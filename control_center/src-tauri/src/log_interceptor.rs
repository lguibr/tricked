use serde_json::json;
use std::io::{BufRead, BufReader, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use tauri::{AppHandle, Emitter};

pub fn spawn_interceptor(app: AppHandle) {
    #[cfg(unix)]
    {
        // Intercept stdout
        match os_pipe::pipe() {
            Ok((rx, tx)) => {
                let tx_clone = tx.try_clone().unwrap();

                // Save real fd to tee outputs back to terminal
                let real_stdout =
                    unsafe { std::fs::File::from_raw_fd(libc::dup(libc::STDOUT_FILENO)) };
                let _real_stderr =
                    unsafe { std::fs::File::from_raw_fd(libc::dup(libc::STDERR_FILENO)) };

                unsafe {
                    libc::dup2(tx.as_raw_fd(), libc::STDOUT_FILENO);
                    libc::dup2(tx_clone.as_raw_fd(), libc::STDERR_FILENO);
                }

                std::thread::spawn(move || {
                    let mut reader = BufReader::new(rx);
                    let mut real_out = real_stdout;
                    let mut line = String::new();

                    let mut log_buffer = Vec::new();
                    let mut last_emit = std::time::Instant::now();

                    while reader.read_line(&mut line).is_ok() {
                        if line.is_empty() {
                            break;
                        }

                        // Tee to the real terminal
                        let _ = real_out.write_all(line.as_bytes());
                        let _ = real_out.flush();

                        // Emit to Tauri payload globally
                        let mut line_trim = line.clone();
                        if line_trim.ends_with('\n') {
                            line_trim.pop();
                            if line_trim.ends_with('\r') {
                                line_trim.pop();
                            }
                        }

                        log_buffer.push(json!({
                            "run_id": "GLOBAL",
                            "line": line_trim
                        }));

                        if last_emit.elapsed() >= std::time::Duration::from_millis(250)
                            || log_buffer.len() >= 100
                        {
                            let _ = app.emit("log_event_batch", json!(log_buffer));
                            log_buffer.clear();
                            last_emit = std::time::Instant::now();
                        }

                        line.clear();
                    }
                });
            }
            Err(e) => {
                eprintln!("[LogInterceptor] Failed to pipe os: {}", e);
            }
        }
    }
}
