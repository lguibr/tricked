use std::io::{BufRead, BufReader};
use os_pipe::pipe;

fn main() {
    let (rx, tx) = pipe().unwrap();
    let file = unsafe { std::fs::File::from_raw_fd(std::os::fd::IntoRawFd::into_raw_fd(tx)) };
    let _r = gag::Redirect::stdout(file).unwrap();
    
    std::thread::spawn(move || {
        let mut reader = BufReader::new(rx);
        let mut line = String::new();
        while reader.read_line(&mut line).is_ok() {
            if line.is_empty() { break; }
            eprint!("{}", line); // Tee to stderr
            line.clear();
        }
    });

    println!("Hello World from Gag!");
    std::thread::sleep(std::time::Duration::from_millis(100));
}
