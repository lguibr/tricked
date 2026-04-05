use tricked_engine::cli::{self, ParsedCommand};
use tricked_engine::train::{runner, tune};

#[cfg(not(test))]
fn resolve_asset(name: &str) -> std::path::PathBuf {
    let mut current = std::env::current_dir().unwrap_or_default();
    loop {
        let candidate = current.join(name);
        if candidate.exists() {
            return candidate;
        }
        if !current.pop() {
            break;
        }
    }
    std::path::PathBuf::from(format!("./{}", name))
}

#[hotpath::main]
fn main() {
    #[cfg(not(test))]
    unsafe {
        extern "C" {
            fn dlopen(
                filename: *const std::ffi::c_char,
                flag: std::ffi::c_int,
            ) -> *mut std::ffi::c_void;
            fn dlerror() -> *const std::ffi::c_char;
        }
        let so_path = resolve_asset("tricked_ops.so");
        let so_path_str = so_path.to_string_lossy().into_owned();
        let c_path = std::ffi::CString::new(so_path_str).unwrap();
        let handle = dlopen(c_path.as_ptr() as *const _, 1 | 256);
        if handle.is_null() {
            let err = std::ffi::CStr::from_ptr(dlerror()).to_string_lossy();
            eprintln!("WARNING: Failed to dlopen ./tricked_ops.so. CUDA Kernel feature packing might crash! Error: {}", err);
        } else {
            println!("🚀 Loaded custom C++ PyTorch operators.");
        }
    }

    match cli::parse_and_build_config() {
        ParsedCommand::Train(cfg_box, max_steps) => {
            runner::run_training(*cfg_box, max_steps);
        }
        ParsedCommand::Tune(tune_cfg) => {
            tune::run_tuning_pipeline(tune_cfg);
        }
    }
}
