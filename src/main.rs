use tricked_engine::cli::{self, ParsedCommand};
use tricked_engine::train::{runner, tune};

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
        let handle = dlopen(c"./tricked_ops.so".as_ptr() as *const _, 1 | 256);
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
