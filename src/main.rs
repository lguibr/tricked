use tricked_engine::cli::{self, ParsedCommand};
use tricked_engine::train::{runner, tune};

#[hotpath::main]
fn main() {
    std::env::set_var("TORCH_CPP_LOG_LEVEL", "ERROR");
    tch::set_num_threads(1);
    tch::set_num_interop_threads(1);

    #[cfg(target_os = "linux")]
    unsafe {
        // Force load libtorch global dependencies to ensure CUDA is detected when using Python's libtorch
        use libloading::os::unix::Library as UnixLibrary;
        // 2 = RTLD_NOW, 256 = RTLD_GLOBAL
        let _ = UnixLibrary::open(Some("libtorch_global_deps.so"), 2 | 256);
        let _ = UnixLibrary::open(Some("libtorch_cuda.so"), 2 | 256);
    }

    #[cfg(not(test))]
    unsafe {
        extern "C" {
            fn dlopen(
                filename: *const std::ffi::c_char,
                flag: std::ffi::c_int,
            ) -> *mut std::ffi::c_void;
            fn dlerror() -> *const std::ffi::c_char;
        }

        let mut ops_path = std::path::PathBuf::from("./tricked_ops.so");
        let mut current = std::env::current_dir().unwrap_or_default();
        loop {
            let candidate = current.join("tricked_ops.so");
            if candidate.exists() {
                ops_path = candidate;
                break;
            }
            if !current.pop() {
                break;
            }
        }

        let path_c = std::ffi::CString::new(ops_path.to_string_lossy().as_ref()).unwrap();
        let handle = dlopen(path_c.as_ptr(), 1 | 256);
        if handle.is_null() {
            let err = std::ffi::CStr::from_ptr(dlerror()).to_string_lossy();
            eprintln!(
                "WARNING: Failed to dlopen {}. CUDA Kernel feature packing might crash! Error: {}",
                ops_path.display(),
                err
            );
        } else {
            println!(
                "🚀 Loaded custom C++ PyTorch operators from {}.",
                ops_path.display()
            );
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
