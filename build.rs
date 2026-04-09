use std::process::Command;

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        // Query Python to get the torch lib path dynamically
        let py_query = "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')";
        if let Ok(output) = Command::new("python3").args(["-c", py_query]).output() {
            if let Ok(path) = String::from_utf8(output.stdout) {
                let path = path.trim();
                println!("cargo:rustc-link-search=native={}", path);
            }
        }

        // Force the linker to keep cuda libraries
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        println!("cargo:rustc-link-lib=dylib=torch_cuda");
        println!("cargo:rustc-link-lib=dylib=c10_cuda");
    }
}
