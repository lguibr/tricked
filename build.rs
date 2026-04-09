fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        // Force the linker to keep cuda libraries
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        println!("cargo:rustc-link-lib=dylib=torch_cuda");
        println!("cargo:rustc-link-lib=dylib=c10_cuda");
    }
}
