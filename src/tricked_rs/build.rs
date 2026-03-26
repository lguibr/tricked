use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../scripts/generators/generate_rust_constants.py");

    let output = Command::new("python")
        .arg("../../scripts/generators/generate_rust_constants.py")
        .output()
        .expect("Failed to execute code generation script");

    if !output.status.success() {
        panic!(
            "Constants generation failed: {:?}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
