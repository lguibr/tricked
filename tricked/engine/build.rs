use std::io::Result;

fn main() -> Result<()> {
    println!("cargo:rustc-link-search=native={}/../../backend/extensions", std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string()));
    
    prost_build::compile_protos(&["../proto_out/tricked.proto"], &["../proto_out/"])?;
    Ok(())
}
