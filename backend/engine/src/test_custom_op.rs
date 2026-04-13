#[test]
fn test_custom_op() {
    let t = tch::Tensor::zeros([10], (tch::Kind::Float, tch::Device::Cpu));
    // Updated to libtricked_ops.so
    unsafe { libloading::Library::new("libtricked_ops.so").unwrap(); }
}