#[test]
fn test_custom_op() {
    let t = tch::Tensor::zeros([10], (tch::Kind::Float, tch::Device::Cpu));
    // How to call custom op? Maybe tch::CModule::load doesn't matter.
    // If I load the library using libloading, then PyTorch registers the TORCH_LIBRARY!
    unsafe { libloading::Library::new("tricked_ops.so").unwrap(); }
    
    // There is no Direct custom op call without CModule in some versions. But wait, we can compile a dummy CModule that calls it from TorchScript! That defeats the purpose.
}
