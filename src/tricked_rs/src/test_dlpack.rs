use tch::Tensor;
fn main() {
    let t = Tensor::zeros([10], (tch::Kind::Float, tch::Device::Cpu));
    let ptr = t.data_ptr();
}
