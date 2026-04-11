use tch::{Tensor, Kind, Device};
fn main() {
    let t = Tensor::zeros([10], (Kind::Float, Device::Cpu));
    let args = [tch::IValue::Tensor(t)];
    // tch::IValue::run_custom_op?
    // Let's check how custom ops are supported.
    // We can grep tch repository or doc locally if it exists.
}
