use tch::{CModule, Device, IValue, Tensor};
fn main() {
    let cm = CModule::load_on_device("nothing.pt", Device::Cpu).unwrap();
    let _ivalue = cm.forward_is(&[IValue::Tensor(Tensor::zeros(
        [1],
        (tch::Kind::Float, Device::Cpu),
    ))]);
}
