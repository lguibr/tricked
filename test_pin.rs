use tch::{Device, Kind, Tensor};
fn main() {
    let t = Tensor::zeros([1_i64], (Kind::Float, Device::Cpu));
    let _t = t.pin_memory(Device::Cpu);
}
