use tch::{CModule, Device, Kind};
fn main() {
    let mut m = CModule::load_data_on_device(&mut [], Device::Cuda(0)).unwrap();
    m.to(Device::Cuda(0), Kind::Half, false);
}
