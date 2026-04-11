use tch::CModule;
fn main() {
    let cm = CModule::load("nothing.pt").unwrap();
    let _x = cm.named_parameters().unwrap();
}
