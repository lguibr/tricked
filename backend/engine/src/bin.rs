fn main() {
    println!("Loading model...");
    let m = tch::CModule::load("backend/workspace/runs/test_123/initial_model.pt");
    println!("{:?}", m.is_ok());
}
