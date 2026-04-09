use tch::Cuda;
fn main() {
    println!("CUDA is available: {}", Cuda::is_available());
}
