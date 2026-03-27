use tch::{Tensor, Kind, Device};

fn main() {
    let indices = Tensor::from_slice(&[0i64, 1, 0, 1]).reshape([2, 2]);
    let values = Tensor::from_slice(&[1.0f32, 2.0]);
    let sparse = Tensor::sparse_coo_tensor(&indices, &values, [2, 2], (Kind::Float, Device::Cpu));
    let dense = Tensor::ones([2, 2], (Kind::Float, Device::Cpu));
    let out = sparse.matmul(&dense);
    println!("Sparse matmul out: {:?}", out.size());
}
