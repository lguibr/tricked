use tch::{Kind, Tensor};

pub fn negative_cosine_similarity(x1: &Tensor, x2: &Tensor) -> Tensor {
    let x1_l2 = x1
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist(&[-1i64][..], true, Kind::Float)
        .sqrt()
        + 1e-8;
    let x2_l2 = x2
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist(&[-1i64][..], true, Kind::Float)
        .sqrt()
        + 1e-8;
    let x1_norm = x1 / x1_l2;
    let x2_norm = x2 / x2_l2;
    -(&x1_norm * &x2_norm).sum_dim_intlist(&[-1i64][..], false, Kind::Float)
}

pub fn soft_cross_entropy(logits: &Tensor, target_probs: &Tensor) -> Tensor {
    let log_probs = logits.log_softmax(-1, Kind::Float);
    -(target_probs * log_probs).sum_dim_intlist(&[-1i64][..], false, Kind::Float)
}

pub fn binary_cross_entropy(logits: &Tensor, targets: &Tensor) -> Tensor {
    logits.binary_cross_entropy_with_logits::<Tensor>(targets, None, None, tch::Reduction::None)
}

pub fn scale_gradient(tensor: &Tensor, scale: f64) -> Tensor {
    tensor * scale + tensor.detach() * (1.0 - scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn test_scale_gradient() {
        let t = Tensor::ones([2, 2], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let scaled = scale_gradient(&t, 0.5);
        let sum = scaled.sum(Kind::Float);
        sum.backward();
        
        let grad = t.grad();
        let grad_val: f32 = grad.mean(Kind::Float).try_into().unwrap();
        assert!((grad_val - 0.5).abs() < 1e-6, "Gradient was not scaled correctly by 0.5");
    }
}
