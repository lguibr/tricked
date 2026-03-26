use tch::{Kind, Tensor};

pub fn negative_cosine_similarity(x1: &Tensor, x2: &Tensor) -> Tensor {
    let x1_l2 = x1
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist(&[-1], true, Kind::Float)
        .sqrt()
        + 1e-8;
    let x2_l2 = x2
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist(&[-1], true, Kind::Float)
        .sqrt()
        + 1e-8;
    let x1_norm = x1 / x1_l2;
    let x2_norm = x2 / x2_l2;
    -(&x1_norm * &x2_norm).sum_dim_intlist(&[-1], false, Kind::Float)
}

pub fn soft_cross_entropy(logits: &Tensor, target_probs: &Tensor) -> Tensor {
    let log_probs = logits.log_softmax(-1, Kind::Float);
    -(target_probs * log_probs).sum_dim_intlist(&[-1], false, Kind::Float)
}

pub fn binary_cross_entropy(logits: &Tensor, targets: &Tensor) -> Tensor {
    logits.binary_cross_entropy_with_logits::<Tensor>(targets, None, None, tch::Reduction::None)
}

pub fn scale_gradient(tensor: &Tensor, scale: f64) -> Tensor {
    tensor * scale + tensor.detach() * (1.0 - scale)
}
