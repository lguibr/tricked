use tch::{Kind, Tensor};

/// Calculates Negative Cosine Similarity:
/// L(x1, x2) = - (x1 · x2) / (||x1||_2 * ||x2||_2)
pub fn negative_cosine_similarity(
    active_projection: &Tensor,
    target_projection: &Tensor,
) -> Tensor {
    assert!(
        i64::try_from(active_projection.isnan().any()).unwrap() == 0,
        "NaN detected in active_projection before cosine similarity"
    );
    assert!(
        i64::try_from(target_projection.isnan().any()).unwrap() == 0,
        "NaN detected in target_projection before cosine similarity"
    );

    let active_l2_norm = active_projection
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist(&[-1i64][..], true, Kind::Float)
        .sqrt()
        + 1e-8;

    let target_l2_norm = target_projection
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist(&[-1i64][..], true, Kind::Float)
        .sqrt()
        + 1e-8;

    let active_normalized = active_projection / active_l2_norm;
    let target_normalized = target_projection / target_l2_norm;

    let similarity_loss = -(&active_normalized * &target_normalized).sum_dim_intlist(
        &[-1i64][..],
        false,
        Kind::Float,
    );

    assert!(
        i64::try_from(similarity_loss.isnan().any()).unwrap() == 0,
        "NaN detected resulting from cosine similarity calculation"
    );

    similarity_loss
}

/// Calculates Soft Cross Entropy Loss:
/// L = - Σ (target_probs * log(softmax(logits)))
pub fn soft_cross_entropy(prediction_logits: &Tensor, target_probabilities: &Tensor) -> Tensor {
    assert!(
        i64::try_from(prediction_logits.isnan().any()).unwrap() == 0,
        "NaN detected in prediction_logits before soft_cross_entropy"
    );

    let logarithm_probabilities = prediction_logits.log_softmax(-1, Kind::Float);
    let cross_entropy_loss = -(target_probabilities * logarithm_probabilities).sum_dim_intlist(
        &[-1i64][..],
        false,
        Kind::Float,
    );

    assert!(
        i64::try_from(cross_entropy_loss.isnan().any()).unwrap() == 0,
        "NaN detected resulting from soft_cross_entropy calculation"
    );

    cross_entropy_loss
}

/// Calculates Binary Cross Entropy Loss:
/// L = - [target * log(σ(logits)) + (1 - target) * log(1 - σ(logits))]
pub fn binary_cross_entropy(prediction_logits: &Tensor, binary_targets: &Tensor) -> Tensor {
    assert!(
        i64::try_from(prediction_logits.isnan().any()).unwrap() == 0,
        "NaN detected in prediction_logits before binary_cross_entropy"
    );

    let bce_loss = prediction_logits.binary_cross_entropy_with_logits::<Tensor>(
        binary_targets,
        None,
        None,
        tch::Reduction::None,
    );

    assert!(
        i64::try_from(bce_loss.isnan().any()).unwrap() == 0,
        "NaN detected resulting from BCE calculation"
    );

    bce_loss
}

/// Scales the gradient passing through the tensor:
/// x_scaled = x * scale + detach(x) * (1 - scale)
pub fn scale_gradient(input_tensor: &Tensor, gradient_scale: f64) -> Tensor {
    input_tensor * gradient_scale + input_tensor.detach() * (1.0 - gradient_scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn test_scale_gradient() {
        let active_tensor =
            Tensor::ones([2, 2], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let scaled_tensor = scale_gradient(&active_tensor, 0.5);
        let sum_loss = scaled_tensor.sum(Kind::Float);
        sum_loss.backward();

        let gradient = active_tensor.grad();
        let gradient_value: f32 = gradient.mean(Kind::Float).try_into().unwrap();
        assert!(
            (gradient_value - 0.5).abs() < 1e-6,
            "Gradient was not scaled correctly by 0.5"
        );
    }

    #[test]
    fn test_cosine_similarity_epsilon_stability() {
        let active_projection = Tensor::zeros([2, 512], (Kind::Float, Device::Cpu));
        let target_projection = Tensor::zeros([2, 512], (Kind::Float, Device::Cpu));
        let loss = negative_cosine_similarity(&active_projection, &target_projection);
        assert_eq!(
            i64::try_from(loss.isnan().any()).unwrap(),
            0,
            "Cosine similarity resulted in NaN on zero tensors!"
        );
    }
}
