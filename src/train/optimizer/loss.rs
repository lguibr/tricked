use tch::{Kind, Tensor};

/// Calculates Negative Cosine Similarity:
/// L(x1, x2) = - (x1 · x2) / (||x1||_2 * ||x2||_2)
pub fn negative_cosine_similarity(
    active_projection_fp16: &Tensor,
    target_projection_fp16: &Tensor,
) -> Tensor {
    let active_projection = active_projection_fp16.to_kind(Kind::Float);
    let target_projection = target_projection_fp16.to_kind(Kind::Float);

    #[cfg(debug_assertions)]
    assert!(
        i64::try_from(active_projection.isnan().any()).unwrap() == 0,
        "NaN detected in active_projection before cosine similarity"
    );
    #[cfg(debug_assertions)]
    assert!(
        i64::try_from(target_projection.isnan().any()).unwrap() == 0,
        "NaN detected in target_projection before cosine similarity"
    );

    let active_l2_norm =
        (active_projection
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist(&[-1i64][..], true, Kind::Float)
            + 1e-8)
            .sqrt();

    let target_l2_norm =
        (target_projection
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist(&[-1i64][..], true, Kind::Float)
            + 1e-8)
            .sqrt();

    let active_normalized = active_projection / active_l2_norm;
    let target_normalized = target_projection / target_l2_norm;

    let similarity_loss = -(&active_normalized * &target_normalized).sum_dim_intlist(
        &[-1i64][..],
        false,
        Kind::Float,
    );

    #[cfg(debug_assertions)]
    assert!(
        i64::try_from(similarity_loss.isnan().any()).unwrap() == 0,
        "NaN detected resulting from cosine similarity calculation"
    );

    similarity_loss
}

pub fn soft_cross_entropy(prediction_logits: &Tensor, target_probabilities: &Tensor) -> Tensor {
    #[cfg(debug_assertions)]
    assert!(
        i64::try_from(prediction_logits.isnan().any()).unwrap() == 0,
        "NaN detected in prediction_logits before soft_cross_entropy"
    );

    let logarithm_probabilities = prediction_logits.log_softmax(-1, Kind::Float);
    // target * log_probs creates a new tensor, which is safe to mutate
    let loss = target_probabilities * logarithm_probabilities;
    let mut cross_entropy_loss = loss.sum_dim_intlist(&[-1i64][..], false, Kind::Float);
    let _ = cross_entropy_loss.neg_();

    #[cfg(debug_assertions)]
    assert!(
        i64::try_from(cross_entropy_loss.isnan().any()).unwrap() == 0,
        "NaN detected resulting from soft_cross_entropy calculation"
    );

    cross_entropy_loss
}

/// Calculates Binary Cross Entropy Loss:
/// L = - [target * log(σ(logits)) + (1 - target) * log(1 - σ(logits))]
pub fn binary_cross_entropy(prediction_logits: &Tensor, binary_targets: &Tensor) -> Tensor {
    #[cfg(debug_assertions)]
    assert!(
        i64::try_from(prediction_logits.isnan().any()).unwrap() == 0,
        "NaN detected in prediction_logits before binary_cross_entropy"
    );

    let binary_cross_entropy_loss = prediction_logits.binary_cross_entropy_with_logits::<Tensor>(
        binary_targets,
        None,
        None,
        tch::Reduction::None,
    );

    #[cfg(debug_assertions)]
    assert!(
        i64::try_from(binary_cross_entropy_loss.isnan().any()).unwrap() == 0,
        "NaN detected resulting from BCE calculation"
    );

    binary_cross_entropy_loss
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
    #[test]
    fn test_soft_cross_entropy_parity() {
        let logits = Tensor::from_slice(&[2.0_f32, 1.0, 0.1]);
        let targets = Tensor::from_slice(&[0.7_f32, 0.2, 0.1]);

        let loss = soft_cross_entropy(&logits, &targets);

        let expected_prob = [
            (2.0f32).exp() / ((2.0f32).exp() + (1.0f32).exp() + (0.1f32).exp()),
            (1.0f32).exp() / ((2.0f32).exp() + (1.0f32).exp() + (0.1f32).exp()),
            (0.1f32).exp() / ((2.0f32).exp() + (1.0f32).exp() + (0.1f32).exp()),
        ];

        let manual_loss = -(0.7 * expected_prob[0].ln()
            + 0.2 * expected_prob[1].ln()
            + 0.1 * expected_prob[2].ln());

        let rust_loss: f32 = loss.try_into().unwrap_or(0.0);
        assert!(
            (rust_loss - manual_loss).abs() < 1e-4,
            "Loss Function parity failed! Rust: {} vs Manual: {}",
            rust_loss,
            manual_loss
        );
    }
}
