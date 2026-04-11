use crate::net::MuZeroNet;
use tch::{nn::Module, Kind, Tensor};

pub fn initial_inference(
    net: &MuZeroNet,
    batched_state: &Tensor,
) -> (Tensor, Tensor, Tensor, Tensor) {
    assert_eq!(
        batched_state.size().len(),
        4,
        "Initial inference batched_state must have 4 dimensions"
    );
    assert_eq!(
        batched_state.size()[1],
        net.spatial_channel_count,
        "Initial inference batched_state spatial channels mismatch"
    );
    let hidden_state = net.representation.forward(batched_state);
    let (value_logits, policy_logits, hidden_state_logits) = net.prediction.forward(&hidden_state);

    let predicted_value_scalar = net.value_support_to_scalar(&value_logits);
    let policy_probabilities = policy_logits
        .softmax(-1, Kind::Float)
        .clamp(1e-4_f64, 1.0_f64)
        .to_kind(Kind::Half);

    (
        hidden_state,
        predicted_value_scalar,
        policy_probabilities,
        hidden_state_logits,
    )
}

pub fn recurrent_inference(
    net: &MuZeroNet,
    hidden_state: &Tensor,
    batched_action: &Tensor,
    batched_piece_identifier: &Tensor,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
    assert_eq!(
        batched_action.size()[0],
        batched_piece_identifier.size()[0],
        "Action and piece identifier batch sizes must match"
    );

    let (hidden_state_next, value_prefix_logits) = net
        .dynamics
        .forward(hidden_state, batched_action, batched_piece_identifier);
    let (value_logits, policy_logits, hidden_state_logits) =
        net.prediction.forward(&hidden_state_next);

    let value_prefix_scalar_prediction = net.reward_support_to_scalar(&value_prefix_logits);
    let value_scalar_prediction = net.value_support_to_scalar(&value_logits);
    let policy_probabilities = policy_logits
        .softmax(-1, Kind::Float)
        .clamp(1e-4_f64, 1.0_f64)
        .to_kind(Kind::Half);

    (
        hidden_state_next,
        value_prefix_scalar_prediction,
        value_scalar_prediction,
        policy_probabilities,
        hidden_state_logits,
    )
}
