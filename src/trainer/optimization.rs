use crate::buffer::ReplayBuffer;
use crate::config::Config;
use crate::network::MuZeroNet;
use crate::trainer::loss::{
    binary_cross_entropy, negative_cosine_similarity, scale_gradient, soft_cross_entropy,
};
use tch::{nn, nn::Module, Device, Kind, Tensor};

pub struct TrainMetrics {
    pub total_loss: f64,
    pub policy_loss: f64,
    pub value_loss: f64,
    pub reward_loss: f64,
}

pub fn train_step(
    neural_model: &MuZeroNet,
    exponential_moving_average_model: &MuZeroNet,
    gradient_optimizer: &mut nn::Optimizer,
    replay_buffer: &ReplayBuffer,
    batched_experience: crate::buffer::replay::BatchTensors,
    sequence_unroll_steps: usize,
) -> TrainMetrics {
    let sequence_unroll_steps = sequence_unroll_steps as i64;

    let batched_state = batched_experience.state_features_batch.to_kind(Kind::Float);
    let batched_action = batched_experience.actions_batch;
    let batched_piece_identifier = batched_experience.piece_identifiers_batch;
    let batched_reward = batched_experience.rewards_batch;
    let batched_target_policy = batched_experience.target_policies_batch;
    let batched_target_value = batched_experience.target_values_batch;
    let batched_transition_state = batched_experience
        .transition_states_batch
        .to_kind(Kind::Float);
    let batched_mask = batched_experience.loss_masks_batch;
    let batched_importance_weight = batched_experience.importance_weights_batch;
    let global_indices = batched_experience.global_indices_sampled;

    gradient_optimizer.zero_grad();
    let scaled_importance_weights = batched_importance_weight;

    let (
        computed_final_loss,
        temporal_difference_errors,
        avg_policy_loss,
        avg_value_loss,
        avg_reward_loss,
    ) = tch::autocast(true, || {
        let mut running_hidden_state = neural_model.representation.forward(&batched_state);

        let (initial_value_logits, initial_policy_logits, initial_hidden_state_logits) =
            neural_model.prediction.forward(&running_hidden_state);

        // Value Loss: Cross-entropy between network value support prediction and target scalar
        let initial_value_targets =
            neural_model.scalar_to_support(&batched_target_value.select(1, 0));
        let initial_value_loss = soft_cross_entropy(&initial_value_logits, &initial_value_targets);

        // Policy Loss: Cross-entropy between network policy vector and MCTS target distribution
        let initial_policy_probabilities_target = batched_target_policy.select(1, 0) + 1e-8;
        let initial_policy_loss =
            soft_cross_entropy(&initial_policy_logits, &initial_policy_probabilities_target);

        let mut initial_binary_cross_entropy = binary_cross_entropy(
            &initial_hidden_state_logits,
            &batched_state.select(1, 19).flatten(1, -1),
        );
        if initial_binary_cross_entropy.dim() > 1 {
            initial_binary_cross_entropy =
                initial_binary_cross_entropy.mean_dim(&[1i64][..], false, Kind::Float);
        }

        let mut cumulative_loss =
            &initial_value_loss + &initial_policy_loss + (&initial_binary_cross_entropy * 0.5);

        let mut value_loss_tracker = initial_value_loss.mean(Kind::Float);
        let mut policy_loss_tracker = initial_policy_loss.mean(Kind::Float);
        let mut reward_loss_tracker = Tensor::zeros_like(&value_loss_tracker);

        for unroll_k in 0..sequence_unroll_steps {
            let action_at_k = batched_action.select(1, unroll_k);
            let piece_identifier_at_k = batched_piece_identifier.select(1, unroll_k);

            let (next_hidden_state_prediction, reward_logits_prediction) =
                neural_model.dynamics.forward(
                    &scale_gradient(&running_hidden_state, 0.5),
                    &action_at_k,
                    &piece_identifier_at_k,
                );
            running_hidden_state = next_hidden_state_prediction;

            let target_hidden_state_projection = tch::no_grad(|| {
                exponential_moving_average_model
                    .representation
                    .forward(&batched_transition_state.select(1, unroll_k))
            });
            let projected_target_representation = tch::no_grad(|| {
                exponential_moving_average_model
                    .projector
                    .forward(&target_hidden_state_projection)
            });

            let projected_active_representation =
                neural_model.projector.forward(&running_hidden_state);
            let (unrolled_value_logits, unrolled_policy_logits, unrolled_hidden_state_logits) =
                neural_model.prediction.forward(&running_hidden_state);

            let reward_targets_support =
                neural_model.scalar_to_support(&batched_reward.select(1, unroll_k));
            let unroll_sequence_mask = batched_mask.select(1, unroll_k + 1);

            let unrolled_reward_loss =
                soft_cross_entropy(&reward_logits_prediction, &reward_targets_support)
                    * &unroll_sequence_mask;

            let value_targets_support =
                neural_model.scalar_to_support(&batched_target_value.select(1, unroll_k + 1));
            let unrolled_value_loss =
                soft_cross_entropy(&unrolled_value_logits, &value_targets_support)
                    * &unroll_sequence_mask;

            let unrolled_policy_targets = batched_target_policy.select(1, unroll_k + 1) + 1e-8;
            let unrolled_policy_loss =
                soft_cross_entropy(&unrolled_policy_logits, &unrolled_policy_targets)
                    * &unroll_sequence_mask;

            reward_loss_tracker += unrolled_reward_loss.mean(Kind::Float);
            value_loss_tracker += unrolled_value_loss.mean(Kind::Float);
            policy_loss_tracker += unrolled_policy_loss.mean(Kind::Float);

            cumulative_loss += &unrolled_reward_loss + &unrolled_value_loss + &unrolled_policy_loss;
            cumulative_loss += negative_cosine_similarity(
                &projected_active_representation,
                &projected_target_representation,
            ) * &unroll_sequence_mask;

            let mut unrolled_binary_cross_entropy = binary_cross_entropy(
                &unrolled_hidden_state_logits,
                &batched_transition_state
                    .select(1, unroll_k)
                    .select(1, 19)
                    .flatten(1, -1),
            );
            if unrolled_binary_cross_entropy.dim() > 1 {
                unrolled_binary_cross_entropy =
                    unrolled_binary_cross_entropy.mean_dim(&[1i64][..], false, Kind::Float);
            }
            cumulative_loss += unrolled_binary_cross_entropy * 0.5 * &unroll_sequence_mask;
        }

        let averaged_scaled_final_loss = (cumulative_loss * scaled_importance_weights)
            .mean(Kind::Float)
            / (sequence_unroll_steps as f64);

        let absolute_temporal_difference_errors = tch::no_grad(|| {
            (neural_model.scalar_to_support(&batched_target_value.select(1, 0))
                - initial_value_logits.softmax(-1, Kind::Float))
            .abs()
            .sum_dim_intlist(&[-1i64][..], false, Kind::Float)
        });

        let divisor = (sequence_unroll_steps + 1) as f64;
        let avg_policy_loss = f64::try_from(policy_loss_tracker / divisor).unwrap_or(0.0);
        let avg_value_loss = f64::try_from(value_loss_tracker / divisor).unwrap_or(0.0);
        let avg_reward_loss = f64::try_from(reward_loss_tracker / divisor).unwrap_or(0.0);

        (
            averaged_scaled_final_loss,
            absolute_temporal_difference_errors,
            avg_policy_loss,
            avg_value_loss,
            avg_reward_loss,
        )
    });

    computed_final_loss.backward();

    gradient_optimizer.clip_grad_norm(5.0);
    gradient_optimizer.step();

    let temporal_difference_f32_vec: Vec<f32> =
        temporal_difference_errors.try_into().unwrap_or_default();
    let temporal_difference_f64_vec: Vec<f64> = temporal_difference_f32_vec
        .into_iter()
        .map(|error_val| error_val as f64)
        .collect();
    replay_buffer.update_priorities(&global_indices, &temporal_difference_f64_vec);

    let final_loss_f64 = f64::try_from(computed_final_loss).unwrap_or(0.0);

    TrainMetrics {
        total_loss: final_loss_f64,
        policy_loss: avg_policy_loss,
        value_loss: avg_value_loss,
        reward_loss: avg_reward_loss,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, nn::OptimizerConfig, Device};

    #[test]
    fn test_train_step_bptt_and_masking() {
        let variable_store = nn::VarStore::new(Device::Cpu);
        let neural_model = MuZeroNet::new(&variable_store.root(), 16, 1, 200);
        let ema_model = MuZeroNet::new(&variable_store.root(), 16, 1, 200);
        let mut gradient_optimizer = nn::Adam::default().build(&variable_store, 1e-3).unwrap();

        let configuration = Config {
            device: "cpu".into(),
            model_checkpoint: "".into(),
            metrics_file: "".into(),
            d_model: 16,
            num_blocks: 1,
            support_size: 200,
            capacity: 100,
            num_games: 1,
            simulations: 10,
            train_batch_size: 2,
            train_epochs: 1,
            num_processes: 1,
            worker_device: "cpu".into(),
            unroll_steps: 2,
            td_steps: 5,
            zmq_inference_port: "".into(),
            zmq_batch_size: 1,
            zmq_timeout_ms: 1,
            max_gumbel_k: 4,
            gumbel_scale: 1.0,
            temp_decay_steps: 10,
            difficulty: 6,
            exploit_starts: vec![],
            temp_boost: false,
            exp_name: "".into(),
            lr_init: 1e-3,
        };

        let replay_buffer = ReplayBuffer::new(100, 2, 8);

        let board_states = vec![[0u64, 0u64]];
        let available_pieces = vec![[0i32, 0, 0]];
        let actions_taken = vec![0i64];
        let piece_identifiers = vec![0i64];
        let rewards_received = vec![1.0f32];
        let policy_targets = vec![[0.0f32; 288]];
        let value_targets = vec![0.5f32];

        replay_buffer.add_game(crate::buffer::replay::OwnedGameData {
            difficulty_setting: 6,
            episode_score: 1.0,
            board_states: board_states.clone(),
            available_pieces: available_pieces.clone(),
            actions_taken: actions_taken.clone(),
            piece_identifiers: piece_identifiers.clone(),
            rewards_received: rewards_received.clone(),
            policy_targets: policy_targets.clone(),
            value_targets: value_targets.clone(),
        });
        replay_buffer.add_game(crate::buffer::replay::OwnedGameData {
            difficulty_setting: 6,
            episode_score: 1.0,
            board_states,
            available_pieces,
            actions_taken,
            piece_identifiers,
            rewards_received,
            policy_targets,
            value_targets,
        });

        let batched_exp = replay_buffer.sample_batch(2, Device::Cpu, 1.0).unwrap();

        train_step(
            &neural_model,
            &ema_model,
            &mut gradient_optimizer,
            &replay_buffer,
            batched_exp,
            configuration.unroll_steps,
        );
    }
}
