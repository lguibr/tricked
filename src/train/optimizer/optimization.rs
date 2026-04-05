use crate::net::MuZeroNet;
use crate::train::buffer::ReplayBuffer;
use crate::train::optimizer::loss::{
    binary_cross_entropy, negative_cosine_similarity, scale_gradient, soft_cross_entropy,
};
use tch::{nn, nn::Module, Kind, Tensor};

pub struct TrainMetrics {
    pub total_loss: f64,
    pub policy_loss: f64,
    pub value_loss: f64,
    pub value_prefix_loss: f64,
}

#[hotpath::measure]
pub fn train_step(
    neural_model: &MuZeroNet,
    exponential_moving_average_model: &MuZeroNet,
    gradient_optimizer: &mut nn::Optimizer,
    replay_buffer: &ReplayBuffer,
    batched_experience_tensors: &crate::train::buffer::BatchTensors,
    sequence_unroll_steps: usize,
) -> TrainMetrics {
    let sequence_unroll_steps = sequence_unroll_steps as i64;

    let batched_state = &batched_experience_tensors.state_features_batch;
    let batched_action = &batched_experience_tensors.actions_batch;
    let batched_piece_identifier = &batched_experience_tensors.piece_identifiers_batch;
    let batched_value_prefix = &batched_experience_tensors.value_prefixs_batch;
    let batched_target_policy = &batched_experience_tensors.target_policies_batch;
    let batched_target_value = &batched_experience_tensors.target_values_batch;
    let batched_transition_boards = &batched_experience_tensors.transition_boards_batch;
    let _batched_transition_actions = &batched_experience_tensors.transition_actions_batch;
    let _batched_transition_metadata = &batched_experience_tensors.transition_metadata_batch;
    let batched_mask = &batched_experience_tensors.loss_masks_batch;
    let batched_importance_weight = &batched_experience_tensors.importance_weights_batch;
    let global_indices = &batched_experience_tensors.global_indices_sampled;

    gradient_optimizer.zero_grad();
    let scaled_importance_weights = batched_importance_weight;

    let (
        computed_final_loss,
        temporal_difference_errors,
        avg_policy_loss,
        avg_value_loss,
        avg_value_prefix_loss,
    ) = tch::autocast(true, || {
        #[cfg(debug_assertions)]
        assert!(
            i64::try_from(batched_state.isnan().any()).unwrap() == 0,
            "batched_state ALREADY HAS NANS!"
        );
        let mut running_hidden_state = neural_model.representation.forward(batched_state);

        let rh_size = running_hidden_state.size();
        assert_eq!(
            rh_size.len(),
            4,
            "Hidden state must strictly be [Batch, Channels, Height, Width]"
        );
        assert_eq!(rh_size[2], 8, "Spatial height must be exactly 8");
        assert_eq!(rh_size[3], 8, "Spatial width must be exactly 8");
        #[cfg(debug_assertions)]
        assert!(
            i64::try_from(running_hidden_state.isnan().any()).unwrap() == 0,
            "NaN detected in running_hidden_state!"
        );

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
        let mut value_prefix_loss_tracker = Tensor::zeros_like(&value_loss_tracker);

        let batch_size = batched_state.size()[0];

        let boards_all = batched_transition_boards; // [Batch, Unroll, 8, 2]
        let b0 = boards_all.select(3, 0).unsqueeze(-1); // [Batch, Unroll, 8, 1]
        let b1 = boards_all.select(3, 1).unsqueeze(-1); // [Batch, Unroll, 8, 1]

        let zero = Tensor::from(0i64).to_device(boards_all.device());
        let mut powers_array = [0i64; 64];
        for (i, power) in powers_array.iter_mut().enumerate() {
            *power = if i == 63 { i64::MIN } else { 1i64 << i };
        }
        let powers = Tensor::from_slice(&powers_array).to_device(boards_all.device());

        let b0_bits = b0
            .bitwise_and_tensor(&powers)
            .not_equal_tensor(&zero)
            .to_kind(Kind::Float);

        let b1_bits = b1
            .bitwise_and_tensor(&powers)
            .not_equal_tensor(&zero)
            .to_kind(Kind::Float);

        let raw_bits = Tensor::cat(&[b0_bits, b1_bits], -1); // [Batch, Unroll, 8, 128]

        let mut feature_planes = vec![];
        for i in 0..8 {
            feature_planes.push(raw_bits.select(2, i).unsqueeze(2));
        }

        let dummy_planes = Tensor::zeros(
            [batch_size, sequence_unroll_steps, 12, 128],
            (Kind::Float, boards_all.device()),
        );
        feature_planes.push(dummy_planes);

        let unrolled_state_features_all =
            Tensor::cat(&feature_planes, 2).view([batch_size, sequence_unroll_steps, 20, 8, 16]);

        for unroll_k in 0..sequence_unroll_steps {
            let action_at_k = batched_action.select(1, unroll_k);
            let piece_identifier_at_k = batched_piece_identifier.select(1, unroll_k);

            let (next_hidden_state_prediction, value_prefix_logits_prediction) =
                neural_model.dynamics.forward(
                    &scale_gradient(&running_hidden_state, 0.5),
                    &action_at_k,
                    &piece_identifier_at_k,
                );
            running_hidden_state = next_hidden_state_prediction;

            let rh_size = running_hidden_state.size();
            assert_eq!(
                rh_size.len(),
                4,
                "Dynamics hidden state must strictly be [Batch, Channels, Height, Width]"
            );
            assert_eq!(rh_size[2], 8, "Spatial height must be exactly 8");
            assert_eq!(rh_size[3], 8, "Spatial width must be exactly 8");
            #[cfg(debug_assertions)]
            assert!(
                i64::try_from(running_hidden_state.isnan().any()).unwrap() == 0,
                "NaN detected in next_hidden_state_prediction!"
            );
            // Target Hidden State via EMA
            let unrolled_state_features = unrolled_state_features_all.select(1, unroll_k);

            let target_hidden_state_projection = tch::no_grad(|| {
                exponential_moving_average_model
                    .representation
                    .forward(&unrolled_state_features)
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

            let value_prefix_targets_support =
                neural_model.scalar_to_support(&batched_value_prefix.select(1, unroll_k));
            let unroll_sequence_mask = batched_mask.select(1, unroll_k + 1);

            let unrolled_value_prefix_loss = soft_cross_entropy(
                &value_prefix_logits_prediction,
                &value_prefix_targets_support,
            ) * &unroll_sequence_mask;

            let value_targets_support =
                neural_model.scalar_to_support(&batched_target_value.select(1, unroll_k + 1));
            let unrolled_value_loss =
                soft_cross_entropy(&unrolled_value_logits, &value_targets_support)
                    * &unroll_sequence_mask;

            let unrolled_policy_targets = batched_target_policy.select(1, unroll_k + 1) + 1e-8;
            let unrolled_policy_loss =
                soft_cross_entropy(&unrolled_policy_logits, &unrolled_policy_targets)
                    * &unroll_sequence_mask;

            value_prefix_loss_tracker += unrolled_value_prefix_loss.mean(Kind::Float);
            value_loss_tracker += unrolled_value_loss.mean(Kind::Float);
            policy_loss_tracker += unrolled_policy_loss.mean(Kind::Float);

            let unroll_scale = 1.0 / (sequence_unroll_steps as f64);

            cumulative_loss +=
                (&unrolled_value_prefix_loss + &unrolled_value_loss + &unrolled_policy_loss)
                    * unroll_scale;
            cumulative_loss += (negative_cosine_similarity(
                &projected_active_representation,
                &projected_target_representation,
            ) * &unroll_sequence_mask)
                * unroll_scale;

            let mut unrolled_binary_cross_entropy = binary_cross_entropy(
                &unrolled_hidden_state_logits,
                &unrolled_state_features.select(1, 19).flatten(1, -1),
            );
            if unrolled_binary_cross_entropy.dim() > 1 {
                unrolled_binary_cross_entropy =
                    unrolled_binary_cross_entropy.mean_dim(&[1i64][..], false, Kind::Float);
            }
            cumulative_loss +=
                (unrolled_binary_cross_entropy * 0.5 * &unroll_sequence_mask) * unroll_scale;
        }

        let averaged_scaled_final_loss =
            (cumulative_loss * scaled_importance_weights).mean(Kind::Float);

        let absolute_temporal_difference_errors = tch::no_grad(|| {
            (neural_model.scalar_to_support(&batched_target_value.select(1, 0))
                - initial_value_logits.softmax(-1, Kind::Float))
            .abs()
            .sum_dim_intlist(&[-1i64][..], false, Kind::Float)
        });

        let divisor = (sequence_unroll_steps + 1) as f64;
        let avg_policy_loss = f64::try_from(policy_loss_tracker / divisor).unwrap_or(0.0);
        let avg_value_loss = f64::try_from(value_loss_tracker / divisor).unwrap_or(0.0);
        let avg_value_prefix_loss =
            f64::try_from(value_prefix_loss_tracker / divisor).unwrap_or(0.0);

        (
            averaged_scaled_final_loss,
            absolute_temporal_difference_errors,
            avg_policy_loss,
            avg_value_loss,
            avg_value_prefix_loss,
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
    replay_buffer.update_priorities(global_indices, &temporal_difference_f64_vec);

    let final_loss_f64 = f64::try_from(computed_final_loss).unwrap_or(0.0);

    TrainMetrics {
        total_loss: final_loss_f64,
        policy_loss: avg_policy_loss,
        value_loss: avg_value_loss,
        value_prefix_loss: avg_value_prefix_loss,
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

        let configuration = crate::config::Config {
            experiment_name_identifier: "test_exp".to_string(),
            device: "cpu".into(),
            paths: crate::config::ExperimentPaths::default(),
            hidden_dimension_size: 16,
            num_blocks: 1,
            support_size: 200,
            buffer_capacity_limit: 100,
            simulations: 10,
            train_batch_size: 2,
            train_epochs: 1,
            num_processes: 1,
            worker_device: "cpu".into(),
            unroll_steps: 2,
            temporal_difference_steps: 5,
            inference_batch_size_limit: 1,
            inference_timeout_ms: 1,
            max_gumbel_k: 4,
            gumbel_scale: 1.0,
            temp_decay_steps: 10,
            difficulty: 6,
            temp_boost: false,
            lr_init: 1e-3,
            reanalyze_ratio: 0.25,
        };

        let replay_buffer = ReplayBuffer::new(100, 2, 8, 32);

        let steps = vec![
            crate::train::buffer::GameStep {
                board_state: [0u64, 0u64],
                available_pieces: [0i32, 0, 0],
                action_taken: 0i64,
                piece_identifier: 0i64,
                value_prefix_received: 1.0f32,
                policy_target: [0.0f32; 288],
                value_target: 0.5f32,
            };
            15
        ];

        replay_buffer.add_game(crate::train::buffer::OwnedGameData {
            difficulty_setting: 6,
            episode_score: 1.0,
            steps: steps.clone(),
            lines_cleared: 0,
            mcts_depth_mean: 0.0,
            mcts_search_time_mean: 0.0,
        });
        replay_buffer.add_game(crate::train::buffer::OwnedGameData {
            difficulty_setting: 6,
            episode_score: 1.0,
            steps,
            lines_cleared: 0,
            mcts_depth_mean: 0.0,
            mcts_search_time_mean: 0.0,
        });

        let mut batched_experience_tensors_opt = None;
        for _ in 0..50 {
            if let Some(batch) = replay_buffer.sample_batch(2, 1.0) {
                batched_experience_tensors_opt = Some(batch);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let batched_experience_tensors = batched_experience_tensors_opt.unwrap();

        train_step(
            &neural_model,
            &ema_model,
            &mut gradient_optimizer,
            &replay_buffer,
            &batched_experience_tensors,
            configuration.unroll_steps,
        );
    }
}
