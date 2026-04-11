use crate::config::Config;
use crate::core::board::GameStateExt;
use crate::mcts::mailbox::{spin_wait, AtomicMailbox};
use crate::mcts::{mcts_search, EvaluationRequest};
use crate::queue::FixedInferenceQueue;
use crate::train::buffer::ReplayBuffer;
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;

pub struct GameLoopExecutionParameters {
    pub configuration: Arc<Config>,
    pub evaluation_transmitter: Arc<FixedInferenceQueue>,
    pub experience_buffer: Arc<ReplayBuffer>,
    pub worker_id: usize,
    pub active_flag: Arc<std::sync::atomic::AtomicBool>,
    pub shared_heatmap: Arc<std::sync::RwLock<[f32; 96]>>,
    pub global_difficulty: Arc<std::sync::atomic::AtomicI32>,
    pub global_gumbel_scale_multiplier: Arc<std::sync::atomic::AtomicU32>,
}

#[hotpath::measure]
pub fn game_loop(parameters: GameLoopExecutionParameters) {
    let configuration = parameters.configuration;
    let evaluation_transmitter = parameters.evaluation_transmitter;
    let experience_buffer = parameters.experience_buffer;
    let worker_id = parameters.worker_id;
    let active_flag = parameters.active_flag;
    let shared_heatmap = parameters.shared_heatmap;
    let global_difficulty = parameters.global_difficulty;
    let global_gumbel_scale_multiplier = parameters.global_gumbel_scale_multiplier;
    let mut thread_rng = rand::thread_rng();
    let _last_spectator_update = std::time::Instant::now();

    loop {
        let current_difficulty = global_difficulty.load(std::sync::atomic::Ordering::Relaxed);
        let mut active_game_state = GameStateExt::new(None, 0, 0, current_difficulty, 0);
        let mut board_history = vec![
            active_game_state.board_bitmask_u128,
            active_game_state.board_bitmask_u128,
        ];
        let mut action_history = Vec::new();
        let mut piece_identifier_history = Vec::new();

        let _last_known_training_steps = 0;

        let mut episode_steps = Vec::with_capacity(100);

        let mut episode_step_count = 0;
        let mut sum_search_time = 0.0_f32;
        let mut sum_depth = 0_usize;
        let initial_mailbox = Arc::new(AtomicMailbox::new());

        for _ in 0..1000 {
            if !active_flag.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }
            if active_game_state.pieces_left == 0 {
                active_game_state.refill_tray();
            }
            if active_game_state.terminal {
                break;
            }

            let board_history_array: [u128; 8] = {
                let mut arr = [0; 8];
                for (i, &b) in board_history.iter().rev().take(8).enumerate() {
                    arr[i] = b;
                }
                arr
            };
            let action_history_array: [i32; 4] = {
                let mut arr = [0; 4];
                for (i, &a) in action_history.iter().rev().take(4).enumerate() {
                    arr[i] = a as i32;
                }
                arr
            };

            if evaluation_transmitter
                .push_batch(
                    worker_id,
                    vec![EvaluationRequest {
                        is_initial: true,
                        board_bitmask: active_game_state.board_bitmask_u128,
                        available_pieces: active_game_state.available,
                        recent_board_history: board_history_array,
                        history_len: std::cmp::min(board_history.len(), 8),
                        recent_action_history: action_history_array,
                        action_history_len: std::cmp::min(action_history.len(), 4),
                        difficulty: current_difficulty,
                        piece_action: 0,
                        piece_id: 0,
                        node_index: 0,
                        generation: 0,
                        worker_id,
                        parent_cache_index: 0,
                        leaf_cache_index: 0,
                        mailbox: initial_mailbox.clone(),
                    }],
                )
                .is_err()
            {
                return;
            }

            evaluation_transmitter
                .blocked_producers
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            let initial_evaluation_response = match spin_wait(&initial_mailbox, &active_flag) {
                Ok(response) => response,
                Err(_) => {
                    evaluation_transmitter
                        .blocked_producers
                        .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    return;
                }
            };

            evaluation_transmitter
                .blocked_producers
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);

            let global_training_steps = experience_buffer
                .state
                .completed_games
                .load(std::sync::atomic::Ordering::Relaxed);
            let multiplier = f32::from_bits(
                global_gumbel_scale_multiplier.load(std::sync::atomic::Ordering::Relaxed),
            );
            let current_gumbel_scale = if global_training_steps < 50_000 {
                configuration.mcts.gumbel_scale
                    * (1.0 - 0.9 * (global_training_steps as f32 / 50_000.0))
            } else {
                configuration.mcts.gumbel_scale * 0.1
            } * multiplier;

            let search_start = std::time::Instant::now();
            let allowed_nodes = (configuration.mcts.simulations as u32 + 32 + 256) * 300;
            let mcts_result = match mcts_search(crate::mcts::MctsParams {
                raw_policy_probabilities: &initial_evaluation_response
                    .child_prior_probabilities_tensor,
                root_cache_index: 0,
                max_tree_nodes: allowed_nodes,
                max_cache_slots: (configuration.mcts.simulations as u32) * 2 + 1000,
                worker_id,
                game_state: &active_game_state,
                total_simulations: configuration.mcts.simulations as usize,
                max_gumbel_k_samples: configuration.mcts.max_gumbel_k as usize,
                gumbel_noise_scale: current_gumbel_scale,
                training_steps: global_training_steps,
                neural_evaluator: &evaluation_transmitter,
                active_flag: active_flag.clone(),
                _seed: None,
                temp_decay_steps: configuration.environment.temp_decay_steps as usize,
                discount_factor: configuration.optimizer.discount_factor,
            }) {
                Ok(result) => result,
                Err(_) => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    break;
                }
            };

            let _search_duration = search_start.elapsed().as_secs_f32() * 1000.0;
            let selected_best_action = mcts_result.0;
            let mcts_visit_distribution = mcts_result.1;
            let latent_value_prediction = mcts_result.2;

            let mut heatmap_data = [0.0_f32; 96];
            for (&action, &visits) in &mcts_visit_distribution {
                let cell_index = (action % 96) as usize;
                heatmap_data[cell_index] += visits as f32;
            }
            let max_val = heatmap_data.iter().copied().fold(0.0_f32, f32::max);
            if max_val > 0.0 {
                for v in &mut heatmap_data {
                    *v /= max_val;
                }
            }
            if let Ok(mut lock) = shared_heatmap.try_write() {
                lock.copy_from_slice(&heatmap_data);
            }

            let _current_max_depth =
                compute_max_depth(&mcts_result.3.arena, mcts_result.3.root_index);

            sum_search_time += _search_duration;
            sum_depth += _current_max_depth;

            if selected_best_action == -1 {
                break;
            }

            let global_training_steps = experience_buffer
                .state
                .completed_games
                .load(std::sync::atomic::Ordering::Relaxed);

            let temperature_decay = get_temperature_decay_factor(
                global_training_steps,
                configuration.environment.temp_decay_steps as usize,
            );
            let mut randomized_action = selected_best_action;
            let (target_policy_probabilities, policy_valid) =
                calculate_policy_targets(&mcts_visit_distribution, temperature_decay);

            if policy_valid {
                randomized_action = sample_action_from_policy(
                    &target_policy_probabilities,
                    &mut thread_rng,
                    selected_best_action,
                );
            }

            let board_slot_index = randomized_action / 96;
            let spatial_position_index = randomized_action % 96;

            let next_game_state = match active_game_state
                .apply_move(board_slot_index as usize, spatial_position_index as usize)
            {
                Some(state) => state,
                None => break,
            };

            let theoretical_max_score = (configuration.environment.difficulty as f32) * 100.0;
            let raw_value_prefix = (next_game_state.score - active_game_state.score) as f32;
            let value_prefix_received = raw_value_prefix / theoretical_max_score;

            let piece_identifier = if active_game_state.available[board_slot_index as usize] == -1 {
                0
            } else {
                active_game_state.available[board_slot_index as usize]
            };
            let composite_action_identifier = piece_identifier * 96 + spatial_position_index;
            episode_steps.push(crate::train::buffer::GameStep {
                board_state: [
                    (active_game_state.board_bitmask_u128 & 0xFFFFFFFFFFFFFFFF) as u64,
                    (active_game_state.board_bitmask_u128 >> 64) as u64,
                ],
                available_pieces: active_game_state.available,
                action_taken: composite_action_identifier as i64,
                piece_identifier: piece_identifier as i64,
                value_prefix_received,
                policy_target: target_policy_probabilities,
                value_target: latent_value_prediction,
            });

            board_history.push(active_game_state.board_bitmask_u128);
            if board_history.len() > 8 {
                board_history.remove(0);
            }

            action_history.push(composite_action_identifier as i64);
            piece_identifier_history.push(piece_identifier);
            active_game_state = next_game_state;
            episode_step_count += 1;
        }

        if episode_step_count > 0 {
            let mcts_depth_mean = sum_depth as f32 / episode_step_count as f32;
            let mcts_search_time_mean = sum_search_time / episode_step_count as f32;

            experience_buffer.add_game(crate::train::buffer::OwnedGameData {
                difficulty_setting: configuration.environment.difficulty,
                episode_score: active_game_state.score as f32,
                steps: episode_steps,
                lines_cleared: active_game_state.total_lines_cleared as u32,
                mcts_depth_mean,
                mcts_search_time_mean,
            });
        }
    }
}

fn compute_max_depth(arena: &[crate::node::LatentNode], node_idx: usize) -> usize {
    if node_idx == usize::MAX {
        return 0;
    }
    let mut max_child_depth = 0;
    let mut child = arena[node_idx]
        .first_child
        .load(std::sync::atomic::Ordering::Relaxed);
    while child != u32::MAX {
        let depth = compute_max_depth(arena, child as usize);
        if depth > max_child_depth {
            max_child_depth = depth;
        }
        child = arena[child as usize]
            .next_sibling
            .load(std::sync::atomic::Ordering::Relaxed);
    }
    max_child_depth + 1
}

#[hotpath::measure]
fn get_temperature_decay_factor(current_step: usize, temperature_decay_steps: usize) -> f32 {
    if current_step < temperature_decay_steps / 2 {
        1.0
    } else if current_step < temperature_decay_steps {
        0.5
    } else {
        0.1
    }
}

#[hotpath::measure]
fn calculate_policy_targets(
    mcts_visit_distribution: &HashMap<i32, i32>,
    temperature_decay: f32,
) -> (Vec<f32>, bool) {
    let mut target_policy_probabilities = vec![0.0f32; 288];
    let mut sum_probabilities = 0.0;

    for (action, visits) in mcts_visit_distribution {
        let powered_probability =
            (*visits as f64).powf(1.0 / (temperature_decay as f64 + 1e-8)) as f32;
        target_policy_probabilities[*action as usize] = powered_probability;
        sum_probabilities += powered_probability;
    }

    if sum_probabilities > 0.0 {
        for probability in target_policy_probabilities.iter_mut() {
            *probability /= sum_probabilities;
        }
        let check_sum: f32 = target_policy_probabilities.iter().sum();
        assert!(
            (check_sum - 1.0).abs() < 1e-4,
            "Normalized target policies must sum to 1.0. Actual: {}",
            check_sum
        );

        (target_policy_probabilities, true)
    } else {
        let uniform_probability = 1.0 / (mcts_visit_distribution.len() as f32);
        for action in mcts_visit_distribution.keys() {
            target_policy_probabilities[*action as usize] = uniform_probability;
        }
        (target_policy_probabilities, false) // Failed normalization
    }
}

#[hotpath::measure]
fn sample_action_from_policy(
    target_policy_probabilities: &[f32],
    random_generator: &mut rand::rngs::ThreadRng,
    default_action: i32,
) -> i32 {
    let uniform_random_sample: f32 = random_generator.gen_range(0.0..=1.0);
    let mut cumulative_probability = 0.0;
    let mut selected_action = default_action;
    for (action_index, &probability) in target_policy_probabilities.iter().enumerate() {
        if probability > 0.0 {
            cumulative_probability += probability;
            if uniform_random_sample <= cumulative_probability {
                selected_action = action_index as i32;
                break;
            }
        }
    }
    selected_action
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::worker::inference::{inference_loop, InferenceLoopParams};
    use arc_swap::ArcSwap;
    use std::sync::Arc;
    use tch::{nn, Device};

    #[test]
    fn test_temperature_decay_schedule() {
        let decay_steps = 30;
        assert_eq!(get_temperature_decay_factor(0, decay_steps), 1.0);
        assert_eq!(get_temperature_decay_factor(14, decay_steps), 1.0);
        assert_eq!(get_temperature_decay_factor(15, decay_steps), 0.5);
        assert_eq!(get_temperature_decay_factor(29, decay_steps), 0.5);
        assert_eq!(get_temperature_decay_factor(30, decay_steps), 0.1);
        assert_eq!(get_temperature_decay_factor(100, decay_steps), 0.1);
    }

    #[test]
    fn test_inference_loop_dynamic_batching_and_disconnect() {
        let inference_queue = crate::queue::FixedInferenceQueue::new(3, 3);
        let variable_store = nn::VarStore::new(Device::Cpu);
        let model_dimension = 16;
        let p_net = Arc::new(crate::net::MuZeroNet::new(
            &variable_store.root(),
            model_dimension,
            1,
            200,
            200,
            crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
            64,
        ));
        let neural_model = Arc::new(ArcSwap::from(p_net));

        let response_mailboxes = Vec::new();
        for i in 0..3 {
            let mailbox = Arc::new(AtomicMailbox::new());
            inference_queue
                .push_batch(
                    i,
                    vec![EvaluationRequest {
                        is_initial: true,
                        board_bitmask: 0,
                        available_pieces: [-1; 3],
                        recent_board_history: [0; 8],
                        history_len: 0,
                        recent_action_history: [0; 4],
                        action_history_len: 0,
                        difficulty: 6,
                        piece_action: 0,
                        piece_id: 0,
                        node_index: 0,
                        generation: 0,
                        worker_id: i,
                        parent_cache_index: 0,
                        leaf_cache_index: 0,
                        mailbox: mailbox.clone(),
                    }],
                )
                .unwrap();
        }

        inference_queue.disconnect_producer();
        inference_queue.disconnect_producer();
        inference_queue.disconnect_producer();
        inference_loop(InferenceLoopParams {
            receiver_queue: inference_queue.clone(),
            shared_neural_model: neural_model,
            cmodule_inference: None,
            model_dimension,
            computation_device: Device::Cpu,
            total_workers: 3,
            maximum_allowed_nodes_in_search_tree: 202,
            inference_batch_size_limit: 1024,
            inference_timeout_milliseconds: 10,
            active_flag: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            shared_queue_saturation: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        });

        for mailbox in response_mailboxes {
            let active_flag = Arc::new(std::sync::atomic::AtomicBool::new(true));
            let evaluator_response: crate::mcts::EvaluationResponse =
                crate::mcts::mailbox::spin_wait(&mailbox, &active_flag)
                    .expect("Failed to receive batched response");
            assert_eq!(
                evaluator_response.child_prior_probabilities_tensor.len(),
                288
            );
        }
    }

    #[test]
    fn test_compute_max_depth() {
        let arena = vec![
            crate::node::LatentNode::new(1.0, -1, 0), // 0: Root
            crate::node::LatentNode::new(0.5, 0, 0),  // 1: Child of Root
            crate::node::LatentNode::new(0.5, 1, 0),  // 2: Child of 1 (Sibling of 3)
            crate::node::LatentNode::new(0.5, 2, 0),  // 3: Child of 1 (Sibling of 2)
            crate::node::LatentNode::new(0.5, 3, 0),  // 4: Sibling of 1
            crate::node::LatentNode::new(0.5, 4, 0),  // 5: Child of 4
        ];

        // Link Root (0) -> [1, 4]
        arena[0]
            .first_child
            .store(1, std::sync::atomic::Ordering::SeqCst);
        arena[1]
            .next_sibling
            .store(4, std::sync::atomic::Ordering::SeqCst);

        // Link Node 1 -> [2, 3]
        arena[1]
            .first_child
            .store(2, std::sync::atomic::Ordering::SeqCst);
        arena[2]
            .next_sibling
            .store(3, std::sync::atomic::Ordering::SeqCst);

        // Link Node 4 -> [5]
        arena[4]
            .first_child
            .store(5, std::sync::atomic::Ordering::SeqCst);

        // Depth: Root(1) -> 1(2) -> 2(3) -> Max depth should be 3
        let depth = compute_max_depth(&arena, 0);
        assert_eq!(depth, 3);

        // Root with no children
        let arena2 = vec![crate::node::LatentNode::new(1.0, -1, 0)];
        assert_eq!(compute_max_depth(&arena2, 0), 1);
    }
}
