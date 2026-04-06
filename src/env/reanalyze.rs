use crossbeam_channel::unbounded;
use std::sync::Arc;

use crate::config::Config;
use crate::train::buffer::ReplayBuffer;

use crate::mcts::{mcts_search, EvaluationRequest, MctsParams};
use crate::queue::FixedInferenceQueue;

pub fn reanalyze_worker_loop(
    configuration: Arc<Config>,
    inference_queue: Arc<FixedInferenceQueue>,
    shared_replay_buffer: Arc<ReplayBuffer>,
    worker_id: usize,
    active_flag: Arc<std::sync::atomic::AtomicBool>,
) {
    loop {
        if !active_flag.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        if shared_replay_buffer.get_length() < configuration.train_batch_size {
            std::thread::sleep(std::time::Duration::from_millis(100));
            continue;
        }

        let mut samples = Vec::new();
        for _ in 0..64 {
            if let Some(sample) = shared_replay_buffer.sample_for_reanalyze() {
                if !sample.1.terminal {
                    samples.push(sample);
                }
            }
        }

        if samples.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(10));
            continue;
        }

        let _: Vec<_> = std::thread::scope(|s| {
            let mut handles = Vec::new();
            for (circular_idx, game_state) in samples.into_iter() {
                let inference_queue_clone = inference_queue.clone();
                let shared_replay_buffer_ref = &*shared_replay_buffer;
                let config_ref = &*configuration;
                let active_flag_clone = active_flag.clone();

                handles.push(s.spawn(move || {
                    let (response_tx, response_rx) = unbounded();

                    let history_boards = shared_replay_buffer_ref
                        .state
                        .get_historical_boards(circular_idx);
                    let action_history = shared_replay_buffer_ref
                        .state
                        .get_historical_actions(circular_idx);

                    let mut board_history_array: [u128; 8] = [0; 8];
                    for (i, &b) in history_boards.iter().take(8).enumerate() {
                        board_history_array[i] = b;
                    }

                    let mut action_history_array: [i32; 4] = [0; 4];
                    for (i, &a) in action_history.iter().take(4).enumerate() {
                        action_history_array[i] = a;
                    }

                    if inference_queue_clone
                        .push_batch(
                            worker_id,
                            vec![EvaluationRequest {
                                is_initial: true,
                                board_bitmask: game_state.board_bitmask_u128,
                                available_pieces: game_state.available,
                                recent_board_history: board_history_array,
                                history_len: std::cmp::min(history_boards.len(), 8),
                                recent_action_history: action_history_array,
                                action_history_len: std::cmp::min(action_history.len(), 4),
                                difficulty: game_state.difficulty,
                                piece_action: 0,
                                piece_id: 0,
                                node_index: 0,
                                generation: 0,
                                worker_id,
                                parent_cache_index: 0,
                                leaf_cache_index: 0,
                                evaluation_request_transmitter: response_tx.clone(),
                            }],
                        )
                        .is_err()
                    {
                        return;
                    }

                    let initial_eval = loop {
                        if !active_flag_clone.load(std::sync::atomic::Ordering::Relaxed) {
                            return;
                        }
                        match response_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                            Ok(resp) => break resp,
                            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                            Err(_) => return,
                        }
                    };

                    let allowed_nodes = (config_ref.simulations as u32 + 32 + 256) * 300;
                    let mcts_params = MctsParams {
                        raw_policy_probabilities: &initial_eval.child_prior_probabilities_tensor,
                        root_cache_index: 0,
                        max_tree_nodes: allowed_nodes,
                        max_cache_slots: (config_ref.simulations as u32) * 2 + 1000,
                        worker_id,
                        game_state: &game_state,
                        total_simulations: config_ref.simulations as usize,
                        max_gumbel_k_samples: 8,
                        gumbel_noise_scale: 1.0,
                        training_steps: 0, // Not primarily for exploitation during reanalyze
                        neural_evaluator: &inference_queue_clone,
                        evaluation_request_transmitter: response_tx.clone(),
                        evaluation_response_receiver: &response_rx,
                        active_flag: active_flag_clone,
                        _seed: None,
                    };

                    if let Ok((_action, visit_counts, value, _tree)) = mcts_search(mcts_params) {
                        let mut target_policy = [0.0f32; 288];
                        let total_visits: f32 = visit_counts.values().map(|&v| v as f32).sum();
                        if total_visits > 0.0 {
                            for (&a, &v) in &visit_counts {
                                if a >= 0 && (a as usize) < 288 {
                                    target_policy[a as usize] = v as f32 / total_visits;
                                }
                            }
                        }

                        shared_replay_buffer_ref.update_reanalyzed_targets(
                            circular_idx,
                            target_policy,
                            value,
                        );
                    }
                }));
            }
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
    }
}
