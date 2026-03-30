use crossbeam_channel::unbounded;
use std::sync::Arc;

use crate::buffer::replay::ReplayBuffer;
use crate::config::Config;

use crate::mcts::{mcts_search, EvalReq, MctsParams};
use crate::queue::FixedInferenceQueue;

use std::sync::RwLock;

pub fn reanalyze_worker_loop(
    configuration: Arc<Config>,
    inference_queue: Arc<FixedInferenceQueue>,
    shared_replay_buffer: Arc<ReplayBuffer>,
    worker_id: usize,
    active_flag: Arc<RwLock<bool>>,
) {
    loop {
        if !*active_flag.read().unwrap() {
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

        std::thread::scope(|s| {
            for (circular_idx, game_state) in samples {
                let inference_queue_clone = inference_queue.clone();
                let shared_replay_buffer_ref = &*shared_replay_buffer;
                let config_ref = &*configuration;
                let active_flag_clone = active_flag.clone();

                s.spawn(move || {
                    let (response_tx, response_rx) = unbounded();

                    if inference_queue_clone
                        .push_batch(
                            worker_id,
                            vec![EvalReq {
                                is_initial: true,
                                board_bitmask: game_state.board_bitmask_u128,
                                available_pieces: game_state.available,
                                recent_board_history: [0; 8],
                                history_len: 0,
                                recent_action_history: [0; 4],
                                action_history_len: 0,
                                difficulty: game_state.difficulty,
                                piece_action: 0,
                                piece_id: 0,
                                node_index: 0,
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
                        if !*active_flag_clone.read().unwrap() {
                            return;
                        }
                        match response_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                            Ok(resp) => break resp,
                            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                            Err(_) => return,
                        }
                    };

                    let mcts_params = MctsParams {
                        raw_policy_probabilities: &initial_eval.child_prior_probabilities_tensor,
                        root_cache_index: 0,
                        maximum_allowed_nodes_in_search_tree: (config_ref.simulations as u32) + 300,
                        worker_id,
                        game_state: &game_state,
                        total_simulations: config_ref.simulations as usize,
                        max_gumbel_k_samples: 8,
                        gumbel_noise_scale: 1.0,
                        previous_tree: None,
                        last_executed_action: None,
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
                });
            }
        });
    }
}
