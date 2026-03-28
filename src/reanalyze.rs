use crossbeam_channel::unbounded;
use std::sync::Arc;

use crate::buffer::replay::ReplayBuffer;
use crate::config::Config;
use crate::features::extract_feature_native;
use crate::mcts::{mcts_search, EvalReq, MctsParams};
use crate::queue::FixedInferenceQueue;

pub fn reanalyze_worker_loop(
    configuration: Arc<Config>,
    inference_queue: Arc<FixedInferenceQueue>,
    shared_replay_buffer: Arc<ReplayBuffer>,
    worker_id: usize,
) {
    loop {
        if shared_replay_buffer.get_length() < configuration.train_batch_size {
            std::thread::sleep(std::time::Duration::from_millis(100));
            continue;
        }

        let sample = shared_replay_buffer.sample_for_reanalyze();
        if let Some((circular_idx, game_state)) = sample {
            if game_state.terminal {
                continue;
            }

            let features_array =
                extract_feature_native(&game_state, None, None, game_state.difficulty);
            let (response_tx, response_rx) = unbounded();

            if inference_queue
                .push(
                    worker_id,
                    EvalReq {
                        is_initial: true,
                        state_feat: Some(features_array),
                        piece_action: 0,
                        piece_id: 0,
                        node_index: 0,
                        worker_id,
                        parent_cache_index: 0,
                        leaf_cache_index: 0,
                        tx: response_tx.clone(),
                    },
                )
                .is_err()
            {
                inference_queue.disconnect_producer();
                return;
            }

            let initial_eval = match response_rx.recv() {
                Ok(resp) => resp,
                Err(_) => {
                    inference_queue.disconnect_producer();
                    return;
                }
            };

            let mcts_params = MctsParams {
                raw_policy_probabilities: &initial_eval.p_next,
                root_cache_index: 0,
                max_nodes: configuration.simulations as u32,
                worker_id,
                game_state: &game_state,
                total_simulations: configuration.simulations as usize,
                max_gumbel_k_samples: 8,
                gumbel_noise_scale: 1.0,
                previous_tree: None,
                last_executed_action: None,
                neural_evaluator: &inference_queue,
                eval_tx: response_tx.clone(),
                eval_rx: &response_rx,
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

                shared_replay_buffer.update_reanalyzed_targets(circular_idx, target_policy, value);
            }
        } else {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
}
