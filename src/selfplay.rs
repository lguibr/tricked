use crate::queue::FixedInferenceQueue;
use crossbeam_channel::unbounded;
use rand::Rng;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tch::{Device, Kind, Tensor};

use crate::board::GameStateExt;
use crate::buffer::ReplayBuffer;
use crate::config::Config;
use crate::features::extract_feature_native;
use crate::mcts::{mcts_search, EvalReq, EvalResp};
use crate::network::MuZeroNet;
use crate::web::TelemetryStore;

struct SafeTensorGuard<'a, T> {
    _tensor: &'a Tensor,
    pub slice: &'a mut [T],
}

impl<'a, T> SafeTensorGuard<'a, T> {
    fn new(tensor: &'a Tensor, len: usize) -> Self {
        Self {
            _tensor: tensor,
            slice: unsafe { std::slice::from_raw_parts_mut(tensor.data_ptr() as *mut T, len) },
        }
    }
}

impl<'a, T> Drop for SafeTensorGuard<'a, T> {
    fn drop(&mut self) {}
}

pub fn inference_loop(
    receiver_queue: Arc<FixedInferenceQueue>,
    shared_neural_model: Arc<RwLock<MuZeroNet>>,
    cmodule_inference: Option<Arc<tch::CModule>>,
    model_dimension: i64,
    computation_device: Device,
    total_workers: usize,
    max_nodes: usize,
) {
    let flat_cache_size = (total_workers * max_nodes) as i64;
    let mut latent_cache = Tensor::zeros(
        [flat_cache_size, model_dimension, 8, 8],
        (Kind::Float, computation_device),
    );

    let mut pinned_initial_states = Tensor::zeros([1024, 20, 8, 16], (Kind::Float, Device::Cpu));
    let mut pinned_recurrent_actions = Tensor::zeros([1024], (Kind::Int64, Device::Cpu));
    let mut pinned_recurrent_ids = Tensor::zeros([1024], (Kind::Int64, Device::Cpu));
    let mut pinned_workers = Tensor::zeros([1024], (Kind::Int64, Device::Cpu));
    let mut pinned_parents = Tensor::zeros([1024], (Kind::Int64, Device::Cpu));
    let mut pinned_nodes = Tensor::zeros([1024], (Kind::Int64, Device::Cpu));

    if computation_device.is_cuda() {
        pinned_initial_states = pinned_initial_states.pin_memory(computation_device);
        pinned_recurrent_actions = pinned_recurrent_actions.pin_memory(computation_device);
        pinned_recurrent_ids = pinned_recurrent_ids.pin_memory(computation_device);
        pinned_workers = pinned_workers.pin_memory(computation_device);
        pinned_parents = pinned_parents.pin_memory(computation_device);
        pinned_nodes = pinned_nodes.pin_memory(computation_device);
    }

    loop {
        let batched_requests =
            match receiver_queue.pop_batch_timeout(1024, std::time::Duration::from_millis(10)) {
                Ok(reqs) if !reqs.is_empty() => reqs,
                Ok(_) => continue,
                Err(_) => break,
            };

        let mut initial_inference_requests = Vec::new();
        let mut recurrent_inference_requests = Vec::new();

        for request in batched_requests.into_iter() {
            if request.is_initial {
                initial_inference_requests.push(request);
            } else {
                recurrent_inference_requests.push(request);
            }
        }

        tch::no_grad(|| {
            tch::autocast(true, || {
                let neural_model = shared_neural_model.read().unwrap();

                if !initial_inference_requests.is_empty() {
                    process_initial_inference(
                        &neural_model,
                        cmodule_inference.as_deref(),
                        initial_inference_requests,
                        max_nodes,
                        computation_device,
                        &mut latent_cache,
                        &mut pinned_initial_states,
                        &mut pinned_workers,
                        &mut pinned_nodes,
                    );
                }

                if !recurrent_inference_requests.is_empty() {
                    process_recurrent_inference(
                        &neural_model,
                        cmodule_inference.as_deref(),
                        recurrent_inference_requests,
                        max_nodes,
                        computation_device,
                        &mut latent_cache,
                        &mut pinned_recurrent_actions,
                        &mut pinned_recurrent_ids,
                        &mut pinned_workers,
                        &mut pinned_parents,
                        &mut pinned_nodes,
                    );
                }
            });
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn process_initial_inference(
    neural_model: &MuZeroNet,
    cmodule_inference: Option<&tch::CModule>,
    inference_requests: Vec<EvalReq>,
    max_nodes: usize,
    computation_device: Device,
    latent_cache: &mut Tensor,
    pinned_initial_states: &mut Tensor,
    pinned_workers: &mut Tensor,
    pinned_nodes: &mut Tensor,
) {
    let batch_size = inference_requests.len();

    let state_view = pinned_initial_states.narrow(0, 0, batch_size as i64);
    {
        let guard = SafeTensorGuard::<f32>::new(&state_view, batch_size * 20 * 8 * 16);
        let mut offset = 0;
        for request in &inference_requests {
            let features_array = request.state_feat.as_ref().unwrap();
            let len = features_array.len();
            guard.slice[offset..offset + len].copy_from_slice(features_array);
            offset += len;
        }
    }

    let state_batch = state_view.to_device(computation_device);

    let (hidden_state_batch, value_batch, policy_batch, _) = if let Some(cmod) = cmodule_inference {
        let ivalue = cmod
            .method_is(
                "initial_inference",
                &[tch::IValue::Tensor(state_batch.copy())],
            )
            .unwrap();
        if let tch::IValue::Tuple(mut tup) = ivalue {
            let reward = if tup.len() == 4 {
                if let tch::IValue::Tensor(r) = tup.remove(3) {
                    r
                } else {
                    unreachable!()
                }
            } else {
                Tensor::zeros([1], (Kind::Float, Device::Cpu))
            };
            let policy = if let tch::IValue::Tensor(p) = tup.remove(2) {
                p
            } else {
                unreachable!()
            };
            let value = if let tch::IValue::Tensor(v) = tup.remove(1) {
                v
            } else {
                unreachable!()
            };
            let hidden = if let tch::IValue::Tensor(h) = tup.remove(0) {
                h
            } else {
                unreachable!()
            };
            (hidden, value, policy, reward)
        } else {
            panic!("Expected Tuple from initial_inference");
        }
    } else {
        neural_model.initial_inference(&state_batch)
    };

    // Save to GPU cache
    let w_view = pinned_workers.narrow(0, 0, batch_size as i64);
    let n_view = pinned_nodes.narrow(0, 0, batch_size as i64);
    {
        let w_guard = SafeTensorGuard::<i64>::new(&w_view, batch_size);
        let n_guard = SafeTensorGuard::<i64>::new(&n_view, batch_size);
        for (i, req) in inference_requests.iter().enumerate() {
            w_guard.slice[i] = req.worker_id as i64;
            n_guard.slice[i] = req.leaf_cache_index as i64;
        }
    }

    let w_tensor = w_view.to_device(computation_device);
    let n_tensor = n_view.to_device(computation_device);

    let max_nodes_tensor = Tensor::from_slice(&[max_nodes as i64]).to_device(computation_device);
    let flat_n_indices = (&w_tensor * &max_nodes_tensor) + &n_tensor;

    let _ = latent_cache.index_copy_(0, &flat_n_indices, &hidden_state_batch);

    let value_predictions_cpu = value_batch.to_device(Device::Cpu).to_kind(Kind::Float);
    let policy_predictions_cpu = policy_batch.to_device(Device::Cpu).to_kind(Kind::Float);

    let value_predictions_f32: Vec<f32> = value_predictions_cpu
        .reshape([-1i64])
        .try_into()
        .unwrap_or_default();
    let policy_predictions_f32: Vec<f32> = policy_predictions_cpu
        .reshape([-1i64])
        .try_into()
        .unwrap_or_default();

    let policy_vector_size = 288;
    for (index, request) in inference_requests.into_iter().enumerate() {
        let start_policy = index * policy_vector_size;
        let end_policy = (index + 1) * policy_vector_size;

        let response = EvalResp {
            reward: 0.0,
            value: value_predictions_f32[index],
            p_next: policy_predictions_f32[start_policy..end_policy].to_vec(),
            node_index: request.node_index,
        };
        let _ = request.tx.send(response);
    }
}

#[allow(clippy::too_many_arguments)]
fn process_recurrent_inference(
    neural_model: &MuZeroNet,
    cmodule_inference: Option<&tch::CModule>,
    inference_requests: Vec<EvalReq>,
    max_nodes: usize,
    computation_device: Device,
    latent_cache: &mut Tensor,
    pinned_actions: &mut Tensor,
    pinned_ids: &mut Tensor,
    pinned_workers: &mut Tensor,
    pinned_parents: &mut Tensor,
    pinned_nodes: &mut Tensor,
) {
    let batch_size = inference_requests.len();

    let actions_view = pinned_actions.narrow(0, 0, batch_size as i64);
    let ids_view = pinned_ids.narrow(0, 0, batch_size as i64);
    let w_view = pinned_workers.narrow(0, 0, batch_size as i64);
    let p_view = pinned_parents.narrow(0, 0, batch_size as i64);
    let n_view = pinned_nodes.narrow(0, 0, batch_size as i64);

    {
        let actions_guard = SafeTensorGuard::<i64>::new(&actions_view, batch_size);
        let ids_guard = SafeTensorGuard::<i64>::new(&ids_view, batch_size);
        let w_guard = SafeTensorGuard::<i64>::new(&w_view, batch_size);
        let p_guard = SafeTensorGuard::<i64>::new(&p_view, batch_size);
        let n_guard = SafeTensorGuard::<i64>::new(&n_view, batch_size);

        for (i, request) in inference_requests.iter().enumerate() {
            actions_guard.slice[i] = request.piece_action;
            ids_guard.slice[i] = request.piece_id;
            w_guard.slice[i] = request.worker_id as i64;
            p_guard.slice[i] = request.parent_cache_index as i64;
            n_guard.slice[i] = request.leaf_cache_index as i64;
        }
    }

    let piece_action_batch = actions_view.to_device(computation_device);
    let piece_identifier_batch = ids_view.to_device(computation_device);
    let w_tensor = w_view.to_device(computation_device);
    let p_tensor = p_view.to_device(computation_device);
    let n_tensor = n_view.to_device(computation_device);

    let max_nodes_tensor = Tensor::from_slice(&[max_nodes as i64]).to_device(computation_device);
    let flat_p_indices = (&w_tensor * &max_nodes_tensor) + &p_tensor;
    let flat_n_indices = (&w_tensor * &max_nodes_tensor) + &n_tensor;

    let hidden_state_batch = latent_cache.index_select(0, &flat_p_indices);

    let (hidden_state_next_batch, reward_batch, value_batch, policy_batch, _) =
        if let Some(cmod) = cmodule_inference {
            let ivalue = cmod
                .method_is(
                    "recurrent_inference",
                    &[
                        tch::IValue::Tensor(hidden_state_batch.copy()),
                        tch::IValue::Tensor(piece_action_batch.copy()),
                        tch::IValue::Tensor(piece_identifier_batch.copy()),
                    ],
                )
                .unwrap();
            if let tch::IValue::Tuple(mut tup) = ivalue {
                let extra = if tup.len() == 5 {
                    if let tch::IValue::Tensor(e) = tup.remove(4) {
                        e
                    } else {
                        unreachable!()
                    }
                } else {
                    Tensor::zeros([1], (Kind::Float, Device::Cpu))
                };
                let policy = if let tch::IValue::Tensor(p) = tup.remove(3) {
                    p
                } else {
                    unreachable!()
                };
                let value = if let tch::IValue::Tensor(v) = tup.remove(2) {
                    v
                } else {
                    unreachable!()
                };
                let reward = if let tch::IValue::Tensor(r) = tup.remove(1) {
                    r
                } else {
                    unreachable!()
                };
                let hidden = if let tch::IValue::Tensor(h) = tup.remove(0) {
                    h
                } else {
                    unreachable!()
                };
                (hidden, reward, value, policy, extra)
            } else {
                panic!("Expected Tuple from recurrent_inference");
            }
        } else {
            neural_model.recurrent_inference(
                &hidden_state_batch,
                &piece_action_batch,
                &piece_identifier_batch,
            )
        };

    let _ = latent_cache.index_copy_(0, &flat_n_indices, &hidden_state_next_batch);

    let reward_predictions_cpu = reward_batch.to_device(Device::Cpu).to_kind(Kind::Float);
    let value_predictions_cpu = value_batch.to_device(Device::Cpu).to_kind(Kind::Float);
    let policy_predictions_cpu = policy_batch.to_device(Device::Cpu).to_kind(Kind::Float);

    let reward_predictions_f32: Vec<f32> = reward_predictions_cpu
        .reshape([-1i64])
        .try_into()
        .unwrap_or_default();
    let value_predictions_f32: Vec<f32> = value_predictions_cpu
        .reshape([-1i64])
        .try_into()
        .unwrap_or_default();
    let policy_predictions_f32: Vec<f32> = policy_predictions_cpu
        .reshape([-1i64])
        .try_into()
        .unwrap_or_default();

    let policy_vector_size = 288;
    for (index, request) in inference_requests.into_iter().enumerate() {
        let start_policy = index * policy_vector_size;
        let end_policy = (index + 1) * policy_vector_size;

        let response = EvalResp {
            reward: reward_predictions_f32[index],
            value: value_predictions_f32[index],
            p_next: policy_predictions_f32[start_policy..end_policy].to_vec(),
            node_index: request.node_index,
        };
        let _ = request.tx.send(response);
    }
}

pub fn game_loop(
    configuration: Arc<Config>,
    evaluation_transmitter: Arc<FixedInferenceQueue>,
    experience_buffer: Arc<ReplayBuffer>,
    telemetry_store: Arc<RwLock<TelemetryStore>>,
    game_logger: Arc<dyn crate::telemetry::GameLogger>,
    worker_id: usize,
) {
    let mut thread_rng = rand::thread_rng();
    let mut last_spectator_update = std::time::Instant::now();
    let mut local_games_played = 0;

    loop {
        let mut active_game_state = GameStateExt::new(None, 0, 0, configuration.difficulty, 0);
        let mut board_history = vec![active_game_state.board, active_game_state.board];
        let mut action_history = Vec::new();
        let mut piece_identifier_history = Vec::new();

        let mut current_tree: Option<crate::mcts::MctsTree> = None;
        let mut last_action: Option<i32> = None;
        let mut last_known_training_steps = 0;

        let mut episode_boards = Vec::new();
        let mut episode_available = Vec::new();
        let mut episode_actions = Vec::new();
        let mut episode_piece_ids = Vec::new();
        let mut episode_rewards = Vec::new();
        let mut episode_policies = Vec::new();
        let mut episode_values = Vec::new();
        let mut episode_features = Vec::new();

        let mut episode_step_count = 0;
        let (response_tx, response_rx) = unbounded();

        for _ in 0..1000 {
            if active_game_state.pieces_left == 0 {
                active_game_state.refill_tray();
            }
            if active_game_state.terminal {
                break;
            }

            let features_array = extract_feature_native(
                &active_game_state,
                Some(board_history.clone()),
                Some(action_history.clone()),
                configuration.difficulty,
            );
            episode_features.push(features_array.clone());

            if evaluation_transmitter
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
                return;
            }

            let initial_evaluation_response = match response_rx.recv() {
                Ok(response) => response,
                Err(_) => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    break;
                }
            };

            let search_start = std::time::Instant::now();
            let mcts_result = match mcts_search(crate::mcts::MctsParams {
                raw_policy_probabilities: &initial_evaluation_response.p_next,
                root_cache_index: 0,
                max_nodes: (configuration.simulations * 2) as u32 + 10,
                worker_id,
                game_state: &active_game_state,
                total_simulations: configuration.simulations as usize,
                max_gumbel_k_samples: configuration.max_gumbel_k as usize,
                gumbel_noise_scale: configuration.gumbel_scale,
                previous_tree: current_tree.take(),
                last_executed_action: last_action,
                neural_evaluator: &evaluation_transmitter,
                eval_tx: response_tx.clone(),
                eval_rx: &response_rx,
                _seed: None,
            }) {
                Ok(result) => result,
                Err(_) => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    break;
                }
            };

            let search_duration = search_start.elapsed().as_secs_f32() * 1000.0;
            let selected_best_action = mcts_result.0;
            let mcts_visit_distribution = mcts_result.1;
            let latent_value_prediction = mcts_result.2;

            if thread_rng.gen_ratio(1, 20) {
                let current_max_depth =
                    compute_max_depth(&mcts_result.3.arena, mcts_result.3.root_index);

                game_logger.log_metric("mcts/search_time_ms", search_duration);
                game_logger.log_metric("mcts/value_prediction", latent_value_prediction);
                game_logger.log_metric("mcts/max_depth", current_max_depth as f32);
            }

            last_action = Some(selected_best_action);
            current_tree = Some(mcts_result.3);

            if selected_best_action == -1 {
                break;
            }

            if last_spectator_update.elapsed() > std::time::Duration::from_millis(500) {
                if let Ok(mut telemetry) = telemetry_store.try_write() {
                    telemetry.spectator_state = Some(active_game_state.clone());
                }
                last_spectator_update = std::time::Instant::now();
            }

            if let Ok(tel) = telemetry_store.try_read() {
                last_known_training_steps = tel.status.training_steps as usize;
            }
            let global_training_steps = last_known_training_steps;

            let temperature_decay = get_temperature_decay_factor(
                global_training_steps,
                configuration.temp_decay_steps as usize,
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

            let theoretical_max_score = (configuration.difficulty as f32) * 100.0;
            let raw_reward = (next_game_state.score - active_game_state.score) as f32;
            let reward_received = raw_reward / theoretical_max_score;

            let piece_identifier = if active_game_state.available[board_slot_index as usize] == -1 {
                0
            } else {
                active_game_state.available[board_slot_index as usize]
            };
            let composite_action_identifier = piece_identifier * 96 + spatial_position_index;

            episode_boards.push([
                (active_game_state.board & 0xFFFFFFFFFFFFFFFF) as u64,
                (active_game_state.board >> 64) as u64,
            ]);
            episode_available.push(active_game_state.available);
            episode_actions.push(composite_action_identifier as i64);
            episode_piece_ids.push(piece_identifier as i64);
            episode_rewards.push(reward_received);

            let mut rigid_policy_array = [0.0f32; 288];
            rigid_policy_array.copy_from_slice(&target_policy_probabilities);
            episode_policies.push(rigid_policy_array);
            episode_values.push(latent_value_prediction);

            board_history.push(active_game_state.board);
            if board_history.len() > 8 {
                board_history.remove(0);
            }

            action_history.push(composite_action_identifier);
            piece_identifier_history.push(piece_identifier);
            active_game_state = next_game_state;
            episode_step_count += 1;
        }

        if episode_step_count > 0 {
            game_logger.log_metric("game/score", active_game_state.score as f32);
            game_logger.log_metric("game/episode_length", episode_step_count as f32);
            game_logger.log_metric("game/difficulty", configuration.difficulty as f32);

            game_logger.log_game_end(
                configuration.difficulty,
                active_game_state.score as f32,
                episode_step_count as i32,
            );

            let mut current_games_count = local_games_played;

            if let Ok(mut tel) = telemetry_store.try_write() {
                tel.status.games_played += 1;
                current_games_count = tel.status.games_played as usize;

                tel.top_games.insert(
                    0,
                    crate::buffer::state::EpisodeMeta {
                        global_start_idx: current_games_count,
                        length: episode_step_count,
                        difficulty: configuration.difficulty,
                        score: active_game_state.score as f32,
                    },
                );
                if tel.top_games.len() > 20 {
                    tel.top_games.truncate(20);
                }
            }

            game_logger.log_trajectory(current_games_count, &episode_features);
            local_games_played += 1;

            experience_buffer.add_game(crate::buffer::replay::OwnedGameData {
                difficulty_setting: configuration.difficulty,
                episode_score: active_game_state.score as f32,
                board_states: episode_boards,
                available_pieces: episode_available,
                actions_taken: episode_actions,
                piece_identifiers: episode_piece_ids,
                rewards_received: episode_rewards,
                policy_targets: episode_policies,
                value_targets: episode_values,
            });
        }
    }
}

fn compute_max_depth(arena: &[crate::node::LatentNode], node_idx: usize) -> usize {
    if node_idx == usize::MAX {
        return 0;
    }
    let mut max_child_depth = 0;
    let mut child = arena[node_idx].first_child;
    while child != u32::MAX {
        let depth = compute_max_depth(arena, child as usize);
        if depth > max_child_depth {
            max_child_depth = depth;
        }
        child = arena[child as usize].next_sibling;
    }
    max_child_depth + 1
}

fn get_temperature_decay_factor(current_step: usize, temperature_decay_steps: usize) -> f32 {
    if current_step < temperature_decay_steps / 2 {
        1.0
    } else if current_step < temperature_decay_steps {
        0.5
    } else {
        0.1
    }
}

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
    use std::sync::{Arc, RwLock};
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
        let neural_model = Arc::new(RwLock::new(crate::network::MuZeroNet::new(
            &variable_store.root(),
            model_dimension,
            1,
            200,
        )));

        let mut response_receivers = Vec::new();
        for i in 0..3 {
            let (answer_tx, answer_rx) = crossbeam_channel::unbounded();
            inference_queue
                .push(
                    i,
                    EvalReq {
                        is_initial: true,
                        state_feat: Some(vec![0.0; 20 * 128]),
                        piece_action: 0,
                        piece_id: 0,
                        node_index: 0,
                        worker_id: i,
                        parent_cache_index: 0,
                        leaf_cache_index: 0,
                        tx: answer_tx,
                    },
                )
                .unwrap();
            response_receivers.push(answer_rx);
        }

        inference_queue.disconnect_producer();
        inference_queue.disconnect_producer();
        inference_queue.disconnect_producer();
        inference_loop(
            inference_queue.clone(),
            neural_model,
            None,
            model_dimension,
            Device::Cpu,
            3,
            2000,
        );

        for receiver in response_receivers {
            let evaluator_response = receiver.recv().expect("Failed to receive batched response");
            assert_eq!(evaluator_response.p_next.len(), 288);
        }
    }

    #[test]
    fn test_compute_max_depth() {
        let mut arena = vec![
            crate::node::LatentNode::new(1.0, -1), // 0: Root
            crate::node::LatentNode::new(0.5, 0),  // 1: Child of Root
            crate::node::LatentNode::new(0.5, 1),  // 2: Child of 1 (Sibling of 3)
            crate::node::LatentNode::new(0.5, 2),  // 3: Child of 1 (Sibling of 2)
            crate::node::LatentNode::new(0.5, 3),  // 4: Sibling of 1
            crate::node::LatentNode::new(0.5, 4),  // 5: Child of 4
        ];

        // Link Root (0) -> [1, 4]
        arena[0].first_child = 1;
        arena[1].next_sibling = 4;

        // Link Node 1 -> [2, 3]
        arena[1].first_child = 2;
        arena[2].next_sibling = 3;

        // Link Node 4 -> [5]
        arena[4].first_child = 5;

        // Depth: Root(1) -> 1(2) -> 2(3) -> Max depth should be 3
        let depth = compute_max_depth(&arena, 0);
        assert_eq!(depth, 3);

        // Root with no children
        let arena2 = vec![crate::node::LatentNode::new(1.0, -1)];
        assert_eq!(compute_max_depth(&arena2, 0), 1);
    }
}
