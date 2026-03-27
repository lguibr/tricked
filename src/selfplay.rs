use crossbeam_channel::{unbounded, Receiver, Sender};
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

pub fn inference_loop(
    receiver_queue: Receiver<EvalReq>,
    shared_neural_model: Arc<RwLock<MuZeroNet>>,
    cmodule_inference: Option<Arc<tch::CModule>>,
    model_dimension: i64,
    computation_device: Device,
    total_workers: usize,
    max_nodes: usize,
) {
    let mut latent_cache = Tensor::zeros(
        [total_workers as i64, max_nodes as i64, model_dimension, 96],
        (Kind::Float, computation_device),
    );

    let mut pinned_initial_states = Tensor::zeros([1024, 20, 96], (Kind::Float, Device::Cpu));
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

    let mut flat_state = vec![0.0f32; 1024 * 1920];
    loop {
        let first_request = match receiver_queue.recv_timeout(std::time::Duration::from_millis(10))
        {
            Ok(request) => request,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        };

        let mut batched_requests = vec![first_request];
        while batched_requests.len() < 1024 {
            if let Ok(request) = receiver_queue.try_recv() {
                batched_requests.push(request);
            } else {
                break;
            }
        }

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
                        model_dimension,
                        computation_device,
                        &mut latent_cache,
                        &mut flat_state,
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
                        model_dimension,
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
    _model_dimension: i64,
    computation_device: Device,
    latent_cache: &mut Tensor,
    flat_state: &mut Vec<f32>,
    pinned_initial_states: &mut Tensor,
    pinned_workers: &mut Tensor,
    pinned_nodes: &mut Tensor,
) {
    let batch_size = inference_requests.len();
    flat_state.clear();
    for request in &inference_requests {
        let features_array = request.state_feat.as_ref().unwrap();
        flat_state.extend_from_slice(features_array);
    }

    let temp_tensor = Tensor::from_slice(flat_state).reshape([batch_size as i64, 20, 96]);

    let mut state_view = pinned_initial_states.narrow(0, 0, batch_size as i64);
    state_view.copy_(&temp_tensor);

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
    let mut w_idx = Vec::with_capacity(batch_size);
    let mut n_idx = Vec::with_capacity(batch_size);
    for req in &inference_requests {
        w_idx.push(req.worker_id as i64);
        n_idx.push(req.leaf_cache_index as i64);
    }

    let mut w_view = pinned_workers.narrow(0, 0, batch_size as i64);
    w_view.copy_(&Tensor::from_slice(&w_idx));
    let mut n_view = pinned_nodes.narrow(0, 0, batch_size as i64);
    n_view.copy_(&Tensor::from_slice(&n_idx));

    let w_tensor = w_view.to_device(computation_device);
    let n_tensor = n_view.to_device(computation_device);

    let _ = latent_cache.index_put_(
        &[Some(w_tensor), Some(n_tensor)],
        &hidden_state_batch,
        false,
    );

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
    _model_dimension: i64,
    computation_device: Device,
    latent_cache: &mut Tensor,
    pinned_actions: &mut Tensor,
    pinned_ids: &mut Tensor,
    pinned_workers: &mut Tensor,
    pinned_parents: &mut Tensor,
    pinned_nodes: &mut Tensor,
) {
    let batch_size = inference_requests.len();
    let mut piece_actions = Vec::with_capacity(batch_size);
    let mut piece_identifiers = Vec::with_capacity(batch_size);
    let mut w_idx = Vec::with_capacity(batch_size);
    let mut p_idx = Vec::with_capacity(batch_size);
    let mut n_idx = Vec::with_capacity(batch_size);

    for request in &inference_requests {
        piece_actions.push(request.piece_action);
        piece_identifiers.push(request.piece_id);
        w_idx.push(request.worker_id as i64);
        p_idx.push(request.parent_cache_index as i64);
        n_idx.push(request.leaf_cache_index as i64);
    }

    let mut actions_view = pinned_actions.narrow(0, 0, batch_size as i64);
    actions_view.copy_(&Tensor::from_slice(&piece_actions));
    let mut ids_view = pinned_ids.narrow(0, 0, batch_size as i64);
    ids_view.copy_(&Tensor::from_slice(&piece_identifiers));

    let mut w_view = pinned_workers.narrow(0, 0, batch_size as i64);
    w_view.copy_(&Tensor::from_slice(&w_idx));
    let mut p_view = pinned_parents.narrow(0, 0, batch_size as i64);
    p_view.copy_(&Tensor::from_slice(&p_idx));
    let mut n_view = pinned_nodes.narrow(0, 0, batch_size as i64);
    n_view.copy_(&Tensor::from_slice(&n_idx));

    let piece_action_batch = actions_view.to_device(computation_device);
    let piece_identifier_batch = ids_view.to_device(computation_device);
    let w_tensor = w_view.to_device(computation_device);
    let p_tensor = p_view.to_device(computation_device);
    let n_tensor = n_view.to_device(computation_device);

    let hidden_state_batch = latent_cache.index(&[Some(w_tensor.copy()), Some(p_tensor)]);

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

    let _ = latent_cache.index_put_(
        &[Some(w_tensor), Some(n_tensor)],
        &hidden_state_next_batch,
        false,
    );

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
    evaluation_transmitter: Sender<EvalReq>,
    experience_buffer: Arc<ReplayBuffer>,
    telemetry_store: Arc<RwLock<TelemetryStore>>,
    game_logger: Arc<dyn crate::telemetry::GameLogger>,
    worker_id: usize,
) {
    let mut thread_rng = rand::thread_rng();
    let mut last_spectator_update = std::time::Instant::now();

    loop {
        let mut active_game_state = GameStateExt::new(None, 0, 0, configuration.difficulty, 0);
        let mut board_history = vec![active_game_state.board, active_game_state.board];
        let mut action_history = Vec::new();
        let mut piece_identifier_history = Vec::new();

        let mut current_tree: Option<crate::mcts::MctsTree> = None;
        let mut last_action: Option<i32> = None;

        let mut episode_boards = Vec::new();
        let mut episode_available = Vec::new();
        let mut episode_actions = Vec::new();
        let mut episode_piece_ids = Vec::new();
        let mut episode_rewards = Vec::new();
        let mut episode_policies = Vec::new();
        let mut episode_values = Vec::new();

        let mut episode_step_count = 0;

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

            let (response_tx, response_rx) = unbounded();
            if evaluation_transmitter
                .send(EvalReq {
                    is_initial: true,
                    state_feat: Some(features_array),
                    piece_action: 0,
                    piece_id: 0,
                    node_index: 0,
                    worker_id,
                    parent_cache_index: 0,
                    leaf_cache_index: 0,
                    tx: response_tx,
                })
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
                _seed: None,
            }) {
                Ok(result) => result,
                Err(_) => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    break;
                }
            };

            let search_duration = search_start.elapsed().as_secs_f32() * 1000.0;
            if thread_rng.gen_ratio(1, 20) {
                game_logger.log_metric("mcts/search_time_ms", search_duration);
            }

            let selected_best_action = mcts_result.0;
            let mcts_visit_distribution = mcts_result.1;
            let latent_value_prediction = mcts_result.2;

            last_action = Some(selected_best_action);
            current_tree = Some(mcts_result.3);

            if selected_best_action == -1 {
                break;
            }

            if last_spectator_update.elapsed() > std::time::Duration::from_millis(500) {
                if let Ok(mut telemetry) = telemetry_store.write() {
                    telemetry.spectator_state = Some(active_game_state.clone());
                }
                last_spectator_update = std::time::Instant::now();
            }

            let global_training_steps = if let Ok(tel) = telemetry_store.read() {
                tel.status.training_steps as usize
            } else {
                0
            };

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

            if let Ok(mut tel) = telemetry_store.write() {
                tel.status.games_played += 1;
                let current_games_count = tel.status.games_played as usize;

                game_logger.log_trajectory(current_games_count, &episode_boards);

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
        let (transmission_queue, receiver_queue) = crossbeam_channel::unbounded();
        let variable_store = nn::VarStore::new(Device::Cpu);
        let model_dimension = 16;
        let neural_model = Arc::new(RwLock::new(crate::network::MuZeroNet::new(
            &variable_store.root(),
            model_dimension,
            1,
            200,
        )));

        let mut response_receivers = Vec::new();
        for _ in 0..3 {
            let (answer_tx, answer_rx) = crossbeam_channel::unbounded();
            transmission_queue
                .send(EvalReq {
                    is_initial: true,
                    state_feat: Some(vec![0.0; 20 * 96]),
                    piece_action: 0,
                    piece_id: 0,
                    node_index: 0,
                    worker_id: 0,
                    parent_cache_index: 0,
                    leaf_cache_index: 0,
                    tx: answer_tx,
                })
                .unwrap();
            response_receivers.push(answer_rx);
        }

        drop(transmission_queue);
        inference_loop(
            receiver_queue,
            neural_model,
            None,
            model_dimension,
            Device::Cpu,
            1,
            2000,
        );

        for receiver in response_receivers {
            let evaluator_response = receiver.recv().expect("Failed to receive batched response");
            assert_eq!(evaluator_response.p_next.len(), 288);
        }
    }
}
