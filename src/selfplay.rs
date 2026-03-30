use crate::queue::FixedInferenceQueue;
use arc_swap::ArcSwap;
use crossbeam_channel::unbounded;
use rand::Rng;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tch::{Device, Kind, Tensor};

use crate::board::GameStateExt;
use crate::buffer::ReplayBuffer;
use crate::config::Config;

use crate::mcts::{mcts_search, EvalReq, EvalResp};
use crate::network::MuZeroNet;
use crate::web::TelemetryStore;

struct SafeTensorGuard<'a, T> {
    _tensor: &'a Tensor,
    pub slice: &'a mut [T],
}

impl<'a, T> SafeTensorGuard<'a, T> {
    fn new(tensor: &'a Tensor, len: usize) -> Self {
        assert!(
            tensor.is_contiguous(),
            "Tensor must be contiguous for raw pointer access"
        );
        Self {
            _tensor: tensor,
            slice: unsafe { std::slice::from_raw_parts_mut(tensor.data_ptr() as *mut T, len) },
        }
    }
}

impl<'a, T> Drop for SafeTensorGuard<'a, T> {
    fn drop(&mut self) {}
}

pub struct InferenceLoopParams {
    pub receiver_queue: Arc<FixedInferenceQueue>,
    pub shared_neural_model: Arc<ArcSwap<MuZeroNet>>,
    pub cmodule_inference: Option<Arc<tch::CModule>>,
    pub model_dimension: i64,
    pub computation_device: Device,
    pub total_workers: usize,
    pub maximum_allowed_nodes_in_search_tree: usize,
    pub inference_batch_size_limit: usize,
    pub inference_timeout_milliseconds: u64,
}

pub fn inference_loop(params: InferenceLoopParams) {
    let receiver_queue = params.receiver_queue;
    let shared_neural_model = params.shared_neural_model;
    let cmodule_inference = params.cmodule_inference;
    let computation_device = params.computation_device;
    let inference_batch_size_limit = params.inference_batch_size_limit;
    let inference_timeout_milliseconds = params.inference_timeout_milliseconds;
    let model_dimension = params.model_dimension;
    let maximum_allowed_nodes_in_search_tree = params.maximum_allowed_nodes_in_search_tree;

    let flat_cache_size =
        (params.total_workers * params.maximum_allowed_nodes_in_search_tree) as i64;
    let mut latent_cache = Tensor::zeros(
        [flat_cache_size, model_dimension, 8, 8],
        (Kind::Float, computation_device),
    );

    let current_batch_size_i64 = inference_batch_size_limit as i64;
    let mut pinned_initial_states = Tensor::zeros(
        [current_batch_size_i64, 20, 8, 16],
        (Kind::Float, Device::Cpu),
    );
    let mut pinned_recurrent_actions =
        Tensor::zeros([current_batch_size_i64], (Kind::Int64, Device::Cpu));
    let mut pinned_recurrent_ids =
        Tensor::zeros([current_batch_size_i64], (Kind::Int64, Device::Cpu));
    let mut pinned_workers = Tensor::zeros([current_batch_size_i64], (Kind::Int64, Device::Cpu));
    let mut pinned_parents = Tensor::zeros([current_batch_size_i64], (Kind::Int64, Device::Cpu));
    let mut pinned_nodes = Tensor::zeros([current_batch_size_i64], (Kind::Int64, Device::Cpu));

    if computation_device.is_cuda() {
        pinned_initial_states = pinned_initial_states.pin_memory(computation_device);
        pinned_recurrent_actions = pinned_recurrent_actions.pin_memory(computation_device);
        pinned_recurrent_ids = pinned_recurrent_ids.pin_memory(computation_device);
        pinned_workers = pinned_workers.pin_memory(computation_device);
        pinned_parents = pinned_parents.pin_memory(computation_device);
        pinned_nodes = pinned_nodes.pin_memory(computation_device);
    }

    let mut batch_count = 0;
    let mut total_batch_size = 0;

    loop {
        let batched_requests = match receiver_queue.pop_batch_timeout(
            inference_batch_size_limit,
            std::time::Duration::from_millis(inference_timeout_milliseconds),
        ) {
            Ok(reqs) if !reqs.is_empty() => reqs,
            Ok(_) => continue,
            Err(_) => break,
        };

        let actual_size = batched_requests.len();
        batch_count += 1;
        total_batch_size += actual_size;

        if batch_count % 500 == 0 {
            let avg = total_batch_size as f32 / batch_count as f32;
            println!(
                "🏎️ [Inference] Dynamic Batching Average Size: {:.1} / {}",
                avg, inference_batch_size_limit
            );
            if batch_count > 10_000 {
                batch_count = 0;
                total_batch_size = 0;
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
            tch::autocast(false, || {
                let neural_model = shared_neural_model.load();

                if !initial_inference_requests.is_empty() {
                    process_initial_inference(
                        &neural_model,
                        cmodule_inference.as_deref(),
                        initial_inference_requests,
                        maximum_allowed_nodes_in_search_tree,
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
                        maximum_allowed_nodes_in_search_tree,
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
    maximum_allowed_nodes_in_search_tree: usize,
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
            let len = 20 * 8 * 16;
            let target_slice = &mut guard.slice[offset..offset + len];
            let hist_len = request.history_len;
            let act_len = request.action_history_len;

            crate::features::extract_feature_native(
                target_slice,
                request.board_bitmask,
                &request.available_pieces,
                &request.recent_board_history[..hist_len],
                &request.recent_action_history[..act_len],
                request.difficulty,
            );
            offset += len;
        }
    }

    let state_batch = state_view.to_device(computation_device);

    let (hidden_state_batch, value_batch, policy_batch, _) = if let Some(cmod) = cmodule_inference {
        let ivalue = match cmod.method_is(
            "initial_inference",
            &[tch::IValue::Tensor(state_batch.copy())],
        ) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("💥 LibTorch C-FFI Exception in initial_inference: {:?}", e);
                panic!("Fatal C-FFI bound error: {:?}", e);
            }
        };
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

    let maximum_allowed_nodes_in_search_tree_tensor =
        Tensor::from_slice(&[maximum_allowed_nodes_in_search_tree as i64])
            .to_device(computation_device);
    let flat_n_indices = (&w_tensor * &maximum_allowed_nodes_in_search_tree_tensor) + &n_tensor;

    let _ = latent_cache.index_copy_(0, &flat_n_indices, &hidden_state_batch);

    std::thread::spawn(move || {
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
                child_prior_probabilities_tensor: policy_predictions_f32[start_policy..end_policy]
                    .to_vec(),
                node_index: request.node_index,
            };
            let _ = request.evaluation_request_transmitter.send(response);
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn process_recurrent_inference(
    neural_model: &MuZeroNet,
    cmodule_inference: Option<&tch::CModule>,
    inference_requests: Vec<EvalReq>,
    maximum_allowed_nodes_in_search_tree: usize,
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

    let maximum_allowed_nodes_in_search_tree_tensor =
        Tensor::from_slice(&[maximum_allowed_nodes_in_search_tree as i64])
            .to_device(computation_device);
    let flat_p_indices = (&w_tensor * &maximum_allowed_nodes_in_search_tree_tensor) + &p_tensor;
    let flat_n_indices = (&w_tensor * &maximum_allowed_nodes_in_search_tree_tensor) + &n_tensor;

    let hidden_state_batch = latent_cache.index_select(0, &flat_p_indices);

    let (hidden_state_next_batch, reward_batch, value_batch, policy_batch, _) =
        if let Some(cmod) = cmodule_inference {
            let ivalue = match cmod.method_is(
                "recurrent_inference",
                &[
                    tch::IValue::Tensor(hidden_state_batch.copy()),
                    tch::IValue::Tensor(piece_action_batch.copy()),
                    tch::IValue::Tensor(piece_identifier_batch.copy()),
                ],
            ) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "💥 LibTorch C-FFI Exception in recurrent_inference: {:?}",
                        e
                    );
                    panic!("Fatal C-FFI bound error: {:?}", e);
                }
            };
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

    std::thread::spawn(move || {
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
                child_prior_probabilities_tensor: policy_predictions_f32[start_policy..end_policy]
                    .to_vec(),
                node_index: request.node_index,
            };
            let _ = request.evaluation_request_transmitter.send(response);
        }
    });
}

pub struct GameLoopExecutionParameters {
    pub configuration: Arc<Config>,
    pub evaluation_transmitter: Arc<FixedInferenceQueue>,
    pub experience_buffer: Arc<ReplayBuffer>,
    pub telemetry_store: Arc<RwLock<TelemetryStore>>,
    pub game_logger: Arc<dyn crate::telemetry::GameLogger>,
    pub worker_id: usize,
    pub perf_counters: Arc<crate::telemetry::PerformanceCounters>,
    pub active_flag: Arc<RwLock<bool>>,
}

pub fn game_loop(parameters: GameLoopExecutionParameters) {
    let configuration = parameters.configuration;
    let evaluation_transmitter = parameters.evaluation_transmitter;
    let experience_buffer = parameters.experience_buffer;
    let telemetry_store = parameters.telemetry_store;
    let game_logger = parameters.game_logger;
    let worker_id = parameters.worker_id;
    let perf_counters = parameters.perf_counters;
    let active_flag = parameters.active_flag;
    let mut thread_rng = rand::thread_rng();
    let mut last_spectator_update = std::time::Instant::now();

    loop {
        let mut active_game_state = GameStateExt::new(None, 0, 0, configuration.difficulty, 0);
        let mut board_history = vec![
            active_game_state.board_bitmask_u128,
            active_game_state.board_bitmask_u128,
        ];
        let mut action_history = Vec::new();
        let mut piece_identifier_history = Vec::new();

        let mut current_tree: Option<crate::mcts::MctsTree> = None;
        let mut last_action: Option<i32> = None;
        let mut last_known_training_steps = 0;

        let mut episode_steps = Vec::with_capacity(100);

        let mut episode_step_count = 0;
        let (response_tx, response_rx) = unbounded();

        for _ in 0..1000 {
            if !*active_flag.read().unwrap() {
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
                    vec![EvalReq {
                        is_initial: true,
                        board_bitmask: active_game_state.board_bitmask_u128,
                        available_pieces: active_game_state.available,
                        recent_board_history: board_history_array,
                        history_len: std::cmp::min(board_history.len(), 8),
                        recent_action_history: action_history_array,
                        action_history_len: std::cmp::min(action_history.len(), 4),
                        difficulty: configuration.difficulty,
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

            let initial_evaluation_response = match response_rx.recv() {
                Ok(response) => response,
                Err(_) => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    break;
                }
            };

            let search_start = std::time::Instant::now();
            let mcts_result = match mcts_search(crate::mcts::MctsParams {
                raw_policy_probabilities: &initial_evaluation_response
                    .child_prior_probabilities_tensor,
                root_cache_index: 0,
                maximum_allowed_nodes_in_search_tree: (configuration.simulations as u32) + 300,
                worker_id,
                game_state: &active_game_state,
                total_simulations: configuration.simulations as usize,
                max_gumbel_k_samples: configuration.max_gumbel_k as usize,
                gumbel_noise_scale: configuration.gumbel_scale,
                previous_tree: current_tree.take(),
                last_executed_action: last_action,
                neural_evaluator: &evaluation_transmitter,
                evaluation_request_transmitter: response_tx.clone(),
                evaluation_response_receiver: &response_rx,
                _seed: None,
            }) {
                Ok(result) => result,
                Err(_) => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    break;
                }
            };

            perf_counters.total_simulations.fetch_add(
                configuration.simulations as u64,
                std::sync::atomic::Ordering::Relaxed,
            );

            let search_duration = search_start.elapsed().as_secs_f32() * 1000.0;
            let selected_best_action = mcts_result.0;
            let mcts_visit_distribution = mcts_result.1;
            let latent_value_prediction = mcts_result.2;

            let current_max_depth =
                compute_max_depth(&mcts_result.3.arena, mcts_result.3.root_index);

            game_logger.log_metric("mcts/search_time_ms", search_duration);
            game_logger.log_metric("mcts/value_prediction", latent_value_prediction);
            game_logger.log_metric("mcts/max_depth", current_max_depth as f32);

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

            if episode_step_count % 10 == 0 {
                if let Ok(tel) = telemetry_store.try_read() {
                    last_known_training_steps = tel.status.training_steps as usize;
                }
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
            let mut rigid_policy_array = [0.0f32; 288];
            rigid_policy_array.copy_from_slice(&target_policy_probabilities);

            episode_steps.push(crate::buffer::replay::GameStep {
                board_state: [
                    (active_game_state.board_bitmask_u128 & 0xFFFFFFFFFFFFFFFF) as u64,
                    (active_game_state.board_bitmask_u128 >> 64) as u64,
                ],
                available_pieces: active_game_state.available,
                action_taken: composite_action_identifier as i64,
                piece_identifier: piece_identifier as i64,
                reward_received,
                policy_target: rigid_policy_array,
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
            perf_counters
                .total_steps
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        if episode_step_count > 0 {
            game_logger.log_metric("game/score", active_game_state.score as f32);
            game_logger.log_metric("game/episode_length", episode_step_count as f32);
            game_logger.log_metric("game/difficulty", configuration.difficulty as f32);
            game_logger.log_metric(
                "game/lines_cleared",
                active_game_state.total_lines_cleared as f32,
            );

            game_logger.log_game_end(
                configuration.difficulty,
                active_game_state.score as f32,
                episode_step_count as i32,
            );

            let current_games_count = perf_counters
                .total_games
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                as usize;

            if let Ok(mut tel) = telemetry_store.write() {
                tel.status.games_played = current_games_count as u64 + 1;

                tel.top_games.insert(
                    0,
                    crate::buffer::state::EpisodeMeta {
                        global_start_storage_index: current_games_count,
                        length: episode_step_count,
                        difficulty: configuration.difficulty,
                        score: active_game_state.score as f32,
                    },
                );
                if tel.top_games.len() > 20 {
                    tel.top_games.truncate(20);
                }
            }

            let boards_u128: Vec<u128> = episode_steps
                .iter()
                .map(|s| ((s.board_state[1] as u128) << 64) | (s.board_state[0] as u128))
                .collect();
            let episode_available: Vec<[i32; 3]> =
                episode_steps.iter().map(|s| s.available_pieces).collect();
            let episode_actions: Vec<i64> = episode_steps.iter().map(|s| s.action_taken).collect();
            let episode_piece_ids: Vec<i64> =
                episode_steps.iter().map(|s| s.piece_identifier).collect();
            game_logger.log_trajectory(
                current_games_count,
                &boards_u128,
                &episode_available,
                &episode_actions,
                &episode_piece_ids,
            );

            experience_buffer.add_game(crate::buffer::replay::OwnedGameData {
                difficulty_setting: configuration.difficulty,
                episode_score: active_game_state.score as f32,
                steps: episode_steps,
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
        let p_net = Arc::new(crate::network::MuZeroNet::new(
            &variable_store.root(),
            model_dimension,
            1,
            200,
        ));
        let neural_model = Arc::new(ArcSwap::from(p_net));

        let mut response_receivers = Vec::new();
        for i in 0..3 {
            let (answer_tx, answer_rx) = crossbeam_channel::unbounded();
            inference_queue
                .push_batch(
                    i,
                    vec![EvalReq {
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
                        worker_id: i,
                        parent_cache_index: 0,
                        leaf_cache_index: 0,
                        evaluation_request_transmitter: answer_tx.clone(),
                    }],
                )
                .unwrap();
            response_receivers.push(answer_rx);
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
            maximum_allowed_nodes_in_search_tree: 2000,
            inference_batch_size_limit: 1024,
            inference_timeout_milliseconds: 10,
        });

        for receiver in response_receivers {
            let evaluator_response = receiver.recv().expect("Failed to receive batched response");
            assert_eq!(
                evaluator_response.child_prior_probabilities_tensor.len(),
                288
            );
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
