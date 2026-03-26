use crossbeam_channel::{unbounded, Receiver, Sender};
use rand::Rng;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
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
    shared_neural_model: Arc<Mutex<MuZeroNet>>,
    model_dimension: i64,
    computation_device: Device,
) {
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
                let neural_model = shared_neural_model.lock().unwrap();

                if !initial_inference_requests.is_empty() {
                    process_initial_inference(
                        &neural_model,
                        initial_inference_requests,
                        model_dimension,
                        computation_device,
                    );
                }

                if !recurrent_inference_requests.is_empty() {
                    process_recurrent_inference(
                        &neural_model,
                        recurrent_inference_requests,
                        model_dimension,
                        computation_device,
                    );
                }
            });
        });
    }
}

fn process_initial_inference(
    neural_model: &MuZeroNet,
    inference_requests: Vec<EvalReq>,
    model_dimension: i64,
    computation_device: Device,
) {
    let state_tensors: Vec<Tensor> = inference_requests
        .iter()
        .map(|request| {
            let features_array = request.state_feat.as_ref().unwrap();
            assert_eq!(
                features_array.len(),
                1920,
                "Initial state feature dimension mismatch"
            );
            Tensor::from_slice(features_array).reshape([20, 96])
        })
        .collect();

    let state_batch = Tensor::stack(&state_tensors, 0)
        .to_device(computation_device)
        .to_kind(Kind::Float);

    assert_eq!(
        state_batch.size(),
        vec![inference_requests.len() as i64, 20, 96],
        "State batch shape assertion failed"
    );

    let (hidden_state_batch, value_batch, policy_batch, _) =
        neural_model.initial_inference(&state_batch);

    let hidden_state_cpu = hidden_state_batch
        .to_device(Device::Cpu)
        .to_kind(Kind::Float);
    let value_predictions_cpu = value_batch.to_device(Device::Cpu).to_kind(Kind::Float);
    let policy_predictions_cpu = policy_batch.to_device(Device::Cpu).to_kind(Kind::Float);

    let hidden_states_f32: Vec<f32> = hidden_state_cpu
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

    let hidden_state_size = (model_dimension as usize) * 96;
    let policy_vector_size = 288;

    if hidden_states_f32.len() == inference_requests.len() * hidden_state_size {
        for (index, request) in inference_requests.into_iter().enumerate() {
            let start_hidden = index * hidden_state_size;
            let end_hidden = (index + 1) * hidden_state_size;
            let start_policy = index * policy_vector_size;
            let end_policy = (index + 1) * policy_vector_size;

            let response = EvalResp {
                h_next: hidden_states_f32[start_hidden..end_hidden].to_vec(),
                reward: 0.0,
                value: value_predictions_f32[index],
                p_next: policy_predictions_f32[start_policy..end_policy].to_vec(),
            };
            let _ = request.tx.send(response);
        }
    }
}

fn process_recurrent_inference(
    neural_model: &MuZeroNet,
    inference_requests: Vec<EvalReq>,
    model_dimension: i64,
    computation_device: Device,
) {
    let hidden_tensors: Vec<Tensor> = inference_requests
        .iter()
        .map(|request| {
            let previous_hidden_state = request.h_last.as_ref().unwrap();
            let channel_count = previous_hidden_state.len() / 96;
            Tensor::from_slice(previous_hidden_state).reshape([channel_count as i64, 96])
        })
        .collect();

    let piece_actions: Vec<i64> = inference_requests
        .iter()
        .map(|request| request.piece_action)
        .collect();
    let piece_identifiers: Vec<i64> = inference_requests
        .iter()
        .map(|request| request.piece_id)
        .collect();

    let hidden_state_batch = Tensor::stack(&hidden_tensors, 0)
        .to_device(computation_device)
        .to_kind(Kind::Float);
    let piece_action_batch = Tensor::from_slice(&piece_actions).to_device(computation_device);
    let piece_identifier_batch =
        Tensor::from_slice(&piece_identifiers).to_device(computation_device);

    assert_eq!(
        hidden_state_batch.size()[0],
        inference_requests.len() as i64,
        "Recurrent hidden batch size mismatch"
    );

    let (hidden_state_next_batch, reward_batch, value_batch, policy_batch, _) = neural_model
        .recurrent_inference(
            &hidden_state_batch,
            &piece_action_batch,
            &piece_identifier_batch,
        );

    let hidden_state_next_cpu = hidden_state_next_batch
        .to_device(Device::Cpu)
        .to_kind(Kind::Float);
    let reward_predictions_cpu = reward_batch.to_device(Device::Cpu).to_kind(Kind::Float);
    let value_predictions_cpu = value_batch.to_device(Device::Cpu).to_kind(Kind::Float);
    let policy_predictions_cpu = policy_batch.to_device(Device::Cpu).to_kind(Kind::Float);

    let hidden_states_next_f32: Vec<f32> = hidden_state_next_cpu
        .reshape([-1i64])
        .try_into()
        .unwrap_or_default();
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

    let hidden_state_size = (model_dimension as usize) * 96;
    let policy_vector_size = 288;

    if hidden_states_next_f32.len() == inference_requests.len() * hidden_state_size {
        for (index, request) in inference_requests.into_iter().enumerate() {
            let start_hidden = index * hidden_state_size;
            let end_hidden = (index + 1) * hidden_state_size;
            let start_policy = index * policy_vector_size;
            let end_policy = (index + 1) * policy_vector_size;

            let response = EvalResp {
                h_next: hidden_states_next_f32[start_hidden..end_hidden].to_vec(),
                reward: reward_predictions_f32[index],
                value: value_predictions_f32[index],
                p_next: policy_predictions_f32[start_policy..end_policy].to_vec(),
            };
            let _ = request.tx.send(response);
        }
    }
}

pub fn game_loop(
    configuration: Arc<Config>,
    evaluation_transmitter: Sender<EvalReq>,
    experience_buffer: Arc<ReplayBuffer>,
    telemetry_store: Arc<RwLock<TelemetryStore>>,
) {
    let mut thread_rng = rand::thread_rng();

    loop {
        let mut active_game_state = GameStateExt::new(None, 0, 0, configuration.difficulty, 0);
        let mut board_history = vec![active_game_state.board, active_game_state.board];
        let mut action_history = Vec::new();
        let mut piece_identifier_history = Vec::new();

        let mut episode_boards = Vec::new();
        let mut episode_available = Vec::new();
        let mut episode_actions = Vec::new();
        let mut episode_piece_ids = Vec::new();
        let mut episode_rewards = Vec::new();
        let mut episode_policies = Vec::new();
        let mut episode_values = Vec::new();

        let mut previous_mcts_tree = None;
        let mut last_executed_action = None;
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
                    h_last: None,
                    piece_action: 0,
                    piece_id: 0,
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

            let mcts_result = match mcts_search(crate::mcts::MctsParams {
                hidden_state_root: &initial_evaluation_response.h_next,
                raw_policy_probabilities: &initial_evaluation_response.p_next,
                game_state: &active_game_state,
                total_simulations: configuration.simulations as usize,
                max_gumbel_k_samples: configuration.max_gumbel_k as usize,
                gumbel_noise_scale: configuration.gumbel_scale,
                previous_tree: previous_mcts_tree,
                last_executed_action,
                neural_evaluator: &evaluation_transmitter,
                _seed: None,
            }) {
                Ok(result) => result,
                Err(_) => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    break;
                }
            };

            let selected_best_action = mcts_result.0;
            let mcts_visit_distribution = mcts_result.1;
            let latent_value_prediction = mcts_result.2;
            previous_mcts_tree = Some(mcts_result.3);

            if selected_best_action == -1 {
                break;
            }

            if thread_rng.gen_ratio(1, 10) {
                if let Ok(mut telemetry) = telemetry_store.write() {
                    telemetry.spectator_state = Some(active_game_state.clone());
                }
            }

            let temperature_decay = get_temperature_decay_factor(
                episode_step_count,
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

            last_executed_action = Some(randomized_action);
            let board_slot_index = randomized_action / 96;
            let spatial_position_index = randomized_action % 96;

            let next_game_state = match active_game_state
                .apply_move(board_slot_index as usize, spatial_position_index as usize)
            {
                Some(state) => state,
                None => break,
            };

            let reward_received = (next_game_state.score - active_game_state.score) as f32;
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
            experience_buffer.add_game(crate::buffer::replay::AddGameParams {
                difficulty_setting: configuration.difficulty,
                episode_score: active_game_state.score as f32,
                board_states: &episode_boards,
                available_pieces: &episode_available,
                actions_taken: &episode_actions,
                piece_identifiers: &episode_piece_ids,
                rewards_received: &episode_rewards,
                policy_targets: &episode_policies,
                value_targets: &episode_values,
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
    use std::sync::{Arc, Mutex};
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
        let neural_model = Arc::new(Mutex::new(crate::network::MuZeroNet::new(
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
                    h_last: None,
                    piece_action: 0,
                    piece_id: 0,
                    tx: answer_tx,
                })
                .unwrap();
            response_receivers.push(answer_rx);
        }

        drop(transmission_queue);
        inference_loop(receiver_queue, neural_model, model_dimension, Device::Cpu);

        for receiver in response_receivers {
            let evaluator_response = receiver.recv().expect("Failed to receive batched response");
            assert_eq!(
                evaluator_response.h_next.len(),
                (model_dimension as usize) * 96
            );
        }
    }
}
