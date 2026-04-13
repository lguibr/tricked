use crate::config::Config;
use crate::queue::FixedInferenceQueue;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tch::{CModule, Device, Kind, Tensor};

pub fn r_inference_worker(
    queue: Arc<FixedInferenceQueue>,
    config: Arc<Config>,
    active_flag: Arc<AtomicBool>,
    initial_model_path: &str,
    recurrent_model_path: &str,
    use_cuda: bool,
) {
    let _no_grad_guard = tch::no_grad_guard();

    let device = if use_cuda && tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    println!("Starting Rust Native Inference Worker on {:?}", device);

    let mut initial_model =
        CModule::load(initial_model_path).expect("Failed to load initial model");
    initial_model.set_eval();
    initial_model.to(device, Kind::Float, true);

    let mut recurrent_model =
        CModule::load(recurrent_model_path).expect("Failed to load recurrent model");
    recurrent_model.set_eval();
    recurrent_model.to(device, Kind::Float, true);

    let inf_batch = config.hardware.inference_batch_size_limit as usize;
    let inf_timeout = config.hardware.inference_timeout_ms as u64;

    let cache_size = config.mcts.simulations * config.hardware.num_processes;
    let hidden_channels = config.architecture.hidden_dimension_size;
    let mut hidden_state_cache =
        Tensor::zeros([cache_size, hidden_channels, 8, 8], (Kind::Float, device));

    while active_flag.load(Ordering::Relaxed) {
        if let Ok((initial_batch, recurrent_batch)) =
            queue.pop_batch_timeout(inf_batch, Duration::from_millis(inf_timeout))
        {
            // Fast spin loop returning directly if timeout occurred without workload
            if initial_batch.is_empty() && recurrent_batch.is_empty() {
                continue;
            }

            // Process Initial Batch
            if !initial_batch.is_empty() {
                let actual_bs = initial_batch.len();
                let mut dense_boards = vec![0.0f32; actual_bs * 20 * 128];
                let mut slots_and_reqs = Vec::with_capacity(actual_bs);

                for (i, guard) in initial_batch.into_iter().enumerate() {
                    let slot = guard.disarm();
                    unsafe {
                        let maybe_req = &mut *queue.metadata[slot].get();
                        if let Some((req, _time)) = maybe_req.take() {
                            let offset = i * 20 * 128;
                            crate::core::features::extract_feature_native(
                                &mut dense_boards[offset..offset + 20 * 128],
                                req.board_bitmask,
                                &req.available_pieces,
                                &req.recent_board_history[..req.history_len],
                                &req.recent_action_history[..req.action_history_len],
                                req.difficulty,
                            );
                            slots_and_reqs.push((slot, req));
                        }
                    }
                }

                if !slots_and_reqs.is_empty() {
                    let valid_bs = slots_and_reqs.len();
                    let f_tensor = tch::Tensor::from_slice(&dense_boards[..valid_bs * 20 * 128])
                        .view([valid_bs as i64, 20, 8, 16])
                        .to_device(device);

                    match initial_model.forward_is(&[tch::IValue::Tensor(f_tensor)]) {
                        Ok(tch::IValue::Tuple(outputs)) => {
                            let hidden_state = if let tch::IValue::Tensor(t) = &outputs[0] {
                                t
                            } else {
                                unreachable!()
                            };
                            let val_logits = if let tch::IValue::Tensor(t) = &outputs[1] {
                                t
                            } else {
                                unreachable!()
                            };
                            let pol_probs = if let tch::IValue::Tensor(t) = &outputs[2] {
                                t
                            } else {
                                unreachable!()
                            };

                            let support_size = config.architecture.value_support_size;
                            let val_argmax = val_logits.argmax(1, false).to_device(Device::Cpu);
                            let pol_cpu = pol_probs.to_device(Device::Cpu);

                            let mut cache_indices = Vec::with_capacity(valid_bs);
                            for (_, req) in slots_and_reqs.iter() {
                                cache_indices.push(req.leaf_cache_index as i64);
                            }
                            let idx_tensor = Tensor::from_slice(&cache_indices).to_device(device);
                            let _ = hidden_state_cache.index_put_(
                                &[Some(&idx_tensor)],
                                hidden_state,
                                false,
                            );

                            for (i, (slot, req)) in slots_and_reqs.into_iter().enumerate() {
                                let mut policy_array = [0.0f32; 288];
                                let pol_slice = pol_cpu.copy();
                                pol_slice
                                    .slice(0, i as i64, i as i64 + 1, 1)
                                    .flatten(0, -1)
                                    .copy_data(&mut policy_array, 288);

                                for p in policy_array.iter_mut() {
                                    if p.is_nan() || p.is_infinite() {
                                        *p = 1.0 / 288.0;
                                    }
                                }

                                let v_idx: i64 = val_argmax.int64_value(&[i as i64]);
                                let scalar_val = (v_idx - support_size) as f32; // direct linear proxy

                                req.mailbox
                                    .write_and_notify(crate::mcts::EvaluationResponse {
                                        child_prior_probabilities_tensor: policy_array,
                                        value_prefix: 0.0,
                                        value: scalar_val,
                                        node_index: req.node_index,
                                        generation: req.generation,
                                    });
                                let _ = queue.free_slots.push(slot);
                            }
                        }
                        Err(e) => {
                            eprintln!("CRITICAL ERROR: initial_model.forward_is failed: {:?}", e);
                            for (slot, req) in slots_and_reqs.into_iter() {
                                req.mailbox
                                    .write_and_notify(crate::mcts::EvaluationResponse {
                                        child_prior_probabilities_tensor: [1.0 / 288.0; 288],
                                        value_prefix: 0.0,
                                        value: 0.0,
                                        node_index: req.node_index,
                                        generation: req.generation,
                                    });
                                let _ = queue.free_slots.push(slot);
                            }
                        }
                        Ok(other) => {
                            eprintln!("CRITICAL ERROR: initial_model.forward_is returned unexpected type: {:?}", other);
                            for (slot, req) in slots_and_reqs.into_iter() {
                                req.mailbox
                                    .write_and_notify(crate::mcts::EvaluationResponse {
                                        child_prior_probabilities_tensor: [1.0 / 288.0; 288],
                                        value_prefix: 0.0,
                                        value: 0.0,
                                        node_index: req.node_index,
                                        generation: req.generation,
                                    });
                                let _ = queue.free_slots.push(slot);
                            }
                        }
                    }
                }
            }

            // Process Recurrent Batch
            if !recurrent_batch.is_empty() {
                let actual_bs = recurrent_batch.len();
                let mut parent_indices = Vec::with_capacity(actual_bs);
                let mut leaf_indices = Vec::with_capacity(actual_bs);
                let mut actions = Vec::with_capacity(actual_bs);
                let mut pieces = Vec::with_capacity(actual_bs);
                let mut slots_and_reqs = Vec::with_capacity(actual_bs);

                for guard in recurrent_batch {
                    let slot = guard.disarm();
                    unsafe {
                        let maybe_req = &mut *queue.metadata[slot].get();
                        if let Some((req, _time)) = maybe_req.take() {
                            parent_indices.push(req.parent_cache_index as i64);
                            leaf_indices.push(req.leaf_cache_index as i64);
                            actions.push(req.piece_action);
                            pieces.push(req.piece_id);
                            slots_and_reqs.push((slot, req));
                        }
                    }
                }

                if !slots_and_reqs.is_empty() {
                    let valid_bs = slots_and_reqs.len();
                    let parent_tensor = Tensor::from_slice(&parent_indices).to_device(device);
                    let gathered_hidden = hidden_state_cache.index(&[Some(&parent_tensor)]);

                    let act_tensor = Tensor::from_slice(&actions)
                        .view([valid_bs as i64, 1])
                        .to_device(device);
                    let piece_tensor = Tensor::from_slice(&pieces)
                        .view([valid_bs as i64, 1])
                        .to_device(device);

                    match recurrent_model.forward_is(&[
                        tch::IValue::Tensor(gathered_hidden),
                        tch::IValue::Tensor(act_tensor),
                        tch::IValue::Tensor(piece_tensor),
                    ]) {
                        Ok(tch::IValue::Tuple(outputs)) => {
                            // outputs: 0: next_hidden, 1: reward_logits, 2: val_logits, 3: pol_probs
                            let next_hidden = if let tch::IValue::Tensor(t) = &outputs[0] {
                                t
                            } else {
                                unreachable!()
                            };
                            let rew_logits = if let tch::IValue::Tensor(t) = &outputs[1] {
                                t
                            } else {
                                unreachable!()
                            };
                            let val_logits = if let tch::IValue::Tensor(t) = &outputs[2] {
                                t
                            } else {
                                unreachable!()
                            };
                            let pol_probs = if let tch::IValue::Tensor(t) = &outputs[3] {
                                t
                            } else {
                                unreachable!()
                            };

                            let leaf_tensor = Tensor::from_slice(&leaf_indices).to_device(device);
                            let _ = hidden_state_cache.index_put_(
                                &[Some(&leaf_tensor)],
                                next_hidden,
                                false,
                            );

                            let support_size = config.architecture.value_support_size;
                            let val_argmax = val_logits.argmax(1, false).to_device(Device::Cpu);
                            let rew_argmax = rew_logits.argmax(1, false).to_device(Device::Cpu);
                            let pol_cpu = pol_probs.to_device(Device::Cpu);

                            for (i, (slot, req)) in slots_and_reqs.into_iter().enumerate() {
                                let mut policy_array = [0.0f32; 288];
                                let pol_slice = pol_cpu.copy();
                                pol_slice
                                    .slice(0, i as i64, i as i64 + 1, 1)
                                    .flatten(0, -1)
                                    .copy_data(&mut policy_array, 288);

                                for p in policy_array.iter_mut() {
                                    if p.is_nan() || p.is_infinite() {
                                        *p = 1.0 / 288.0;
                                    }
                                }

                                let v_idx: i64 = val_argmax.int64_value(&[i as i64]);
                                let scalar_val = (v_idx - support_size) as f32;

                                let r_idx: i64 = rew_argmax.int64_value(&[i as i64]);
                                let scalar_rew = (r_idx - support_size) as f32;

                                req.mailbox
                                    .write_and_notify(crate::mcts::EvaluationResponse {
                                        child_prior_probabilities_tensor: policy_array,
                                        value_prefix: scalar_rew,
                                        value: scalar_val,
                                        node_index: req.node_index,
                                        generation: req.generation,
                                    });
                                let _ = queue.free_slots.push(slot);
                            }
                        }
                        Err(e) => {
                            eprintln!("CRITICAL ERROR: recurrent_model.forward_is failed: {:?}", e);
                            for (slot, req) in slots_and_reqs.into_iter() {
                                req.mailbox
                                    .write_and_notify(crate::mcts::EvaluationResponse {
                                        child_prior_probabilities_tensor: [1.0 / 288.0; 288],
                                        value_prefix: 0.0,
                                        value: 0.0,
                                        node_index: req.node_index,
                                        generation: req.generation,
                                    });
                                let _ = queue.free_slots.push(slot);
                            }
                        }
                        Ok(other) => {
                            eprintln!("CRITICAL ERROR: recurrent_model.forward_is returned unexpected type: {:?}", other);
                            for (slot, req) in slots_and_reqs.into_iter() {
                                req.mailbox
                                    .write_and_notify(crate::mcts::EvaluationResponse {
                                        child_prior_probabilities_tensor: [1.0 / 288.0; 288],
                                        value_prefix: 0.0,
                                        value: 0.0,
                                        node_index: req.node_index,
                                        generation: req.generation,
                                    });
                                let _ = queue.free_slots.push(slot);
                            }
                        }
                    }
                }
            }
        }
    }
}
