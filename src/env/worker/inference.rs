use crate::queue::FixedInferenceQueue;
use arc_swap::ArcSwap;
use std::sync::Arc;
use tch::{Device, Kind, Tensor};

use crate::mcts::EvaluationResponse;
use crate::net::MuZeroNet;

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
    pub active_flag: Arc<std::sync::atomic::AtomicBool>,
    pub shared_queue_saturation: Arc<std::sync::atomic::AtomicU32>,
}

#[hotpath::measure]
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
    let mut pinned_workers = Tensor::zeros([current_batch_size_i64], (Kind::Int64, Device::Cpu));
    let mut pinned_parents = Tensor::zeros([current_batch_size_i64], (Kind::Int64, Device::Cpu));
    let mut pinned_nodes = Tensor::zeros([current_batch_size_i64], (Kind::Int64, Device::Cpu));

    let mut gpu_workers =
        Tensor::zeros([current_batch_size_i64], (Kind::Int64, computation_device));
    let mut gpu_parents =
        Tensor::zeros([current_batch_size_i64], (Kind::Int64, computation_device));
    let mut gpu_nodes = Tensor::zeros([current_batch_size_i64], (Kind::Int64, computation_device));

    if computation_device.is_cuda() {
        pinned_workers = pinned_workers.pin_memory(computation_device);
        pinned_parents = pinned_parents.pin_memory(computation_device);
        pinned_nodes = pinned_nodes.pin_memory(computation_device);
    }

    let mut batch_count = 0;
    let mut total_batch_size = 0;

    #[cfg(not(test))]
    {
        // Custom ops now loaded in main()
    }

    loop {
        if !params
            .active_flag
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            break;
        }
        let (initial_requests, recurrent_requests) = match receiver_queue.pop_batch_timeout(
            inference_batch_size_limit,
            std::time::Duration::from_millis(inference_timeout_milliseconds),
        ) {
            Ok((i, r)) if !i.is_empty() || !r.is_empty() => (i, r),
            Ok(_) => continue,
            Err(_) => break,
        };

        let actual_size = initial_requests.len() + recurrent_requests.len();
        batch_count += 1;
        total_batch_size += actual_size;

        let saturation = actual_size as f32 / inference_batch_size_limit as f32;
        let current_sat = f32::from_bits(
            params
                .shared_queue_saturation
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        let new_sat = if current_sat == 0.0 {
            saturation
        } else {
            current_sat * 0.95 + saturation * 0.05
        };
        params
            .shared_queue_saturation
            .store(new_sat.to_bits(), std::sync::atomic::Ordering::Relaxed);

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

        let i_empty = initial_requests.is_empty();
        let r_empty = recurrent_requests.is_empty();

        tch::no_grad(|| {
            let neural_model = shared_neural_model.load();

            if !i_empty {
                process_initial_inference(
                    &neural_model,
                    cmodule_inference.as_deref(),
                    initial_requests,
                    receiver_queue.clone(),
                    inference_batch_size_limit,
                    maximum_allowed_nodes_in_search_tree,
                    computation_device,
                    &mut latent_cache,
                    &mut pinned_workers,
                    &mut pinned_nodes,
                    &mut gpu_workers,
                    &mut gpu_nodes,
                );
            }

            if !r_empty {
                process_recurrent_inference(
                    &neural_model,
                    cmodule_inference.as_deref(),
                    recurrent_requests,
                    receiver_queue.clone(),
                    inference_batch_size_limit,
                    maximum_allowed_nodes_in_search_tree,
                    computation_device,
                    &mut latent_cache,
                    &mut pinned_workers,
                    &mut pinned_parents,
                    &mut pinned_nodes,
                    &mut gpu_workers,
                    &mut gpu_parents,
                    &mut gpu_nodes,
                );
            }
        });
    }
}

#[hotpath::measure]
#[allow(clippy::too_many_arguments)]
fn process_initial_inference(
    neural_model: &MuZeroNet,
    cmodule_inference: Option<&tch::CModule>,
    initial_slots: Vec<crate::queue::QueueSlotGuard>,
    queue: Arc<FixedInferenceQueue>,
    inference_batch_size_limit: usize,
    maximum_allowed_nodes_in_search_tree: usize,
    computation_device: Device,
    latent_cache: &mut Tensor,
    pinned_workers: &mut Tensor,
    pinned_nodes: &mut Tensor,
    gpu_workers: &mut Tensor,
    gpu_nodes: &mut Tensor,
) {
    let actual_size = initial_slots.len();
    if actual_size == 0 {
        return;
    }

    let mut slots_i64: Vec<i64> = initial_slots
        .iter()
        .map(|guard| guard.slot as i64)
        .collect();
    let pad_val = slots_i64[0];
    while slots_i64.len() < inference_batch_size_limit {
        slots_i64.push(pad_val);
    }

    let batch_size = inference_batch_size_limit;
    let index_tensor = Tensor::from_slice(&slots_i64);

    let boards = queue
        .initial_boards_pinned
        .index_select(0, &index_tensor)
        .to_device(computation_device);
    let avail = queue
        .initial_avail_pinned
        .index_select(0, &index_tensor)
        .to_device(computation_device);
    let hist = queue
        .initial_hist_pinned
        .index_select(0, &index_tensor)
        .to_device(computation_device);
    let acts = queue
        .initial_acts_pinned
        .index_select(0, &index_tensor)
        .to_device(computation_device);
    let diff = queue
        .initial_diff_pinned
        .index_select(0, &index_tensor)
        .to_device(computation_device);

    let state_batch = if computation_device.is_cuda() {
        neural_model.extract_initial_features(&boards, &avail, &hist, &acts, &diff)
    } else {
        Tensor::zeros(
            [batch_size as i64, neural_model.spatial_channel_count, 8, 16],
            (Kind::Float, computation_device),
        )
    };

    let (hidden_state_batch, value_batch, policy_batch, _) = if let Some(cmod) = cmodule_inference {
        let ivalue = match cmod.method_is("initial_inference", &[tch::IValue::Tensor(state_batch)])
        {
            Ok(v) => v,
            Err(e) => {
                eprintln!("💥 LibTorch C-FFI Exception in initial_inference: {:?}", e);
                panic!("Fatal C-FFI bound error: {:?}", e);
            }
        };
        if let tch::IValue::Tuple(mut tup) = ivalue {
            let value_prefix = if tup.len() == 4 {
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
            (hidden, value, policy, value_prefix)
        } else {
            panic!("Expected Tuple from initial_inference");
        }
    } else {
        neural_model.initial_inference(&state_batch)
    };

    let w_view = pinned_workers.narrow(0, 0, batch_size as i64);
    let n_view = pinned_nodes.narrow(0, 0, batch_size as i64);
    let mut evaluation_requests = Vec::with_capacity(batch_size);

    {
        let w_guard = SafeTensorGuard::<i64>::new(&w_view, batch_size);
        let n_guard = SafeTensorGuard::<i64>::new(&n_view, batch_size);
        for (i, guard) in initial_slots.iter().enumerate() {
            let slot = guard.slot;
            let (req, ts) = unsafe { (*queue.metadata[slot].get()).take().unwrap() };
            queue.latency_sum_nanos.fetch_add(
                ts.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            queue
                .latency_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            w_guard.slice[i] = req.worker_id as i64;
            n_guard.slice[i] = req.leaf_cache_index as i64;
            evaluation_requests.push(req);
        }
        let pad_w = w_guard.slice[0];
        let pad_n = n_guard.slice[0];
        for i in actual_size..batch_size {
            w_guard.slice[i] = pad_w;
            n_guard.slice[i] = pad_n;
        }
    }

    let mut w_tensor = gpu_workers.narrow(0, 0, batch_size as i64);
    w_tensor.copy_(&w_view);
    let mut n_tensor = gpu_nodes.narrow(0, 0, batch_size as i64);
    n_tensor.copy_(&n_view);

    let maximum_allowed_nodes_in_search_tree_tensor =
        Tensor::from_slice(&[maximum_allowed_nodes_in_search_tree as i64])
            .to_device(computation_device);
    let flat_n_indices = (&w_tensor * &maximum_allowed_nodes_in_search_tree_tensor) + &n_tensor;

    let current_sz = actual_size as i64;
    let valid_n_indices = flat_n_indices.narrow(0, 0, current_sz);
    let valid_hidden = hidden_state_batch.narrow(0, 0, current_sz);

    let _ = latent_cache.index_copy_(0, &valid_n_indices, &valid_hidden);

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
        for (index, request) in evaluation_requests.into_iter().enumerate() {
            let start_policy = index * policy_vector_size;
            let end_policy = (index + 1) * policy_vector_size;

            let response = EvaluationResponse {
                value_prefix: 0.0,
                value: value_predictions_f32[index],
                child_prior_probabilities_tensor: policy_predictions_f32[start_policy..end_policy]
                    .try_into()
                    .unwrap(),
                node_index: request.node_index,
                generation: request.generation,
            };
            let _ = request.evaluation_request_transmitter.send(response);
        }

        // initial_slots will naturally drop here, and their QueueSlotGuards will return slots to free_tx.
    });
}

#[hotpath::measure]
#[allow(clippy::too_many_arguments)]
fn process_recurrent_inference(
    neural_model: &MuZeroNet,
    cmodule_inference: Option<&tch::CModule>,
    recurrent_slots: Vec<crate::queue::QueueSlotGuard>,
    queue: Arc<FixedInferenceQueue>,
    inference_batch_size_limit: usize,
    maximum_allowed_nodes_in_search_tree: usize,
    computation_device: Device,
    latent_cache: &mut Tensor,
    pinned_workers: &mut Tensor,
    pinned_parents: &mut Tensor,
    pinned_nodes: &mut Tensor,
    gpu_workers: &mut Tensor,
    gpu_parents: &mut Tensor,
    gpu_nodes: &mut Tensor,
) {
    let actual_size = recurrent_slots.len();
    if actual_size == 0 {
        return;
    }

    let mut slots_i64: Vec<i64> = recurrent_slots
        .iter()
        .map(|guard| guard.slot as i64)
        .collect();
    let pad_val = slots_i64[0];
    while slots_i64.len() < inference_batch_size_limit {
        slots_i64.push(pad_val);
    }

    let batch_size = inference_batch_size_limit;
    let index_tensor = Tensor::from_slice(&slots_i64);

    let piece_action_batch = queue
        .recurrent_actions_pinned
        .index_select(0, &index_tensor)
        .to_device(computation_device);
    let piece_identifier_batch = queue
        .recurrent_ids_pinned
        .index_select(0, &index_tensor)
        .to_device(computation_device);

    let w_view = pinned_workers.narrow(0, 0, batch_size as i64);
    let p_view = pinned_parents.narrow(0, 0, batch_size as i64);
    let n_view = pinned_nodes.narrow(0, 0, batch_size as i64);
    let mut evaluation_requests = Vec::with_capacity(batch_size);

    {
        let w_guard = SafeTensorGuard::<i64>::new(&w_view, batch_size);
        let p_guard = SafeTensorGuard::<i64>::new(&p_view, batch_size);
        let n_guard = SafeTensorGuard::<i64>::new(&n_view, batch_size);

        for (i, guard) in recurrent_slots.iter().enumerate() {
            let slot = guard.slot;
            let (req, ts) = unsafe { (*queue.metadata[slot].get()).take().unwrap() };
            queue.latency_sum_nanos.fetch_add(
                ts.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            queue
                .latency_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            w_guard.slice[i] = req.worker_id as i64;
            p_guard.slice[i] = req.parent_cache_index as i64;
            n_guard.slice[i] = req.leaf_cache_index as i64;
            evaluation_requests.push(req);
        }
        let pad_w = w_guard.slice[0];
        let pad_p = p_guard.slice[0];
        let pad_n = n_guard.slice[0];
        for i in actual_size..batch_size {
            w_guard.slice[i] = pad_w;
            p_guard.slice[i] = pad_p;
            n_guard.slice[i] = pad_n;
        }
    }

    let mut w_tensor = gpu_workers.narrow(0, 0, batch_size as i64);
    w_tensor.copy_(&w_view);
    let mut p_tensor = gpu_parents.narrow(0, 0, batch_size as i64);
    p_tensor.copy_(&p_view);
    let mut n_tensor = gpu_nodes.narrow(0, 0, batch_size as i64);
    n_tensor.copy_(&n_view);

    let maximum_allowed_nodes_in_search_tree_tensor =
        Tensor::from_slice(&[maximum_allowed_nodes_in_search_tree as i64])
            .to_device(computation_device);
    let flat_p_indices = (&w_tensor * &maximum_allowed_nodes_in_search_tree_tensor) + &p_tensor;
    let flat_n_indices = (&w_tensor * &maximum_allowed_nodes_in_search_tree_tensor) + &n_tensor;

    let hidden_state_batch = latent_cache.index_select(0, &flat_p_indices);

    let (hidden_state_next_batch, value_prefix_batch, value_batch, policy_batch, _) =
        if let Some(cmod) = cmodule_inference {
            let ivalue = match cmod.method_is(
                "recurrent_inference",
                &[
                    tch::IValue::Tensor(hidden_state_batch),
                    tch::IValue::Tensor(piece_action_batch),
                    tch::IValue::Tensor(piece_identifier_batch),
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
                let value_prefix = if let tch::IValue::Tensor(r) = tup.remove(1) {
                    r
                } else {
                    unreachable!()
                };
                let hidden = if let tch::IValue::Tensor(h) = tup.remove(0) {
                    h
                } else {
                    unreachable!()
                };
                (hidden, value_prefix, value, policy, extra)
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

    let current_sz = actual_size as i64;
    let valid_n_indices = flat_n_indices.narrow(0, 0, current_sz);
    let valid_hidden_next = hidden_state_next_batch.narrow(0, 0, current_sz);

    let _ = latent_cache.index_copy_(0, &valid_n_indices, &valid_hidden_next);

    std::thread::spawn(move || {
        let value_prefix_predictions_cpu = value_prefix_batch
            .to_device(Device::Cpu)
            .to_kind(Kind::Float);
        let value_predictions_cpu = value_batch.to_device(Device::Cpu).to_kind(Kind::Float);
        let policy_predictions_cpu = policy_batch.to_device(Device::Cpu).to_kind(Kind::Float);

        let value_prefix_predictions_f32: Vec<f32> = value_prefix_predictions_cpu
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
        for (index, request) in evaluation_requests.into_iter().enumerate() {
            let start_policy = index * policy_vector_size;
            let end_policy = (index + 1) * policy_vector_size;

            let response = EvaluationResponse {
                value_prefix: value_prefix_predictions_f32[index],
                value: value_predictions_f32[index],
                child_prior_probabilities_tensor: policy_predictions_f32[start_policy..end_policy]
                    .try_into()
                    .unwrap(),
                node_index: request.node_index,
                generation: request.generation,
            };
            let _ = request.evaluation_request_transmitter.send(response);
        }

        // recurrent_slots will naturally drop here, and their QueueSlotGuards will return slots to free_tx.
    });
}
