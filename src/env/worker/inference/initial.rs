use crate::mcts::EvaluationResponse;
use crate::net::MuZeroNet;
use crate::queue::FixedInferenceQueue;
use std::sync::Arc;
use tch::{Device, Kind, Tensor};
use crate::env::worker::inference::loop_runner::SafeTensorGuard;

#[hotpath::measure]
#[allow(clippy::too_many_arguments)]
pub fn process_initial_inference(
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
            request.mailbox.write_and_notify(response);
        }
    });
}
