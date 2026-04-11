use crate::queue::FixedInferenceQueue;
use arc_swap::ArcSwap;
use std::sync::Arc;
use tch::{Device, Kind, Tensor};


use crate::net::MuZeroNet;
use crate::env::worker::inference::initial::process_initial_inference;
use crate::env::worker::inference::recurrent::process_recurrent_inference;

pub(crate) struct SafeTensorGuard<'a, T> {
    _tensor: &'a Tensor,
    pub slice: &'a mut [T],
}

impl<'a, T> SafeTensorGuard<'a, T> {
    pub(crate) fn new(tensor: &'a Tensor, len: usize) -> Self {
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

    // Memory pinning removed to avoid PyTorch deprecation warnings

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




