use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicI32, Ordering};
use std::sync::RwLock;
use tch::Device;
use crate::queue::FixedInferenceQueue;
use crate::train::buffer::ReplayBuffer;
use crate::config::Config;
use crate::env::worker as selfplay;
use crate::env::reanalyze;
use crate::net::MuZeroNet;
use arc_swap::ArcSwap;

#[allow(clippy::too_many_arguments)]
pub fn spawn_inference_thread(
    inference_queue: Arc<FixedInferenceQueue>,
    active_inference_net: Arc<ArcSwap<MuZeroNet>>,
    cmodule_inference: Option<Arc<tch::CModule>>,
    active_training_flag: Arc<AtomicBool>,
    configuration_model_dimension: i64,
    computation_device: Device,
    total_workers: usize,
    max_nodes: usize,
    inference_batch_size_limit: usize,
    inference_timeout_milliseconds: u64,
    shared_queue_saturation: Arc<AtomicU32>,
) {
    let _ = std::thread::Builder::new()
        .name("inference".into())
        .spawn(move || {
            while active_training_flag.load(Ordering::Relaxed) {
                selfplay::inference_loop(selfplay::InferenceLoopParams {
                    receiver_queue: Arc::clone(&inference_queue),
                    shared_neural_model: Arc::clone(&active_inference_net),
                    cmodule_inference: cmodule_inference.clone(),
                    model_dimension: configuration_model_dimension,
                    computation_device,
                    total_workers,
                    maximum_allowed_nodes_in_search_tree: max_nodes,
                    inference_batch_size_limit,
                    inference_timeout_milliseconds,
                    active_flag: Arc::clone(&active_training_flag),
                    shared_queue_saturation: Arc::clone(&shared_queue_saturation),
                });
            }
        });
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_mcts_workers(
    worker_count: usize,
    configuration_arc: Arc<Config>,
    inference_queue: Arc<FixedInferenceQueue>,
    shared_replay_buffer: Arc<ReplayBuffer>,
    active_training_flag: Arc<AtomicBool>,
    shared_heatmap: Arc<RwLock<[f32; 96]>>,
    global_difficulty: Arc<AtomicI32>,
    global_gumbel_scale_multiplier: Arc<AtomicU32>,
) {
    for worker_id in 0..worker_count {
        let thread_configuration = Arc::clone(&configuration_arc);
        let thread_evaluation_sender = Arc::clone(&inference_queue);
        let thread_replay_buffer = Arc::clone(&shared_replay_buffer);
        let thread_active_flag = Arc::clone(&active_training_flag);
        let thread_heatmap = Arc::clone(&shared_heatmap);
        let thread_difficulty = Arc::clone(&global_difficulty);
        let thread_gumbel_multiplier = Arc::clone(&global_gumbel_scale_multiplier);

        let _ = std::thread::Builder::new()
            .name(format!("mcts-worker-{}", worker_id))
            .spawn(move || {
                while thread_active_flag.load(Ordering::Relaxed) {
                    selfplay::game_loop(selfplay::GameLoopExecutionParameters {
                        configuration: Arc::clone(&thread_configuration),
                        evaluation_transmitter: Arc::clone(&thread_evaluation_sender),
                        experience_buffer: Arc::clone(&thread_replay_buffer),
                        worker_id,
                        active_flag: Arc::clone(&thread_active_flag),
                        shared_heatmap: Arc::clone(&thread_heatmap),
                        global_difficulty: Arc::clone(&thread_difficulty),
                        global_gumbel_scale_multiplier: Arc::clone(&thread_gumbel_multiplier),
                    });
                }
            });
    }
}

pub fn spawn_reanalyze_workers(
    base_worker_id: usize,
    worker_count: usize,
    configuration_arc: Arc<Config>,
    inference_queue: Arc<FixedInferenceQueue>,
    shared_replay_buffer: Arc<ReplayBuffer>,
    active_training_flag: Arc<AtomicBool>,
) {
    for worker_index in 0..worker_count {
        let worker_id = base_worker_id + worker_index;
        let thread_configuration = Arc::clone(&configuration_arc);
        let thread_evaluation_sender = Arc::clone(&inference_queue);
        let thread_replay_buffer = Arc::clone(&shared_replay_buffer);
        let thread_active_flag = Arc::clone(&active_training_flag);

        let _ = std::thread::Builder::new()
            .name(format!("reanalyze-{}", worker_id))
            .spawn(move || {
                while thread_active_flag.load(Ordering::Relaxed) {
                    reanalyze::reanalyze_worker_loop(
                        Arc::clone(&thread_configuration),
                        Arc::clone(&thread_evaluation_sender),
                        Arc::clone(&thread_replay_buffer),
                        worker_id,
                        Arc::clone(&thread_active_flag),
                    );
                }
            });
    }
}
