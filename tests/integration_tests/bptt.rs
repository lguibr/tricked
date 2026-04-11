use crate::common::get_test_config;
use tricked_engine::net::MuZeroNet;
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Tensor};

#[test]
fn test_end_to_end_bptt_flow() {
    if !tch::Cuda::is_available() {
        println!("Skipping test on CPU to save CI resources");
        return;
    }
    tch::set_num_threads(1);
    tch::manual_seed(42);
    let mut cfg = get_test_config();
    cfg.hardware.device = "cpu".to_string();
    cfg.hardware.worker_device = "cpu".to_string();
    cfg.hardware.num_processes = 1;
    // Keep dimensions extremely low so we don't block tests for too long
    cfg.mcts.simulations = 2;
    cfg.optimizer.train_batch_size = 2;
    cfg.optimizer.temporal_difference_steps = 2;
    cfg.optimizer.unroll_steps = 2;
    cfg.architecture.hidden_dimension_size = 16;
    cfg.architecture.num_blocks = 1;
    cfg.architecture.value_support_size = 10;
    cfg.architecture.reward_support_size = 10;

    let configuration_arc = std::sync::Arc::new(cfg);
    let shared_replay_buffer = std::sync::Arc::new(tricked_engine::train::buffer::ReplayBuffer::new(
        100,
        configuration_arc.optimizer.temporal_difference_steps,
        configuration_arc.optimizer.unroll_steps,
        configuration_arc.optimizer.train_batch_size,
        None,
        configuration_arc.optimizer.discount_factor,
        configuration_arc.optimizer.td_lambda,
    ));

    let computation_device = Device::Cpu;
    let training_var_store = nn::VarStore::new(computation_device);
    let inference_var_store = nn::VarStore::new(computation_device);
    let exponential_moving_average_var_store = nn::VarStore::new(computation_device);

    let training_network = MuZeroNet::new(
        &training_var_store.root(),
        configuration_arc.architecture.hidden_dimension_size,
        configuration_arc.architecture.num_blocks,
        configuration_arc.architecture.value_support_size,
        configuration_arc.architecture.reward_support_size,
        configuration_arc.architecture.spatial_channel_count,
        configuration_arc.architecture.hole_predictor_dim,
    );
    let _ema_network = MuZeroNet::new(
        &exponential_moving_average_var_store.root(),
        configuration_arc.architecture.hidden_dimension_size,
        configuration_arc.architecture.num_blocks,
        configuration_arc.architecture.value_support_size,
        configuration_arc.architecture.reward_support_size,
        configuration_arc.architecture.spatial_channel_count,
        configuration_arc.architecture.hole_predictor_dim,
    );
    let inference_net_a = std::sync::Arc::new(MuZeroNet::new(
        &inference_var_store.root(),
        configuration_arc.architecture.hidden_dimension_size,
        configuration_arc.architecture.num_blocks,
        configuration_arc.architecture.value_support_size,
        configuration_arc.architecture.reward_support_size,
        configuration_arc.architecture.spatial_channel_count,
        configuration_arc.architecture.hole_predictor_dim,
    ));
    let active_inference_net = std::sync::Arc::new(arc_swap::ArcSwap::from(
        std::sync::Arc::clone(&inference_net_a),
    ));

    let _dummy_obs = Tensor::zeros(
        [
            1,
            tricked_engine::core::features::NATIVE_FEATURE_CHANNELS as i64,
            8,
            16,
        ],
        (tch::Kind::Float, Device::Cpu),
    );
    let _dummy_hidden = training_network
        .representation
        .forward_t(&_dummy_obs, false);
    let _ = training_network.prediction.forward(&_dummy_hidden);
    let _ = training_network.dynamics.forward(
        &_dummy_hidden,
        &Tensor::zeros([1], (tch::Kind::Int64, Device::Cpu)),
        &Tensor::zeros([1], (tch::Kind::Int64, Device::Cpu)),
    );

    let active_training_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let inference_active_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    
    // NOTE: FixedInferenceQueue is not pub(crate) inside mcts/queue or env ?
    let inference_queue = std::sync::Arc::new(tricked_engine::queue::FixedInferenceQueue::new(
        configuration_arc.optimizer.buffer_capacity_limit,
        2,
    ));

    // Inference Thread
    let thread_evaluation_receiver = std::sync::Arc::clone(&inference_queue);
    let thread_network_mutex = std::sync::Arc::clone(&active_inference_net);
    let thread_active_flag = std::sync::Arc::clone(&inference_active_flag);
    let configuration_model_dimension = configuration_arc.architecture.hidden_dimension_size;
    let inference_hnd = std::thread::spawn(move || {
        tricked_engine::env::worker::inference_loop(tricked_engine::env::worker::InferenceLoopParams {
            receiver_queue: std::sync::Arc::clone(&thread_evaluation_receiver),
            shared_neural_model: std::sync::Arc::clone(&thread_network_mutex),
            cmodule_inference: None,
            model_dimension: configuration_model_dimension,
            computation_device,
            total_workers: 2, 
            maximum_allowed_nodes_in_search_tree: 1000,
            inference_batch_size_limit: 1,
            inference_timeout_milliseconds: 1,
            active_flag: std::sync::Arc::clone(&thread_active_flag),
            shared_queue_saturation: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
        });
    });

    // Worker Thread
    let worker_configuration = std::sync::Arc::clone(&configuration_arc);
    let worker_evaluation_sender = std::sync::Arc::clone(&inference_queue);
    let worker_replay_buffer = std::sync::Arc::clone(&shared_replay_buffer);
    let worker_active_flag = std::sync::Arc::clone(&active_training_flag);
    let worker_hnd = std::thread::spawn(move || {
        tricked_engine::env::worker::game_loop(tricked_engine::env::worker::GameLoopExecutionParameters {
            configuration: std::sync::Arc::clone(&worker_configuration),
            evaluation_transmitter: std::sync::Arc::clone(&worker_evaluation_sender),
            experience_buffer: std::sync::Arc::clone(&worker_replay_buffer),
            worker_id: 0,
            active_flag: std::sync::Arc::clone(&worker_active_flag),
            shared_heatmap: std::sync::Arc::new(std::sync::RwLock::new([0.0; 96])),
            global_difficulty: std::sync::Arc::new(std::sync::atomic::AtomicI32::new(
                worker_configuration.environment.difficulty,
            )),
            global_gumbel_scale_multiplier: std::sync::Arc::new(
                std::sync::atomic::AtomicU32::new(1.0_f32.to_bits()),
            ),
        });
    });

    let mut gradient_optimizer = nn::Adam::default()
        .build(&training_var_store, 0.01)
        .unwrap();
    let mut training_steps = 0;
    let mut initial_loss = f64::MAX;
    let mut final_loss = 0.0_f64;

    let prefetch_batch_size = configuration_arc.optimizer.train_batch_size;
    let mut games_seen = 0;

    while active_training_flag.load(std::sync::atomic::Ordering::Relaxed) {
        let current_games = shared_replay_buffer
            .state
            .completed_games
            .load(std::sync::atomic::Ordering::Relaxed);

        if current_games < games_seen + 1 {
            std::thread::sleep(std::time::Duration::from_millis(50));
            continue;
        }

        if let Some(mut batch) = shared_replay_buffer.sample_batch(prefetch_batch_size, 1.0) {
            games_seen += 1;

            let mut arena = tricked_engine::train::arena::PinnedBatchTensors::new(
                prefetch_batch_size,
                configuration_arc.optimizer.unroll_steps,
                computation_device,
            );
            arena.copy_from_unpinned(&batch);

            let mut gpu_arena = tricked_engine::train::arena::GpuBatchTensors::new(
                prefetch_batch_size,
                configuration_arc.optimizer.unroll_steps,
                computation_device,
            );
            gpu_arena.copy_from_pinned(&arena);

            batch.state_features_batch = gpu_arena.state_features.shallow_clone();
            batch.actions_batch = gpu_arena.actions.shallow_clone();
            batch.piece_identifiers_batch = gpu_arena.piece_identifiers.shallow_clone();
            batch.value_prefixs_batch = gpu_arena.value_prefixs.shallow_clone();
            batch.target_policies_batch = gpu_arena.target_policies.shallow_clone();
            batch.target_values_batch = gpu_arena.target_values.shallow_clone();
            batch.model_values_batch = gpu_arena.model_values.shallow_clone();

            batch.loss_masks_batch = gpu_arena.loss_masks.shallow_clone();
            batch.importance_weights_batch = gpu_arena.importance_weights.shallow_clone();

            let mut cmodule_bptt = {
                let abs_bptt_path = "/tmp/bptt_kernel_test_end_to_end.pt";
                if !std::path::Path::new(abs_bptt_path).exists() {
                    let _ = std::process::Command::new("python3")
                        .args([
                            "scripts/export_bptt.py",
                            "--blocks",
                            "1",
                            "--channels",
                            "16",
                            "--support",
                            "10",
                            "--spatial-channels",
                            &configuration_arc
                                .architecture
                                .spatial_channel_count
                                .to_string(),
                            "--output",
                            abs_bptt_path,
                        ])
                        .status();
                }
                tch::CModule::load(abs_bptt_path).unwrap()
            };

            let step_metrics = tricked_engine::train::optimizer::optimization::train_step(
                &training_network,
                &mut gradient_optimizer,
                &shared_replay_buffer,
                &batch,
                configuration_arc.optimizer.unroll_steps,
                &training_var_store,
                &mut cmodule_bptt,
            );

            assert!(
                !step_metrics.total_loss.is_nan(),
                "NaN detected in total loss!"
            );

            if training_steps == 0 {
                initial_loss = step_metrics.total_loss;
            }
            final_loss = step_metrics.total_loss;
            training_steps += 1;

            if training_steps >= 5 {
                break;
            }
        } else {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    active_training_flag.store(false, std::sync::atomic::Ordering::SeqCst);
    let _ = worker_hnd.join();

    inference_active_flag.store(false, std::sync::atomic::Ordering::SeqCst);
    let _ = inference_hnd.join();

    assert!(
        final_loss < initial_loss + 1.0,
        "Loss exploded! Initial: {}, Final: {}",
        initial_loss,
        final_loss
    );
}
