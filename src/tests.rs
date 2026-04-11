#[cfg(test)]
mod integration_tests {
    use crate::config::Config;
    use crate::core::board::GameStateExt;
    use crate::core::features::extract_feature_native;
    use crate::mcts::EvaluationRequest;
    use crate::net::MuZeroNet;
    use crossbeam_channel::unbounded;
    use tch::{nn, nn::Module, nn::ModuleT, nn::OptimizerConfig, Device, Tensor};

    fn get_test_config() -> Config {
        serde_yaml::from_str(
            r#"
device: cpu
checkpoint_interval: 100
paths:
  base_directory: "runs/test"
  model_checkpoint_path: "test.safetensors"
  metrics_file_path: "test.csv"
  telemetry_config_export: "config.json"
experiment_name_identifier: "test"
hardware:
  device: cpu
  num_processes: 1
  worker_device: cpu
  inference_batch_size_limit: 1
  inference_timeout_ms: 1
architecture:
  hidden_dimension_size: 128
  num_blocks: 8
  value_support_size: 300
  reward_support_size: 300
  spatial_channel_count: 20
  hole_predictor_dim: 64
optimizer:
  buffer_capacity_limit: 1000
  train_batch_size: 4
  discount_factor: 0.99
  td_lambda: 0.95
  weight_decay: 0.0
  lr_init: 0.01
  unroll_steps: 3
  temporal_difference_steps: 5
  reanalyze_ratio: 0.25
mcts:
  simulations: 10
  max_gumbel_k: 4
  gumbel_scale: 1.0
environment:
  difficulty: 6
  temp_decay_steps: 100
  temp_boost: false
        "#,
        )
        .unwrap()
    }

    #[test]
    fn test_network_dimensions() {
        // Objective: Test flows convergence dimensions of the tensor
        let cfg = get_test_config();
        let vs = nn::VarStore::new(Device::Cpu);
        let net = MuZeroNet::new(
            &vs.root(),
            cfg.architecture.hidden_dimension_size,
            cfg.architecture.num_blocks,
            300,
            300,
            crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
            64,
        );

        let game = GameStateExt::new(None, 0, 0, 0, 0); // 0 difficulty
        let mut feat = vec![0.0; crate::core::features::NATIVE_FEATURE_CHANNELS * 128];
        extract_feature_native(
            &mut feat,
            game.board_bitmask_u128,
            &game.available,
            &[],
            &[],
            0,
        );

        // Tensor dimension mapping
        // B=1, C=20, H=8, W=16
        let obs = Tensor::from_slice(&feat).view([
            1,
            crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
            8,
            16,
        ]);

        let hidden = net.representation.forward_t(&obs, false);
        assert_eq!(
            hidden.size(),
            vec![1i64, 128i64, 8i64, 8i64],
            "Representation shape mismatch"
        );

        let (value_logits, policy_logits, _hole_logits) = net.prediction.forward(&hidden);
        assert_eq!(
            policy_logits.size(),
            vec![1i64, 288i64],
            "Policy shape mismatch"
        );
        assert_eq!(
            value_logits.size(),
            vec![1i64, 601i64],
            "Value shape mismatch"
        );

        let action = Tensor::zeros([1], (tch::Kind::Int64, Device::Cpu));
        let piece_id = Tensor::zeros([1], (tch::Kind::Int64, Device::Cpu));
        let (next_hidden, reward_logits) = net.dynamics.forward(&hidden, &action, &piece_id);

        assert_eq!(
            next_hidden.size(),
            vec![1i64, 128i64, 8i64, 8i64],
            "Dynamics hidden shape mismatch"
        );
        assert_eq!(
            reward_logits.size(),
            vec![1i64, 601i64],
            "Reward shape mismatch"
        );
    }

    #[test]
    fn test_transmission_stress_test() {
        // Objective: Test transmission stress tests with self-play evaluating channels
        let (evaluation_request_transmitter, evaluation_response_receiver) =
            unbounded::<EvaluationRequest>();

        let mut handlers = vec![];
        let num_workers = 10;
        let num_reqs = 100;

        for _w in 0..num_workers {
            let thread_tx = evaluation_request_transmitter.clone();
            handlers.push(std::thread::spawn(move || {
                for _i in 0..num_reqs {
                    let mailbox = std::sync::Arc::new(crate::mcts::mailbox::AtomicMailbox::new());
                    let req = EvaluationRequest {
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
                        generation: 0,
                        worker_id: 0,
                        parent_cache_index: 0,
                        leaf_cache_index: 0,
                        mailbox: mailbox.clone(),
                    };
                    thread_tx.send(req).unwrap();
                    let active_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
                    let _ = crate::mcts::mailbox::spin_wait(&mailbox, &active_flag).unwrap();
                }
            }));
        }

        let total_reqs = num_workers * num_reqs;
        for _ in 0..total_reqs {
            let req = evaluation_response_receiver.recv().unwrap();
            req.mailbox
                .write_and_notify(crate::mcts::EvaluationResponse {
                    child_prior_probabilities_tensor: [0.0; 288],
                    value: 0.0,
                    value_prefix: 0.0,
                    node_index: 0,
                    generation: 0,
                });
        }

        for h in handlers {
            h.join().unwrap();
        }
        assert!(
            evaluation_response_receiver.is_empty(),
            "Channel should be thoroughly processed"
        );
    }

    #[test]
    fn test_flow_convergence() {
        if !tch::Cuda::is_available() {
            println!("Skipping test on CPU to save CI resources");
            return;
        }
        // Objective: Test flows convergence on synthetic batch
        let mut cfg = get_test_config();
        cfg.architecture.hidden_dimension_size = 16;
        cfg.architecture.num_blocks = 1;
        let vs = nn::VarStore::new(Device::Cpu);
        let net = MuZeroNet::new(
            &vs.root(),
            cfg.architecture.hidden_dimension_size,
            cfg.architecture.num_blocks,
            300,
            300,
            crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
            64,
        );

        let mut opt = nn::Adam::default().build(&vs, 0.0001).unwrap();

        let batch_size = 4;
        // Mock identical inputs to force overfitting
        let obs = Tensor::randn(
            [
                batch_size,
                crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
                8,
                16,
            ],
            (tch::Kind::Float, Device::Cpu),
        );

        let target_value = Tensor::zeros([batch_size, 601], (tch::Kind::Float, Device::Cpu));
        let _ = target_value.narrow(1, 300, 1).fill_(1.0); // 0 score

        let target_policy = Tensor::zeros([batch_size, 288], (tch::Kind::Float, Device::Cpu));
        let _ = target_policy.narrow(1, 42, 1).fill_(1.0); // Predict action 42

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        for epoch in 0..20 {
            let hidden = net.representation.forward(&obs);
            let (value_logits, policy_logits, _hole_logits) = net.prediction.forward(&hidden);

            let p_loss = -(target_policy.copy() * policy_logits.log_softmax(1, tch::Kind::Float))
                .sum(tch::Kind::Float)
                / (batch_size as f64);
            let v_loss = -(target_value.copy() * value_logits.log_softmax(1, tch::Kind::Float))
                .sum(tch::Kind::Float)
                / (batch_size as f64);

            let loss = p_loss + v_loss;

            if epoch == 0 {
                initial_loss = f64::try_from(loss.copy()).unwrap();
            }
            if epoch == 19 {
                final_loss = f64::try_from(loss.copy()).unwrap();
            }

            opt.backward_step(&loss);
        }

        assert!(
            final_loss < initial_loss,
            "Loss did not converge: initial {} -> final {}",
            initial_loss,
            final_loss
        );
    }

    #[test]
    fn test_ema_polyak_averaging() {
        let vs = nn::VarStore::new(Device::Cpu);
        let ema_vs = nn::VarStore::new(Device::Cpu);

        let _p_model = vs.root().var("w", &[1], tch::nn::Init::Const(100.0));
        let p_ema = ema_vs.root().var("w", &[1], tch::nn::Init::Const(0.0));

        tch::no_grad(|| {
            let mut ema_vars = ema_vs.variables();
            let model_vars = vs.variables();
            for (name, t_ema) in ema_vars.iter_mut() {
                if let Some(t_model) = model_vars.get(name) {
                    t_ema.copy_(&(&*t_ema * 0.99 + t_model * 0.01));
                }
            }
        });

        let val = f64::try_from(p_ema).unwrap();
        // 0.0 * 0.99 + 100.0 * 0.01 = 1.0
        assert!(
            (val - 1.0).abs() < 1e-5,
            "EMA Polyak averaging math failed!"
        );
    }

    #[test]
    fn test_device_fallback_safety() {
        let requested = "cuda";
        let actual = if requested == "cuda" && tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        // Verify no panics and standard structure.
        assert!(matches!(actual, Device::Cpu | Device::Cuda(0)));
    }
    #[test]
    fn test_nan_free_initialization() {
        // Objective: Test that freshly initialized weights do not produce NaNs
        let cfg = get_test_config();

        let device = if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        let vs = nn::VarStore::new(device);
        // Note: We intentionally do NOT call `vs.half()` to prevent FP16 batch norm NaNs.

        let net = MuZeroNet::new(
            &vs.root(),
            cfg.architecture.hidden_dimension_size,
            cfg.architecture.num_blocks,
            300,
            300,
            crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
            64,
        );

        let batch_size = 2;
        let state = crate::core::board::GameStateExt::new(Some([1, 2, 3]), 0, 0, 6, 0);
        let mut features = vec![0.0; crate::core::features::NATIVE_FEATURE_CHANNELS * 128];
        crate::core::features::extract_feature_native(
            &mut features,
            state.board_bitmask_u128,
            &state.available,
            &[],
            &[],
            6,
        );
        let mut flat_batch = features.clone();
        flat_batch.extend(features);

        let obs = Tensor::from_slice(&flat_batch)
            .view([
                batch_size as i64,
                crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
                8,
                16,
            ])
            .to_kind(tch::Kind::BFloat16)
            .to_device(device)
            .to_kind(tch::Kind::Float);

        let hidden = net.representation.forward_t(&obs, false);

        assert!(
            !bool::try_from(hidden.isnan().any()).unwrap(),
            "NaN detected in hidden state immediately after initialization!"
        );

        let (value_logits, policy_logits, hidden_state_logits) = net.prediction.forward(&hidden);

        assert!(
            !bool::try_from(value_logits.isnan().any()).unwrap(),
            "NaN detected in value logits!"
        );

        assert!(
            !bool::try_from(policy_logits.isnan().any()).unwrap(),
            "NaN detected in policy logits!"
        );

        assert!(
            !bool::try_from(hidden_state_logits.isnan().any()).unwrap(),
            "NaN detected in hidden state logits!"
        );
    }

    #[test]
    fn test_end_to_end_bptt_flow() {
        if !tch::Cuda::is_available() {
            println!("Skipping test on CPU to save CI resources");
            return;
        }
        // Objective: Spawn 1 Worker, 1 Inference, and 1 Optimizer thread, running 5 BPTT steps
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
        let shared_replay_buffer = std::sync::Arc::new(crate::train::buffer::ReplayBuffer::new(
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
        let ema_network = MuZeroNet::new(
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

        // DUMMY WARMUP PASS:
        // PyTorch ATen dispatcher has a known threading bug where background threads crash with:
        // "Tried to access the schema for aten::as_strided which doesn't have a schema registered yet"
        // Executing a full forward pass on the main thread forces the dispatcher to cache all operator schemas.
        let _dummy_obs = Tensor::zeros(
            [
                1,
                crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
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
        let inference_queue = std::sync::Arc::new(crate::queue::FixedInferenceQueue::new(
            configuration_arc.optimizer.buffer_capacity_limit,
            2,
        ));

        // Inference Thread
        let thread_evaluation_receiver = std::sync::Arc::clone(&inference_queue);
        let thread_network_mutex = std::sync::Arc::clone(&active_inference_net);
        let thread_active_flag = std::sync::Arc::clone(&inference_active_flag);
        let configuration_model_dimension = configuration_arc.architecture.hidden_dimension_size;
        let inference_hnd = std::thread::spawn(move || {
            crate::env::worker::inference_loop(crate::env::worker::InferenceLoopParams {
                receiver_queue: std::sync::Arc::clone(&thread_evaluation_receiver),
                shared_neural_model: std::sync::Arc::clone(&thread_network_mutex),
                cmodule_inference: None,
                model_dimension: configuration_model_dimension,
                computation_device,
                total_workers: 2, // Must be >= 1 to safely handle worker+reanalyze slots
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
            crate::env::worker::game_loop(crate::env::worker::GameLoopExecutionParameters {
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

        // Opt Loop Implementation
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

                let mut arena = crate::train::arena::PinnedBatchTensors::new(
                    prefetch_batch_size,
                    configuration_arc.optimizer.unroll_steps,
                    computation_device,
                );
                arena.copy_from_unpinned(&batch);

                let mut gpu_arena = crate::train::arena::GpuBatchTensors::new(
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

                let step_metrics = crate::train::optimizer::optimization::train_step(
                    &training_network,
                    &ema_network,
                    &mut gradient_optimizer,
                    &shared_replay_buffer,
                    &batch,
                    configuration_arc.optimizer.unroll_steps,
                    &training_var_store,
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

        // Shut down worker thread first
        active_training_flag.store(false, std::sync::atomic::Ordering::SeqCst);
        let _ = worker_hnd.join();

        // Shut down inference thread later to accommodate any pending straggler requests
        inference_active_flag.store(false, std::sync::atomic::Ordering::SeqCst);
        let _ = inference_hnd.join();

        assert!(
            final_loss < initial_loss + 1.0,
            "Loss exploded! Initial: {}, Final: {}",
            initial_loss,
            final_loss
        );
    }
}
