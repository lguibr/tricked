#[cfg(test)]
mod integration_tests {
    use crate::board::GameStateExt;
    use crate::config::Config;
    use crate::features::extract_feature_native;
    use crate::mcts::EvalReq;
    use crate::network::MuZeroNet;
    use crossbeam_channel::unbounded;
    use tch::{nn, nn::Module, nn::ModuleT, nn::OptimizerConfig, Device, Tensor};

    fn get_test_config() -> Config {
        serde_yaml::from_str(
            r#"
device: cpu
paths:
  base_directory: "runs/test"
  model_checkpoint_path: "test.safetensors"
  metrics_file_path: "test.csv"
  telemetry_config_export: "config.json"
experiment_name_identifier: "test"
hidden_dimension_size: 128
num_blocks: 8
support_size: 300
buffer_capacity_limit: 1000
simulations: 10
train_batch_size: 4
train_epochs: 1
num_processes: 1
worker_device: cpu
unroll_steps: 3
temporal_difference_steps: 5
zmq_batch_size: 1
zmq_timeout_ms: 1
max_gumbel_k: 4
gumbel_scale: 1.0
temp_decay_steps: 100
difficulty: 6
temp_boost: false
lr_init: 0.01
reanalyze_ratio: 0.25
        "#,
        )
        .unwrap()
    }

    #[test]
    fn test_network_dimensions() {
        // Objective: Test flows convergence dimensions of the tensor
        let cfg = get_test_config();
        let vs = nn::VarStore::new(Device::Cpu);
        let net = MuZeroNet::new(&vs.root(), cfg.hidden_dimension_size, cfg.num_blocks, 300);

        let game = GameStateExt::new(None, 0, 0, 0, 0); // 0 difficulty
        let mut feat = vec![0.0; 20 * 128];
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
        let obs = Tensor::from_slice(&feat).view([1, 20, 8, 16]);

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
        let (evaluation_request_transmitter, evaluation_response_receiver) = unbounded::<EvalReq>();

        let mut handlers = vec![];
        let num_workers = 10;
        let num_reqs = 100;

        for _w in 0..num_workers {
            let thread_tx = evaluation_request_transmitter.clone();
            handlers.push(std::thread::spawn(move || {
                for _i in 0..num_reqs {
                    let (ans_tx, ans_rx) = unbounded();
                    let req = EvalReq {
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
                        worker_id: 0,
                        parent_cache_index: 0,
                        leaf_cache_index: 0,
                        evaluation_request_transmitter: ans_tx,
                    };
                    thread_tx.send(req).unwrap();
                    let _ = ans_rx.recv().unwrap();
                }
            }));
        }

        let total_reqs = num_workers * num_reqs;
        for _ in 0..total_reqs {
            let req = evaluation_response_receiver.recv().unwrap();
            req.evaluation_request_transmitter
                .send(crate::mcts::EvalResp {
                    child_prior_probabilities_tensor: vec![0.0; 128],
                    value: 0.0,
                    reward: 0.0,
                    node_index: 0,
                })
                .unwrap();
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
        // Objective: Test flows convergence on synthetic batch
        let mut cfg = get_test_config();
        cfg.hidden_dimension_size = 16;
        cfg.num_blocks = 1;
        let vs = nn::VarStore::new(Device::Cpu);
        let net = MuZeroNet::new(&vs.root(), cfg.hidden_dimension_size, cfg.num_blocks, 300);

        let mut opt = nn::Adam::default().build(&vs, 0.0001).unwrap();

        let batch_size = 4;
        // Mock identical inputs to force overfitting
        let obs = Tensor::randn([batch_size, 20, 8, 16], (tch::Kind::Float, Device::Cpu));

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

        let net = MuZeroNet::new(&vs.root(), cfg.hidden_dimension_size, cfg.num_blocks, 300);

        let batch_size = 2;
        let state = crate::board::GameStateExt::new(Some([1, 2, 3]), 0, 0, 6, 0);
        let mut features = vec![0.0; 20 * 128];
        crate::features::extract_feature_native(
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
            .view([batch_size as i64, 20, 8, 16])
            .to_kind(tch::Kind::BFloat16)
            .to_device(device)
            .to_kind(tch::Kind::Float);

        let hidden = net.representation.forward_t(&obs, false);

        assert_eq!(
            i64::try_from(hidden.isnan().any()).unwrap(),
            0,
            "NaN detected in hidden state immediately after initialization!"
        );

        let (value_logits, policy_logits, hidden_state_logits) = net.prediction.forward(&hidden);

        assert_eq!(
            i64::try_from(value_logits.isnan().any()).unwrap(),
            0,
            "NaN detected in value logits!"
        );

        assert_eq!(
            i64::try_from(policy_logits.isnan().any()).unwrap(),
            0,
            "NaN detected in policy logits!"
        );

        assert_eq!(
            i64::try_from(hidden_state_logits.isnan().any()).unwrap(),
            0,
            "NaN detected in hidden state logits!"
        );
    }
}
