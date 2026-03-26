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
model_checkpoint: "test.safetensors"
metrics_file: "test.csv"
d_model: 32
num_blocks: 2
support_size: 300
capacity: 1000
num_games: 1
simulations: 10
train_batch_size: 4
train_epochs: 1
num_processes: 1
worker_device: cpu
unroll_steps: 3
td_steps: 5
zmq_inference_port: "5555"
zmq_batch_size: 1
zmq_timeout_ms: 1
max_gumbel_k: 4
gumbel_scale: 1.0
temp_decay_steps: 100
difficulty: 6
exploit_starts: []
temp_boost: false
exp_name: "test"
lr_init: 0.01
        "#,
        )
        .unwrap()
    }

    #[test]
    fn test_network_dimensions() {
        // Objective: Test flows convergence dimensions of the tensor
        let cfg = get_test_config();
        let vs = nn::VarStore::new(Device::Cpu);
        let net = MuZeroNet::new(&vs.root(), cfg.d_model, cfg.num_blocks, 300);

        let game = GameStateExt::new(None, 0, 0, 0, 0); // 0 difficulty
        let feat = extract_feature_native(&game, None, None, 0);

        // Tensor dimension mapping
        // B=1, C=20, L=96
        let obs = Tensor::from_slice(&feat).view([1, 20, 96]);

        let hidden = net.representation.forward_t(&obs, false);
        assert_eq!(
            hidden.size(),
            vec![1i64, 32i64, 96i64],
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

        let action = Tensor::zeros(&[1], (tch::Kind::Int64, Device::Cpu));
        let piece_id = Tensor::zeros(&[1], (tch::Kind::Int64, Device::Cpu));
        let (next_hidden, reward_logits) = net.dynamics.forward(&hidden, &action, &piece_id);

        assert_eq!(
            next_hidden.size(),
            vec![1i64, 32i64, 96i64],
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
        let (tx, rx) = unbounded::<EvalReq>();

        let mut handlers = vec![];
        let num_workers = 10;
        let num_reqs = 100;

        for _w in 0..num_workers {
            let thread_tx = tx.clone();
            handlers.push(std::thread::spawn(move || {
                for _i in 0..num_reqs {
                    let (ans_tx, ans_rx) = unbounded();
                    let req = EvalReq {
                        is_initial: true,
                        state_feat: Some(vec![0.0; 20 * 96]),
                        h_last: None,
                        piece_action: 0,
                        piece_id: 0,
                        tx: ans_tx,
                    };
                    thread_tx.send(req).unwrap();
                    let _ = ans_rx.recv().unwrap();
                }
            }));
        }

        let total_reqs = num_workers * num_reqs;
        for _ in 0..total_reqs {
            let req = rx.recv().unwrap();
            req.tx
                .send(crate::mcts::EvalResp {
                    h_next: vec![0.0; 32 * 96],
                    p_next: vec![0.0; 96],
                    value: 0.0,
                    reward: 0.0,
                })
                .unwrap();
        }

        for h in handlers {
            h.join().unwrap();
        }
        assert!(rx.is_empty(), "Channel should be thoroughly processed");
    }

    #[test]
    fn test_flow_convergence() {
        // Objective: Test flows convergence on synthetic batch
        let cfg = get_test_config();
        let vs = nn::VarStore::new(Device::Cpu);
        let net = MuZeroNet::new(&vs.root(), cfg.d_model, cfg.num_blocks, 300);

        let mut opt = nn::Adam::default().build(&vs, cfg.lr_init).unwrap();

        let batch_size = 4;
        // Mock identical inputs to force overfitting
        let obs = Tensor::zeros(&[batch_size, 20, 96], (tch::Kind::Float, Device::Cpu));

        let target_value = Tensor::zeros(&[batch_size, 601], (tch::Kind::Float, Device::Cpu));
        let _ = target_value.narrow(1, 300, 1).fill_(1.0); // 0 score

        let target_policy = Tensor::zeros(&[batch_size, 288], (tch::Kind::Float, Device::Cpu));
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
}
