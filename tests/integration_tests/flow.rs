use crate::common::get_test_config;
use tricked_engine::net::MuZeroNet;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};

#[test]
fn test_flow_convergence() {
    if !tch::Cuda::is_available() {
        println!("Skipping test on CPU to save CI resources");
        return;
    }
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
        tricked_engine::core::features::NATIVE_FEATURE_CHANNELS as i64,
        64,
    );

    let mut opt = nn::Adam::default().build(&vs, 0.0001).unwrap();

    let batch_size = 4;
    let obs = Tensor::randn(
        [
            batch_size,
            tricked_engine::core::features::NATIVE_FEATURE_CHANNELS as i64,
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
