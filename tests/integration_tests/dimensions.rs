use crate::common::get_test_config;
use tricked_engine::core::board::GameStateExt;
use tricked_engine::core::features::extract_feature_native;
use tricked_engine::net::MuZeroNet;
use tch::{nn, nn::ModuleT, Device, Tensor};

#[test]
fn test_network_dimensions() {
    let cfg = get_test_config();
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

    let game = GameStateExt::new(None, 0, 0, 0, 0); // 0 difficulty
    let mut feat = vec![0.0; tricked_engine::core::features::NATIVE_FEATURE_CHANNELS * 128];
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
        tricked_engine::core::features::NATIVE_FEATURE_CHANNELS as i64,
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
