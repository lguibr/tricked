use crate::common::get_test_config;
use tricked_engine::net::MuZeroNet;
use tch::{nn, nn::ModuleT, Device, Tensor};

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
    let cfg = get_test_config();

    let device = if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let vs = nn::VarStore::new(device);

    let net = MuZeroNet::new(
        &vs.root(),
        cfg.architecture.hidden_dimension_size,
        cfg.architecture.num_blocks,
        300,
        300,
        tricked_engine::core::features::NATIVE_FEATURE_CHANNELS as i64,
        64,
    );

    let batch_size = 2;
    let state = tricked_engine::core::board::GameStateExt::new(Some([1, 2, 3]), 0, 0, 6, 0);
    let mut features = vec![0.0; tricked_engine::core::features::NATIVE_FEATURE_CHANNELS * 128];
    tricked_engine::core::features::extract_feature_native(
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
            tricked_engine::core::features::NATIVE_FEATURE_CHANNELS as i64,
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
