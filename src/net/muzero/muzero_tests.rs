use super::*;
use tch::{nn, Device, Kind};

#[test]
fn test_muzero_nan_safety() {
    let variable_store = nn::VarStore::new(Device::Cpu);
    let neural_engine = MuZeroNet::new(
        &variable_store.root(),
        256,
        4,
        300,
        300,
        crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
        64,
    );
    let batch_size = 1;

    let batched_state = Tensor::zeros(
        [
            batch_size,
            crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
            8,
            16,
        ],
        (Kind::Float, Device::Cpu),
    );

    let (hidden_state, value_scalar, policy_probs, hidden_state_logits) =
        neural_engine.initial_inference(&batched_state);
    assert_eq!(
        i64::try_from(hidden_state.isnan().any()).unwrap(),
        0,
        "NaN in representation"
    );
    assert_eq!(
        i64::try_from(value_scalar.isnan().any()).unwrap(),
        0,
        "NaN in initial value"
    );
    assert_eq!(
        i64::try_from(policy_probs.isnan().any()).unwrap(),
        0,
        "NaN in initial policy"
    );
    assert_eq!(
        i64::try_from(hidden_state_logits.isnan().any()).unwrap(),
        0,
        "NaN in hole logits"
    );

    let batched_action = Tensor::zeros([batch_size], (Kind::Int64, Device::Cpu));
    let batched_piece_identifier = Tensor::zeros([batch_size], (Kind::Int64, Device::Cpu));

    let (
        hidden_state_next,
        value_prefix_scalar,
        value_scalar_next,
        policy_probs_next,
        hidden_state_logits_next,
    ) = neural_engine.recurrent_inference(
        &hidden_state,
        &batched_action,
        &batched_piece_identifier,
    );

    assert_eq!(
        i64::try_from(hidden_state_next.isnan().any()).unwrap(),
        0,
        "NaN in recurrent hidden_state"
    );
    assert_eq!(
        i64::try_from(value_prefix_scalar.isnan().any()).unwrap(),
        0,
        "NaN in recurrent value_prefix"
    );
    assert_eq!(
        i64::try_from(value_scalar_next.isnan().any()).unwrap(),
        0,
        "NaN in recurrent value"
    );
    assert_eq!(
        i64::try_from(policy_probs_next.isnan().any()).unwrap(),
        0,
        "NaN in recurrent policy"
    );
    assert_eq!(
        i64::try_from(hidden_state_logits_next.isnan().any()).unwrap(),
        0,
        "NaN in recurrent logits"
    );
}

#[test]
fn test_support_vector_round_trip() {
    let variable_store = nn::VarStore::new(Device::Cpu);
    let neural_engine = MuZeroNet::new(
        &variable_store.root(),
        256,
        4,
        300,
        300,
        crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
        64,
    );

    // Test strictly positive scalars as the domain was deliberately shifted to [0, +inf)
    let original_scalars = Tensor::from_slice(&[0.0_f32, 10.0, 0.5, 5.5, 299.9]);

    // 1. Scalar to Support (Probabilities)
    let support_probs = neural_engine.scalar_to_value_support(&original_scalars);

    // 2. Convert Probabilities to Logits to feed into support_to_scalar
    // Add a tiny epsilon to prevent log(0) -> -inf which breaks softmax math
    let logits = (support_probs + 1e-9).log();

    // 3. Support to Scalar
    let reconstructed_scalars = neural_engine.value_support_to_scalar(&logits);

    let diff = (&original_scalars - &reconstructed_scalars).abs();
    let max_diff: f32 = diff.max().try_into().unwrap_or(1.0);

    // Max diff should be small (accounting for 32-bit float / epsilons across sqrt/pow operations)
    assert!(
        max_diff < 0.1,
        "Support Vector Math round-trip failed! Max delta: {}",
        max_diff
    );
}

#[test]
fn test_initial_inference_channels_match() {
    let vs = nn::VarStore::new(Device::Cpu);
    let net = MuZeroNet::new(&vs.root(), 64, 2, 300, 300, 64, 64);
    // Provide input matching configured spatial_channel_count (64)
    let input = Tensor::zeros([2, 64, 8, 16], (Kind::Float, Device::Cpu));
    let (hidden, _, _, _) = net.initial_inference(&input);

    // Output hidden size must match model_dimension (64)
    assert_eq!(hidden.size(), [2, 64, 8, 8]);
}

#[test]
fn test_extract_initial_features_padding() {
    let vs = nn::VarStore::new(Device::Cpu);
    // Network expects 64 spatial channels
    let net = MuZeroNet::new(&vs.root(), 64, 1, 10, 10, 64, 64);

    let batch_size = 3;
    let boards = Tensor::zeros([batch_size as i64], (Kind::Int64, Device::Cpu));
    let avail = Tensor::zeros([batch_size as i64, 3], (Kind::Int, Device::Cpu));
    let hist = Tensor::zeros([batch_size as i64, 10], (Kind::Int64, Device::Cpu));
    let acts = Tensor::zeros([batch_size as i64, 10], (Kind::Int, Device::Cpu));
    let diff = Tensor::zeros([batch_size as i64], (Kind::Int, Device::Cpu));

    let features = net.extract_initial_features(&boards, &avail, &hist, &acts, &diff);

    // The resulting tensor MUST be padded to 64 channels, not NATIVE_FEATURE_CHANNELS (20)
    assert_eq!(features.size(), [batch_size as i64, 64, 8, 16]);
}

#[test]
fn test_extract_unrolled_features_padding() {
    let vs = nn::VarStore::new(Device::Cpu);
    let net = MuZeroNet::new(&vs.root(), 64, 1, 10, 10, 64, 64);
    let batch_size = 2;
    let unroll_steps = 4;

    let boards = Tensor::zeros(
        [batch_size as i64, unroll_steps as i64],
        (Kind::Int64, Device::Cpu),
    );
    let hist = Tensor::zeros(
        [batch_size as i64, unroll_steps as i64, 10],
        (Kind::Int64, Device::Cpu),
    );

    let features = net.extract_unrolled_features(&boards, &hist);

    // Ensure that padding dimension concat logic works properly
    assert_eq!(
        features.size(),
        [batch_size as i64, unroll_steps as i64, 64, 8, 16]
    );
}

#[test]
fn test_spatial_channel_count_identity_padding() {
    let vs = nn::VarStore::new(Device::Cpu);
    // Setup config to precisely match Native channels (e.g. 20)
    let native_channels = crate::core::features::NATIVE_FEATURE_CHANNELS as i64;
    let net = MuZeroNet::new(&vs.root(), 64, 1, 10, 10, native_channels, 64);

    let boards = Tensor::zeros([1], (Kind::Int64, Device::Cpu));
    let avail = Tensor::zeros([1, 3], (Kind::Int, Device::Cpu));
    let hist = Tensor::zeros([1, 10], (Kind::Int64, Device::Cpu));
    let acts = Tensor::zeros([1, 10], (Kind::Int, Device::Cpu));
    let diff = Tensor::zeros([1], (Kind::Int, Device::Cpu));

    let features = net.extract_initial_features(&boards, &avail, &hist, &acts, &diff);

    // Should exactly equal native channels without panic
    assert_eq!(features.size(), [1, native_channels, 8, 16]);
}

#[test]
fn test_representation_forward_channel_doubling() {
    let vs = nn::VarStore::new(Device::Cpu);
    // Setup config with 64 spatial channels
    let net = MuZeroNet::new(&vs.root(), 128, 1, 10, 10, 64, 64);

    // Input 64 channels (the 8x16 hex grid format)
    let state = Tensor::zeros([2, 64, 8, 16], (Kind::Float, Device::Cpu));

    let hidden = net.representation.forward(&state);
    // Hex (8x16) to Cartesian (8x8) doubles channels: 64 -> 128 correctly to match model_dimension
    assert_eq!(hidden.size(), [2, 128, 8, 8]);
}

#[test]
fn test_dynamics_forward_input_channels() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model_dim = 128;
    let net = MuZeroNet::new(&vs.root(), model_dim, 2, 10, 10, 64, 64);

    let batch_size = 2;
    let hidden = Tensor::zeros([batch_size, model_dim, 8, 8], (Kind::Float, Device::Cpu));
    let action = Tensor::zeros([batch_size], (Kind::Int64, Device::Cpu));
    let piece_id = Tensor::zeros([batch_size], (Kind::Int64, Device::Cpu));

    let (next_hidden, reward) = net.dynamics.forward(&hidden, &action, &piece_id);

    assert_eq!(next_hidden.size(), [batch_size, model_dim, 8, 8]);
    assert_eq!(reward.size(), [batch_size, 21]); // reward_support_size (10 * 2 + 1)
}

#[test]
fn test_cuda_cpu_feature_parity() {
    if !tch::Cuda::is_available() {
        println!("Skipping CUDA vs CPU parity test because CUDA is not available on this machine.");
        return;
    }

    let vs_cpu = nn::VarStore::new(Device::Cpu);
    let net_cpu = MuZeroNet::new(
        &vs_cpu.root(),
        64,
        1,
        10,
        10,
        crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
        64,
    );

    let vs_cuda = nn::VarStore::new(Device::Cuda(0));
    let net_cuda = MuZeroNet::new(
        &vs_cuda.root(),
        64,
        1,
        10,
        10,
        crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
        64,
    );

    let batch_size = 1000;

    let boards_cpu = Tensor::randint(i64::MAX, [batch_size as i64, 2], (Kind::Int64, Device::Cpu));
    let avail_cpu = Tensor::randint(82, [batch_size as i64, 3], (Kind::Int, Device::Cpu)) - 1;
    let hist_cpu = Tensor::randint(
        i64::MAX,
        [batch_size as i64, 7, 2],
        (Kind::Int64, Device::Cpu),
    );
    let acts_cpu = Tensor::randint(289, [batch_size as i64, 3], (Kind::Int, Device::Cpu)) - 1;
    let diff_cpu = Tensor::randint(6, [batch_size as i64], (Kind::Int, Device::Cpu)) + 1;

    let boards_cuda = boards_cpu.to_device(Device::Cuda(0));
    let avail_cuda = avail_cpu.to_device(Device::Cuda(0));
    let hist_cuda = hist_cpu.to_device(Device::Cuda(0));
    let acts_cuda = acts_cpu.to_device(Device::Cuda(0));
    let diff_cuda = diff_cpu.to_device(Device::Cuda(0));

    let features_cpu =
        net_cpu.extract_initial_features(&boards_cpu, &avail_cpu, &hist_cpu, &acts_cpu, &diff_cpu);
    let features_cuda = net_cuda.extract_initial_features(
        &boards_cuda,
        &avail_cuda,
        &hist_cuda,
        &acts_cuda,
        &diff_cuda,
    );

    let features_cuda_on_cpu = features_cuda.to_device(Device::Cpu);

    let diff = (&features_cpu - &features_cuda_on_cpu).abs();
    let max_diff: f32 = diff.max().try_into().unwrap_or(1.0);

    assert!(
        max_diff == 0.0,
        "CUDA vs CPU extraction parity failed! Max diff: {}",
        max_diff
    );
}
