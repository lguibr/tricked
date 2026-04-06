use crate::core::constants::STANDARD_PIECES;
use crate::core::features::CANONICAL_PIECE_MASKS;
use crate::net::{DynamicsNet, PredictionNet, ProjectorNet, RepresentationNet};
use crate::node::COMPACT_PIECE_MASKS;
use tch::{nn, nn::Module, Kind, Tensor};

#[derive(Debug)]
pub struct MuZeroNet {
    pub representation: RepresentationNet,
    pub dynamics: DynamicsNet,
    pub prediction: PredictionNet,
    pub projector: ProjectorNet,
    pub value_support_size: i64,
    pub reward_support_size: i64,
    pub spatial_channel_count: i64,
    pub epsilon_factor: f64,
    pub math_cmodule: tch::CModule,
    pub canonical_tensor: Tensor,
    pub compact_tensor: Tensor,
    pub standard_tensor: Tensor,
    pub num_standard_pieces: i32,
}

unsafe impl Sync for MuZeroNet {}
unsafe impl Send for MuZeroNet {}

impl MuZeroNet {
    pub fn new(
        variable_store: &nn::Path,
        model_dimension: i64,
        convolution_blocks: i64,
        value_support_size: i64,
        reward_support_size: i64,
        spatial_channel_count: i64,
        hole_predictor_dim: i64,
    ) -> Self {
        let representation = RepresentationNet::new(
            &(variable_store / "representation"),
            model_dimension,
            convolution_blocks,
            spatial_channel_count,
        );
        let dynamics = DynamicsNet::new(
            &(variable_store / "dynamics"),
            model_dimension,
            convolution_blocks,
            reward_support_size,
        );
        let prediction = PredictionNet::new(
            &(variable_store / "prediction"),
            model_dimension,
            value_support_size,
            288,
            hole_predictor_dim,
        );
        let projector =
            ProjectorNet::new(&(variable_store / "projector"), model_dimension, 512, 128);

        let mut math_data = std::io::Cursor::new(include_bytes!("../../assets/math_kernels.pt"));
        let math_cmodule =
            tch::CModule::load_data_on_device(&mut math_data, variable_store.device())
                .expect("Failed to load embedded math_kernels.pt");

        let mut canonical_flat = Vec::with_capacity(CANONICAL_PIECE_MASKS.len() * 128);
        for mask in CANONICAL_PIECE_MASKS.iter() {
            for &idx in mask {
                canonical_flat.push(idx as i32);
            }
            for _ in mask.len()..128 {
                canonical_flat.push(-1);
            }
        }

        let mut compact_flat = Vec::with_capacity(COMPACT_PIECE_MASKS.len() * 128);
        for piece_masks in COMPACT_PIECE_MASKS.iter() {
            for &(_rot, mask) in piece_masks {
                compact_flat.push((mask & 0xFFFFFFFFFFFFFFFF) as i64);
                compact_flat.push((mask >> 64) as i64);
            }
            for _ in piece_masks.len()..64 {
                compact_flat.push(0);
                compact_flat.push(0);
            }
        }

        let mut standard_flat = Vec::with_capacity(STANDARD_PIECES.len() * 192);
        for piece in STANDARD_PIECES.iter() {
            for &mask in piece.iter() {
                standard_flat.push((mask & 0xFFFFFFFFFFFFFFFF) as i64);
                standard_flat.push((mask >> 64) as i64);
            }
        }

        let device = variable_store.device();
        let canonical_tensor = Tensor::from_slice(&canonical_flat).to_device(device);
        let compact_tensor = Tensor::from_slice(&compact_flat).to_device(device);
        let standard_tensor = Tensor::from_slice(&standard_flat).to_device(device);

        Self {
            representation,
            dynamics,
            prediction,
            projector,
            value_support_size,
            reward_support_size,
            spatial_channel_count,
            epsilon_factor: 0.001,
            math_cmodule,
            canonical_tensor,
            compact_tensor,
            standard_tensor,
            num_standard_pieces: STANDARD_PIECES.len() as i32,
        }
    }

    pub fn value_support_to_scalar(&self, logits_prediction: &Tensor) -> Tensor {
        let ivalue = self
            .math_cmodule
            .method_is(
                "support_to_scalar",
                &[
                    tch::IValue::Tensor(logits_prediction.copy()),
                    tch::IValue::Int(self.value_support_size),
                    tch::IValue::Double(self.epsilon_factor),
                ],
            )
            .expect("math_cmodule support_to_scalar failed");

        if let tch::IValue::Tensor(t) = ivalue {
            t
        } else {
            unreachable!("math_kernels support_to_scalar must return a Tensor")
        }
    }

    pub fn reward_support_to_scalar(&self, logits_prediction: &Tensor) -> Tensor {
        let ivalue = self
            .math_cmodule
            .method_is(
                "support_to_scalar",
                &[
                    tch::IValue::Tensor(logits_prediction.copy()),
                    tch::IValue::Int(self.reward_support_size),
                    tch::IValue::Double(self.epsilon_factor),
                ],
            )
            .expect("math_cmodule support_to_scalar failed");

        if let tch::IValue::Tensor(t) = ivalue {
            t
        } else {
            unreachable!("math_kernels support_to_scalar must return a Tensor")
        }
    }

    pub fn scalar_to_value_support(&self, scalar_prediction: &Tensor) -> Tensor {
        let ivalue = self
            .math_cmodule
            .method_is(
                "scalar_to_support",
                &[
                    tch::IValue::Tensor(scalar_prediction.copy()),
                    tch::IValue::Int(self.value_support_size),
                    tch::IValue::Double(self.epsilon_factor),
                ],
            )
            .expect("math_cmodule scalar_to_support failed");

        if let tch::IValue::Tensor(t) = ivalue {
            t
        } else {
            unreachable!("math_kernels scalar_to_support must return a Tensor")
        }
    }

    pub fn scalar_to_reward_support(&self, scalar_prediction: &Tensor) -> Tensor {
        let ivalue = self
            .math_cmodule
            .method_is(
                "scalar_to_support",
                &[
                    tch::IValue::Tensor(scalar_prediction.copy()),
                    tch::IValue::Int(self.reward_support_size),
                    tch::IValue::Double(self.epsilon_factor),
                ],
            )
            .expect("math_cmodule scalar_to_support failed");

        if let tch::IValue::Tensor(t) = ivalue {
            t
        } else {
            unreachable!("math_kernels scalar_to_support must return a Tensor")
        }
    }

    pub fn extract_initial_features(
        &self,
        boards: &Tensor,
        avail: &Tensor,
        hist: &Tensor,
        acts: &Tensor,
        diff: &Tensor,
    ) -> Tensor {
        let batch_size = boards.size()[0] as i32;
        let mut out = Tensor::zeros(
            [
                batch_size as i64,
                crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
                8,
                16,
            ],
            (tch::Kind::Float, boards.device()),
        );

        if !boards.device().is_cuda() {
            let mut out_data = vec![
                0.0f32;
                (batch_size * crate::core::features::NATIVE_FEATURE_CHANNELS as i32 * 128)
                    as usize
            ];
            let boards_data = Vec::<i64>::try_from(boards).unwrap_or_default();
            let avail_data = Vec::<i32>::try_from(avail).unwrap_or_default();
            let hist_data = Vec::<i64>::try_from(hist).unwrap_or_default();
            let acts_data = Vec::<i32>::try_from(acts).unwrap_or_default();
            let diff_data = Vec::<i32>::try_from(diff).unwrap_or_default();

            for b in 0..batch_size as usize {
                let board_low = boards_data.get(b * 2).copied().unwrap_or(0);
                let board_high = boards_data.get(b * 2 + 1).copied().unwrap_or(0);
                let board = (board_low as u64 as u128) | ((board_high as u64 as u128) << 64);

                let avail_arr = [
                    avail_data.get(b * 3).copied().unwrap_or(-1),
                    avail_data.get(b * 3 + 1).copied().unwrap_or(-1),
                    avail_data.get(b * 3 + 2).copied().unwrap_or(-1),
                ];

                let mut history = Vec::with_capacity(7);
                for i in (0..7).rev() {
                    let hl = hist_data.get(b * 14 + i * 2).copied().unwrap_or(0);
                    let hh = hist_data.get(b * 14 + i * 2 + 1).copied().unwrap_or(0);
                    history.push((hl as u64 as u128) | ((hh as u64 as u128) << 64));
                }

                let mut actions = Vec::with_capacity(3);
                for i in (0..3).rev() {
                    actions.push(acts_data.get(b * 3 + i).copied().unwrap_or(-1));
                }

                let diff_val = diff_data.get(b).copied().unwrap_or(0);

                let start_idx = b * crate::core::features::NATIVE_FEATURE_CHANNELS * 128;
                let end_idx = start_idx + crate::core::features::NATIVE_FEATURE_CHANNELS * 128;
                crate::core::features::extract_feature_native(
                    &mut out_data[start_idx..end_idx],
                    board,
                    &avail_arr,
                    &history,
                    &actions,
                    diff_val,
                );
            }
            out = Tensor::from_slice(&out_data)
                .view([
                    batch_size as i64,
                    crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
                    8,
                    16,
                ])
                .to_device(boards.device());
        } else {
            unsafe {
                let lib_paths = [
                    "tricked_ops.so",
                    "../tricked_ops.so",
                    "../../tricked_ops.so",
                    "../../../tricked_ops.so",
                    "./scripts/tricked_ops.so",
                ];
                let mut handle = None;
                for path in lib_paths {
                    if let Ok(lib) = libloading::Library::new(path) {
                        handle = Some(lib);
                        break;
                    }
                }
                if let Some(lib) = handle {
                    if let Ok(func) = lib.get::<unsafe extern "C" fn(
                        *const i64,
                        *const i32,
                        *const i64,
                        *const i32,
                        *const i32,
                        *mut f32,
                        *const i32,
                        *const i64,
                        *const i64,
                        i32,
                        i32,
                    )>(b"launch_extract_features")
                    {
                        func(
                            boards.data_ptr() as *const i64,
                            avail.data_ptr() as *const i32,
                            hist.data_ptr() as *const i64,
                            acts.data_ptr() as *const i32,
                            diff.data_ptr() as *const i32,
                            out.data_ptr() as *mut f32,
                            self.canonical_tensor.data_ptr() as *const i32,
                            self.compact_tensor.data_ptr() as *const i64,
                            self.standard_tensor.data_ptr() as *const i64,
                            batch_size,
                            self.num_standard_pieces,
                        );
                    } else {
                        eprintln!(
                            "WARNING: Could not find launch_extract_features in tricked_ops.so"
                        );
                    }
                    std::mem::forget(lib);
                } else {
                    eprintln!(
                        "WARNING: Could not load tricked_ops.so for extract_initial_features"
                    );
                }
            }
        }

        if self.spatial_channel_count > crate::core::features::NATIVE_FEATURE_CHANNELS as i64 {
            let padding = Tensor::zeros(
                [
                    batch_size as i64,
                    self.spatial_channel_count
                        - crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
                    8,
                    16,
                ],
                (tch::Kind::Float, boards.device()),
            );
            out = Tensor::cat(&[&out, &padding], 1);
        }
        out
    }

    pub fn extract_unrolled_features(&self, boards: &Tensor, hist: &Tensor) -> Tensor {
        let batch_size = boards.size()[0] as i32;
        let unroll_steps = boards.size()[1] as i32;
        let mut out = Tensor::zeros(
            [
                batch_size as i64,
                unroll_steps as i64,
                crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
                8,
                16,
            ],
            (tch::Kind::Float, boards.device()),
        );

        if !boards.device().is_cuda() {
            let mut out_data = vec![
                0.0f32;
                (batch_size
                    * unroll_steps
                    * crate::core::features::NATIVE_FEATURE_CHANNELS as i32
                    * 128) as usize
            ];
            let boards_data = Vec::<i64>::try_from(boards).unwrap_or_default();
            let hist_data = Vec::<i64>::try_from(hist).unwrap_or_default();

            for b in 0..batch_size as usize {
                for u in 0..unroll_steps as usize {
                    let board_low = boards_data
                        .get((b * (unroll_steps as usize) + u) * 2)
                        .copied()
                        .unwrap_or(0);
                    let board_high = boards_data
                        .get((b * (unroll_steps as usize) + u) * 2 + 1)
                        .copied()
                        .unwrap_or(0);
                    let board = (board_low as u64 as u128) | ((board_high as u64 as u128) << 64);

                    let mut history = Vec::with_capacity(7);
                    for i in (0..7).rev() {
                        let hl = hist_data
                            .get((b * (unroll_steps as usize) + u) * 14 + i * 2)
                            .copied()
                            .unwrap_or(0);
                        let hh = hist_data
                            .get((b * (unroll_steps as usize) + u) * 14 + i * 2 + 1)
                            .copied()
                            .unwrap_or(0);
                        history.push((hl as u64 as u128) | ((hh as u64 as u128) << 64));
                    }

                    let start_idx = (b * (unroll_steps as usize) + u)
                        * crate::core::features::NATIVE_FEATURE_CHANNELS
                        * 128;
                    let out_slice = &mut out_data[start_idx
                        ..start_idx + crate::core::features::NATIVE_FEATURE_CHANNELS * 128];

                    let fill_channel = |out: &mut [f32], c_idx: usize, mut bits: u128| {
                        let offset = c_idx * 128;
                        bits &= (1_u128 << 96) - 1;
                        while bits != 0 {
                            let bit_index = bits.trailing_zeros() as usize;
                            out[offset + crate::core::features::get_spatial_idx(bit_index)] = 1.0;
                            bits &= bits - 1;
                        }
                    };

                    fill_channel(out_slice, 0, board);
                    for (i, &h) in history.iter().rev().enumerate() {
                        fill_channel(out_slice, i + 1, h);
                    }
                }
            }
            out = Tensor::from_slice(&out_data)
                .view([
                    batch_size as i64,
                    unroll_steps as i64,
                    crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
                    8,
                    16,
                ])
                .to_device(boards.device());
        } else {
            unsafe {
                let lib_paths = [
                    "tricked_ops.so",
                    "../tricked_ops.so",
                    "../../tricked_ops.so",
                    "../../../tricked_ops.so",
                    "./scripts/tricked_ops.so",
                ];
                let mut handle = None;
                for path in lib_paths {
                    if let Ok(lib) = libloading::Library::new(path) {
                        handle = Some(lib);
                        break;
                    }
                }
                if let Some(lib) = handle {
                    if let Ok(func) =
                        lib.get::<unsafe extern "C" fn(*const i64, *const i64, *mut f32, i32, i32)>(
                            b"launch_extract_unrolled_features",
                        )
                    {
                        func(
                            boards.data_ptr() as *const i64,
                            hist.data_ptr() as *const i64,
                            out.data_ptr() as *mut f32,
                            batch_size,
                            unroll_steps,
                        );
                    } else {
                        eprintln!("WARNING: Could not find function symbol in tricked_ops.so");
                    }
                    std::mem::forget(lib);
                } else {
                    eprintln!("WARNING: Could not load tricked_ops.so, extract_unrolled_features will return zeros");
                }
            }
        } // End of else block

        if self.spatial_channel_count > crate::core::features::NATIVE_FEATURE_CHANNELS as i64 {
            let padding = Tensor::zeros(
                [
                    batch_size as i64,
                    unroll_steps as i64,
                    self.spatial_channel_count
                        - crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
                    8,
                    16,
                ],
                (tch::Kind::Float, boards.device()),
            );
            out = Tensor::cat(&[&out, &padding], 2);
        }
        out
    }

    pub fn initial_inference(&self, batched_state: &Tensor) -> (Tensor, Tensor, Tensor, Tensor) {
        assert_eq!(
            batched_state.size().len(),
            4,
            "Initial inference batched_state must have 4 dimensions"
        );
        assert_eq!(
            batched_state.size()[1],
            self.spatial_channel_count,
            "Initial inference batched_state spatial channels mismatch"
        );
        let hidden_state = self.representation.forward(batched_state);
        let (value_logits, policy_logits, hidden_state_logits) =
            self.prediction.forward(&hidden_state);

        let predicted_value_scalar = self.value_support_to_scalar(&value_logits);
        let policy_probabilities = policy_logits.softmax(-1, Kind::Float);

        (
            hidden_state,
            predicted_value_scalar,
            policy_probabilities,
            hidden_state_logits,
        )
    }

    pub fn recurrent_inference(
        &self,
        hidden_state: &Tensor,
        batched_action: &Tensor,
        batched_piece_identifier: &Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        assert_eq!(
            batched_action.size()[0],
            batched_piece_identifier.size()[0],
            "Action and piece identifier batch sizes must match"
        );

        let (hidden_state_next, value_prefix_logits) =
            self.dynamics
                .forward(hidden_state, batched_action, batched_piece_identifier);
        let (value_logits, policy_logits, hidden_state_logits) =
            self.prediction.forward(&hidden_state_next);

        let value_prefix_scalar_prediction = self.reward_support_to_scalar(&value_prefix_logits);
        let value_scalar_prediction = self.value_support_to_scalar(&value_logits);
        let policy_probabilities = policy_logits.softmax(-1, Kind::Float);

        (
            hidden_state_next,
            value_prefix_scalar_prediction,
            value_scalar_prediction,
            policy_probabilities,
            hidden_state_logits,
        )
    }
}

#[cfg(test)]
mod tests {
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
}
