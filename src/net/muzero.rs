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
            canonical_tensor,
            compact_tensor,
            standard_tensor,
            num_standard_pieces: STANDARD_PIECES.len() as i32,
        }
    }

    fn support_to_scalar_fused(&self, logits: &Tensor, support_size: i64, epsilon: f64) -> Tensor {
        let batch_size = logits.size()[0] as i32;
        let out = Tensor::zeros([batch_size as i64], (tch::Kind::Float, logits.device()));
        if !logits.device().is_cuda() {
            let probs = logits.softmax(-1, tch::Kind::Float);
            let support = Tensor::arange(2 * support_size + 1, (tch::Kind::Float, logits.device()))
                - support_size as f64;
            let expected_value =
                (probs * support).sum_dim_intlist(Some(&[-1][..]), false, tch::Kind::Float);
            let sgn = expected_value.sign();
            let abs_x = expected_value.abs();

            let eps = epsilon;
            let term1 = ((&abs_x + (1.0 + eps)) * (4.0 * eps) + 1.0).sqrt() - 1.0;
            let term2 = term1 / (2.0 * eps);
            let inv = sgn * (term2.pow_tensor_scalar(2.0) - 1.0);
            return inv;
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
                    if let Ok(func) = lib
                        .get::<unsafe extern "C" fn(*const f32, *mut f32, i32, i32, f32)>(
                            b"launch_support_to_scalar",
                        )
                    {
                        func(
                            logits.data_ptr() as *const f32,
                            out.data_ptr() as *mut f32,
                            batch_size,
                            support_size as i32,
                            epsilon as f32,
                        );
                    } else {
                        eprintln!(
                            "WARNING: Could not find launch_support_to_scalar in tricked_ops.so"
                        );
                    }
                    std::mem::forget(lib);
                }
            }
        }
        out
    }

    fn scalar_to_support_fused(&self, scalar: &Tensor, support_size: i64, epsilon: f64) -> Tensor {
        let batch_size = scalar.size()[0] as i32;
        let mut out = Tensor::zeros(
            [batch_size as i64, 2 * support_size + 1],
            (tch::Kind::Float, scalar.device()),
        );
        if !scalar.device().is_cuda() {
            let safe_scalar = scalar.nan_to_num(0.0, 0.0, 0.0);
            let transformed = safe_scalar.sign() * ((safe_scalar.abs() + 1.0).sqrt() - 1.0)
                + safe_scalar.copy() * epsilon;
            let clamped = transformed
                .reshape([-1])
                .clamp(-support_size as f64, support_size as f64);
            let shifted = clamped + support_size as f64;
            let floor_val = shifted.floor();
            let ceil_val = shifted.ceil();

            let upper_prob = shifted.copy() - floor_val.copy();
            let lower_prob = 1.0 - upper_prob.copy();

            let lower_idx = floor_val.to_kind(tch::Kind::Int64);
            let upper_idx = ceil_val.to_kind(tch::Kind::Int64);

            let batch_indices =
                Tensor::arange(batch_size as i64, (tch::Kind::Int64, scalar.device()));

            let _ = out.index_put_(
                &[Some(batch_indices.copy()), Some(lower_idx)],
                &lower_prob,
                true,
            );
            let _ = out.index_put_(
                &[Some(batch_indices.copy()), Some(upper_idx)],
                &upper_prob,
                true,
            );
            return out;
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
                    if let Ok(func) = lib
                        .get::<unsafe extern "C" fn(*const f32, *mut f32, i32, i32, f32)>(
                            b"launch_scalar_to_support",
                        )
                    {
                        func(
                            scalar.data_ptr() as *const f32,
                            out.data_ptr() as *mut f32,
                            batch_size,
                            support_size as i32,
                            epsilon as f32,
                        );
                    } else {
                        eprintln!(
                            "WARNING: Could not find launch_scalar_to_support in tricked_ops.so"
                        );
                    }
                    std::mem::forget(lib);
                }
            }
        }
        out
    }

    pub fn value_support_to_scalar(&self, logits_prediction: &Tensor) -> Tensor {
        self.support_to_scalar_fused(
            logits_prediction,
            self.value_support_size,
            self.epsilon_factor,
        )
    }

    pub fn reward_support_to_scalar(&self, logits_prediction: &Tensor) -> Tensor {
        self.support_to_scalar_fused(
            logits_prediction,
            self.reward_support_size,
            self.epsilon_factor,
        )
    }

    pub fn scalar_to_value_support(&self, scalar_prediction: &Tensor) -> Tensor {
        self.scalar_to_support_fused(
            scalar_prediction,
            self.value_support_size,
            self.epsilon_factor,
        )
    }

    pub fn scalar_to_reward_support(&self, scalar_prediction: &Tensor) -> Tensor {
        self.scalar_to_support_fused(
            scalar_prediction,
            self.reward_support_size,
            self.epsilon_factor,
        )
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
        let policy_probabilities = policy_logits
            .softmax(-1, Kind::Float)
            .clamp(1e-4_f64, 1.0_f64)
            .to_kind(Kind::Half);

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
        let policy_probabilities = policy_logits
            .softmax(-1, Kind::Float)
            .clamp(1e-4_f64, 1.0_f64)
            .to_kind(Kind::Half);

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
mod muzero_tests;
