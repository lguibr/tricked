use crate::core::constants::STANDARD_PIECES;
use crate::core::features::CANONICAL_PIECE_MASKS;
use crate::net::{DynamicsNet, PredictionNet, ProjectorNet, RepresentationNet};
use crate::node::COMPACT_PIECE_MASKS;
use tch::{nn, Tensor};

use crate::net::muzero::features::{extract_initial_features, extract_unrolled_features};
use crate::net::muzero::support::{scalar_to_support_fused, support_to_scalar_fused};
use crate::net::muzero::inference::{initial_inference, recurrent_inference};

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

    pub fn value_support_to_scalar(&self, logits_prediction: &Tensor) -> Tensor {
        support_to_scalar_fused(
            logits_prediction,
            self.value_support_size,
            self.epsilon_factor,
        )
    }

    pub fn reward_support_to_scalar(&self, logits_prediction: &Tensor) -> Tensor {
        support_to_scalar_fused(
            logits_prediction,
            self.reward_support_size,
            self.epsilon_factor,
        )
    }

    pub fn scalar_to_value_support(&self, scalar_prediction: &Tensor) -> Tensor {
        scalar_to_support_fused(
            scalar_prediction,
            self.value_support_size,
            self.epsilon_factor,
        )
    }

    pub fn scalar_to_reward_support(&self, scalar_prediction: &Tensor) -> Tensor {
        scalar_to_support_fused(
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
        extract_initial_features(
            self.spatial_channel_count,
            &self.canonical_tensor,
            &self.compact_tensor,
            &self.standard_tensor,
            self.num_standard_pieces,
            boards,
            avail,
            hist,
            acts,
            diff,
        )
    }

    pub fn extract_unrolled_features(&self, boards: &Tensor, hist: &Tensor) -> Tensor {
        extract_unrolled_features(self.spatial_channel_count, boards, hist)
    }

    pub fn initial_inference(&self, batched_state: &Tensor) -> (Tensor, Tensor, Tensor, Tensor) {
        initial_inference(self, batched_state)
    }

    pub fn recurrent_inference(
        &self,
        hidden_state: &Tensor,
        batched_action: &Tensor,
        batched_piece_identifier: &Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        recurrent_inference(self, hidden_state, batched_action, batched_piece_identifier)
    }
}
