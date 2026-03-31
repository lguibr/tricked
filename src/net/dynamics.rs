use crate::net::FlattenedResNetBlock;
use tch::{nn, nn::Module, Kind, Tensor};

#[derive(Debug)]
pub struct DynamicsNet {
    piece_emb: nn::Embedding,
    pos_emb: nn::Embedding,
    proj_in: nn::Conv2D,
    blocks: Vec<FlattenedResNetBlock>,
    scale_norm: nn::LayerNorm,
    reward_cond: nn::Conv2D,
    reward_fc1: nn::Linear,
    reward_norm: nn::LayerNorm,
    reward_fc2: nn::Linear,
}

impl DynamicsNet {
    pub fn new(
        variable_store: &nn::Path,
        model_dimension: i64,
        num_blocks: i64,
        support_size: i64,
    ) -> Self {
        let piece_emb = nn::embedding(
            &(variable_store / "piece_emb"),
            48,
            model_dimension,
            Default::default(),
        );
        let pos_emb = nn::embedding(
            &(variable_store / "pos_emb"),
            96,
            model_dimension,
            Default::default(),
        );
        let proj_config = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let proj_in = nn::conv2d(
            &(variable_store / "proj_in"),
            model_dimension * 2,
            model_dimension,
            3,
            proj_config,
        );

        let mut blocks = Vec::new();
        let blocks_vs = variable_store / "blocks";
        for i in 0..num_blocks {
            blocks.push(FlattenedResNetBlock::new(
                &(&blocks_vs / i),
                model_dimension,
                128,
            ));
        }
        let scale_norm = nn::layer_norm(
            &(variable_store / "scale_norm"),
            vec![model_dimension],
            Default::default(),
        );

        let conv2d_config = nn::ConvConfig::default();
        let reward_cond = nn::conv2d(
            &(variable_store / "reward_cond"),
            model_dimension * 2,
            model_dimension,
            1,
            conv2d_config,
        );

        let reward_fc1 = nn::linear(
            &(variable_store / "reward_fc1"),
            model_dimension,
            64,
            Default::default(),
        );
        let reward_norm = nn::layer_norm(
            &(variable_store / "reward_norm"),
            vec![64],
            Default::default(),
        );
        let reward_fc2 = nn::linear(
            &(variable_store / "reward_fc2"),
            64,
            2 * support_size + 1,
            Default::default(),
        );

        Self {
            piece_emb,
            pos_emb,
            proj_in,
            blocks,
            scale_norm,
            reward_cond,
            reward_fc1,
            reward_norm,
            reward_fc2,
        }
    }

    pub fn forward(
        &self,
        hidden_state: &Tensor,
        batched_action: &Tensor,
        batched_piece_identifier: &Tensor,
    ) -> (Tensor, Tensor) {
        assert_eq!(
            hidden_state.size().len(),
            4,
            "Dynamics forward requires a 4D hidden_state tensor"
        );
        assert_eq!(
            batched_action.size()[0],
            batched_piece_identifier.size()[0],
            "Action sizes must match"
        );

        let position_indices = batched_action.remainder(96);
        let action_embeddings = self.piece_emb.forward(batched_piece_identifier)
            + self.pos_emb.forward(&position_indices);

        let action_expanded = action_embeddings
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand([-1, -1, 8, 8], false);
        let concatenated_features = Tensor::cat(&[hidden_state, &action_expanded], 1);

        let reward_convolutions_mish = self.reward_cond.forward(&concatenated_features).mish();
        let hidden_state_average_pooled =
            reward_convolutions_mish.mean_dim(&[2i64, 3i64][..], false, Kind::Float);

        let reward_features = self
            .reward_norm
            .forward(&self.reward_fc1.forward(&hidden_state_average_pooled))
            .mish();
        let reward_logits = self.reward_fc2.forward(&reward_features);

        let mut hidden_state_next = self.proj_in.forward(&concatenated_features);
        for block in &self.blocks {
            hidden_state_next = block.forward(&hidden_state_next);
        }
        hidden_state_next = self
            .scale_norm
            .forward(&hidden_state_next.permute([0, 2, 3, 1]))
            .permute([0, 3, 1, 2]);

        let final_hidden_state_next = hidden_state_next;

        assert_eq!(
            final_hidden_state_next.size(),
            hidden_state.size(),
            "Hidden state dynamic dims drifted!"
        );

        (final_hidden_state_next, reward_logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device};

    #[test]
    fn test_dynamics_action_conditioning() {
        let variable_store = nn::VarStore::new(Device::Cpu);
        let dynamics_network = DynamicsNet::new(&variable_store.root(), 16, 1, 300);

        let batch_size = 2;
        let hidden_state = Tensor::zeros([batch_size, 16, 8, 8], (Kind::Float, Device::Cpu));
        let batched_action = Tensor::zeros([batch_size], (Kind::Int64, Device::Cpu));
        let batched_piece_identifier = Tensor::zeros([batch_size], (Kind::Int64, Device::Cpu));

        let (hidden_state_next, reward_logits) =
            dynamics_network.forward(&hidden_state, &batched_action, &batched_piece_identifier);

        assert_eq!(
            hidden_state_next.size(),
            vec![batch_size, 16, 8, 8],
            "Latent state dimensions incorrect after dynamics forward pass"
        );
        assert_eq!(
            reward_logits.size(),
            vec![batch_size, 601],
            "Reward logits boundaries do not match 2*support + 1"
        );
    }
}
