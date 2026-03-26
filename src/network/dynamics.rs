use tch::{nn, nn::Module, Kind, Tensor};
use crate::network::FlattenedResNetBlock;

#[derive(Debug)]
pub struct DynamicsNet {
    piece_emb: nn::Embedding,
    pos_emb: nn::Embedding,
    proj_in: nn::Linear,
    blocks: Vec<FlattenedResNetBlock>,
    scale_norm: nn::LayerNorm,
    reward_cond: nn::Conv1D,
    reward_fc1: nn::Linear,
    reward_norm: nn::LayerNorm,
    reward_fc2: nn::Linear,
}

impl DynamicsNet {
    pub fn new(vs: &nn::Path, d_model: i64, num_blocks: i64, support_size: i64) -> Self {
        let piece_emb = nn::embedding(&(vs / "piece_emb"), 48, d_model, Default::default());
        let pos_emb = nn::embedding(&(vs / "pos_emb"), 96, d_model, Default::default());
        let proj_in = nn::linear(&(vs / "proj_in"), d_model * 2, d_model, Default::default());

        let mut blocks = Vec::new();
        let blk_vs = vs / "blocks";
        for i in 0..num_blocks {
            blocks.push(FlattenedResNetBlock::new(&(blk_vs / i), d_model, 96));
        }
        let scale_norm = nn::layer_norm(&(vs / "scale_norm"), vec![d_model], Default::default());

        let mut conv1d_cfg = nn::ConvConfig::default();
        conv1d_cfg.kernel_size = 1;
        let reward_cond = nn::conv1d(&(vs / "reward_cond"), d_model * 2, d_model, 1, conv1d_cfg);

        let reward_fc1 = nn::linear(&(vs / "reward_fc1"), d_model, 64, Default::default());
        let reward_norm = nn::layer_norm(&(vs / "reward_norm"), vec![64], Default::default());
        let reward_fc2 = nn::linear(
            (&(vs / "reward_fc2")),
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

    pub fn forward(&self, h: &Tensor, a: &Tensor, piece_id: &Tensor) -> (Tensor, Tensor) {
        let pos_idx = a.remainder(96);
        let a_emb = self.piece_emb.forward(piece_id) + self.pos_emb.forward(&pos_idx);

        let a_expanded = a_emb.unsqueeze(-1).expand(&[-1, -1, h.size()[2]], false);
        let x = Tensor::cat(&[h, &a_expanded], 1);

        let r_conv = self.reward_cond.forward(&x).mish();
        let h_t_pooled = r_conv.mean_dim(&[2], false, Kind::Float);

        let r = self.reward_norm
            .forward(&self.reward_fc1.forward(&h_t_pooled))
            .mish();
        let reward_logits = self.reward_fc2.forward(&r);

        let mut h_next = self.proj_in.forward(&x.transpose(1, 2));
        for block in &self.blocks {
            h_next = block.forward(&h_next);
        }
        h_next = self.scale_norm.forward(&h_next);
        (h_next.transpose(1, 2), reward_logits)
    }
}
