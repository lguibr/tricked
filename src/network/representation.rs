use crate::network::FlattenedResNetBlock;
use tch::{nn, nn::Module, Tensor};

#[derive(Debug)]
pub struct RepresentationNet {
    proj_in: nn::Linear,
    blocks: Vec<FlattenedResNetBlock>,
    scale_norm: nn::LayerNorm,
}

impl RepresentationNet {
    pub fn new(vs: &nn::Path, d_model: i64, num_blocks: i64) -> Self {
        let proj_in = nn::linear(&(vs / "proj_in"), 20, d_model, Default::default());
        let mut blocks = Vec::new();
        let blk_vs = vs / "blocks";
        for i in 0..num_blocks {
            blocks.push(FlattenedResNetBlock::new(&(&blk_vs / i), d_model, 96));
        }
        let scale_norm = nn::layer_norm(&(vs / "scale_norm"), vec![d_model], Default::default());
        Self {
            proj_in,
            blocks,
            scale_norm,
        }
    }
}

impl Module for RepresentationNet {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = self.proj_in.forward(&x.transpose(1, 2));
        for block in &self.blocks {
            h = block.forward(&h);
        }
        self.scale_norm.forward(&h).transpose(1, 2)
    }
}
