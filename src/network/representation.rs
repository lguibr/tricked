use crate::network::FlattenedResNetBlock;
use tch::{nn, nn::Module, Tensor};

#[derive(Debug)]
pub struct RepresentationNet {
    proj_in: nn::Conv2D,
    blocks: Vec<FlattenedResNetBlock>,
    scale_norm: nn::LayerNorm,
}

impl RepresentationNet {
    pub fn new(vs: &nn::Path, d_model: i64, num_blocks: i64) -> Self {
        let config = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let proj_in = nn::conv2d(&(vs / "proj_in"), 40, d_model, 3, config);
        let mut blocks = Vec::new();
        let blk_vs = vs / "blocks";
        for i in 0..num_blocks {
            blocks.push(FlattenedResNetBlock::new(&(&blk_vs / i), d_model, 128));
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
        let batch = x.size()[0];
        let x_reshaped = x
            .view([batch, 20, 8, 8, 2])
            .permute([0, 1, 4, 2, 3])
            .reshape([batch, 40, 8, 8]);
        let mut h = self.proj_in.forward(&x_reshaped);
        for block in &self.blocks {
            h = block.forward(&h);
        }
        self.scale_norm
            .forward(&h.permute([0, 2, 3, 1]))
            .permute([0, 3, 1, 2])
    }
}
