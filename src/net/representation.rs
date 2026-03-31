use crate::net::FlattenedResNetBlock;
use tch::{nn, nn::Module, Tensor};

#[derive(Debug)]
pub struct RepresentationNet {
    proj_in: nn::Conv2D,
    blocks: Vec<FlattenedResNetBlock>,
    scale_norm: nn::LayerNorm,
}

impl RepresentationNet {
    pub fn new(vs: &nn::Path, hidden_dimension_size: i64, num_blocks: i64) -> Self {
        let config = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let proj_in = nn::conv2d(&(vs / "proj_in"), 40, hidden_dimension_size, 3, config);
        let mut blocks = Vec::new();
        let blk_vs = vs / "blocks";
        for i in 0..num_blocks {
            blocks.push(FlattenedResNetBlock::new(
                &(&blk_vs / i),
                hidden_dimension_size,
                128,
            ));
        }
        let scale_norm = nn::layer_norm(
            &(vs / "scale_norm"),
            vec![hidden_dimension_size],
            Default::default(),
        );
        Self {
            proj_in,
            blocks,
            scale_norm,
        }
    }
}

impl Module for RepresentationNet {
    fn forward(&self, input_tensor_batch_channel_height_width: &Tensor) -> Tensor {
        let input_shape = input_tensor_batch_channel_height_width.size();
        assert_eq!(
            input_shape.len(),
            4,
            "RepresentationNet requires [Batch, 20, 8, 16] input"
        );
        let batch = input_tensor_batch_channel_height_width.size()[0];
        let x_reshaped = input_tensor_batch_channel_height_width
            .view([batch, 20, 8, 8, 2])
            .permute([0, 1, 4, 2, 3])
            .reshape([batch, 40, 8, 8]);
        let mut h = self.proj_in.forward(&x_reshaped);
        for block in &self.blocks {
            h = block.forward(&h);
        }
        self.scale_norm
            .forward(&h.permute([0, 2, 3, 1]).contiguous())
            .permute([0, 3, 1, 2])
            .contiguous()
    }
}
