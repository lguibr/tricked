use tch::{nn, nn::Module, Tensor};
use crate::network::GraphConv1d;

#[derive(Debug)]
pub struct FlattenedResNetBlock {
    conv1: GraphConv1d,
    conv2: GraphConv1d,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
}

impl FlattenedResNetBlock {
    pub fn new(vs: &nn::Path, d_model: i64, grid_size: i64) -> Self {
        let conv1 = GraphConv1d::new(&(vs / "conv1"), d_model, d_model, grid_size);
        let conv2 = GraphConv1d::new(&(vs / "conv2"), d_model, d_model, grid_size);
        let norm1 = nn::layer_norm(&(vs / "norm1"), vec![d_model], Default::default());
        let norm2 = nn::layer_norm(&(vs / "norm2"), vec![d_model], Default::default());
        Self {
            conv1,
            conv2,
            norm1,
            norm2,
        }
    }
}

impl Module for FlattenedResNetBlock {
    fn forward(&self, x: &Tensor) -> Tensor {
        let residual = x;
        let mut out = x.transpose(1, 2);
        out = self.conv1.forward(&out).transpose(1, 2);
        out = self.norm1.forward(&out).mish();
        out = out.transpose(1, 2);
        out = self.conv2.forward(&out).transpose(1, 2);
        out = self.norm2.forward(&out);
        residual + out
    }
}
