use tch::{nn, nn::Module, Tensor};

#[derive(Debug)]
pub struct FlattenedResNetBlock {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
}

impl FlattenedResNetBlock {
    pub fn new(vs: &nn::Path, d_model: i64, _grid_size: i64) -> Self {
        let config = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1 = nn::conv2d(&(vs / "conv1"), d_model, d_model, 3, config);
        let conv2 = nn::conv2d(&(vs / "conv2"), d_model, d_model, 3, config);

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
        let mut out = self.conv1.forward(x);
        out = self
            .norm1
            .forward(&out.permute([0, 2, 3, 1]))
            .permute([0, 3, 1, 2])
            .mish();
        out = self.conv2.forward(&out);
        out = self
            .norm2
            .forward(&out.permute([0, 2, 3, 1]))
            .permute([0, 3, 1, 2]);
        (residual + out).mish()
    }
}
