use tch::{nn, nn::Module, Kind, Tensor};

#[derive(Debug)]
pub struct ProjectorNet {
    proj: nn::Linear,
    norm1: nn::LayerNorm,
    fc1: nn::Linear,
    norm2: nn::LayerNorm,
    fc2: nn::Linear,
}

impl ProjectorNet {
    pub fn new(vs: &nn::Path, d_model: i64, proj_dim: i64, out_dim: i64) -> Self {
        let proj = nn::linear(&(vs / "proj"), d_model, d_model / 2, Default::default());
        let norm1 = nn::layer_norm(&(vs / "norm1"), vec![d_model / 2], Default::default());
        let fc1 = nn::linear(&(vs / "fc1"), d_model / 2, proj_dim, Default::default());
        let norm2 = nn::layer_norm(&(vs / "norm2"), vec![proj_dim], Default::default());
        let fc2 = nn::linear(&(vs / "fc2"), proj_dim, out_dim, Default::default());
        Self {
            proj,
            norm1,
            fc1,
            norm2,
            fc2,
        }
    }

    pub fn forward(&self, h: &Tensor) -> Tensor {
        let x = self
            .norm1
            .forward(&self.proj.forward(&h.permute([0, 2, 3, 1])))
            .mish()
            .mean_dim(&[1i64, 2i64][..], false, Kind::Float);
        let x = self.norm2.forward(&self.fc1.forward(&x)).mish();
        self.fc2.forward(&x)
    }
}
