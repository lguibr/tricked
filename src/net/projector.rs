use tch::{nn, nn::Module, Kind, Tensor};

#[derive(Debug)]
pub struct ProjectorNet {
    proj: nn::Linear,
    norm1: nn::LayerNorm,
    fully_connected_layer_1: nn::Linear,
    norm2: nn::LayerNorm,
    fc2: nn::Linear,
}

impl ProjectorNet {
    pub fn new(variable_store: &nn::Path, hidden_dimension_size: i64, proj_dim: i64, out_dim: i64) -> Self {
        let proj = nn::linear(
            &(variable_store / "proj"),
            hidden_dimension_size,
            hidden_dimension_size / 2,
            Default::default(),
        );
        let norm1 = nn::layer_norm(
            &(variable_store / "norm1"),
            vec![hidden_dimension_size / 2],
            Default::default(),
        );
        let fully_connected_layer_1 = nn::linear(
            &(variable_store / "fully_connected_layer_1"),
            hidden_dimension_size / 2,
            proj_dim,
            Default::default(),
        );
        let norm2 = nn::layer_norm(&(variable_store / "norm2"), vec![proj_dim], Default::default());
        let fc2 = nn::linear(&(variable_store / "fc2"), proj_dim, out_dim, Default::default());
        Self {
            proj,
            norm1,
            fully_connected_layer_1,
            norm2,
            fc2,
        }
    }

    pub fn forward(&self, hidden_state_tensor: &Tensor) -> Tensor {
        let intermediate_features = self
            .norm1
            .forward(&self.proj.forward(&hidden_state_tensor.permute([0, 2, 3, 1])))
            .mish()
            .mean_dim(&[1i64, 2i64][..], false, Kind::Float);
        let intermediate_features = self.norm2.forward(&self.fully_connected_layer_1.forward(&intermediate_features)).mish();
        self.fc2.forward(&intermediate_features)
    }
}
