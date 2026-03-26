use tch::{nn, nn::Module, Kind, Tensor};

#[derive(Debug)]
struct HolePredictor {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl HolePredictor {
    fn new(vs: &nn::Path, d_model: i64) -> Self {
        Self {
            fc1: nn::linear(&(vs / "0"), d_model, 64, Default::default()),
            fc2: nn::linear(&(vs / "2"), 64, 1, Default::default()),
        }
    }
}

impl Module for HolePredictor {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.fc2.forward(&self.fc1.forward(x).mish())
    }
}

#[derive(Debug)]
pub struct PredictionNet {
    val_proj: nn::Linear,
    val_norm: nn::LayerNorm,
    value_fc1: nn::Linear,
    value_fc2: nn::Linear,

    pol_proj: nn::Linear,
    pol_norm: nn::LayerNorm,
    policy_fc1: nn::Linear,

    hole_predictor: HolePredictor,
}

impl PredictionNet {
    pub fn new(vs: &nn::Path, d_model: i64, support_size: i64, num_actions: i64) -> Self {
        let val_proj = nn::linear(&(vs / "val_proj"), d_model, d_model / 2, Default::default());
        let val_norm = nn::layer_norm(&(vs / "val_norm"), vec![d_model / 2], Default::default());
        let value_fc1 = nn::linear(&(vs / "value_fc1"), d_model / 2, 64, Default::default());
        let value_fc2 = nn::linear(
            &(vs / "value_fc2"),
            64,
            2 * support_size + 1,
            Default::default(),
        );

        let pol_proj = nn::linear(&(vs / "pol_proj"), d_model, d_model / 2, Default::default());
        let pol_norm = nn::layer_norm(&(vs / "pol_norm"), vec![d_model / 2], Default::default());
        let policy_fc1 = nn::linear(
            &(vs / "policy_fc1"),
            d_model / 2,
            num_actions,
            Default::default(),
        );

        let hole_predictor = HolePredictor::new(&(vs / "hole_predictor"), d_model);

        Self {
            val_proj,
            val_norm,
            value_fc1,
            value_fc2,
            pol_proj,
            pol_norm,
            policy_fc1,
            hole_predictor,
        }
    }

    pub fn forward(&self, h: &Tensor) -> (Tensor, Tensor, Tensor) {
        let x = h.transpose(1, 2);
        let v = self
            .val_norm
            .forward(&self.val_proj.forward(&x))
            .mish()
            .mean_dim(&[1], false, Kind::Float);
        let v = self.value_fc1.forward(&v).mish();
        let value_logits = self.value_fc2.forward(&v);

        let p = self
            .pol_norm
            .forward(&self.pol_proj.forward(&x))
            .mish()
            .mean_dim(&[1], false, Kind::Float);
        let policy_logits = self.policy_fc1.forward(&p);

        let hole_logits = self.hole_predictor.forward(&x).squeeze_dim(-1);

        (value_logits, policy_logits, hole_logits)
    }
}
