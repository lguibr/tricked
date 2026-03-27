use tch::{nn, nn::Module, Kind, Tensor};

#[derive(Debug)]
struct HolePredictor {
    feature_layer_1: nn::Linear,
    feature_layer_2: nn::Linear,
}

impl HolePredictor {
    fn new(variable_store: &nn::Path, model_dimension: i64) -> Self {
        Self {
            feature_layer_1: nn::linear(
                &(variable_store / "0"),
                model_dimension,
                64,
                Default::default(),
            ),
            feature_layer_2: nn::linear(&(variable_store / "2"), 64, 1, Default::default()),
        }
    }
}

impl Module for HolePredictor {
    fn forward(&self, network_input: &Tensor) -> Tensor {
        self.feature_layer_2
            .forward(&self.feature_layer_1.forward(network_input).mish())
    }
}

#[derive(Debug)]
pub struct PredictionNet {
    value_projection: nn::Linear,
    value_normalization: nn::LayerNorm,
    value_layer_1: nn::Linear,
    value_layer_2: nn::Linear,

    policy_projection: nn::Linear,
    policy_normalization: nn::LayerNorm,
    policy_layer_1: nn::Linear,

    hole_predictor: HolePredictor,
}

impl PredictionNet {
    pub fn new(
        variable_store: &nn::Path,
        model_dimension: i64,
        support_size: i64,
        action_count: i64,
    ) -> Self {
        let value_projection = nn::linear(
            &(variable_store / "val_proj"),
            model_dimension,
            model_dimension / 2,
            Default::default(),
        );
        let value_normalization = nn::layer_norm(
            &(variable_store / "val_norm"),
            vec![model_dimension / 2],
            Default::default(),
        );
        let value_layer_1 = nn::linear(
            &(variable_store / "value_fc1"),
            model_dimension / 2,
            64,
            Default::default(),
        );
        let value_layer_2 = nn::linear(
            &(variable_store / "value_fc2"),
            64,
            2 * support_size + 1,
            Default::default(),
        );

        let policy_projection = nn::linear(
            &(variable_store / "pol_proj"),
            model_dimension,
            model_dimension / 2,
            Default::default(),
        );
        let policy_normalization = nn::layer_norm(
            &(variable_store / "pol_norm"),
            vec![model_dimension / 2],
            Default::default(),
        );
        let policy_layer_1 = nn::linear(
            &(variable_store / "policy_fc1"),
            model_dimension / 2,
            action_count,
            Default::default(),
        );

        let hole_predictor =
            HolePredictor::new(&(variable_store / "hole_predictor"), model_dimension);

        Self {
            value_projection,
            value_normalization,
            value_layer_1,
            value_layer_2,
            policy_projection,
            policy_normalization,
            policy_layer_1,
            hole_predictor,
        }
    }

    pub fn forward(&self, hidden_state: &Tensor) -> (Tensor, Tensor, Tensor) {
        assert_eq!(
            hidden_state.size().len(),
            3,
            "Prediction forward requires a 3D hidden_state tensor"
        );

        let transposed_hidden_state = hidden_state.transpose(1, 2);

        let value_features_mish = self
            .value_normalization
            .forward(&self.value_projection.forward(&transposed_hidden_state))
            .mish()
            .mean_dim(&[1i64][..], false, Kind::Float);

        let value_intermediate = self.value_layer_1.forward(&value_features_mish).mish();
        let value_logits = self.value_layer_2.forward(&value_intermediate);

        let policy_features_mish = self
            .policy_normalization
            .forward(&self.policy_projection.forward(&transposed_hidden_state))
            .mish()
            .mean_dim(&[1i64][..], false, Kind::Float);

        let policy_logits = self.policy_layer_1.forward(&policy_features_mish);

        let hole_logits = self
            .hole_predictor
            .forward(&transposed_hidden_state)
            .squeeze_dim(-1);

        (value_logits, policy_logits, hole_logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device, Kind};

    #[test]
    fn test_prediction_output_shapes() {
        let variable_store = nn::VarStore::new(Device::Cpu);
        let prediction_network = PredictionNet::new(&variable_store.root(), 16, 300, 288);

        let batch_size = 2;
        let hidden_state = Tensor::zeros([batch_size, 16, 96], (Kind::Float, Device::Cpu));

        let (value_logits, policy_logits, hole_logits) = prediction_network.forward(&hidden_state);

        assert_eq!(
            value_logits.size(),
            vec![batch_size, 601],
            "Value support boundaries do not match"
        );
        assert_eq!(
            policy_logits.size(),
            vec![batch_size, 288],
            "Action count boundaries do not match"
        );
        assert_eq!(
            hole_logits.size(),
            vec![batch_size, 96],
            "Spatial hole masking size does not match"
        );
    }
}
