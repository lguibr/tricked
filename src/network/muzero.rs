use crate::network::{DynamicsNet, PredictionNet, ProjectorNet, RepresentationNet};
use tch::{nn, nn::Module, Kind, Tensor};

#[derive(Debug)]
pub struct MuZeroNet {
    pub representation: RepresentationNet,
    pub dynamics: DynamicsNet,
    pub prediction: PredictionNet,
    pub projector: ProjectorNet,
    pub support_size: i64,
    pub epsilon_factor: f64,
    pub support_vector: Tensor,
}

impl MuZeroNet {
    pub fn new(
        variable_store: &nn::Path,
        model_dimension: i64,
        convolution_blocks: i64,
        support_size: i64,
    ) -> Self {
        let representation = RepresentationNet::new(
            &(variable_store / "representation"),
            model_dimension,
            convolution_blocks,
        );
        let dynamics = DynamicsNet::new(
            &(variable_store / "dynamics"),
            model_dimension,
            convolution_blocks,
            support_size,
        );
        let prediction = PredictionNet::new(
            &(variable_store / "prediction"),
            model_dimension,
            support_size,
            288,
        );
        let projector =
            ProjectorNet::new(&(variable_store / "projector"), model_dimension, 512, 128);

        let support_vector = Tensor::arange_start_step(
            -support_size as f64,
            (support_size + 1) as f64,
            1.0,
            (Kind::Float, variable_store.device()),
        );

        Self {
            representation,
            dynamics,
            prediction,
            projector,
            support_size,
            epsilon_factor: 0.001,
            support_vector,
        }
    }

    pub fn support_to_scalar(&self, logits_prediction: &Tensor) -> Tensor {
        let float_logits = logits_prediction.to_kind(Kind::Float);
        let softmax_probabilities = float_logits.softmax(-1, Kind::Float);
        let symmetric_scalar = (&softmax_probabilities * &self.support_vector).sum_dim_intlist(
            &[-1i64][..],
            false,
            Kind::Float,
        );

        let epsilon = self.epsilon_factor;
        let absolute_symmetric = symmetric_scalar.abs();
        let clamped_absolute = absolute_symmetric.clamp(0.0, self.support_size as f64);

        let scaled_inversion =
            (((&clamped_absolute + (1.0 + epsilon)) * (4.0 * epsilon) + 1.0).sqrt() - 1.0)
                / (2.0 * epsilon);
        let final_value = scaled_inversion.pow_tensor_scalar(2.0) - 1.0;

        symmetric_scalar.sign() * final_value
    }

    pub fn scalar_to_support(&self, scalar_prediction: &Tensor) -> Tensor {
        let safe_scalar = scalar_prediction.nan_to_num(0.0, Some(0.0), Some(0.0));
        let symmetric_scalar = safe_scalar.sign() * ((safe_scalar.abs() + 1.0).sqrt() - 1.0)
            + self.epsilon_factor * &safe_scalar;

        let clamped_scalar = symmetric_scalar
            .reshape([-1])
            .clamp(-self.support_size as f64, self.support_size as f64);

        let mut support_probabilities = Tensor::zeros(
            [clamped_scalar.size()[0], 2 * self.support_size + 1],
            (Kind::Float, safe_scalar.device()),
        );

        let floor_value = clamped_scalar.floor();
        let ceiling_value = clamped_scalar.ceil();

        let upper_probability = &clamped_scalar - &floor_value;
        let lower_probability = -&upper_probability + 1.0;

        let lower_index = (&floor_value + self.support_size as f64).to_kind(Kind::Int64);
        let upper_index = (&ceiling_value + self.support_size as f64).to_kind(Kind::Int64);

        support_probabilities = support_probabilities.scatter_add(
            1,
            &lower_index.unsqueeze(1),
            &lower_probability.unsqueeze(1),
        );
        support_probabilities = support_probabilities.scatter_add(
            1,
            &upper_index.unsqueeze(1),
            &upper_probability.unsqueeze(1),
        );

        support_probabilities
    }

    pub fn initial_inference(&self, batched_state: &Tensor) -> (Tensor, Tensor, Tensor, Tensor) {
        assert_eq!(
            batched_state.size()[1],
            20,
            "Initial inference batched_state must have 20 spatial channels"
        );
        let hidden_state = self.representation.forward(batched_state);
        let (value_logits, policy_logits, hidden_state_logits) =
            self.prediction.forward(&hidden_state);

        let predicted_value_scalar = self.support_to_scalar(&value_logits);
        let policy_probabilities = policy_logits.softmax(-1, Kind::Float);

        (
            hidden_state,
            predicted_value_scalar,
            policy_probabilities,
            hidden_state_logits,
        )
    }

    pub fn recurrent_inference(
        &self,
        hidden_state: &Tensor,
        batched_action: &Tensor,
        batched_piece_identifier: &Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        assert_eq!(
            batched_action.size()[0],
            batched_piece_identifier.size()[0],
            "Action and piece identifier batch sizes must match"
        );

        let (hidden_state_next, reward_logits) =
            self.dynamics
                .forward(hidden_state, batched_action, batched_piece_identifier);
        let (value_logits, policy_logits, hidden_state_logits) =
            self.prediction.forward(&hidden_state_next);

        let reward_scalar_prediction = self.support_to_scalar(&reward_logits);
        let value_scalar_prediction = self.support_to_scalar(&value_logits);
        let policy_probabilities = policy_logits.softmax(-1, Kind::Float);

        (
            hidden_state_next,
            reward_scalar_prediction,
            value_scalar_prediction,
            policy_probabilities,
            hidden_state_logits,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device, Kind};

    #[test]
    fn test_muzero_nan_safety() {
        let variable_store = nn::VarStore::new(Device::Cpu);
        let neural_engine = MuZeroNet::new(&variable_store.root(), 256, 4, 300);

        let batch_size = 2;
        let batched_state = Tensor::zeros(&[batch_size, 20, 96], (Kind::Float, Device::Cpu));

        let (hidden_state, value_scalar, policy_probs, hidden_state_logits) =
            neural_engine.initial_inference(&batched_state);
        assert_eq!(
            i64::try_from(hidden_state.isnan().any()).unwrap(),
            0,
            "NaN in representation"
        );
        assert_eq!(
            i64::try_from(value_scalar.isnan().any()).unwrap(),
            0,
            "NaN in initial value"
        );
        assert_eq!(
            i64::try_from(policy_probs.isnan().any()).unwrap(),
            0,
            "NaN in initial policy"
        );
        assert_eq!(
            i64::try_from(hidden_state_logits.isnan().any()).unwrap(),
            0,
            "NaN in hole logits"
        );

        let batched_action = Tensor::zeros(&[batch_size], (Kind::Int64, Device::Cpu));
        let batched_piece_identifier = Tensor::zeros(&[batch_size], (Kind::Int64, Device::Cpu));

        let (
            hidden_state_next,
            reward_scalar,
            value_scalar_next,
            policy_probs_next,
            hidden_state_logits_next,
        ) = neural_engine.recurrent_inference(
            &hidden_state,
            &batched_action,
            &batched_piece_identifier,
        );

        assert_eq!(
            i64::try_from(hidden_state_next.isnan().any()).unwrap(),
            0,
            "NaN in recurrent hidden_state"
        );
        assert_eq!(
            i64::try_from(reward_scalar.isnan().any()).unwrap(),
            0,
            "NaN in recurrent reward"
        );
        assert_eq!(
            i64::try_from(value_scalar_next.isnan().any()).unwrap(),
            0,
            "NaN in recurrent value"
        );
        assert_eq!(
            i64::try_from(policy_probs_next.isnan().any()).unwrap(),
            0,
            "NaN in recurrent policy"
        );
        assert_eq!(
            i64::try_from(hidden_state_logits_next.isnan().any()).unwrap(),
            0,
            "NaN in recurrent logits"
        );
    }
}
