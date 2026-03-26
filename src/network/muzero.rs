use crate::network::{DynamicsNet, PredictionNet, ProjectorNet, RepresentationNet};
use tch::{nn, nn::Module, Kind, Tensor};

#[derive(Debug)]
pub struct MuZeroNet {
    pub representation: RepresentationNet,
    pub dynamics: DynamicsNet,
    pub prediction: PredictionNet,
    pub projector: ProjectorNet,
    pub support_size: i64,
    pub epsilon: f64,
    pub support_vector: Tensor,
}

impl MuZeroNet {
    pub fn new(vs: &nn::Path, d_model: i64, num_blocks: i64, support_size: i64) -> Self {
        let representation = RepresentationNet::new(&(vs / "representation"), d_model, num_blocks);
        let dynamics = DynamicsNet::new(&(vs / "dynamics"), d_model, num_blocks, support_size);
        let prediction = PredictionNet::new(&(vs / "prediction"), d_model, support_size, 288);
        let projector = ProjectorNet::new(&(vs / "projector"), d_model, 512, 128);

        let support_vector = Tensor::arange_start_step(
            -support_size as f64,
            (support_size + 1) as f64,
            1.0,
            (Kind::Float, vs.device()),
        );

        Self {
            representation,
            dynamics,
            prediction,
            projector,
            support_size,
            epsilon: 0.001,
            support_vector,
        }
    }

    pub fn support_to_scalar(&self, logits: &Tensor) -> Tensor {
        let logits = logits.to_kind(Kind::Float);
        let probs = logits.softmax(-1, Kind::Float);
        let sym_scalar =
            (&probs * &self.support_vector).sum_dim_intlist(&[-1i64][..], false, Kind::Float);

        let epsilon = self.epsilon;
        let y = sym_scalar.abs();
        let y = y.clamp(0.0, self.support_size as f64);

        let z = (((&y + (1.0 + epsilon)) * (4.0 * epsilon) + 1.0).sqrt() - 1.0) / (2.0 * epsilon);
        let x = z.pow_tensor_scalar(2.0) - 1.0;

        sym_scalar.sign() * x
    }

    pub fn scalar_to_support(&self, scalar: &Tensor) -> Tensor {
        let scalar = scalar.nan_to_num(0.0, Some(0.0), Some(0.0));
        let sym_scalar =
            scalar.sign() * ((scalar.abs() + 1.0).sqrt() - 1.0) + self.epsilon * &scalar;
        let sym_scalar = sym_scalar
            .reshape([-1])
            .clamp(-self.support_size as f64, self.support_size as f64);

        let mut probabilities = Tensor::zeros(
            [sym_scalar.size()[0], 2 * self.support_size + 1],
            (Kind::Float, scalar.device()),
        );

        let lower = sym_scalar.floor();
        let upper = sym_scalar.ceil();

        let p_upper = &sym_scalar - &lower;
        let p_lower = -&p_upper + 1.0;

        let lower_idx = (&lower + self.support_size as f64).to_kind(Kind::Int64);
        let upper_idx = (&upper + self.support_size as f64).to_kind(Kind::Int64);

        probabilities =
            probabilities.scatter_add(1, &lower_idx.unsqueeze(1), &p_lower.unsqueeze(1));
        probabilities =
            probabilities.scatter_add(1, &upper_idx.unsqueeze(1), &p_upper.unsqueeze(1));

        probabilities
    }

    pub fn initial_inference(&self, s: &Tensor) -> (Tensor, Tensor, Tensor, Tensor) {
        let h = self.representation.forward(s);
        let (value_logits, policy_logits, hole_logits) = self.prediction.forward(&h);
        let value_scalar = self.support_to_scalar(&value_logits);
        let policy_probs = policy_logits.softmax(-1, Kind::Float);
        (h, value_scalar, policy_probs, hole_logits)
    }

    pub fn recurrent_inference(
        &self,
        h: &Tensor,
        a: &Tensor,
        piece_id: &Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let (h_next, reward_logits) = self.dynamics.forward(h, a, piece_id);
        let (value_logits, policy_logits, hole_logits) = self.prediction.forward(&h_next);
        let reward_scalar = self.support_to_scalar(&reward_logits);
        let value_scalar = self.support_to_scalar(&value_logits);
        let policy_probs = policy_logits.softmax(-1, Kind::Float);
        (
            h_next,
            reward_scalar,
            value_scalar,
            policy_probs,
            hole_logits,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device, Kind};

    #[test]
    fn test_muzero_nan_safety() {
        let vs = nn::VarStore::new(Device::Cpu);
        let net = MuZeroNet::new(&vs.root(), 256, 4, 300);

        let batch_size = 2;
        let s = Tensor::zeros(&[batch_size, 20, 96], (Kind::Float, Device::Cpu));

        let (h, value_scalar, policy_probs, hole_logits) = net.initial_inference(&s);
        assert_eq!(
            i64::try_from(h.isnan().any()).unwrap(),
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
            i64::try_from(hole_logits.isnan().any()).unwrap(),
            0,
            "NaN in hole logits"
        );

        let a = Tensor::zeros(&[batch_size], (Kind::Int64, Device::Cpu));
        let piece_id = Tensor::zeros(&[batch_size], (Kind::Int64, Device::Cpu));

        let (h_next, reward_scalar, value_scalar_r, policy_probs_r, hole_logits_r) =
            net.recurrent_inference(&h, &a, &piece_id);

        assert_eq!(
            i64::try_from(h_next.isnan().any()).unwrap(),
            0,
            "NaN in dynamics"
        );
        assert_eq!(
            i64::try_from(reward_scalar.isnan().any()).unwrap(),
            0,
            "NaN in recurrent reward"
        );
        assert_eq!(
            i64::try_from(value_scalar_r.isnan().any()).unwrap(),
            0,
            "NaN in recurrent value"
        );
        assert_eq!(
            i64::try_from(policy_probs_r.isnan().any()).unwrap(),
            0,
            "NaN in recurrent policy"
        );
        assert_eq!(
            i64::try_from(hole_logits_r.isnan().any()).unwrap(),
            0,
            "NaN in recurrent holes"
        );
    }
}
