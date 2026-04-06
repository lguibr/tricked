use crate::net::{DynamicsNet, PredictionNet, ProjectorNet, RepresentationNet};
use tch::{nn, nn::Module, Kind, Tensor};

#[derive(Debug)]
pub struct MuZeroNet {
    pub representation: RepresentationNet,
    pub dynamics: DynamicsNet,
    pub prediction: PredictionNet,
    pub projector: ProjectorNet,
    pub value_support_size: i64,
    pub reward_support_size: i64,
    pub epsilon_factor: f64,
    pub math_cmodule: tch::CModule,
}

unsafe impl Sync for MuZeroNet {}
unsafe impl Send for MuZeroNet {}

impl MuZeroNet {
    pub fn new(
        variable_store: &nn::Path,
        model_dimension: i64,
        convolution_blocks: i64,
        value_support_size: i64,
        reward_support_size: i64,
        spatial_channel_count: i64,
        hole_predictor_dim: i64,
    ) -> Self {
        let representation = RepresentationNet::new(
            &(variable_store / "representation"),
            model_dimension,
            convolution_blocks,
            spatial_channel_count,
        );
        let dynamics = DynamicsNet::new(
            &(variable_store / "dynamics"),
            model_dimension,
            convolution_blocks,
            reward_support_size,
        );
        let prediction = PredictionNet::new(
            &(variable_store / "prediction"),
            model_dimension,
            value_support_size,
            288,
            hole_predictor_dim,
        );
        let projector =
            ProjectorNet::new(&(variable_store / "projector"), model_dimension, 512, 128);

        let mut math_data = std::io::Cursor::new(include_bytes!("../../assets/math_kernels.pt"));
        let math_cmodule =
            tch::CModule::load_data_on_device(&mut math_data, variable_store.device())
                .expect("Failed to load embedded math_kernels.pt");

        Self {
            representation,
            dynamics,
            prediction,
            projector,
            value_support_size,
            reward_support_size,
            epsilon_factor: 0.001,
            math_cmodule,
        }
    }

    pub fn value_support_to_scalar(&self, logits_prediction: &Tensor) -> Tensor {
        let ivalue = self
            .math_cmodule
            .method_is(
                "support_to_scalar",
                &[
                    tch::IValue::Tensor(logits_prediction.copy()),
                    tch::IValue::Int(self.value_support_size),
                    tch::IValue::Double(self.epsilon_factor),
                ],
            )
            .expect("math_cmodule support_to_scalar failed");

        if let tch::IValue::Tensor(t) = ivalue {
            t
        } else {
            unreachable!("math_kernels support_to_scalar must return a Tensor")
        }
    }

    pub fn reward_support_to_scalar(&self, logits_prediction: &Tensor) -> Tensor {
        let ivalue = self
            .math_cmodule
            .method_is(
                "support_to_scalar",
                &[
                    tch::IValue::Tensor(logits_prediction.copy()),
                    tch::IValue::Int(self.reward_support_size),
                    tch::IValue::Double(self.epsilon_factor),
                ],
            )
            .expect("math_cmodule support_to_scalar failed");

        if let tch::IValue::Tensor(t) = ivalue {
            t
        } else {
            unreachable!("math_kernels support_to_scalar must return a Tensor")
        }
    }

    pub fn scalar_to_value_support(&self, scalar_prediction: &Tensor) -> Tensor {
        let ivalue = self
            .math_cmodule
            .method_is(
                "scalar_to_support",
                &[
                    tch::IValue::Tensor(scalar_prediction.copy()),
                    tch::IValue::Int(self.value_support_size),
                    tch::IValue::Double(self.epsilon_factor),
                ],
            )
            .expect("math_cmodule scalar_to_support failed");

        if let tch::IValue::Tensor(t) = ivalue {
            t
        } else {
            unreachable!("math_kernels scalar_to_support must return a Tensor")
        }
    }

    pub fn scalar_to_reward_support(&self, scalar_prediction: &Tensor) -> Tensor {
        let ivalue = self
            .math_cmodule
            .method_is(
                "scalar_to_support",
                &[
                    tch::IValue::Tensor(scalar_prediction.copy()),
                    tch::IValue::Int(self.reward_support_size),
                    tch::IValue::Double(self.epsilon_factor),
                ],
            )
            .expect("math_cmodule scalar_to_support failed");

        if let tch::IValue::Tensor(t) = ivalue {
            t
        } else {
            unreachable!("math_kernels scalar_to_support must return a Tensor")
        }
    }

    pub fn extract_unrolled_features(&self, boards: &Tensor, hist: &Tensor) -> Tensor {
        let batch_size = boards.size()[0] as i32;
        let unroll_steps = boards.size()[1] as i32;
        let out = Tensor::zeros(
            [batch_size as i64, unroll_steps as i64, 20, 8, 16],
            (tch::Kind::Float, boards.device()),
        );

        unsafe {
            let lib_paths = [
                "tricked_ops.so",
                "../tricked_ops.so",
                "./scripts/tricked_ops.so",
            ];
            let mut handle = None;
            for path in lib_paths {
                if let Ok(lib) = libloading::Library::new(path) {
                    handle = Some(lib);
                    break;
                }
            }
            if let Some(lib) = handle {
                if let Ok(func) =
                    lib.get::<unsafe extern "C" fn(*const i64, *const i64, *mut f32, i32, i32)>(
                        b"launch_extract_unrolled_features",
                    )
                {
                    func(
                        boards.data_ptr() as *const i64,
                        hist.data_ptr() as *const i64,
                        out.data_ptr() as *mut f32,
                        batch_size,
                        unroll_steps,
                    );
                } else {
                    eprintln!("WARNING: Could not find function symbol in tricked_ops.so");
                }
                std::mem::forget(lib);
            } else {
                eprintln!("WARNING: Could not load tricked_ops.so, extract_unrolled_features will return zeros");
            }
        }
        out
    }

    pub fn initial_inference(&self, batched_state: &Tensor) -> (Tensor, Tensor, Tensor, Tensor) {
        assert_eq!(
            batched_state.size().len(),
            4,
            "Initial inference batched_state must have 4 dimensions"
        );
        assert_eq!(
            batched_state.size()[1],
            20,
            "Initial inference batched_state must have 20 spatial channels"
        );
        let hidden_state = self.representation.forward(batched_state);
        let (value_logits, policy_logits, hidden_state_logits) =
            self.prediction.forward(&hidden_state);

        let predicted_value_scalar = self.value_support_to_scalar(&value_logits);
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

        let (hidden_state_next, value_prefix_logits) =
            self.dynamics
                .forward(hidden_state, batched_action, batched_piece_identifier);
        let (value_logits, policy_logits, hidden_state_logits) =
            self.prediction.forward(&hidden_state_next);

        let value_prefix_scalar_prediction = self.reward_support_to_scalar(&value_prefix_logits);
        let value_scalar_prediction = self.value_support_to_scalar(&value_logits);
        let policy_probabilities = policy_logits.softmax(-1, Kind::Float);

        (
            hidden_state_next,
            value_prefix_scalar_prediction,
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
        let neural_engine = MuZeroNet::new(&variable_store.root(), 256, 4, 300, 300, 20, 64);

        let batch_size = 2;
        let batched_state = Tensor::zeros([batch_size, 20, 8, 16], (Kind::Float, Device::Cpu));

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

        let batched_action = Tensor::zeros([batch_size], (Kind::Int64, Device::Cpu));
        let batched_piece_identifier = Tensor::zeros([batch_size], (Kind::Int64, Device::Cpu));

        let (
            hidden_state_next,
            value_prefix_scalar,
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
            i64::try_from(value_prefix_scalar.isnan().any()).unwrap(),
            0,
            "NaN in recurrent value_prefix"
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

    #[test]
    fn test_support_vector_round_trip() {
        let variable_store = nn::VarStore::new(Device::Cpu);
        let neural_engine = MuZeroNet::new(&variable_store.root(), 256, 4, 300, 300, 20, 64);

        // Test strictly positive scalars as the domain was deliberately shifted to [0, +inf)
        let original_scalars = Tensor::from_slice(&[0.0_f32, 10.0, 0.5, 5.5, 299.9]);

        // 1. Scalar to Support (Probabilities)
        let support_probs = neural_engine.scalar_to_value_support(&original_scalars);

        // 2. Convert Probabilities to Logits to feed into support_to_scalar
        // Add a tiny epsilon to prevent log(0) -> -inf which breaks softmax math
        let logits = (support_probs + 1e-9).log();

        // 3. Support to Scalar
        let reconstructed_scalars = neural_engine.value_support_to_scalar(&logits);

        let diff = (&original_scalars - &reconstructed_scalars).abs();
        let max_diff: f32 = diff.max().try_into().unwrap_or(1.0);

        // Max diff should be small (accounting for 32-bit float / epsilons across sqrt/pow operations)
        assert!(
            max_diff < 0.1,
            "Support Vector Math round-trip failed! Max delta: {}",
            max_diff
        );
    }
}
