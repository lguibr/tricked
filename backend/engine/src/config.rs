pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/tricked.rs"));
}
use prost::Message;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExperimentPaths {
    pub base_directory: String,
    pub model_checkpoint_path: String,
    pub metrics_file_path: String,
    pub experiment_name_identifier: String,
    pub workspace_db: Option<String>,
}

impl ExperimentPaths {
    pub fn new(experiment_name: &str) -> Self {
        let base_directory = format!("runs/{}", experiment_name);
        Self {
            model_checkpoint_path: format!(
                "{}/{}_weights.safetensors",
                base_directory, experiment_name
            ),
            metrics_file_path: format!("{}/{}_metrics.csv", base_directory, experiment_name),
            base_directory: base_directory.clone(),
            experiment_name_identifier: experiment_name.to_string(),
            workspace_db: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HardwareConfig {
    pub device: String,
    pub num_processes: i64,
    pub worker_device: String,
    pub inference_batch_size_limit: i64,
    pub inference_timeout_ms: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    pub hidden_dimension_size: i64,
    pub num_blocks: i64,
    pub value_support_size: i64,
    pub reward_support_size: i64,
    pub spatial_channel_count: i64,
    pub hole_predictor_dim: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub buffer_capacity_limit: usize,
    pub train_batch_size: usize,
    pub discount_factor: f32,
    pub td_lambda: f32,
    pub weight_decay: f64,
    pub lr_init: f64,
    pub unroll_steps: usize,
    pub temporal_difference_steps: usize,
    pub reanalyze_ratio: f32,
    pub max_steps: i32,
    pub prioritized_replay_alpha: f64,
    pub prioritized_replay_beta: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MctsConfig {
    pub simulations: i64,
    pub max_gumbel_k: i64,
    pub gumbel_scale: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    pub difficulty: i32,
    pub temp_decay_steps: i64,
    pub temp_boost: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub experiment_name_identifier: String,
    pub checkpoint_interval: usize,
    #[serde(default)]
    pub hardware: HardwareConfig,
    #[serde(default)]
    pub architecture: ArchitectureConfig,
    #[serde(default)]
    pub optimizer: OptimizerConfig,
    #[serde(default)]
    pub mcts: MctsConfig,
    #[serde(default)]
    pub environment: EnvironmentConfig,
    #[serde(skip)]
    pub paths: ExperimentPaths,
}

impl Config {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, prost::DecodeError> {
        let proto = pb::TrickedConfig::decode(bytes)?;
        let mut hw = proto.hardware.unwrap_or_default();
        if hw.inference_batch_size_limit == 0 {
            hw.inference_batch_size_limit = 64;
        }
        if hw.inference_timeout_ms == 0 {
            hw.inference_timeout_ms = 5;
        }
        let arch = proto.architecture.unwrap_or_default();
        let opt = proto.optimizer.unwrap_or_default();
        let mc = proto.mcts.unwrap_or_default();
        let env = proto.environment.unwrap_or_default();

        Ok(Self {
            paths: ExperimentPaths::new(&proto.experiment_name_identifier),
            experiment_name_identifier: proto.experiment_name_identifier,
            checkpoint_interval: proto.checkpoint_interval as usize,
            hardware: HardwareConfig {
                device: hw.device,
                num_processes: hw.num_processes as i64,
                worker_device: hw.worker_device,
                inference_batch_size_limit: hw.inference_batch_size_limit as i64,
                inference_timeout_ms: hw.inference_timeout_ms as i64,
            },
            architecture: ArchitectureConfig {
                hidden_dimension_size: arch.hidden_dimension_size as i64,
                num_blocks: arch.num_blocks as i64,
                value_support_size: arch.value_support_size as i64,
                reward_support_size: arch.reward_support_size as i64,
                spatial_channel_count: arch.spatial_channel_count as i64,
                hole_predictor_dim: arch.hole_predictor_dim as i64,
            },
            optimizer: OptimizerConfig {
                buffer_capacity_limit: opt.buffer_capacity_limit as usize,
                train_batch_size: opt.train_batch_size as usize,
                discount_factor: opt.discount_factor,
                td_lambda: opt.td_lambda,
                weight_decay: opt.weight_decay as f64,
                lr_init: opt.lr_init as f64,
                unroll_steps: opt.unroll_steps as usize,
                temporal_difference_steps: opt.temporal_difference_steps as usize,
                reanalyze_ratio: opt.reanalyze_ratio,
                max_steps: opt.max_steps,
                prioritized_replay_alpha: opt.prioritized_replay_alpha.unwrap_or(0.6) as f64,
                prioritized_replay_beta: opt.prioritized_replay_beta.unwrap_or(0.4) as f64,
            },
            mcts: MctsConfig {
                simulations: mc.simulations as i64,
                max_gumbel_k: mc.max_gumbel_k as i64,
                gumbel_scale: mc.gumbel_scale,
            },
            environment: EnvironmentConfig {
                difficulty: env.difficulty,
                temp_decay_steps: env.temp_decay_steps as i64,
                temp_boost: env.temp_boost,
            },
        })
    }
}
