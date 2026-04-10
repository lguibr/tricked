use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
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

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct HardwareConfig {
    pub device: String,
    pub num_processes: i64,
    pub worker_device: String,
    pub inference_batch_size_limit: i64,
    pub inference_timeout_ms: i64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ArchitectureConfig {
    pub hidden_dimension_size: i64,
    pub num_blocks: i64,
    pub value_support_size: i64,
    pub reward_support_size: i64,
    pub spatial_channel_count: i64,
    pub hole_predictor_dim: i64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
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
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct MctsConfig {
    pub simulations: i64,
    pub max_gumbel_k: i64,
    pub gumbel_scale: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct EnvironmentConfig {
    pub difficulty: i32,
    pub temp_decay_steps: i64,
    pub temp_boost: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
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
    
    #[serde(skip, default = "default_paths")]
    pub paths: ExperimentPaths,
}

fn default_paths() -> ExperimentPaths {
    ExperimentPaths::new("default_fallback")
}
