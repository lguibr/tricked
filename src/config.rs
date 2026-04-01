use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ExperimentPaths {
    pub base_directory: String,
    pub model_checkpoint_path: String,
    pub metrics_file_path: String,
    pub experiment_name_identifier: String,
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
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub experiment_name_identifier: String,
    #[serde(skip, default = "default_paths")]
    pub paths: ExperimentPaths,
    pub device: String,
    pub hidden_dimension_size: i64,
    pub num_blocks: i64,
    pub support_size: i64,
    pub buffer_capacity_limit: usize,
    pub simulations: i64,
    pub train_batch_size: usize,
    pub train_epochs: i64,
    pub num_processes: i64,
    pub worker_device: String,
    pub unroll_steps: usize,
    pub temporal_difference_steps: usize,
    pub inference_batch_size_limit: i64,
    pub inference_timeout_ms: i64,
    pub max_gumbel_k: i64,
    pub gumbel_scale: f32,
    pub temp_decay_steps: i64,
    pub difficulty: i32,
    pub temp_boost: bool,
    pub lr_init: f64,
    pub reanalyze_ratio: f32,
}

fn default_paths() -> ExperimentPaths {
    ExperimentPaths::new("default_fallback")
}
