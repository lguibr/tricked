use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub device: String,
    pub model_checkpoint: String,
    pub metrics_file: String,
    pub d_model: i64,
    pub num_blocks: i64,
    pub support_size: i64,
    pub capacity: usize,
    pub num_games: i64,
    pub simulations: i64,
    pub train_batch_size: usize,
    pub train_epochs: i64,
    pub num_processes: i64,
    pub worker_device: String,
    pub unroll_steps: usize,
    pub td_steps: usize,
    pub zmq_inference_port: String,
    pub zmq_batch_size: i64,
    pub zmq_timeout_ms: i64,
    pub max_gumbel_k: i64,
    pub gumbel_scale: f32,
    pub temp_decay_steps: i64,
    pub difficulty: i32,
    pub exploit_starts: Vec<i32>,
    pub temp_boost: bool,
    pub exp_name: String,
    pub lr_init: f64,
}

impl Config {
    pub fn load_yaml(path: &str) -> Self {
        let content = std::fs::read_to_string(path).expect("Could not read config.yaml");
        serde_yaml::from_str(&content).expect("Could not parse config.yaml")
    }
}
