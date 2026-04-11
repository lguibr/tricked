use tricked_engine::config::Config;

pub fn get_test_config() -> Config {
    serde_yaml::from_str(
        r#"
device: cpu
checkpoint_interval: 100
paths:
  base_directory: "runs/test"
  model_checkpoint_path: "test.safetensors"
  metrics_file_path: "test.csv"
  telemetry_config_export: "config.json"
experiment_name_identifier: "test"
hardware:
  device: cpu
  num_processes: 1
  worker_device: cpu
  inference_batch_size_limit: 1
  inference_timeout_ms: 1
architecture:
  hidden_dimension_size: 128
  num_blocks: 8
  value_support_size: 300
  reward_support_size: 300
  spatial_channel_count: 20
  hole_predictor_dim: 64
optimizer:
  buffer_capacity_limit: 1000
  train_batch_size: 4
  discount_factor: 0.99
  td_lambda: 0.95
  weight_decay: 0.0
  lr_init: 0.01
  unroll_steps: 3
  temporal_difference_steps: 5
  reanalyze_ratio: 0.25
mcts:
  simulations: 10
  max_gumbel_k: 4
  gumbel_scale: 1.0
environment:
  difficulty: 6
  temp_decay_steps: 100
  temp_boost: false
        "#,
    )
    .unwrap()
}
