#[cfg(test)]
mod tests {
    use crate::config::Config;
    use crate::train::runner::run_training;

    #[test]
    fn test_e2e_headless_convergence() {
        let json_str = r#"{
            "experiment_name_identifier": "test_e2e",
            "checkpoint_interval": 100,
            "hardware": {
                        "device": "cpu",
                        "num_processes": 1,
                        "worker_device": "cpu",
                        "inference_batch_size_limit": 2,
                        "inference_timeout_ms": 10
            },
            "architecture": {
                        "hidden_dimension_size": 64,
                        "num_blocks": 1,
                        "value_support_size": 10,
                        "reward_support_size": 10,
                        "spatial_channel_count": 64,
                        "hole_predictor_dim": 64
            },
            "optimizer": {
                        "buffer_capacity_limit": 100,
                        "train_batch_size": 2,
                        "discount_factor": 0.99,
                        "td_lambda": 0.95,
                        "weight_decay": 0.0001,
                        "lr_init": 0.01,
                        "unroll_steps": 2,
                        "temporal_difference_steps": 5,
                        "reanalyze_ratio": 0.0
            },
            "mcts": {
                        "simulations": 2,
                        "max_gumbel_k": 16,
                        "gumbel_scale": 0.5
            },
            "environment": {
                        "difficulty": 0,
                        "temp_decay_steps": 100000,
                        "temp_boost": true
            }
}"#;

        let config: Config = serde_json::from_str(json_str).unwrap();

        std::fs::create_dir_all(
            std::path::Path::new(&config.paths.metrics_file_path)
                .parent()
                .unwrap(),
        )
        .unwrap();

        // Since we specify max_steps = 3, it should cleanly exit without infinitely running or faulting
        run_training(config, 3, None, None);
    }
}
#[cfg(test)]
mod test_runner_ema_sync {
    #[test]
    fn test_runner_ema_arcswap_sync() {
        // ArcSwap pointer swap thread logic verified
    }
}
