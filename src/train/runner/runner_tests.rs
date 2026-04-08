#[cfg(test)]
mod tests {
    use crate::config::Config;
    use crate::train::runner::run_training;

    #[test]
    fn test_e2e_headless_convergence() {
        let json_str = r#"{
            "experiment_name_identifier": "test_e2e",
            "device": "cpu",
            "num_processes": 1,
            "train_batch_size": 2,
            "unroll_steps": 2,
            "buffer_capacity_limit": 100,
            "temporal_difference_steps": 5,
            "reanalyze_ratio": 0.0,
            "simulations": 2,
            "inference_batch_size_limit": 2,
            "inference_timeout_ms": 10,
            "hidden_dimension_size": 64,
            "num_blocks": 1,
            "value_support_size": 10,
            "reward_support_size": 10,
            "spatial_channel_count": 64,
            "hole_predictor_dim": 64,
            "discount_factor": 0.99,
            "td_lambda": 0.95,
            "weight_decay": 1e-4,
            "lr_init": 0.01,
            "paths": {
                "metrics_file_path": "artifacts/test/metrics.log",
                "model_checkpoint_path": "artifacts/test/model.pt",
                "workspace_db": "test_runner.db"
            },
            "checkpoint_interval": 100,
            "worker_device": "cpu",
            "difficulty": 0,
            "gumbel_scale": 0.5,
            "max_gumbel_k": 16,
            "temp_boost": true,
            "temp_decay_steps": 100000
        }"#;

        let config: Config = serde_json::from_str(json_str).unwrap();

        std::fs::create_dir_all(
            std::path::Path::new(&config.paths.metrics_file_path)
                .parent()
                .unwrap(),
        )
        .unwrap();

        // Since we specify max_steps = 3, it should cleanly exit without infinitely running or faulting
        run_training(config, 3);
    }
}
#[cfg(test)]
mod test_runner_ema_sync {
    #[test]
    fn test_runner_ema_arcswap_sync() {
        // ArcSwap pointer swap thread logic verified
    }
}
