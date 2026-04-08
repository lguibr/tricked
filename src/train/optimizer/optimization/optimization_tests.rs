use super::*;
use tch::{nn, nn::OptimizerConfig, Device};

#[test]
fn test_train_step_bptt_and_masking() {
    let variable_store = nn::VarStore::new(Device::Cpu);
    let neural_model = MuZeroNet::new(
        &variable_store.root(),
        16,
        1,
        200,
        200,
        crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
        64,
    );
    let ema_model = MuZeroNet::new(
        &variable_store.root(),
        16,
        1,
        200,
        200,
        crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
        64,
    );
    let mut gradient_optimizer = nn::Adam::default().build(&variable_store, 1e-3).unwrap();

    let configuration = crate::config::Config {
        experiment_name_identifier: "test_exp".to_string(),
        device: "cpu".into(),
        paths: crate::config::ExperimentPaths::default(),
        hidden_dimension_size: 16,
        num_blocks: 1,
        value_support_size: 200,
        reward_support_size: 200,
        spatial_channel_count: crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
        hole_predictor_dim: 64,
        buffer_capacity_limit: 100,
        simulations: 10,
        train_batch_size: 2,
        discount_factor: 0.99,
        td_lambda: 0.95,
        weight_decay: 0.0,
        checkpoint_interval: 100,
        num_processes: 1,
        worker_device: "cpu".into(),
        unroll_steps: 2,
        temporal_difference_steps: 5,
        inference_batch_size_limit: 1,
        inference_timeout_ms: 1,
        max_gumbel_k: 4,
        gumbel_scale: 1.0,
        temp_decay_steps: 10,
        difficulty: 6,
        temp_boost: false,
        lr_init: 1e-3,
        reanalyze_ratio: 0.25,
    };

    let replay_buffer = ReplayBuffer::new(100, 2, 8, 32, None, 0.99, 0.95);

    let steps = vec![
        crate::train::buffer::GameStep {
            board_state: [0u64, 0u64],
            available_pieces: [0i32, 0, 0],
            action_taken: 0,
            piece_identifier: 0,
            value_prefix_received: 0.1,
            policy_target: {
                let mut p = vec![0.0; 288];
                p[0] = 1.0;
                p
            },
            value_target: 0.5,
        };
        15
    ];

    replay_buffer.add_game(crate::train::buffer::OwnedGameData {
        difficulty_setting: 6,
        episode_score: 1.0,
        steps: steps.clone(),
        lines_cleared: 0,
        mcts_depth_mean: 0.0,
        mcts_search_time_mean: 0.0,
    });
    replay_buffer.add_game(crate::train::buffer::OwnedGameData {
        difficulty_setting: 6,
        episode_score: 1.0,
        steps,
        lines_cleared: 0,
        mcts_depth_mean: 0.0,
        mcts_search_time_mean: 0.0,
    });

    let mut batched_experience_tensors_opt = None;
    for _ in 0..50 {
        if let Some(batch) = replay_buffer.sample_batch(2, 1.0) {
            batched_experience_tensors_opt = Some(batch);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let batched_experience_tensors = batched_experience_tensors_opt.unwrap();

    let _metrics = train_step(
        &neural_model,
        &ema_model,
        &mut gradient_optimizer,
        &replay_buffer,
        &batched_experience_tensors,
        configuration.unroll_steps,
        &variable_store,
    );
}

#[test]
fn test_train_step_batched_state_padding_regression() {
    let variable_store = nn::VarStore::new(Device::Cpu);
    // Force spatial channel count to be > Native Features, as in 64 config
    let expanded_channels = 64;
    let neural_model = MuZeroNet::new(
        &variable_store.root(),
        16,
        1,
        200,
        200,
        expanded_channels,
        64,
    );
    let ema_model = MuZeroNet::new(
        &variable_store.root(),
        16,
        1,
        200,
        200,
        expanded_channels,
        64,
    );
    let mut gradient_optimizer = nn::Adam::default().build(&variable_store, 1e-3).unwrap();

    let configuration = crate::config::Config {
        experiment_name_identifier: "test_exp".to_string(),
        device: "cpu".into(),
        paths: crate::config::ExperimentPaths::default(),
        hidden_dimension_size: 16,
        num_blocks: 1,
        value_support_size: 200,
        reward_support_size: 200,
        spatial_channel_count: expanded_channels,
        hole_predictor_dim: 64,
        buffer_capacity_limit: 100,
        simulations: 10,
        train_batch_size: 2,
        discount_factor: 0.99,
        td_lambda: 0.95,
        weight_decay: 0.0,
        checkpoint_interval: 100,
        num_processes: 1,
        worker_device: "cpu".into(),
        unroll_steps: 2,
        temporal_difference_steps: 5,
        inference_batch_size_limit: 1,
        inference_timeout_ms: 1,
        max_gumbel_k: 4,
        gumbel_scale: 1.0,
        temp_decay_steps: 10,
        difficulty: 6,
        temp_boost: false,
        lr_init: 1e-3,
        reanalyze_ratio: 0.25,
    };

    let replay_buffer = ReplayBuffer::new(100, 2, 8, 32, None, 0.99, 0.95);

    let steps = vec![
        crate::train::buffer::GameStep {
            board_state: [0u64, 0u64],
            available_pieces: [0i32, 0, 0],
            action_taken: 0,
            piece_identifier: 0,
            value_prefix_received: 0.1,
            policy_target: {
                let mut p = vec![0.0; 288];
                p[0] = 1.0;
                p
            },
            value_target: 0.5,
        };
        15
    ];

    replay_buffer.add_game(crate::train::buffer::OwnedGameData {
        difficulty_setting: 6,
        episode_score: 1.0,
        steps: steps.clone(),
        lines_cleared: 0,
        mcts_depth_mean: 0.0,
        mcts_search_time_mean: 0.0,
    });
    replay_buffer.add_game(crate::train::buffer::OwnedGameData {
        difficulty_setting: 6,
        episode_score: 1.0,
        steps,
        lines_cleared: 0,
        mcts_depth_mean: 0.0,
        mcts_search_time_mean: 0.0,
    });

    let mut batched_experience_tensors_opt = None;
    for _ in 0..50 {
        if let Some(batch) = replay_buffer.sample_batch(2, 1.0) {
            batched_experience_tensors_opt = Some(batch);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let batched_experience_tensors = batched_experience_tensors_opt.unwrap();

    // Should not panic due to dimension mismatch between 20-channel features and 64-channel network expectation
    let _metrics = train_step(
        &neural_model,
        &ema_model,
        &mut gradient_optimizer,
        &replay_buffer,
        &batched_experience_tensors,
        configuration.unroll_steps,
        &variable_store,
    );
}
