use std::sync::{Arc, RwLock};
use std::thread;
use tch::{nn, nn::OptimizerConfig, Device};

use crate::config::Config;
use crate::env::reanalyze;
use crate::env::worker as selfplay;
use crate::net::MuZeroNet;
use crate::queue;
use crate::train::buffer::ReplayBuffer;
use crate::train::optimizer as trainer;

#[hotpath::measure]
pub fn run_training(config: Config, max_steps: usize) {
    println!(
        "🚀 Starting Tricked AI Native Engine (CLI Mode) for experiment: {}",
        config.experiment_name_identifier
    );

    let configuration_arc = Arc::new(config);

    assert!(configuration_arc.buffer_capacity_limit > configuration_arc.train_batch_size);
    assert!(configuration_arc.temporal_difference_steps > 0);
    assert!(configuration_arc.num_processes > 0);

    let shared_replay_buffer = Arc::new(ReplayBuffer::new(
        configuration_arc.buffer_capacity_limit,
        configuration_arc.unroll_steps,
        configuration_arc.temporal_difference_steps,
        configuration_arc.train_batch_size,
    ));

    let computation_device = if configuration_arc.device == "cuda" && tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    let mut training_var_store = nn::VarStore::new(computation_device);
    let mut inference_var_store = nn::VarStore::new(computation_device);
    let exponential_moving_average_var_store = nn::VarStore::new(computation_device);

    let training_network = MuZeroNet::new(
        &training_var_store.root(),
        configuration_arc.hidden_dimension_size,
        configuration_arc.num_blocks,
        configuration_arc.support_size,
    );
    let ema_network = MuZeroNet::new(
        &exponential_moving_average_var_store.root(),
        configuration_arc.hidden_dimension_size,
        configuration_arc.num_blocks,
        configuration_arc.support_size,
    );

    let mut inference_var_store_b = nn::VarStore::new(computation_device);
    let inference_net_a = Arc::new(MuZeroNet::new(
        &inference_var_store.root(),
        configuration_arc.hidden_dimension_size,
        configuration_arc.num_blocks,
        configuration_arc.support_size,
    ));
    let inference_net_b = Arc::new(MuZeroNet::new(
        &inference_var_store_b.root(),
        configuration_arc.hidden_dimension_size,
        configuration_arc.num_blocks,
        configuration_arc.support_size,
    ));

    let active_inference_net = Arc::new(arc_swap::ArcSwap::from(Arc::clone(&inference_net_a)));
    let mut cmodule_inference: Option<Arc<tch::CModule>> = None;

    if !configuration_arc.paths.model_checkpoint_path.is_empty() {
        if std::path::Path::new(&configuration_arc.paths.model_checkpoint_path).exists() {
            if configuration_arc
                .paths
                .model_checkpoint_path
                .ends_with(".pt")
            {
                println!(
                    "🚀 Loading TorchScript model: {}",
                    configuration_arc.paths.model_checkpoint_path
                );
                cmodule_inference = tch::CModule::load_on_device(
                    &configuration_arc.paths.model_checkpoint_path,
                    computation_device,
                )
                .ok()
                .map(Arc::new);
            } else {
                println!(
                    "🚀 Loading Native Rust weights: {}",
                    configuration_arc.paths.model_checkpoint_path
                );
                let _ = training_var_store.load(&configuration_arc.paths.model_checkpoint_path);
                inference_var_store.copy(&training_var_store).unwrap();
                inference_var_store_b.copy(&training_var_store).unwrap();
            }
        } else {
            println!(
                "⚠️ Checkpoint '{}' not found. Init weights.",
                configuration_arc.paths.model_checkpoint_path
            );
            inference_var_store.copy(&training_var_store).unwrap();
            inference_var_store_b.copy(&training_var_store).unwrap();
            std::fs::create_dir_all(
                std::path::Path::new(&configuration_arc.paths.model_checkpoint_path)
                    .parent()
                    .unwrap(),
            )
            .unwrap();
            let _ = training_var_store.save(&configuration_arc.paths.model_checkpoint_path);
        }
    } else {
        inference_var_store.copy(&training_var_store).unwrap();
        inference_var_store_b.copy(&training_var_store).unwrap();
    }

    tch::no_grad(|| {
        for (name, tensor) in training_var_store.variables().iter() {
            assert!(
                i64::try_from(tensor.isnan().any()).unwrap() == 0,
                "NaN detected in weights '{name}'"
            );
        }
    });

    let active_training_flag = Arc::new(RwLock::new(true));

    let workspace_db_path = configuration_arc
        .paths
        .workspace_db
        .clone()
        .unwrap_or_else(|| "tricked_workspace.db".to_string());

    let telemetry_logger = crate::telemetry::TelemetryLogger::new(workspace_db_path);

    let stdin_active_flag = Arc::clone(&active_training_flag);
    thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut buffer = String::new();
        while *stdin_active_flag.read().unwrap() {
            buffer.clear();
            if let Ok(bytes) = stdin.read_line(&mut buffer) {
                if bytes == 0 {
                    break;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&buffer) {
                    if v.get("cmd").and_then(|c| c.as_str()) == Some("stop") {
                        if let Ok(mut flag) = stdin_active_flag.write() {
                            *flag = false;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(1500));
                        std::process::exit(0);
                    }
                }
            }
        }
    });

    let config_path = std::path::Path::new(&configuration_arc.paths.metrics_file_path)
        .parent()
        .unwrap()
        .join("config.json");
    let config_json = serde_json::to_string_pretty(&*configuration_arc).unwrap();
    std::fs::write(config_path, config_json).unwrap();

    let reanalyze_worker_count = std::cmp::max(1, configuration_arc.num_processes / 4);
    let total_workers = configuration_arc.num_processes + reanalyze_worker_count;
    let inference_queue = Arc::new(queue::FixedInferenceQueue::new(
        configuration_arc.buffer_capacity_limit,
        total_workers as usize,
    ));

    for _ in 0..1 {
        let thread_evaluation_receiver = Arc::clone(&inference_queue);
        let thread_network_mutex = Arc::clone(&active_inference_net);
        let thread_cmodule = cmodule_inference.clone();
        let thread_active_flag = Arc::clone(&active_training_flag);
        let configuration_model_dimension = configuration_arc.hidden_dimension_size;
        // The GC actively frees nodes every step. We only need bound guarantees for a single search step.
        let max_nodes = (configuration_arc.simulations as usize) * 2 + 1000;
        let inference_batch_size_limit = configuration_arc.inference_batch_size_limit as usize;
        let inference_timeout_milliseconds = configuration_arc.inference_timeout_ms as u64;

        thread::spawn(move || {
            while *thread_active_flag.read().unwrap() {
                selfplay::inference_loop(selfplay::InferenceLoopParams {
                    receiver_queue: Arc::clone(&thread_evaluation_receiver),
                    shared_neural_model: Arc::clone(&thread_network_mutex),
                    cmodule_inference: thread_cmodule.clone(),
                    model_dimension: configuration_model_dimension,
                    computation_device,
                    total_workers: total_workers as usize,
                    maximum_allowed_nodes_in_search_tree: max_nodes,
                    inference_batch_size_limit,
                    inference_timeout_milliseconds,
                    active_flag: Arc::clone(&thread_active_flag),
                });
            }
        });
    }

    let selfplay_worker_count = configuration_arc.num_processes;
    for worker_id in 0..selfplay_worker_count {
        let thread_configuration = Arc::clone(&configuration_arc);
        let thread_evaluation_sender = Arc::clone(&inference_queue);
        let thread_replay_buffer = Arc::clone(&shared_replay_buffer);
        let thread_active_flag = Arc::clone(&active_training_flag);

        thread::spawn(move || {
            while *thread_active_flag.read().unwrap() {
                selfplay::game_loop(selfplay::GameLoopExecutionParameters {
                    configuration: Arc::clone(&thread_configuration),
                    evaluation_transmitter: Arc::clone(&thread_evaluation_sender),
                    experience_buffer: Arc::clone(&thread_replay_buffer),
                    worker_id: worker_id as usize,
                    active_flag: Arc::clone(&thread_active_flag),
                });
            }
        });
    }

    for worker_index in 0..reanalyze_worker_count {
        let worker_id = selfplay_worker_count + worker_index;
        let thread_configuration = Arc::clone(&configuration_arc);
        let thread_evaluation_sender = Arc::clone(&inference_queue);
        let thread_replay_buffer = Arc::clone(&shared_replay_buffer);
        let thread_active_flag = Arc::clone(&active_training_flag);

        thread::spawn(move || {
            while *thread_active_flag.read().unwrap() {
                reanalyze::reanalyze_worker_loop(
                    Arc::clone(&thread_configuration),
                    Arc::clone(&thread_evaluation_sender),
                    Arc::clone(&thread_replay_buffer),
                    worker_id as usize,
                    Arc::clone(&thread_active_flag),
                );
            }
        });
    }

    let prefetch_replay_buffer = Arc::clone(&shared_replay_buffer);
    let prefetch_active_flag = Arc::clone(&active_training_flag);
    let (prefetch_tx, prefetch_rx) = crossbeam_channel::bounded(4);
    let prefetch_device = computation_device;
    let prefetch_batch_size = configuration_arc.train_batch_size;

    let unroll_steps = configuration_arc.unroll_steps;
    let prefetch_max_steps = max_steps as f64;

    thread::spawn(move || {
        const BUFFER_COUNT: usize = 8;
        let mut pinned_arenas: Vec<_> = (0..BUFFER_COUNT)
            .map(|_| {
                crate::train::arena::PinnedBatchTensors::new(
                    prefetch_batch_size,
                    unroll_steps,
                    prefetch_device,
                )
            })
            .collect();
        let mut gpu_arenas: Vec<_> = (0..BUFFER_COUNT)
            .map(|_| {
                crate::train::arena::GpuBatchTensors::new(
                    prefetch_batch_size,
                    unroll_steps,
                    prefetch_device,
                )
            })
            .collect();

        let mut cycle = 0;

        while *prefetch_active_flag.read().unwrap() {
            if prefetch_replay_buffer.get_length() < prefetch_batch_size {
                thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
            let current_step = prefetch_replay_buffer
                .state
                .completed_games
                .load(std::sync::atomic::Ordering::Relaxed) as f64;
            let beta = (0.4 + 0.6 * (current_step / prefetch_max_steps.max(100_000.0))).min(1.0);

            if let Some(mut batch) = prefetch_replay_buffer.sample_batch(prefetch_batch_size, beta)
            {
                let idx = cycle % BUFFER_COUNT;
                pinned_arenas[idx].copy_from_unpinned(&batch);
                gpu_arenas[idx].copy_from_pinned(&pinned_arenas[idx]);

                batch.state_features_batch = gpu_arenas[idx].state_features.shallow_clone();
                batch.actions_batch = gpu_arenas[idx].actions.shallow_clone();
                batch.piece_identifiers_batch = gpu_arenas[idx].piece_identifiers.shallow_clone();
                batch.value_prefixs_batch = gpu_arenas[idx].value_prefixs.shallow_clone();
                batch.target_policies_batch = gpu_arenas[idx].target_policies.shallow_clone();
                batch.target_values_batch = gpu_arenas[idx].target_values.shallow_clone();
                batch.model_values_batch = gpu_arenas[idx].model_values.shallow_clone();
                batch.transition_boards_batch = gpu_arenas[idx].transition_boards.shallow_clone();
                batch.transition_actions_batch = gpu_arenas[idx].transition_actions.shallow_clone();
                batch.transition_metadata_batch =
                    gpu_arenas[idx].transition_metadata.shallow_clone();
                batch.loss_masks_batch = gpu_arenas[idx].loss_masks.shallow_clone();
                batch.importance_weights_batch = gpu_arenas[idx].importance_weights.shallow_clone();

                if prefetch_tx.send(batch).is_err() {
                    break;
                }
                cycle += 1;
            } else {
                thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    });

    let mut gradient_optimizer = nn::Adam::default()
        .build(&training_var_store, configuration_arc.lr_init)
        .unwrap();
    let mut last_trained_games = 0;
    let games_per_train_step = 1;

    let optimizer_network_arcswap = Arc::clone(&active_inference_net);
    let mut active_is_a = true;

    let optimizer_replay_buffer = Arc::clone(&shared_replay_buffer);
    let optimizer_configuration = Arc::clone(&configuration_arc);
    let optimizer_active_flag = Arc::clone(&active_training_flag);

    let mut training_steps = 0;

    let mut sys = sysinfo::System::new_all();
    let mut networks = sysinfo::Networks::new_with_refreshed_list();

    fn get_disk_io_bytes() -> (u64, u64) {
        if let Ok(content) = std::fs::read_to_string("/proc/diskstats") {
            let mut read_bytes = 0;
            let mut write_bytes = 0;
            for line in content.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 10 {
                    let name = parts[2];
                    if !name.starts_with("loop") && !name.starts_with("ram") {
                        if let (Ok(r), Ok(w)) = (parts[5].parse::<u64>(), parts[9].parse::<u64>()) {
                            read_bytes += r * 512;
                            write_bytes += w * 512;
                        }
                    }
                }
            }
            return (read_bytes, write_bytes);
        }
        (0, 0)
    }

    let (mut last_disk_read, mut last_disk_write) = get_disk_io_bytes();
    let mut last_time = std::time::Instant::now();
    let mut local_episodes = Vec::new();

    while *optimizer_active_flag.read().unwrap() {
        let current_games = optimizer_replay_buffer
            .state
            .completed_games
            .load(std::sync::atomic::Ordering::Relaxed);

        if current_games < last_trained_games + games_per_train_step {
            thread::sleep(std::time::Duration::from_millis(10));
            continue;
        }

        let mut batched_experience_tensorserience =
            match prefetch_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(batch) => batch,
                Err(_) => continue,
            };

        last_trained_games += games_per_train_step;

        let lr_multiplier = if (training_steps as f64) < 10000.0 {
            1.0
        } else if (training_steps as f64) < 50000.0 {
            0.1
        } else {
            0.01
        };
        let current_lr = optimizer_configuration.lr_init * lr_multiplier;
        gradient_optimizer.set_lr(current_lr);

        let step_metrics = trainer::optimization::train_step(
            &training_network,
            &ema_network,
            &mut gradient_optimizer,
            &optimizer_replay_buffer,
            &batched_experience_tensorserience,
            optimizer_configuration.unroll_steps,
        );

        if let Some(arena) = batched_experience_tensorserience.arena.take() {
            optimizer_replay_buffer.return_arena(arena);
        }

        sys.refresh_cpu_usage();
        sys.refresh_memory();
        networks.refresh_list();

        let now = std::time::Instant::now();
        let dt = now.duration_since(last_time).as_secs_f64().max(0.1);
        last_time = now;

        let mut rx_bytes = 0;
        let mut tx_bytes = 0;
        for (_, net) in &networks {
            rx_bytes += net.received();
            tx_bytes += net.transmitted();
        }
        let network_rx_mbps = (rx_bytes as f64 / dt) / 1024.0 / 1024.0;
        let network_tx_mbps = (tx_bytes as f64 / dt) / 1024.0 / 1024.0;

        let (cur_read, cur_write) = get_disk_io_bytes();
        let disk_read_mbps =
            (cur_read.saturating_sub(last_disk_read) as f64 / dt) / 1024.0 / 1024.0;
        let disk_write_mbps =
            (cur_write.saturating_sub(last_disk_write) as f64 / dt) / 1024.0 / 1024.0;
        last_disk_read = cur_read;
        last_disk_write = cur_write;

        let cpu_usage = sys.global_cpu_info().cpu_usage();
        let ram_usage = sys.used_memory() as f32 / 1024.0 / 1024.0;
        let (gpu_usage, vram_usage) = get_gpu_metrics();

        let disks = sysinfo::Disks::new_with_refreshed_list();
        let mut total_disk = 0;
        let mut used_disk = 0;
        for disk in &disks {
            total_disk += disk.total_space();
            used_disk += disk.total_space() - disk.available_space();
        }
        let disk_usage_pct = if total_disk > 0 {
            (used_disk as f64 / total_disk as f64) * 100.0
        } else {
            0.0
        };

        let mut score_min = 0.0_f32;
        let mut score_max = 0.0_f32;
        let mut score_mean = 0.0_f32;
        let mut score_med = 0.0_f32;
        let mut lines_cleared = 0_u32;
        let mut mcts_depth = 0.0_f32;
        let mut mcts_search_time = 0.0_f32;

        while let Some(episode) = optimizer_replay_buffer.state.episodes.pop() {
            local_episodes.push(episode);
        }

        let current_global_write_index = optimizer_replay_buffer
            .state
            .global_write_storage_index
            .load(std::sync::atomic::Ordering::Relaxed);
        let buffer_capacity = optimizer_configuration.buffer_capacity_limit;

        let remove_count = local_episodes
            .iter()
            .take_while(|episode| {
                episode.global_start_storage_index + buffer_capacity < current_global_write_index
            })
            .count();
        if remove_count > 0 {
            local_episodes.drain(0..remove_count);
        }

        let count = local_episodes.len();
        if count > 0 {
            let mut scores: Vec<f32> = local_episodes.iter().map(|e| e.score).collect();
            scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            score_min = scores[0];
            score_max = *scores.last().unwrap_or(&0.0);
            score_med = scores[count / 2];
            score_mean = scores.iter().sum::<f32>() / count as f32;

            let total_lines: u32 = local_episodes.iter().map(|e| e.lines_cleared).sum();
            lines_cleared = total_lines / count as u32;

            let sum_depth: f32 = local_episodes.iter().map(|e| e.mcts_depth_mean).sum();
            mcts_depth = sum_depth / count as f32;

            let sum_time: f32 = local_episodes.iter().map(|e| e.mcts_search_time_mean).sum();
            mcts_search_time = sum_time / count as f32;
        }

        let winrate_mean = (score_mean + 1.0) / 2.0;

        let json_metric = serde_json::json!({
            "step": training_steps,
            "total_loss": step_metrics.total_loss,
            "policy_loss": step_metrics.policy_loss,
            "value_loss": step_metrics.value_loss,
            "value_prefix_loss": step_metrics.value_prefix_loss,
            "lr": current_lr,
            "game_score_min": score_min,
            "game_score_max": score_max,
            "game_score_med": score_med,
            "game_score_mean": score_mean,
            "winrate_mean": winrate_mean,
            "game_lines_cleared": lines_cleared,
            "game_count": current_games,
            "ram_usage_mb": ram_usage,
            "gpu_usage_pct": gpu_usage,
            "cpu_usage_pct": cpu_usage,
            "io_usage": disk_read_mbps + disk_write_mbps,
            "disk_usage_pct": disk_usage_pct,
            "vram_usage_mb": vram_usage,
            "mcts_depth_mean": mcts_depth,
            "mcts_search_time_mean": mcts_search_time,
            "network_tx_mbps": network_tx_mbps,
            "network_rx_mbps": network_rx_mbps,
            "disk_read_mbps": disk_read_mbps,
            "disk_write_mbps": disk_write_mbps,
        });

        telemetry_logger.send_stdout(json_metric.to_string());

        telemetry_logger.send_metric(crate::telemetry::TelemetryData {
            run_id: optimizer_configuration.experiment_name_identifier.clone(),
            step: training_steps,
            total_loss: step_metrics.total_loss as f32,
            policy_loss: step_metrics.policy_loss as f32,
            value_loss: step_metrics.value_loss as f32,
            reward_loss: step_metrics.value_prefix_loss as f32,
            lr: current_lr,
            game_score_min: score_min,
            game_score_max: score_max,
            game_score_med: score_med,
            game_score_mean: score_mean,
            winrate_mean,
            game_lines_cleared: lines_cleared,
            game_count: current_games,
            ram_usage_mb: ram_usage,
            gpu_usage_pct: gpu_usage,
            cpu_usage_pct: cpu_usage,
            io_usage: (disk_read_mbps + disk_write_mbps) as f32,
            disk_usage_pct,
            vram_usage_mb: vram_usage,
            mcts_depth_mean: mcts_depth,
            mcts_search_time_mean: mcts_search_time,
            network_tx_mbps,
            network_rx_mbps,
            disk_read_mbps,
            disk_write_mbps,
        });

        training_steps += 1;

        if training_steps % 100 == 0 {
            println!(
                "🔄 Step {} | Games: {} | Loss: {:.4}",
                training_steps, current_games, step_metrics.total_loss
            );
            if !optimizer_configuration
                .paths
                .model_checkpoint_path
                .is_empty()
            {
                let _ =
                    training_var_store.save(&optimizer_configuration.paths.model_checkpoint_path);
            }
        }

        if training_steps % 50 == 0 {
            let active_is_a_local = active_is_a;

            let mut target_vars = if active_is_a_local {
                inference_var_store_b.variables()
            } else {
                inference_var_store.variables()
            };

            let src_vars = training_var_store.variables();

            let arcswap_clone = Arc::clone(&optimizer_network_arcswap);
            let net_a = Arc::clone(&inference_net_a);
            let net_b = Arc::clone(&inference_net_b);

            std::thread::spawn(move || {
                tch::no_grad(|| {
                    for (name, target_tensor) in target_vars.iter_mut() {
                        if let Some(src_tensor) = src_vars.get(name) {
                            target_tensor.copy_(src_tensor);
                        }
                    }
                    if active_is_a_local {
                        arcswap_clone.store(net_b);
                    } else {
                        arcswap_clone.store(net_a);
                    }
                });
            });

            active_is_a = !active_is_a;
        }

        tch::no_grad(|| {
            let mut exponential_moving_average_variables =
                exponential_moving_average_var_store.variables();
            let active_network_variables = training_var_store.variables();
            for (tensor_name, ema_tensor_mut) in exponential_moving_average_variables.iter_mut() {
                if let Some(active_tensor) = active_network_variables.get(tensor_name) {
                    let ema_decay_rate = 0.99;
                    let updated_tensor =
                        &*ema_tensor_mut * ema_decay_rate + active_tensor * (1.0 - ema_decay_rate);
                    ema_tensor_mut.copy_(&updated_tensor);
                }
            }
        });

        if max_steps > 0 && training_steps >= max_steps {
            println!(
                "✅ Hit max training steps limit ({}). Shutting down...",
                max_steps
            );
            println!("FINAL_EVAL_SCORE: {}", step_metrics.total_loss);
            *optimizer_active_flag.write().unwrap() = false;
            std::thread::sleep(std::time::Duration::from_millis(1500));
            std::process::exit(0);
        }
    }
}

pub fn get_gpu_metrics() -> (f32, f32) {
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=utilization.gpu,memory.used")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        if let Ok(out_str) = String::from_utf8(output.stdout) {
            if let Some(first_line) = out_str.trim().lines().next() {
                let parts: Vec<&str> = first_line.split(", ").collect();
                if parts.len() == 2 {
                    let gpu_util = parts[0].parse::<f32>().unwrap_or(0.0);
                    let vram_used = parts[1].parse::<f32>().unwrap_or(0.0);
                    return (gpu_util, vram_used);
                }
            }
        }
    }
    (0.0, 0.0)
}
