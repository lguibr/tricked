use std::sync::Arc;
use std::thread;
use tch::{nn, nn::OptimizerConfig, Device};

use crate::config::Config;
// workers moved
use crate::net::MuZeroNet;
use crate::queue;
use crate::train::buffer::ReplayBuffer;
use crate::train::optimizer as trainer;

use super::workers::{spawn_inference_thread, spawn_mcts_workers, spawn_reanalyze_workers};
use super::prefetch::spawn_prefetch_thread;
use super::telemetry::spawn_telemetry_thread;

#[hotpath::measure]
pub fn run_training(
    config: Config,
    max_steps: usize,
    external_abort: Option<Arc<std::sync::atomic::AtomicBool>>,
    on_metric: Option<Box<dyn Fn(serde_json::Value) + Send + Sync>>,
) -> (f64, f64) {
    println!(
        "🚀 Starting Tricked AI Native Engine (CLI Mode) for experiment: {}",
        config.experiment_name_identifier
    );

    let configuration_arc = Arc::new(config);

    assert!(
        configuration_arc.optimizer.buffer_capacity_limit
            > configuration_arc.optimizer.train_batch_size
    );
    assert!(configuration_arc.optimizer.temporal_difference_steps > 0);
    assert!(configuration_arc.hardware.num_processes > 0);

    let artifacts_dir = std::path::Path::new(&configuration_arc.paths.metrics_file_path)
        .parent()
        .map(|p| p.to_string_lossy().to_string());

    let shared_replay_buffer = Arc::new(ReplayBuffer::new(
        configuration_arc.optimizer.buffer_capacity_limit,
        configuration_arc.optimizer.unroll_steps,
        configuration_arc.optimizer.temporal_difference_steps,
        configuration_arc.optimizer.train_batch_size,
        artifacts_dir,
        configuration_arc.optimizer.discount_factor,
        configuration_arc.optimizer.td_lambda,
    ));

    #[cfg(unix)]
    unsafe {
        // Horrific workaround for Python pip PyTorch >= 2.1 splitting CUDA libraries dynamically
        let _ = libc::dlopen(
            c"libc10_cuda.so".as_ptr(),
            libc::RTLD_NOW | libc::RTLD_GLOBAL,
        );
        let _ = libc::dlopen(
            c"libtorch_cuda.so".as_ptr(),
            libc::RTLD_NOW | libc::RTLD_GLOBAL,
        );
    }

    let computation_device = if configuration_arc.hardware.device.starts_with("cuda")
        && tch::Cuda::is_available()
    {
        // extract the device index if possible, else 0
        Device::Cuda(0) // Simplification for now
    } else if tch::Cuda::is_available() {
        panic!("❌ CUDA GPU detected, but CPU fallback was triggered by config ('{}')! To prevent severe performance degradation, Tricked AI refuses to run on CPU when a GPU is present.", configuration_arc.hardware.device);
    } else {
        Device::Cpu
    };

    println!("🚀 Hardware detected: {:?}", computation_device);

    let mut training_var_store = nn::VarStore::new(computation_device);
    let mut inference_var_store = nn::VarStore::new(computation_device);
    let exponential_moving_average_var_store = nn::VarStore::new(computation_device);

    let training_network = MuZeroNet::new(
        &training_var_store.root(),
        configuration_arc.architecture.hidden_dimension_size,
        configuration_arc.architecture.num_blocks,
        configuration_arc.architecture.value_support_size,
        configuration_arc.architecture.reward_support_size,
        configuration_arc.architecture.spatial_channel_count,
        configuration_arc.architecture.hole_predictor_dim,
    );
    let _ema_network = MuZeroNet::new(
        &exponential_moving_average_var_store.root(),
        configuration_arc.architecture.hidden_dimension_size,
        configuration_arc.architecture.num_blocks,
        configuration_arc.architecture.value_support_size,
        configuration_arc.architecture.reward_support_size,
        configuration_arc.architecture.spatial_channel_count,
        configuration_arc.architecture.hole_predictor_dim,
    );

    let mut inference_var_store_b = nn::VarStore::new(computation_device);
    let inference_net_a = Arc::new(MuZeroNet::new(
        &inference_var_store.root(),
        configuration_arc.architecture.hidden_dimension_size,
        configuration_arc.architecture.num_blocks,
        configuration_arc.architecture.value_support_size,
        configuration_arc.architecture.reward_support_size,
        configuration_arc.architecture.spatial_channel_count,
        configuration_arc.architecture.hole_predictor_dim,
    ));
    let inference_net_b = Arc::new(MuZeroNet::new(
        &inference_var_store_b.root(),
        configuration_arc.architecture.hidden_dimension_size,
        configuration_arc.architecture.num_blocks,
        configuration_arc.architecture.value_support_size,
        configuration_arc.architecture.reward_support_size,
        configuration_arc.architecture.spatial_channel_count,
        configuration_arc.architecture.hole_predictor_dim,
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

    let active_training_flag =
        external_abort.unwrap_or_else(|| Arc::new(std::sync::atomic::AtomicBool::new(true)));

    let workspace_db_path = configuration_arc
        .paths
        .workspace_db
        .clone()
        .unwrap_or_else(|| "tricked_workspace.db".to_string());

    let telemetry_logger = crate::telemetry::TelemetryLogger::new(workspace_db_path.clone());

    let shared_queue_saturation = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let shared_heatmap = Arc::new(std::sync::RwLock::new([0.0_f32; 96]));

    let initial_difficulty = if configuration_arc.environment.difficulty < 3 {
        3
    } else {
        configuration_arc.environment.difficulty
    };
    let global_difficulty = Arc::new(std::sync::atomic::AtomicI32::new(initial_difficulty));

    let global_gumbel_scale_multiplier =
        Arc::new(std::sync::atomic::AtomicU32::new(1.0_f32.to_bits()));

    let stdin_active_flag = Arc::clone(&active_training_flag);
    use std::io::IsTerminal;
    if std::io::stdin().is_terminal() {
        let _ = std::thread::Builder::new()
            .name("stdin".into())
            .spawn(move || {
                let stdin = std::io::stdin();
                let mut buffer = String::new();
                while stdin_active_flag.load(std::sync::atomic::Ordering::Relaxed) {
                    buffer.clear();
                    if let Ok(bytes) = stdin.read_line(&mut buffer) {
                        if bytes == 0 {
                            break;
                        }
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&buffer) {
                            if v.get("cmd").and_then(|c| c.as_str()) == Some("stop") {
                                stdin_active_flag.store(false, std::sync::atomic::Ordering::SeqCst);
                                std::thread::sleep(std::time::Duration::from_millis(1500));
                                std::process::exit(0);
                            }
                        }
                    }
                }
            });
    }

    let config_path = std::path::Path::new(&configuration_arc.paths.metrics_file_path)
        .parent()
        .unwrap()
        .join("config.json");
    let config_json = serde_json::to_string_pretty(&*configuration_arc).unwrap();
    std::fs::write(config_path, config_json).unwrap();

    let reanalyze_worker_count = std::cmp::max(
        1,
        (configuration_arc.hardware.num_processes as f32
            * configuration_arc.optimizer.reanalyze_ratio)
            .round() as i64,
    );
    let total_workers = configuration_arc.hardware.num_processes + reanalyze_worker_count;
    let inference_queue = Arc::new(queue::FixedInferenceQueue::new(
        configuration_arc.optimizer.buffer_capacity_limit,
        total_workers as usize,
    ));

    spawn_inference_thread(
        Arc::clone(&inference_queue),
        Arc::clone(&active_inference_net),
        cmodule_inference.clone(),
        Arc::clone(&active_training_flag),
        configuration_arc.architecture.hidden_dimension_size,
        computation_device,
        total_workers as usize,
        (configuration_arc.mcts.simulations as usize) * 2 + 1000,
        configuration_arc.hardware.inference_batch_size_limit as usize,
        configuration_arc.hardware.inference_timeout_ms as u64,
        Arc::clone(&shared_queue_saturation),
    );

    let selfplay_worker_count = configuration_arc.hardware.num_processes;
    spawn_mcts_workers(
        selfplay_worker_count as usize,
        Arc::clone(&configuration_arc),
        Arc::clone(&inference_queue),
        Arc::clone(&shared_replay_buffer),
        Arc::clone(&active_training_flag),
        Arc::clone(&shared_heatmap),
        Arc::clone(&global_difficulty),
        Arc::clone(&global_gumbel_scale_multiplier),
    );

    spawn_reanalyze_workers(
        selfplay_worker_count as usize,
        reanalyze_worker_count as usize,
        Arc::clone(&configuration_arc),
        Arc::clone(&inference_queue),
        Arc::clone(&shared_replay_buffer),
        Arc::clone(&active_training_flag),
    );

    let (prefetch_tx, prefetch_rx) = crossbeam_channel::bounded(4);
    spawn_prefetch_thread(
        Arc::clone(&shared_replay_buffer),
        Arc::clone(&active_training_flag),
        prefetch_tx,
        computation_device,
        configuration_arc.optimizer.train_batch_size,
        configuration_arc.optimizer.unroll_steps,
        max_steps as f64,
    );

    let adam_cfg = nn::Adam {
        wd: configuration_arc.optimizer.weight_decay,
        ..Default::default()
    };
    let mut gradient_optimizer = adam_cfg
        .build(&training_var_store, configuration_arc.optimizer.lr_init)
        .unwrap();
    let mut last_trained_games = 0;
    let games_per_train_step = 1;

    let optimizer_network_arcswap = Arc::clone(&active_inference_net);
    let mut active_is_a = true;

    let optimizer_replay_buffer = Arc::clone(&shared_replay_buffer);
    let optimizer_configuration = Arc::clone(&configuration_arc);
    let optimizer_active_flag = Arc::clone(&active_training_flag);

    let mut training_steps = 0;
    let mut accumulated_elapsed_time = 0.0_f64;

    // --- HISTORY RESUME LOGIC ---
    match rusqlite::Connection::open(&workspace_db_path) {
        Ok(c) => {
            match c.prepare("SELECT step, game_count, elapsed_time, difficulty FROM metrics WHERE run_id = ?1 ORDER BY step DESC LIMIT 1") {
                Ok(mut stmt) => {
                    match stmt.query_row(rusqlite::params![&optimizer_configuration.experiment_name_identifier], |r| {
                        Ok((
                            r.get::<_, i64>(0)?,
                            r.get::<_, i64>(1)?,
                            r.get::<_, f64>(2).unwrap_or(0.0),
                            r.get::<_, f64>(3).unwrap_or(0.0),
                        ))
                    }) {
                        Ok(row) => {
                            training_steps = row.0 as usize + 1;
                            last_trained_games = row.1 as usize;
                            accumulated_elapsed_time = row.2;
                            global_difficulty.store(row.3 as i32, std::sync::atomic::Ordering::SeqCst);
                            println!("🔄 RESUMED RUN: Starting at step {}, elapsed: {:.2}s, difficulty: {}", training_steps, accumulated_elapsed_time, row.3);
                        },
                        Err(rusqlite::Error::QueryReturnedNoRows) => println!("ℹ️ NO PREVIOUS RUN METRICS. Starting fresh."),
                        Err(e) => println!("⚠️ RESUME QUERY ERROR: {:?}", e),
                    }
                },
                Err(e) => println!("⚠️ RESUME SQL PREPARE ERROR: {:?}", e),
            }
        },
        Err(e) => println!("❌ COULD NOT OPEN DB FOR RESUME: {:?}", e),
    }
    // ----------------------------

    let shared_cpu_usage = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let shared_ram_usage = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let shared_disk_read = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let shared_disk_write = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let shared_net_rx = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let shared_net_tx = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let shared_disk_pct = Arc::new(std::sync::atomic::AtomicU32::new(0));

    let telemetry_active = Arc::clone(&active_training_flag);
    let t_cpu = Arc::clone(&shared_cpu_usage);
    let t_ram = Arc::clone(&shared_ram_usage);
    let t_disk_r = Arc::clone(&shared_disk_read);
    let t_disk_w = Arc::clone(&shared_disk_write);
    let t_net_rx = Arc::clone(&shared_net_rx);
    let t_net_tx = Arc::clone(&shared_net_tx);
    let t_disk_pct = Arc::clone(&shared_disk_pct);

    spawn_telemetry_thread(
        Arc::clone(&telemetry_active),
        Arc::clone(&t_cpu),
        Arc::clone(&t_ram),
        Arc::clone(&t_disk_r),
        Arc::clone(&t_disk_w),
        Arc::clone(&t_net_rx),
        Arc::clone(&t_net_tx),
        Arc::clone(&t_disk_pct),
    );

    let mut local_episodes = Vec::new();
    let training_start_time = std::time::Instant::now();

    let abs_bptt_path = format!(
        "/tmp/bptt_kernel_b{}_c{}_s{}.pt",
        configuration_arc.architecture.num_blocks,
        configuration_arc.architecture.hidden_dimension_size,
        configuration_arc.architecture.value_support_size
    );

    if !std::path::Path::new(&abs_bptt_path).exists() {
        println!(
            "🚀 JIT Compiling Fused Native Training Kernel to {}...",
            abs_bptt_path
        );
        let status = std::process::Command::new("python3")
            .args([
                "scripts/export_bptt.py",
                "--blocks",
                &configuration_arc.architecture.num_blocks.to_string(),
                "--channels",
                &configuration_arc
                    .architecture
                    .hidden_dimension_size
                    .to_string(),
                "--support",
                &configuration_arc
                    .architecture
                    .value_support_size
                    .to_string(),
                "--spatial-channels",
                &configuration_arc
                    .architecture
                    .spatial_channel_count
                    .to_string(),
                "--output",
                &abs_bptt_path,
            ])
            .status()
            .expect("Failed to execute BPTT exporter python script");
        if !status.success() {
            panic!("Python JIT exporter failed. Check Torch installation!");
        }
    }

    let mut cmodule_bptt = tch::CModule::load_on_device(&abs_bptt_path, computation_device)
        .expect("Failed to load compiled BPTT kernel from disk");

    let mut final_loss = f64::MAX;
    let mut final_mcts_time = f64::MAX;

    while optimizer_active_flag.load(std::sync::atomic::Ordering::Relaxed) {
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
        let current_lr = optimizer_configuration.optimizer.lr_init * lr_multiplier;
        gradient_optimizer.set_lr(current_lr);

        let step_metrics = trainer::optimization::train_step(
            &training_network,
            &mut gradient_optimizer,
            &optimizer_replay_buffer,
            &batched_experience_tensorserience,
            optimizer_configuration.optimizer.unroll_steps,
            &training_var_store,
            &mut cmodule_bptt,
        );

        final_loss = step_metrics.total_loss as f64;

        if let Some(arena) = batched_experience_tensorserience.arena.take() {
            optimizer_replay_buffer.return_arena(arena);
        }

        let cpu_usage = f32::from_bits(shared_cpu_usage.load(std::sync::atomic::Ordering::Relaxed));
        let ram_usage = f32::from_bits(shared_ram_usage.load(std::sync::atomic::Ordering::Relaxed));
        let disk_read_mbps = f64::from(f32::from_bits(
            shared_disk_read.load(std::sync::atomic::Ordering::Relaxed),
        ));
        let disk_write_mbps = f64::from(f32::from_bits(
            shared_disk_write.load(std::sync::atomic::Ordering::Relaxed),
        ));
        let network_rx_mbps = f64::from(f32::from_bits(
            shared_net_rx.load(std::sync::atomic::Ordering::Relaxed),
        ));
        let network_tx_mbps = f64::from(f32::from_bits(
            shared_net_tx.load(std::sync::atomic::Ordering::Relaxed),
        ));
        let disk_usage_pct = f64::from(f32::from_bits(
            shared_disk_pct.load(std::sync::atomic::Ordering::Relaxed),
        ));

        let (gpu_usage, vram_usage) = get_gpu_metrics();

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
        let buffer_capacity = optimizer_configuration.optimizer.buffer_capacity_limit;

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
            final_mcts_time = mcts_search_time as f64;
        }

        let winrate_mean = (score_mean + 1.0) / 2.0;

        let current_sat =
            f32::from_bits(shared_queue_saturation.load(std::sync::atomic::Ordering::Relaxed));

        let latency_sum = inference_queue
            .latency_sum_nanos
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        let latency_count = inference_queue
            .latency_count
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        let queue_latency_us = if latency_count > 0 {
            (latency_sum / latency_count) / 1000
        } else {
            0
        };

        let sumtree_contention_us =
            optimizer_replay_buffer.state.per.get_and_reset_contention() / 1000;

        let total_trained = training_steps as f64
            * optimizer_configuration.optimizer.train_batch_size as f64
            * (optimizer_configuration.optimizer.unroll_steps as f64 + 1.0);
        let current_transitions = optimizer_replay_buffer
            .state
            .global_write_storage_index
            .load(std::sync::atomic::Ordering::Relaxed);
        let sps_vs_tps = if current_transitions > 0 {
            (total_trained / current_transitions as f64) as f32
        } else {
            0.0
        };

        let current_heatmap = {
            if let Ok(lock) = shared_heatmap.read() {
                lock.to_vec()
            } else {
                vec![0.0; 96]
            }
        };

        let elapsed_time = accumulated_elapsed_time + training_start_time.elapsed().as_secs_f64();

        let json_metric = serde_json::json!({
            "step": training_steps,
            "elapsed_time": elapsed_time,
            "total_loss": step_metrics.total_loss,
            "policy_loss": step_metrics.policy_loss,
            "value_loss": step_metrics.value_loss,
            "value_prefix_loss": step_metrics.value_prefix_loss,
            "policy_entropy": step_metrics.policy_entropy,
            "gradient_norm": step_metrics.gradient_norm,
            "representation_drift": step_metrics.representation_drift,
            "mean_td_error": step_metrics.mean_td_error,
            "queue_saturation_ratio": current_sat,
            "queue_latency_us": queue_latency_us,
            "sumtree_contention_us": sumtree_contention_us,
            "action_space_entropy": step_metrics.action_space_entropy,
            "layer_gradient_norms": step_metrics.layer_gradient_norms,
            "sps_vs_tps": sps_vs_tps,
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
        if let Some(f) = &on_metric {
            f(json_metric);
        }

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
            elapsed_time,
            network_tx_mbps,
            network_rx_mbps,
            disk_read_mbps,
            disk_write_mbps,
            policy_entropy: step_metrics.policy_entropy as f32,
            gradient_norm: step_metrics.gradient_norm as f32,
            representation_drift: step_metrics.representation_drift as f32,
            mean_td_error: step_metrics.mean_td_error as f32,
            queue_saturation_ratio: current_sat,
            sps_vs_tps,
            queue_latency_us: queue_latency_us as f32,
            sumtree_contention_us: sumtree_contention_us as f32,
            action_space_entropy: step_metrics.action_space_entropy as f32,
            layer_gradient_norms: step_metrics.layer_gradient_norms.clone(),
            spatial_heatmap: current_heatmap,
            difficulty: global_difficulty.load(std::sync::atomic::Ordering::Relaxed) as f32,
        });

        // --- SOTA CURRICULUM MANAGER ---
        // Dynamically scales game complexity (piece generation boundaries) as the agent masters current topological difficulties.
        let current_difficulty = global_difficulty.load(std::sync::atomic::Ordering::Relaxed);
        if current_difficulty < 6 {
            let target_score = match current_difficulty {
                0 => 800.0, // Single fragments are trivial
                1 => 500.0,
                2 => 350.0,
                3 => 250.0,
                _ => 150.0,
            };
            if score_mean >= target_score && training_steps > 50 {
                let next_diff = current_difficulty + 1;
                global_difficulty.store(next_diff, std::sync::atomic::Ordering::SeqCst);
                telemetry_logger.send_stdout(format!(
                    "🏆 CURRICULUM UPGRADE: Agent mastered Difficulty {} (EMA Score: {:.1}). Escalating to {}!",
                    current_difficulty, score_mean, next_diff
                ));
            }
        }
        // -------------------------------

        let current_multiplier = f32::from_bits(
            global_gumbel_scale_multiplier.load(std::sync::atomic::Ordering::Relaxed),
        );
        if step_metrics.action_space_entropy < 0.1 && current_multiplier < 10.0 {
            let next_multiplier = current_multiplier * 1.05;
            global_gumbel_scale_multiplier.store(
                next_multiplier.to_bits(),
                std::sync::atomic::Ordering::SeqCst,
            );
            telemetry_logger.send_stdout(format!(
                "⚠️ ENTROPY COLLAPSE IMMUNIZATION: action_space_entropy {:.3} < 0.1. Boosted exploration multiplier to {:.2}!",
                step_metrics.action_space_entropy, next_multiplier
            ));
        } else if step_metrics.action_space_entropy > 0.5 && current_multiplier > 1.0 {
            let next_multiplier = (current_multiplier * 0.99).max(1.0);
            global_gumbel_scale_multiplier.store(
                next_multiplier.to_bits(),
                std::sync::atomic::Ordering::SeqCst,
            );
        }

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

            let _ = std::thread::Builder::new()
                .name("ema-swap".into())
                .spawn(move || {
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
                    *ema_tensor_mut *= ema_decay_rate;
                    *ema_tensor_mut += active_tensor * (1.0 - ema_decay_rate);
                }
            }
        });

        if max_steps > 0 && training_steps >= max_steps {
            println!("🛑 Max steps ({}) reached. Graceful exit.", max_steps);
            active_training_flag.store(false, std::sync::atomic::Ordering::SeqCst);
            break;
        }
    }

    telemetry_logger.close();
    println!("✅ Native Tricked AI Engine Session Completed.");
    (final_loss, final_mcts_time)
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

