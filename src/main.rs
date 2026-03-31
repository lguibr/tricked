use clap::{Parser, Subcommand};
use crossbeam_channel::unbounded;
use std::sync::{Arc, RwLock};
use std::thread;
use tch::{nn, nn::OptimizerConfig, Device};

use tricked_engine::config::Config;
use tricked_engine::env::reanalyze;
use tricked_engine::env::worker as selfplay;
use tricked_engine::net::MuZeroNet;
use tricked_engine::queue;
use tricked_engine::train::buffer::ReplayBuffer;
use tricked_engine::train::optimizer as trainer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    Train {
        /// The name of the experiment for logging and paths
        #[arg(long, default_value = "cli_run")]
        experiment_name: String,

        /// Optional path to a JSON/YAML config file
        #[arg(short, long)]
        config: Option<String>,

        /// Overrides for hyperparameters
        #[arg(long)]
        lr_init: Option<f64>,
        #[arg(long)]
        simulations: Option<i64>,
        #[arg(long)]
        unroll_steps: Option<usize>,
        #[arg(long)]
        temporal_difference_steps: Option<usize>,
        #[arg(long)]
        reanalyze_ratio: Option<f32>,
        #[arg(long)]
        support_size: Option<i64>,
        #[arg(long)]
        temp_decay_steps: Option<i64>,

        /// Max training steps to run before exiting (0 = infinite)
        #[arg(long, default_value = "0")]
        max_steps: usize,
    },
}

fn main() {
    let cli = Cli::parse();
    let Commands::Train {
        experiment_name,
        config,
        lr_init,
        simulations,
        unroll_steps,
        temporal_difference_steps,
        reanalyze_ratio,
        support_size,
        temp_decay_steps,
        max_steps,
    } = cli.command;
    let mut cfg = if let Some(path) = config {
        let file = std::fs::File::open(&path).expect("Failed to open config file");
        let mut parsed: Config = if path.ends_with(".yaml") || path.ends_with(".yml") {
            serde_yaml::from_reader(file).expect("Failed to parse YAML config")
        } else {
            serde_json::from_reader(file).expect("Failed to parse JSON config")
        };
        parsed.experiment_name_identifier = experiment_name.clone();
        parsed.paths = tricked_engine::config::ExperimentPaths::new(&experiment_name);
        parsed
    } else {
        Config {
            experiment_name_identifier: experiment_name.clone(),
            paths: tricked_engine::config::ExperimentPaths::new(&experiment_name),
            device: "cuda".to_string(),
            hidden_dimension_size: 256,
            num_blocks: 10,
            support_size: 300,
            buffer_capacity_limit: 1_000_000,
            simulations: 200,
            train_batch_size: 256,
            train_epochs: 1000,
            num_processes: 4,
            worker_device: "cpu".to_string(),
            unroll_steps: 15,
            temporal_difference_steps: 15,
            zmq_batch_size: 64,
            zmq_timeout_ms: 5,
            max_gumbel_k: 16,
            gumbel_scale: 1.0,
            temp_decay_steps: 10000,
            difficulty: 6,
            temp_boost: true,
            lr_init: 0.0003,
            reanalyze_ratio: 0.0,
        }
    };
    if let Some(v) = lr_init {
        cfg.lr_init = v;
    }
    if let Some(v) = simulations {
        cfg.simulations = v;
    }
    if let Some(v) = unroll_steps {
        cfg.unroll_steps = v;
    }
    if let Some(v) = temporal_difference_steps {
        cfg.temporal_difference_steps = v;
    }
    if let Some(v) = reanalyze_ratio {
        cfg.reanalyze_ratio = v;
    }
    if let Some(v) = support_size {
        cfg.support_size = v;
    }
    if let Some(v) = temp_decay_steps {
        cfg.temp_decay_steps = v;
    }

    run_training(cfg, max_steps);
}

fn run_training(config: Config, max_steps: usize) {
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

    std::fs::create_dir_all(
        std::path::Path::new(&configuration_arc.paths.metrics_file_path)
            .parent()
            .unwrap(),
    )
    .unwrap();
    let mut csv_writer =
        csv::Writer::from_path(&configuration_arc.paths.metrics_file_path).unwrap();
    csv_writer
        .write_record(&[
            "step",
            "total_loss",
            "policy_loss",
            "value_loss",
            "reward_loss",
            "lr",
        ])
        .unwrap();
    let csv_mutex = Arc::new(std::sync::Mutex::new(csv_writer));

    let reanalyze_worker_count = std::cmp::max(1, configuration_arc.num_processes / 4);
    let total_workers = configuration_arc.num_processes + reanalyze_worker_count;
    let inference_queue = Arc::new(queue::FixedInferenceQueue::new(
        total_workers as usize,
        total_workers as usize,
    ));

    for _ in 0..1 {
        let thread_evaluation_receiver = Arc::clone(&inference_queue);
        let thread_network_mutex = Arc::clone(&active_inference_net);
        let thread_cmodule = cmodule_inference.clone();
        let thread_active_flag = Arc::clone(&active_training_flag);
        let configuration_model_dimension = configuration_arc.hidden_dimension_size;
        let max_nodes = (configuration_arc.simulations as usize) + 300;
        let inference_batch_size_limit = configuration_arc.zmq_batch_size as usize;
        let inference_timeout_milliseconds = configuration_arc.zmq_timeout_ms as u64;

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
    let (prefetch_tx, prefetch_rx) = unbounded();
    let prefetch_device = computation_device;
    let prefetch_batch_size = configuration_arc.train_batch_size;

    thread::spawn(move || {
        while *prefetch_active_flag.read().unwrap() {
            if prefetch_replay_buffer.get_length() < prefetch_batch_size {
                thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
            let current_step = prefetch_replay_buffer
                .state
                .completed_games
                .load(std::sync::atomic::Ordering::Relaxed) as f64;
            let beta = (0.4 + 0.6 * (current_step / 100_000.0)).min(1.0);

            if let Some(batch) =
                prefetch_replay_buffer.sample_batch(prefetch_batch_size, prefetch_device, beta)
            {
                if prefetch_tx.send(batch).is_err() {
                    break;
                }
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

    while *optimizer_active_flag.read().unwrap() {
        let current_games = optimizer_replay_buffer
            .state
            .completed_games
            .load(std::sync::atomic::Ordering::Relaxed);

        if current_games < last_trained_games + games_per_train_step {
            thread::sleep(std::time::Duration::from_millis(10));
            continue;
        }

        let batched_experience_tensorserience =
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
            batched_experience_tensorserience,
            optimizer_configuration.unroll_steps,
        );

        if let Ok(mut writer) = csv_mutex.lock() {
            let _ = writer.write_record(&[
                training_steps.to_string(),
                step_metrics.total_loss.to_string(),
                step_metrics.policy_loss.to_string(),
                step_metrics.value_loss.to_string(),
                step_metrics.reward_loss.to_string(),
                current_lr.to_string(),
            ]);
            let _ = writer.flush();
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
            tch::no_grad(|| {
                if active_is_a {
                    inference_var_store_b.copy(&training_var_store).unwrap();
                    optimizer_network_arcswap.store(Arc::clone(&inference_net_b));
                } else {
                    inference_var_store.copy(&training_var_store).unwrap();
                    optimizer_network_arcswap.store(Arc::clone(&inference_net_a));
                }
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
            break;
        }
    }
}
