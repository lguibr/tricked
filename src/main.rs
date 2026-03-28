mod board;
mod buffer;
mod config;
mod constants;
mod features;
mod mcts;
mod network;
mod node;
mod queue;
mod reanalyze;
mod selfplay;
mod serialization;
mod sumtree;
mod telemetry;
mod trainer;
mod web;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod performance_benches;

use crossbeam_channel::unbounded;
use std::sync::{Arc, RwLock};
use std::thread;
use tch::{nn, nn::OptimizerConfig, Device};
use tower_http::cors::CorsLayer;

use crate::board::GameStateExt;
use crate::buffer::ReplayBuffer;
use crate::network::MuZeroNet;
use crate::web::{api_router, ws_router, AppState, EngineCommand, TelemetryStore};

#[tokio::main]
async fn main() {
    println!("🚀 Starting Tricked AI Native Engine");

    let shared_telemetry = Arc::new(RwLock::new(TelemetryStore::default()));
    let current_game_state = Arc::new(RwLock::new(GameStateExt::new(None, 0, 0, 6, 0)));
    let active_difficulty_level = Arc::new(RwLock::new(6));

    let (command_sender, command_receiver) = unbounded::<EngineCommand>();

    let evaluation_transmission_state = Arc::new(RwLock::new(None));

    let application_state = AppState {
        current_game: current_game_state,
        current_difficulty: active_difficulty_level,
        telemetry: Arc::clone(&shared_telemetry),
        cmd_sender: command_sender,
        eval_tx: Arc::clone(&evaluation_transmission_state),
    };

    let application_router = axum::Router::new()
        .nest("/api", api_router())
        .merge(ws_router())
        .layer(CorsLayer::permissive())
        .with_state(application_state);

    let tcp_listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();

    let telemetry_reference = Arc::clone(&shared_telemetry);

    thread::spawn(move || {
        let mut shutdown_flags: Vec<Arc<RwLock<bool>>> = vec![];

        loop {
            match command_receiver.recv() {
                Ok(EngineCommand::StartTraining(training_configuration)) => {
                    println!(
                        "🚀 Starting new training session: {}",
                        training_configuration.exp_name
                    );
                    let configuration_arc = Arc::new(*training_configuration);
                    let shared_replay_buffer = Arc::new(ReplayBuffer::new(
                        configuration_arc.capacity,
                        configuration_arc.unroll_steps,
                        configuration_arc.td_steps,
                    ));

                    // Use CUDA if listed in config, fallback to CPU
                    let computation_device =
                        if configuration_arc.device == "cuda" && tch::Cuda::is_available() {
                            Device::Cuda(0)
                        } else {
                            Device::Cpu
                        };

                    let mut training_var_store = nn::VarStore::new(computation_device);
                    let mut inference_var_store = nn::VarStore::new(computation_device);
                    let exponential_moving_average_var_store =
                        nn::VarStore::new(computation_device);

                    let training_network = MuZeroNet::new(
                        &training_var_store.root(),
                        configuration_arc.d_model,
                        configuration_arc.num_blocks,
                        configuration_arc.support_size,
                    );

                    let ema_network = MuZeroNet::new(
                        &exponential_moving_average_var_store.root(),
                        configuration_arc.d_model,
                        configuration_arc.num_blocks,
                        configuration_arc.support_size,
                    );

                    let neural_network_mutex = Arc::new(RwLock::new(MuZeroNet::new(
                        &inference_var_store.root(),
                        configuration_arc.d_model,
                        configuration_arc.num_blocks,
                        configuration_arc.support_size,
                    )));

                    let mut cmodule_inference: Option<Arc<tch::CModule>> = None;

                    if !configuration_arc.model_checkpoint.is_empty() {
                        if std::path::Path::new(&configuration_arc.model_checkpoint).exists() {
                            if configuration_arc.model_checkpoint.ends_with(".pt") {
                                println!(
                                    "🚀 Loading TorchScript model for pure inference: {}",
                                    configuration_arc.model_checkpoint
                                );
                                cmodule_inference = match tch::CModule::load_on_device(
                                    &configuration_arc.model_checkpoint,
                                    computation_device,
                                ) {
                                    Ok(module) => Some(Arc::new(module)),
                                    Err(e) => {
                                        println!(
                                            "Warning: Failed to load TorchScript module: {}",
                                            e
                                        );
                                        None
                                    }
                                };
                            } else {
                                println!(
                                    "🚀 Loading Native Rust weights from: {}",
                                    configuration_arc.model_checkpoint
                                );
                                training_var_store
                                    .load(&configuration_arc.model_checkpoint)
                                    .unwrap_or_else(|e| {
                                        println!("Warning: Failed to load weights: {}", e)
                                    });
                                inference_var_store.copy(&training_var_store).unwrap();
                            }
                        } else {
                            println!(
                                "⚠️ Warning: Checkpoint '{}' not found. Starting with newly initialized weights.",
                                configuration_arc.model_checkpoint
                            );
                            inference_var_store.copy(&training_var_store).unwrap();
                        }
                    } else {
                        inference_var_store.copy(&training_var_store).unwrap();
                    }

                    let active_training_flag = Arc::new(RwLock::new(true));
                    shutdown_flags.push(Arc::clone(&active_training_flag));

                    let reanalyze_worker_count =
                        std::cmp::max(1, configuration_arc.num_processes / 4);
                    let total_workers = configuration_arc.num_processes + reanalyze_worker_count;

                    let inference_queue = crate::queue::FixedInferenceQueue::new(
                        total_workers as usize,
                        total_workers as usize,
                    );

                    // Spawn Inference threads
                    let inference_thread_count = 1;
                    for _ in 0..inference_thread_count {
                        let thread_evaluation_receiver = inference_queue.clone();
                        let thread_network_mutex = Arc::clone(&neural_network_mutex);
                        let thread_cmodule = cmodule_inference.clone();
                        let thread_active_flag = Arc::clone(&active_training_flag);
                        let configuration_model_dimension = configuration_arc.d_model;
                        let max_nodes = (configuration_arc.simulations * 2) as usize + 10;

                        thread::spawn(move || {
                            while *thread_active_flag.read().unwrap() {
                                selfplay::inference_loop(
                                    thread_evaluation_receiver.clone(),
                                    Arc::clone(&thread_network_mutex),
                                    thread_cmodule.clone(),
                                    configuration_model_dimension,
                                    computation_device,
                                    total_workers as usize,
                                    max_nodes,
                                );
                            }
                        });
                    }

                    // Spawn Selfplay threads
                    let redis_logger =
                        Arc::new(crate::telemetry::RedisLogger::new("redis://127.0.0.1/"))
                            as Arc<dyn crate::telemetry::GameLogger>;

                    if let Ok(config_json) = serde_json::to_string(&*configuration_arc) {
                        redis_logger.log_config(&config_json);
                    }

                    let selfplay_worker_count = configuration_arc.num_processes;
                    for worker_id in 0..selfplay_worker_count {
                        let thread_configuration = Arc::clone(&configuration_arc);
                        let thread_evaluation_sender = inference_queue.clone();
                        let thread_replay_buffer = Arc::clone(&shared_replay_buffer);
                        let thread_telemetry_store = Arc::clone(&telemetry_reference);
                        let thread_active_flag = Arc::clone(&active_training_flag);
                        let thread_logger = Arc::clone(&redis_logger);

                        thread::spawn(move || {
                            while *thread_active_flag.read().unwrap() {
                                selfplay::game_loop(
                                    Arc::clone(&thread_configuration),
                                    thread_evaluation_sender.clone(),
                                    Arc::clone(&thread_replay_buffer),
                                    Arc::clone(&thread_telemetry_store),
                                    Arc::clone(&thread_logger),
                                    worker_id as usize,
                                );
                            }
                        });
                    }

                    for worker_index in 0..reanalyze_worker_count {
                        let worker_id = selfplay_worker_count + worker_index;
                        let thread_configuration = Arc::clone(&configuration_arc);
                        let thread_evaluation_sender = inference_queue.clone();
                        let thread_replay_buffer = Arc::clone(&shared_replay_buffer);
                        let thread_active_flag = Arc::clone(&active_training_flag);

                        thread::spawn(move || {
                            while *thread_active_flag.read().unwrap() {
                                reanalyze::reanalyze_worker_loop(
                                    Arc::clone(&thread_configuration),
                                    thread_evaluation_sender.clone(),
                                    Arc::clone(&thread_replay_buffer),
                                    worker_id as usize,
                                );
                            }
                        });
                    }

                    // Spawn Optimizer Loop
                    let mut gradient_optimizer = nn::Adam::default()
                        .build(&training_var_store, configuration_arc.lr_init)
                        .unwrap();
                    let mut last_trained_games = 0;
                    let games_per_train_step = 1;

                    let optimizer_network_mutex = Arc::clone(&neural_network_mutex);
                    let optimizer_replay_buffer = Arc::clone(&shared_replay_buffer);
                    let optimizer_configuration = Arc::clone(&configuration_arc);
                    let optimizer_active_flag = Arc::clone(&active_training_flag);
                    let optimizer_telemetry = Arc::clone(&shared_telemetry);
                    let optimizer_logger = Arc::clone(&redis_logger);
                    thread::spawn(move || {
                        while *optimizer_active_flag.read().unwrap() {
                            let current_games = optimizer_replay_buffer
                                .state
                                .completed_games
                                .load(std::sync::atomic::Ordering::Relaxed);

                            // 2. Gate the optimizer: Only train if we have enough NEW games AND enough total data
                            if current_games < last_trained_games + games_per_train_step
                                || optimizer_replay_buffer.get_length()
                                    < optimizer_configuration.train_batch_size
                            {
                                thread::sleep(std::time::Duration::from_millis(10));
                                continue;
                            }

                            // 3. Update our tracker
                            last_trained_games = current_games;

                            {
                                let training_steps =
                                    optimizer_telemetry.read().unwrap().status.training_steps
                                        as f64;
                                let lr_multiplier = if training_steps < 10000.0 {
                                    1.0
                                } else if training_steps < 50000.0 {
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
                                    &optimizer_configuration,
                                    computation_device,
                                );

                                optimizer_logger.log_training_step(
                                    step_metrics.total_loss as f32,
                                    step_metrics.policy_loss as f32,
                                    step_metrics.value_loss as f32,
                                    step_metrics.reward_loss as f32,
                                );
                                optimizer_logger.log_metric("learning_rate", current_lr as f32);

                                if let Ok(mut telemetry_lock) = optimizer_telemetry.write() {
                                    telemetry_lock.status.loss_total =
                                        step_metrics.total_loss as f32;
                                    telemetry_lock.status.training_steps += 1;

                                    if telemetry_lock.status.training_steps % 100 == 0
                                        && !optimizer_configuration.model_checkpoint.is_empty()
                                    {
                                        let _ = training_var_store
                                            .save(&optimizer_configuration.model_checkpoint);
                                    }
                                }
                            }

                            // Synchronization: Lock inference read access briefly to copy weights
                            if optimizer_telemetry.read().unwrap().status.training_steps % 50 == 0 {
                                let _network_reference = optimizer_network_mutex.write().unwrap();
                                tch::no_grad(|| {
                                    inference_var_store.copy(&training_var_store).unwrap();
                                });
                            }

                            tch::no_grad(|| {
                                let mut exponential_moving_average_variables =
                                    exponential_moving_average_var_store.variables();
                                let active_network_variables = training_var_store.variables();
                                for (tensor_name, ema_tensor_mut) in
                                    exponential_moving_average_variables.iter_mut()
                                {
                                    if let Some(active_tensor) =
                                        active_network_variables.get(tensor_name)
                                    {
                                        let ema_decay_rate = 0.99;
                                        let updated_tensor = &*ema_tensor_mut * ema_decay_rate
                                            + active_tensor * (1.0 - ema_decay_rate);
                                        ema_tensor_mut.copy_(&updated_tensor);
                                    }
                                }
                            });
                        }
                    });
                }
                Ok(EngineCommand::StopTraining) => {
                    for flag in shutdown_flags.iter() {
                        if let Ok(mut write_guard) = flag.write() {
                            *write_guard = false;
                        }
                    }
                    shutdown_flags.clear();
                    println!("🛑 Stopped all training threads.");
                }
                Err(_) => break,
            }
        }
    });

    println!("🌍 Server running at http://localhost:8000");
    axum::serve(tcp_listener, application_router).await.unwrap();
}
