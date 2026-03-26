mod board;
mod buffer;
mod config;
mod constants;
mod features;
mod mcts;
mod neighbors;
mod network;
mod node;
mod selfplay;
mod serialization;
mod sumtree;
mod telemetry;
mod trainer;
mod web;

#[cfg(test)]
mod tests;

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

                    let variable_store = nn::VarStore::new(computation_device);
                    let exponential_moving_average_var_store =
                        nn::VarStore::new(computation_device);

                    let neural_network_mutex = Arc::new(RwLock::new(MuZeroNet::new(
                        &variable_store.root(),
                        configuration_arc.d_model,
                        configuration_arc.num_blocks,
                        configuration_arc.support_size,
                    )));
                    let ema_network_mutex = Arc::new(RwLock::new(MuZeroNet::new(
                        &exponential_moving_average_var_store.root(),
                        configuration_arc.d_model,
                        configuration_arc.num_blocks,
                        configuration_arc.support_size,
                    )));

                    let active_training_flag = Arc::new(RwLock::new(true));
                    shutdown_flags.push(Arc::clone(&active_training_flag));

                    let (evaluation_sender, evaluation_receiver) = unbounded();

                    // Spawn Inference threads
                    let inference_thread_count = 4;
                    for _ in 0..inference_thread_count {
                        let thread_evaluation_receiver = evaluation_receiver.clone();
                        let thread_network_mutex = Arc::clone(&neural_network_mutex);
                        let thread_active_flag = Arc::clone(&active_training_flag);
                        let configuration_model_dimension = configuration_arc.d_model;

                        thread::spawn(move || {
                            while *thread_active_flag.read().unwrap() {
                                selfplay::inference_loop(
                                    thread_evaluation_receiver.clone(),
                                    Arc::clone(&thread_network_mutex),
                                    configuration_model_dimension,
                                    computation_device,
                                );
                            }
                        });
                    }

                    // Spawn Selfplay threads
                    let redis_logger =
                        Arc::new(crate::telemetry::RedisLogger::new("redis://127.0.0.1/"))
                            as Arc<dyn crate::telemetry::GameLogger>;

                    let selfplay_worker_count = configuration_arc.num_processes;
                    for _ in 0..selfplay_worker_count {
                        let thread_configuration = Arc::clone(&configuration_arc);
                        let thread_evaluation_sender = evaluation_sender.clone();
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
                                );
                            }
                        });
                    }

                    // Spawn Optimizer Loop
                    let mut gradient_optimizer = nn::Adam::default()
                        .build(&variable_store, configuration_arc.lr_init)
                        .unwrap();
                    let optimizer_network_mutex = Arc::clone(&neural_network_mutex);
                    let optimizer_ema_mutex = Arc::clone(&ema_network_mutex);
                    let optimizer_replay_buffer = Arc::clone(&shared_replay_buffer);
                    let optimizer_configuration = Arc::clone(&configuration_arc);
                    let optimizer_active_flag = Arc::clone(&active_training_flag);
                    let optimizer_telemetry = Arc::clone(&shared_telemetry);
                    let optimizer_logger = Arc::clone(&redis_logger);
                    thread::spawn(move || {
                        while *optimizer_active_flag.read().unwrap() {
                            if optimizer_replay_buffer.get_length()
                                < optimizer_configuration.train_batch_size
                            {
                                thread::sleep(std::time::Duration::from_millis(100));
                                continue;
                            }

                            {
                                let network_reference = optimizer_network_mutex.write().unwrap();
                                let ema_reference = optimizer_ema_mutex.write().unwrap();
                                let step_loss = trainer::optimization::train_step(
                                    &network_reference,
                                    &ema_reference,
                                    &mut gradient_optimizer,
                                    &optimizer_replay_buffer,
                                    &optimizer_configuration,
                                    computation_device,
                                );

                                optimizer_logger.log_training_step(step_loss as f32);
                                if let Ok(mut telemetry_lock) = optimizer_telemetry.write() {
                                    telemetry_lock.status.loss_total = step_loss as f32;
                                }
                            }

                            tch::no_grad(|| {
                                let mut exponential_moving_average_variables =
                                    exponential_moving_average_var_store.variables();
                                let active_network_variables = variable_store.variables();
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

                            // Yield to prevent complete starvation of the inference threads that share the network mutex
                            std::thread::sleep(std::time::Duration::from_millis(100));
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
