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
mod trainer;
mod web;

#[cfg(test)]
mod tests;

use crossbeam_channel::unbounded;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use tch::{nn, nn::OptimizerConfig, Device};

use crate::board::GameStateExt;
use crate::buffer::ReplayBuffer;
use crate::network::MuZeroNet;
use crate::web::{api_router, ws_router, AppState, EngineCommand, TelemetryStore};

#[tokio::main]
async fn main() {
    println!("🚀 Starting Tricked AI Native Engine");

    let telemetry = Arc::new(RwLock::new(TelemetryStore::default()));
    let current_game = Arc::new(RwLock::new(GameStateExt::new(None, 0, 0, 6, 0)));
    let current_difficulty = Arc::new(RwLock::new(6));

    let (cmd_tx, cmd_rx) = unbounded::<EngineCommand>();

    let eval_tx_state = Arc::new(RwLock::new(None));

    let app_state = AppState {
        current_game,
        current_difficulty,
        telemetry: Arc::clone(&telemetry),
        cmd_sender: cmd_tx,
        eval_tx: Arc::clone(&eval_tx_state),
    };

    let app = axum::Router::new()
        .merge(api_router())
        .merge(ws_router())
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();

    let tel_ref = Arc::clone(&telemetry);

    thread::spawn(move || {
        let mut shutdown_flags: Vec<Arc<RwLock<bool>>> = vec![];

        loop {
            match cmd_rx.recv() {
                Ok(EngineCommand::StartTraining(cfg)) => {
                    println!("🚀 Starting new training session: {}", cfg.exp_name);
                    let cfg_arc = Arc::new(*cfg);
                    let buffer = Arc::new(ReplayBuffer::new(
                        cfg_arc.capacity,
                        cfg_arc.unroll_steps,
                        cfg_arc.td_steps,
                    ));

                    // Use CUDA if listed in config, fallback to CPU
                    let device = if cfg_arc.device == "cuda" && tch::Cuda::is_available() {
                        Device::Cuda(0)
                    } else {
                        Device::Cpu
                    };

                    let vs = nn::VarStore::new(device);
                    let ema_vs = nn::VarStore::new(device);

                    let m_lock = Arc::new(Mutex::new(MuZeroNet::new(
                        &vs.root(),
                        cfg_arc.d_model,
                        cfg_arc.num_blocks,
                        cfg_arc.support_size,
                    )));
                    let ema_lock = Arc::new(Mutex::new(MuZeroNet::new(
                        &ema_vs.root(),
                        cfg_arc.d_model,
                        cfg_arc.num_blocks,
                        cfg_arc.support_size,
                    )));

                    let is_active = Arc::new(RwLock::new(true));
                    shutdown_flags.push(Arc::clone(&is_active));

                    let (eval_tx, eval_rx) = unbounded();

                    // Spawn Inference threads
                    let num_infer = 4;
                    for _ in 0..num_infer {
                        let e_rx = eval_rx.clone();
                        let m_lock_infer = Arc::clone(&m_lock);
                        let a_flag = Arc::clone(&is_active);
                        let d_model = cfg_arc.d_model;

                        thread::spawn(move || {
                            while *a_flag.read().unwrap() {
                                selfplay::inference_loop(
                                    e_rx.clone(),
                                    Arc::clone(&m_lock_infer),
                                    d_model,
                                    device,
                                );
                            }
                        });
                    }

                    // Spawn Selfplay threads
                    let num_workers = cfg_arc.num_processes;
                    for _ in 0..num_workers {
                        let cfg = Arc::clone(&cfg_arc);
                        let tx = eval_tx.clone();
                        let buf = Arc::clone(&buffer);
                        let tel = Arc::clone(&tel_ref);
                        let a_flag = Arc::clone(&is_active);

                        thread::spawn(move || {
                            while *a_flag.read().unwrap() {
                                selfplay::game_loop(
                                    Arc::clone(&cfg),
                                    tx.clone(),
                                    Arc::clone(&buf),
                                    Arc::clone(&tel),
                                );
                            }
                        });
                    }

                    // Spawn Optimizer Loop
                    let mut opt = nn::Adam::default().build(&vs, cfg_arc.lr_init).unwrap();
                    let t_model_lock = Arc::clone(&m_lock);
                    let t_ema_lock = Arc::clone(&ema_lock);
                    let t_buf = Arc::clone(&buffer);
                    let t_cfg = Arc::clone(&cfg_arc);
                    let t_flag = Arc::clone(&is_active);
                    let t_tel = Arc::clone(&tel_ref);

                    thread::spawn(move || {
                        let mut iters = 0;
                        while *t_flag.read().unwrap() {
                            if t_buf.get_length() < t_cfg.train_batch_size * 2 {
                                std::thread::sleep(std::time::Duration::from_millis(500));
                                continue;
                            }

                            let tr_model = t_model_lock.lock().unwrap();
                            let tr_ema = t_ema_lock.lock().unwrap();
                            crate::trainer::train_step(
                                &tr_model, &tr_ema, &mut opt, &t_buf, &t_cfg, device,
                            );
                            drop(tr_model);
                            drop(tr_ema);

                            // EMA Update
                            tch::no_grad(|| {
                                let mut ema_vars = ema_vs.variables();
                                let model_vars = vs.variables();
                                for (name, t_ema) in ema_vars.iter_mut() {
                                    if let Some(t_model) = model_vars.get(name) {
                                        t_ema.copy_(
                                            &(&*t_ema * 0.99 + t_model * 0.01),
                                        );
                                    }
                                }
                            });

                            iters += 1;
                            if iters % 100 == 0 {
                                if let Ok(mut tel) = t_tel.write() {
                                    tel.status.loss_total = 0.0; // Real logging logic in trainer but simplifying here
                                }
                            }
                        }
                    });
                }
                Ok(EngineCommand::StopTraining) => {
                    println!("🛑 Stopping training");
                    for flag in shutdown_flags.iter_mut() {
                        *flag.write().unwrap() = false;
                    }
                    shutdown_flags.clear();
                }
                Err(_) => break,
            }
        }
    });

    println!("🌐 Axum Web Server listening on 0.0.0.0:8000");
    axum::serve(listener, app).await.unwrap();
}
