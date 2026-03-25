use crossbeam_channel::{Receiver, Sender, unbounded};
use pyo3::prelude::*;
use rand::Rng;
use std::sync::{Arc, Mutex};
use std::thread;
use tch::{CModule, Device, Tensor};

use crate::board::GameStateExt;
use crate::features::extract_feature_native;
use crate::mcts::{EvalReq, EvalResp, mcts_search};

#[derive(Clone)]
pub struct AppConfig {
    pub d_model: i64,
    pub num_blocks: i64,
    pub support_size: i64,
    pub simulations: usize,
    pub max_gumbel_k: usize,
    pub gumbel_scale: f32,
    pub difficulty: i32,
    pub temp_decay_steps: usize,
    pub push_port: String,
    pub sub_port: String,
}

fn inference_loop(rx: Receiver<EvalReq>, sub_port: String) {
    let device = Device::Cuda(0);
    let mut model: Option<CModule> = None;
    
    let ctx = zmq::Context::new();
    let subscriber = ctx.socket(zmq::SUB).unwrap();
    subscriber.connect(&sub_port).unwrap();
    subscriber.set_subscribe(b"").unwrap();

    loop {
        // Hot Reload check via ZMQ IPC
        if let Ok(msg) = subscriber.recv_bytes(zmq::DONTWAIT) {
            println!("🔄 Received updated JIT model via IPC. Hot reloading...");
            if let Ok(new_model) = CModule::load_data_on_device(&mut msg.as_slice(), device) {
                model = Some(new_model);
            } else {
                println!("⚠️ Failed to load JIT model from IPC bytes.");
            }
        }

        if model.is_none() {
            std::thread::sleep(std::time::Duration::from_millis(100));
            continue;
        }

        let first_req = match rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(req) => req,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        };

        let mut reqs = vec![first_req];
        while reqs.len() < 1024 {
            if let Ok(req) = rx.try_recv() {
                reqs.push(req);
            } else {
                break;
            }
        }
        
        let Some(ref m) = model else { unreachable!() };

        let mut initial_reqs = Vec::new();
        let mut recurrent_reqs = Vec::new();

        for req in reqs.into_iter() {
            if req.is_initial {
                initial_reqs.push(req);
            } else {
                recurrent_reqs.push(req);
            }
        }

        tch::no_grad(|| {
            if !initial_reqs.is_empty() {
                let s_tensors: Vec<Tensor> = initial_reqs
                    .iter()
                    .map(|r| Tensor::from_slice(r.state_feat.as_ref().unwrap()).view([20, 96]))
                    .collect();
                let s_batch = Tensor::stack(&s_tensors, 0).to_device(device);

                let outputs = m
                    .method_is("initial_inference_jit", &[tch::IValue::from(s_batch)])
                    .unwrap();
                let tuple = if let tch::IValue::Tuple(t) = outputs { t } else { panic!("Expected tuple") };
                let h_next_batch = if let tch::IValue::Tensor(t) = &tuple[0] { t.copy() } else { panic!() }.to_device(Device::Cpu).contiguous();
                let value_batch = if let tch::IValue::Tensor(t) = &tuple[1] { t.copy() } else { panic!() }.to_device(Device::Cpu).contiguous();
                let policy_batch = if let tch::IValue::Tensor(t) = &tuple[2] { t.copy() } else { panic!() }.to_device(Device::Cpu).contiguous();

                let h_next_f32: Vec<f32> = h_next_batch.view([-1]).try_into().unwrap();
                let value_f32: Vec<f32> = value_batch.view([-1]).try_into().unwrap();
                let policy_f32: Vec<f32> = policy_batch.view([-1]).try_into().unwrap();

                let h_size = h_next_f32.len() / initial_reqs.len();
                let p_size = policy_f32.len() / initial_reqs.len();

                for (i, req) in initial_reqs.into_iter().enumerate() {
                    let resp = EvalResp {
                        h_next: h_next_f32[i * h_size..(i + 1) * h_size].to_vec(),
                        reward: 0.0,
                        value: value_f32[i],
                        p_next: policy_f32[i * p_size..(i + 1) * p_size].to_vec(),
                    };
                    let _ = req.tx.send(resp);
                }
            }

            if !recurrent_reqs.is_empty() {
                let h_tensors: Vec<Tensor> = recurrent_reqs
                    .iter()
                    .map(|r| {
                        let h = r.h_last.as_ref().unwrap();
                        let channels = h.len() / 96;
                        Tensor::from_slice(h).view([channels as i64, 96])
                    })
                    .collect();

                let a_tensors: Vec<Tensor> = recurrent_reqs
                    .iter()
                    .map(|r| Tensor::from_slice(&[r.piece_action]))
                    .collect();
                let p_tensors: Vec<Tensor> = recurrent_reqs
                    .iter()
                    .map(|r| Tensor::from_slice(&[r.piece_id]))
                    .collect();

                let h_batch = Tensor::stack(&h_tensors, 0).to_device(device);
                let a_batch = Tensor::stack(&a_tensors, 0).to_device(device);
                let p_batch = Tensor::stack(&p_tensors, 0).to_device(device);

                let outputs = m
                    .forward_is(&[
                        tch::IValue::from(h_batch),
                        tch::IValue::from(a_batch),
                        tch::IValue::from(p_batch),
                    ])
                    .unwrap();
                let tuple = if let tch::IValue::Tuple(t) = outputs { t } else { panic!("Expected tuple") };
                let h_next_batch = if let tch::IValue::Tensor(t) = &tuple[0] { t.copy() } else { panic!() }.to_device(Device::Cpu).contiguous();
                let reward_batch = if let tch::IValue::Tensor(t) = &tuple[1] { t.copy() } else { panic!() }.to_device(Device::Cpu).contiguous();
                let value_batch = if let tch::IValue::Tensor(t) = &tuple[2] { t.copy() } else { panic!() }.to_device(Device::Cpu).contiguous();
                let policy_batch = if let tch::IValue::Tensor(t) = &tuple[3] { t.copy() } else { panic!() }.to_device(Device::Cpu).contiguous();

                let h_next_f32: Vec<f32> = h_next_batch.view([-1]).try_into().unwrap();
                let reward_f32: Vec<f32> = reward_batch.view([-1]).try_into().unwrap();
                let value_f32: Vec<f32> = value_batch.view([-1]).try_into().unwrap();
                let policy_f32: Vec<f32> = policy_batch.view([-1]).try_into().unwrap();

                let h_size = h_next_f32.len() / recurrent_reqs.len();
                let p_size = policy_f32.len() / recurrent_reqs.len();

                for (i, req) in recurrent_reqs.into_iter().enumerate() {
                    let resp = EvalResp {
                        h_next: h_next_f32[i * h_size..(i + 1) * h_size].to_vec(),
                        reward: reward_f32[i],
                        value: value_f32[i],
                        p_next: policy_f32[i * p_size..(i + 1) * p_size].to_vec(),
                    };
                    let _ = req.tx.send(resp);
                }
            }
        });
    }
}

fn game_loop(cfg: Arc<AppConfig>, eval_tx: Sender<EvalReq>, pusher: Arc<Mutex<zmq::Socket>>, logger: Arc<dyn crate::telemetry::GameLogger>) {
    let mut rng = rand::thread_rng();
    let worker_pid = std::process::id() as i32;

    loop {
        let mut state = GameStateExt::new(None, 0, 0, cfg.difficulty, 0);
        let mut history = vec![state.board, state.board];
        let mut prefix_actions = Vec::new();
        let mut prefix_piece_ids = Vec::new();

        let mut ep_boards = Vec::new();
        let mut ep_available = Vec::new();
        let mut ep_actions = Vec::new();
        let mut ep_p_ids = Vec::new();
        let mut ep_rewards = Vec::new();
        let mut ep_policies = Vec::new();
        let mut ep_values = Vec::new();

        let mut prev_tree = None;
        let mut last_action = None;
        let mut step = 0;

        for _ in 0..1000 {
            if state.pieces_left == 0 {
                state.refill_tray();
            }
            if state.terminal {
                break;
            }

            let feat = extract_feature_native(
                &state,
                Some(history.clone()),
                Some(prefix_actions.clone()),
                cfg.difficulty,
            );

            let (tx, rx) = unbounded();
            eval_tx
                .send(EvalReq {
                    is_initial: true,
                    state_feat: Some(feat.clone()),
                    h_last: None,
                    piece_action: 0,
                    piece_id: 0,
                    tx,
                })
                .unwrap();
            let initial_resp = rx.recv().unwrap();

            let mcts_res = mcts_search(
                &initial_resp.h_next,
                &initial_resp.p_next,
                &state,
                cfg.simulations,
                cfg.max_gumbel_k,
                cfg.gumbel_scale,
                prev_tree,
                last_action,
                &eval_tx,
                None,
            )
            .unwrap();

            let best_action = mcts_res.0;
            let action_visits = mcts_res.1;
            let latent_val = mcts_res.2;
            prev_tree = Some(mcts_res.3);

            if best_action == -1 {
                break;
            }

            let mut top_moves: Vec<(i32, i32)> = action_visits.clone().into_iter().collect();
            top_moves.sort_by(|a, b| b.1.cmp(&a.1));
            top_moves.truncate(5);

            let mut hole_logits = Vec::new();
            if let Some(h_last) = initial_resp.h_next.clone().get(..96) {
                hole_logits.extend_from_slice(h_last);
            }

            logger.log_spectator_update(&crate::telemetry::SpectatorMetrics {
                worker: worker_pid,
                board: state.board.to_string(),
                score: state.score,
                pieces_left: state.pieces_left,
                terminal: state.terminal,
                available: state.available.clone(),
                hole_logits,
            });

            let temp = if step < (cfg.temp_decay_steps / 2) {
                1.0
            } else if step < cfg.temp_decay_steps {
                0.5
            } else {
                0.1
            };

            let mut policy_target = vec![0.0f32; 288];
            let mut sum_probs = 0.0;
            for (act, visits) in &action_visits {
                let prob = (*visits as f64).powf(1.0 / temp) as f32;
                policy_target[*act as usize] = prob;
                sum_probs += prob;
            }
            if sum_probs > 0.0 {
                for p in policy_target.iter_mut() {
                    *p /= sum_probs;
                }
            } else {
                let uniform = 1.0 / (action_visits.len() as f32);
                for (act, _) in &action_visits {
                    policy_target[*act as usize] = uniform;
                }
            }

            let u: f32 = rng.gen_range(0.0..=1.0);
            let mut cumulative = 0.0;
            let mut chosen = best_action;
            for (act, &prob) in policy_target.iter().enumerate() {
                if prob > 0.0 {
                    cumulative += prob;
                    if u <= cumulative {
                        chosen = act as i32;
                        break;
                    }
                }
            }

            last_action = Some(chosen);
            let slot = chosen / 96;
            let pos = chosen % 96;

            let next_state = state.apply_move(slot as usize, pos as usize).unwrap();
            let reward = (next_state.score - state.score) as f32;

            let piece_id = if state.available[slot as usize] == -1 {
                0
            } else {
                state.available[slot as usize]
            };
            let piece_action = piece_id * 96 + pos;

            ep_boards.push(state.board);
            ep_available.extend_from_slice(&state.available);
            ep_actions.push(piece_action as i64);
            ep_p_ids.push(piece_id as i64);
            ep_rewards.push(reward);
            ep_policies.extend_from_slice(&policy_target);
            ep_values.push(latent_val);

            history.push(state.board);
            if history.len() > 8 {
                history.remove(0);
            }

            prefix_actions.push(piece_action);
            prefix_piece_ids.push(piece_id);
            state = next_state;
            step += 1;
        }

        if step > 0 {
            let payload = crate::serialization::serialize_trajectory(
                cfg.difficulty,
                state.score as f32,
                step as u64,
                &ep_boards,
                &ep_available,
                &ep_actions,
                &ep_p_ids,
                &ep_rewards,
                &ep_policies,
                &ep_values,
            );

            pusher.lock().unwrap().send(&payload, 0).unwrap();
        }
    }
}

#[pyfunction]
pub fn run_self_play_worker(
    d_model: i64,
    num_blocks: i64,
    support_size: i64,
    simulations: usize,
    max_gumbel_k: usize,
    gumbel_scale: f32,
    difficulty: i32,
    temp_decay_steps: usize,
    push_port: String,
    sub_port: String,
) {
    let cfg = Arc::new(AppConfig {
        d_model,
        num_blocks,
        support_size,
        simulations,
        max_gumbel_k,
        gumbel_scale,
        difficulty,
        temp_decay_steps,
        push_port,
        sub_port: sub_port.clone(),
    });

    println!("🔥 Starting Tricked Native Self-Play Engine (via PyO3)");

    let ctx = zmq::Context::new();
    let pusher = ctx.socket(zmq::PUSH).unwrap();
    pusher.connect(&cfg.push_port).unwrap();
    let mt_pusher = Arc::new(Mutex::new(pusher));

    let (eval_tx, eval_rx) = unbounded();

    let sub_port_clone = sub_port.clone();
    thread::spawn(move || {
        inference_loop(eval_rx, sub_port_clone);
    });

    let redis_url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string());
    let logger: Arc<dyn crate::telemetry::GameLogger> = Arc::new(crate::telemetry::RedisLogger::new(&redis_url));

    let num_threads = 120;
    println!("⚔️ Spawning {} Native MCTS Worker Threads...", num_threads);
    let mut handles = Vec::new();
    for _ in 0..num_threads {
        let t_cfg = Arc::clone(&cfg);
        let t_tx = eval_tx.clone();
        let t_push = Arc::clone(&mt_pusher);
        let t_logger = Arc::clone(&logger);
        handles.push(thread::spawn(move || {
            game_loop(t_cfg, t_tx, t_push, t_logger);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}
