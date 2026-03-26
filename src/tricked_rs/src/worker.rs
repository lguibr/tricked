use crossbeam_channel::{unbounded, Receiver, Sender};
use pyo3::prelude::*;
use rand::Rng;
use redis::Commands;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use tch::{CModule, Device, Tensor};

use crate::board::GameStateExt;
use crate::features::extract_feature_native;
use crate::mcts::{mcts_search, EvalReq, EvalResp};

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
    pub redis_url: String,
}

fn inference_loop(rx: Receiver<EvalReq>, shared_model: Arc<RwLock<Option<tch::CModule>>>, d_model: i64) {
    let device = Device::Cuda(0);

    loop {
        if shared_model.read().unwrap().is_none() {
            std::thread::sleep(std::time::Duration::from_millis(50));
            continue;
        }

        let first_req = match rx.recv_timeout(std::time::Duration::from_millis(10)) {
            Ok(req) => req,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        };

        let m_guard = shared_model.read().unwrap();

        let mut reqs = vec![first_req];
        while reqs.len() < 1024 {
            if let Ok(req) = rx.try_recv() {
                reqs.push(req);
            } else {
                break;
            }
        }

        let m = m_guard.as_ref().unwrap();

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

                // FP16 Cast before sending to GPU (Task 1.3)
                let s_batch = Tensor::stack(&s_tensors, 0)
                    .to_device(device)
                    .to_kind(tch::Kind::Half);

                let outputs = match m.method_is("initial_inference_jit", &[tch::IValue::from(s_batch)]) {
                    Ok(out) => out,
                    Err(e) => {
                        println!("🔥🔥🔥 FATAL JIT ERROR in initial_inference_jit: {:?}", e);
                        panic!("initial_inference_jit failed: {:?}", e);
                    }
                };
                let tuple = if let tch::IValue::Tuple(t) = outputs {
                    t
                } else {
                    panic!("Expected tuple")
                };

                // Cast outputs back to FP32 before extracting
                let h_next_batch = if let tch::IValue::Tensor(t) = &tuple[0] {
                    t.copy()
                } else {
                    panic!()
                }
                .to_kind(tch::Kind::Float)
                .to_device(Device::Cpu)
                .contiguous();
                let value_batch = if let tch::IValue::Tensor(t) = &tuple[1] {
                    t.copy()
                } else {
                    panic!()
                }
                .to_kind(tch::Kind::Float)
                .to_device(Device::Cpu)
                .contiguous();
                let policy_batch = if let tch::IValue::Tensor(t) = &tuple[2] {
                    t.copy()
                } else {
                    panic!()
                }
                .to_kind(tch::Kind::Float)
                .to_device(Device::Cpu)
                .contiguous();

                let h_next_f32: Vec<f32> = h_next_batch.view([-1]).try_into().unwrap_or_default();
                let value_f32: Vec<f32> = value_batch.view([-1]).try_into().unwrap_or_default();
                let policy_f32: Vec<f32> = policy_batch.view([-1]).try_into().unwrap_or_default();

                let h_size = (d_model as usize) * 96; // Dynamic d_model mapping
                let p_size = 288;

                if h_next_f32.len() == initial_reqs.len() * h_size {
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
            }

            if !recurrent_reqs.is_empty() {
                let h_tensors: Vec<Tensor> = recurrent_reqs
                    .iter()
                    .map(|r| {
                        let h = r.h_last.as_ref().unwrap();
                        let channels = h.len() / 96; // Space is artificially padded to 96
                        Tensor::from_slice(h).view([channels as i64, 96])
                    })
                    .collect();

                let a_actions: Vec<i64> = recurrent_reqs
                    .iter()
                    .map(|r| r.piece_action)
                    .collect();
                let p_ids: Vec<i64> = recurrent_reqs
                    .iter()
                    .map(|r| r.piece_id)
                    .collect();

                let h_batch = Tensor::stack(&h_tensors, 0)
                    .to_device(device)
                    .to_kind(tch::Kind::Half);
                let a_batch = Tensor::from_slice(&a_actions).to_device(device);
                let p_batch = Tensor::from_slice(&p_ids).to_device(device);

                let outputs = match m.forward_is(&[
                    tch::IValue::from(h_batch),
                    tch::IValue::from(a_batch),
                    tch::IValue::from(p_batch),
                ]) {
                    Ok(out) => out,
                    Err(e) => {
                        println!("🔥🔥🔥 FATAL JIT ERROR in recurrent_inference_jit: {:?}", e);
                        panic!("recurrent_inference_jit failed: {:?}", e);
                    }
                };

                let tuple = if let tch::IValue::Tuple(t) = outputs {
                    t
                } else {
                    panic!("Expected tuple")
                };

                // Cast outputs back to FP32
                let h_next_batch = if let tch::IValue::Tensor(t) = &tuple[0] {
                    t.copy()
                } else {
                    panic!()
                }
                .to_kind(tch::Kind::Float)
                .to_device(Device::Cpu)
                .contiguous();
                let reward_batch = if let tch::IValue::Tensor(t) = &tuple[1] {
                    t.copy()
                } else {
                    panic!()
                }
                .to_kind(tch::Kind::Float)
                .to_device(Device::Cpu)
                .contiguous();
                let value_batch = if let tch::IValue::Tensor(t) = &tuple[2] {
                    t.copy()
                } else {
                    panic!()
                }
                .to_kind(tch::Kind::Float)
                .to_device(Device::Cpu)
                .contiguous();
                let policy_batch = if let tch::IValue::Tensor(t) = &tuple[3] {
                    t.copy()
                } else {
                    panic!()
                }
                .to_kind(tch::Kind::Float)
                .to_device(Device::Cpu)
                .contiguous();

                let h_next_f32: Vec<f32> = h_next_batch.view([-1]).try_into().unwrap_or_default();
                let reward_f32: Vec<f32> = reward_batch.view([-1]).try_into().unwrap_or_default();
                let value_f32: Vec<f32> = value_batch.view([-1]).try_into().unwrap_or_default();
                let policy_f32: Vec<f32> = policy_batch.view([-1]).try_into().unwrap_or_default();

                let h_size = (d_model as usize) * 96;
                let p_size = 288;

                if h_next_f32.len() == recurrent_reqs.len() * h_size {
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
            }
        });
    }
}

fn fetch_dynamic_temp(redis_client: &redis::Client, default_steps: usize, step: usize) -> f32 {
    if let Ok(mut con) = redis_client.get_connection() {
        if let Ok(temp_override) = con.get::<_, f32>("tricked_gumbel_temp") {
            return temp_override;
        }
    }
    // Fallback logic
    if step < (default_steps / 2) {
        1.0
    } else if step < default_steps {
        0.5
    } else {
        0.1
    }
}

fn game_loop(
    cfg: Arc<AppConfig>,
    eval_tx: Sender<EvalReq>,
    pusher: Arc<Mutex<zmq::Socket>>,
    logger: Arc<dyn crate::telemetry::GameLogger>,
    redis_client: Arc<redis::Client>,
) {
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
            if let Err(_) = eval_tx.send(EvalReq {
                is_initial: true,
                state_feat: Some(feat.clone()),
                h_last: None,
                piece_action: 0,
                piece_id: 0,
                tx,
            }) {
                return;
            }

            let initial_resp = match rx.recv() {
                Ok(res) => res,
                Err(_) => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    break;
                }
            };

            let mcts_res = match mcts_search(
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
            ) {
                Ok(res) => res,
                Err(_) => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    break;
                }
            };

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

            // Task 2.3: Dynamic Temperature Scheduling
            let temp = fetch_dynamic_temp(&redis_client, cfg.temp_decay_steps, step);

            let mut policy_target = vec![0.0f32; 288];
            let mut sum_probs = 0.0;
            for (act, visits) in &action_visits {
                let prob = (*visits as f64).powf(1.0 / (temp as f64 + 1e-8)) as f32;
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

            let next_state = match state.apply_move(slot as usize, pos as usize) {
                Some(s) => s,
                None => {
                    println!("Warning: invalid move chosen in Gumbel MCTS, skipping iteration.");
                    break;
                }
            };
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

            if let Err(_) = pusher.lock().unwrap().send(&payload, 0) {
                return;
            }
        }
    }
}

#[pyfunction]
pub fn run_self_play_worker(
    py: Python,
    d_model: i64,
    num_blocks: i64,
    support_size: i64,
    simulations: usize,
    max_gumbel_k: usize,
    gumbel_scale: f32,
    difficulty: i32,
    temp_decay_steps: usize,
    push_port: String,
    redis_url: String, // Replaced sub_port (Task 3.1)
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
        redis_url: redis_url.clone(),
    });

    println!("🔥 Starting Tricked Native Self-Play Engine (via PyO3)");

    let ctx = zmq::Context::new();
    let pusher = ctx.socket(zmq::PUSH).unwrap();
    pusher.connect(&cfg.push_port).unwrap();
    let mt_pusher = Arc::new(Mutex::new(pusher));

    let (eval_tx, eval_rx) = unbounded();

    let redis_client = Arc::new(redis::Client::open(cfg.redis_url.clone()).unwrap());

    // Central Model Updater Thread fetching from Redis
    let shared_model = Arc::new(RwLock::new(None));
    let updater_model = Arc::clone(&shared_model);
    let updater_redis = Arc::clone(&redis_client);

    thread::spawn(move || {
        let device = Device::Cuda(0);
        let mut con = updater_redis.get_connection().unwrap();
        let mut pubsub = con.as_pubsub();
        let _ = pubsub.subscribe("model_updates");

        // Decompression helper
        let load_model = |bytes: Vec<u8>| {
            use std::io::Read;
            // Decompress LZ4 frames generated by Python's lz4.frame
            let mut decoder = lz4_flex::frame::FrameDecoder::new(bytes.as_slice());
            let mut decompressed = Vec::new();
            if let Err(e) = decoder.read_to_end(&mut decompressed) {
                println!("❌ Failed to decompress JIT payload: {}", e);
                return;
            }
            match CModule::load_data_on_device(&mut decompressed.as_slice(), device) {
                Ok(new_model) => {
                    println!("✅ Successfully loaded JIT model native CModule (FP16) on GPU! Size: {} bytes", decompressed.len());
                    *updater_model.write().unwrap() = Some(new_model);
                }
                Err(e) => {
                    println!("❌ Failed to load decompressed JIT payload onto device. Size: {} bytes. Error: {:?}", decompressed.len(), e);
                }
            }
        };

        // Try initial fetch
        let mut dl_con = updater_redis.get_connection().unwrap();
        if let Ok(bytes) = redis::cmd("GET")
            .arg("model_weights")
            .query::<Vec<u8>>(&mut dl_con)
        {
            println!("🔄 Loading initial compressed model from Redis Parameter Server... (FP16)");
            load_model(bytes);
        }

        loop {
            if let Ok(_) = pubsub.get_message() {
                if let Ok(bytes) = redis::cmd("GET")
                    .arg("model_weights")
                    .query::<Vec<u8>>(&mut dl_con)
                {
                    println!(
                        "🔄 Received compressed JIT model via Redis. Hot reloading in FP16..."
                    );
                    load_model(bytes);
                }
            }
        }
    });

    // Task 2.1: Multi-Stream LibTorch Inference
    let num_infer_threads = 4;
    println!(
        "⚔️ Spawning {} Native LibTorch Inference Threads...",
        num_infer_threads
    );
    for _ in 0..num_infer_threads {
        let t_rx = eval_rx.clone();
        let t_model = Arc::clone(&shared_model);
        thread::spawn(move || {
            inference_loop(t_rx, t_model, d_model);
        });
    }

    let logger: Arc<dyn crate::telemetry::GameLogger> =
        Arc::new(crate::telemetry::RedisLogger::new(&cfg.redis_url));

    let num_threads = 120;
    println!("⚔️ Spawning {} Native MCTS Worker Threads...", num_threads);
    let mut handles = Vec::new();
    for _ in 0..num_threads {
        let t_cfg = Arc::clone(&cfg);
        let t_tx = eval_tx.clone();
        let t_push = Arc::clone(&mt_pusher);
        let t_logger = Arc::clone(&logger);
        let t_redis = Arc::clone(&redis_client);
        handles.push(thread::spawn(move || {
            game_loop(t_cfg, t_tx, t_push, t_logger, t_redis);
        }));
    }

    py.allow_threads(|| {
        for h in handles {
            h.join().unwrap();
        }
    });
}
