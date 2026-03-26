use crossbeam_channel::{unbounded, Receiver, Sender};
use rand::Rng;
use std::sync::{Arc, Mutex, RwLock};
use tch::{Device, Kind, Tensor};

use crate::board::GameStateExt;
use crate::buffer::ReplayBuffer;
use crate::config::Config;
use crate::features::extract_feature_native;
use crate::mcts::{mcts_search, EvalReq, EvalResp};
use crate::network::MuZeroNet;
use crate::web::TelemetryStore;

pub fn inference_loop(
    rx: Receiver<EvalReq>,
    shared_model: Arc<Mutex<MuZeroNet>>,
    d_model: i64,
    device: Device,
) {
    loop {
        let first_req = match rx.recv_timeout(std::time::Duration::from_millis(10)) {
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
            tch::autocast(true, || {
                let m = shared_model.lock().unwrap();

                if !initial_reqs.is_empty() {
                    let s_tensors: Vec<Tensor> = initial_reqs
                        .iter()
                        .map(|r| Tensor::from_slice(r.state_feat.as_ref().unwrap()).view([20, 96]))
                        .collect();

                    let s_batch = Tensor::stack(&s_tensors, 0)
                        .to_device(device)
                        .to_kind(Kind::Float);

                    let (h_next_batch, value_batch, policy_batch, _) =
                        m.initial_inference(&s_batch);

                    let h_next_cpu = h_next_batch.to_device(Device::Cpu).to_kind(Kind::Float);
                    let value_cpu = value_batch.to_device(Device::Cpu).to_kind(Kind::Float);
                    let policy_cpu = policy_batch.to_device(Device::Cpu).to_kind(Kind::Float);

                    let h_next_f32: Vec<f32> =
                        h_next_cpu.view([-1i64]).try_into().unwrap_or_default();
                    let value_f32: Vec<f32> =
                        value_cpu.view([-1i64]).try_into().unwrap_or_default();
                    let policy_f32: Vec<f32> =
                        policy_cpu.view([-1i64]).try_into().unwrap_or_default();

                    let h_size = (d_model as usize) * 96;
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

                    let a_actions: Vec<i64> =
                        recurrent_reqs.iter().map(|r| r.piece_action).collect();
                    let p_ids: Vec<i64> = recurrent_reqs.iter().map(|r| r.piece_id).collect();

                    let h_batch = Tensor::stack(&h_tensors, 0)
                        .to_device(device)
                        .to_kind(Kind::Float);
                    let a_batch = Tensor::from_slice(&a_actions).to_device(device);
                    let p_batch = Tensor::from_slice(&p_ids).to_device(device);

                    let (h_next_batch, reward_batch, value_batch, policy_batch, _) =
                        m.recurrent_inference(&h_batch, &a_batch, &p_batch);

                    let h_next_cpu = h_next_batch.to_device(Device::Cpu).to_kind(Kind::Float);
                    let reward_cpu = reward_batch.to_device(Device::Cpu).to_kind(Kind::Float);
                    let value_cpu = value_batch.to_device(Device::Cpu).to_kind(Kind::Float);
                    let policy_cpu = policy_batch.to_device(Device::Cpu).to_kind(Kind::Float);

                    let h_next_f32: Vec<f32> =
                        h_next_cpu.view([-1i64]).try_into().unwrap_or_default();
                    let reward_f32: Vec<f32> =
                        reward_cpu.view([-1i64]).try_into().unwrap_or_default();
                    let value_f32: Vec<f32> =
                        value_cpu.view([-1i64]).try_into().unwrap_or_default();
                    let policy_f32: Vec<f32> =
                        policy_cpu.view([-1i64]).try_into().unwrap_or_default();

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
        });
    }
}

pub fn game_loop(
    cfg: Arc<Config>,
    eval_tx: Sender<EvalReq>,
    buffer: Arc<ReplayBuffer>,
    telemetry: Arc<RwLock<TelemetryStore>>,
) {
    let mut rng = rand::thread_rng();

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
            if eval_tx.send(EvalReq {
                is_initial: true,
                state_feat: Some(feat.clone()),
                h_last: None,
                piece_action: 0,
                piece_id: 0,
                tx,
            }).is_err() {
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
                cfg.simulations as usize,
                cfg.max_gumbel_k as usize,
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

            // Update Telemetry
            if rng.gen_ratio(1, 10) {
                if let Ok(mut tel) = telemetry.write() {
                    tel.spectator_state = Some(state.clone());
                }
            }

            // Dynamic temperature
            let temp = if step < (cfg.temp_decay_steps / 2) as usize {
                1.0
            } else if step < cfg.temp_decay_steps as usize {
                0.5
            } else {
                0.1
            };

            let mut policy_target = vec![0.0f32; 288];
            let mut sum_probs = 0.0;
            for (act, visits) in &action_visits {
                let prob = (*visits as f64).powf(1.0 / (temp + 1e-8)) as f32;
                policy_target[*act as usize] = prob;
                sum_probs += prob;
            }
            if sum_probs > 0.0 {
                for p in policy_target.iter_mut() {
                    *p /= sum_probs;
                }
            } else {
                let uniform = 1.0 / (action_visits.len() as f32);
                for act in action_visits.keys() {
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
                    println!("Warning: invalid move chosen in Gumbel MCTS, skipping.");
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

            ep_boards.push([
                (state.board & 0xFFFFFFFFFFFFFFFF) as u64,
                (state.board >> 64) as u64,
            ]);
            ep_available.push([state.available[0], state.available[1], state.available[2]]);
            ep_actions.push(piece_action as i64);
            ep_p_ids.push(piece_id as i64);
            ep_rewards.push(reward);

            let mut p_arr = [0.0f32; 288];
            p_arr.copy_from_slice(&policy_target);
            ep_policies.push(p_arr);

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
            buffer.add_game(
                cfg.difficulty,
                state.score as f32,
                &ep_boards,
                &ep_available,
                &ep_actions,
                &ep_p_ids,
                &ep_rewards,
                &ep_policies,
                &ep_values,
            );
        }
    }
}
