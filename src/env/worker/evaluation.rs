use crate::config::Config;
use crate::core::board::GameStateExt;
use crate::net::MuZeroNet;
use std::sync::Arc;
use tch::{nn, Device, Kind, Tensor};

#[derive(serde::Serialize, Clone)]
pub struct EvaluationStepData {
    pub board_low: u64,
    pub board_high: u64,
    pub score: i32,
    pub lines_cleared: i32,
    pub available: [i32; 3],
    pub terminal: bool,
    pub pieces_left: i32,
    pub selected_piece_id: i32,
    pub selected_action: i32,
}

pub fn run_evaluation(
    config: Config,
    checkpoint_path: String,
    external_abort: Arc<std::sync::atomic::AtomicBool>,
    on_step: Box<dyn Fn(EvaluationStepData) + Send + Sync>,
) {
    println!("🚀 Starting Isolated Evaluation Loop using checkpoint: {}", checkpoint_path);

    let computation_device = if config.hardware.device.starts_with("cuda") && tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    let mut inference_var_store = nn::VarStore::new(computation_device);
    let neural_model = MuZeroNet::new(
        &inference_var_store.root(),
        config.architecture.hidden_dimension_size,
        config.architecture.num_blocks,
        config.architecture.value_support_size,
        config.architecture.reward_support_size,
        config.architecture.spatial_channel_count,
        config.architecture.hole_predictor_dim,
    );

    let mut cmodule_inference: Option<tch::CModule> = None;

    if std::path::Path::new(&checkpoint_path).exists() {
        if checkpoint_path.ends_with(".pt") {
            println!("🚀 Loading TorchScript checkpoint for evaluation...");
            cmodule_inference = tch::CModule::load_on_device(&checkpoint_path, computation_device).ok();
        } else {
            println!("🚀 Loading Native Rust weights for evaluation...");
            let _ = inference_var_store.load(&checkpoint_path);
        }
    } else {
        println!("⚠️ Checkpoint not found! Evaluating with random initialization: {}", checkpoint_path);
    }

    let initial_difficulty = if config.environment.difficulty < 3 {
        3
    } else {
        config.environment.difficulty
    };

    let mut active_game_state = GameStateExt::new(None, 0, 0, initial_difficulty, 0);
    let mut board_history = vec![
        active_game_state.board_bitmask_u128,
        active_game_state.board_bitmask_u128,
    ];
    let mut action_history: Vec<i32> = vec![];

    if active_game_state.pieces_left == 0 {
        active_game_state.refill_tray();
    }

    on_step(EvaluationStepData {
        board_low: (active_game_state.board_bitmask_u128 & 0xFFFFFFFFFFFFFFFF) as u64,
        board_high: (active_game_state.board_bitmask_u128 >> 64) as u64,
        score: active_game_state.score,
        lines_cleared: active_game_state.total_lines_cleared,
        available: active_game_state.available,
        terminal: active_game_state.terminal,
        pieces_left: active_game_state.pieces_left,
        selected_piece_id: -1,
        selected_action: -1,
    });

    std::thread::sleep(std::time::Duration::from_millis(1500));

    loop {
        if !external_abort.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }

        if active_game_state.terminal {
            println!("🏁 Evaluation Game Finished. Score: {}", active_game_state.score);
            active_game_state = GameStateExt::new(None, 0, 0, initial_difficulty, 0);
            board_history = vec![
                active_game_state.board_bitmask_u128,
                active_game_state.board_bitmask_u128,
            ];
            action_history.clear();
            if active_game_state.pieces_left == 0 {
                active_game_state.refill_tray();
            }
            std::thread::sleep(std::time::Duration::from_millis(2000));
            continue;
        }

        if active_game_state.pieces_left == 0 {
            active_game_state.refill_tray();
        }

        let board_history_array: [u128; 8] = {
            let mut arr = [0; 8];
            for (i, &b) in board_history.iter().rev().take(8).enumerate() {
                arr[i] = b;
            }
            arr
        };
        let action_history_array: [i32; 4] = {
            let mut arr = [0; 4];
            for (i, &a) in action_history.iter().rev().take(4).enumerate() {
                arr[i] = a as i32;
            }
            arr
        };

        // Network Forward Pass (Raw NN Greedy)
        let mut selected_action: i32 = -1;

        tch::no_grad(|| {
            let boards_tensor = Tensor::from_slice(&[
                active_game_state.board_bitmask_u128 as i64,
                (active_game_state.board_bitmask_u128 >> 64) as i64,
            ]).view([1, 2]);

            let avail_tensor = Tensor::from_slice(&active_game_state.available).view([1, 3]);
            let hist_data: Vec<i64> = board_history_array.iter().flat_map(|&b| vec![b as i64, (b >> 64) as i64]).collect();
            let hist_tensor = Tensor::from_slice(&hist_data).view([1, 16]);
            let acts_tensor = Tensor::from_slice(&action_history_array).view([1, 4]);
            let diff_tensor = Tensor::from_slice(&[initial_difficulty]).view([1, 1]);

            let boards = boards_tensor.to_device(computation_device);
            let avail = avail_tensor.to_device(computation_device);
            let hist = hist_tensor.to_device(computation_device);
            let acts = acts_tensor.to_device(computation_device);
            let diff = diff_tensor.to_device(computation_device);

            let state_batch = if computation_device.is_cuda() {
                neural_model.extract_initial_features(&boards, &avail, &hist, &acts, &diff)
            } else {
                Tensor::zeros(
                    [1, config.architecture.spatial_channel_count as i64, 8, 16],
                    (Kind::Float, computation_device),
                )
            };

            let policy_batch = if let Some(cmod) = &cmodule_inference {
                let ivalue = cmod.method_is("initial_inference", &[tch::IValue::Tensor(state_batch)]).unwrap();
                if let tch::IValue::Tuple(mut tup) = ivalue {
                    if let tch::IValue::Tensor(p) = tup.remove(2) {
                        p
                    } else { unreachable!() }
                } else { unreachable!() }
            } else {
                let (_, _, p, _) = neural_model.initial_inference(&state_batch);
                p
            };

            let policy_cpu = policy_batch.to_device(Device::Cpu).to_kind(Kind::Float);
            let policy_probs: Vec<f32> = policy_cpu.reshape([-1]).try_into().unwrap_or_default();

            // Mask invalid actions
            let mut highest_prob = -1.0;

            for slot in 0..3 {
                let pid = active_game_state.available[slot];
                if pid == -1 { continue; }
                for cell in 0..96 {
                    let piece_mask = crate::core::constants::STANDARD_PIECES[pid as usize][cell];
                    if piece_mask != 0 && (active_game_state.board_bitmask_u128 & piece_mask) == 0 {
                        let action_idx = (slot * 96) + cell;
                        let prob = policy_probs[action_idx];
                        if prob > highest_prob {
                            highest_prob = prob;
                            selected_action = action_idx as i32;
                        }
                    }
                }
            }
        });

        if selected_action == -1 {
            println!("⚠️ NO VALID ACTIONS IDENTIFIED BY NETWORK! Assuming terminal state fallthrough.");
            active_game_state.terminal = true;
        } else {
            let slot = (selected_action / 96) as usize;
            let cell = (selected_action % 96) as usize;
            let pid = active_game_state.available[slot];

            if let Some(next_state) = active_game_state.apply_move(slot, cell) {
                board_history.push(active_game_state.board_bitmask_u128);
                if board_history.len() > 8 { board_history.remove(0); }
                action_history.push((pid * 96) + cell as i32);
                
                // Emitting the "PRE" state + the action first
                on_step(EvaluationStepData {
                    board_low: (active_game_state.board_bitmask_u128 & 0xFFFFFFFFFFFFFFFF) as u64,
                    board_high: (active_game_state.board_bitmask_u128 >> 64) as u64,
                    score: active_game_state.score,
                    lines_cleared: active_game_state.total_lines_cleared,
                    available: active_game_state.available,
                    terminal: active_game_state.terminal,
                    pieces_left: active_game_state.pieces_left,
                    selected_piece_id: pid,
                    selected_action: selected_action,
                });
                
                std::thread::sleep(std::time::Duration::from_millis(500));
                
                active_game_state = next_state;

                // Emitting the resulting state
                on_step(EvaluationStepData {
                    board_low: (active_game_state.board_bitmask_u128 & 0xFFFFFFFFFFFFFFFF) as u64,
                    board_high: (active_game_state.board_bitmask_u128 >> 64) as u64,
                    score: active_game_state.score,
                    lines_cleared: active_game_state.total_lines_cleared,
                    available: active_game_state.available,
                    terminal: active_game_state.terminal,
                    pieces_left: active_game_state.pieces_left,
                    selected_piece_id: -1,
                    selected_action: -1,
                });

            } else {
                active_game_state.terminal = true;
            }
        }
        
        std::thread::sleep(std::time::Duration::from_millis(300));
    }
}
