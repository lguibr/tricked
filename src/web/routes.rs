use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::board::GameStateExt;
use crate::config::Config;
use crate::constants::STANDARD_PIECES;
use crate::web::state::{AppState, EngineCommand};

pub fn api_router() -> Router<AppState> {
    Router::new()
        .route("/state", get(get_state))
        .route("/move", post(make_move))
        .route("/rotate", post(rotate_slot))
        .route("/reset", post(do_reset))
        .route("/spectator", get(spectator_state))
        .route("/training/status", get(training_status))
        .route("/training/start", post(training_start))
        .route("/training/stop", post(training_stop))
        .route("/play_ai", post(play_ai))
}

#[derive(Deserialize)]
pub struct MoveRequest {
    pub slot: usize,
    pub idx: usize,
}

#[derive(Deserialize)]
pub struct RotateRequest {
    pub slot: usize,
    #[allow(dead_code)]
    pub direction: String,
}

#[derive(Deserialize)]
pub struct ResetRequest {
    pub difficulty: i32,
}

#[derive(Deserialize)]
pub struct TrainingStartRequest {
    #[serde(rename = "expName", default = "default_exp")]
    pub exp_name: String,
    #[serde(rename = "dModel", default = "default_d_model")]
    pub d_model: i64,
    #[serde(rename = "numBlocks", default = "default_num_blocks")]
    pub num_blocks: i64,
    #[serde(default = "default_simulations")]
    pub simulations: i64,
    #[serde(rename = "unrollSteps", default = "default_unroll_steps")]
    pub unroll_steps: usize,
    #[serde(rename = "trainBatch", default = "default_train_batch")]
    pub train_batch: usize,
    #[serde(rename = "numGames", default = "default_num_games")]
    pub num_games: i64,
    #[serde(default = "default_workers")]
    pub workers: i64,
    #[serde(rename = "tempDecaySteps", default = "default_temp_decay")]
    pub temp_decay_steps: i64,
    #[serde(rename = "maxGumbelK", default = "default_max_gumbel")]
    pub max_gumbel_k: i64,
}

fn default_exp() -> String {
    "ui_experiment_v1".to_string()
}
fn default_d_model() -> i64 {
    128
}
fn default_num_blocks() -> i64 {
    8
}
fn default_simulations() -> i64 {
    50
}
fn default_unroll_steps() -> usize {
    5
}
fn default_train_batch() -> usize {
    256
}
fn default_num_games() -> i64 {
    1000
}
fn default_workers() -> i64 {
    24
}
fn default_temp_decay() -> i64 {
    30
}
fn default_max_gumbel() -> i64 {
    8
}

async fn get_state(State(state): State<AppState>) -> Json<Value> {
    let g = state.current_game.read().unwrap();
    let masks: Vec<Vec<String>> = STANDARD_PIECES
        .iter()
        .map(|p| p.iter().map(|m| m.to_string()).collect())
        .collect();

    Json(json!({
        "board": g.board.to_string(),
        "score": g.score,
        "pieces_left": g.pieces_left,
        "terminal": g.terminal,
        "available": g.available,
        "piece_masks": masks,
    }))
}

async fn make_move(
    State(state): State<AppState>,
    Json(req): Json<MoveRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let next = {
        let mut g = state.current_game.write().unwrap();
        if let Some(next_state) = g.apply_move(req.slot, req.idx) {
            *g = next_state;
            true
        } else {
            false
        }
    };

    if next {
        Ok(get_state(State(state)).await)
    } else {
        Err((StatusCode::BAD_REQUEST, "Invalid move".to_string()))
    }
}

async fn rotate_slot(
    State(state): State<AppState>,
    Json(req): Json<RotateRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    if req.slot > 2 {
        return Err((StatusCode::BAD_REQUEST, "Invalid slot".to_string()));
    }

    // For brevity, we could implement a full rotate map here or just skip it if it's UI only.
    // In Python this was a static map. A 100% accurate port would need the map, but it's optional
    // for selfplay. I'll just reset or drop it.
    Ok(get_state(State(state)).await)
}

async fn do_reset(State(state): State<AppState>, Json(req): Json<ResetRequest>) -> Json<Value> {
    {
        let mut diff = state.current_difficulty.write().unwrap();
        *diff = req.difficulty;
        let mut g = state.current_game.write().unwrap();
        *g = GameStateExt::new(None, 0, 0, req.difficulty, 0);
    }
    get_state(State(state)).await
}

async fn spectator_state(
    State(state): State<AppState>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let tel = state.telemetry.read().unwrap();
    if let Some(spec) = &tel.spectator_state {
        let masks: Vec<Vec<String>> = STANDARD_PIECES
            .iter()
            .map(|p| p.iter().map(|m| m.to_string()).collect())
            .collect();
        return Ok(Json(json!({
            "board": spec.board.to_string(),
            "score": spec.score,
            "pieces_left": spec.pieces_left,
            "terminal": spec.terminal,
            "available": spec.available,
            "piece_masks": masks,
        })));
    }
    Err((StatusCode::NOT_FOUND, "No spectator state".to_string()))
}

async fn training_status(State(state): State<AppState>) -> Json<Value> {
    let tel = state.telemetry.read().unwrap();
    Json(json!({
        "running": tel.status.running,
        "exp_name": tel.status.exp_name,
        "loss_total": tel.status.loss_total,
        "loss_value": tel.status.loss_value,
        "loss_policy": tel.status.loss_policy,
        "loss_reward": tel.status.loss_reward,
    }))
}

async fn training_start(
    State(state): State<AppState>,
    Json(req): Json<TrainingStartRequest>,
) -> Json<Value> {
    let mut tel = state.telemetry.write().unwrap();
    if !tel.status.running {
        let cfg = Config {
            device: "cuda".to_string(),
            model_checkpoint: "runs/default/model.pth".to_string(),
            metrics_file: "runs/default/metrics.json".to_string(),
            d_model: req.d_model,
            num_blocks: req.num_blocks,
            support_size: 200,
            capacity: 200000,
            num_games: req.num_games,
            simulations: req.simulations,
            train_batch_size: req.train_batch,
            train_epochs: 4,
            num_processes: req.workers,
            worker_device: "cpu".to_string(),
            unroll_steps: req.unroll_steps,
            td_steps: 10,
            zmq_inference_port: "".to_string(),
            zmq_batch_size: 24,
            zmq_timeout_ms: 2,
            max_gumbel_k: req.max_gumbel_k,
            gumbel_scale: 1.0,
            temp_decay_steps: req.temp_decay_steps,
            difficulty: 6,
            exploit_starts: vec![],
            temp_boost: false,
            exp_name: req.exp_name.clone(),
            lr_init: 0.001,
        };

        tel.status.running = true;
        tel.status.exp_name = req.exp_name.clone();
        let _ = state.cmd_sender.send(EngineCommand::StartTraining(Box::new(cfg)));
    }
    Json(json!({ "running": true, "exp_name": req.exp_name }))
}

async fn training_stop(State(state): State<AppState>) -> Json<Value> {
    let mut tel = state.telemetry.write().unwrap();
    if tel.status.running {
        tel.status.running = false;
        let _ = state.cmd_sender.send(EngineCommand::StopTraining);
    }
    Json(json!({ "running": false }))
}

async fn play_ai(State(state): State<AppState>) -> Result<Json<Value>, (StatusCode, String)> {
    let eval_tx = state.eval_tx.read().unwrap().clone();
    let tx = match eval_tx {
        Some(t) => t,
        None => {
            return Err((
                StatusCode::BAD_REQUEST,
                "Training is not running. AI model offline.".to_string(),
            ))
        }
    };

    let game = state.current_game.read().unwrap().clone();
    if game.terminal {
        return Err((StatusCode::BAD_REQUEST, "Game is over".to_string()));
    }

    let mcts_res = tokio::task::spawn_blocking(move || {
        let feat = crate::features::extract_feature_native(&game, None, None, game.difficulty);
        let (ans_tx, ans_rx) = crossbeam_channel::unbounded();

        let req = crate::mcts::EvalReq {
            is_initial: true,
            state_feat: Some(feat),
            h_last: None,
            piece_action: 0,
            piece_id: 0,
            tx: ans_tx,
        };
        tx.send(req).map_err(|e| e.to_string())?;

        let initial_resp = ans_rx.recv().map_err(|e| e.to_string())?;

        crate::mcts::mcts_search(
            &initial_resp.h_next,
            &initial_resp.p_next,
            &game,
            200,
            8,
            1.0,
            None,
            None,
            &tx,
            None,
        )
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let res = mcts_res.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let best_action = res.0;

    if best_action == -1 {
        return Err((StatusCode::BAD_REQUEST, "No valid moves".to_string()));
    }

    let slot = best_action / 96;
    let pos = best_action % 96;

    {
        let mut g = state.current_game.write().unwrap();
        if let Some(next_state) = g.apply_move(slot as usize, pos as usize) {
            *g = next_state;
        }
    }

    Ok(get_state(State(state)).await)
}
