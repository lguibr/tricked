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
        .route("/games/:game_id", get(get_game_replay))
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
    #[serde(default = "default_exp")]
    pub experiment_name_identifier: String,
    #[serde(default = "default_hidden_dimension_size")]
    pub hidden_dimension_size: i64,
    #[serde(default = "default_num_blocks")]
    pub num_blocks: i64,
    #[serde(default = "default_simulations")]
    pub simulations: i64,
    #[serde(default = "default_unroll_steps")]
    pub unroll_steps: usize,
    #[serde(default = "default_train_batch")]
    pub train_batch_size: usize,
    #[serde(default = "default_workers")]
    pub num_processes: i64,
    #[serde(default = "default_temp_decay")]
    pub temp_decay_steps: i64,
    #[serde(default = "default_max_gumbel")]
    pub max_gumbel_k: i64,

    // UI Forge Extra Configs
    #[serde(default = "default_buffer_capacity_limit")]
    pub buffer_capacity_limit: usize,
    #[serde(default = "default_train_epochs")]
    pub train_epochs: i64,
    #[serde(default = "default_temporal_difference_steps")]
    pub temporal_difference_steps: usize,
    #[serde(default = "default_gumbel_scale")]
    pub gumbel_scale: f32,
    #[serde(default = "default_temp_boost")]
    pub temp_boost: bool,
    #[serde(default = "default_lr_init")]
    pub lr_init: f64,
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(default = "default_worker_device")]
    pub worker_device: String,
    #[serde(default = "default_zmq_batch_size")]
    pub zmq_batch_size: i64,
    #[serde(default = "default_zmq_timeout_ms")]
    pub zmq_timeout_ms: i64,
    #[serde(default = "default_support_size")]
    pub support_size: i64,
}

fn default_exp() -> String {
    "ui_experiment_v1".to_string()
}
fn default_hidden_dimension_size() -> i64 {
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
fn default_workers() -> i64 {
    256
}
fn default_temp_decay() -> i64 {
    30
}
fn default_max_gumbel() -> i64 {
    8
}
fn default_buffer_capacity_limit() -> usize {
    100_000
}
fn default_train_epochs() -> i64 {
    1
}
fn default_temporal_difference_steps() -> usize {
    5
}
fn default_gumbel_scale() -> f32 {
    1.0
}
fn default_temp_boost() -> bool {
    false
}
fn default_lr_init() -> f64 {
    0.001
}
fn default_device() -> String {
    "cuda".to_string()
}
fn default_worker_device() -> String {
    "cpu".to_string()
}
fn default_zmq_batch_size() -> i64 {
    16
}
fn default_zmq_timeout_ms() -> i64 {
    5
}
fn default_support_size() -> i64 {
    300
}

async fn get_state(State(application_state): State<AppState>) -> Json<Value> {
    let active_game_state = application_state.current_game.read().unwrap();
    let piece_binary_masks: Vec<Vec<String>> = STANDARD_PIECES
        .iter()
        .map(|piece_layout| {
            piece_layout
                .iter()
                .map(|bitboard_mask| bitboard_mask.to_string())
                .collect()
        })
        .collect();

    Json(json!({
        "board": active_game_state.board_bitmask_u128.to_string(),
        "score": active_game_state.score,
        "pieces_left": active_game_state.pieces_left,
        "terminal": active_game_state.terminal,
        "available": active_game_state.available,
        "piece_masks": piece_binary_masks,
    }))
}

async fn make_move(
    State(application_state): State<AppState>,
    Json(move_request): Json<MoveRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let move_success = {
        let mut active_game_state = application_state.current_game.write().unwrap();
        if let Some(next_state) = active_game_state.apply_move(move_request.slot, move_request.idx)
        {
            *active_game_state = next_state;
            true
        } else {
            false
        }
    };

    if move_success {
        Ok(get_state(State(application_state.clone())).await)
    } else {
        Err((StatusCode::BAD_REQUEST, "Invalid move".to_string()))
    }
}

async fn rotate_slot(
    State(application_state): State<AppState>,
    Json(rotate_request): Json<RotateRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    if rotate_request.slot > 2 {
        return Err((StatusCode::BAD_REQUEST, "Invalid slot".to_string()));
    }
    // Tiger Style observance: Unimplemented features must safely default to safe NO-OP rather than Panics or silently dropping required tasks.
    Ok(get_state(State(application_state.clone())).await)
}

async fn do_reset(
    State(application_state): State<AppState>,
    Json(reset_request): Json<ResetRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    {
        let mut active_game_state = application_state.current_game.write().unwrap();
        *active_game_state = GameStateExt::new(None, 0, 0, reset_request.difficulty, 0);

        let mut current_difficulty_state = application_state.current_difficulty.write().unwrap();
        *current_difficulty_state = reset_request.difficulty;
    }

    Ok(get_state(State(application_state.clone())).await)
}

async fn spectator_state(State(application_state): State<AppState>) -> Json<Value> {
    let shared_telemetry = application_state.telemetry.read().unwrap();

    let piece_binary_masks: Vec<Vec<String>> = STANDARD_PIECES
        .iter()
        .map(|piece_layout| {
            piece_layout
                .iter()
                .map(|bitboard_mask| bitboard_mask.to_string())
                .collect()
        })
        .collect();

    if let Some(spectator_metrics) = &shared_telemetry.spectator_state {
        Json(json!({
            "board": spectator_metrics.board_bitmask_u128.to_string(),
            "score": spectator_metrics.score,
            "pieces_left": spectator_metrics.pieces_left,
            "terminal": spectator_metrics.terminal,
            "available": spectator_metrics.available,
            "piece_masks": piece_binary_masks,
        }))
    } else {
        Json(json!({ "error": "No spectator state available" }))
    }
}

async fn training_status(State(application_state): State<AppState>) -> Json<Value> {
    let shared_telemetry = application_state.telemetry.read().unwrap();
    Json(json!({
        "running": shared_telemetry.status.running,
        "experiment_name_identifier": shared_telemetry.status.experiment_name_identifier,
        "loss_total": shared_telemetry.status.loss_total,
        "loss_value": shared_telemetry.status.loss_value,
        "loss_policy": shared_telemetry.status.loss_policy,
        "loss_reward": shared_telemetry.status.loss_reward,
        "games_played": shared_telemetry.status.games_played,
        "training_steps": shared_telemetry.status.training_steps,
        "top_games": shared_telemetry.top_games,
    }))
}

async fn training_start(
    State(application_state): State<AppState>,
    Json(start_request): Json<TrainingStartRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let mut shared_telemetry = application_state.telemetry.write().unwrap();
    if shared_telemetry.status.running {
        return Err((
            StatusCode::BAD_REQUEST,
            "Training already running".to_string(),
        ));
    }

    shared_telemetry.status.running = true;
    shared_telemetry.status.experiment_name_identifier =
        start_request.experiment_name_identifier.clone();

    let experiment_paths =
        crate::config::ExperimentPaths::new(&start_request.experiment_name_identifier);
    let exp_dir = &experiment_paths.base_directory;
    if std::path::Path::new(exp_dir).exists() {
        shared_telemetry.status.running = false;
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Experiment directory '{}' already exists. Please choose a unique experiment name.",
                exp_dir
            ),
        ));
    }
    std::fs::create_dir_all(exp_dir).unwrap_or(());

    let engine_configuration = Config {
        experiment_name_identifier: start_request.experiment_name_identifier.clone(),
        device: start_request.device.clone(),
        paths: experiment_paths.clone(),
        hidden_dimension_size: start_request.hidden_dimension_size,
        num_blocks: start_request.num_blocks,
        support_size: start_request.support_size,
        buffer_capacity_limit: start_request.buffer_capacity_limit,
        simulations: start_request.simulations,
        train_batch_size: start_request.train_batch_size,
        train_epochs: start_request.train_epochs,
        num_processes: start_request.num_processes,
        worker_device: start_request.worker_device.clone(),
        unroll_steps: start_request.unroll_steps,
        temporal_difference_steps: start_request.temporal_difference_steps,
        zmq_batch_size: start_request.zmq_batch_size,
        zmq_timeout_ms: start_request.zmq_timeout_ms,
        max_gumbel_k: start_request.max_gumbel_k,
        gumbel_scale: start_request.gumbel_scale,
        temp_decay_steps: start_request.temp_decay_steps,
        difficulty: 6,
        temp_boost: start_request.temp_boost,
        lr_init: start_request.lr_init,
    };

    let _ = application_state
        .cmd_sender
        .send(EngineCommand::StartTraining(Box::new(engine_configuration)));

    let payload = serde_json::json!({
        "type": "experiment_started",
        "experiment_name_identifier": start_request.experiment_name_identifier.clone()
    });

    if let Ok(client) = redis::Client::open("redis://127.0.0.1:6379/") {
        if let Ok(mut con) = client.get_connection() {
            let _: () = redis::cmd("SET")
                .arg("tricked_current_exp")
                .arg(&start_request.experiment_name_identifier)
                .query(&mut con)
                .unwrap_or(());

            let _: () = redis::cmd("PUBLISH")
                .arg("tricked_config")
                .arg(payload.to_string())
                .query(&mut con)
                .unwrap_or(());
        }
    }

    Ok(Json(json!({ "status": "started" })))
}

async fn training_stop(State(application_state): State<AppState>) -> Json<Value> {
    let mut shared_telemetry = application_state.telemetry.write().unwrap();
    if shared_telemetry.status.running {
        shared_telemetry.status.running = false;
        let _ = application_state
            .cmd_sender
            .send(EngineCommand::StopTraining);
    }
    Json(json!({ "status": "stopped" }))
}

use axum::extract::Path;

async fn get_game_replay(Path(game_id): Path<usize>) -> Result<Json<Value>, (StatusCode, String)> {
    let mut conn = redis::Client::open("redis://127.0.0.1:6379/")
        .unwrap()
        .get_connection()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let replay_data: String = redis::cmd("HGET")
        .arg("tricked_replays")
        .arg(game_id.to_string())
        .query(&mut conn)
        .map_err(|_| (StatusCode::NOT_FOUND, "Game not found".to_string()))?;

    let parsed: Value = serde_json::from_str(&replay_data)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(parsed))
}
