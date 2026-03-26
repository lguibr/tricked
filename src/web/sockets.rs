use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};

use serde_json::json;
use tokio::time::{interval, Duration};

use crate::constants::STANDARD_PIECES;
use crate::web::state::AppState;

pub fn ws_router() -> Router<AppState> {
    Router::new().route("/ws", get(ws_handler))
}

async fn ws_handler(
    websocket_upgrade: WebSocketUpgrade,
    State(application_state): State<AppState>,
) -> impl IntoResponse {
    websocket_upgrade.on_upgrade(|active_socket| handle_socket(active_socket, application_state))
}

async fn handle_socket(mut active_socket: WebSocket, application_state: AppState) {
    let mut broadcast_ticker = interval(Duration::from_millis(50));
    let mut last_tick = std::time::Instant::now();
    let mut last_games_count = 0;
    loop {
        tokio::select! {
            _ = broadcast_ticker.tick() => {
                let telemetry_payload = {
                    let mut shared_telemetry = application_state.telemetry.write().unwrap();
                    let elapsed = last_tick.elapsed().as_secs_f32();
                    if elapsed >= 1.0 {
                        let current_games = shared_telemetry.status.games_played;
                        let newly_finished_games = current_games.saturating_sub(last_games_count);
                        let instant_gps = newly_finished_games as f32 / elapsed;

                        let previous_gps = shared_telemetry.status.games_per_second;
                        if previous_gps == 0.0 {
                            shared_telemetry.status.games_per_second = instant_gps;
                        } else {
                            shared_telemetry.status.games_per_second = (0.2 * instant_gps) + (0.8 * previous_gps);
                        }

                        last_games_count = current_games;
                        last_tick = std::time::Instant::now();
                    }
                    let mut spectator_json_payload = serde_json::Value::Null;

                    if let Some(spectator_metrics) = &shared_telemetry.spectator_state {
                        let piece_binary_masks: Vec<Vec<String>> = STANDARD_PIECES
                            .iter()
                            .map(|piece_layout| {
                                piece_layout
                                    .iter()
                                    .map(|bitboard_mask| bitboard_mask.to_string())
                                    .collect()
                            })
                            .collect();

                        spectator_json_payload = json!({
                            "board": spectator_metrics.board.to_string(),
                            "score": spectator_metrics.score,
                            "pieces_left": spectator_metrics.pieces_left,
                            "terminal": spectator_metrics.terminal,
                            "available": spectator_metrics.available,
                            "piece_masks": piece_binary_masks,
                        });
                    }

                    let mut top_games_json = serde_json::Value::Array(Vec::new());
                    if !shared_telemetry.top_games.is_empty() {
                        top_games_json = serde_json::json!(shared_telemetry.top_games);
                    }

                    json!({
                        "type": "sync",
                        "spectator": spectator_json_payload,
                        "status": {
                            "running": shared_telemetry.status.running,
                            "loss_total": shared_telemetry.status.loss_total,
                            "games_per_second": shared_telemetry.status.games_per_second,
                            "top_games": top_games_json,
                        }
                    })
                };

                if active_socket
                    .send(Message::Text(telemetry_payload.to_string()))
                    .await
                    .is_err()
                {
                    break;
                }
            }
            msg = active_socket.recv() => {
                if let Some(Ok(Message::Close(_))) | None = msg {
                    break;
                }
            }
        }
    }
}
