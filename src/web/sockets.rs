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
    let piece_binary_masks: Vec<Vec<String>> = STANDARD_PIECES
        .iter()
        .map(|piece_layout| {
            piece_layout
                .iter()
                .map(|bitboard_mask| bitboard_mask.to_string())
                .collect()
        })
        .collect();

    let mut broadcast_ticker = interval(Duration::from_millis(50));
    loop {
        tokio::select! {
            _ = broadcast_ticker.tick() => {
                let telemetry_payload = {
                    let shared_telemetry = application_state.telemetry.read().unwrap();
                    let mut spectator_json_payload = serde_json::Value::Null;

                    if let Some(spectator_metrics) = &shared_telemetry.spectator_state {
                        spectator_json_payload = json!({
                            "board": spectator_metrics.board.to_string(),
                            "score": spectator_metrics.score,
                            "pieces_left": spectator_metrics.pieces_left,
                            "terminal": spectator_metrics.terminal,
                            "available": spectator_metrics.available,
                            "piece_masks": &piece_binary_masks,
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
                            // We just send games_played. The UI will calculate GPS.
                            "games_played": shared_telemetry.status.games_played,
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
