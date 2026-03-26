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

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: AppState) {
    let mut ticker = interval(Duration::from_millis(50));
    loop {
        ticker.tick().await;

        let payload = {
            let tel = state.telemetry.read().unwrap();
            let mut spec_json = serde_json::Value::Null;

            if let Some(spec) = &tel.spectator_state {
                let masks: Vec<Vec<String>> = STANDARD_PIECES
                    .iter()
                    .map(|p| p.iter().map(|m| m.to_string()).collect())
                    .collect();

                spec_json = json!({
                    "board": spec.board.to_string(),
                    "score": spec.score,
                    "pieces_left": spec.pieces_left,
                    "terminal": spec.terminal,
                    "available": spec.available,
                    "piece_masks": masks,
                });
            }

            json!({
                "type": "sync",
                "spectator": spec_json,
                "status": {
                    "running": tel.status.running,
                    "loss_total": tel.status.loss_total,
                }
            })
        };

        if socket
            .send(Message::Text(payload.to_string()))
            .await
            .is_err()
        {
            break;
        }
    }
}
