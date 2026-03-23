import json
from collections.abc import Callable
from typing import Any

from flask_socketio import SocketIO

import tricked_web.state as st
from tricked.env.pieces import STANDARD_PIECES

def register_sockets(socketio: SocketIO) -> Callable[[], None]:
    def background_telemetry_thread() -> None:
        import time
        last_iter = 0
        last_time = time.time()
        games_per_second = 0.0
        last_top_games_time = 0.0

        while True:
            try:
                r = st.get_redis()
                spectators = r.hgetall("spectator")
                if spectators:
                    best_state = None
                    best_score = -1
                    for pid, state_json in spectators.items():
                        state = json.loads(state_json)
                        if state["score"] > best_score:
                            best_score = state["score"]
                            best_state = state
                    if best_state:
                        best_state["piece_masks"] = [[str(m) for m in p] for p in STANDARD_PIECES]
                        socketio.emit('spectator', best_state)

                is_running = False
                if st.training_process is not None and st.training_process.poll() is None:
                    is_running = True
                    
                status_data: dict[str, Any] = {"running": is_running}
                if is_running:
                    status_json = r.get("training_status")
                    if status_json:
                        parsed_status = json.loads(status_json)
                        status_data.update(parsed_status)
                        
                        curr_iter = int(r.get("total_games_played") or 0)
                        curr_time = time.time()
                        dt = curr_time - last_time
                        if dt > 1.0:
                            if curr_iter >= last_iter:
                                games_per_second = (curr_iter - last_iter) / dt
                            else:
                                games_per_second = curr_iter / dt
                            last_iter = curr_iter
                            last_time = curr_time
                        status_data["games_per_second"] = round(games_per_second, 2)
                        
                socketio.emit('status', status_data)

                curr_time = time.time()
                if curr_time - last_top_games_time > 2.0:
                    last_top_games_time = curr_time
                    games_json = r.lrange("games_history", 0, 1000)
                    games = []
                    for i, g_str in enumerate(games_json):
                        g = json.loads(g_str)
                        moves = g.get("moves", [])
                        final_board = "0"
                        if moves:
                            last_move = moves[-1]
                            final_board = last_move.get("board", "0") if isinstance(last_move, dict) else str(last_move)
                        games.append({
                            "id": len(games_json) - i,
                            "difficulty": g.get("difficulty", 6),
                            "score": g.get("score", 0),
                            "steps": g.get("steps", 0),
                            "board": final_board
                        })
                    games.sort(key=lambda x: x["score"], reverse=True)
                    socketio.emit('top_games', games[:32])

                socketio.sleep(0.5)
            except Exception as e: 
                print("SocketIO Telemetry Error:", str(e)) 
                socketio.sleep(2) 

    @socketio.on('connect')  
    def on_connect() -> None:
        print("Client connected via WebSocket")

    return background_telemetry_thread
