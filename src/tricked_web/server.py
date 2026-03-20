import os
import random
import subprocess
import sys
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS

from tricked.env.pieces import STANDARD_PIECES
from tricked.env.state import GameState

app = Flask(__name__)
CORS(app)

# Global state for simple local server play
current_state = GameState()
current_difficulty = 6

training_process: subprocess.Popen[Any] | None = None
tb_process: subprocess.Popen[Any] | None = None


def get_difficulty_filtered_pieces(max_triangles: int) -> list[int]:
    valid_ids = []
    for p_id in range(len(STANDARD_PIECES)):
        for mask in STANDARD_PIECES[p_id]:
            if mask != 0:
                if int(mask).bit_count() <= max_triangles:
                    valid_ids.append(p_id)
                break
    return valid_ids


def reset_game(difficulty: int = 6) -> None:
    global current_state, current_difficulty
    current_difficulty = difficulty
    current_state = GameState()
    valid_piece_ids = get_difficulty_filtered_pieces(current_difficulty)
    current_state.available = [
        random.choice(valid_piece_ids),
        random.choice(valid_piece_ids),
        random.choice(valid_piece_ids),
    ]





@app.route("/api/state", methods=["GET"])
def get_state() -> Any:
    return jsonify(
        {
            "board": str(current_state.board),
            "score": current_state.score,
            "pieces_left": current_state.pieces_left,
            "terminal": current_state.terminal,
            "available": current_state.available,
            "piece_masks": [[str(m) for m in p] for p in STANDARD_PIECES],
        }
    )


@app.route("/api/move", methods=["POST"])
def make_move() -> Any:
    global current_state
    data = request.json or {}
    slot = data.get("slot")
    idx = data.get("idx")

    if slot is None or idx is None:
        return jsonify({"error": "Missing slot or idx"}), 400

    next_state = current_state.apply_move(int(slot), int(idx))
    if next_state is None:
        return jsonify({"error": "Invalid move mathematically."}), 400

    current_state = next_state

    return get_state()


ROTATION_MAP_RIGHT = {
    0: 16,
    1: 2,
    2: 22,
    3: 8,
    4: 7,
    5: 29,
    6: 23,
    7: 21,
    8: 3,
    9: 4,
    10: 9,
    11: 14,
    12: 0,
    13: 15,
    14: 12,
    15: 1,
    16: 19,
    17: 6,
    18: 13,
    19: 11,
    20: 10,
    21: 20,
    22: 18,
    23: 17,
    29: 5,
}

ROTATION_MAP_LEFT = {v: k for k, v in ROTATION_MAP_RIGHT.items()}


@app.route("/api/rotate", methods=["POST"])
def rotate_slot() -> Any:
    global current_state
    data = request.json or {}
    slot = data.get("slot")
    direction = data.get("direction", "right")

    if slot is None or slot < 0 or slot > 2:
        return jsonify({"error": "Invalid slot"}), 400

    avail = current_state.available
    p_id = avail[slot]
    if p_id != -1:
        if direction == "left" and p_id in ROTATION_MAP_LEFT:
            avail[slot] = ROTATION_MAP_LEFT[p_id]
            current_state.available = avail
            current_state.check_terminal()
        elif direction == "right" and p_id in ROTATION_MAP_RIGHT:
            avail[slot] = ROTATION_MAP_RIGHT[p_id]
            current_state.available = avail
            current_state.check_terminal()

    return get_state()


@app.route("/api/reset", methods=["POST"])
def do_reset() -> Any:
    data = request.json or {}
    diff = int(data.get("difficulty", 6))
    reset_game(diff)
    return get_state()


@app.route("/api/spectator", methods=["GET"])
def spectator_state() -> Any:
    import json
    import os
    import sqlite3
    
    db_path = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "experience.db")
    if not os.path.exists(db_path):
        return jsonify({"error": "No spectators found"}), 404
        
    try:
        conn = sqlite3.connect(db_path, timeout=1)
        row = conn.execute("SELECT state FROM spectator ORDER BY score DESC LIMIT 1").fetchone()
        conn.close()
        if row is None:
            return jsonify({"error": "No spectators found"}), 404
            
        best_state = json.loads(row[0])
        best_state["piece_masks"] = [[str(m) for m in p] for p in STANDARD_PIECES]
        return jsonify(best_state)
    except Exception:
        return jsonify({"error": "Failed to read SQLite"}), 500

@app.route("/api/games/top", methods=["GET"])
def get_top_games() -> Any:
    import os
    import sqlite3
    
    db_path = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "experience.db")
    if not os.path.exists(db_path):
        return jsonify([])
        
    try:
        conn = sqlite3.connect(db_path, timeout=1)
        max_diff_row = conn.execute("SELECT MAX(difficulty) FROM games").fetchone()
        max_diff = max_diff_row[0] if max_diff_row and max_diff_row[0] is not None else 1
        
        # Fetch the top 32 strictly locked to the highest active difficulty
        cursor = conn.execute("SELECT id, difficulty, score, steps, moves FROM games WHERE difficulty = ? ORDER BY score DESC, id DESC LIMIT 32", (max_diff,))
        games = []
        import json
        for row in cursor.fetchall():
            moves = json.loads(row[4]) if row[4] else []
            final_board = str(0)
            if moves:
                last_move = moves[-1]
                final_board = last_move["board"] if isinstance(last_move, dict) else last_move
                
            games.append({
                "id": row[0],
                "difficulty": row[1],
                "score": row[2],
                "steps": row[3],
                "board": final_board
            })
        conn.close()
        return jsonify(games)
    except Exception:
        return jsonify([])

@app.route("/api/games/<int:game_id>", methods=["GET"])
def get_game_replay(game_id: int) -> Any:
    import json
    import os
    import sqlite3
    
    db_path = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "experience.db")
    if not os.path.exists(db_path):
        return jsonify({"error": "Database not found"}), 404
        
    try:
        conn = sqlite3.connect(db_path, timeout=1)
        row = conn.execute("SELECT difficulty, score, steps, moves FROM games WHERE id = ?", (game_id,)).fetchone()
        conn.close()
        
        if row is None:
            return jsonify({"error": "Game not found"}), 404
            
        replay_data = {
            "difficulty": row[0],
            "score": row[1],
            "steps": row[2],
            "moves": json.loads(row[3])
        }
        return jsonify(replay_data)
    except Exception:
        return jsonify({"error": "Failed to read replay"}), 500


@app.route("/api/training/status", methods=["GET"])
def training_status() -> Any:
    global training_process
    is_running = False
    if training_process is not None:
        if training_process.poll() is None:
            is_running = True
            
    status_data: dict[str, Any] = {"running": is_running}
    if is_running:
        import json
        import os
        import sqlite3
        db_path = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "experience.db")
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path, timeout=1)
                row = conn.execute("SELECT status_json FROM training_status WHERE id=1").fetchone()
                conn.close()
                if row:
                    status_data.update(json.loads(row[0]))
            except Exception:
                pass

    return jsonify(status_data)


@app.route("/api/training/start", methods=["POST"])
def training_start() -> Any:
    global training_process
    global tb_process
    if training_process is None or training_process.poll() is not None:
        env = os.environ.copy()
        env["ENABLE_WEB_UI"] = "0"
        training_process = subprocess.Popen(
            [sys.executable, "src/tricked/main.py"],
            env=env,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        )
        if tb_process is None or tb_process.poll() is not None:
            tb_process = subprocess.Popen(
                [sys.executable, "-m", "tensorboard.main", "--logdir", "runs", "--port", "6006"],
                env=env,
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
            )
    return jsonify({"running": True})


@app.route("/api/training/stop", methods=["POST"])
def training_stop() -> Any:
    global training_process
    global tb_process
    if training_process is not None and training_process.poll() is None:
        training_process.terminate()
        try:
            training_process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            training_process.kill()  # pragma: no cover
    training_process = None
    
    if tb_process is not None:
        tb_process.terminate()
        tb_process = None
        
    return jsonify({"running": False})


if __name__ == "__main__":
    reset_game()
    app.run(debug=True, port=8080)
