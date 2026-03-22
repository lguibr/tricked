import json
import os
import random
import subprocess
import sys
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
from tricked_engine import GameStateExt as GameState

from tricked.env.pieces import STANDARD_PIECES

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global state for simple local server play
current_state = GameState()
current_difficulty = 6

training_process: subprocess.Popen[Any] | None = None

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
    """
    Proxies a kinetic fragment projection request into the PyO3 native Rust engine.

    Retrieves the payload comprising the target fragment scalar `idx` and the slot `slot`.
    Attempts a physical structural assignment. If successful natively, the Triango State 
    is fully advanced via PyO3 returning the updated configuration map.

    Returns:
        JSON representation of the unified Triango map or an Error dictionary.
    """
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


# --- REDIS CONNECTION MANAGER ---
_redis_pool = None
def get_redis() -> Any:
    global _redis_pool
    if _redis_pool is None:
        import redis
        host = os.environ.get("REDIS_HOST", "localhost")
        try:
            _redis_pool = redis.Redis(host=host, port=6379, db=0, decode_responses=True)
            _redis_pool.ping()
        except Exception:
            fallback = "localhost" if host == "redis" else "redis"
            _redis_pool = redis.Redis(host=fallback, port=6379, db=0, decode_responses=True)
    return _redis_pool

@app.route("/api/spectator", methods=["GET"])
def spectator_state() -> Any:
    try:
        r = get_redis()
        spectators = r.hgetall("spectator")
        if not spectators:
            return jsonify({"error": "No spectators found"}), 404
            
        best_state = None
        best_score = -1
        for pid, state_json in spectators.items():
            state = json.loads(state_json)
            if state["score"] > best_score:
                best_score = state["score"]
                best_state = state
                
        if best_state:
            best_state["piece_masks"] = [[str(m) for m in p] for p in STANDARD_PIECES]
            return jsonify(best_state)
        return jsonify({"error": "No spectators processing"}), 404
    except Exception:
        return jsonify({"error": "Failed to read Redis IPC"}), 500

@app.route("/api/games/top", methods=["GET"])
def get_top_games() -> Any:
    try:
        difficulty_filter = request.args.get("difficulty", type=int)
        limit = request.args.get("limit", default=32, type=int)

        r = get_redis()
        games_json = r.lrange("games_history", 0, 1000)
        games = []
        for i, g_str in enumerate(games_json):
            g = json.loads(g_str)
            diff = g.get("difficulty", 6)

            if difficulty_filter is not None and diff != difficulty_filter:
                continue

            moves = g.get("moves", [])
            final_board = str(0)
            if moves:
                last_move = moves[-1]
                final_board = last_move["board"] if isinstance(last_move, dict) else str(last_move)
                
            games.append({
                "id": len(games_json) - i,
                "difficulty": diff,
                "score": g.get("score", 0),
                "steps": g.get("steps", 0),
                "board": final_board
            })
            
        games.sort(key=lambda x: x["score"], reverse=True)
        return jsonify(games[:limit])
    except Exception:
        return jsonify([])

@app.route("/api/games/<int:game_id>", methods=["GET"])
def get_game_replay(game_id: int) -> Any:
    try:
        r = get_redis()
        games_json = r.lrange("games_history", 0, 1000)
        idx = len(games_json) - game_id
        if 0 <= idx < len(games_json):
            g = json.loads(games_json[idx])
            return jsonify({
                "difficulty": g.get("difficulty", 6),
                "score": g.get("score", 0),
                "steps": g.get("steps", 0),
                "moves": g.get("moves", [])
            })
        return jsonify({"error": "Game not found"}), 404
    except Exception:
        return jsonify({"error": "Failed to read replay from Redis"}), 500

@app.route("/api/training/status", methods=["GET"])
def training_status() -> Any:
    global training_process
    is_running = False
    if training_process is not None:
        if training_process.poll() is None:
            is_running = True
            
    status_data: dict[str, Any] = {"running": is_running}
    if is_running:
        try:
            r = get_redis()
            status_json = r.get("training_status")
            if status_json:
                status_data.update(json.loads(status_json))
        except Exception:
            pass

    return jsonify(status_data)



def get_exp_dir(exp_name: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "runs", exp_name))

@app.route("/api/experiment/<exp_name>", methods=["GET"])
def get_experiment(exp_name: str) -> Any:
    manifest_path = os.path.join(get_exp_dir(exp_name), "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return jsonify({"exists": True, "config": json.load(f)})
    return jsonify({"exists": False})

@app.route("/api/experiment/<exp_name>", methods=["DELETE"])
def delete_experiment(exp_name: str) -> Any:
    import shutil
    exp_dir = get_exp_dir(exp_name)
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Not found"}), 404

@app.route("/api/experiments", methods=["GET"])
def list_experiments() -> Any:
    runs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "runs"))
    experiments = []
    if os.path.exists(runs_dir):
        for entry in os.listdir(runs_dir):
            exp_dir = os.path.join(runs_dir, entry)
            if os.path.isdir(exp_dir):
                manifest_path = os.path.join(exp_dir, "manifest.json")
                if os.path.exists(manifest_path):
                    with open(manifest_path) as f:
                        try:
                            config = json.load(f)
                            experiments.append({"name": entry, "config": config})
                        except Exception:
                            pass
    # Sort experiments by name desc
    experiments.sort(key=lambda x: x["name"], reverse=True)
    return jsonify(experiments)

@app.route("/api/training/start", methods=["POST"])
def training_start() -> Any:
    global training_process
    if training_process is None or training_process.poll() is not None:
        data = request.json or {}
        exp_name = data.get("expName", "ui_experiment_v1")
        exp_dir = get_exp_dir(exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        manifest_path = os.path.join(exp_dir, "manifest.json")

        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                saved_config = json.load(f)
                data["dModel"] = saved_config.get("dModel", data.get("dModel"))
                data["numBlocks"] = saved_config.get("numBlocks", data.get("numBlocks"))
        else:
            with open(manifest_path, "w") as f:
                json.dump(data, f, indent=4)

        env = os.environ.copy()
        env["ENABLE_WEB_UI"] = "0"
        env["PYTHONPATH"] = "src"  # <--- Crucial fix for the Subprocess
        env["EXP_NAME"] = exp_name
        env["D_MODEL"] = str(data.get("dModel", 128))
        env["NUM_BLOCKS"] = str(data.get("numBlocks", 8))
        env["SIMULATIONS"] = str(data.get("simulations", 50))
        env["UNROLL_STEPS"] = str(data.get("unrollSteps", 5))
        env["TRAIN_BATCH"] = str(data.get("trainBatch", 256))
        env["NUM_GAMES"] = str(data.get("numGames", 1000))
        env["WORKERS"] = str(data.get("workers", 24))
        env["TEMP_DECAY_STEPS"] = str(data.get("tempDecaySteps", 30))
        env["MAX_GUMBEL_K"] = str(data.get("maxGumbelK", 8))
        if "WANDB_API_KEY" in os.environ:
            env["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
        elif "WANDB_API_KEY" in env:
            del env["WANDB_API_KEY"]
            
        env["WANDB_BASE_URL"] = os.environ.get("WANDB_BASE_URL", "http://localhost:8081")

        training_process = subprocess.Popen(
            [sys.executable, "src/tricked/main.py", "--headless"],
            env=env,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        )
    return jsonify({"running": True, "exp_name": exp_name})

@app.route("/api/training/stop", methods=["POST"])
def training_stop() -> Any:
    global training_process
    if training_process is not None and training_process.poll() is None:
        training_process.terminate()
        try:
            training_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            training_process.kill()
    training_process = None
    
    return jsonify({"running": False})


def background_telemetry_thread() -> None:
    while True:
        try:
            r = get_redis()
            
            # Publish Spectator State
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

            # Publish Training Status
            is_running = False
            if training_process is not None and training_process.poll() is None:
                is_running = True
                
            status_data = {"running": is_running}
            if is_running:
                status_json = r.get("training_status")
                if status_json:
                    status_data.update(json.loads(status_json))
            socketio.emit('status', status_data)

            socketio.sleep(0.5) # 2 Hz telemetry stream
        except Exception as e:
            print("SocketIO Telemetry Error:", str(e))
            socketio.sleep(2)

@socketio.on('connect')  # type: ignore
def on_connect() -> None:
    print("Client connected via WebSocket")
