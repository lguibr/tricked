import random

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from tricked.env.pieces import STANDARD_PIECES
from tricked.env.state import GameState

app = Flask(__name__)
CORS(app)

# Global state for simple local server play
current_state = GameState()
current_difficulty = 6


def get_difficulty_filtered_pieces(max_triangles: int) -> list[int]:
    valid_ids = []
    for p_id in range(len(STANDARD_PIECES)):
        for mask in STANDARD_PIECES[p_id]:
            if mask != 0:
                if int(mask).bit_count() <= max_triangles:
                    valid_ids.append(p_id)
                break
    return valid_ids


from typing import Any

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


@app.route("/")
def index() -> str:
    return render_template("index.html")


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
    data = request.json
    slot = data.get("slot")
    idx = data.get("idx")

    if slot is None or idx is None:
        return jsonify({"error": "Missing slot or idx"}), 400

    next_state = current_state.apply_move(int(slot), int(idx))
    if next_state is None:
        return jsonify({"error": "Invalid move mathematically."}), 400

    current_state = next_state

    # Enforce difficulty bounds immediately after Rust refill
    if current_state.pieces_left == 3:
        valid_piece_ids = get_difficulty_filtered_pieces(current_difficulty)
        current_state.available = [
            random.choice(valid_piece_ids),
            random.choice(valid_piece_ids),
            random.choice(valid_piece_ids),
        ]
        current_state.check_terminal()

    return get_state()


ROTATION_MAP = {
    0: 13,
    1: 16,
    2: 1,
    3: 8,
    4: 9,
    5: 25,
    6: 18,
    7: 4,
    8: 3,
    9: 10,
    10: 21,
    11: 20,
    12: 12,
    13: 15,
    14: 19,
    15: 11,
    16: 14,
    17: 0,
    18: 24,
    19: 23,
    20: 17,
    21: 22,
    22: 7,
    23: 2,
    24: 6,
    25: 5,
}


@app.route("/api/rotate", methods=["POST"])
def rotate_slot() -> Any:
    global current_state
    data = request.json
    slot = data.get("slot")

    if slot is None or slot < 0 or slot > 2:
        return jsonify({"error": "Invalid slot"}), 400

    avail = current_state.available
    p_id = avail[slot]
    if p_id != -1 and p_id in ROTATION_MAP:
        avail[slot] = ROTATION_MAP[p_id]
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
    best_state = None
    best_score = -1

    import glob
    import json

    for fpath in glob.glob("/tmp/tricked_worker_*.json"):
        try:
            with open(fpath) as f:
                data = json.load(f)
                if data["score"] > best_score:
                    best_score = data["score"]
                    best_state = data
        except Exception:
            pass  # File might be half-written or locked briefly

    if best_state is None:
        return jsonify({"error": "No spectators found"}), 404

    best_state["piece_masks"] = [[str(m) for m in p] for p in STANDARD_PIECES]
    return jsonify(best_state)


if __name__ == "__main__":
    reset_game()
    app.run(debug=True, port=8080)
