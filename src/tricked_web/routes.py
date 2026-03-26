import json
import os
import shutil
import subprocess
import sys
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import tricked_web.state as st
from tricked.env.pieces import STANDARD_PIECES

router = APIRouter(prefix="/api")

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


def get_exp_dir(exp_name: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "runs", exp_name))


class MoveRequest(BaseModel):
    slot: int
    idx: int


class RotateRequest(BaseModel):
    slot: int
    direction: str = "right"


class ResetRequest(BaseModel):
    difficulty: int = 6


class TrainingStartRequest(BaseModel):
    expName: str = "ui_experiment_v1"
    dModel: int = 128
    numBlocks: int = 8
    simulations: int = 50
    unrollSteps: int = 5
    trainBatch: int = 256
    numGames: int = 1000
    workers: int = 24
    tempDecaySteps: int = 30
    maxGumbelK: int = 8


@router.get("/state")
def get_state() -> Any:
    return {
        "board": str(st.current_state.board),
        "score": st.current_state.score,
        "pieces_left": st.current_state.pieces_left,
        "terminal": st.current_state.terminal,
        "available": st.current_state.available,
        "piece_masks": [[str(m) for m in p] for p in STANDARD_PIECES],
    }


@router.post("/move")
def make_move(req: MoveRequest) -> Any:
    next_state = st.current_state.apply_move(req.slot, req.idx)
    if next_state is None:
        raise HTTPException(status_code=400, detail="Invalid move mathematically.")

    st.current_state = next_state
    return get_state()


@router.post("/rotate")
def rotate_slot(req: RotateRequest) -> Any:
    if req.slot < 0 or req.slot > 2:
        raise HTTPException(status_code=400, detail="Invalid slot")

    avail = st.current_state.available
    p_id = avail[req.slot]
    if p_id != -1:
        if req.direction == "left" and p_id in ROTATION_MAP_LEFT:
            avail[req.slot] = ROTATION_MAP_LEFT[p_id]
            st.current_state.available = avail
            st.current_state.check_terminal()
        elif req.direction == "right" and p_id in ROTATION_MAP_RIGHT:
            avail[req.slot] = ROTATION_MAP_RIGHT[p_id]
            st.current_state.available = avail
            st.current_state.check_terminal()

    return get_state()


@router.post("/reset")
def do_reset(req: ResetRequest) -> Any:
    st.reset_game(req.difficulty)
    return get_state()


@router.get("/spectator")
def spectator_state() -> Any:
    try:
        r = st.get_redis()
        spectators = r.hgetall("spectator")
        if not spectators:
            raise HTTPException(status_code=404, detail="No spectators found")
        best_state = None
        best_score = -1
        for pid, state_json in spectators.items():
            state = json.loads(state_json)
            if state["score"] > best_score:
                best_score = state["score"]
                best_state = state
        if best_state:
            best_state["piece_masks"] = [[str(m) for m in p] for p in STANDARD_PIECES]
            return best_state
        raise HTTPException(status_code=404, detail="No spectators processing")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read Redis IPC")


@router.get("/games/top")
def get_top_games(difficulty: int | None = None, limit: int = 32) -> Any:
    try:
        r = st.get_redis()
        games_json = r.lrange("games_history", 0, 1000)
        games = []
        for i, g_str in enumerate(games_json):
            g = json.loads(g_str)
            diff = g.get("difficulty", 6)
            if difficulty is not None and diff != difficulty:
                continue
            moves = g.get("moves", [])
            final_board = str(0)
            if moves:
                last_move = moves[-1]
                final_board = last_move["board"] if isinstance(last_move, dict) else str(last_move)
            games.append(
                {
                    "id": len(games_json) - i,
                    "difficulty": diff,
                    "score": g.get("score", 0),
                    "steps": g.get("steps", 0),
                    "board": final_board,
                }
            )
        games.sort(key=lambda x: x["score"], reverse=True)
        return games[:limit]
    except Exception:
        return []


@router.get("/games/{game_id}")
def get_game_replay(game_id: int) -> Any:
    try:
        r = st.get_redis()
        games_json = r.lrange("games_history", 0, 1000)
        idx = len(games_json) - game_id
        if 0 <= idx < len(games_json):
            g = json.loads(games_json[idx])
            return {
                "difficulty": g.get("difficulty", 6),
                "score": g.get("score", 0),
                "steps": g.get("steps", 0),
                "moves": g.get("moves", []),
            }
        raise HTTPException(status_code=404, detail="Game not found")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching game replay: {e}")
        raise HTTPException(status_code=500, detail="Failed to read replay from Redis")


@router.get("/training/status")
def training_status() -> Any:
    is_running = False
    if st.training_process is not None:
        if st.training_process.poll() is None:
            is_running = True
    status_data: dict[str, Any] = {"running": is_running}
    if is_running:
        try:
            r = st.get_redis()
            status_json = r.get("training_status")
            if status_json:
                status_data.update(json.loads(status_json))
        except Exception:
            pass
    return status_data


@router.get("/experiment/{exp_name}")
def get_experiment(exp_name: str) -> Any:
    manifest_path = os.path.join(get_exp_dir(exp_name), "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return {"exists": True, "config": json.load(f)}
    return {"exists": False}


@router.delete("/experiment/{exp_name}")
def delete_experiment(exp_name: str) -> Any:
    exp_dir = get_exp_dir(exp_name)
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
        return {"success": True}
    raise HTTPException(status_code=404, detail="Not found")


@router.get("/experiments")
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
    experiments.sort(key=lambda x: x["name"], reverse=True)
    return experiments


@router.post("/training/start")
def training_start(req: TrainingStartRequest) -> Any:
    if st.training_process is None or st.training_process.poll() is not None:
        data = req.model_dump()
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
        env["PYTHONPATH"] = "src"
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

        cmd = [
            sys.executable,
            "src/tricked/main.py",
            f"exp_name={exp_name}",
            f"d_model={data.get('dModel', 128)}",
            f"num_blocks={data.get('numBlocks', 8)}",
            f"simulations={data.get('simulations', 50)}",
            f"unroll_steps={data.get('unrollSteps', 5)}",
            f"train_batch_size={data.get('trainBatch', 256)}",
            f"num_games={data.get('numGames', 1000)}",
            f"num_processes={data.get('workers', 24)}",
            f"temp_decay_steps={data.get('tempDecaySteps', 30)}",
            f"max_gumbel_k={data.get('maxGumbelK', 8)}",
        ]

        st.training_process = subprocess.Popen(
            cmd,
            env=env,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        )
    return {"running": True, "exp_name": req.expName}


@router.post("/training/stop")
def training_stop() -> Any:
    if st.training_process is not None and st.training_process.poll() is None:
        st.training_process.terminate()
        try:
            st.training_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            st.training_process.kill()
    st.training_process = None
    return {"running": False}
