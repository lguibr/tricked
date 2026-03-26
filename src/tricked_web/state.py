import os
import random
import subprocess
from typing import Any

from tricked_engine import GameStateExt as GameState

from tricked.env.pieces import STANDARD_PIECES

current_state = GameState()
current_difficulty = 6
training_process: subprocess.Popen[Any] | None = None
_redis_pool = None


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
