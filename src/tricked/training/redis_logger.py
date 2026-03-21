"""
Redis-based telemetry logging for extreme high-throughput IPC.
Replaces SQL locks with in-memory JSON streams.
"""

import json
from typing import Any

# Global pool to avoid multiprocessing fork issues
_redis_client = None

def get_redis():
    global _redis_client
    if _redis_client is None:
        import redis
        # Standard local redis port
        _redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    return _redis_client

def init_db() -> None:
    try:
        r = get_redis()
        r.ping()
        print("🚀 Connected to Redis IPC Telemetry Server.")
        r.delete("spectator") 
        r.delete("training_status")
    except Exception as e:
        print(f"⚠️ Redis Connection Failed. Please ensure 'redis-server' is running natively: {e}")

def update_spectator(worker_pid: int, state_dict: dict[str, Any]) -> None:
    try:
        get_redis().hset("spectator", str(worker_pid), json.dumps(state_dict))
    except Exception:
        pass

def log_game(difficulty: int, score: float, steps: int, history_states: list[dict[str, Any]]) -> None:
    try:
        data = {
            "difficulty": difficulty,
            "score": int(score),
            "steps": steps,
            "moves": history_states
        }
        r = get_redis()
        r.lpush("games_history", json.dumps(data))
        r.ltrim("games_history", 0, 1000)
    except Exception:
        pass

def update_training_status(status_dict: dict[str, Any]) -> None:
    try:
        get_redis().set("training_status", json.dumps(status_dict))
    except Exception:
        pass
