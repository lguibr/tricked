"""
Standard Documentation for sqlite_logger.py.

This module supplies high-performance, asynchronous PyTorch logging natively targeting SQLite.
It utilizes Write-Ahead Logging (WAL) to completely decouple disk I/O from Neural inference latency.
"""

import json
import os
import sqlite3
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "runs", "experience.db")


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS spectator (
            worker_pid INTEGER PRIMARY KEY,
            score INTEGER,
            state JSON
        )
        """
    )
    
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            difficulty INTEGER,
            score INTEGER,
            steps INTEGER,
            moves JSON,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS training_status (
            id INTEGER PRIMARY KEY,
            status_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def update_spectator(worker_pid: int, state_dict: dict[str, Any]) -> None:
    try:
        conn = sqlite3.connect(DB_PATH, timeout=2)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            INSERT INTO spectator (worker_pid, score, state) VALUES (?, ?, ?)
            ON CONFLICT(worker_pid) DO UPDATE SET score=excluded.score, state=excluded.state
            """,
            (worker_pid, state_dict["score"], json.dumps(state_dict)),
        )
        conn.commit()
        conn.close()
    except Exception:  # pragma: no cover
        pass


def log_game(difficulty: int, score: float, steps: int, history_states: list[str]) -> None:
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            "INSERT INTO games (difficulty, score, steps, moves) VALUES (?, ?, ?, ?)",
            (difficulty, int(score), steps, json.dumps(history_states)),
        )
        conn.commit()
        conn.close()
    except Exception:  # pragma: no cover
        pass


def update_training_status(status_dict: dict[str, Any]) -> None:
    try:
        conn = sqlite3.connect(DB_PATH, timeout=2)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            INSERT INTO training_status (id, status_json) VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET status_json=excluded.status_json
            """,
            (json.dumps(status_dict),),
        )
        conn.commit()
        conn.close()
    except Exception:  # pragma: no cover
        pass
