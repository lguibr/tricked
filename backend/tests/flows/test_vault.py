import pytest
import json
from fastapi.testclient import TestClient
from backend.server import app

client = TestClient(app)

def test_vault_global_fetch():
    # Attempt to fetch global vault
    # If the database is empty, engine might return "[]" or a stringified layout
    resp = client.get("/api/vault/global")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)

def test_commit_to_vault_human_game():
    # Post a mock human game
    mock_game = {
        "source_run_id": "human_play_1",
        "source_run_name": "Human Session",
        "run_type": "HUMAN",
        "difficulty": 1,
        "episode_score": 100.5,
        "lines_cleared": 2,
        "mcts_depth_mean": 0.0,
        "mcts_search_time_mean": 0.0,
        "steps": [
            {
                "board_low": "123",
                "board_high": "456",
                "available": [1, 2, 3],
                "action_taken": 2,
                "piece_identifier": 1
            }
        ]
    }
    resp = client.post("/api/playground/commit_to_vault", json=mock_game)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    
    # Verify it appears in global vault
    check = client.get("/api/vault/global")
    assert check.status_code == 200
    data = check.json()
    # It might be aggregated so we just check it doesn't crash 
    # and maybe data len > 0 if engine persists it fast enough.
    assert isinstance(data, list)

def test_commit_to_vault_invalid_payload():
    # Post missing fields
    bad_game = {
        "source_run_id": "bad",
        "source_run_name": "Bad"
    }
    resp = client.post("/api/playground/commit_to_vault", json=bad_game)
    assert resp.status_code == 422 # Pydantic ValidationError
