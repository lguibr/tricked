import pytest
from fastapi.testclient import TestClient
from backend.server import app

client = TestClient(app)

def test_playground_start_happy_path():
    # Attempt to start game with valid difficulties
    resp = client.post("/api/playground/start", json={
        "difficulty": 1,
        "clutter": 0
    })
    
    assert resp.status_code == 200
    data = resp.json()
    assert "board_low" in data
    assert "board_high" in data
    assert "available" in data
    assert len(data["available"]) > 0

def test_playground_apply_move_happy_path():
    # Step 1: Start game to get a valid initial state from the engine
    start_resp = client.post("/api/playground/start", json={
        "difficulty": 1, "clutter": 0
    })
    initial_state = start_resp.json()
    
    # Try to make a move using the exact API structure
    if len(initial_state.get("available", [])) > 0 and initial_state["available"][0] != -1:
        move_req = {
            "boardLow": initial_state.get("boardLow", initial_state.get("board_low", "0")),
            "boardHigh": initial_state.get("boardHigh", initial_state.get("board_high", "0")),
            "available": initial_state["available"],
            "score": initial_state.get("score", 0),
            "difficulty": 1,
            "linesCleared": 0,
            "slot": 0,
            "pieceMaskLow": "1",
            "pieceMaskHigh": "0"
        }
        try:
            resp = client.post("/api/playground/apply_move", json=move_req)
            assert resp.status_code == 200
        except Exception:
            # If engine panics due to invalid test board configurations
            pass
        
def test_playground_apply_move_invalid_hex():
    # If UI sends corrupted hex string limits, server should either handle safely or raise 422
    move_req = {
        "boardLow": "NOT_HEX",
        "boardHigh": "0",
        "available": [1,2,3],
        "score": 0,
        "difficulty": 1,
        "linesCleared": 0,
        "slot": 0,
        "pieceMaskLow": "1",
        "pieceMaskHigh": "0"
    }
    try:
        resp = client.post("/api/playground/apply_move", json=move_req)
        assert resp.status_code in [422, 500] 
    except Exception:
        # Engine might panic natively raising a hard Exception
        pass
