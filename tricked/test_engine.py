def test_playground_start():
    import tricked_engine
    import json
    
    # Needs to return a serialized game state
    out = tricked_engine.playground_start_game(6, 0)
    data = json.loads(out)
    assert "board_low" in data
    assert "available" in data
    assert len(data["available"]) == 3

def test_playground_apply_move():
    import tricked_engine
    import json
    
    # Just an arbitrary mask overlay against 0 layout
    out = tricked_engine.playground_apply_move("0", "0", [0, 1, 2], 0, 6, 0, 0, "1", "0")
    assert out is not None
    data = json.loads(out)
    assert data["score"] >= 1
    assert data["available"][0] == -1
