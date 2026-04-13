import pytest
import os
import time
import collections
from fastapi.testclient import TestClient
from backend.server import app, PROJECT_ROOT, pm

client = TestClient(app)

def test_terminal_websocket_tailing():
    run_id = "test_terminal_run"
    run_dir = os.path.join(PROJECT_ROOT, "backend", "workspace", "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "output.log")
    
    # Write initial 1000 lines
    with open(log_path, "w") as f:
        for i in range(1000):
            f.write(f"Line {i}\n")
            
    # Mock the process manager state
    pm.active_run = {"run_id": run_id, "pid": 1234, "type": "TRAIN"}
    pm.log_buffer = collections.deque([f"Line {i}" for i in range(500, 1000)], maxlen=500)
    
    with client.websocket_connect(f"/api/ws/runs/{run_id}/logs") as websocket:
        # Initial connect should fetch last 500 lines
        data = websocket.receive_json()
        assert isinstance(data, list)
        assert len(data) == 500
        assert data[0] == "Line 500"
        assert data[-1] == "Line 999"

        # Now append a line
        for sub in pm.log_subscribers:
            sub.put_nowait(["Line 1000"])
            
        # Should stream the diff
        diff = websocket.receive_json()
        assert isinstance(diff, list)
        assert len(diff) == 1
        assert diff[0] == "Line 1000"
        
        # Manually close to break the server tail loop!
        websocket.close()

def test_terminal_websocket_file_read():
    run_id = "test_terminal_run_file"
    run_dir = os.path.join(PROJECT_ROOT, "backend", "workspace", "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "output.log")
    
    # Write initial 1000 lines
    with open(log_path, "w") as f:
        for i in range(1000):
            f.write(f"Line {i}\n")
            
    # Ensure no active run
    pm.active_run = None
    
    with client.websocket_connect(f"/api/ws/runs/{run_id}/logs") as websocket:
        # Initial connect should fetch last 500 lines from file
        data = websocket.receive_json()
        assert isinstance(data, list)
        assert len(data) == 500
        assert data[0] == "Line 500"
        assert data[-1] == "Line 999"
        
        websocket.close()
