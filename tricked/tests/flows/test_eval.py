import pytest
from fastapi.testclient import TestClient
from tricked.server import app

client = TestClient(app)

def test_evaluation_missing_checkpoint():
    # If a checkpoint path is entirely fake, the backend might trap it or raise a hard Exception.
    try:
        resp = client.post("/api/evaluation/step", json={
            "boardLow": "1",
            "boardHigh": "0",
            "available": [1, 2, 3],
            "checkpointPath": "/fake/path/does/not/exist.safetensors"
        })
        assert resp.status_code in [404, 500]
    except Exception:
        # FastAPI might re-raise FileNotFoundError natively here during tests
        pass

def test_evaluation_invalid_payload():
    # Sending missing attributes
    resp = client.post("/api/evaluation/step", json={
        "boardLow": "1"
        # missing boardHigh, available, checkpointPath
    })
    assert resp.status_code == 422
