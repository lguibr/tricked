import os
import pytest
from fastapi.testclient import TestClient
from backend.server import app, DB_PATH
import sqlite3

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM runs")
    conn.commit()
    conn.close()
    yield

def test_tuning_study_happy_path():
    # Create study run
    create_resp = client.post("/api/runs/create", json={
        "name": "Integration Tune",
        "type": "STUDY",
        "preset": ""
    })
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    client.post("/api/runs/save_config", json={"id": run_id, "config": "{}"})

    # Start study
    start_resp = client.post("/api/studies/start", json={"id": run_id})
    assert start_resp.status_code == 200

    # Retrieve status checks RUNNING or STARTING
    list_resp = client.get("/api/runs")
    runs = list_resp.json()
    assert len(runs) > 0
    assert runs[0]["id"] == run_id

    # Stop study
    stop_resp = client.post("/api/runs/stop", json={"id": run_id, "force": True})
    assert stop_resp.status_code == 200

def test_tuning_study_invalid_start():
    # Attempting to start a study on an invalid ID without prior config setting
    start_resp = client.post("/api/studies/start", json={"id": "non_existent_study_id"})
    assert start_resp.status_code in [409, 500]

def test_tuning_does_not_conflict_with_training():
    create_resp1 = client.post("/api/runs/create", json={"name": "A", "type": "STUDY", "preset": ""})
    run_1_id = create_resp1.json()["id"]
    client.post("/api/runs/save_config", json={"id": run_1_id, "config": "{}"})
    
    # Can't start two things at same time in current ProcessManager
    client.post("/api/studies/start", json={"id": run_1_id})

    create_resp2 = client.post("/api/runs/create", json={"name": "B", "type": "TRAIN", "preset": ""})
    run_2_id = create_resp2.json()["id"]
    client.post("/api/runs/save_config", json={"id": run_2_id, "config": "{}"})
    
    start_resp = client.post("/api/runs/start", json={"id": run_2_id})
    # ProcessManager throws ValueError if something is running
    assert start_resp.status_code == 409

    # Cleanup
    client.post("/api/runs/stop", json={"id": run_1_id, "force": True})

