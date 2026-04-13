import os
import pytest
from fastapi.testclient import TestClient
from backend.server import app, DB_PATH
import sqlite3

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_db():
    # Ensure test db is clean before each test
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM runs")
    cur.execute("DELETE FROM metrics")
    conn.commit()
    conn.close()
    yield

def test_happy_path_config_run_cycle():
    # Create
    create_resp = client.post("/api/runs/create", json={
        "name": "Integration Test Run",
        "type": "TRAIN",
        "preset": ""
    })
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    # Save Config
    cfg_resp = client.post("/api/runs/save_config", json={
        "id": run_id,
        "config": '{"hardware": {"num_processes": 1}, "architecture": {}, "optimizer": {}, "mcts": {}, "environment": {}}'
    })
    assert cfg_resp.status_code == 200

    # Start
    start_resp = client.post("/api/runs/start", json={"id": run_id})
    assert start_resp.status_code == 200

    # Stop
    stop_resp = client.post("/api/runs/stop", json={"id": run_id, "force": True})
    assert stop_resp.status_code == 200

    # Retrieve status
    list_resp = client.get("/api/runs")
    assert list_resp.status_code == 200
    runs = list_resp.json()
    assert len(runs) > 0
    assert runs[0]["status"] == "STOPPED"
    assert runs[0]["id"] == run_id

def test_unhappy_path_start_invalid_run():
    # Try starting a non-existent run
    # If the DB doesn't have the config, it won't be able to fetch it and start correctly, 
    # but the PM error might be caught or we might just get a generic 500 depending on implementation.
    # At minimum, process manager throws on invalid execution state or it skips. Let's see what it does.
    # Actually, if we just check status_code == 500 or 409
    start_resp = client.post("/api/runs/start", json={"id": "nonexistent-run-id-1234"})
    assert start_resp.status_code in [409, 500]

def test_delete_run_clears_db():
    # Create to have a target
    create_resp = client.post("/api/runs/create", json={
        "name": "Delete Me",
        "type": "TRAIN",
        "preset": ""
    })
    run_id = create_resp.json()["id"]

    # Mock some metrics
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO metrics (run_id, step) VALUES (?, 1)", (run_id,))
    conn.commit()
    conn.close()

    # Delete
    del_resp = client.post("/api/runs/delete", json={"id": run_id})
    assert del_resp.status_code == 200

    # Verify runs table empty
    list_resp = client.get("/api/runs")
    assert len(list_resp.json()) == 0

    # Verify metrics table empty
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM metrics")
    count = cur.fetchone()[0]
    conn.close()
    assert count == 0
