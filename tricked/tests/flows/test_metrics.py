import pytest
import sqlite3
import time
from fastapi.testclient import TestClient
from tricked.server import app, DB_PATH
from tricked.proto_out.tricked_pb2 import MetricHistory

client = TestClient(app)

def test_metrics_websocket_history():
    # Setup mock metrics
    run_id = "test_ws_metric_run"
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM metrics")
    cur.execute("INSERT INTO metrics (run_id, step, total_loss) VALUES (?, 1, 0.5)", (run_id,))
    cur.execute("INSERT INTO metrics (run_id, step, total_loss) VALUES (?, 2, 0.2)", (run_id,))
    conn.commit()
    conn.close()

    with client.websocket_connect(f"/api/ws/runs/{run_id}/metrics") as websocket:
        # Initial connect should fetch both metrics from step > -1
        data = websocket.receive_bytes()
        history = MetricHistory()
        history.ParseFromString(data)
        
        assert len(history.metrics) == 2
        assert history.metrics[0].step == 1
        assert history.metrics[0].total_loss == pytest.approx(0.5)
        assert history.metrics[1].step == 2
        assert history.metrics[1].total_loss == pytest.approx(0.2)

def test_metrics_websocket_differential():
    run_id = "test_ws_metric_diff"
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM metrics")
    cur.execute("INSERT INTO metrics (run_id, step, value_loss) VALUES (?, 5, 0.1)", (run_id,))
    conn.commit()
    conn.close()

    from tricked.server import pm
    pm.active_run = {"run_id": run_id}

    with client.websocket_connect(f"/api/ws/runs/{run_id}/metrics") as websocket:
        init_data = websocket.receive_bytes()
        
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO metrics (run_id, step, value_loss) VALUES (?, 10, 0.05)", (run_id,))
        conn.commit()
        conn.close()

        diff_data = websocket.receive_bytes()
        diff_history = MetricHistory()
        diff_history.ParseFromString(diff_data)
        
        assert len(diff_history.metrics) == 1
        assert diff_history.metrics[0].step == 10
        assert diff_history.metrics[0].value_loss == pytest.approx(0.05)
        
    pm.active_run = None
