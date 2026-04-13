import pytest
from fastapi.testclient import TestClient
from tricked.server import app

client = TestClient(app)

def test_hardware_telemetry_get():
    # Hit the basic HTTP endpoint
    resp = client.get("/api/hardware")
    assert resp.status_code == 200
    assert resp.headers.get("content-type") == "application/x-protobuf"
    
    # Assert body parses to proto natively
    from tricked.proto_out.tricked_pb2 import HardwareMetrics
    parsed = HardwareMetrics()
    parsed.ParseFromString(resp.content)
    
    assert parsed.cpu_usage >= 0.0
    assert getattr(parsed, "ram_usage_mb", getattr(parsed, "ram_usage_pct", 0.0)) >= 0.0
    assert parsed.machine_identifier != ""

def test_hardware_telemetry_ws():
    with client.websocket_connect("/api/ws/hardware") as websocket:
        data = websocket.receive_bytes()
        
        from tricked.proto_out.tricked_pb2 import HardwareMetrics
        parsed_ws = HardwareMetrics()
        parsed_ws.ParseFromString(data)
        
        assert parsed_ws.cpu_usage >= 0.0
        assert parsed_ws.machine_identifier != ""
