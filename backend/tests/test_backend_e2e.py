import pytest
from fastapi.testclient import TestClient
from backend.server import app

def test_full_e2e_backend_telemetry_flow():
    with TestClient(app) as client:
        # Check standard endpoint route for binary encoding
        response = client.get("/api/hardware")
        
        assert response.status_code == 200
        assert response.headers.get("content-type") == "application/x-protobuf"
        assert isinstance(response.content, bytes)
        assert len(response.content) > 0

        from backend.proto_out.tricked_pb2 import HardwareMetrics
        parsed = HardwareMetrics()
        parsed.ParseFromString(response.content)
        assert parsed.cpu_usage >= 0.0
        assert parsed.machine_identifier == "Python Orchestrator"

        # Check WebSocket endpoint
        with client.websocket_connect("/api/ws/hardware") as websocket:
            binary_data = websocket.receive_bytes()
            assert isinstance(binary_data, bytes)
            assert len(binary_data) > 0
            
            parsed_ws = HardwareMetrics()
            parsed_ws.ParseFromString(binary_data)
            assert parsed_ws.cpu_usage >= 0.0
            assert parsed_ws.machine_identifier == "Python Orchestrator"
