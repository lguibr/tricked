import pytest
from backend.config_schema import TrickedConfig
from backend.proto_out.tricked_pb2 import TrickedConfig as ProtoTrickedConfig
from google.protobuf.json_format import ParseError

def test_config_strict_parsing():
    # Valid raw JSON testing complete struct mapping
    valid_json = """
    {
        "experiment_name_identifier": "test_run",
        "checkpoint_interval": 50,
        "hardware": {
            "device": "cuda:0",
            "num_processes": 8,
            "inference_batch_size_limit": 256
        },
        "mcts": {
            "simulations": 100
        }
    }
    """
    cfg = TrickedConfig.parse_raw(valid_json)
    
    # Assert pydantic wrapper wraps protobuf struct successfully
    assert cfg.proto.experiment_name_identifier == "test_run"
    assert cfg.proto.mcts.simulations == 100
    assert cfg.proto.hardware.num_processes == 8

    # Ensure validation strips out garbage
    invalid_dimension_json = """
    {
        "experiment_name_identifier": "bad_run",
        "mcts": {
            "simulations": "not_an_int"
        }
    }
    """
    
    with pytest.raises((ValueError, ParseError)):
        TrickedConfig.parse_raw(invalid_dimension_json)

def test_rust_serialization_boundary():
    # Confirm the config safely yields raw bytes for the Rust boundary
    cfg = TrickedConfig()
    cfg.proto.mcts.simulations = 42
    
    mcts_bytes = cfg.get_mcts_bytes()
    assert isinstance(mcts_bytes, bytes)
    assert len(mcts_bytes) > 0
