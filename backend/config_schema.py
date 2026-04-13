from backend.proto_out.tricked_pb2 import TrickedConfig as ProtoTrickedConfig
from google.protobuf.json_format import Parse, MessageToJson, ParseError
import json

class TrickedConfig:
    """
    A strict wrapper around the compiled Protobuf schema to ensure fail-fast guarantees
    and to expose seamless hooks for the pure byte extraction used in PyO3/Rust boundaries.
    """
    def __init__(self, proto_obj=None):
        if proto_obj is None:
            self.proto = ProtoTrickedConfig()
        else:
            self.proto = proto_obj

    def __getattr__(self, name):
        return getattr(self.proto, name)

    @classmethod
    def parse_raw(cls, json_str: str) -> "TrickedConfig":
        try:
            proto = Parse(json_str, ProtoTrickedConfig(), ignore_unknown_fields=False)
            return cls(proto)
        except ParseError as e:
            raise ValueError(f"Strict Protobuf Config Parsing Failed: {e}")

    def json(self) -> str:
        # Serializes cleanly to standard JSON if needed for Database saving
        return MessageToJson(self.proto, preserving_proto_field_name=True)
        
    def get_full_bytes(self) -> bytes:
        return self.proto.SerializeToString()
        
    def get_mcts_bytes(self) -> bytes:
        return self.proto.mcts.SerializeToString()

