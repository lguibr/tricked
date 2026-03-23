"""
Standard Documentation for search.py.

This module supplies the core execution logic for the `mcts` namespace, heavily typed and tested for production distribution.
"""

from typing import Any

import torch
import tricked_engine
from tricked_engine import GameStateExt as GameState

from tricked.model.network import MuZeroNet


class DummyRoot:
    def __init__(self, value: float):
        self.value = value

class MuZeroMCTS:
    def __init__(self, model: MuZeroNet, device: torch.device, hw_config: dict[str, Any] | None = None):
        self.model = model
        self.device = device
        self.hw_config = hw_config or {}
        
        import zmq
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.DEALER)
        self.zmq_port = self.hw_config.get("zmq_inference_port", "tcp://127.0.0.1:5555")
        self.zmq_socket.connect(self.zmq_port)

    def search(
        self, 
        root_state: GameState, 
        history: list[int] | None = None, 
        action_history: list[int] | None = None,
        difficulty: int = 1,
        simulations: int = 50,
        hw_config: dict[str, Any] | None = None
    ) -> tuple[int | None, dict[int, int], Any]:
        """
        Delegates pure structural processing to Native Rust MCTS.
        - Exclusively handles the `INITIAL` ZMQ GPU fetch inside Python.
        - `RECURRENT` passes execute entirely in fully-compiled Rust IPC loops.
        """
        with torch.no_grad():
            import pickle
            payload = pickle.dumps({
                "board": root_state.board,
                "available": root_state.available,
                "history": history or [],
                "action_history": action_history or [],
                "difficulty": difficulty
            })
            self.zmq_socket.send_multipart([b"INITIAL", payload])
            h0_bytes, policy_bytes = self.zmq_socket.recv_multipart()
            
            max_gumbel_k = self.hw_config.get("max_gumbel_k", 8)
            gumbel_scale = hw_config.get("gumbel_scale", 1.0) if hw_config else 1.0

            best_action, visits, root_value = tricked_engine.mcts_search(
                h0_bytes,
                policy_bytes,
                root_state,
                simulations,
                max_gumbel_k,
                gumbel_scale,
                self.zmq_port
            )

            if best_action == -1:
                return None, {}, DummyRoot(0.0)

            return best_action, visits, DummyRoot(root_value)
