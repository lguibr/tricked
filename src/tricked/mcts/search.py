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
    def __init__(self, model: MuZeroNet, device: torch.device, hw_config: Any = None):
        self.model = model
        self.device = device
        self.model = model
        self.device = device
        self.hw_config = hw_config or {}

    def search(
        self, 
        root_state: GameState, 
        history: list[int] | None = None, 
        action_history: list[int] | None = None,
        difficulty: int = 1,
        simulations: int = 50,
        hw_config: Any = None
    ) -> tuple[int | None, dict[int, int], Any]:
        """
        Delegates pure structural processing to Native Rust MCTS.
        - Exclusively handles the `INITIAL` ZMQ GPU fetch inside Python.
        - `RECURRENT` passes execute entirely in fully-compiled Rust IPC loops.
        """
        with torch.no_grad():
            import numpy as np

            from tricked.mcts.features import extract_feature

            x = extract_feature(root_state)
            if isinstance(x, np.ndarray):
                x_t = torch.from_numpy(x).unsqueeze(0).to(self.device).float()
            else:
                x_t = x.clone().detach().unsqueeze(0).to(self.device).float()
            
            with torch.autocast(device_type=self.device.type, enabled=(self.device.type=="cuda")):
                h0 = self.model.representation(x_t)
                _, p0, _ = self.model.prediction(h0)
            
            h0_bytes = h0.cpu().numpy().astype(np.float32).tobytes()
            policy_bytes = p0.cpu().numpy().astype(np.float32).tobytes()

            # Pass safely to C++ LibTorch module in Rust
            max_gumbel_k = self.hw_config.max_gumbel_k # type: ignore
            gumbel_scale = hw_config.gumbel_scale if hw_config else 1.0

            best_action, visits, root_value = tricked_engine.mcts_search(
                h0_bytes,
                policy_bytes,
                root_state,
                simulations,
                max_gumbel_k,
                gumbel_scale
            )

            if best_action == -1:
                return None, {}, DummyRoot(0.0)

            return best_action, visits, DummyRoot(root_value)
