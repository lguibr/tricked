"""
Standard Documentation for features.py.

This module supplies the core execution logic for the `mcts` namespace, heavily typed and tested for production distribution.
"""

import torch
from tricked_engine import GameStateExt as GameState

from tricked.env.constants import TOTAL_TRIANGLES, get_neighbors
from tricked.env.pieces import get_piece_overlay, get_valid_placement_mask


def extract_feature(
    state: GameState, 
    history: list[int] | None = None, 
    action_history: list[int] | None = None, 
    difficulty: int = 1
) -> torch.Tensor:
    
    feature = torch.zeros(20, TOTAL_TRIANGLES, dtype=torch.float32)

    def fill_channel(channel_idx: int, board_int: int) -> None:
        for i in range(TOTAL_TRIANGLES):
            if (board_int >> i) & 1:
                feature[channel_idx, i] = 1.0

    history = history or []
    
    fill_channel(0, state.board)
    
    for i in range(1, 8):
        if len(history) >= i:
            fill_channel(i, history[-i])
        else:
            fill_channel(i, state.board)  

    action_history = action_history or []
    for i in range(3):
        if len(action_history) > i:
            act = action_history[-(i+1)]
            slot = act // TOTAL_TRIANGLES
            idx = act % TOTAL_TRIANGLES
            if idx < TOTAL_TRIANGLES:
                feature[8 + i, idx] = (slot + 1) * 0.33

    for slot in range(3):
        p_id = state.available[slot]
        if p_id == -1:
            continue
        overlay = get_piece_overlay(p_id)
        valid_mask = get_valid_placement_mask(p_id, state.board)
        for i in range(TOTAL_TRIANGLES):
            if overlay[i] == 1:
                feature[11 + (slot * 2), i] = 1.0
            if valid_mask[i] == 1:
                feature[12 + (slot * 2), i] = 1.0

    feature[17, :] = 1.0 / 22.0

    feature[18, :] = float(difficulty) / 6.0  

    for i in range(TOTAL_TRIANGLES):
        is_filled = (state.board >> i) & 1
        if not is_filled:
            
            surrounded = True
            for n in get_neighbors(i):
                if not ((state.board >> n) & 1):
                    surrounded = False
                    break
            if surrounded:
                feature[19, i] = 1.0  

    return feature
