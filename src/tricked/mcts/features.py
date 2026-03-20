"""
Standard Documentation for features.py.

This module supplies the core execution logic for the `mcts` namespace, heavily typed and tested for production distribution.
"""

import torch

from tricked.env.constants import TOTAL_TRIANGLES, get_neighbors
from tricked.env.pieces import get_piece_overlay, get_valid_placement_mask
from tricked.env.state import GameState


def extract_feature(
    state: GameState, 
    history: list[int] | None = None, 
    action_history: list[int] | None = None, 
    difficulty: int = 1
) -> torch.Tensor:
    # Build a [20, 96] spatial tensor
    # 0-7: Board History (t, t-1, ... t-7)
    # 8-10: Action History (last 3 actions)
    # 11-12: Slot 0 Geom, Mask
    # 13-14: Slot 1 Geom, Mask
    # 15-16: Slot 2 Geom, Mask
    # 17: Probability Map (Remaining Pieces)
    # 18: Difficulty Scalar
    # 19: Explicit Hole Mask

    feature = torch.zeros(20, TOTAL_TRIANGLES, dtype=torch.float32)

    def fill_channel(channel_idx: int, board_int: int) -> None:
        bin_str = bin(board_int)[2:].zfill(TOTAL_TRIANGLES)[::-1]
        for i in range(TOTAL_TRIANGLES):
            if i < len(bin_str) and bin_str[i] == "1":
                feature[channel_idx, i] = 1.0

    # Task 1: Deeper Temporal History Projection (8 frames)
    history = history or []
    # Current board is t
    fill_channel(0, state.board)
    
    # Fill up to 7 past frames
    for i in range(1, 8):
        if len(history) >= i:
            fill_channel(i, history[-i])
        else:
            fill_channel(i, state.board)  # Pad with current board

    # Task 2: Action Trajectory Broadcast Planes
    action_history = action_history or []
    for i in range(3):
        if len(action_history) > i:
            act = action_history[-(i+1)]
            slot = act // TOTAL_TRIANGLES
            idx = act % TOTAL_TRIANGLES
            if idx < TOTAL_TRIANGLES:
                feature[8 + i, idx] = 1.0

    # Channels 11-16: Piece Overlays and Valid Masks
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

    # Task 3: Stochastic Piece Pool Probability Map (approximate uniform prior)
    feature[17, :] = 1.0 / 22.0

    # Task 4: Curriculum/Turn Scalar Broadcaster
    feature[18, :] = float(difficulty) / 6.0  # Normalized to max difficulty 6

    # Task P2: Deterministic "Dead Space" Injection (Explicit Hole Mask)
    board_bin = bin(state.board)[2:].zfill(TOTAL_TRIANGLES)[::-1]
    for i in range(TOTAL_TRIANGLES):
        is_filled = (i < len(board_bin) and board_bin[i] == "1")
        if not is_filled:
            # Check if all physical neighbors are filled
            neighbors = get_neighbors(i)
            surrounded = True
            for n in neighbors:
                n_filled = (n < len(board_bin) and board_bin[n] == "1")
                if not n_filled:
                    surrounded = False
                    break
            if surrounded:
                feature[19, i] = 1.0  # Trap Map Hole identified

    return feature
