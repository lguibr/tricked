"""
Standard Documentation for features.py.

This module supplies the core execution logic for the `mcts` namespace, heavily typed and tested for production distribution.
"""

import torch

from tricked.env.constants import TOTAL_TRIANGLES
from tricked.env.pieces import get_piece_overlay, get_valid_placement_mask
from tricked.env.state import GameState


def extract_feature(state: GameState, history: list[int] | None = None) -> torch.Tensor:
    # Build a [9, 96] spatial tensor
    # Channel 0: Current Board (t)
    # Channel 1: Board (t-2) - Oldest in history window
    # Channel 2: Board (t-1)
    # Channel 3: Geometry of piece in slot 0
    # Channel 4: Valid Mask of piece in slot 0
    # Channel 5: Geometry of piece in slot 1
    # Channel 6: Valid Mask of piece in slot 1
    # Channel 7: Geometry of piece in slot 2
    # Channel 8: Valid Mask of piece in slot 2

    feature = torch.zeros(9, 96, dtype=torch.float32)

    def fill_channel(channel_idx: int, board_int: int) -> None:
        bin_str = bin(board_int)[2:].zfill(TOTAL_TRIANGLES)[::-1]
        for i in range(TOTAL_TRIANGLES):
            if i < len(bin_str) and bin_str[i] == "1":
                feature[channel_idx, i] = 1.0

    fill_channel(0, state.board)

    if history and len(history) >= 2:
        fill_channel(1, history[-2])
        fill_channel(2, history[-1])
    else:
        fill_channel(1, state.board)
        fill_channel(2, state.board)

    # Channels 3-8: Piece Overlays and Valid Masks
    for slot in range(3):
        p_id = state.available[slot]
        if p_id == -1:
            continue
        overlay = get_piece_overlay(p_id)
        valid_mask = get_valid_placement_mask(p_id, state.board)
        for i in range(TOTAL_TRIANGLES):
            if overlay[i] == 1:
                feature[(slot * 2) + 3, i] = 1.0
            if valid_mask[i] == 1:
                feature[(slot * 2) + 4, i] = 1.0

    return feature
