"""
Human Interactive CLI for Tricked Game.
Supports difficulty filters that limit the maximum triangle bounds allowed to spawn.
"""

import os
import sys

# Ensure the module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random

from tricked.env.pieces import STANDARD_PIECES
from tricked.env.state import GameState
from tricked.print_pieces import str_piece_mask


def get_difficulty_filtered_pieces(max_triangles: int) -> list[int]:
    """Return piece IDs (0..25) that possess <= max_triangles."""
    valid_ids = []
    for p_id in range(len(STANDARD_PIECES)):
        # Find the first non-zero placement to count triangles
        for mask in STANDARD_PIECES[p_id]:
            if mask != 0:
                triangles = int(mask).bit_count()
                if triangles <= max_triangles:
                    valid_ids.append(p_id)
                break
    return valid_ids


def render_board(board_mask: int) -> None:
    print("\n--- Current Board State ---")
    print(str_piece_mask(board_mask))
    print("---------------------------\n")


def human_loop() -> None:
    print("Welcome to Tricked: AlphaZero Mathematical Engine Terminal Edition")
    print("Select Difficulty (Max triangles per piece):")
    print("1) Easy (1-Triangle Only)")
    print("3) Normal (Up to 3-Triangles)")
    print("6) Hard (All Pieces Allowed)")

    try:
        max_t = int(input("> ") or "6")
    except ValueError:
        max_t = 6

    valid_piece_ids = get_difficulty_filtered_pieces(max_t)
    print(f"Loaded {len(valid_piece_ids)} symmetric shapes bound by <= {max_t} complexity.")

    state = GameState()

    # Overwrite the standard random tray with our difficulty filter initially
    state.available = [
        random.choice(valid_piece_ids),
        random.choice(valid_piece_ids),
        random.choice(valid_piece_ids),
    ]

    while not state.terminal:
        render_board(state.board)

        print(f"Score: {state.score} | Pieces Left: {state.pieces_left}")
        print("Tray Contains:")
        for slot, p_id in enumerate(state.available):
            if p_id != -1:
                print(
                    f"Slot {slot} -> Piece ID {p_id} ({int(next(m for m in STANDARD_PIECES[p_id] if m != 0)).bit_count()} triangles)"
                )

        # User input loop
        move = input("\nEnter move [slot index] e.g. '0 45', or 'q' to quit: ")
        if move.lower().strip() == "q":
            print("Cowardly abandoning the equation...")
            break

        try:
            slot_str, idx_str = move.split()
            slot = int(slot_str)
            idx = int(idx_str)

            p_id = state.available[slot]
            if p_id == -1:
                print("Invalid slot. It is empty.")
                continue

            mask = STANDARD_PIECES[p_id][idx]
            if mask == 0:
                print("Invalid coordinate placement logic for this shape.")
                continue

            if (state.board & mask) != 0:
                print("Collision! Those coordinates are physically occupied.")
                continue

            # Perform Move
            next_state = state.apply_move(slot, idx)
            if next_state is None:
                print("Engine rejected move (Logical bounds check failed).")
                continue

            state = next_state

            # Post-move validation: if tray was refilled, strictly enforce difficulty bounds
            if state.pieces_left == 3:
                # Engine refilled it with standard randomness. We overwrite it securely.
                state.available = [
                    random.choice(valid_piece_ids),
                    random.choice(valid_piece_ids),
                    random.choice(valid_piece_ids),
                ]
                # Recheck engine terminal logic in case the filtered spawn trapped the player
                state.check_terminal()

        except ValueError:
            print(
                "Invalid syntax. Specify slot (0/1/2) and geometry index (0..95) separated by a space."
            )
            continue

    print("\n====================")
    print(f"GAME OVER! Final Mathematical Score: {state.score}")
    render_board(state.board)


if __name__ == "__main__":
    human_loop()
