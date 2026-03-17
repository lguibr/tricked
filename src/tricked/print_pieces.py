import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tricked.env.constants import ROW_LENGTHS, TOTAL_TRIANGLES, get_row_col, is_up
from tricked.env.pieces import STANDARD_PIECES


def str_piece_mask(m: int) -> str:
    """Returns a string representation of a raw bitboard mask."""
    if m == 0:
        return ""

    points = []
    for i in range(TOTAL_TRIANGLES):
        if (m & (1 << i)) != 0:
            r, c = get_row_col(i)
            # convert to approximate x, y for text printing
            # y goes down
            x = c - ROW_LENGTHS[r] * 0.5
            points.append((r, x, is_up(r, c)))

    if not points:
        return ""

    min_r = min(p[0] for p in points)
    max_r = max(p[0] for p in points)

    lines = []
    for r in range(min_r, max_r + 1):
        # find bounding columns for this printed block to center it roughly
        line = ""
        for c in range(ROW_LENGTHS[r]):
            idx = -1
            for p in range(TOTAL_TRIANGLES):
                pr, pc = get_row_col(p)
                if pr == r and pc == c:
                    idx = p

            if idx != -1 and (m & (1 << idx)) != 0:
                line += "^" if is_up(r, c) else "v"
            else:
                line += "."
        lines.append(" " * (15 - ROW_LENGTHS[r]) + line)

    return "\n".join(lines)


def print_piece(p_id: int) -> None:
    # find the first valid placement of p_id
    m = 0
    for i in range(TOTAL_TRIANGLES):
        if STANDARD_PIECES[p_id][i] != 0:
            m = STANDARD_PIECES[p_id][i]
            break

    if m == 0:
        return

    print(f"--- PIECE {p_id} ---")
    print(str_piece_mask(m))


if __name__ == "__main__":
    for i in range(12):
        print_piece(i)
