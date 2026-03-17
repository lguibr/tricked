import sys

sys.path.append("/Users/lg/lab/tricked/")
from tricked.env.pieces import COORD_TO_INDEX, INDEX_TO_COORD, STANDARD_PIECES
from tricked.symmetry import D12_PERMUTATIONS


from typing import Any, Tuple, Optional

def normalize_shape(bitmask: int) -> Any:
    """
    Given a bitmask of a shape on the board, returns canonical offsets.
    The canonical anchor is the topmost, then leftmost coordinate.
    """
    # find all coords in the bitmask
    coords = []
    for i in range(96):
        if (bitmask & (1 << i)) != 0:
            coords.append(INDEX_TO_COORD[i])

    if not coords:
        return None, None

    # Find min Y (topmost), then min X (leftmost)
    # in INDEX_TO_COORD, coords are (cx, cy, cz) where cy is Y (goes down), cx is X (goes right), cz is parity?
    # Actually wait, cz is parity 1 or 0?
    # Let's just find the min lexicographical tuple.
    anchor = min(coords, key=lambda c: (c[1], c[0], c[2]))

    ax, ay, az = anchor

    offsets = []
    for cx, cy, cz in coords:
        offsets.append((cx - ax, cy - ay, cz - az))

    offsets.sort()
    # Parity mapping mathematically implies up/down based on column parity vs row
    parity = anchor[2]
    return tuple(offsets), parity == 1


def generate_closed_piece_set() -> None:
    unique_shapes = set()

    for p_id in range(12):
        for t_idx, perm in enumerate(D12_PERMUTATIONS):
            # start with a valid placement in the middle of the board to avoid edge clipping during rotation
            # wait, if a shape is on the edge, rotating it might clip it.
            # so we should use a placement near the center

            # just try ALL placements for this piece. Any valid rotation will just be a translated shape.
            for i in range(96):
                m = STANDARD_PIECES[p_id][i]
                if m == 0:
                    continue

                nm = 0
                for bit in range(96):
                    if (m & (1 << bit)) != 0:
                        nm |= 1 << perm[bit]

                offsets, anchor_is_up = normalize_shape(nm)
                if offsets:
                    unique_shapes.add((anchor_is_up, offsets))

    print(f"Discovered {len(unique_shapes)} unique canonical shapes under D12 closure.")

    # Render them as python code
    print("CLOSED_PIECE_DEFS = [")
    for anchor_is_up, offsets in sorted(list(unique_shapes)):
        req_up = anchor_is_up
        req_down = not anchor_is_up
        print(f"    PieceDef({req_up}, {req_down}, {list(offsets)}),")
    print("]")


if __name__ == "__main__":
    generate_closed_piece_set()
