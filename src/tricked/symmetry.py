import math
from typing import Any

TOTAL_TRIANGLES = 96
ROW_LENGTHS = [9, 11, 13, 15, 15, 13, 11, 9]


def get_row_col(idx: int) -> tuple[int, int]:
    rem = idx
    for r in range(8):
        if rem < ROW_LENGTHS[r]:
            return r, rem
        rem -= ROW_LENGTHS[r]
    return -1, -1


def is_up(r: int, c: int) -> bool:
    if r < 4:
        return c % 2 == 0
    return c % 2 == 1


def generate_board_coordinates() -> list[tuple[int, float, float, bool]]:
    pts = []
    for i in range(96):
        row, col = get_row_col(i)
        y = -(row - 3.5)
        w = ROW_LENGTHS[row]
        x = col - (w - 1) / 2.0
        up = is_up(row, col)
        pts.append((i, x, y, up))
    return pts


def rotate_transform(x: float, y: float, degrees: float) -> tuple[float, float]:
    true_y = y * (math.sqrt(3) / 2.0)
    true_x = x * 0.5

    rad = math.radians(degrees)
    c = math.cos(rad)
    s = math.sin(rad)

    nx = true_x * c - true_y * s
    ny = true_x * s + true_y * c

    new_x = nx / 0.5
    new_y = ny / (math.sqrt(3) / 2.0)
    return new_x, new_y


def mirror_transform(x: float, y: float) -> tuple[float, float]:
    return -x, y


def compute_mapping(
    pts: list[tuple[int, float, float, bool]], transform_fn: Any
) -> tuple[int, ...]:
    mapping = [0] * 96

    for i, x, y, up in pts:
        nx, ny = transform_fn(x, y)

        best_j = -1
        best_d = 999999.0
        for j, jx, jy, jup in pts:

            d = (nx - jx) ** 2 + (ny - jy) ** 2
            if d < best_d:
                best_d = d
                best_j = j

        mapping[int(i)] = int(best_j)

    return tuple(mapping)


def generate_d12_permutations() -> list[tuple[int, ...]]:
    """Returns 12 permutation tuples mapping indices 0..95 to their new locations."""
    pts = generate_board_coordinates()
    perms = []

    for angle in [0, 60, 120, 180, 240, 300]:

        def t_fn(x: float, y: float, a: float = angle) -> tuple[float, float]:
            return rotate_transform(x, y, a)

        mapping = compute_mapping(pts, t_fn)
        perms.append(mapping)

    for angle in [0, 60, 120, 180, 240, 300]:

        def t_fn2(x: float, y: float, a: float = angle) -> tuple[float, float]:
            return rotate_transform(*mirror_transform(x, y), a)

        mapping = compute_mapping(pts, t_fn2)
        perms.append(mapping)

    return perms


D12_PERMUTATIONS = generate_d12_permutations()

if __name__ == "__main__":
    for i, p in enumerate(D12_PERMUTATIONS):
        s = set(p)
        print(f"Perm{i} Length {len(s)} expected 96")
        if len(s) != 96:

            missing = set(range(96)) - s
            print(f"  Missing: {missing}")

    p_mirror = D12_PERMUTATIONS[6]
    print("Row 3 Mirror Map: ", [p_mirror[i] for i in range(33, 48)])

    import sys

    sys.path.append("/Users/lg/lab/tricked/")
    try:
        from tricked.env.pieces import STANDARD_PIECES

        mask_to_action: dict[int, tuple[int, int]] = {}
        for p_id in range(12):
            for i in range(96):
                m = STANDARD_PIECES[p_id][i]
                if m != 0:
                    mask_to_action[m] = (p_id, i)

        is_closed = True
        broken_t: dict[int, set[int]] = {}
        for t_idx, perm in enumerate(D12_PERMUTATIONS):
            closed_for_t = True
            for p_id in range(12):
                for i in range(96):
                    m = STANDARD_PIECES[p_id][i]
                    if m == 0:
                        continue

                    nm = 0
                    for bit in range(96):
                        if (m & (1 << bit)) != 0:

                            nm |= 1 << perm[bit]

                    if nm not in mask_to_action:

                        closed_for_t = False
                        is_closed = False
                        if t_idx not in broken_t:
                            broken_t[t_idx] = set()
                        broken_t[t_idx].add(p_id)

            print(f"Transform {t_idx} Closure = {closed_for_t}")
            if not closed_for_t:
                print(f"  Broken Pieces: {sorted(list(broken_t[t_idx]))}")

    except Exception as e:
        print("Failed to check closure:", e)
