import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tricked.env.pieces import STANDARD_PIECES
from tricked.symmetry import D12_PERMUTATIONS

perm = D12_PERMUTATIONS[1]

mask_to_pid = {}
for p_id, masks in enumerate(STANDARD_PIECES):
    for m in masks:
        if m != 0:
            mask_to_pid[m] = p_id

rotation_map = {}

for p_id, masks in enumerate(STANDARD_PIECES):
    found = False
    for m in masks:
        if m == 0:
            continue

        nm = 0
        for bit in range(96):
            if (m & (1 << bit)) != 0:
                nm |= 1 << perm[bit]

        if nm in mask_to_pid:
            rotation_map[p_id] = mask_to_pid[nm]
            found = True
            break

    if not found:
        print(f"Warning: Could not find valid on-board rotation for Piece {p_id}")

print("const ROTATION_MAP = {")
for k, v in rotation_map.items():
    print(f"    {k}: {v},")
print("};")
