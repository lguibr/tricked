import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tricked.env.pieces import STANDARD_PIECES
from tricked.symmetry import D12_PERMUTATIONS


def find_rotation_map() -> dict[int, int]:
    rotation_map = {}
    
    # D12_PERMUTATIONS[1] is 60 deg, [2] is 120, [3] is 180, [4] is 240, [5] is 300 (-60 degrees)
    # Right click rotation in UI = clockwise = 300 degrees mathematical.
    perm = D12_PERMUTATIONS[5]
    
    num_pieces = len(STANDARD_PIECES)
    for p_id in range(num_pieces):
        # find the first valid non-zero mask for p_id near the center
        m = 0
        best_dist = 999999
        for mask in STANDARD_PIECES[p_id]:
            if mask != 0:
                dist = sum(abs(bit - 48) for bit in range(96) if (mask & (1 << bit)) != 0)
                if dist < best_dist:
                    best_dist = dist
                    m = mask
        
        if m == 0:
            continue
            
        nm = 0
        for bit in range(96):
            if (m & (1 << bit)) != 0:
                nm |= 1 << perm[bit]
                
        found = False
        for other_p_id in range(num_pieces):
            if nm in STANDARD_PIECES[other_p_id]:
                rotation_map[p_id] = other_p_id
                found = True
                break
                
        if not found:
            print(f"ERROR: No match for p_id {p_id}")
            
    return rotation_map

if __name__ == "__main__":
    rmap = find_rotation_map()
    print("ROTATION_MAP = {")
    for k in sorted(rmap.keys()):
        print(f"    {k}: {rmap[k]},")
    print("}")
