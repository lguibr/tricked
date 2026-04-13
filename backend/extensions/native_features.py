import torch
import libtricked_ops
import json
import os

# Load masks automatically at import time
def _load_masks():
    masks_path = os.path.join(os.path.dirname(__file__), "..", "masks.json")
    with open(masks_path, "r") as f:
        data = json.load(f)
    
    # Canonical mask: 2D array padded with -1
    max_canonical_len = max(len(m) for m in data["canonical"])
    canonical_padded = []
    for m in data["canonical"]:
        canonical_padded.append(m + [-1] * (max_canonical_len - len(m)))
    canonical_t = torch.tensor(canonical_padded, dtype=torch.int32, device="cuda")
    
    # Compact mask: 3D array padded
    max_compact_len = max(len(m) for m in data["compact"])
    compact_padded = []
    for m in data["compact"]:
        padded_m = m + [[0, 0]] * (max_compact_len - len(m))
        compact_padded.append(padded_m)
    compact_t = torch.tensor(compact_padded, dtype=torch.int64, device="cuda")
    
    # Standard mask: 3D array padded
    max_std_len = max(len(m) for m in data["standard"])
    std_padded = []
    for m in data["standard"]:
        padded_m = m + [[0, 0]] * (max_std_len - len(m))
        std_padded.append(padded_m)
    std_t = torch.tensor(std_padded, dtype=torch.int64, device="cuda")
    
    return canonical_t, compact_t, std_t

_CANONICAL, _COMPACT, _STANDARD = None, None, None

def _ensure_masks():
    global _CANONICAL, _COMPACT, _STANDARD
    if _CANONICAL is None:
        _CANONICAL, _COMPACT, _STANDARD = _load_masks()

def launch_extract_features(boards: torch.Tensor, avail: torch.Tensor, hist: torch.Tensor, acts: torch.Tensor, diff: torch.Tensor, unroll_steps: int = 1) -> torch.Tensor:
    _ensure_masks()
    return libtricked_ops.extract_features(boards, avail, hist, acts, diff, _CANONICAL, _COMPACT, _STANDARD, unroll_steps)