# mypy: ignore-errors
import time

# C++ Native Implementations (Legacy) -> Replaced by Rust
from tricked_engine import GameStateExt as RsGameState

# Python Pure Implementations
from tricked.env.state import GameState as PyGameState

# No initialization needed for pure rust module

# Because search.py was modified to use CppNode, we need to temporarily inject the PyNode
# back in or just mock the pure python pass for benchmarking.
# Actually, we can just run a native `select_child` and `expand` recursive simulation benchmark!


def benchmark_simulation():
    # 1. Pure Python GameState Deep Expansion (No neural net, raw throughput)
    print("--- Simulating 5000 random deep traversals ---")

    start_time = time.time()
    for _ in range(5000):
        s = PyGameState()
        while not s.terminal:
            # Find first available move manually without MCTS
            slot, idx = -1, -1
            from tricked.env.pieces import STANDARD_PIECES

            for s_i in range(3):
                p_id = s.available[s_i]
                if p_id == -1:
                    continue
                # find valid overlay
                has_move = False
                for m_idx, m_mask in enumerate(STANDARD_PIECES[p_id]):
                    if m_mask != 0 and (s.board & m_mask) == 0:
                        slot, idx = s_i, m_idx
                        has_move = True
                        break
                if has_move:
                    break

            if slot != -1:
                s = s.apply_move(slot, idx)
                if s is None:
                    break
            else:
                break
    py_time = time.time() - start_time
    print(f"Python GameState Traversal Time: {py_time:.4f}s")

    # 2. Native Rust GameState Deep Expansion
    start_time = time.time()
    for _ in range(5000):
        s = RsGameState()
        while not s.terminal:
            from tricked.env.pieces import STANDARD_PIECES

            has_move = False
            for s_i in range(3):
                p_id = s.available[s_i]
                if p_id == -1:
                    continue
                for m_idx, m_mask in enumerate(STANDARD_PIECES[p_id]):
                    next_s = s.apply_move(s_i, m_idx)
                    if next_s is not None:
                        s = next_s
                        has_move = True
                        break
                if has_move:
                    break
            if not has_move:
                break
    cpp_time = time.time() - start_time
    print(f"Rust Node+GameState Traversal Time: {cpp_time:.4f}s")

    speedup = py_time / cpp_time if cpp_time > 0 else 0
    print(f"--> NATIVE RUST IS {speedup:.2f}x FASTER! <--")


if __name__ == "__main__":
    benchmark_simulation()
