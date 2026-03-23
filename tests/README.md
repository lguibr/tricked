# Tricked Tests

Test harness utilizing pytest infrastructure to guarantee algorithmic soundness spanning geometry, network prediction shape alignment, and zero-sum parity.

## Modules
- `test_pieces_coverage.py` & `test_edges.py`: Asserts physical space definitions haven't drifted.
- `test_symmery.py`: Verifies Dihedral Group D12 rotation accuracy.
- `test_mcts.py` / `test_self_play.py`: Runs microscopic self-play iterations for deterministic outcomes.

Run via `./test.sh` at the project root.
