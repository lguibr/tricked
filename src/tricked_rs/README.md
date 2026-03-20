# Tricked Rust Core (`tricked_rs`)

## Abstract
The `tricked_rs` module constitutes the primary mathematical engine behind the Tri-Ecosystem. By actively resolving Python's Global Interpreter Lock (GIL) constraint via **PyO3 FFI bindings**, this native library evaluates complex environment dynamics, sequential bitboard topology, and Monte Carlo Latent Node initializations inside strict millisecond thresholds.

## Architectural Data Structures
### 1. The Triango 96-node Bitboard (`u128`)
Geometrically, the Triango board operates on a highly irregular 96-tile triangular mesh geometry avoiding canonical 2D plane logic. 
- Representation: The board state is heavily compressed into an integer `u128` (since 96 < 128). 
- Execution: Fragment collisions are calculated structurally using a pure `(target_mask & board_state) == 0` bitwise evaluation. Line clears process across all 48 symmetrical vectors explicitly tracking `(board_state & line) == line`.

### 2. State Duplication vs Extrapolation
Historically, reinforcement learning engines instantiate entirely new Python dictionaries/heaps per action unroll.
- The `GameStateExt` PyClass leverages strict `#derive([Clone])` and returns **new instances exclusively upon state mutation** (`apply_move()`).
- This functional structural dynamic ensures safe parallel reads natively passing the identical pointer downstream whenever the tree node acts essentially "read-only".

## Python Bindings (PyO3)
Maturin safely translates the `GameStateExt` C-struct into PyTorch-friendly Python endpoints seamlessly:
- `@property:` Read attributes transparently (e.g., `.score`).
- `__init__:` Bootstraps logic (spawning pieces via rand logic entirely in C++).
- Thread safety: `check_terminal()` iteratively proves game-over conditionals without exposing memory race conditions to the CPU pool workers running the Search instances.

## Build Requirements
This library strictly enforces `cargo`, governed structurally by `./run.sh` script execution leveraging `maturin develop --release`.
