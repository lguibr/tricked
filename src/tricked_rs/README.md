# Tricked RS (Rust Backend)

Performance-critical paths of the engine, particularly the Monte Carlo Tree Search, reimplemented in safe and wildly optimized Rust bindings.

## Overview
- `lib.rs` and `node.rs`: Provide Python hooks using `pyo3` and fundamental graph node data structures.
- `board.rs`: Mirrors the bitmasking operation of `env/pieces.py`.
- `mcts.rs`: Multithreaded, zero-cost MCTS rollout logic ensuring high batch volumes required to saturating the Tensor cores on the python side.
