# Tricked Scripts

This directory contains utility scripts for the Tricked engine.

## Code Generators
Scripts in `generators/` synthesize code and data tables based on the core python models:
* `generate_all.py`: Executes all generators.
* `generate_rotation_map.py` & `compute_rotation_map.py`: Calculate mappings for rotational symmetries of the board and pieces.
* `generate_rust_constants.py`: Generates the Rust constants (`constants.rs`) containing standard pieces and masks to ensure engine consistency across Python and Rust.

## Utilities
* `benchmark.py`: Performs performance testing on the engine components (MCTS, simulator, neural networks).
* `play_human.py`: A terminal-based interface allowing humans to play the game securely against engine constraints or difficulty filters.
