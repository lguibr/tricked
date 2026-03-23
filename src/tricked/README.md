# Tricked Core Engine

The core modeling engine for Tricked, defining the geometries, standard pieces, rules of engagement, and neural/MCTS infrastructure.

## Structure
- `env/`: Board state, geometry mapping (`constants.py`), and piece masking rules (`pieces.py`, `piece_defs.py`).
- `mcts/`: Monte Carlo Tree Search routines, features engineering, and fast traversal.
- `model/`: PyTorch neural networks optimized for spatial awareness of the Tricked board via specific 2D convolutional layouts.
- `training/`: Core loops for self-play, evaluation against random/human baselines, and synchronization with `redis_logger` for metrics.

## Philosophy
Code here is maintained clean of inline operational noise, leaning on docstrings and semantic naming functions.
