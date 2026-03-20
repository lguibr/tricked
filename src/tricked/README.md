# Python MuZero Engine (`tricked`)

## Abstract
The `tricked` core embodies the complete implementation of the EfficientZero V2 and AlphaZero reinforcement learning algorithms operating over the Triango coordinate system. It dictates the architectural lifecycle of generating self-play trajectories, buffering historical sequences, and stabilizing Neural Network weights using massive GPU backpropagation unrolls.

## Architectural Layers

### 1. `training/` (Backpropagation & Hardware Loops)
Centralizes the mathematics needed to iteratively converge the Neural Network via Monte Carlo estimation.
- **`trainer.py`**: Executes BPTT (Backpropagation Through Time) over $K$-unroll steps (default: 5). Operates using modern **M0RV Advantage-Weighted targets** and **Contrastive Self-Supervised Projection (SimSiam)**. Gradients strictly flow inside `.register_hook(0.5)` to halve magnitude dynamically.
- **`self_play.py`**: Harnesses `multiprocessing` to isolate search evaluations simultaneously. Orchestrates Dirichlet noise injection inside the Latent Node expansion for bounded organic discovery. 

### 2. `mcts/` (Procedural Tree Search)
Implements the core Latent Monte Carlo Tree Search heavily inspired by `Gumbel MuZero`.
- Abandons standard UCB counts for the mathematical Sequential Halving UCB formulations (`PUCT`).
- Values derived during search bypass the Environment immediately, operating purely on Latent States (`model.recurrent_inference`) avoiding IPC overhead for physics translations.

### 3. `model/` (Neural Representation & Dynamics)
Constructs the actual PyTorch `nn.Module` orchestrating topological knowledge capability.
- Implements a sophisticated `GraphConv1d` layer. Standard convolutions hallucinate structural shapes when applied to 96-node triangular meshes; this graph-centric operation maps directly over the constant Adjacency Matrix (`A`) preventing grid-bleed.
- Employs Newton-Raphson hardware-accelerated loops for `support_to_scalar` inverted Symlog transforms.

## Execution Pattern (`main.py`)
`main.py` is an infinite-execution daemon. It runs isolated Self-Play clusters in epoch intervals, retrieves the absolute best 10,000 game frames into the `ReplayBuffer`, heavily mutates the PyTorch checkpoint using `torch.optim.Adam` and `StepLR`, and inherently scales the `Curriculum/Difficulty` mechanism automatically upon consecutive score mastery.
