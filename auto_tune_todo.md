# Auto-Tune Refactoring & Stabilization Roadmap

This document outlines the planned improvements for the `auto_tune.py` toolkit and the Tricked AI Engine's experiment management lifecycle.

## 🏆 Achieved Milestones

- **Discovered the "Juice Spot"**: Uncovered mathematical proof that the engine favors massive width over depth, hitting **0.58 Games/Second** with `p64_z42` and `b2048`.
- **Golden Ratio Filtering**: Implemented hard bounding logic in `auto_tune.py` to immediately discard unbalanced configurations (enforcing a 1.5x - 2.0x ZMQ-to-Worker ratio) with clear CLI warnings.
- **Chronological TensorBoard Sorting**: Enforced strict `YY-MM-DD-HH-MM-SS` timestamp prefixes so TensorBoard natively groups and tracks tuning runs sequentially.
- **Focused Cartesian Grid**: Radically pruned the hyperparameter search space away from timeout-inducing gargantuan dimensions to intensely map the high-yield `p64/b2048` subspace.

## Phase 1: Thread Lifecycle & Engine Isolation (Urgent)

**The Problem (Metric & Queue Leaking):**
As seen in the logs, a run locked to `p1_z1` (1 process, batch size 1) emitted a dynamic batching average of `85.0 / 85`. This mathematically proves that **threads from the previous run are surviving the termination signal**. 
Because large MCTS searches take longer than the 5-second cooldown sleep, the old threads are still executing `mcts_search()` when `auto_tune.py` sends the next `StartTraining` signal. The threads then incorrectly assume they are allowed to continue, bleeding their old configuration context into the new run's inference queue.

- [ ] **Implement Generation UUIDs:** Replace the globally shared `active_flag` boolean with an atomic `generation_id` counter. Threads will store the `generation_id` they were spawned with and instantly self-terminate if the global engine generation advances, destroying zombies permanently.
- [ ] **Inference Queue Purging:** Ensure the bounded ZMQ/Crossbeam channels are completely drained and discarded between training runs, preventing old batched requests from executing under a new run's context.

## Phase 2: SOTA Auto-Sweep Heuristics (Optuna)

**The Problem:** 
A static Cartesian grid search filters configs statically, but executing all permutations in a randomized or interleaved order is highly inefficient. Building custom Bayesian optimizers or grid search early-ejection heuristics is reinventing the wheel.

- [ ] **Optuna Integration:** Replace the manual grid search with `optuna`. Use `TPESampler` (Tree-structured Parzen Estimator) to intelligently explore the "juice spot" configurations without hardcoding Cartesian loops.
- [ ] **Automated Pruning:** Use Optuna's `MedianPruner` or `HyperbandPruner` to instantly abort underperforming trials based on intermediate metrics (e.g., games_per_second at 15-second marks) instead of manually building "Early Ejection" logic.

## Phase 3: Observability & Dashboarding

**The Problem:**
Building a custom terminal UI (with `textual` or `rich`) and a custom leaderboard requires massive boilerplate and maintenance. We need to operate as a full master control CLI without the overhead.

- [ ] **Optuna Dashboard:** Leverage `optuna-dashboard` to provide a rich, interactive web UI for real-time experiment tracking, hyperparameters importance evaluation, and Pareto front visualization. (Tip: Also available directly as a VS Code or Jupyter Lab extension!)
- [ ] **Eliminate Custom TUI:** Strip out planning for custom Terminal UIs and interactive pause/skip commands, relying on Optuna's native study management and SQLite backend for resuming or killing trials.
