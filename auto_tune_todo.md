# Auto-Tune Refactoring & Stabilization Roadmap

This document outlines the planned improvements for the `auto_tune.py` toolkit and the Tricked AI Engine's experiment management lifecycle.

## Phase 1: Thread Lifecycle & Engine Isolation (Urgent)

**The Problem (Metric & Queue Leaking):**
As seen in the logs, a run locked to `p1_z1` (1 process, batch size 1) emitted a dynamic batching average of `85.0 / 85`. This mathematically proves that **threads from the previous run are surviving the termination signal**. 
Because large MCTS searches take longer than the 5-second cooldown sleep, the old threads are still executing `mcts_search()` when `auto_tune.py` sends the next `StartTraining` signal. The threads then incorrectly assume they are allowed to continue, bleeding their old configuration context into the new run's inference queue.

- [ ] **Implement Generation UUIDs:** Replace the globally shared `active_flag` boolean with an atomic `generation_id` counter. Threads will store the `generation_id` they were spawned with and instantly self-terminate if the global engine generation advances, destroying zombies permanently.
- [ ] **Inference Queue Purging:** Ensure the bounded ZMQ/Crossbeam channels are completely drained and discarded between training runs, preventing old batched requests from executing under a new run's context.

## Phase 2: Intelligent Auto-Sweep Heuristics

**The Problem:** 
A static Cartesian grid search filters configs statically, but executing all permutations in a randomized or interleaved order is highly inefficient when seeking optimal boundaries.

- [ ] **Orthogonal Divide & Conquer:** Instead of a full grid, sweep one hyperparameter dimension while anchoring the others to their optimal "juice spot" (e.g., `s16_d128`), map the gradient, and then switch axes.
- [ ] **Early Ejection (Warmup Phase):** Rather than blindly waiting 180 seconds for mathematically impossible networks (e.g., `d2048`), run a 15-second "probe" phase. If 0 games are completed, immediately cleanly abort the flight instead of waiting 165 more seconds.
- [ ] **Bayesian Optimization Framework:** Integrate a lightweight probabilistic model (e.g., Optuna/Ray Tune concepts) to automatically construct the next candidate permutations based on previous success (measured by `game/lines_cleared`).

## Phase 3: CLI Toolkit Usability & UI/UX

**The Problem:**
The tuner logs gracefully with `rich`, but it operates as a rigid, fire-and-forget shell script. It needs to operate as a full master control CLI.

- [ ] **Sweep Topology Dashboard:** Before executing, print a rich table summarizing the exact dimensions, the total combinatorial space (e.g., 243), the Golden Ratio filtered count (e.g., 81), and the estimated time to completion.
- [ ] **Interactive TUI (Textualize):** Refactor the terminal output into a live `textual` app with persistent sidebars. Allow the user to press `[S]` to skip the current configuration, `[P]` to pause the run, or `[Q]` to gracefully abort and save the best checkpoint so far.
- [ ] **Live Leaderboard:** Maintain a persistent top-5 leaderboard in the bottom panel sorted by `games_per_second` and `lines_cleared`, updating dynamically as permutations complete.
