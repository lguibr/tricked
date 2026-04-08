# Large Source Files Analysis (> 240 lines)

This study identifies all source code files (`.rs`, `.py`, `.ts`, `.tsx`) in the workspace that exceed 240 lines of code, excluding build artifacts and dependencies (`venv`, `node_modules`, `target`, `.git`, `dist`).

For the most extreme cases, an architectural analysis is provided to explain *why* the file is so large and *how* to refactor it utilizing data modules, locality, and elegant design patterns.

---

## 📊 File Size Metrics (Lines & Characters)

The following 41 files exceed the 240-line threshold:

| Lines | Characters | File Path |
|-------:|-----------:|-----------|
| 8,654 | 155,521 | `./src/core/constants.rs` |
| 959 | 40,335 | `./src/train/runner.rs` |
| 853 | 31,624 | `./src/net/muzero.rs` |
| 822 | 31,016 | `./control_center/src/components/OptunaStudyDashboard.tsx` |
| 779 | 23,807 | `./control_center/src/components/ui/sidebar.tsx` |
| 760 | 26,282 | `./src/queue.rs` |
| 669 | 21,446 | `./control_center/src-tauri/src/commands.rs` |
| 603 | 22,191 | `./src/tests.rs` |
| 596 | 23,346 | `./src/train/optimizer/optimization.rs` |
| 590 | 21,762 | `./src/env/worker/inference.rs` |
| 518 | 19,058 | `./src/performance_benches.rs` |
| 518 | 16,944 | `./control_center/src/components/MetricsDashboard.tsx` |
| 500 | 19,443 | `./src/env/worker/self_play.rs` |
| 498 | 19,900 | `./src/train/buffer/batcher.rs` |
| 452 | 15,981 | `./src/sumtree.rs` |
| 447 | 15,294 | `./src/core/board_tests.rs` |
| 428 | 16,061 | `./src/train/tune.rs` |
| 425 | 13,755 | `./control_center/src/components/dashboard/HardwareMiniDashboard.tsx` |
| 416 | 16,029 | `./src/train/buffer/state.rs` |
| 405 | 17,830 | `./control_center/src/components/execution/RunsSidebarList.tsx` |
| 404 | 13,393 | `./control_center/src/components/playground/TrickedPlayground.tsx` |
| 403 | 14,178 | `./src/bin/mc_metrics.rs` |
| 396 | 13,538 | `./control_center/src/components/execution/CreateStudySidebar.tsx` |
| 393 | 17,788 | `./src/telemetry.rs` |
| 389 | 11,747 | `./src/core/features.rs` |
| 383 | 13,057 | `./control_center/src/components/execution/HydraConfigViewer.tsx` |
| 374 | 13,103 | `./src/train/buffer/writer.rs` |
| 366 | 10,854 | `./control_center/src/components/execution/CreateSimpleRunSidebar.tsx` |
| 351 | 11,567 | `./src/cli.rs` |
| 339 | 12,363 | `./src/node.rs` |
| 321 | 14,394 | `./control_center/src-tauri/src/db.rs` |
| 313 | 8,971 | `./control_center/src/components/dashboard/MetricChart.tsx` |
| 307 | 10,670 | `./src/mcts/gumbel.rs` |
| 292 | 10,106 | `./control_center/src/App.tsx` |
| 290 | 10,290 | `./control_center/src-tauri/src/execution.rs` |
| 255 | 8,536 | `./src/train/buffer/core.rs` |
| 252 | 6,916 | `./control_center/src/store/useAppStore.ts` |
| 251 | 9,305 | `./src/mcts/tree_ops.rs` |
| 244 | 8,307 | `./src/core/board.rs` |
| 242 | 7,186 | `./control_center/src/components/execution/LossStackedArea.tsx` |
| 242 | 6,065 | `./control_center/src/components/ui/field.tsx` |

---

## 🏗️ Deep Architectural Analysis & Refactoring Proposals

For the top monolithic files, here is a breakdown of why they have grown so large and how to beautifully refactor them using **Data Modules** and **Locality of Behavior (LoB)** to maintain clean, scalable boundaries.

### 1. `src/core/constants.rs` (8,654 lines)

* **Why it's so big:** It contains massive hardcoded 2D arrays and bitboards (`ALL_MASKS`, `STANDARD_PIECES`). Rust arrays of this size bloat the syntax tree and significantly slow down compilation times.
* **How to fix:** Move data to a true **Data Module**. Export the raw arrays to a binary (`.bin`) or JSON file in an `assets/` folder. Load them at compile time using `include_bytes!()` and decode them safely, or use `lazy_static!` to parse them at runtime. This separates the rigid schema logic from gigabytes of static data.

### 2. `src/train/runner.rs` (959 lines)

* **Why it's so big:** Contains a classic "God function" (`run_training` is >800 lines) handling configuration, initialization, the primary training loop, telemetry sync, EMA backpropagation, and SQLite transactions all within a single block. Tests are also bundled at the bottom.
* **How to fix:** Implement the **State/Strategy Pattern**:
  * Extract the state into a `TrainingPipeline` struct.
  * Break `run_training` into smaller, encapsulated phases: `pipeline.initialize()`, `pipeline.step()`, `pipeline.sync_telemetry()`.
  * Extract the EMA updating logic into a dedicated module (`src/train/ema.rs`) to keep update mechanics localized.

### 3. `src/net/muzero.rs` (853 lines)

* **Why it's so big:** The `MuZeroNet` structure houses everything: the Representation, Dynamics, and Prediction network sub-modules, scaling mappings, tensor transformations, and >300 lines of unit testing logic.
* **How to fix:** Enforce boundary locality by splitting sub-networks into their own domains:
  * Create `src/net/representation.rs`, `src/net/dynamics.rs`, and `src/net/prediction.rs`.
  * Keep `src/net/muzero.rs` purely as the facade/orchestrator that stitches the sub-modules together.
  * Move the `cfg(test)` module to an external integration test or a dedicated `tests/` subdirectory to halve the file size instantly.

### 4. `control_center/src/components/OptunaStudyDashboard.tsx` (822 lines)

* **Why it's so big:** A frontend monolithic component that manages deep component state, parses backend configuration logic, fetches IPC Tauri payloads, and defines custom UI wrappers (like `GlassCard`) inline.
* **How to fix:** Adopt **Custom Hooks & Presentational Boundaries**:
  * Move Tauri IPC logic and state to a custom hook (e.g., `useOptunaStudy()`).
  * Extract `GlassCard` and layout blocks into the `components/ui/` directory.
  * Offload chart configuration (ECharts formatters) to a `lib/chart-utils.ts` data module, leaving this file strictly for compositional UI layout.

### 5. `src/queue.rs` (760 lines)

* **Why it's so big:** Packages the core `FixedInferenceQueue` and `QueueSlotGuard` with exceptionally heavy, 500+ line massively concurrent fuzzing test suites.
* **How to fix:** Simply move the test suite into a dedicated `tests/queue_tests.rs` file. By keeping tests local to the workspace but structurally decoupled to the `tests/` directory, the core logic stays under 200 lines of highly readable, locational code.

### 6. `control_center/src-tauri/src/commands.rs` (669 lines)

* **Why it's so big:** Handles all IPC (Inter-Process Communication) commands for the entire application (runs, configurations, tuning, playground, database flush).
* **How to fix:** Break into modular handlers within a `commands/` directory:
  * `commands/runs.rs`
  * `commands/tuning.rs`
  * `commands/playground.rs`
  * Register these loosely coupled modules natively to the Tauri builder in `main.rs`.

### 7. `src/train/optimizer/optimization.rs` (596 lines)

* **Why it's so big:** Features a highly complex 300-line `train_step` computing gradients alongside intense test cases simulating BPTT padding and unroll states.
* **How to fix:** Create a distinct `LossCalculator` data module to handle cross-entropy and scaling mathematical formulas. Extract the tensor preparations to a `BatchProcessor` struct.

### 8. `src/env/worker/inference.rs` (590 lines)

* **Why it's so big:** Features two monolithic routines `process_initial_inference` and `process_recurrent_inference`. They manage deep torch tensor memory handling, caching, device mapping, and thread pinning.
* **How to fix:** Encapsulate memory buffers inside a strictly defined `InferenceContext` struct. Create functional locational scopes for `TensorAllocation`, `DeviceSync`, and `CallbackDispatch`.

### 9. `control_center/src/components/MetricsDashboard.tsx` (518 lines)

* **Why it's so big:** Inline sub-components like `LayerNormsDisplay` bloat the parent component next to fetching intervals.
* **How to fix:** Relocate the `LayerNormsDisplay` into `components/execution/LayerNormsDisplay.tsx`. Move metric transformation array mappings into pure functions inside a `data_modules/` or `lib/` directory so the main file remains declarative UI.
