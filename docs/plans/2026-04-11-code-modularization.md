# Code Modularization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the massive `runner.rs`, `muzero.rs`, `inference.rs`, and `tests.rs` files into highly localized, single-responsibility Rust modules without altering any underlying runtime logic or performance.

**Architecture:** We will convert existing standalone files into module directories (e.g., `src/net/muzero.rs` becomes `src/net/muzero/mod.rs`), breaking their internal logic into isolated submodules for Thread Orchestration, Tensor Preparation, FFI Interop, and Network Transforms. We will also relocate integration tests to Cargo's native black-box `tests/` root directory.

**Tech Stack:** Rust (cargo), LibTorch/tch-rs, PyTorch C-FFI.

---

### Task 1: Create `tests/` Integration Suite

**Files:**
- Create: `tests/integration_tests/mod.rs`
- Create: `tests/integration_tests/dimensions.rs`
- Create: `tests/integration_tests/flow.rs`
- Create: `tests/integration_tests/stress.rs`
- Create: `tests/integration_tests/bptt.rs`
- Modify: `src/tests.rs` (Delete/Clear)

**Step 1: Write the integration framework**
Set up the black-box testing structure in Cargo. Create the standard integration test entrypoint. Move `get_test_config()` into a shared `tests/integration_tests/common.rs` or similar helper module.

**Step 2: Migrate test groups**
- Move `test_network_dimensions` to `dimensions.rs`.
- Move `test_flow_convergence` and `test_ema_polyak_averaging` to `flow.rs`.
- Move `test_transmission_stress_test` to `stress.rs`.
- Move `test_end_to_end_bptt_flow` to `bptt.rs`.
- Move `test_device_fallback_safety` and `test_nan_free_initialization` to a `safety.rs` test.

**Step 3: Run test to verify compilation and visibility**
Run: `cargo test --test integration_tests`
Expected: FAIL with visibility / privacy errors (since `tests/` runs identically to an external crate, many internal `pub(crate)` structures may need to become `pub` or tested differently).

**Step 4: Fix visibilities to pass tests**
Export the necessary types in the main library (`lib.rs` / `main.rs`) and fix any import paths (e.g., `crate::` becomes `tricked::`).
Run: `cargo test`
Expected: PASS

**Step 5: Commit**
```bash
git add src/tests.rs tests/
git commit -m "refactor: migrate tests.rs to cargo native tests/ folder integration suite"
```

---

### Task 2: Refactor `src/net/muzero.rs`

**Files:**
- Create: `src/net/muzero/mod.rs`
- Create: `src/net/muzero/network.rs`
- Create: `src/net/muzero/ffi.rs`
- Create: `src/net/muzero/support.rs`
- Create: `src/net/muzero/inference.rs`
- Create: `src/net/muzero/features.rs`
- Modify: `src/net/muzero.rs` (Delete, replacing with tree)

**Step 1: Extract `MuZeroNet` definition**
Move the `MuZeroNet` struct block and its `new()` initializer into `network.rs`. Adjust visibilities.

**Step 2: Extract FFI and Support Logic**
Move `support_to_scalar_fused` and `scalar_to_support_fused` (and their `tricked_ops.so` unsafe C bindings) into `support.rs` and `ffi.rs`. Expose them to the struct.

**Step 3: Extract Feature Fallbacks**
Move the massive CPU-fallback tensor-packing algorithms (`extract_initial_features` and `extract_unrolled_features`) into `features.rs`. 

**Step 4: Re-bundle in `mod.rs`**
In `src/net/muzero/mod.rs`, ensure that `pub use network::MuZeroNet;` is present so that `crate::net::MuZeroNet` functions exactly as it did before.

**Step 5: Run tests to verify**
Run: `cargo test`
Expected: PASS

**Step 6: Commit**
```bash
git add src/net/muzero.rs src/net/muzero/
git commit -m "refactor: split muzero.rs into isolated domains (ffi, support, features)"
```

---

### Task 3: Refactor `src/env/worker/inference.rs`

**Files:**
- Create: `src/env/worker/inference/mod.rs`
- Create: `src/env/worker/inference/loop_runner.rs`
- Create: `src/env/worker/inference/initial.rs`
- Create: `src/env/worker/inference/recurrent.rs`
- Modify: `src/env/worker/inference.rs` (Delete, replacing with tree)
- Modify: `src/env/worker/mod.rs` (Update imports)

**Step 1: Isolate the main loop**
Move `InferenceLoopParams` and the main `inference_loop` while loop to `loop_runner.rs`. 

**Step 2: Isolate Tensor Handlers**
Move the mathematically dense `process_initial_inference` function to `initial.rs`.
Move `process_recurrent_inference` to `recurrent.rs`.

**Step 3: Wire together in `mod.rs`**
Expose `inference_loop` publicly in `mod.rs` so the runner can still access `crate::env::worker::inference_loop`. Ensure the internal memory `SafeTensorGuard` utilities are shared appropriately.

**Step 4: Check build integrity**
Run: `cargo check`
Expected: PASS

**Step 5: Commit**
```bash
git add src/env/worker/inference.rs src/env/worker/inference/
git commit -m "refactor: decompose inference.rs into independent loop and process handlers"
```

---

### Task 4: Refactor `src/train/runner.rs`

**Files:**
- Create: `src/train/runner/mod.rs`
- Create: `src/train/runner/telemetry.rs`
- Create: `src/train/runner/workers.rs`
- Create: `src/train/runner/prefetch.rs`
- Create: `src/train/runner/optimizer.rs`
- Modify: `src/train/runner.rs` (Delete, replacing with tree)
- Modify: `src/train/mod.rs` (Update module inclusion)

**Step 1: Isolate Thread Spawners**
Move the `sysinfo` loop thread to `telemetry.rs`.
Move the environment worker loops (inference, self-play, reanalyze) to `workers.rs`.
Move the BPTT batch pipeliner to `prefetch.rs`.

**Step 2: Isolate the Optimization Execution**
Extract the `while active_training_flag` loop that executes `trainer::optimization::train_step` into an `optimizer.rs` loop logic.

**Step 3: Construct the Core Assembler**
In `mod.rs`, keep `pub fn run_training` strictly as the orchestrator. It should construct the devices, databases, and VarStores, and then call out to the spawner sub-functions from `workers.rs`, `prefetch.rs`, and finally yield execution to `optimizer.rs`.

**Step 4: Run comprehensive verification**
Run: `cargo build --release`
Run: `cargo test`
Expected: PASS (and identical behavior checking)

**Step 5: Commit**
```bash
git add src/train/runner.rs src/train/runner/
git commit -m "refactor: decompose giant run_training engine into sub-spawners"
```
