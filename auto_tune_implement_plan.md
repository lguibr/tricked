# Auto-Tune Advanced Implementation Plan

## Goal Description
Upgrade the `auto_tune.py` script and the Tricked AI Engine telemetry to prevent zombie threads from bleeding metrics across runs, implement an intelligent Bayesian search heuristic, and provide a rich Terminal UI for live experiment control.

## Proposed Changes

### Engine Threading & Core
#### [MODIFY] src/main.rs
- Inject a `generation_id` UUID into the `StartTraining` HTTP command payload.
- Update tracking channels so each thread maintains its own birth generation variable.

#### [MODIFY] src/selfplay.rs
- Validate `generation_id` before invoking ZMQ or updating telemetry, enforcing instant self-termination on generation mismatch.
- Ensure `FixedInferenceQueue` dynamically scrubs old generation payloads.

### Auto-Tuner Heuristics & Control
#### [MODIFY] scripts/auto_tune.py
- **Optuna Integration**: Replace `itertools.product` Cartesian grid mapping with `optuna.create_study()`. Use `TPESampler` for intelligent probabilistic search.
- **Optuna Pruning**: Add `trial.report` and `trial.should_prune` during the existing evaluation loops to abort slow configurations early.
- **Persistent Storage & UI**: Configure a SQLite backend for Optuna to allow out-of-the-box use of `optuna-dashboard` instead of building a custom Terminal UI.
