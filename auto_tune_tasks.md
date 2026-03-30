# Auto-Tune & Telemetry Action Checklist

- [ ] **Phase 1: Thread Isolation Layer**
  - [ ] Generate atomic `generation_uuid` per `auto_tune.py` start command.
  - [ ] Modify `main.rs` Thread Spawns to deeply carry the UUID into the worker scope.
  - [ ] Kill active jobs in `selfplay.rs` if `thread_uuid != active_engine_uuid`.
- [ ] **Phase 2: SOTA Auto-Tuning Integration (Optuna)**
  - [ ] Replace combinatorial Cartesian parsing with `optuna.create_study`.
  - [ ] Define dynamic search spaces using `trial.suggest_categorical`.
  - [ ] Implement `trial.report()` and `trial.should_prune()` for early trial ejection.
  - [ ] Force ZMQ & Replay Queue wholesale flushing via a new HTTP reset signal between runs.
- [ ] **Phase 3: Dashboard & Observability**
  - [ ] Configure `optuna` to use a persistent SQLite storage backend (`sqlite:///autotune.db`).
  - [ ] Launch `optuna-dashboard` to provide a live UI, eliminating the need to build a custom `textual` app.
  - [ ] Ensure TensorBoard logs seamlessly align with Optuna trial IDs.
