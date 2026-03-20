# Tricked Flask API Gateway (`tricked_web`)

## Abstract
The `tricked_web` module acts as the definitive bridge between the high-performance native PyTorch/Rust ecosystems and the thin-client Svelte UI. It is a strictly asynchronous boundary enforcing game-rules validation, SQLite telemetry streaming, and remote subprocess execution for the Triango engine.

## Architectural Endpoints
### `POST /api/move`
Transacts physical fragment placements into the PyO3 `GameStateExt`. Refuses syntactically invalid overlap permutations explicitly via the Rust validation layer, returning stateless validation structures.

### `GET /api/training/status`
A decoupled observer polling the `sqlite_logger` telemetry stream to serialize realtime Neural Backpropagation states (Iteration, Score, Loss trajectories) directly to the web client.

### `POST /api/training/start`
Architecturally segregates the PyTorch training loop (`main.py`) into a fully disjoint subprocess daemon, allowing asynchronous execution across cores without locking the UI HTTP threads.

## Security & Concurrency
The API enforces strict global thread locking (`global current_state`) on standard moves to ensure atomicity across parallel HTTP worker connections during spectator scenarios.
