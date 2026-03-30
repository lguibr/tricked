# Agent Instructions

## Project Context
- **Architecture**: 100% Rust monolithic RL engine (`tch-rs`/LibTorch) + React/Vite/Tailwind UI dashboard.
- **Infrastructure**: Refer to `README.md` for directory layout (`/src`, `/ui`) and environment setup.

## Quality Gates & Style (STYLE.md)
- **MANDATORY**: Refer to `STYLE.md` before making structural or game-loop changes.
- **Mathematical Correctness**: Assert tensor sizes/types; fail fast on NaNs.
- **Performance Limits**: No dynamic allocation inside the MCTS simulation loop. Use `crossbeam` batches.
- **Explicit Naming**: Do not abbreviate variable names (e.g., use `policy_probabilities` over `probs`).

## Commit Attribution
AI commits MUST include:
```
Co-Authored-By: Antigravity <noreply@example.com>
```

## File-Scoped Commands
| Task | Command |
|------|---------|
| Format Rust | `make format` |
| Lint Rust | `make lint` |
| Test Rust | `make test` |
| Run Engine | `cargo run --release` |
| Test UI | `cd ui && npx vitest` |
| Run UI Dev | `cd ui && npm run dev` |
