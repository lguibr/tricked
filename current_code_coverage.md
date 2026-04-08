# Tricked AI Code Coverage Dossier

This document provides a summary of the current code coverage for the Tricked AI project, split by the core engine (Rust) and the Control Center frontend (TypeScript).

## 🦀 Rust Core Engine Coverage

Generated using `cargo llvm-cov`. Note that branch coverage is not tracked by default in this run.

| Metric | Covered / Total | Percentage |
|---|---|---|
| **Lines** | 7,535 / 9,579 | **78.66%** |
| **Functions** | 335 / 469 | **71.43%** |
| **Regions** | 10,229 / 13,372 | **76.50%** |
| **Branches** | N/A | **N/A** |

### File-Level Highlights (Rust)

* **Most Covered Modules**:
  * `src/net/dynamics.rs` (100% Lines, 100% Funcs)
  * `src/net/prediction.rs` (100% Lines, 100% Funcs)
  * `src/queue.rs` (99.27% Lines, 100% Funcs)
* **Least Covered Modules**:
  * `control_center/src-tauri/src/telemetry.rs` (0% Lines, 0% Funcs)
  * `control_center/src-tauri/src/execution.rs` (2.98% Lines, 6.25% Funcs)
  * `src/cli.rs` (36.18% Lines, 28.57% Funcs)

---

## 🦕 TypeScript Control Center Coverage

Generated using `vitest run --coverage` (V8). The Control Center currently has low test coverage, as many UI components lack specific unit tests.

| Metric | Percentage |
|---|---|
| **Statements** | **16.71%** |
| **Branches** | **4.76%** |
| **Functions** | **16.77%** |
| **Lines** | **16.01%** |

### File-Level Highlights (TypeScript)

* **Most Covered Modules**:
  * `src/components/execution/StudiesWorkspace.tsx` (76.92% Lines, 75% Funcs)
  * `src/components/ui/button.tsx` (100% Lines, 100% Funcs)
  * `src/components/ui/resizable.tsx` (100% Lines, 100% Funcs)
* **Least Covered Modules**:
  * `src/components/OptunaStudyDashboard.tsx` (1.33% Lines, 0% Funcs)
  * `src/components/playground/Playground.tsx` (1.36% Lines, 0% Funcs)
  * `src/components/execution/OptunaLogsViewer.tsx` (2.77% Lines, 8.33% Funcs)
  * `src/store/useTuningStore.ts` (7.89% Lines, 10% Funcs)

---
*Generated automatically during continuous integration and telemetry analysis.*
