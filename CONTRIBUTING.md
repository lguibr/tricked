# Contributing to Tricked AI Engine

Thank you for your interest in contributing! Tricked isn't just an experimental AI codebase; it's a heavily optimized production engine pursuing mathematical performance ceilings. To maintain extreme bounds on memory safety, concurrency tracking, and scale processing, all contributions must strictly abide by our Zero-Debt methodology.

> **Note to Contributors:** We are extremely welcoming to developers of all backgrounds! We absolutely welcome and encourage contributions generated using LLMs, AI coding assistants (like Copilot, Cursor, etc.), or traditional IDEs. We will gladly review each and every PR, regardless of how it was created, to help you get it merged. Our goal is to make contributing as accessible and fun as possible!

## Zero-Debt Policy

The core rule of Tricked is **Zero Debt.** This implies:
* **No Warning Suppression:** We **do not** tolerate the usage of `#[allow(dead_code)]`, `#[allow(unused)]` or `#[allow(clippy::all)]` tags.
* **No Logic Omits:** Failed testing parameters should trigger a rewrite of the assertion's logic bounds organically, not an `#[ignore]` tag.
* **Orphan Code Pruning:** Deprecated structs, configuration options, debug print statements, layout files, or unused python scripts must immediately be eradicated. We do not preserve obsolete files.

## Workflow 

1. **Test Driven Implementation:** Write testing metrics that natively expose panics on edge constraints dynamically.
2. **Run The Gauntlet:** A Pull Request qualifies only if it locally handles:
```bash
make lint
make test
cargo bench
```
3. **Architectural Parity:** Ensure any additions mapping the back-end (Rust parameters, JSON parsing) mirror evenly alongside the React UI Forge parameters. Unused parameters must be purged out of the user interface seamlessly. 
4. **Hardware Validation:** Check your concurrency overhead. Using atomic operators (`AtomicI64`) and asynchronous channels holds priority far ahead of any `Mutex` implementation paths for hot-loop scaling.
