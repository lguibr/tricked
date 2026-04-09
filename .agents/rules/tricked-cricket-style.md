---
trigger: always_on
---


You are working in the `tricked` repository. You must **STRICTLY adhere to the Cricket Style philosophy** defined in `CRICKET_STYLE.md`.

## 1. The Sanctity of Language (Zero Abbreviations)

- **NEVER use abbreviations** in variable names, function names, or struct names.
- Bad: `obs`, `val`, `td_steps`, `net`.
- Good: `batched_observations`, `predicted_value_scalar`, `temporal_difference_steps`, `neural_network`.
- Keystrokes are infinite; cognitive capacity is precious. A variable name must be a complete, unbroken thought.

## 2. Encode Shape in Tensors

- The shape and semantic meaning of multi-dimensional data must be woven directly into the variable name.
- Bad: `policy_output`
- Good: `target_policies_batch_time_action`
- A shape mismatch is a silent killer. The dimensions of reality must be spoken aloud in the name of the thing itself.

## 3. The Duality of Mind (Rust/CPU) and Muscle (CUDA/GPU)

- The CPU is the mind. It handles branching, unpredictability, and Monte Carlo Tree Search. It must remain agile and lock-free.
- The GPU is the muscle. It only handles huge blocks of geometry. Do not put control logic, decision trees, or branching on the GPU.
- Respect the PCIe Boundary: only cross the bridge when necessary, passing pre-allocated batches to avoid starvation.

## 4. Maximum Leverage & Constraints

- You are acting for a solo developer fighting infinite compute. Everything you write must be designed for maximum leverage, doing exactly what's needed and nothing more.
- When traversing the Monte Carlo trees, respect the flow of the lock-free architecture. Use isolated workers communicating via queues/channels, and avoid blocking locks entirely.

Keep the thoughts clear. Keep the names whole. Keep the leaps massive.
