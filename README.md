<div align="center">
  <img src="logo.png" alt="Tricked AI Logo" width="250" />

  <h1>Tricked 🔺</h1>
  <p><b>High-Performance SOTA Mathematical Engine & Gumbel MuZero Tree Search</b></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/Rust-1.76-000000?style=for-the-badge&logo=rust&logoColor=white" />
    <img src="https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
    <img src="https://img.shields.io/badge/SvelteKit-5-FF3E00?style=for-the-badge&logo=svelte&logoColor=white" />
    <img src="https://img.shields.io/badge/Tailwind_CSS-v4-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" />
    <img src="https://img.shields.io/badge/Docker-CUDA_Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
    <img src="https://img.shields.io/badge/Coverage-100%25-brightgreen?style=for-the-badge" />
  </p>
</div>

---

## 🌌 Ecosystem Architecture
Tricked is a multi-language ecosystem optimized entirely to prevent standard abstraction delays. We translate raw bitwise board matrices from Python natively into a **Rust (PyO3) Verification Engine**, achieving zero-cost tensor bounds for ultra-concurrent MCTS expansions.

The system features a completely decoupled Svelte 5 frontend interacting seamlessly with a Flask JSON Core.

```mermaid
graph TD;
    subgraph UI [Black Dev Web Client]
        Svelte[SvelteKit V5 Runes] --> |Tailwind v4 SVG Matrix| Svelte
    end
    
    subgraph Python [Core AI Backend]
        Flask[Flask JSON API] --> Svelte
        MuZero[Gumbel MuZero Agent] --> |Self-Play Workers| Buffer[Experience Replay]
        Trainer[PyTorch Model] --> Buffer
    end
    
    subgraph Rust [Native Engine]
        PyO3[PyO3 FFI Bridge] --> Bitboards[u128 Bitwise Board]
        Bitboards --> Validator[D12 Symmetry Validator]
    end

    Flask -.-> |API Requests| PyO3
    MuZero -.-> |MCTS Expansions| PyO3
```

## 🔥 High-Fidelity Features

1. **Rust PyO3 Engine (`tricked_rs`)**: Total mathematical bound verification with zero memory leaks. 120-degree Tri-Coordinate mathematics seamlessly map (9,11,13,15,15,13,11,9) layout structures directly to `u128` integers.
2. **Gumbel MuZero Reinforcement**: Replaced legacy scalar heuristics with Two-Hot Cross Entropy and Spatial ResNets for maximum predictive precision.
3. **Live Spectator Mode**: Ultra-high-performance UI tracking. Subprocesses asynchronously dump lock-free mathematical permutations polled by the Svelte Client at 100ms rates without bottlenecking the PyTorch Training cycles.
4. **Progressive Difficulty Curriculum**: Automated threshold promotion scaling up Dihedral spatial reasoning from simple isolated anchors all the way to 26 perfectly mathematically closed D12 piece definitions.

---

## 🚀 Execution & Bootstrap

### 1. Cloud / RTX Docker Container (Recommended)
Containerize the entire ecosystem effortlessly spanning physical RTX nodes:
```bash
docker build -t tricked-ai:latest .
docker run --gpus all -p 6006:6006 -p 8080:8080 -p 5173:5173 tricked-ai:latest
```
*Intercept the dynamic loss curves natively on: `http://localhost:6006`*
*View the live HUD Spectator natively on: `http://localhost:5173`*

### 2. Manual Source Compilation
Install the ecosystem targeting native Python optimization. `pip install` transparently invokes `maturin` to bind the Rust structs.
```bash
python3 -m pip install -e .
cd ui && npm install && npm run dev
```

### 3. Training Orchestrator
We expose a globally registered training orchestrator accessible universally:
```bash
tricked-train
```

### 4. Regenerating Shapes & Rust Constants
If you modify the underlying mathematical grid or symmetry definitions, you must regenerate the canonical piece data and compiling the Rust engine:

```bash
# 1. Regenerates Python PieceDefs and Rust bitmasks
python scripts/generate_all.py 

# 2. Recompile the Rust PyO3 Engine for the local environment
maturin develop --release
```

---
<div align="center">
  <i>Engineered for Maximum Capability • Pure Mathematical Strategy</i>
</div>
