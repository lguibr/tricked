
<div align="center">

![Tricked AI](logo.png)

# Tricked AI Engine

**A Zero-Debt, Lock-Free MuZero Reinforcement Learning Engine & Control Center**

[![Rust](https://img.shields.io/badge/Rust-1.80+-orange.svg?logo=rust)](https://www.rust-lang.org)
[![Tauri](https://img.shields.io/badge/Tauri-2.0-24c8db.svg?logo=tauri)](https://v2.tauri.app/)
[![React](https://img.shields.io/badge/React-19.1-61dafb.svg?logo=react)](https://react.dev/)
[![CUDA](https://img.shields.io/badge/CUDA-13.2-76b900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Optuna](https://img.shields.io/badge/Optuna-4.0-blue.svg?logo=optuna)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

## 🌌 Introduction

**Tricked** is a high-performance, strictly lock-free Reinforcement Learning engine engineered to solve a custom topological board puzzle. It trains AlphaZero and MuZero-style general board game agents, utilizing a rigorous **zero-debt Rust architecture** that extracts 100% throughput from multi-core CPU and GPU platforms without falling victim to cache starvation, lock contention, or memory saturation.

Beyond the core training algorithms, the Tricked ecosystem features a natively integrated **Tauri + React Control Center**—a production-grade MLOps dashboard providing real-time hardware telemetry, live UDP log routing, dynamic hyperparameter tuning via Optuna, and interactive environment playgrounds.

This repository serves as both a cutting-edge RL research lab and a showcase of modern Rust systems engineering applied to artificial intelligence.

---

## 🏗️ 1. High-Level Ecosystem Architecture

The Tricked platform is radically decoupled into two primary domains: the headless, hyper-optimized Rust/CUDA training engine ("The Muscle"), and the responsive Tauri/React-based Control Center ("The Eyes"). Relieving the GPU training hotpath from UI and telemetry overhead guarantees zero latency spikes during inference.

```mermaid
graph TD
    classDef UI fill:#0ea5e9,stroke:#0369a1,stroke-width:2px,color:#fff;
    classDef Engine fill:#f97316,stroke:#c2410c,stroke-width:2px,color:#fff;
    classDef DB fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff;

    subgraph Control_Center [Tauri App / React UI]
        React[React 19 Frontend<br/>Tailwind, ECharts, Shadcn]:::UI
        Telemetry[60 FPS requestAnimationFrame<br/>Metrics & Observability]:::UI
        TB[Tauri Rust Backend<br/>Process & Dashboard Manager]:::UI
        
        React <-->|IPC Commands & Events| TB
        Telemetry <-->|Binary UDP Stream| TB
    end

    subgraph Workspace [Shared Stateful Storage]
        DB[(SQLite Workspace DB)]:::DB
        OptDB[(Optuna Study DB)]:::DB
        CFG[JSON Run Configs]:::DB
        SAF[Safetensors Checkpoints]:::DB
    end

    subgraph Tricked_Engine [Rust Core Sidecar]
        CLI[Engine CLI & Config Parser]:::Engine
        MCTS[Lock-Free MCTS / Self-Play Arena]:::Engine
        BPTT[MuZero BPTT Optimizer]:::Engine
        CMOD[CUDA C++ PyTorch Ops]:::Engine
    end

    TB -->|Spawns Isolated Sidecar| CLI
    CLI --> MCTS
    MCTS <--> BPTT
    BPTT <--> CMOD
    
    TB <-->|Reads/Writes| DB
    CLI <-->|Reads/Writes| DB
    CLI -->|Persists Network| SAF
    CLI -->|Records Loss metrics| OptDB
```

---

## 🎮 2. Game Mechanics & The Topological Environment

**Tricked** is effectively a single-player topological survival puzzle demanding massive combinatorial spatial reasoning. The reinforcement learning agent must continuously clear lines to manage board density, surviving as long as possible by chaining multi-axis intersecting combos.

*   **The Grid:** A regular hexagon composed of exactly **96 equilateral triangles** configured with a side length of 4 units.
*   **Mathematical Coordinate System:** The board is represented natively as a raw `u128` bitmask. This allows engine logic to process complex line clears, intersections, and collision detections at near-zero latency executing `ALL_MASKS` bitwise comparisons.
*   **D6 Rotational Symmetry (Data Augmentation):** To artificially expand the training distribution, Tricked employs a mathematically complete D6 rotational symmetry set consisting of **89 distinct piece rotations**. This massively accelerates network generalization without the agent knowing it is being fed augmented trajectories.
*   **Scoring & Terminal State:** Clearing lines spanning any axis grants points, with overlapping intersections acting as exponential combo multipliers. The episode terminates when the board's clutter mathematically prevents the placement of *any* of the remaining pieces in the agent's 3-piece tray.

---

## 🧠 3. Core Engine Architecture (The Mind vs. Muscle)

The greatest sin of modern AI engineering is asking the logical mind to lift boulders, or asking the GPU muscle to solve tree-search riddles. Tricked enforces a hard, impenetrable hardware boundary between chronological tree-search, and concurrent geometric tensor arithmetic.

### I. Gumbel AlphaZero MCTS & Parallel Garbage Collection

The Monte Carlo Tree Search operates using a state-of-the-art **Sequential Halving** pipeline injected with **Gumbel Noise**, aggressively pruning the search space to find the optimal policy. 

To eliminate locking overhead, MCTS nodes are allocated from a lock-free `ArrayQueue`, guaranteeing zero-allocation runtime traversals. Furthermore, Tricked parallelizes MCTS garbage collection to prevent worker starvation and entirely avoids `spin-loop` architecture by utilizing event-driven, atomic inference hand-offs.

```mermaid
sequenceDiagram
    participant W as CPU Worker (Lock-Free)
    participant T as Stack-Allocated MCTS Tree
    participant Q as Event-Driven Inference Queue
    participant GPU as PyTorch/CUDA

    W->>T: Initialize Tree & Root Node
    W->>T: Inject Gumbel Noise Variables
    
    loop Sequential Halving Pass
        W->>T: Lock-Free Traverse (DFS)
        T-->>W: Target Leaf Node Index
        W->>Q: Atomic Push EvaluationRequest (Batching)
        Q->>GPU: Batched Forward Pass (No Spin-Loops)
        GPU-->>Q: Value, Policy, Hidden State
        Q-->>W: Asynchronous EvaluationResponse
        W->>T: Backpropagate Value & Expand Node
        W->>T: Prune Sub-Optimal Candidates (Halving)
    end
    W->>W: Compute Final Action Distribution
```

### II. Lock-Free CPU/GPU Memory Pipeline

Memory limits are finite. The GPU should never sit idle waiting for the CPU to arrange tensors. 
Tricked completely decouples tensor formatting using a dedicated `Prefetch` threading hierarchy. It formats observation arenas directly into explicitly pinned host memory (`PinnedBatchTensors`), allowing PyTorch to execute non-blocking, asynchronous PCIe DMA transfers directly into `GpuBatchTensors`.

```mermaid
flowchart LR
    classDef C1 fill:#6366f1,stroke:#4338ca,stroke-width:2px,color:#fff;
    classDef C2 fill:#ec4899,stroke:#be185d,stroke-width:2px,color:#fff;

    A[Parallel MCTS Workers]:::C1 -->|Game Trajectories| B(Replay Buffer / PER Shards)
    B --> C[(SumTree Priority Tracker)]:::C1
    C -->|Uncontended Lock| E{Background Prefetcher}:::C1
    
    E -->|Format directly to RAM| F[PinnedBatchTensors]:::C2
    F -->|PCIe Async DMA Transfer| G[GpuBatchTensors]:::C2
    G --> H((CUDA BPTT Optimizer)):::C2
    H -->|Temporal Difference Errors| C
```

### III. MuZero Unrolled BPTT Network Architecture

The optimizer unrolls the dynamics network over arbitrary temporal structures, computing Soft Cross Entropy and Negative Cosine Similarity against an Exponential Moving Average (EMA) momentum target network. This explicitly prevents representation collapse in environments with sparse rewards. Tricked has completely removed FP64 precision liabilities from the hotpath, allowing pure FP32/TensorFloat32 (TF32) throughput.

```mermaid
graph TD
    classDef Net fill:#8b5cf6,stroke:#5b21b6,color:#fff;
    classDef Target fill:#14b8a6,stroke:#0f766e,color:#fff;
    classDef Loss fill:#ef4444,stroke:#b91c1c,color:#fff;

    subgraph Unrolled BPTT Execution [Online Network]
        S0[Batched Initial State] --> Rep[Representation Net]:::Net
        Rep --> H0[Hidden State t=0]
        H0 --> Pred0[Prediction Net]:::Net
        Pred0 --> P0[Policy 0] & V0[Value 0]
        
        H0 & A1[Action Vector 1] --> Dyn1[Dynamics Net]:::Net
        Dyn1 --> H1[Hidden State t=1] & R1[Reward 1]
        H1 --> Pred1[Prediction Net]:::Net
        Pred1 --> P1[Policy 1] & V1[Value 1]
    end
    
    subgraph EMA Target Generation [Momentum Network]
        EMA[Target Representation Net]:::Target
        S1[Target State t=1] --> EMA
        EMA --> TH1[Target Hidden State]:::Target
    end
    
    H1 -.->|Negative Cosine Similarity| SimLoss[Similarity Loss]:::Loss
    TH1 -.-> SimLoss
    
    P1 -.->|Soft Cross Entropy| MCTS_P[MCTS Augmented Policy]
    MCTS_P -.-> PolLoss[Policy Loss]:::Loss
    
    V1 -.->|Huber / MSE| TD_V[N-Step TD Return]
    TD_V -.-> ValLoss[Value Loss]:::Loss
```

---

## 🖥️ 4. The MLOps Control Center (Tauri + React 19)

Tricked isn’t just a headless AI—it ships with a production-grade, highly-themed Control Center to manage the immense stream of metrics. Utilizing React 19, Tailwind, ECharts, and Shadcn, the interface guarantees actionable visual feedback at 60 FPS.

### Deep Observability & Telemetry Pipeline
We abandoned standard synchronous `stdout` string logging. System telemetry operates via a highly-optimized **Binary UDP Protocol**, sending thousands of metrics per second over localhost. The frontend ingests these via a Rust Tauri command, passing them into pre-allocated mutable React references rendering via `requestAnimationFrame` to ensure zero UI jank.

*   **Granular System Diagnostics:** A unified, toggleable **Treemap & Sunburst visualization** displaying CPU core topography, alongside specific PID-level RAM, VRAM, and GPU utilization for all internal engine and sidecar processes.
*   **Hierarchical Theming:** Visual accents distinctly highlight the relationship between overarching "Runs" and associated "Runnages" (MCTS workers, BPTT servers).
*   **Live Metrics:** High-fidelity, gradient-smoothed area charts monitor critical AI health signs dynamically: 
    * *Queue Latency & Spin-Wait Cycles*
    * *SumTree Lock Contention Ratios*
    * *Layer-Wise Neural Network Gradient Norms*
    * *Action Space Entropy & KL Divergences*
    * *Replay Buffer Prioritization Heatmaps*

---

## 🔬 5. Optuna Tuning Lab

Configuring hyperparameters for AlphaZero engines is notoriously difficult. The Control Center features an **Optuna Tuning Lab** seamlessly integrated into the React frontend. It fetches completed trials from the `unified_optuna_study.db` SQLite database, plotting multi-objective parameter success.

*   View **Pareto Fronts** charting Hardware Limits (FPS/Throughput) vs. Evaluation Loss.
*   Inspect parameter importance via hyper-dimensional filtering.
*   Interact with real-time trial pruning distributions to see exactly where failing theories are killed. 

### Tuning Pipeline Architecture

```mermaid
sequenceDiagram
    box rgb(30, 41, 59) Tauri UI
    participant UI as Control Center
    end
    
    box rgb(15, 23, 42) Tricked Training
    participant Eng as Engine (Sidecar)
    participant Optuna as Optuna Study
    participant DB as SQLite DB
    end

    UI->>Eng: Start Tuning Study (Trials: 50, Bounds: {...})
    
    loop Every Single Trial
        Eng->>Optuna: Request Hyperparameters (optuna_ask.py)
        Optuna-->>Eng: Sample JSON Bounds (Batch Size, LR, C_PUCT, etc)
        
        Eng->>Eng: Instantiate Workers & Run BPTT Optimizer
        
        par Telemetry & Logging
            Eng->>DB: Stream Live Loss & Hardware Saturations (Binary UDP)
            DB-->>UI: Real-Time ECharts Metric Updates
        end
        
        Eng->>Optuna: Report Final Evaluation Loss & Throughput Penalty
        Optuna->>DB: Update Pareto Fronts & Feature Importance
    end
    
    UI->>DB: Fetch Final Pareto Configurations
```

---

## 🕹️ 6. Tricked Interactive Playground

To assist in behavioral debugging, the UI includes an **Interactive Playground** mirroring the AI's exact spatial sandbox. 
By utilizing `GameStateExt` bindings compiled via WASM/Tauri IPC, human players compete using the identical mathematical constraints the Rust engine natively executes. This allows researchers to playtest difficulty parameters, verify the complete D6 rotational symmetry augmentation visually, and confirm terminal state logic before firing up the GPU cluster.

---

## 🚀 7. Installation & Setup

Tricked relies on strict formatting hooks and requires the user to satisfy an assortment of specific ML tools. 

### Prerequisites
*   **Rust Toolchain:** `stable` (1.80+)
*   **Node.js:** `v20.x` or higher
*   **NVIDIA CUDA Toolkit:** `13.2+` (Required for custom C++ operations)
*   **Python:** 3.11+ via the `uv` blazing fast package manager.

### Standard Build Workflow

Tricked uses a unified `Makefile` that orchestrates building custom PyTorch C++ extensions (`tricked_ops.so`), compiling the Rust backend, resolving `pnpm` frontend dependencies, and tying it all into the Tauri binary. It also acts as the primary defense hook for our zero-debt CI expectations.

```bash
# 1. Clone the repository
git clone https://github.com/Tricked-AI/Tricked.git
cd Tricked

# 2. Build everything (CUDA ops, Rust server, React frontend)
make all

# 3. Launch the Control Center GUI in Developer Mode
make dev
```

### Headless CLI Mode

For dedicated training clusters (e.g., Slurm / Kubernetes environments), the UI can be bypassed entirely. 

```bash
cargo run --release --bin tricked_engine -- train \
    --experiment-name "resnet_v2_baseline" \
    --simulations 400 \
    --train-batch-size 2048 \
    --lr-init 0.005 \
    --workers 16 \
    --gumbel-scale 1.5
```

---

## 🛡️ License

Tricked is distributed under the **MIT License**. For an exhaustive explanation of warranties and liabilities, see the [LICENSE](LICENSE) file.
