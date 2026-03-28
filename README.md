<div align="center">
  <img src="logo.png" alt="tricked logo" width="120" />
  <h1>Tricked AI Engine</h1>
  <p><em>A High-Performance, Native Reinforcement Learning Engine</em></p>

  ![Rust](https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/LibTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
  ![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
  ![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?&style=for-the-badge&logo=redis&logoColor=white)
</div>

**Tricked** is an elite, native Reinforcement Learning engine implementing a **Gumbel MuZero** agent to master a topological puzzle game on a 96-triangle hexagonal grid. 

Optimized specifically for solo-developer hardware (RTX 3080 Ti Mobile, i9, 64GB RAM), the engine bypasses the Python GIL entirely, leveraging a **Rust backend** and `tch-rs` (LibTorch) to orchestrate massively concurrent self-play, MCTS search, and network optimization.

---

## 🏗️ 1. END-TO-END SYSTEM TOPOLOGY

This diagram illustrates the macro architecture, showing how the Rust Engine, Redis, and the Unified React UI interact via 10-second REST polling.

```mermaid
graph TD
    subgraph React_Frontend [Unified React Dashboard]
        UI_Control[Mission Control]
        UI_Forge[The Forge Config]
        UI_Vault[The Vault Replays]
    end

    subgraph Axum_Web_Server[Rust Axum API]
        API_Status[/api/training/status/]
        API_Start[/api/training/start/]
        API_Games[/api/games/latest/]
    end

    subgraph Rust_Core_Engine[Tricked AI Engine]
        SP[Self-Play Workers x32]
        MCTS[Gumbel MCTS]
        OPT[Adam Optimizer]
        PER[(Prioritized Replay Buffer)]
    end

    subgraph External_Services [Data Layer]
        REDIS[(Redis Pub/Sub & Hash)]
        TB[TensorBoard Logger]
    end

    UI_Control -- "10s Poll" --> API_Status
    UI_Vault -- "10s Poll" --> API_Games
    UI_Forge -- "POST Config" --> API_Start

    API_Start --> SP
    SP <--> MCTS
    SP -- "Push Trajectories" --> PER
    PER -- "Sample Batch" --> OPT
    OPT -- "Update Weights" --> MCTS

    SP -- "Log Games" --> REDIS
    OPT -- "Log Loss" --> REDIS
    REDIS -- "Stream" --> TB
```

---

## 🧠 2. GUMBEL MUZERO MCTS EXECUTION FLOW

The core decision-making algorithm. This details how Sequential Halving and Gumbel Noise are injected into the Monte Carlo Tree Search to ensure optimal exploration without the Python GIL overhead.

```mermaid
graph LR
    A[Start Search] --> B{Is Node Expanded?}
    B -- No --> C[Initial Inference]
    C --> D[Expand Root Node]
    D --> E[Inject Gumbel Noise]
    
    B -- Yes --> E
    
    E --> F[Select Top K Actions]
    F --> G[Sequential Halving Loop]
    
    subgraph Halving_Phase [Sequential Halving]
        G --> H[Traverse to Leaf]
        H --> I[Recurrent Inference]
        I --> J[Backpropagate Value]
        J --> K{Phase Complete?}
        K -- No --> H
        K -- Yes --> L[Prune Bottom 50% Actions]
    end
    
    L --> M{Only 1 Action Left?}
    M -- No --> G
    M -- Yes --> N[Compute Final Policy]
    N --> O[Execute Move in Env]
```

---

## 🕸️ 3. NEURAL NETWORK ARCHITECTURE

The MuZero model is split into three distinct ResNet-based networks. The environment is mapped to a 20-channel 8x16 spatial tensor.

```mermaid
graph TD
    Input[State Tensor: 20x8x16] --> RepNet
    
    subgraph Representation_Network
        RepNet[Conv2D Projection] --> R_Res1[ResNet Block 1]
        R_Res1 --> R_ResN[ResNet Block N]
        R_ResN --> HiddenState[Hidden State: d_model x 8 x 8]
    end

    HiddenState --> PredNet
    HiddenState --> DynNet

    subgraph Prediction_Network
        PredNet[LayerNorm] --> P_Val[Value Head]
        PredNet --> P_Pol[Policy Head]
        P_Val --> ValOut[Scalar Value]
        P_Pol --> PolOut[Action Logits: 288]
    end

    subgraph Dynamics_Network
        DynNet[Concat Action + Hidden] --> D_Res1[ResNet Block 1]
        D_Res1 --> D_ResN[ResNet Block N]
        D_ResN --> NextHidden[Next Hidden State]
        DynNet --> D_Rew[Reward Head]
        D_Rew --> RewOut[Scalar Reward]
    end
```

---

## 💻 4. HARDWARE RESOURCE MAPPING (RTX 3080 Ti Mobile)

How the engine maps threads and memory to a solo-developer laptop (16GB VRAM, 14-Core CPU).

```mermaid
graph TD
    subgraph CPU_RAM[64GB System RAM / i9 CPU]
        W1[Worker 1] --> Q[Fixed Inference Queue]
        W2[Worker 2] --> Q
        WN[Worker 32] --> Q
        PER[(Replay Buffer: 100k Capacity)]
    end

    subgraph GPU_VRAM[RTX 3080 Ti Mobile - 16GB VRAM]
        Q -- "Batch Size: 1024" --> Inf[Inference Engine FP16]
        Inf -- "Hidden States" --> Cache[(Latent Tensor Cache)]
        
        PER -- "Train Batch: 512" --> Opt[Adam Optimizer]
        Opt --> Weights[Model Weights]
        Weights -. "EMA Sync" .-> Inf
    end
```

---

## 🔄 5. DATA PIPELINE & REPLAY BUFFER

The lifecycle of a game trajectory from generation to optimization.

```mermaid
sequenceDiagram
    participant SP as Self-Play Worker
    participant Env as Hex Grid Env
    participant PER as Prioritized Replay Buffer
    participant Optim as Optimizer
    
    loop Every Step
        SP->>Env: Apply Action
        Env-->>SP: Next State, Reward
        SP->>SP: Store in Local History
    end
    
    SP->>PER: Push OwnedGameData (Boards, Actions, Policies)
    Note over PER: Calculate Difficulty Penalty
    PER->>PER: Insert into Segment Tree (SumTree)
    
    loop Every Train Step
        Optim->>PER: Sample Batch (Proportional to Priority)
        PER-->>Optim: BatchTensors + Importance Weights
        Optim->>Optim: Compute BCE & Soft-Cross Entropy Loss
        Optim->>Optim: Backpropagate & Step
        Optim->>PER: Update TD-Errors (New Priorities)
    end
```

---

## 🌐 6. UNIFIED UI POLLING ARCHITECTURE

The refactored, memory-efficient UI architecture. WebSockets have been removed to save 2GB of RAM. The UI now uses lightweight 10-second polling.

```mermaid
graph LR
    subgraph Browser [React SPA]
        Dash[Unified Dashboard]
        Timer((10s Interval))
    end

    subgraph Axum [Rust Backend]
        API_S["/api/training/status"]
        API_G["/api/games/latest"]
    end

    subgraph Redis [Redis Cache]
        Hash[tricked_replays]
        PubSub[tricked_metrics]
    end

    Timer -->|Fetch| API_S
    Timer -->|Fetch| API_G
    
    API_S -->|Read| PubSub
    API_G -->|HGET| Hash
    
    Dash -.->|Render| Board[SVG Hex Board]
```

---

## 🚀 Getting Started

### Prerequisites
*   **Rust**: Standard `cargo` toolchain (1.75+).
*   **Node.js**: v18+ for the frontend.
*   **Redis**: Running on `localhost:6379`.
*   **Hardware**: Optimized for RTX 3080 Ti Mobile (16GB VRAM).

### 1. Launch the Engine & TensorBoard
```bash
docker-compose up -d redis
make run
```
* TensorBoard: `http://localhost:6006`
* Axum API: `http://localhost:8000`

### 2. Start the Unified UI
```bash
cd ui
npm install
npm run dev
```
* Navigate to `http://localhost:5173` to access the Unified Dashboard.
