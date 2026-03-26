<img src="logo.png" alt="tricked logo" width="100" />

# Tricked


![Rust](https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white)
![PyTorch](https://img.shields.io/badge/LibTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Tailwind](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?&style=for-the-badge&logo=redis&logoColor=white)

**Tricked** is a high-performance, native Reinforcement Learning engine implementing a **MuZero-inspired** agent to master a 96-bit topological hex-grid puzzle game. 

By bypassing the Python GIL entirely, the engine leverages Rust and `tch-rs` (LibTorch) to orchestrate massively concurrent self-play, MCTS search, and network optimization—all monitored through a stunning real-time React web dashboard.

---

## ✨ Key Features

*   🧠 **Native MuZero Architecture**: Full implementation of Representation, Dynamics, and Prediction networks using Graph Convolutions (GCN) and ResNet blocks.
*   ⚡ **Zero Python Overhead**: 100% Rust backend using `tch-rs`. Multi-threaded self-play and training run concurrently without GIL contention.
*   🌳 **Advanced MCTS**: Features Gumbel MuZero search, Sequential Halving, and dynamic $K$-samples for highly efficient exploration.
*   🧩 **Bitboard Topology**: Ultra-fast $O(1)$ collision, line-clear detection, and scoring using `u128` bitwise operations on a 96-triangle grid.
*   💾 **Sharded PER**: Prioritized Experience Replay backed by concurrent SumTrees and ring buffers for lock-free experience gathering.
*   🎛️ **Mission Control Dashboard**: A beautiful React/Vite/Tailwind web app featuring real-time telemetry, trajectory scrubbing, and live hyperparameter tuning.

---

## 🏗️ Architecture Overview

The codebase is split into two primary domains:

### 1. The Engine (`/src`)
A Tokio-driven asynchronous Rust backend.
*   **`board.rs` & `features.rs`**: The high-performance game logic. Encodes topological game states and translates them into spatial tensor features.
*   **`network/`**: The neural network definitions (Dynamics, Representation, Prediction) utilizing `tch-rs` to interface with LibTorch/CUDA.
*   **`mcts.rs` & `node.rs`**: The Monte Carlo Tree Search engine featuring Gumbel noise injection and policy normalization.
*   **`selfplay.rs`**: Actor threads running the game loops and generating trajectories.
*   **`trainer/`**: The optimizer loop applying Adam, computing BCE/Soft-Cross Entropy, and maintaining Polyak Exponential Moving Averages (EMA) of network weights.
*   **`web/`**: An Axum web server exposing REST endpoints and WebSockets for the UI.

### 2. The Dashboard (`/ui`)
A modern SPA built with React, Vite, and Tailwind CSS.
*   **Mission Control**: Watch the AI play live, view the active grid, and monitor Engine Vitals (Games Per Second).
*   **The Forge**: Interactively tweak transformer architecture (d_model, blocks), MCTS routing, and hardware I/O settings with visual resource estimation.
*   **The Vault**: Scrub through past games loaded from Redis, visualizing step-by-step trajectories and AI "death traps" (hole logits).

---

## 🚀 Getting Started

### Prerequisites
*   **Rust**: Standard `cargo` toolchain (1.75+ recommended).
*   **Node.js**: v18+ and `npm` or `yarn` for the frontend.
*   **LibTorch**: The `tch` crate handles this automatically in most setups, but you may need to set `LIBTORCH` environment variables if compiling against a custom CUDA installation.
*   **Redis**: Requires a local Redis server running on `localhost:6379` for trajectory caching and event pub/sub.

### 1. Start the Backend Engine

```bash
# Clone the repository
git clone https://github.com/lguibr/tricked.git
cd tricked

# Ensure Redis is running (e.g., via Docker)
docker run -d -p 6379:6379 redis

# Build and run the Rust engine (Release mode is crucial for RL performance)
cargo run --release
```
*The Axum server will start listening on `http://0.0.0.0:8000`.*

### 2. Start the Frontend Dashboard

```bash
# Open a new terminal instance
cd ui

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```
*Navigate to `http://localhost:5173` in your browser to access Mission Control.*

---

## 🎮 Web Interface Tour

### 📡 Mission Control
Your primary operations center. Connects via WebSocket to stream live board states, current scores, and network predictions as the AI plays out MCTS simulations.

### 🔨 The Forge
Configure the brain of your agent before spinning up a training run. Tweak:
*   **Network Size**: Embedding dimensions, ResNet blocks.
*   **Search**: Simulations, Unroll/TD steps, Gumbel Scale.
*   **Hardware**: Choose between `cuda`, `mps` (Apple Silicon), or `cpu`.

### 🔒 The Vault
Post-game forensic analysis. Re-load `trajectory` chunks from Redis and step through them frame-by-frame to analyze policy distributions and understand where the AI failed.

---

## ⚙️ Configuration

The engine uses a `config.yaml` file (located in `conf/config.yaml`) defining the default state for the orchestrator. You can override these dynamically via **The Forge** UI, which constructs a JSON payload sent to the backend `/api/training/start` endpoint.

---

## 🛠️ Development & Testing

**Formatting & Linting (Rust):**
```bash
make format
make lint
```

**Running Engine Tests:**
The engine includes comprehensive tests for bitboard topology, MCTS sequence halving, and NaN-safety in neural network layers.
```bash
make test
```

**Running UI Tests:**
```bash
cd ui
npx vitest
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.