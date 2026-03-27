<div align="center">
  <img src="logo.png" alt="tricked logo" width="120" />
  <h1>Tricked AI Engine</h1>
  <p><em>A High-Performance, Native Reinforcement Learning Engine</em></p>

  ![Rust](https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/LibTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
  ![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
  ![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?&style=for-the-badge&logo=redis&logoColor=white)
</div>

**Tricked** is an elite, native Reinforcement Learning engine implementing a **MuZero-inspired** agent to master a topological puzzle game. 

By bypassing the Python GIL entirely, the engine leverages a **Rust backend** and `tch-rs` (LibTorch) to orchestrate massively concurrent self-play, MCTS search, and network optimization. The entire system is monitored through a stunning real-time **React Web UI** and bridged to **TensorBoard** for deep metric analysis.

---

## ✨ Key Architectural Features

*   🧠 **Native MuZero Architecture**: Full implementation of Representation, Dynamics, and Prediction networks using ResNet blocks.
*   📐 **Rhombus 2D Coordinates Triangular Grid**: The 96-triangle environment is mathematically mapped to a 2D rhombus coordinate system (`HEX_TO_2D_MAP`), allowing the neural network to process spatial convolutions natively on an 8x16 grid.
*   🎯 **Centralized Sprites of Piece Buffers**: Piece geometries are canonicalized and centered before being fed into the network. This provides translation invariance, allowing the AI to recognize shapes instantly regardless of where they are placed on the board.
*   ⚡ **Zero Python Overhead**: 100% Rust backend. Multi-threaded self-play and training run concurrently without GIL contention.
*   📊 **TensorBoard Integration**: A lightweight Python bridge (`tb_logger.py`) subscribes to Redis events, streaming live loss metrics, Q-values, and GPS directly to TensorBoard.
*   🌐 **Mission Control Web Server**: An Axum-powered web server exposing REST endpoints and WebSockets to a beautiful React/Vite/Tailwind SPA.

---

## 🏗️ System Topology

The codebase is split into three primary domains:

### 1. The Rust Backend (`/src`)
A Tokio-driven asynchronous engine.
*   **`features.rs` & `math.ts`**: Encodes the **rhombus 2D coordinates triangular grid** and generates the **centralized sprites of the piece buffers** for the 20-channel spatial tensor.
*   **`network/`**: The neural network definitions utilizing `tch-rs` to interface with LibTorch/CUDA.
*   **`mcts.rs`**: The Monte Carlo Tree Search engine featuring Gumbel noise injection and Sequential Halving.
*   **`trainer/`**: The optimizer loop applying Adam, computing BCE/Soft-Cross Entropy, and maintaining Polyak EMA weights.

### 2. The Web UI (`/ui`)
A modern SPA built with React, Vite, and Tailwind CSS.
*   **Mission Control**: Watch the AI play live via WebSockets.
*   **The Forge**: Interactively tweak transformer architecture, MCTS routing, and hardware I/O settings.
*   **The Vault**: Scrub through past games loaded from Redis, visualizing step-by-step trajectories.

### 3. The Telemetry Bridge (`/scripts`)
*   **`tb_logger.py`**: Listens to Redis Pub/Sub channels and writes standard `SummaryWriter` logs for TensorBoard visualization.

---

## 🚀 Getting Started

### Prerequisites
*   **Rust**: Standard `cargo` toolchain (1.75+).
*   **Node.js**: v18+ for the frontend.
*   **Redis**: Running on `localhost:6379`.
*   **Python 3**: For TensorBoard logging.

### 1. Launch the Engine & TensorBoard
The included Makefile handles setting up the Python virtual environment, launching TensorBoard, and compiling the Rust backend in release mode.

```bash
# Ensure Redis is running
docker-compose up -d redis

# Build and run the Rust engine + TensorBoard logger
make run
```
* TensorBoard will be available at `http://localhost:6006`
* The Axum API will be available at `http://localhost:8000`

### 2. Start the Web UI
```bash
cd ui
npm install
npm run dev
```
* Navigate to `http://localhost:5173` to access Mission Control.

---

## ⚙️ Configuration (The Forge)
The engine uses a dynamic configuration system. You can override hyperparameters dynamically via **The Forge** UI, which constructs a JSON payload sent to the backend `/api/training/start` endpoint. Adjust embedding dimensions, Gumbel scales, and TD-bootstrap steps on the fly.
