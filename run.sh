#!/usr/bin/env bash

# Wipe any lingering python daemons silently so the environment is strictly fresh when running
pkill -f src/tricked/main.py || true

set -e

echo "🚀 Booting Tricked UI & Engine..."
source .venv/bin/activate

# 🌟 CRITICAL FIX: Tell Python where the 'tricked' packages live
export PYTHONPATH=src

# 0. Boot Infrastructure Dependencies (as art)
echo "🐳 Automagically booting Redis & WandB-Local..."
docker compose up -d redis wandb-local || docker-compose up -d redis wandb-local

export REDIS_HOST=localhost
export WANDB_BASE_URL=http://localhost:8081
unset WANDB_API_KEY

# 1. Generate synced Rust constants from Python source of truth
echo "⚙️ Syncing Python mathematical grid to Rust constants..."
python3 scripts/generators/generate_rust_constants.py

# 2. Compile native Rust PyO3 engine
echo "🦀 Compiling high-performance Rust engine..."
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="$(python3 -c 'import torch; import os; print(os.path.dirname(torch.__file__) + "/lib")'):/usr/lib/python3.13/config-3.13-x86_64-linux-gnu:$LD_LIBRARY_PATH"
export PYTHON_SYS_EXECUTABLE="$(pwd)/.venv/bin/python"
maturin develop --release --manifest-path src/tricked_rs/Cargo.toml
cargo build --release --bin self_play_worker --manifest-path src/tricked_rs/Cargo.toml

# 3. Start Training Daemon
echo "🤖 Starting Training Daemon..."
echo "💡 Note: Web UI is decoupled. To run the UI, use docker-compose up -d tricked-ai or run src/tricked_web/server.py independently."
python3 src/tricked/main.py
