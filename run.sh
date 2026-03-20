#!/usr/bin/env bash

# Wipe any lingering python daemons silently so the environment is strictly fresh when running
pkill -f src/tricked_web/server.py || true
pkill -f tensorboard.main || true
pkill -f src/tricked/main.py || true

set -e

echo "🚀 Booting Tricked UI & Engine..."
source .venv/bin/activate

# 1. Generate synced Rust constants from Python source of truth
echo "⚙️ Syncing Python mathematical grid to Rust constants..."
python3 scripts/generators/generate_rust_constants.py

# 2. Compile native Rust PyO3 engine
echo "🦀 Compiling high-performance Rust engine..."
maturin develop --release --manifest-path src/tricked_rs/Cargo.toml

# 3. Start Python backend server in background
echo "🐍 Starting Flask API Backend..."
python3 src/tricked_web/server.py &
BACKEND_PID=$!

# Register cleanup BEFORE blocking foreground to guarantee execution on Ctrl+C
trap "echo 'Shutting down daemons...'; kill $BACKEND_PID || true; pkill -f tensorboard.main || true; pkill -f src/tricked/main.py || true" EXIT INT TERM

# 4. Start SvelteKit UI in foreground
echo "⚡ Starting SvelteKit UI..."
cd ui && npm install && npm run dev
