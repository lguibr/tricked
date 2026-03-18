#!/usr/bin/env bash

# Wipe lingering daemons silently
pkill -f src/tricked_web/server.py || true
pkill -f tensorboard.main || true
pkill -f src/tricked/main.py || true

set -e

echo "🚀 Booting Tricked UI & Engine (Docker Ecosystem)..."

# 1. Generate synced Rust constants
echo "⚙️ Syncing Python mathematical grid to Rust constants..."
python3 scripts/generate_rust_constants.py

# 2. Compile native Rust PyO3 engine
echo "🦀 Compiling high-performance Rust engine..."
maturin develop --release --manifest-path src/tricked_rs/Cargo.toml

# 3. Start Python backend server in background
echo "🐍 Starting Flask API Backend..."
python3 src/tricked_web/server.py &
BACKEND_PID=$!

# Register cleanup BEFORE blocking foreground
trap "echo 'Shutting down daemons...'; kill $BACKEND_PID || true; pkill -f tensorboard.main || true; pkill -f src/tricked/main.py || true" EXIT INT TERM

# 4. Start SvelteKit UI in foreground
echo "⚡ Starting SvelteKit UI..."
cd ui && npm install && npm run dev
