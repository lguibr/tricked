#!/bin/bash
set -e

echo "Starting Tricked Triple-Daemon Orchestrator..."

if [ "${ENABLE_TENSORBOARD:-1}" = "1" ]; then
    # 1. Daemonize TensorBoard in the background on port 6006
    echo "Initializing TensorBoard daemon on port 6006..."
    tensorboard --logdir=runs/tricked_muzero --host 0.0.0.0 --port 6006 &
    # Wait slightly for tensorboard to bind
    sleep 2
else
    echo "TensorBoard daemon is disabled (ENABLE_TENSORBOARD=0)."
fi

# 2. Boot the core MuZero AI Trainer in the foreground.
if [ "${ENABLE_WEB_UI:-1}" = "1" ]; then
    echo "Starting MuZero Neural Trainer (and implicitly Web UI on 8080)..."
else
    echo "Starting MuZero Neural Trainer (Web UI disabled)..."
fi
exec python3 -m src.tricked.main
