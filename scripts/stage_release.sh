#!/bin/bash
# Staging script to move shared libraries into the Tauri assets directory for AppImage bundling

set -e

# Change down to project root
cd "$(dirname "$0")/.."

echo "Staging libraries for Linux release..."

ASSETS_DIR="control_center/src-tauri/assets/libs"
VENV_DIR="venv"

# Find python version programmatically
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# Make sure assets directory exists
mkdir -p "$ASSETS_DIR"

echo "Copying LibTorch dependencies..."
# Copy Torch libs
cp -a $VENV_DIR/lib/python${PY_VER}/site-packages/torch/lib/*.so* "$ASSETS_DIR/"

echo "Copying NVIDIA cu13 dependencies..."
# Copy NVIDIA cu13 libs
# Some directories might not have lib/, suppressing error just in case
for d in $VENV_DIR/lib/python${PY_VER}/site-packages/nvidia/*; do
    if [ -d "$d/lib" ]; then
        cp -a "$d/lib"/*.so* "$ASSETS_DIR/" 2>/dev/null || true
    fi
done

echo "Copying custom CUDA ops..."
if [ -f "tricked_ops.so" ]; then
    cp "tricked_ops.so" "$ASSETS_DIR/"
fi

# Copy models
mkdir -p "control_center/src-tauri/assets/models"
if [ -f "assets/math_kernels.pt" ]; then
    cp "assets/math_kernels.pt" "control_center/src-tauri/assets/models/"
fi
if [ -f "feature_extractor.pt" ]; then
    cp "feature_extractor.pt" "control_center/src-tauri/assets/models/"
fi

echo "Staging completed. Libraries placed in $ASSETS_DIR"
