#!/bin/bash
# Tricked AI - 4 Quadrant Hotpath Profiling Sweep
set -e

mkdir -p reports

echo "Compiling with hotpath profiling enabled..."
cargo build --release --bin tricked_engine --features="hotpath"

echo "=========================================="
echo "Running 1: Small Model + Shallow Search (6 sims)"
cargo run --release --bin tricked_engine --features="hotpath" -- train --experiment-name small_shallow --resnet-blocks 4 --resnet-channels 64 --simulations 6 --max-steps 5 > reports/hotpath_small_shallow.txt || true
echo "Completed Quadrant 1"

echo "=========================================="
echo "Running 2: Small Model + Deep Search (128 sims)"
cargo run --release --bin tricked_engine --features="hotpath" -- train --experiment-name small_deep --resnet-blocks 4 --resnet-channels 64 --simulations 128 --max-steps 5 > reports/hotpath_small_deep.txt || true
echo "Completed Quadrant 2"

echo "=========================================="
echo "Running 3: Big Model + Shallow Search (6 sims)"
cargo run --release --bin tricked_engine --features="hotpath" -- train --experiment-name big_shallow --resnet-blocks 10 --resnet-channels 256 --simulations 6 --max-steps 5 > reports/hotpath_big_shallow.txt || true
echo "Completed Quadrant 3"

echo "=========================================="
echo "Running 4: Big Model + Deep Search (128 sims)"
cargo run --release --bin tricked_engine --features="hotpath" -- train --experiment-name big_deep --resnet-blocks 10 --resnet-channels 256 --simulations 128 --max-steps 5 > reports/hotpath_big_deep.txt || true
echo "Completed Quadrant 4"

echo "=========================================="
echo "All profiling sweeps completed. Reports dumped to the 'reports/' directory."
