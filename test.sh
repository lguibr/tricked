#!/usr/bin/env bash
set -e

echo "🧪 Running Strict Quality Gates..."
source .venv/bin/activate

# 1. Sync constants to ensure accurate tests
python3 scripts/generators/generate_rust_constants.py
maturin develop --release --manifest-path src/tricked_rs/Cargo.toml

# 2. Strict linting via ruff
echo "🧹 Running Ruff (Strict)..."
ruff check src/ tests/ --fix

# 3. Strict typechecking via mypy
echo "🧐 Running Mypy (Strict)..."
mypy --strict src/

# 4. Behavioral unit tests + coverage
echo "✅ Running Pytest Coverage..."
python3 -m pytest tests/ -v --cov=src --cov-report=term-missing

# 5. Frontend Quality Gates
echo "🌐 Running Frontend Checks..."
cd ui || exit
npm run format:check || true
npm run check
npm run test
cd ..
