.PHONY: check test lint format all coverage

all: format lint test build

format:
	cargo fmt

lint:
	cargo clippy --all-targets --all-features -- -D warnings

test:
	cargo test --release

build:
	cargo build --release

run:
	@echo "🔥 Starting Telemetry Daemon in background..."
	$(MAKE) telemetry & cargo run --release --bin tricked_engine

benchmark:
	@echo "🚀 Running 100 Million Game Monte Carlo Performance Baseline..."
	cargo run --release --bin mc_metrics -- 100000000

tune:
	@echo "📦 Ensuring python dependencies for auto-tune..."
	python3 -m venv venv
	./venv/bin/pip install -q requests rich optuna optuna-dashboard optunahub cmaes
	@echo "⚙️  Starting Auto-Tuner Optimization..."
	./venv/bin/python3 scripts/auto_tune.py

coverage:
	cargo tarpaulin --all-features --branch --engine llvm --out Html --output-dir target/coverage
	@echo "Coverage report generated at target/coverage/tarpaulin-report.html"

telemetry:
	@echo "📦 Ensuring python dependencies for telemetry..."
	python3 -m venv venv
	./venv/bin/pip install -q redis tensorboardX python-dotenv tensorboard "setuptools<70" psutil pynvml
	@echo "📊 Starting TensorBoard server in background..."
	./venv/bin/tensorboard --logdir runs --port 6006 --bind_all > /dev/null 2>&1 &
	@echo "🌐 TensorBoard available at http://localhost:6006"
	@echo "🚀 Launching Python TensorBoard Bridge..."
	./venv/bin/python3 scripts/tb_logger.py
