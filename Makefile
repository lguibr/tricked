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
	@echo "🔥 Starting Tricked AI Native Engine (CLI Mode)..."
	cargo run --release --features=hotpath,hotpath-alloc --bin tricked_engine -- train --experiment-name test_run --unroll-steps 5 --temporal-difference-steps 5

benchmark:
	@echo "🚀 Running 100 Million Game Monte Carlo Performance Baseline..."
	cargo run --release --bin mc_metrics -- 100000000

telemetry:
	@echo "📦 Ensuring Python dependencies for telemetry..."
	python3 -m venv venv
	./venv/bin/pip install -q streamlit pandas
	@echo "🌐 Starting Tricked AI Telemetry Dashboard..."
	./venv/bin/streamlit run scripts/dashboard.py
	
tune:
	@echo "📦 Ensuring python dependencies for auto-tune..."
	python3 -m venv venv
	./venv/bin/pip install -q requests rich optuna optuna-dashboard optunahub cmaes pandas numpy pymoo disjoint_set gpytorch plotly wandb optuna-integration
	./venv/bin/pip install -q --no-deps --no-build-isolation hebo
	@echo "⚙️  Starting Auto-Tuner Optimization..."
	./venv/bin/python3 scripts/tune.py

dashboard:
	@echo "🌐 Starting Optuna Dashboard..."
	./venv/bin/optuna-dashboard sqlite:///autotune.db --port 8080
