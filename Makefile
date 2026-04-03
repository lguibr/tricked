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
	
hardware-tune:
	@echo "📦 Ensuring python dependencies for auto-tune..."
	python3 -m venv venv
	./venv/bin/pip install -q optunahub cmaes pandas optuna-integration
	./venv/bin/pip install -q --no-deps --no-build-isolation hebo
	@echo "⚙️  Starting Hardware Tuning Phase..."
	./venv/bin/python3 studies/hardware_tune.py --config scripts/configs/big.json --trials 30 --max-steps 15 --timeout 400

learning-tune:
	@echo "📦 Ensuring python dependencies for learning-tune..."
	python3 -m venv venv
	./venv/bin/pip install -q optunahub cmaes pandas optuna-integration
	./venv/bin/pip install -q --no-deps --no-build-isolation hebo
	@echo "⚙️  Starting Semantic Learning Velocity Tuning Phase..."
	./venv/bin/python3 studies/learning_tune.py --config scripts/configs/big.json --trials 50 --max-steps 50 --timeout 1800

dashboard:
	@echo "🌐 Starting Optuna Dashboard for Studies..."
	./venv/bin/optuna-dashboard sqlite:///studies/hardware_optuna_study.db --port 8080

profile-check:
	@echo "🔥 Running Automated Hotpath Profiling Sequence..."
	bash scripts/profile_hotpath.sh
	@echo "🔍 Validating Performance Limits to ensure zero regressions..."
	python3 scripts/check_profiling.py
