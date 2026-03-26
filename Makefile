.PHONY: check test lint format all coverage

all: format lint test build

format:
	cargo fmt

lint:
	cargo clippy --all-targets --all-features -- -D warnings

test:
	cargo test

build:
	cargo build --release

run:
	cargo run --release

coverage:
	cargo tarpaulin --all-features --branch --engine llvm --out Html --output-dir target/coverage
	@echo "Coverage report generated at target/coverage/tarpaulin-report.html"

telemetry:
	@echo "📦 Ensuring python dependencies for telemetry..."
	python3 -m venv venv
	./venv/bin/pip install -q redis wandb python-dotenv
	@echo "🚀 Launching Python WandB Bridge..."
	./venv/bin/python3 scripts/wandb_logger.py

telemetry:
	@echo "📦 Ensuring python dependencies for telemetry..."
	python3 -m venv venv
	./venv/bin/pip install -q redis wandb python-dotenv
	@echo "🚀 Launching Python WandB Bridge..."
	./venv/bin/python3 scripts/wandb_logger.py
