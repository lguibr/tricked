SHELL := /usr/bin/env zsh
.PHONY: setup build run-native run-docker clean test run

# Automatically find the PyTorch library path for Rust linking (safe against fresh installs via suppression)
TORCH_LIB := $(shell . .venv/bin/activate 2>/dev/null && python3 -c 'import torch; import os; print(os.path.dirname(torch.__file__) + "/lib")' 2>/dev/null)
export LD_LIBRARY_PATH := $(TORCH_LIB):$(LD_LIBRARY_PATH)
export LIBTORCH_USE_PYTORCH := 1
export LIBTORCH_BYPASS_VERSION_CHECK := 1
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY := 1
export PYTHONPATH := $(PWD)/src

setup:
	@echo "📦 Setting up Python environment..."
	python3 -m venv .venv && . .venv/bin/activate && pip install torch
	@echo "🌐 Setting up UI environment..."
	cd ui && npm install

build:
	@echo "🦀 Compiling Rust Engine..."
	. .venv/bin/activate && cargo build --release

run-native: build
	@echo "🧹 Cleaning orphaned processes..."
	-@pkill -f 'vite' || true
	-@rm -f backend.pid ui/ui.pid
	@echo "🔁 Starting Infrastructure (Redis)..."
	docker compose up -d redis
	@echo "🚀 Starting Pure Rust Tricked Backend & UI..."
	. .venv/bin/activate && export WANDB_MODE=offline REDIS_HOST=127.0.0.1 && \
	cargo run --release --bin tricked_engine & echo $$! > ../../backend.pid; \
	cd ../../ui && export BACKEND_HOST=localhost && npm run dev -- --host 0.0.0.0 & echo $$! > ui.pid; \
	trap 'kill $$(cat ../../backend.pid ui/ui.pid 2>/dev/null) 2>/dev/null || true; rm -f ../../backend.pid ui.pid' INT TERM EXIT; \
	wait

# Alias wrapper for backwards compatibility
run: run-native

run-docker:
	docker compose up --build

test: build
	@echo "🧪 Running Unit Tests & Convergence Pipeline..."
	. .venv/bin/activate && export WANDB_MODE=offline REDIS_HOST=127.0.0.1 && \
	pytest -v tests/test_convergence.py
	. .venv/bin/activate && export WANDB_MODE=offline REDIS_HOST=127.0.0.1 && \
	pytest -v tests/

clean:
	docker compose down -v
	rm -rf .venv target src/tricked_rs/target outputs multirun backend.pid ui/ui.pid
