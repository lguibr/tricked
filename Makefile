.PHONY: check test lint format all coverage setup

PY_VER := $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
TORCH_ENV := . venv/bin/activate && export LIBTORCH_USE_PYTORCH=1 && export LIBTORCH_BYPASS_VERSION_CHECK=1 && export LIBTORCH_CXX11_ABI=1 && export LD_LIBRARY_PATH=$(PWD)/venv/lib/python$(PY_VER)/site-packages/torch/lib:$$LD_LIBRARY_PATH

all: setup format lint test build

setup:
	@if [ ! -d "venv" ]; then \
		python3 -m venv venv && \
		. venv/bin/activate && pip install torch optuna optunahub pandas streamlit; \
	fi
	@if [ ! -d "control_center/node_modules" ]; then \
		cd control_center && npm install; \
	fi

format:
	cargo fmt
	cd control_center && npm run format

lint: sidecar
	$(TORCH_ENV) && cargo clippy --all-targets --all-features -- -D warnings
	cd control_center && npm run typecheck

test: sidecar
	$(TORCH_ENV) && cargo test --release
	cd control_center/src-tauri && cargo test --release
	cd control_center && npm run test

sidecar:
	mkdir -p assets
	. venv/bin/activate && cd scripts && python build_pure_so.py
	. venv/bin/activate && cd scripts && python export_math_kernels.py ../assets/math_kernels.pt
	$(TORCH_ENV) && cargo build --release --bin tricked_engine
	mkdir -p control_center/src-tauri/bin
	cp target/release/tricked_engine control_center/src-tauri/bin/tricked_engine-$$(rustc -vV | grep host | awk '{print $$2}')

build: sidecar
	$(TORCH_ENV) && cargo build --release
	cd control_center && npm run build

dev: sidecar
	$(TORCH_ENV) && cd control_center && npm run tauri dev

start: dev
