.PHONY: check test lint format all coverage setup

PY_VER := $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
TORCH_ENV := . venv/bin/activate && export LIBTORCH_USE_PYTORCH=1 && export TORCH_CUDA_VERSION=cu130 && export LIBTORCH_BYPASS_VERSION_CHECK=1 && export LIBTORCH_CXX11_ABI=1 && export LD_LIBRARY_PATH=$(PWD)/venv/lib/python$(PY_VER)/site-packages/torch/lib:$(PWD)/venv/lib/python$(PY_VER)/site-packages/nvidia/cu13/lib:$$LD_LIBRARY_PATH && export TORCH_CPP_LOG_LEVEL=ERROR

all: setup format lint test build

setup:
	@if [ ! -d "venv" ]; then \
		python3 -m venv venv && \
		. venv/bin/activate && pip install torch; \
	fi
	@if [ ! -d "control_center/node_modules" ]; then \
		cd control_center && npm install; \
	fi

format:
	cargo fmt
	cd control_center && npm run format

lint: setup_assets
	$(TORCH_ENV) && cargo clippy --all-targets --all-features -- -D warnings
	cd control_center && npm run typecheck

test: setup_assets
	$(TORCH_ENV) && cargo test --release
	$(TORCH_ENV) && cd control_center/src-tauri && cargo test --release
	cd control_center && npm run test

setup_assets:
	mkdir -p assets
	. venv/bin/activate && cd scripts && python build_pure_so.py

build: setup_assets
	@echo "Compiling via official Tauri CLI to inject Asset Bundle natively..."
	$(TORCH_ENV) && cd control_center && npm run tauri build -- --no-bundle

dev: setup_assets
	$(TORCH_ENV) && cd control_center && npm run tauri dev

start: setup_assets
	@if [ ! -f "target/release/control_center" ]; then \
		echo "Optimized binary not found. Running 'make build' first..."; \
		$(MAKE) build; \
	fi
	@echo "Booting the optimized native release build..."
	$(TORCH_ENV) && ./target/release/control_center
	
release: setup_assets
	@echo "Staging libraries for Linux release..."
	./scripts/stage_release.sh
	@echo "Compiling optimized unmanaged release packages (.AppImage, .deb)..."
	$(TORCH_ENV) && cd control_center && npm run tauri build
	@echo "Success! Packages located in: control_center/src-tauri/target/release/bundle/"
