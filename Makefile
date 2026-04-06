.PHONY: check test lint format all coverage

all: format lint test build

format:
	cargo fmt
	cd control_center && yarn format

lint:
	cargo clippy --all-targets --all-features -- -D warnings
	cd control_center && yarn typecheck

test:
	cargo test --release
	cd control_center/src-tauri && cargo test --release
	cd control_center && yarn test

sidecar:
	. venv/bin/activate && cd scripts && python build_pure_so.py
	. venv/bin/activate && export LIBTORCH_USE_PYTORCH=1 && export LIBTORCH_BYPASS_VERSION_CHECK=1 && cargo build --release --bin tricked_engine
	mkdir -p control_center/src-tauri/bin
	cp target/release/tricked_engine control_center/src-tauri/bin/tricked_engine-$$(rustc -vV | grep host | awk '{print $$2}')

build: sidecar
	. venv/bin/activate && export LIBTORCH_USE_PYTORCH=1 && export LIBTORCH_BYPASS_VERSION_CHECK=1 && cargo build --release
	cd control_center && yarn build

dev: sidecar
	. venv/bin/activate && export LIBTORCH_USE_PYTORCH=1 && export LIBTORCH_BYPASS_VERSION_CHECK=1 && export LD_LIBRARY_PATH=$$(pwd)/venv/lib/python3.13/site-packages/torch/lib:$$LD_LIBRARY_PATH && cd control_center && yarn tauri dev

start: dev
