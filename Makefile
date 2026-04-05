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
	cargo build --release --bin tricked_engine
	mkdir -p control_center/src-tauri/bin
	cp target/release/tricked_engine control_center/src-tauri/bin/tricked_engine-$$(rustc -vV | grep host | awk '{print $$2}')

build: sidecar
	cargo build --release
	cd control_center && yarn build

dev: sidecar
	cd control_center && yarn tauri dev

start: dev
