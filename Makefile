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

build:
	cargo build --release
	cd control_center && yarn build
