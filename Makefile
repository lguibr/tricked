.PHONY: check test lint format all

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
