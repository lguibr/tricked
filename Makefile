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
