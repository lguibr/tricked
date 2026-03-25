.PHONY: build test-rust test-python run

build:
	@echo "Compiling Rust Engine..."
	maturin develop --release

test-rust:
	cd src/tricked_rs && cargo test

test-python:
	pytest tests/ -v

run: build
	@echo "Starting Ray Cluster and Training..."
	python src/tricked/main.py
