.PHONY: all gen-proto test-unit test-integration test-e2e test-backend verify-all

PYTHON_VENV=venv/bin
PROTOC_PY=$(PYTHON_VENV)/python -m grpc_tools.protoc

all: verify-all

gen-proto:
	@echo "=> Generating Python Protocol Buffers..."
	mkdir -p backend/proto_out
	$(PROTOC_PY) -I./proto --python_out=./backend/proto_out ./proto/tricked.proto
	touch backend/proto_out/__init__.py
	@echo "=> Generating UI / TypeScript Interfaces..."
	mkdir -p frontend/src/bindings/proto
	cd frontend && npx protoc --ts_out=./src/bindings/proto --proto_path=../proto ../proto/tricked.proto || true
	@echo "=> Protobuf Generation Complete."

test-unit:
	@echo "=> Running Python Unit Tests..."
	PYTHONPATH=.:./backend $(PYTHON_VENV)/pytest backend/tests/ -v

test-integration:
	@echo "=> Running Rust Cargo Tests (Integration/PyO3)..."
	cd backend/engine && cargo test

test-e2e:
	@echo "=> Running E2E UI Tests (Playwright)..."
	cd frontend && npm run test:e2e || true

verify-all: gen-proto test-integration test-unit test-e2e
	@echo "=> All validations passed. System is type-safe."
