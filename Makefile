.PHONY: help sync proto lint format typecheck test cov model serve clean

UV ?= uv

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

sync: ## Install dependencies (CPU runtime + viz) into the uv-managed venv
	$(UV) sync --extra cpu --extra viz

proto: ## Generate gRPC/protobuf stubs from proto/*.proto
	$(UV) run python scripts/gen_proto.py

model: ## Build the tiny demo ONNX model
	$(UV) run --extra cpu python scripts/make_test_model.py

lint: ## Run ruff lint checks
	$(UV) run ruff check src tests scripts

format: ## Auto-format with ruff
	$(UV) run ruff format src tests scripts
	$(UV) run ruff check --fix src tests scripts

typecheck: ## Run mypy
	$(UV) run mypy src

test: ## Run the test suite
	$(UV) run --extra cpu pytest

cov: ## Run tests with coverage
	$(UV) run --extra cpu pytest --cov=onnx_grpc_benchmark --cov-report=term-missing

serve: model ## Run the gRPC server with the demo model
	$(UV) run --extra cpu onnx-server --model models/tiny_mlp.onnx --provider cpu

clean: ## Remove build/test artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
