.PHONY: help sync sync-export proto lint format typecheck test cov model serve clean

UV ?= uv

# `make model` parameters: MODEL is tiny|bge-m3, PRECISION is fp32|fp16|int8.
MODEL ?= tiny
PRECISION ?= fp32

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

sync: ## Install dependencies into the uv-managed venv
	$(UV) sync

sync-export: ## Install the optional model-export deps (transformers/torch/optimum)
	$(UV) sync --group export

proto: ## Generate gRPC/protobuf stubs from proto/*.proto
	$(UV) run python scripts/gen_proto.py

model: ## Build a model + tokenizer (MODEL=tiny|bge-m3 PRECISION=fp32|fp16|int8)
	$(UV) run --group export python scripts/make_model.py --model $(MODEL) --precision $(PRECISION)

lint: ## Run ruff lint checks
	$(UV) run ruff check src tests scripts

format: ## Auto-format with ruff
	$(UV) run ruff format src tests scripts
	$(UV) run ruff check --fix src tests scripts

typecheck: ## Run mypy
	$(UV) run mypy src

test: ## Run the test suite
	$(UV) run pytest

cov: ## Run tests with coverage
	$(UV) run pytest --cov=bge_m3_bench --cov-report=term-missing

clean: ## Remove build/test artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
