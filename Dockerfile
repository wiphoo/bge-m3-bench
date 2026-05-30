# syntax=docker/dockerfile:1
# CPU image for the ONNX Runtime gRPC benchmark service (Milestone 5).
# Reproducible: pinned base, deterministic uv sync from the committed lockfile.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# Install uv (static binary) from the official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first for layer caching.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy source and the generated protobuf stubs, then install the project.
# README.md is required: pyproject.toml declares it as the project readme, so
# the build backend needs it present to install the project.
COPY README.md ./
COPY src ./src
COPY scripts ./scripts
COPY proto ./proto
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Default demo model (override by mounting a volume at /models).
RUN uv run python scripts/make_test_model.py --out /models/tiny_mlp.onnx

EXPOSE 50051
ENV ONNX_GRPC_HOST=0.0.0.0 ONNX_GRPC_PORT=50051 ONNX_GRPC_PROVIDER=cpu

ENTRYPOINT ["onnx-server"]
CMD ["--model", "/models/tiny_mlp.onnx", "--provider", "cpu"]
