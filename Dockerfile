# syntax=docker/dockerfile:1
# CPU image for the BGE-M3 embedding gRPC service.
# Mount the model + tokenizer at runtime (no weights are baked in).
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY README.md ./
COPY src ./src
COPY proto ./proto
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 50051
ENV BGE_M3_HOST=0.0.0.0 BGE_M3_PORT=50051 BGE_M3_PROVIDER=cpu
# Provide --model and --tokenizer (e.g. mount a volume at /models).
ENTRYPOINT ["bge-m3-server"]
CMD ["--model", "/models/model.onnx", "--tokenizer", "/models/tokenizer.json"]
