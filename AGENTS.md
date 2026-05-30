# Repository Agent Instructions

## Project Overview

`bge-m3-bench` is a production-ready gRPC service wrapping ONNX Runtime plus a
reproducible benchmark platform. It benchmarks ONNX inference locally, over
gRPC, in Docker, on a VPS, or in Kubernetes. Benchmark results capture latency
statistics, output validation status, and environment metadata.

The Python distribution is named `onnx-grpc-benchmark`.

## Tech Stack

- **Language:** Python `>=3.12` (ruff/mypy target `py312`).
- **Package manager:** `uv` (lockfile `uv.lock`, all work runs through `uv run`).
- **Build backend:** `hatchling`; wheel packages `src/onnx_grpc_benchmark`.
- **Core deps:** `onnxruntime`, `onnx`, `numpy`, `grpcio`,
  `grpcio-health-checking`, `protobuf`, `psutil`, `click`.
- **Optional extras:** `viz` (matplotlib/pandas/jupyter), `gpu`
  (`onnxruntime-gpu`), `openvino` (`onnxruntime-openvino`).
- **Tooling:** ruff (lint + format), mypy (strict-ish: `disallow_untyped_defs`),
  pytest (+ pytest-cov), pre-commit.
- **RPC:** protobuf/gRPC; stubs generated from `proto/inference.proto`.
- **Deployment:** Docker (`Dockerfile`, `Dockerfile.cuda`, `Dockerfile.openvino`)
  and Kubernetes (`deploy/k8s/`).
- **CI:** GitHub Actions — `.github/workflows/ci.yml` (lint/typecheck/test +
  image build/smoke-test) and `.github/workflows/publish.yml`.

## Repository Structure

- `src/onnx_grpc_benchmark/` — main package.
  - `benchmark/` — CLI (`cli.py`), `runner.py`, `dataset.py`, `stats.py`,
    `validation.py`, `report.py`, `viz.py`.
  - `server/` — gRPC server (`grpc_server.py`, `__main__.py`), `service.py`,
    `runtime.py`, `registry.py`, `serialization.py`, `client.py`.
  - `common/` — `config.py`, `logging.py`, `providers.py`, `dtypes.py`.
  - `metadata/` — environment metadata `collector.py`.
  - `generated/` — generated protobuf/gRPC stubs. **Do not hand-edit.**
- `proto/inference.proto` — RPC contract; regenerate stubs after changes.
- `scripts/` — `gen_proto.py`, `make_test_model.py`, `build_report.py`.
- `deploy/k8s/` — `deployment.yaml`, `benchmark-job.yaml`, `run-benchmark.sh`.
- `docs/` — architecture, metadata schema, and E2E runbooks (local/gRPC,
  Docker, VPS, Kubernetes), plus model export and grpcurl guides.
- `models/`, `results/` — demo model and benchmark output artifacts.

Console entry points: `onnx-bench` (benchmark CLI) and `onnx-server` (gRPC
server).

## Agent Operating Rules

Agents working in this repository must:

1. Read this file before making changes.
2. Understand the current task before editing code.
3. Inspect related files before proposing a solution.
4. Prefer small, focused changes.
5. Keep implementation and tests aligned.
6. Update documentation when behavior changes.
7. Do not introduce unrelated refactors.
8. Do not remove existing functionality unless explicitly requested.
9. Run relevant checks before declaring completion.
10. Report clearly what changed, what was tested, and what remains.

## Development Workflow

1. Read the request/issue and identify affected modules.
2. Make the smallest safe change.
3. If you change `proto/inference.proto`, regenerate stubs with `make proto`
   (never edit `src/onnx_grpc_benchmark/generated/` by hand).
4. Add or update tests under `tests/`.
5. Run `make lint`, `make typecheck`, and `make test` (or the underlying
   `uv run` commands) before declaring completion.
6. Summarize the result.

All commands run through `uv`. Run `make sync` first to set up the venv.

### Common commands

```bash
make sync       # uv sync --all-extras (install deps)
make model      # build models/tiny_mlp.onnx demo model
make proto      # regenerate protobuf/gRPC stubs from proto/*.proto
make lint       # uv run ruff check src tests scripts
make format     # uv run ruff format + ruff check --fix
make typecheck  # uv run mypy src
make test       # uv run pytest
make cov        # pytest with coverage
make serve      # run local gRPC server with the demo model
```

## Branch Rules

Use focused branches: `<type>/<short-description>`, e.g.
`feat/streaming-infer`, `fix/grpc-health-check`, `chore/ci-image-cache`.

## Commit Rules

Use clear, scoped commit messages, e.g.
`feat(server): add batched inference endpoint`,
`fix(benchmark): correct p99 latency calc`,
`chore(ci): cache uv downloads`.

## Testing Requirements

Run the most relevant checks before finishing:

```bash
uv run ruff check src tests scripts
uv run mypy src
uv run pytest
```

If a check cannot be run, explain why. ruff and mypy exclude the generated
protobuf stubs by configuration; do not work around those exclusions.

## Code Quality Rules

- Keep code readable and maintainable; prefer explicit names.
- All functions must be typed (`disallow_untyped_defs` is enabled).
- Respect ruff rule sets `E, F, I, UP, B, C4, SIM, RUF` and line length 100.
- Avoid large rewrites unless required.
- Keep the gRPC/proto contract and CLI flags backward compatible unless the
  task says otherwise.
- Validate inputs at boundaries; handle errors intentionally.
- Keep secrets out of source control.

## Documentation Rules

Update documentation when changing setup steps, commands, environment
variables, the gRPC/CLI API, deployment behavior (Docker/k8s), the metadata
schema, or the agent workflow. Relevant docs live in `docs/` and `README.md`.

## Security Rules

Agents must not:

- commit secrets, tokens, private keys, or credentials;
- expose internal infrastructure unnecessarily;
- weaken authentication or authorization;
- disable security checks without explicit approval;
- add dependencies without checking purpose and risk.

## Pull Request Rules

PR descriptions should include:

```markdown
## Summary

- What changed
- Why it changed

## Tests

- [ ] Unit tests
- [ ] Integration tests
- [ ] Lint/typecheck
- [ ] Manual verification

## Notes

Mention risks, follow-ups, skipped checks, or known limitations.
```

## Agent File Policy

This file is the canonical instruction source for all coding agents.

Tool-specific files must symlink to this file:

```text
CLAUDE.md    -> AGENTS.md
GEMINI.md    -> AGENTS.md
codex.md     -> AGENTS.md
opencode.md  -> AGENTS.md
```

Do not copy this file into tool-specific files. Do not maintain separate
duplicated instruction files unless a tool absolutely requires a different
format.

## Completion Checklist

Before finishing, report:

- Files changed
- Summary of changes
- Tests/checks run
- Checks not run
- Risks or follow-up work
