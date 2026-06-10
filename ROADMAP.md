# Roadmap

Planned enhancements, roughly in priority order. Not commitments — open an issue
to discuss before picking one up.

- **Server thread config passthrough** — honor `intra_op_threads = 0` and
  `inter_op_threads = 0` (ONNX Runtime defaults) cleanly end-to-end when running
  the server.
- **CLI stack migration** — move from `click`/`argparse` to `typer` + `rich`,
  with `pydantic` for config validation.
- **Intel CPU optimization** — optional OpenVINO execution provider for Intel
  CPUs.
- **GPU support** — flesh out the `cuda` provider seam to actually run on GPU.
