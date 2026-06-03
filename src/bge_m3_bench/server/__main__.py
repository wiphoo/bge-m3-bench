"""Server entrypoint: ``python -m bge_m3_bench.server`` / ``bge-m3-server``."""

from __future__ import annotations

import argparse
from dataclasses import replace

from ..common.config import ServerConfig
from ..common.logging import configure_logging
from .grpc_server import serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BGE-M3 embedding gRPC server")
    parser.add_argument("--model", help="Path to the ONNX embedding model.")
    parser.add_argument("--tokenizer", help="Path to tokenizer.json.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--provider", default=None, help="cpu|cuda")
    parser.add_argument("--pooling", default=None, help="none|cls|mean")
    parser.add_argument("--normalize", dest="normalize", action="store_true", default=None)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="gRPC server thread pool size (max concurrent in-flight requests).",
    )
    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=None,
        help=(
            "ONNX Runtime intra-op threads. -1 (default) = auto: "
            "max(1, usable_cores // max_workers) — usable_cores is the host "
            "physical count capped by any cgroup quota / affinity — to avoid CPU "
            "oversubscription under concurrency. 0 = ORT default (all cores). >0 = explicit."
        ),
    )
    parser.add_argument(
        "--inter-op-threads",
        type=int,
        default=None,
        help="ONNX Runtime inter-op threads (0 = ORT default).",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ServerConfig:
    """Merge parsed CLI args over the env-derived base config.

    Uses ``is None`` rather than ``or`` for numeric fields so an explicit ``0``
    (ONNX Runtime default thread count) is honored instead of falling back.
    """
    base = ServerConfig.from_env()
    return replace(
        base,
        host=args.host or base.host,
        port=args.port or base.port,
        max_workers=base.max_workers if args.max_workers is None else args.max_workers,
        provider=args.provider or base.provider,
        model_path=args.model or base.model_path,
        tokenizer_path=args.tokenizer or base.tokenizer_path,
        pooling=args.pooling or base.pooling,
        normalize=base.normalize if args.normalize is None else args.normalize,
        max_length=args.max_length or base.max_length,
        intra_op_threads=(
            base.intra_op_threads if args.intra_op_threads is None else args.intra_op_threads
        ),
        inter_op_threads=(
            base.inter_op_threads if args.inter_op_threads is None else args.inter_op_threads
        ),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    configure_logging(config.log_level)

    if not config.model_path or not config.tokenizer_path:
        parser.error("--model and --tokenizer are required (or set BGE_M3_MODEL/BGE_M3_TOKENIZER)")

    serve(config)


if __name__ == "__main__":
    main()
