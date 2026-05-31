#!/usr/bin/env python3
"""Generate gRPC/protobuf stubs into the package's ``generated`` directory.

grpcio-tools emits ``import embedding_pb2 as ...`` which only works if the
output dir is on sys.path. We rewrite it to a package-relative import so the
stubs import cleanly as ``bge_m3_bench.generated.*``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = ROOT / "proto"
OUT_DIR = ROOT / "src" / "bge_m3_bench" / "generated"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    protos = sorted(PROTO_DIR.glob("*.proto"))
    if not protos:
        print("no .proto files found", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        *[str(p) for p in protos],
    ]
    subprocess.run(cmd, check=True)

    # Rewrite absolute stub imports to package-relative imports.
    for grpc_file in OUT_DIR.glob("*_pb2_grpc.py"):
        text = grpc_file.read_text()
        text = re.sub(r"^import (\w+_pb2) as", r"from . import \1 as", text, flags=re.MULTILINE)
        grpc_file.write_text(text)

    print(f"generated stubs in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
