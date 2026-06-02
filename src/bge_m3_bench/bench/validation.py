"""Embedding validation: cheap correctness checks on returned embeddings.

Always checks finiteness, dimension, norm and zero-vectors. When a local
reference embedder is available, also compares against it (cosine similarity +
max abs diff) to catch a wrong model served or a serialization defect.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_embeddings(
    embeddings: np.ndarray,
    *,
    normalize: bool,
    reference: np.ndarray | None = None,
    norm_tol: float = 1e-2,
    cosine_floor: float = 0.99,
) -> tuple[dict[str, Any], bool, list[str]]:
    """Return ``(report, passed, reasons)``.

    ``report`` matches the summary's ``validation`` section.
    """
    norms = np.linalg.norm(embeddings, axis=1)
    nan_count = int(np.isnan(embeddings).sum())
    inf_count = int(np.isinf(embeddings).sum())
    zero_count = int((norms < 1e-6).sum())

    report: dict[str, Any] = {
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "embedding_norm_mean": round(float(norms.mean()), 6),
        "embedding_norm_std": round(float(norms.std()), 6),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_vector_count": zero_count,
        "cosine_similarity_mean_vs_reference": None,
        "max_abs_diff_vs_reference": None,
    }

    reasons: list[str] = []
    if nan_count or inf_count:
        reasons.append(f"non-finite values (nan={nan_count}, inf={inf_count})")
    if zero_count:
        reasons.append(f"{zero_count} zero vector(s)")
    if normalize:
        # Check each row, not just the batch mean: offsetting bad vectors
        # (e.g. norms 0.5 and 1.5) average to ~1.0 and would slip through.
        max_dev = float(np.abs(norms - 1.0).max())
        if max_dev > norm_tol:
            worst = float(norms[np.abs(norms - 1.0).argmax()])
            reasons.append(f"worst row norm {worst:.4f} not ~1.0 (max dev {max_dev:.4f})")

    if reference is not None:
        if reference.shape != embeddings.shape:
            reasons.append(
                f"reference shape {reference.shape} != embeddings shape {embeddings.shape}"
            )
        else:
            a = embeddings.astype(np.float64)
            b = reference.astype(np.float64)
            denom = np.clip(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12, None)
            cosine = float(((a * b).sum(axis=1) / denom).mean())
            max_abs = float(np.abs(a - b).max())
            report["cosine_similarity_mean_vs_reference"] = round(cosine, 6)
            report["max_abs_diff_vs_reference"] = round(max_abs, 6)
            if cosine < cosine_floor:
                reasons.append(f"cosine vs reference {cosine:.4f} < {cosine_floor}")

    return report, not reasons, reasons
