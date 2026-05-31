from __future__ import annotations

import numpy as np
import pytest

from bge_m3_bench.server.embedder import Embedder


def test_cls_pooling_normalized(embedder, texts):
    out = embedder.embed(texts)
    assert out.embeddings.shape == (len(texts), 8)
    assert out.embeddings.dtype == np.float32
    assert np.all(np.isfinite(out.embeddings))
    # Normalized -> unit L2 norm.
    norms = np.linalg.norm(out.embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5, atol=1e-5)
    # Timings are populated and non-negative.
    assert out.tokenize_us >= 0 and out.inference_us >= 0 and out.postprocess_us >= 0
    np.testing.assert_array_equal(out.token_counts > 0, True)


def test_mean_pooling_shape(model, tokenizer, texts):
    emb = Embedder(model, tokenizer, pooling="mean", normalize=False)
    out = emb.embed(texts)
    assert out.embeddings.shape == (len(texts), 8)
    assert np.all(np.isfinite(out.embeddings))


def test_embed_is_deterministic(embedder, texts):
    a = embedder.embed(texts).embeddings
    b = embedder.embed(texts).embeddings
    np.testing.assert_array_equal(a, b)


def test_unknown_pooling_rejected(model, tokenizer):
    with pytest.raises(ValueError, match="unknown pooling"):
        Embedder(model, tokenizer, pooling="bogus")


def test_pooling_none_requires_2d(model, tokenizer, texts):
    # The tiny model emits a 3-D last_hidden_state; pooling=none must reject it.
    emb = Embedder(model, tokenizer, pooling="none", normalize=False)
    with pytest.raises(ValueError, match="2-D"):
        emb.embed(texts)
