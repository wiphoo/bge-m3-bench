from __future__ import annotations

import numpy as np


def test_encode_batch_shapes_and_counts(tokenizer, texts):
    result = tokenizer.encode_batch(texts)
    n = len(texts)
    assert result.input_ids.shape[0] == n
    assert result.input_ids.shape == result.attention_mask.shape
    assert result.input_ids.dtype == np.int64
    # token_counts == number of non-pad tokens per input.
    np.testing.assert_array_equal(result.token_counts, result.attention_mask.sum(axis=1))
    # Padded to a common length; the longest input has no padding.
    assert result.token_counts.max() == result.input_ids.shape[1]


def test_cls_token_first(tokenizer, texts):
    # The template wraps every input as [CLS] ... [SEP]; id 2 is [CLS].
    result = tokenizer.encode_batch(texts)
    assert set(result.input_ids[:, 0].tolist()) == {2}


def test_truncation(tiny_tokenizer_path):
    from bge_m3_bench.server.tokenizer import BgeTokenizer

    tok = BgeTokenizer.from_file(tiny_tokenizer_path, max_length=3)
    result = tok.encode_batch(["hello world foo bar baz the quick"])
    assert result.input_ids.shape[1] <= 3


def test_fixed_pad_length_gives_static_shape(tiny_tokenizer_path):
    from bge_m3_bench.server.tokenizer import BgeTokenizer

    # pad_length < max_length, and an input whose token count exceeds pad_length:
    # the shape must still be exactly pad_length (padding short inputs up and
    # truncating long ones down), so the CoreML EP sees one static input shape.
    tok = BgeTokenizer.from_file(tiny_tokenizer_path, max_length=64, pad_length=8)
    short = tok.encode_batch(["hi"])
    over_length = tok.encode_batch(["the quick brown fox jumps over the lazy dog again and again"])
    assert short.input_ids.shape[1] == 8
    assert short.attention_mask.shape[1] == 8
    assert over_length.input_ids.shape[1] == 8  # truncated down to the fixed length
    # The over-length input has no padding (every position is a real token).
    assert int(over_length.attention_mask.sum()) == 8
