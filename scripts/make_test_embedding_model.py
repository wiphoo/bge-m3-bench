#!/usr/bin/env python3
"""Build a tiny, deterministic embedding model + tokenizer for tests and demos.

The model embeds token ids via a fixed lookup table and returns a
``last_hidden_state`` of shape ``[batch, seq, HIDDEN]`` — enough to exercise
cls/mean pooling and normalization without downloading BGE-M3. The tokenizer is
a tiny WordLevel tokenizer that wraps inputs with ``[CLS] ... [SEP]`` so cls
pooling is meaningful.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

VOCAB = 32
HIDDEN = 8


def build_model(seed: int = 0) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)
    table = rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32)

    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["batch", "seq"]
    )
    out = helper.make_tensor_value_info(
        "last_hidden_state", TensorProto.FLOAT, ["batch", "seq", HIDDEN]
    )

    w = numpy_helper.from_array(table, name="embed_table")
    # last_hidden_state[b, s, :] = embed_table[input_ids[b, s], :]
    gather = helper.make_node("Gather", ["embed_table", "input_ids"], ["last_hidden_state"], axis=0)

    graph = helper.make_graph(
        [gather],
        "tiny_embed",
        [input_ids, attention_mask],  # attention_mask declared (used by mean pooling on the server)
        [out],
        initializer=[w],
    )
    model = helper.make_model(
        graph,
        producer_name="bge-m3-bench",
        opset_imports=[helper.make_operatorsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def build_tokenizer() -> Any:
    """Return a tiny WordLevel ``tokenizers.Tokenizer`` wrapping [CLS] ... [SEP]."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    words = ["hello", "world", "foo", "bar", "baz", "the", "quick", "brown", "fox", "lorem"]
    for i, word in enumerate(words):
        vocab[word] = 4 + i

    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return tok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/tiny_embed.onnx")
    parser.add_argument("--tokenizer", default="models/tiny_tokenizer.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(args.seed), out)

    tok_path = Path(args.tokenizer)
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    build_tokenizer().save(str(tok_path))  # type: ignore[attr-defined]

    print(f"wrote {out} and {tok_path}")


if __name__ == "__main__":
    main()
