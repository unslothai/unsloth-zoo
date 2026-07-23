# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression test for #121: add_new_tokens must not shrink a padded embedding.

Some models ship an embedding padded LARGER than the tokenizer vocab (Gemma3:
262208 rows vs 262145 tokens). The old code resized to len(tokenizer), which
SHRANK the matrix and silently destroyed the already-trained rows past the
tokenizer length (no exception raised). The fix never resizes below the existing
embedding, fills only the genuinely-new rows with the trained mean, keeps tied
weights tied, and keeps config.vocab_size equal to the real matrix row count so
the model still round-trips through save_pretrained/from_pretrained.

Fully synthetic (tiny tied Llama, ~100-token tokenizer), CPU only, no download.
"""

import tempfile

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from unsloth_zoo.tokenizer_utils import add_new_tokens

VOCAB_TOK = 100  # real tokens in the tokenizer
PADDED = 128     # embedding rows shipped by the model (padding = [100, 128))


def _build(vocab_tok, padded):
    torch.manual_seed(0)
    vocab = {f"tok{i}": i for i in range(vocab_tok)}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="tok0"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="tok0", pad_token="tok1",
    )
    cfg = LlamaConfig(
        vocab_size=padded, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model, tokenizer


def test_padded_embedding_preserves_trained_rows_and_places_new_tokens():
    model, tokenizer = _build(VOCAB_TOK, PADDED)
    emb = model.get_input_embeddings().weight

    # Snapshot everything the fix must preserve, and mark the alignment padding
    # rows [VOCAB_TOK, PADDED) with a distinctive sentinel so we can tell if a
    # new token wrongly resolves to a leftover padding row.
    trained_rows = emb.detach()[:VOCAB_TOK].clone()
    with torch.no_grad():
        emb[VOCAB_TOK:PADDED] = 7.0
    sentinel_rows = emb.detach()[VOCAB_TOK:PADDED].clone()
    assert emb.data_ptr() == model.get_output_embeddings().weight.data_ptr()

    new_tokens = ["<newA>", "<newB>"]
    add_new_tokens(model, tokenizer, new_tokens=new_tokens)

    emb2 = model.get_input_embeddings().weight
    head2 = model.get_output_embeddings().weight

    # 1. No shrink: all trained rows survive byte-identical.
    assert emb2.shape[0] == PADDED, "padded embedding must not shrink"
    assert torch.equal(emb2[:VOCAB_TOK], trained_rows), "trained rows corrupted"

    # 2. New tokens land at their real tokenizer IDs, each pointing at its OWN
    #    mean-initialised row (not a leftover padding row).
    new_len = len(tokenizer)
    for offset, tok in enumerate(new_tokens):
        tid = tokenizer.convert_tokens_to_ids(tok)
        assert tid == VOCAB_TOK + offset
        assert tokenizer(tok, add_special_tokens=False).input_ids == [tid]
        row = emb2[tid]
        assert not torch.allclose(row, torch.full_like(row, 7.0)), \
            "new token resolved to a leftover padding row"
        # Row is real (finite, non-zero), i.e. mean-initialised.
        assert torch.isfinite(row).all() and row.abs().sum() > 0

    # 3. mean-init touched ONLY the genuinely-new rows: the leftover padding
    #    beyond the new tokens is exactly as shipped.
    assert torch.equal(emb2[new_len:PADDED], sentinel_rows[new_len - VOCAB_TOK:]), \
        "alignment padding rows were overwritten"

    # 4. Tied weights stay tied, and share storage at a new-token row.
    assert emb2.data_ptr() == head2.data_ptr(), "tie broken by resize"
    assert torch.equal(emb2[VOCAB_TOK], head2[VOCAB_TOK])

    # 5. config.vocab_size tracks the matrix, so the model round-trips.
    assert model.config.vocab_size == PADDED
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        reloaded = LlamaForCausalLM.from_pretrained(d)
    assert reloaded.get_input_embeddings().weight.shape[0] == PADDED
    assert reloaded.config.vocab_size == PADDED

    # 6. Forward pass over the new IDs works.
    with torch.no_grad():
        logits = model(torch.tensor([[VOCAB_TOK, VOCAB_TOK + 1, 3, 4]])).logits
    assert logits.shape[-1] == PADDED


def test_non_padded_embedding_still_grows_normally():
    # embedding == tokenizer length: the ordinary case must be unaffected.
    model, tokenizer = _build(VOCAB_TOK, VOCAB_TOK)
    trained_rows = model.get_input_embeddings().weight.detach()[:VOCAB_TOK].clone()

    add_new_tokens(model, tokenizer, new_tokens=["<newA>", "<newB>"])

    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    assert emb.shape[0] == VOCAB_TOK + 2, "non-padded model must grow to fit"
    assert torch.equal(emb[:VOCAB_TOK], trained_rows), "trained rows corrupted"
    assert model.config.vocab_size == VOCAB_TOK + 2
    assert emb.data_ptr() == head.data_ptr()
    for offset in range(2):
        row = emb[VOCAB_TOK + offset]
        assert torch.isfinite(row).all() and row.abs().sum() > 0


if __name__ == "__main__":
    test_padded_embedding_preserves_trained_rows_and_places_new_tokens()
    test_non_padded_embedding_still_grows_normally()
    print("ok")
