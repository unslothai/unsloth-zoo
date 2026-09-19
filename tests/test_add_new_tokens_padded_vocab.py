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

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from unsloth_zoo.tokenizer_utils import add_new_tokens, fix_untrained_tokens

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

    assert emb2.shape[0] == PADDED, "padded embedding must not shrink"
    assert torch.equal(emb2[:VOCAB_TOK], trained_rows), "trained rows corrupted"

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

    assert emb2.data_ptr() == head2.data_ptr(), "tie broken by resize"
    assert torch.equal(emb2[VOCAB_TOK], head2[VOCAB_TOK])

    assert model.config.vocab_size == PADDED
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        reloaded = LlamaForCausalLM.from_pretrained(d)
    assert reloaded.get_input_embeddings().weight.shape[0] == PADDED
    assert reloaded.config.vocab_size == PADDED

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


def test_add_new_token_keeps_negative_trained_rows_in_the_mean():
    model, tokenizer = _build(4, 4)
    weight = model.get_input_embeddings().weight
    with torch.no_grad():
        weight[0].fill_(-1)
        weight[1].fill_(-2)
        weight[2].fill_(-3)
        weight[3].zero_()
    expected = torch.full((model.config.hidden_size,), -2.0)
    add_new_tokens(model, tokenizer, new_tokens=["<new>"])
    new_id = tokenizer.convert_tokens_to_ids("<new>")
    torch.testing.assert_close(model.get_input_embeddings().weight[new_id], expected)
    loss = model(torch.tensor([[new_id, 0]]), labels=torch.tensor([[new_id, 0]])).loss
    assert torch.isfinite(loss)


@pytest.mark.parametrize("row_value, untrained", [(-3.0, False), (0.0, True), (1e-18, True), (-1e-18, True)])
def test_frozen_token_validation_uses_magnitude(row_value: float, untrained: bool):
    model, tokenizer = _build(4, 4)
    with torch.no_grad():
        model.get_input_embeddings().weight.fill_(1)
        model.get_input_embeddings().weight[2].fill_(row_value)
    model.requires_grad_(False)
    original = model.get_input_embeddings().weight.clone()
    dataset = Dataset.from_dict({"input_ids": [[2, 3]]})
    if untrained:
        with pytest.raises(ValueError, match="Untrained tokens"):
            fix_untrained_tokens(model, tokenizer, dataset)
    else:
        fix_untrained_tokens(model, tokenizer, dataset)
    torch.testing.assert_close(model.get_input_embeddings().weight, original)
