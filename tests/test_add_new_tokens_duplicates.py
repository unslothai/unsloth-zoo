# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""A repeated entry in ``new_tokens`` must not shift later interpolation rows."""

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from unsloth_zoo.tokenizer_utils import add_new_tokens

VOCAB_TOK = 100


def _build(padded):
    torch.manual_seed(0)
    vocab = {f"tok{i}": i for i in range(VOCAB_TOK)}
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
    return LlamaForCausalLM(cfg).eval(), tokenizer


@pytest.mark.parametrize("padded", [VOCAB_TOK, VOCAB_TOK + 8])
def test_repeated_new_token_keeps_one_row_per_token(padded):
    model, tokenizer = _build(padded)
    emb = model.get_input_embeddings().weight
    before = emb.detach().clone()
    first, second = "tok5 tok6", "tok7 tok8"

    add_new_tokens(
        model, tokenizer, new_tokens = [first, first, second], method = "interpolation",
    )

    emb = model.get_input_embeddings().weight
    assert len(tokenizer) == VOCAB_TOK + 2
    assert emb.shape[0] == padded + (0 if padded > VOCAB_TOK else 2)
    assert torch.equal(emb[:VOCAB_TOK], before[:VOCAB_TOK])
    mean = before.float().mean(0)
    for token, pieces in ((first, [5, 6]), (second, [7, 8])):
        row = emb[tokenizer.convert_tokens_to_ids(token)]
        expected = mean * 0.5 + before[pieces].float().mean(0) * 0.5
        assert torch.allclose(row, expected, atol = 1e-5), token
    if padded > VOCAB_TOK:
        assert torch.equal(emb[VOCAB_TOK + 2:padded], before[VOCAB_TOK + 2:padded])
