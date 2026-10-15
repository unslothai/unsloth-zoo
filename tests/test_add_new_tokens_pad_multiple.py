# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in row alignment for add_new_tokens (pad_to_multiple_of / UNSLOTH_VOCAB_PAD_MULTIPLE).

Default: the embedding grows to exactly len(tokenizer) (unchanged behaviour).
Opt-in: rows round up to the multiple, the tokenizer length is unchanged, the new token row
is mean-initialised as before, the alignment rows are deterministic and listed in
generation_config.suppress_tokens so HF generate never emits them, and the model round-trips
through save_pretrained / from_pretrained with the padded config.vocab_size.

CPU only, tiny model from the local HF cache (hf-internal-testing/tiny-random-LlamaForCausalLM).
"""

import os

import pytest
import torch

transformers = pytest.importorskip("transformers")

from transformers import AutoModelForCausalLM, AutoTokenizer

from unsloth_zoo.tokenizer_utils import add_new_tokens

REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _load():
    try:
        model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype = torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(REPO)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"tiny model unavailable: {exc}")
    model.eval()
    return model, tokenizer


def _rows(model):
    return (
        model.get_input_embeddings().weight.shape[0],
        model.get_output_embeddings().weight.shape[0],
        model.config.vocab_size,
    )


def test_default_path_rows_equal_tokenizer_length(monkeypatch):
    monkeypatch.delenv("UNSLOTH_VOCAB_PAD_MULTIPLE", raising = False)
    model, tokenizer = _load()
    add_new_tokens(model, tokenizer, new_tokens = ["<unsloth_new_tok>"])
    n = len(tokenizer)
    assert _rows(model) == (n, n, n)
    assert not (getattr(model.generation_config, "suppress_tokens", None) or [])


@pytest.mark.parametrize("via_env", [False, True])
def test_opt_in_pads_rows_and_suppresses_them(monkeypatch, tmp_path, via_env):
    monkeypatch.delenv("UNSLOTH_VOCAB_PAD_MULTIPLE", raising = False)
    # Reference row from the default path (same weights, same mean init).
    ref_model, ref_tok = _load()
    add_new_tokens(ref_model, ref_tok, new_tokens = ["<unsloth_new_tok>"])
    new_id = ref_tok.convert_tokens_to_ids("<unsloth_new_tok>")
    ref_emb  = ref_model.get_input_embeddings ().weight[new_id].detach().clone()
    ref_head = ref_model.get_output_embeddings().weight[new_id].detach().clone()

    model, tokenizer = _load()
    old_rows = model.get_input_embeddings().weight.shape[0]
    old_trained = model.get_input_embeddings().weight[:old_rows].detach().clone()
    if via_env:
        monkeypatch.setenv("UNSLOTH_VOCAB_PAD_MULTIPLE", "64")
        add_new_tokens(model, tokenizer, new_tokens = ["<unsloth_new_tok>"])
    else:
        add_new_tokens(model, tokenizer, new_tokens = ["<unsloth_new_tok>"], pad_to_multiple_of = 64)

    n = len(tokenizer)
    assert n == len(ref_tok) == old_rows + 1, "tokenizer length must not change"
    padded = -(-n // 64) * 64
    assert padded > n
    assert _rows(model) == (padded, padded, padded)

    emb  = model.get_input_embeddings ().weight
    head = model.get_output_embeddings().weight
    assert torch.equal(emb[:old_rows], old_trained), "trained rows changed"
    assert torch.equal(emb [new_id], ref_emb),  "new token row must be initialised as before"
    assert torch.equal(head[new_id], ref_head), "new token lm_head row must be initialised as before"
    # Alignment rows are deterministic (the trained mean), not random.
    assert torch.equal(emb [n:], ref_emb .expand(padded - n, -1))
    assert torch.equal(head[n:], ref_head.expand(padded - n, -1))

    assert list(model.generation_config.suppress_tokens) == list(range(n, padded))

    # Make the alignment rows win the argmax: generate must still never emit them.
    with torch.no_grad():
        head[n:] = 0.0
        head[n:, 0] = 1e4
    inputs = tokenizer("hello world", return_tensors = "pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    assert int(logits[0, -1].argmax()) >= n, "sanity: the rigged rows must dominate the raw logits"
    out = model.generate(**inputs, max_new_tokens = 4, do_sample = False, pad_token_id = 0)
    assert int(out[0, inputs["input_ids"].shape[1]:].max()) < n
    with torch.no_grad():
        head[n:] = ref_head

    # Merged save round trip with the padded vocab.
    model.save_pretrained(tmp_path / "m")
    tokenizer.save_pretrained(tmp_path / "m")
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "m", torch_dtype = torch.float32)
    tok2 = AutoTokenizer.from_pretrained(tmp_path / "m")
    assert len(tok2) == n
    assert _rows(reloaded) == (padded, padded, padded)
    assert torch.equal(reloaded.get_input_embeddings().weight, emb.detach())
    assert list(reloaded.generation_config.suppress_tokens) == list(range(n, padded))


def test_opt_in_noop_when_existing_padding_absorbs_tokens(monkeypatch):
    monkeypatch.delenv("UNSLOTH_VOCAB_PAD_MULTIPLE", raising = False)
    model, tokenizer = _load()
    rows = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(rows + 10, mean_resizing = False)  # ship 10 spare rows
    add_new_tokens(model, tokenizer, new_tokens = ["<unsloth_new_tok>"], pad_to_multiple_of = 64)
    assert _rows(model) == (rows + 10, rows + 10, rows + 10)
    assert not (getattr(model.generation_config, "suppress_tokens", None) or [])


def test_invalid_env_is_ignored(monkeypatch):
    monkeypatch.setenv("UNSLOTH_VOCAB_PAD_MULTIPLE", "abc")
    model, tokenizer = _load()
    add_new_tokens(model, tokenizer, new_tokens = ["<unsloth_new_tok>"])
    n = len(tokenizer)
    assert _rows(model) == (n, n, n)
