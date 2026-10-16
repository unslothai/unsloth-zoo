# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in row alignment for add_new_tokens (pad_to_multiple_of / UNSLOTH_VOCAB_PAD_MULTIPLE)."""

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
    model.resize_token_embeddings(rows + 10, mean_resizing = False)
    add_new_tokens(model, tokenizer, new_tokens = ["<unsloth_new_tok>"], pad_to_multiple_of = 64)
    assert _rows(model) == (rows + 10, rows + 10, rows + 10)
    assert not (getattr(model.generation_config, "suppress_tokens", None) or [])


def test_invalid_env_is_ignored(monkeypatch):
    monkeypatch.setenv("UNSLOTH_VOCAB_PAD_MULTIPLE", "abc")
    model, tokenizer = _load()
    add_new_tokens(model, tokenizer, new_tokens = ["<unsloth_new_tok>"])
    n = len(tokenizer)
    assert _rows(model) == (n, n, n)


def test_second_call_unsuppresses_the_rows_it_turns_into_tokens(monkeypatch):
    monkeypatch.delenv("UNSLOTH_VOCAB_PAD_MULTIPLE", raising = False)
    model, tokenizer = _load()
    add_new_tokens(model, tokenizer, new_tokens = ["<unsloth_a>"], pad_to_multiple_of = 64)
    n1 = len(tokenizer)
    padded = model.get_input_embeddings().weight.shape[0]
    assert list(model.generation_config.suppress_tokens) == list(range(n1, padded))
    add_new_tokens(model, tokenizer, new_tokens = ["<unsloth_b>"], pad_to_multiple_of = 64)
    new_id = tokenizer.convert_tokens_to_ids("<unsloth_b>")
    assert new_id == n1 and _rows(model) == (padded, padded, padded)
    assert list(model.generation_config.suppress_tokens) == list(range(n1 + 1, padded))


def test_padding_off_keeps_suppression_it_does_not_own(monkeypatch):
    # The model's own suppress_tokens may name a spare row a new token now takes: without
    # padding, add_new_tokens must leave that list exactly as it was.
    monkeypatch.delenv("UNSLOTH_VOCAB_PAD_MULTIPLE", raising = False)
    model, tokenizer = _load()
    rows = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(rows + 16, mean_resizing = False)
    model.generation_config.suppress_tokens = [7, rows]
    add_new_tokens(model, tokenizer, new_tokens = ["<unsloth_new_tok>"])
    assert tokenizer.convert_tokens_to_ids("<unsloth_new_tok>") == rows
    assert list(model.generation_config.suppress_tokens) == [7, rows]
    assert not hasattr(model.config, "unsloth_vocab_pad_ids")


def test_padded_call_releases_only_its_own_rows(monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_VOCAB_PAD_MULTIPLE", raising = False)
    model, tokenizer = _load()
    add_new_tokens(model, tokenizer, new_tokens = ["<unsloth_a>"], pad_to_multiple_of = 64)
    n1 = len(tokenizer)
    padded = model.get_input_embeddings().weight.shape[0]
    assert model.config.unsloth_vocab_pad_ids == list(range(n1, padded))
    # The caller suppresses an id of its own: still suppressed after it becomes a token.
    model.generation_config.suppress_tokens = [7] + list(model.generation_config.suppress_tokens)

    # Ownership survives save / load, so a later call (padding off) still releases the reused row.
    model.save_pretrained(tmp_path / "m")
    tokenizer.save_pretrained(tmp_path / "m")
    model = AutoModelForCausalLM.from_pretrained(tmp_path / "m", torch_dtype = torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(tmp_path / "m")
    assert model.config.unsloth_vocab_pad_ids == list(range(n1, padded))
    add_new_tokens(model, tokenizer, new_tokens = ["<unsloth_b>", "<unsloth_c>"])
    assert tokenizer.convert_tokens_to_ids("<unsloth_c>") == n1 + 1
    assert list(model.generation_config.suppress_tokens) == [7] + list(range(n1 + 2, padded))
    assert model.config.unsloth_vocab_pad_ids == list(range(n1 + 2, padded))


def test_vllm_generate_biases_padded_rows(monkeypatch):
    # vLLM ignores suppress_tokens and samples every config.vocab_size row.
    import ast
    import copy
    import functools
    import pathlib
    import types

    import unsloth_zoo
    path = pathlib.Path(unsloth_zoo.__file__).parent / "vllm_utils.py"
    wanted = {"_padded_vocab_ids", "suppress_vllm_padded_vocab"}
    nodes = [n for n in ast.parse(path.read_text(encoding = "utf-8-sig")).body
             if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    ns = {"copy": copy, "functools": functools}
    exec(compile(ast.Module(body = nodes, type_ignores = []), "vllm_utils_helpers", "exec"), ns)

    model, tokenizer = _load()
    assert ns["_padded_vocab_ids"](model.config) == []
    add_new_tokens(model, tokenizer, new_tokens = ["<unsloth_a>"], pad_to_multiple_of = 64)
    ids = ns["_padded_vocab_ids"](model.config)
    assert ids == list(range(len(tokenizer), model.config.vocab_size))

    class Params:
        def __init__(self, logit_bias = None):
            self.logit_bias = logit_bias

    seen = []

    class LLM:
        def get_default_sampling_params(self):
            return Params()

        def generate(self, prompts, sampling_params = None, use_tqdm = True):
            seen.append(sampling_params)
            return ["out"]

    llm = LLM()
    ns["suppress_vllm_padded_vocab"](llm, ids)
    ns["suppress_vllm_padded_vocab"](llm, ids)
    mine = Params({5: 2.0})
    assert llm.generate(["a"], mine, use_tqdm = False) == ["out"]
    llm.generate(["a"])
    llm.generate(["a", "b"], sampling_params = [Params(), Params({ids[0]: 3.0})])
    want = {i: -100.0 for i in ids}
    assert seen[0].logit_bias == {5: 2.0, **want} and mine.logit_bias == {5: 2.0}
    assert seen[1].logit_bias == want
    assert [p.logit_bias for p in seen[2]] == [want, want]
    assert len(seen) == 3
