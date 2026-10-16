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


def test_legacy_padded_model_releases_its_rows(monkeypatch, tmp_path):
    # Saved by padding code that suppressed the alignment rows but did not record them.
    monkeypatch.delenv("UNSLOTH_VOCAB_PAD_MULTIPLE", raising = False)
    model, tokenizer = _load()
    add_new_tokens(model, tokenizer, new_tokens = ["<unsloth_a>"], pad_to_multiple_of = 64)
    n1 = len(tokenizer)
    padded = model.get_input_embeddings().weight.shape[0]
    del model.config.unsloth_vocab_pad_ids
    del model.config.unsloth_vocab_pad_suppressed_ids
    model.save_pretrained(tmp_path / "m")
    tokenizer.save_pretrained(tmp_path / "m")
    model = AutoModelForCausalLM.from_pretrained(tmp_path / "m", torch_dtype = torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(tmp_path / "m")
    assert not hasattr(model.config, "unsloth_vocab_pad_ids")
    assert list(model.generation_config.suppress_tokens) == list(range(n1, padded))
    add_new_tokens(model, tokenizer, new_tokens = ["<unsloth_b>"])
    assert tokenizer.convert_tokens_to_ids("<unsloth_b>") == n1
    assert list(model.generation_config.suppress_tokens) == list(range(n1 + 1, padded))
    assert model.config.unsloth_vocab_pad_ids == list(range(n1 + 1, padded))


def test_alignment_rows_already_suppressed_are_still_recorded_for_vllm(monkeypatch):
    # vLLM only biases the recorded rows, so a row the caller already suppressed must be recorded
    # too; it stays the caller's suppression when it later becomes a token.
    monkeypatch.delenv("UNSLOTH_VOCAB_PAD_MULTIPLE", raising = False)
    model, tokenizer = _load()
    n0 = len(tokenizer)
    model.generation_config.suppress_tokens = [7, n0 + 1]
    add_new_tokens(model, tokenizer, new_tokens = ["<unsloth_a>"], pad_to_multiple_of = 64)
    n1 = len(tokenizer)
    padded = model.get_input_embeddings().weight.shape[0]
    assert n1 == n0 + 1 and n1 < padded
    assert model.config.unsloth_vocab_pad_ids == list(range(n1, padded))
    assert model.config.unsloth_vocab_pad_suppressed_ids == list(range(n1 + 1, padded))
    assert sorted(model.generation_config.suppress_tokens) == [7] + list(range(n1, padded))

    add_new_tokens(model, tokenizer, new_tokens = ["<unsloth_b>", "<unsloth_c>"])
    assert tokenizer.convert_tokens_to_ids("<unsloth_b>") == n1
    # n1 was the caller's: still suppressed. n1 + 1 was ours: released. Neither is an alignment row now.
    assert sorted(model.generation_config.suppress_tokens) == [7, n1] + list(range(n1 + 2, padded))
    assert model.config.unsloth_vocab_pad_ids == list(range(n1 + 2, padded))
    assert model.config.unsloth_vocab_pad_suppressed_ids == list(range(n1 + 2, padded))


def _vllm_wrapper_ns():
    import ast
    import copy
    import functools
    import inspect
    import pathlib

    import unsloth_zoo
    path = pathlib.Path(unsloth_zoo.__file__).parent / "vllm_utils.py"
    wanted = {"_padded_vocab_ids", "suppress_vllm_padded_vocab"}
    nodes = [n for n in ast.parse(path.read_text(encoding = "utf-8-sig")).body
             if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    ns = {"copy": copy, "functools": functools, "inspect": inspect}
    exec(compile(ast.Module(body = nodes, type_ignores = []), "vllm_utils_helpers", "exec"), ns)
    return ns


class _Params:
    def __init__(self, logit_bias = None, n = 1):
        self.logit_bias = logit_bias
        self.n = n


class _PoolingParams:
    pass


class _EngineCoreRequest:
    def __init__(self, request_id, params):
        self.request_id = request_id
        is_sampling = hasattr(params, "logit_bias")
        self.sampling_params = copy_params(params) if is_sampling else None
        self.pooling_params = None if is_sampling else params


def copy_params(params):
    import copy
    return copy.copy(params)


class _Client:
    def __init__(self):
        self.seen = []

    def add_request(self, request):
        self.seen.append(request)


class _AsyncClient:
    def __init__(self):
        self.seen = []

    async def add_request_async(self, request):
        self.seen.append(request)


class _LLMEngine:
    # vLLM 0.29 LLMEngine.add_request: process_inputs (clones params) or a prebuilt
    # EngineCoreRequest whose own params win, then n > 1 fan-out, then engine_core.add_request.
    def __init__(self):
        self.engine_core = _Client()

    def add_request(self, request_id, prompt, params):
        request = prompt if isinstance(prompt, _EngineCoreRequest) else _EngineCoreRequest(request_id, params)
        n = getattr(request.sampling_params, "n", 1)
        for i in range(n):
            child = request if n == 1 else _EngineCoreRequest(f"{i}_{request_id}", request.sampling_params)
            self.engine_core.add_request(child)
        return request_id


class _LLM:
    # vLLM 0.29 LLM: generate / chat / enqueue all end in llm_engine.add_request.
    def __init__(self):
        self.llm_engine = _LLMEngine()

    def _submit(self, prompts, params):
        params = params if isinstance(params, list) else [params or _Params()] * len(prompts)
        return [self.llm_engine.add_request(str(i), p, q) for i, (p, q) in enumerate(zip(prompts, params))]

    def generate(self, prompts, sampling_params = None):
        return self._submit(prompts, sampling_params)

    def chat(self, messages, sampling_params = None):
        return self._submit(messages, sampling_params)

    def enqueue(self, prompts, sampling_params = None):
        return self._submit(prompts, sampling_params)


IDS = list(range(2000, 2063))
WANT = {i: -100.0 for i in IDS}


def test_vllm_generate_biases_padded_rows():
    ns = _vllm_wrapper_ns()
    model, tokenizer = _load()
    assert ns["_padded_vocab_ids"](model.config) == []
    add_new_tokens(model, tokenizer, new_tokens = ["<unsloth_a>"], pad_to_multiple_of = 64)
    assert ns["_padded_vocab_ids"](model.config) == list(range(len(tokenizer), model.config.vocab_size))

    llm = _LLM()
    generate, add_request = llm.generate, llm.llm_engine.add_request
    ns["suppress_vllm_padded_vocab"](llm, IDS)
    ns["suppress_vllm_padded_vocab"](llm, IDS)
    assert llm.generate == generate and llm.llm_engine.add_request == add_request
    mine = _Params({5: 2.0})
    llm.generate(["a"], mine)
    llm.generate(["a", "b"], sampling_params = [_Params(), _Params({IDS[0]: 3.0})])
    llm.chat([["hi"]], mine)
    llm.enqueue(["a"], _Params(n = 2))
    seen = llm.llm_engine.engine_core.seen
    assert [r.sampling_params.logit_bias for r in seen] == [{5: 2.0, **WANT}, WANT, WANT, {5: 2.0, **WANT}, WANT, WANT]
    assert mine.logit_bias == {5: 2.0}


def test_vllm_padded_vocab_biases_prebuilt_engine_core_request():
    # LLMEngine.add_request(request_id, EngineCoreRequest, params) sends the request's own params.
    engine = _LLMEngine()
    _vllm_wrapper_ns()["suppress_vllm_padded_vocab"](engine, IDS)
    prebuilt = _EngineCoreRequest("r0", _Params({1: 1.0}))
    engine.add_request("r0", prebuilt, _Params())
    pooling = _PoolingParams()
    engine.add_request("r1", "p", pooling)
    seen = engine.engine_core.seen
    assert seen[0].sampling_params.logit_bias == {1: 1.0, **WANT}
    assert seen[1].sampling_params is None and seen[1].pooling_params is pooling


def test_vllm_padded_vocab_biases_async_streaming_chunks():
    # AsyncLLM streaming input: each chunk's own SamplingParams go through process_inputs and
    # straight to engine_core.add_request_async, never through AsyncLLM.add_request.
    import asyncio

    class AsyncLLM:
        def __init__(self):
            self.engine_core = _AsyncClient()

        async def stream(self, chunks):
            for i, params in enumerate(chunks):
                await self.engine_core.add_request_async(_EngineCoreRequest("s", params))

    llm = AsyncLLM()
    _vllm_wrapper_ns()["suppress_vllm_padded_vocab"](llm, IDS)
    asyncio.run(llm.stream([_Params(), _Params({9: 1.0})]))
    assert [r.sampling_params.logit_bias for r in llm.engine_core.seen] == [WANT, {9: 1.0, **WANT}]


def test_vllm_padded_vocab_not_biased_under_speculative_decoding(capsys):
    # vLLM 0.29 SamplingParams._validate_spec_decode raises on any logit_bias with speculation.
    llm = _LLM()
    add_request = llm.llm_engine.engine_core.add_request
    _vllm_wrapper_ns()["suppress_vllm_padded_vocab"](llm, IDS, speculative = True)
    assert llm.llm_engine.engine_core.add_request == add_request
    llm.generate(["a"], _Params())
    assert llm.llm_engine.engine_core.seen[0].sampling_params.logit_bias is None
    assert "speculative decoding" in capsys.readouterr().out


def test_vllm_padded_vocab_bias_respects_the_logit_bias_limit(capsys):
    # vLLM 0.29 Model Runner V2 raises past MAX_NUM_LOGIT_BIAS_TOKENS (1024) entries per request.
    wrap = _vllm_wrapper_ns()["suppress_vllm_padded_vocab"]
    big = _LLMEngine()
    wrap(big, list(range(2000, 3100)))  # pad_to_multiple_of = 2048 territory
    big.add_request("r0", "p", _Params())
    big.add_request("r1", "p", _Params())
    assert [r.sampling_params.logit_bias for r in big.engine_core.seen] == [None, None]

    small = _LLMEngine()
    wrap(small, IDS)
    small.add_request("r2", "p", _Params({i: 1.0 for i in range(1000)}))
    small.add_request("r3", "p", _Params({i: 1.0 for i in range(900)}))
    assert [len(r.sampling_params.logit_bias) for r in small.engine_core.seen] == [1000, 963]
    assert capsys.readouterr().out.count("exceed vLLM's limit of 1024") == 2  # once per engine
