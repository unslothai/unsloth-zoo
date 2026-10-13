# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DeepSpeed ZeRO-3 init partitions every weight at load into a 1-D empty placeholder, the real
shape in `ds_shape` (unsloth#225). Rebuilding the embedding / lm_head around it asserted 2-D,
and the untrained-token check reduced over a dimension it no longer has."""

import contextlib
import enum
import sys
import types

import pytest
import torch

transformers = pytest.importorskip("transformers")

from unsloth_zoo import tokenizer_utils
from unsloth_zoo.patching_utils import patch_model_and_tokenizer


class _Status(enum.Enum):
    AVAILABLE = 1
    NOT_AVAILABLE = 2


def _partition(param):
    """What deepspeed.zero.Init leaves behind: the full tensor kept aside, data emptied."""
    param.ds_shape = param.shape
    param._full = param.data.clone()
    param.data = torch.empty(0, dtype = param.dtype)
    param.ds_status = _Status.NOT_AVAILABLE


def _tiny_llama():
    config = transformers.LlamaConfig(
        vocab_size = 8, hidden_size = 4, intermediate_size = 8, num_hidden_layers = 1,
        num_attention_heads = 1, num_key_value_heads = 1, tie_word_embeddings = False,
    )
    return transformers.LlamaForCausalLM(config)


class _Tokenizer:
    chat_template = "<|extra_0|> in the template"

    def __len__(self):
        return 12

    def convert_ids_to_tokens(self, ids):
        return [f"<|extra_{i}|>" for i in range(len(ids))]


@pytest.mark.parametrize("tokenizer_len, resized", [(12, [12]), (8, [])])
def test_patch_keeps_partitioned_embedding_modules(monkeypatch, tokenizer_len, resized):
    model = _tiny_llama()
    embed, head = model.model.embed_tokens, model.lm_head
    _partition(embed.weight)
    _partition(head.weight)
    # transformers' resize_token_embeddings gathers and re-partitions under ZeRO-3; record the call.
    calls = []
    monkeypatch.setattr(model, "resize_token_embeddings", lambda n: calls.append(n))
    tokenizer = _Tokenizer()
    monkeypatch.setattr(_Tokenizer, "__len__", lambda self: tokenizer_len)

    patch_model_and_tokenizer(model, tokenizer, downcast_rope = False)

    # Sized from ds_shape, not the empty placeholder, and only when the tokenizer is longer.
    assert calls == resized
    assert model.model.embed_tokens is embed and model.lm_head is head
    assert embed.weight.shape == (0,) and head.weight.shape == (0,)


@pytest.fixture
def fake_gather(monkeypatch):
    calls = []

    @contextlib.contextmanager
    def GatheredParameters(params, modifier_rank = None):
        assert not torch.is_inference_mode_enabled(), "all-gather must write param.data outside inference mode"
        calls.append((len(params), modifier_rank))
        for p in params:
            p.data = p._full
            p.ds_status = _Status.AVAILABLE
        yield
        for p in params:
            p._full = p.data
            p.data = torch.empty(0, dtype = p.dtype)
            p.ds_status = _Status.NOT_AVAILABLE

    deepspeed = types.ModuleType("deepspeed")
    deepspeed.zero = types.SimpleNamespace(GatheredParameters = GatheredParameters)
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed)
    return calls


def test_fix_untrained_tokens_gathers_partitioned_weights(fake_gather):
    model = _tiny_llama()
    with torch.no_grad():
        for weight in (model.model.embed_tokens.weight, model.lm_head.weight):
            weight.fill_(1.0)
            weight[4].zero_()
            weight[5].zero_()
    _partition(model.model.embed_tokens.weight)
    _partition(model.lm_head.weight)

    tokenizer_utils.fix_untrained_tokens(
        model, _Tokenizer(), [{"input_ids" : [1, 2, 4]}, {"input_ids" : [2, 3, 5]}],
    )

    # One gather of both matrices, rank 0's reset broadcast, placeholders restored.
    assert fake_gather == [(2, 0)]
    for weight in (model.model.embed_tokens.weight, model.lm_head.weight):
        assert weight.shape == (0,)
        full = weight._full
        assert torch.all(full[:4] == 1.0)
        assert not torch.equal(full[4], torch.zeros(4))
        assert not torch.equal(full[5], torch.zeros(4))


def test_fix_untrained_tokens_does_not_gather_unpartitioned_weights(fake_gather):
    model = _tiny_llama()
    tokenizer_utils.fix_untrained_tokens(model, _Tokenizer(), [{"input_ids" : [1]}])
    assert fake_gather == []


@pytest.fixture
def live_zero3(monkeypatch):
    """Make transformers report a live ZeRO-3 config (or none) the way TrainingArguments does."""
    import weakref
    from transformers.integrations import deepspeed as hf_deepspeed

    class _Config:
        def __init__(self, zero3): self.zero3 = zero3
        def is_zero3(self): return self.zero3

    held = []

    def set_state(zero3):
        if zero3 is None:
            monkeypatch.setattr(hf_deepspeed, "_hf_deepspeed_config_weak_ref", None)
            return
        held.append(_Config(zero3))
        monkeypatch.setattr(hf_deepspeed, "_hf_deepspeed_config_weak_ref", weakref.ref(held[-1]))

    return set_state


@pytest.mark.parametrize("factory", ["_eager_during_decode", "eager_under_zero3"])
def test_compiled_regions_run_eager_under_zero3(live_zero3, factory):
    """Inside torch.compile ZeRO-3's parameter getter returns the empty placeholder, so Unsloth's
    compiled regions (generated modules, patch_function, temporary patches) run eager under it."""
    from unsloth_zoo.temporary_patches import utils

    calls = []
    dispatch = getattr(utils, factory)(lambda: calls.append("eager"), lambda: calls.append("compiled"))
    for state in (None, False, True, None):
        live_zero3(state)
        dispatch()
    assert calls == ["compiled", "compiled", "eager", "compiled"]
    assert dispatch._unsloth_compiled_func is not None


def test_patch_function_compiled_patch_runs_eager_under_zero3(live_zero3):
    from unsloth_zoo.temporary_patches.utils import patch_function

    class _Target:
        def forward(self, x):
            return x

    traced = []

    def forward(self, x):
        # Runs at trace time under torch.compile (True), on every call when eager (False).
        traced.append(torch.compiler.is_compiling())
        return x + 1

    patch_function(_Target, "forward", forward, fullgraph = True)
    live_zero3(True)
    for _ in range(2):
        assert _Target.forward(_Target(), torch.tensor(1.0)).item() == 2.0
    assert traced == [False, False], traced
