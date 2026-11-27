# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A 16-bit fast_inference rebuild must not need bitsandbytes; only 4-bit quant_state weights import it."""

import sys

import pytest
import torch

transformers = pytest.importorskip("transformers")


@pytest.fixture
def vu():
    try:
        from unsloth_zoo import vllm_utils
    except ImportError as e:
        pytest.skip(f"vllm_utils does not import here: {e}")
    return vllm_utils


def _tiny_state():
    config = transformers.LlamaConfig(
        vocab_size = 64, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 64,
        tie_word_embeddings = False,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).to(torch.float16)
    return config, {k: v.detach().clone() for k, v in model.state_dict().items()}


def _block_bitsandbytes(monkeypatch):
    for name in list(sys.modules):
        if name == "bitsandbytes" or name.startswith("bitsandbytes."):
            monkeypatch.delitem(sys.modules, name)
    for name in ("bitsandbytes", "bitsandbytes.nn", "bitsandbytes.nn.modules"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError):
        from bitsandbytes.nn.modules import Linear4bit  # noqa: F401


def test_16bit_convert_runs_without_bitsandbytes(vu, monkeypatch):
    monkeypatch.setattr(vu, "get_target_device", lambda *a, **k: torch.device("cpu"))
    config, state = _tiny_state()
    _block_bitsandbytes(monkeypatch)
    new_model = vu.convert_vllm_to_huggingface(state, config, torch.float16)
    got = new_model.state_dict()
    for key in ("model.layers.0.self_attn.q_proj.weight", "model.layers.1.mlp.down_proj.weight", "lm_head.weight"):
        assert torch.equal(got[key].cpu(), state[key]), key
    assert not any(type(m).__name__ == "Linear4bit" for m in new_model.modules())


def test_4bit_weights_still_import_bitsandbytes(vu, monkeypatch):
    # The 4-bit branch keeps needing bitsandbytes: with it blocked the error must surface, not be swallowed.
    monkeypatch.setattr(vu, "get_target_device", lambda *a, **k: torch.device("cpu"))
    config, state = _tiny_state()
    state["model.layers.0.self_attn.q_proj.weight.quant_state"] = object()
    _block_bitsandbytes(monkeypatch)
    with pytest.raises(ImportError):
        vu.convert_vllm_to_huggingface(state, config, torch.float16)
