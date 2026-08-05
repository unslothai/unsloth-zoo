# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Regression tests for ``temporary_patches/qwen3_5_float32.py``.

The dtype mismatch reported in unsloth#7506 (``BFloat16 != Half``
RuntimeError inside Qwen3.5 linear layers) is reproduced on CPU by
keeping fp16 weights but feeding them bf16 activations. The temporary
patch normalizes activations to the actual weight dtype under
``UNSLOTH_FORCE_FLOAT32=1``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")
pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")
pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

import torch  # noqa: E402
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen  # noqa: E402

from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES  # noqa: E402
from unsloth_zoo.temporary_patches.qwen3_5_float32 import (  # noqa: E402
    _unsloth_is_default_causal_lm_loss,
)


def _apply_temporary_patches():
    """Apply all registered temporary patches (they gate on environment vars)."""
    for patch_fn in TEMPORARY_PATCHES:
        patch_fn()


def _tiny_text_config(layer_types=("linear_attention", "full_attention")):
    return Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=128,
        rope_theta=10000.0,
        head_dim=16,
        attention_dropout=0.0,
        rms_norm_eps=1e-06,
        layer_types=list(layer_types),
        tie_word_embeddings=False,
        partial_rotary_factor=0.25,
    )


def test_qwen3_5_mlp_dtype_mismatch_fixed(monkeypatch):
    """``Qwen3_5MLP`` must accept bf16 activations when weights are fp16."""
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    _apply_temporary_patches()

    config = _tiny_text_config()
    mlp = qwen.Qwen3_5MLP(config, config.intermediate_size).to(torch.float16)
    x = torch.randn(2, 4, config.hidden_size, dtype=torch.bfloat16)

    out = mlp(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_qwen3_5_attention_dtype_mismatch_fixed(monkeypatch):
    """``Qwen3_5Attention`` must accept bf16 hidden states and RoPE when weights are fp16."""
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    _apply_temporary_patches()

    config = _tiny_text_config()
    attn = qwen.Qwen3_5Attention(config, layer_idx=0).to(torch.float16)
    rotary = qwen.Qwen3_5TextRotaryEmbedding(config)

    hidden = torch.randn(2, 4, config.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(4).unsqueeze(0).expand(2, -1)
    cos, sin = rotary(hidden, position_ids)

    out, _ = attn(
        hidden,
        position_embeddings=(cos, sin),
        position_ids=position_ids,
        use_cache=False,
    )
    assert out.shape == hidden.shape
    assert out.dtype == hidden.dtype


def test_qwen3_5_for_causal_lm_dtype_mismatch_fixed(monkeypatch):
    """``Qwen3_5ForCausalLM`` must complete forward passes with fp16 weights."""
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    _apply_temporary_patches()

    config = _tiny_text_config(layer_types=("full_attention", "full_attention"))
    model = qwen.Qwen3_5ForCausalLM(config).to(torch.float16)
    input_ids = torch.randint(0, config.vocab_size, (2, 4))

    outputs = model(input_ids, use_cache=False)
    assert outputs.logits.shape == (2, 4, config.vocab_size)

    # The wrapper must preserve the standard Transformers tuple-output contract.
    tuple_outputs = model(input_ids, use_cache=False, return_dict=False)
    assert isinstance(tuple_outputs, tuple)
    assert tuple_outputs[0].shape == (2, 4, config.vocab_size)


def test_qwen3_5_for_causal_lm_respects_output_hidden_states_config(monkeypatch):
    """`config.output_hidden_states=True` must propagate even without the kwarg."""
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    _apply_temporary_patches()

    config = _tiny_text_config(layer_types=("full_attention", "full_attention"))
    config.output_hidden_states = True
    model = qwen.Qwen3_5ForCausalLM(config).to(torch.float16)
    input_ids = torch.randint(0, config.vocab_size, (2, 4))

    outputs = model(input_ids, use_cache=False)
    assert outputs.logits.shape == (2, 4, config.vocab_size)
    assert outputs.hidden_states is not None


@pytest.mark.parametrize("name", ["ForCausalLMLoss", "UnslothForCausalLMLoss"])
def test_default_causal_lm_loss_accepted(name):
    """The default (and Unsloth-patched) loss names must take the fused path."""
    fake_loss = type("_FakeLoss", (), {"__name__": name})()
    assert _unsloth_is_default_causal_lm_loss(fake_loss) is True


def test_custom_loss_rejected():
    """Custom losses must fall back to the logits + loss_function branch."""
    fake_loss = type("_FakeLoss", (), {"__name__": "CustomFocalLoss"})()
    assert _unsloth_is_default_causal_lm_loss(fake_loss) is False
