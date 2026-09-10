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
    _unsloth_cast_position_embeddings,
    _unsloth_weight_dtype,
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


def test_qwen3_5_gated_delta_net_dtype_mismatch_fixed(monkeypatch):
    """``Qwen3_5GatedDeltaNet`` must accept bf16 activations when weights are fp16."""
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    _apply_temporary_patches()

    config = _tiny_text_config(layer_types=("linear_attention",))
    block = qwen.Qwen3_5GatedDeltaNet(config, layer_idx=0).to(torch.float16)
    x = torch.randn(2, 4, config.hidden_size, dtype=torch.bfloat16)

    out = block(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_unsloth_weight_dtype_skips_quantized_and_missing():
    """`_unsloth_weight_dtype` must refuse quantized or absent weights."""
    assert _unsloth_weight_dtype(None) is None

    linear = torch.nn.Linear(4, 4)
    linear.weight = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float32))
    assert _unsloth_weight_dtype(linear) is torch.float32

    # Fake quantized weight
    param = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float16))
    param.quant_state = object()
    linear.weight = param
    assert _unsloth_weight_dtype(linear) is None


def test_unsloth_cast_position_embeddings():
    """`_unsloth_cast_position_embeddings` casts cos/sin to the target dtype."""
    cos = torch.randn(1, 8, 16, dtype=torch.float32)
    sin = torch.randn(1, 8, 16, dtype=torch.float32)
    out = _unsloth_cast_position_embeddings((cos, sin), torch.float16)
    assert out[0].dtype is torch.float16
    assert out[1].dtype is torch.float16

    assert _unsloth_cast_position_embeddings(None, torch.float16) is None
