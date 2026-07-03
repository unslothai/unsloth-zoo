"""Tests for the prequantized per-expert bnb-4bit MoE loading converters.

tf4-era unsloth-bnb-4bit checkpoints store MoE experts as one quantized tensor per
expert per projection (`experts.<N>.gate_proj.weight` + absmax/quant_map/quant_state
aux keys). Under transformers v5 the model's MergeModulelist converters match those
keys first, byte-concat packed uint8 as if bf16, and drop every aux key, so
`_bnb4bit_per_expert_conversions` builds prepended "twin" converters that collect the
aux keys and reassemble one stacked Params4bit per projection.

These tests drive the twin-construction and routing logic on CPU with synthetic
converters; the full checkpoint path is exercised by the model loading integration.
"""

from __future__ import annotations

import types

import pytest
import torch

pytest.importorskip(
    "transformers.core_model_loading",
    reason="requires transformers v5 core_model_loading",
)

from transformers.core_model_loading import ConversionOps, WeightConverter

from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import (
    _AUX_SUFFIXES,
    _bnb4bit_per_expert_conversions,
    _quantstate_absmax_fp32,
)


class _StubMerge(ConversionOps):
    """Stands in for the model's MergeModulelist op: stacks collected experts."""

    def __init__(self):
        self.calls = []

    def convert(self, input_dict, **kwargs):
        self.calls.append((dict(input_dict), kwargs))
        merged = torch.stack([v for vs in input_dict.values() for v in (vs if isinstance(vs, list) else [vs])])
        target = kwargs.get("target_patterns")
        key = target[0] if isinstance(target, (list, tuple)) else target
        return {key: merged}


def _expert_merge_converter(target="model.layers.*.mlp.experts.gate_up_proj"):
    return WeightConverter(
        source_patterns=[
            "model.layers.*.mlp.experts.*.gate_proj.weight",
            "model.layers.*.mlp.experts.*.up_proj.weight",
        ],
        target_patterns=target,
        operations=[_StubMerge()],
    )


def test_twin_built_for_per_expert_expert_converter():
    conv = _expert_merge_converter()
    twins = _bnb4bit_per_expert_conversions([conv], hf_quantizer=None)
    assert len(twins) == 1
    twin = twins[0]

    # Aux keys come first so they are collected alongside the weights.
    n_base = len(conv.source_patterns)
    expected_aux = [b + suf for b in conv.source_patterns for suf in _AUX_SUFFIXES]
    assert list(twin.source_patterns[: len(expected_aux)]) == expected_aux

    # Bare weight patterns are `$`-anchored so pattern_to_converter (last-wins)
    # does not let the twin shadow the model's own converter for bf16 checkpoints.
    anchored = list(twin.source_patterns[len(expected_aux):])
    assert anchored == [s + "$" for s in conv.source_patterns]
    assert twin.target_patterns[0] == conv.target_patterns[0]
    assert type(twin.operations[0]).__name__ == "_PerExpertStackDeserialize"


def test_non_expert_converters_ignored():
    attn = WeightConverter(
        source_patterns=["model.layers.*.self_attn.q_proj.weight"],
        target_patterns="model.layers.*.self_attn.q_proj.weight",
        operations=[_StubMerge()],
    )
    non_weight = WeightConverter(
        source_patterns=["model.layers.*.mlp.experts.*.gate_proj.bias"],
        target_patterns="model.layers.*.mlp.experts.gate_up_proj",
        operations=[_StubMerge()],
    )
    assert _bnb4bit_per_expert_conversions([attn, non_weight], hf_quantizer=None) == []


def test_down_proj_converter_also_twinned():
    conv = WeightConverter(
        source_patterns=["model.layers.*.mlp.experts.*.down_proj.weight"],
        target_patterns="model.layers.*.mlp.experts.down_proj",
        operations=[_StubMerge()],
    )
    twins = _bnb4bit_per_expert_conversions([conv], hf_quantizer=None)
    assert len(twins) == 1
    assert twins[0].target_patterns[0].endswith("down_proj")


def test_unquantized_layer_falls_back_to_original_merge():
    """Layers whose experts were skipped by dynamic quantization carry no aux keys;
    the twin must reproduce the model's own merge for them."""
    conv = _expert_merge_converter()
    twin = _bnb4bit_per_expert_conversions([conv], hf_quantizer=None)[0]
    op = twin.operations[0]

    n_experts = 3
    input_dict = {
        anchored: [torch.full((4, 2), float(e)) for e in range(n_experts)]
        for anchored in op.anchored_sources
    }
    out = op.convert(
        dict(input_dict),
        model=None,
        full_layer_name="model.layers.0.mlp.experts.gate_up_proj",
        target_patterns=[conv.target_patterns[0]],
    )
    key = conv.target_patterns[0]
    assert key in out
    assert out[key].shape[0] == n_experts * len(op.base_sources)


def test_quantstate_absmax_fp32_plain():
    qs = types.SimpleNamespace(nested=False, absmax=torch.tensor([1.0, 2.5], dtype=torch.bfloat16))
    out = _quantstate_absmax_fp32(qs)
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.tensor([1.0, 2.5]))
