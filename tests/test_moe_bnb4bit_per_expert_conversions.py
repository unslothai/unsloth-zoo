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


class _StubTranspose(ConversionOps):
    """Stands in for the model's Transpose op on fused expert params."""

    def __init__(self):
        self.calls = 0

    def convert(self, input_dict, **kwargs):
        self.calls += 1
        return {k: (v[0] if isinstance(v, list) else v).transpose(1, 2) for k, v in input_dict.items()}


def _fused_passthrough_converter(name="model.layers.*.mlp.experts.gate_up_proj"):
    return WeightConverter(
        source_patterns=name,
        target_patterns=name,
        operations=[_StubTranspose()],
    )


def test_fused_passthrough_converter_twinned():
    """Model converters over a fused expert param (e.g. qwen3_vl_moe's Transpose)
    precede appended quantizer conversions, so they need a prepended twin too."""
    conv = _fused_passthrough_converter()
    twins = _bnb4bit_per_expert_conversions([conv], hf_quantizer=None)
    assert len(twins) == 1
    twin = twins[0]
    base = conv.source_patterns[0]
    assert list(twin.source_patterns) == [base + suf for suf in _AUX_SUFFIXES] + [base + "$"]
    assert twin.target_patterns[0] == conv.target_patterns[0]
    assert type(twin.operations[0]).__name__ == "_FusedExpertDeserialize"


def test_fused_unquantized_falls_back_to_original_ops():
    conv = _fused_passthrough_converter()
    twin = _bnb4bit_per_expert_conversions([conv], hf_quantizer=None)[0]
    op = twin.operations[0]
    orig = op.original_ops[0]

    fused = torch.randn(3, 4, 2)
    out = op.convert(
        {op.anchored_source: fused},
        model=None,
        full_layer_name="model.layers.0.mlp.experts.gate_up_proj",
        target_patterns=[conv.target_patterns[0]],
    )
    assert orig.calls == 1
    assert out[op.base_source].shape == (3, 2, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for bnb 4-bit")
def test_fused_prequantized_builds_params4bit():
    import bitsandbytes as bnb
    import torch.nn as nn

    conv = _fused_passthrough_converter()
    twin = _bnb4bit_per_expert_conversions([conv], hf_quantizer=None)[0]
    op = twin.operations[0]

    w = torch.randn(2, 8, 16, device="cuda", dtype=torch.bfloat16)
    packed, qs = bnb.functional.quantize_4bit(w, quant_type="nf4")
    base = op.base_source
    input_dict = {base + "." + k: v for k, v in qs.as_dict(packed=True).items()}
    input_dict[op.anchored_source] = packed

    experts = nn.Module()
    experts.gate_up_proj = nn.Parameter(torch.empty(1), requires_grad=False)
    mlp = nn.Module(); mlp.experts = experts
    layer = nn.Module(); layer.mlp = mlp
    layers = nn.ModuleList([layer])
    inner = nn.Module(); inner.layers = layers
    model = nn.Module(); model.model = inner

    out = op.convert(
        input_dict,
        model=model,
        full_layer_name="model.layers.0.mlp.experts.gate_up_proj",
        target_patterns=[conv.target_patterns[0]],
    )
    new_param = out[conv.target_patterns[0]]
    assert type(new_param).__name__ == "Params4bit"
    assert tuple(new_param._original_shape) == (2, 8, 16)
    deq = bnb.functional.dequantize_4bit(new_param.data, new_param.quant_state)
    assert torch.allclose(deq.float(), w.float(), atol=0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for bnb 4-bit")
def test_fused_prequantized_orientation_mismatch_raises():
    """A blob quantized in the transposed (checkpoint) layout must fail loudly:
    a transpose op cannot be applied to packed 4-bit data."""
    import bitsandbytes as bnb
    import torch.nn as nn

    conv = _fused_passthrough_converter()
    twin = _bnb4bit_per_expert_conversions([conv], hf_quantizer=None)[0]
    op = twin.operations[0]

    w = torch.randn(2, 16, 8, device="cuda", dtype=torch.bfloat16)  # transposed layout
    packed, qs = bnb.functional.quantize_4bit(w, quant_type="nf4")
    base = op.base_source
    input_dict = {base + "." + k: v for k, v in qs.as_dict(packed=True).items()}
    input_dict[op.anchored_source] = packed

    experts = nn.Module()
    # Real-rank meta param carrying the model layout.
    experts.gate_up_proj = nn.Parameter(torch.empty(2, 8, 16), requires_grad=False)
    mlp = nn.Module(); mlp.experts = experts
    layer = nn.Module(); layer.mlp = mlp
    layers = nn.ModuleList([layer])
    inner = nn.Module(); inner.layers = layers
    model = nn.Module(); model.model = inner

    with pytest.raises(ValueError, match="model layout"):
        op.convert(
            input_dict,
            model=model,
            full_layer_name="model.layers.0.mlp.experts.gate_up_proj",
            target_patterns=[conv.target_patterns[0]],
        )


def _stack_input_dict(op, weights_per_src):
    """Build the per-expert list input_dict for _PerExpertStackDeserialize.

    weights_per_src: [src][expert] float weight tensors (out, in) on CUDA.
    """
    import bitsandbytes as bnb

    input_dict = {}
    for (base, anch), expert_ws in zip(
        zip(op.base_sources, op.anchored_sources), weights_per_src
    ):
        packed_list = []
        aux_lists = {}
        for w in expert_ws:
            packed, qs = bnb.functional.quantize_4bit(w, quant_type="nf4")
            packed_list.append(packed)
            for k, v in qs.as_dict(packed=True).items():
                aux_lists.setdefault("." + k, []).append(v)
        input_dict[anch] = packed_list
        for suf, vals in aux_lists.items():
            input_dict[base + suf] = vals
    return input_dict


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for bnb 4-bit")
@pytest.mark.parametrize(
    "out_dim,in_dim,exact",
    [
        (8, 16, True),   # 128 elems per segment: whole 64-blocks, raw concat is exact
        (6, 10, False),  # 60 elems: partial block, must take the repack path
    ],
)
def test_stack_prequantized_block_alignment(out_dim, in_dim, exact):
    import bitsandbytes as bnb

    torch.manual_seed(0)
    conv = _expert_merge_converter()
    twin = _bnb4bit_per_expert_conversions([conv], hf_quantizer=None)[0]
    op = twin.operations[0]

    n_experts = 2
    weights_per_src = [
        [torch.randn(out_dim, in_dim, device="cuda", dtype=torch.bfloat16) for _ in range(n_experts)]
        for _ in range(len(op.base_sources))
    ]
    input_dict = _stack_input_dict(op, weights_per_src)

    from transformers.quantizers.quantizers_utils import get_module_from_name  # noqa: F401
    import torch.nn as nn
    experts = nn.Module()
    experts.gate_up_proj = nn.Parameter(torch.empty(1), requires_grad=False)
    mlp = nn.Module(); mlp.experts = experts
    layer = nn.Module(); layer.mlp = mlp
    inner = nn.Module(); inner.layers = nn.ModuleList([layer])
    model = nn.Module(); model.model = inner

    out = op.convert(
        input_dict,
        model=model,
        full_layer_name="model.layers.0.mlp.experts.gate_up_proj",
        target_patterns=[conv.target_patterns[0]],
    )
    new_param = out[conv.target_patterns[0]]
    # Reference: what the per-expert quantized segments actually dequantize to.
    seg_ref = torch.stack([
        torch.cat([
            bnb.functional.dequantize_4bit(
                *bnb.functional.quantize_4bit(weights_per_src[s][e], quant_type="nf4")
            ).float()
            for s in range(len(weights_per_src))
        ], dim=0)
        for e in range(n_experts)
    ])
    assert tuple(new_param._original_shape) == tuple(seg_ref.shape)
    deq = bnb.functional.dequantize_4bit(new_param.data, new_param.quant_state).float()
    deq = deq.reshape(seg_ref.shape)
    err = (deq - seg_ref).abs().max().item()
    if exact:
        # Same bytes and absmax as the per-expert quantization: dequant is identical.
        assert err == 0.0, err
    else:
        # Repack path adds one requant of already-quantized values (~0.16 here);
        # raw concat instead mis-scales across the partial block boundary.
        assert err < 0.25, err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for bnb 4-bit")
def test_plain_dequant_skips_bnb_embedding4bit():
    import bitsandbytes as bnb
    import torch.nn as nn

    from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import (
        _dequantize_plain_param_slots,
    )

    class Holder(nn.Module):
        pass

    model = Holder()
    model.emb = bnb.nn.Embedding4bit(16, 64)
    model.emb = model.emb.cuda()  # quantizes weight into Params4bit
    model.plain = Holder()
    packed, qs = bnb.functional.quantize_4bit(
        torch.randn(4, 64, device="cuda", dtype=torch.bfloat16), quant_type="nf4"
    )
    from bitsandbytes.nn import Params4bit
    router = torch.Tensor._make_subclass(Params4bit, packed)
    router.requires_grad = False
    router.quant_state = qs
    model.plain.weight = router

    _dequantize_plain_param_slots(model)

    # bnb module keeps its quantized weight; the genuinely plain slot converts.
    assert type(model.emb.weight).__name__ == "Params4bit"
    assert getattr(model.emb.weight, "quant_state", None) is not None
    assert type(model.plain.weight).__name__ == "Parameter"
    assert model.plain.weight.shape == (4, 64)
