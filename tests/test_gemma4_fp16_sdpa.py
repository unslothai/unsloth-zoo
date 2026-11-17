# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Forced float32 Gemma-4 attention on CUDA runs SDPA on float16 q / k / v, within float16 tolerance of float32."""

import pytest
import torch

from unsloth_zoo.temporary_patches import gemma4_banded_attention as gba
from unsloth_zoo.temporary_patches import gemma4_float32 as patches
from unsloth_zoo.temporary_patches import utils as patch_utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the float16 SDPA path is CUDA only")

S, HIDDEN, HEAD_DIM = 48, 64, 32


@pytest.fixture
def gemma4(monkeypatch):
    module = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    monkeypatch.setattr(patch_utils, "UNSLOTH_COMPILE_DISABLE", True)
    original = module.Gemma4TextAttention.forward
    monkeypatch.setattr(module.Gemma4TextAttention, "forward", original)
    return module, original


def _attention(module, layer, dtype, head_dim=HEAD_DIM):
    config = module.Gemma4TextConfig(
        vocab_size=32, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=head_dim, global_head_dim=head_dim,
        num_global_key_value_heads=2, layer_types=["sliding_attention", "full_attention"],
        sliding_window=16, num_kv_shared_layers=0,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return module.Gemma4TextAttention(config, layer).to("cuda", dtype).train()


def _mask(lengths, w):
    seg = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
    i = torch.arange(sum(lengths))
    m = (seg[:, None] == seg[None, :]) & (i[None, :] <= i[:, None])
    if w:
        m &= i[None, :] > i[:, None] - w
    return m[None, None].cuda()


def _run(module, attention, inputs, layer, lengths, head_dim=HEAD_DIM, packed=True):
    angles = torch.randn(1, S, head_dim, generator=torch.Generator().manual_seed(1))
    dtype = torch.float32 if inputs.dtype == torch.float16 else inputs.dtype  # the forced path upcasts RoPE itself
    position_embeddings = angles.cos().to("cuda", dtype), angles.sin().to("cuda", dtype)
    kwargs = {} if lengths is None or not packed else {"packed_seq_lengths": torch.tensor(lengths)}
    out, _ = attention(inputs, position_embeddings, _mask(lengths or [S], 16 if layer == 0 else None),
                       shared_kv_states={}, **kwargs)
    params = [p for p in attention.parameters() if p.requires_grad]
    grads = torch.autograd.grad(out.to(dtype).square().mean(), [inputs] + params)
    return out, grads


def _rel_err(x, ref):
    return ((x.double() - ref).norm() / ref.norm()).item()


def _capture(monkeypatch):
    seen = []
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def capture(q, k, v, **kwargs):
        seen.append(q.dtype)
        return sdpa(q, k, v, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", capture)
    return seen


# Head dim 512 is Gemma-4's global layers. Each path is measured against a float64 forward on its own weights.
# Float16 SDPA operands stay well below the error of the upstream bfloat16 forward on the same inputs (what every
# bf16 GPU runs; B200: 0.07x to 0.16x for the output and every grad) and within 2x the float32 operands' error
# (1.1x to 1.6x). A fixed tolerance against the float32 path does not hold at 512: there both paths differ by
# about the float16 rounding of the output they share.
@pytest.mark.parametrize("head_dim", [32, 512])
@pytest.mark.parametrize("lengths", [None, [16, 16, 8, 8]])
@pytest.mark.parametrize("layer", [0, 1])
def test_forced_float32_attention_runs_float16_sdpa_within_tolerance(gemma4, monkeypatch, layer, lengths, head_dim):
    gemma4, original = gemma4
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    seen = _capture(monkeypatch)
    inputs = torch.randn(1, S, HIDDEN, generator=torch.Generator().manual_seed(2)).to("cuda", torch.float16)
    results = {}
    for flag in ("0", "1"):  # float32 SDPA operands (the kill switch), then the default float16 operands
        monkeypatch.setenv("UNSLOTH_GEMMA4_FP16_SDPA", flag)
        gemma4.Gemma4TextAttention.forward = original  # the closure reads the switch at patch time
        patches.patch_Gemma4TextAttention()
        attention = _attention(gemma4, layer, torch.float16, head_dim)
        seen.clear()
        engaged = gba._PACKED_DOC_ENGAGED[0]
        results[flag] = _run(gemma4, attention, inputs.clone().requires_grad_(), layer, lengths, head_dim)
        assert seen and set(seen) == {torch.float16 if flag == "1" else torch.float32}
        assert gba._PACKED_DOC_ENGAGED[0] == engaged + (lengths is not None)
    gemma4.Gemma4TextAttention.forward = original

    def against_float64(dtype, outs=None):
        exact = _attention(gemma4, layer, dtype, head_dim).double()
        ref = _run(gemma4, exact, inputs.double().requires_grad_(), layer, lengths, head_dim, packed=False)
        if outs is None:  # the upstream forward in this dtype on the same inputs
            outs = _run(gemma4, _attention(gemma4, layer, dtype, head_dim),
                        inputs.to(dtype).requires_grad_(), layer, lengths, head_dim, packed=False)
        return [_rel_err(a, r) for a, r in zip([outs[0], *outs[1]], [ref[0], *ref[1]])]

    e16 = against_float64(torch.float16, results["1"])
    e32 = against_float64(torch.float16, results["0"])
    ebf16 = against_float64(torch.bfloat16)
    assert results["1"][0].dtype == torch.float16
    assert all(torch.isfinite(x).all() for x in [results["1"][0], *results["1"][1]])
    for i, (a, b, c) in enumerate(zip(e16, e32, ebf16)):
        assert a <= 0.5 * c and a <= 2 * b + 1e-5, (i, a, b, c)


def test_bfloat16_attention_is_not_patched(gemma4, monkeypatch):
    gemma4, original = gemma4
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "0")
    patches.patch_Gemma4TextAttention()
    assert gemma4.Gemma4TextAttention.forward is original
    seen = _capture(monkeypatch)
    attention = _attention(gemma4, 1, torch.bfloat16)
    _run(gemma4, attention, torch.randn(1, S, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True), 1, None)
    assert seen and set(seen) == {torch.bfloat16}


_MASKED_ROWS = [0, 5, 30]


def _mask_with_empty_rows(kind):
    allowed = torch.ones(S, S, dtype=torch.bool).tril()
    allowed[_MASKED_ROWS] = False  # fully masked query rows
    if kind == "bool":
        return allowed[None, None].cuda()
    fill = float("-inf") if kind == "-inf" else torch.finfo(torch.float32).min
    return torch.zeros(1, 1, S, S).masked_fill(~allowed, fill).cuda()


@pytest.mark.parametrize("layer", [0, 1])
def test_fully_masked_rows_agree_across_mask_types_and_the_float32_path(gemma4, monkeypatch, layer):
    gemma4, original = gemma4
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    angles = torch.randn(1, S, HEAD_DIM, generator=torch.Generator().manual_seed(1))
    position_embeddings = angles.cos().cuda(), angles.sin().cuda()
    inputs = torch.randn(1, S, HIDDEN, device="cuda", dtype=torch.float16)
    results = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("UNSLOTH_GEMMA4_FP16_SDPA", flag)
        gemma4.Gemma4TextAttention.forward = original
        patches.patch_Gemma4TextAttention()
        attention = _attention(gemma4, layer, torch.float16)
        params = [p for p in attention.parameters() if p.requires_grad]
        for kind in ("bool", "-inf", "float32 min"):
            x = inputs.clone().requires_grad_()
            out, _ = attention(x, position_embeddings, _mask_with_empty_rows(kind), shared_kv_states={})
            grads = torch.autograd.grad(out.float().square().mean(), [x] + params)
            results[flag, kind] = out, grads
    ref, ref_grads = results["0", "bool"]
    assert torch.equal(ref[0, _MASKED_ROWS], torch.zeros_like(ref[0, _MASKED_ROWS]))
    for key, (out, grads) in results.items():
        assert torch.equal(out[0, _MASKED_ROWS], ref[0, _MASKED_ROWS]), key
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-3, atol=2e-3, msg=str(key))
        for g, r in zip(grads, ref_grads):
            assert torch.isfinite(g).all(), key
            torch.testing.assert_close(g.float(), r.float(), rtol=1e-2, atol=1e-2 * r.float().abs().max().item(),
                                       msg=str(key))
