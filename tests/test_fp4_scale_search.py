# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
"""MXFP4 ``mse`` shared-exponent search and the magnitude-domain NVFP4 4over6 pick for routed-expert stacks (CPU)."""

import pytest
import torch

from unsloth_zoo.packed_experts import (
    NVFP4_GROUP,
    MX_GROUP,
    _E2M1,
    _pack_nibbles,
    _round_e2m1,
    mxfp4_weight_quant_method,
    nvfp4_weight_quant_method,
    quantize_experts_mxfp4,
    quantize_experts_nvfp4,
)

_LUT = torch.tensor(_E2M1)


def _weights(E = 3, N = 32, K = 256, seed = 0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(E, N, K, generator = g) * torch.logspace(-2, 0.5, K)
    w[:, 0, :3] *= 40
    return w


def _values(codes):
    v = _LUT[(codes & 7).long()]
    return torch.where((codes & 8).bool(), -v, v)


def _legacy_4over6(rows):
    """4over6 exactly as before the search existed (global = 256 * 6 / amax, /4 kept when strictly lower)."""
    E, N, K = rows.shape
    w = rows.to(torch.float32)
    amax = w.abs().amax(dim = (1, 2))
    g = torch.where(amax > 0, (256.0 * 6.0) / amax, torch.ones_like(amax)).clamp(max = 2.0**127)
    blocks = w.view(E, N, K // NVFP4_GROUP, NVFP4_GROUP)
    bmax = blocks.abs().amax(-1)
    out = []
    for top in (6.0, 4.0):
        s = (bmax / top * g[:, None, None]).clamp(max = 448.0).to(torch.float8_e4m3fn)
        d = s.to(torch.float32) / g[:, None, None]
        q = _round_e2m1(torch.where(d[..., None] > 0, blocks / d[..., None], torch.zeros_like(blocks)))
        out.append((q, s, ((_values(q) * d[..., None] - blocks) ** 2).sum(-1)))
    (q6, s6, e6), (q4, s4, e4) = out
    take4 = e4 < e6
    q = torch.where(take4[..., None], q4, q6)
    s = torch.where(take4, s4.view(torch.uint8), s6.view(torch.uint8))
    return _pack_nibbles(q.reshape(E, N, K)), s, g


def test_method_resolution(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MXFP4_WEIGHT_QUANT", raising = False)
    assert mxfp4_weight_quant_method() == "mse"
    for value, want in (("mse", "mse"), ("search", "mse"), ("rtn", "rtn"), ("OCP", "rtn"), ("0", "rtn")):
        monkeypatch.setenv("UNSLOTH_MXFP4_WEIGHT_QUANT", value)
        assert mxfp4_weight_quant_method() == want
    assert mxfp4_weight_quant_method("mse") == "mse"  # the argument wins
    monkeypatch.setenv("UNSLOTH_MXFP4_WEIGHT_QUANT", "max6")
    with pytest.raises(ValueError, match = "rtn, mse"):
        mxfp4_weight_quant_method()
    with pytest.raises(ValueError):
        nvfp4_weight_quant_method("mse")  # NVFP4 keeps max6 / 4over6 only


def test_4over6_unchanged_bit_for_bit():
    for seed in range(3):
        w = _weights(seed = seed)
        w[0, 1, 16:32] *= 2.0 ** -40  # a block near the e4m3 subnormal range
        q = quantize_experts_nvfp4(w, torch.bfloat16, method = "4over6")
        codes, s, g = _legacy_4over6(w)
        assert torch.equal(q.data, codes)
        assert torch.equal(q.packed_scales.view(torch.uint8), s)
        assert torch.equal(q.packed_global[:, 0], g)


def _mx_dense(q):
    return q.dequantize(torch.float32)


def _mx_legacy(rows):
    E, N, K = rows.shape
    w = rows.to(torch.float32).view(E, N, K // MX_GROUP, MX_GROUP)
    bmax = w.abs().amax(-1)
    e = torch.floor(torch.log2(bmax.clamp(min = 2.0**-126))) - 2
    e = torch.where(bmax > 0, e, torch.full_like(e, -127.0)).clamp(-127, 127)
    return _pack_nibbles(_round_e2m1(w / torch.exp2(e)[..., None])), (e + 127).to(torch.uint8)


def test_mxfp4_rtn_unchanged_and_mse_never_worse():
    for seed in range(3):
        w = _weights(seed = seed)
        rtn = quantize_experts_mxfp4(w, torch.bfloat16, method = "rtn")
        blocks, scales = _mx_legacy(w)
        assert torch.equal(rtn.data, blocks) and torch.equal(rtn.mxfp4_scales, scales)
        mse = quantize_experts_mxfp4(w, torch.bfloat16, method = "mse")
        d = mse.mxfp4_scales.to(torch.int16) - rtn.mxfp4_scales.to(torch.int16)
        assert set(d.unique().tolist()) <= {0, 1}  # only ever the next exponent up
        assert (d == 1).any()
        wb = w.view(*w.shape[:2], -1, MX_GROUP)
        e_rtn = ((_mx_dense(rtn).view_as(wb) - wb) ** 2).sum(-1)
        e_mse = ((_mx_dense(mse).view_as(wb) - wb) ** 2).sum(-1)
        assert (e_mse <= e_rtn).all()
        assert e_mse.sum() < 0.97 * e_rtn.sum()


def test_mxfp4_mse_equals_the_full_search():
    """The clipping shortcut (only blocks with bmax > 6 * 2^e try e + 1) picks what trying every block picks."""
    for seed in range(3):
        w = _weights(seed = seed).to(torch.float32)
        q = quantize_experts_mxfp4(w, torch.bfloat16, method = "mse")
        b = w.view(*w.shape[:2], -1, MX_GROUP)
        bmax = b.abs().amax(-1)
        e = torch.floor(torch.log2(bmax.clamp(min = 2.0**-126))) - 2
        e = torch.where(bmax > 0, e, torch.full_like(e, -127.0)).clamp(-127, 127)
        errs = []
        for ee in (e, (e + 1).clamp(max = 127)):
            sc = torch.exp2(ee)[..., None]
            errs.append(((_values(_round_e2m1(b / sc)) * sc - b) ** 2).sum(-1))
        want = torch.where(errs[1] < errs[0], (e + 1).clamp(max = 127), e)
        assert torch.equal(q.mxfp4_scales, (want + 127).to(torch.uint8))


def test_mxfp4_mse_zero_block_and_env(monkeypatch):
    w = _weights()
    w[1] = 0
    monkeypatch.setenv("UNSLOTH_MXFP4_WEIGHT_QUANT", "mse")
    q = quantize_experts_mxfp4(w, torch.bfloat16)
    assert torch.equal(_mx_dense(q)[1], torch.zeros_like(w[1]))
    assert torch.equal(q.mxfp4_scales[1], torch.zeros_like(q.mxfp4_scales[1]))  # e8m0 2^-127, as RTN writes
    assert torch.equal(q.data, quantize_experts_mxfp4(w, torch.bfloat16, method = "mse").data)


def test_e2m1_magnitude_equals_the_code_round_trip():
    from unsloth_zoo.packed_experts import _e2m1_magnitude
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    pts = torch.cat([
        mids, _LUT, torch.nextafter(mids, torch.tensor(9.0)), torch.nextafter(mids, torch.tensor(-9.0)),
        torch.rand(200_000, generator = torch.Generator().manual_seed(0)) * 8,
        torch.tensor([6.0001, 7.0, 1e30, float("inf"), 0.0, 1e-45]),
    ])
    assert torch.equal(_e2m1_magnitude(pts), _values(_round_e2m1(pts)))
