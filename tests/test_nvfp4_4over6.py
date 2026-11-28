"""NVFP4 4over6 block scales in quantize_experts_nvfp4 (CPU)."""

import pytest
import torch

from unsloth_zoo.packed_experts import (
    NVFP4_GROUP,
    _E2M1,
    _pack_nibbles,
    _round_e2m1,
    nvfp4_experts_dequantize_torch,
    nvfp4_weight_quant_method,
    quantize_experts_nvfp4,
)


def _weights(E = 3, N = 32, K = 256, seed = 0):
    g = torch.Generator().manual_seed(seed)
    # Per-column spread plus a few outliers, so block maxima vary over orders of magnitude.
    w = torch.randn(E, N, K, generator = g) * torch.logspace(-2, 0.5, K)
    w[:, 0, :3] *= 40
    return w


def _legacy_rtn(rows):
    """The round-to-nearest quantizer before 4over6 (global = 448 * 6 / amax)."""
    E, N, K = rows.shape
    w = rows.to(torch.float32)
    amax = w.abs().amax(dim = (1, 2))
    g = torch.where(amax > 0, (448.0 * 6.0) / amax, torch.ones_like(amax))
    blocks = w.view(E, N, K // NVFP4_GROUP, NVFP4_GROUP)
    s = (blocks.abs().amax(dim = -1) / 6.0 * g[:, None, None]).clamp(max = 448.0).to(torch.float8_e4m3fn)
    decode = s.to(torch.float32) / g[:, None, None]
    y = torch.where(decode[..., None] > 0, blocks / decode[..., None], torch.zeros_like(blocks))
    return _pack_nibbles(_round_e2m1(y).reshape(E, N, K)), s, g


def _decode(q, dtype = torch.float32):
    return nvfp4_experts_dequantize_torch(q.data, q.packed_scales, q.packed_global, dtype)


def _rel_mse(q, w):
    return (((_decode(q) - w) ** 2).sum() / (w ** 2).sum()).item()


def test_method_resolution(monkeypatch):
    monkeypatch.delenv("UNSLOTH_NVFP4_WEIGHT_QUANT", raising = False)
    assert nvfp4_weight_quant_method() == "4over6"
    for value, want in (("max6", "max6"), ("RTN", "max6"), ("0", "max6"), ("4/6", "4over6"), ("four-over-six", "4over6")):
        monkeypatch.setenv("UNSLOTH_NVFP4_WEIGHT_QUANT", value)
        assert nvfp4_weight_quant_method() == want
    monkeypatch.setenv("UNSLOTH_NVFP4_WEIGHT_QUANT", "max6")
    assert nvfp4_weight_quant_method("4over6") == "4over6"  # the argument wins
    monkeypatch.setenv("UNSLOTH_NVFP4_WEIGHT_QUANT", "max5")
    with pytest.raises(ValueError):
        nvfp4_weight_quant_method()


def test_max6_is_the_previous_rtn_bit_for_bit():
    w = _weights()
    q = quantize_experts_nvfp4(w, torch.bfloat16, method = "max6")
    codes, s, g = _legacy_rtn(w)
    assert torch.equal(q.data, codes)
    assert torch.equal(q.packed_scales.view(torch.uint8), s.view(torch.uint8))
    assert torch.equal(q.packed_global[:, 0], g)


def test_4over6_scales_and_selection():
    w = _weights()
    q = quantize_experts_nvfp4(w, torch.bfloat16, method = "4over6")
    amax = w.abs().amax(dim = (1, 2))
    assert torch.equal(q.packed_global[:, 0], (256.0 * 6.0) / amax)
    s = q.packed_scales.view(torch.float8_e4m3fn).to(torch.float32)
    assert s.max().item() <= 384.0  # the max/4 candidate tops out at 256 * 6 / 4
    # Per block the kept choice is never worse than the max/6 candidate under the same global scale.
    g = q.packed_global[:, 0]
    blocks = w.view(*w.shape[:2], -1, NVFP4_GROUP)
    s6 = (blocks.abs().amax(-1) / 6.0 * g[:, None, None]).clamp(max = 448.0).to(torch.float8_e4m3fn)
    d6 = s6.to(torch.float32) / g[:, None, None]
    y6 = torch.where(d6[..., None] > 0, blocks / d6[..., None], torch.zeros_like(blocks))
    lut = torch.tensor(_E2M1)
    c6 = _round_e2m1(y6)
    v6 = torch.where((c6 & 8).bool(), -lut[(c6 & 7).long()], lut[(c6 & 7).long()]) * d6[..., None]
    err6 = ((v6 - blocks) ** 2).sum(-1)
    err = ((_decode(q).view_as(blocks) - blocks) ** 2).sum(-1)
    assert (err <= err6 * (1 + 1e-6) + 1e-30).all()
    took4 = (s != s6.to(torch.float32))
    assert 0 < took4.float().mean().item() < 1  # both choices occur on this data


def test_4over6_lowers_weight_error():
    for seed in range(3):
        w = _weights(seed = seed)
        rtn = _rel_mse(quantize_experts_nvfp4(w, torch.bfloat16, method = "max6"), w)
        fos = _rel_mse(quantize_experts_nvfp4(w, torch.bfloat16, method = "4over6"), w)
        assert fos < rtn * 0.95, (seed, rtn, fos)


def test_4over6_zero_expert_decodes_to_zero():
    w = _weights()
    w[1] = 0
    q = quantize_experts_nvfp4(w, torch.bfloat16, method = "4over6")
    assert torch.equal(_decode(q)[1], torch.zeros_like(w[1]))
    assert torch.isfinite(q.packed_global).all()


@pytest.mark.parametrize("method", ["max6", "4over6"])
def test_compressed_tensors_decompress_is_bit_identical(method):
    pytest.importorskip("compressed_tensors")
    from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
    from compressed_tensors.quantization import preset_name_to_scheme

    scheme = preset_name_to_scheme("NVFP4A16", ["Linear"])
    w = _weights()
    q = quantize_experts_nvfp4(w, torch.bfloat16, method = method)
    ours = _decode(q, torch.bfloat16)

    def ct(scale):
        out = []
        for e in range(w.shape[0]):
            sd = {"weight_packed": q.data[e], "weight_scale": scale[e], "weight_global_scale": q.packed_global[e]}
            out.append(NVFP4PackedCompressor.decompress(sd, scheme)["weight"])
        return torch.stack(out)

    scale = q.packed_scales.view(torch.float8_e4m3fn)
    theirs = ct(scale)
    assert theirs.dtype == torch.bfloat16
    assert torch.equal(theirs, ours)
    # Negative control: one block scale one code step off must break bit identity.
    bad = scale.view(torch.uint8).clone()
    bad[0, 0, 0] += 1
    assert not torch.equal(ct(bad.view(torch.float8_e4m3fn)), ours)


@pytest.mark.parametrize("method", ["max6", "4over6"])
def test_tiny_amax_expert_keeps_a_finite_global_scale(method):
    """An expert stack with amax near 1e-37 overflowed top * 6 / amax to inf; its zero blocks decoded 0 * inf = NaN."""
    from unsloth_zoo.packed_experts import quantize_experts_nvfp4
    gen = torch.Generator().manual_seed(0)
    w = torch.zeros(2, 16, 32)
    w[0] = torch.randn(16, 32, generator = gen)
    w[1, :, :16] = torch.randn(16, 16, generator = gen) * 1e-37
    param = quantize_experts_nvfp4(w, torch.bfloat16, method = method)
    glob = param._companions()[1]
    assert torch.isfinite(glob).all()
    dense = param.dequantize(torch.float32)
    assert torch.isfinite(dense).all()
    assert torch.equal(dense[1, :, 16:], torch.zeros(16, 16))
    # In-range experts are untouched by the cap.
    ref = quantize_experts_nvfp4(w[:1].clone(), torch.bfloat16, method = method)
    assert torch.equal(param.data[:1], ref.data) and torch.equal(glob[:1], ref._companions()[1])
