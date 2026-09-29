# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Per-expert FP8 paths must multiply by `weight_scale_inv` (w = q * s, as FP8Linear), never invert it; e8m0 has no CUDA reciprocal."""

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils_fp8 as F

E8M0 = getattr(torch, "float8_e8m0fnu", None)
needs_e8m0 = pytest.mark.skipif(E8M0 is None, reason = "torch without float8_e8m0fnu")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


def _pow2_scales(shape, device):
    exponents = torch.randint(-20, 4, shape, device = device).to(torch.float32)
    return torch.exp2(exponents)


class _Experts(torch.nn.Module):
    def __init__(self, scales):
        super().__init__()
        E = scales.shape[0]
        self.num_experts = E
        self.block_size = [128, 128]
        self.gate_up_proj = torch.nn.Parameter(
            torch.zeros(E, 256, 256, device = scales.device).to(torch.float8_e4m3fn),
            requires_grad = False,
        )
        self.gate_up_proj_scale_inv = torch.nn.Parameter(scales, requires_grad = False)


@needs_e8m0
@needs_cuda
def test_linear_quant_state_slice_accepts_e8m0_on_cuda():
    s = _pow2_scales((4, 2, 2), "cuda")
    with pytest.raises(NotImplementedError, match = "Float8_e8m0fnu"):
        s.to(E8M0).reciprocal()
    from_e8m0 = F._slice_fp8_linear_quant_state(_Experts(s.to(E8M0)), "gate_up_proj", 1)
    from_f32 = F._slice_fp8_linear_quant_state(_Experts(s.clone()), "gate_up_proj", 1)
    assert torch.equal(from_e8m0.to(torch.float32), from_f32)


@needs_e8m0
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks = needs_cuda)])
def test_expert_slice_dequant_matches_the_float32_scale(device):
    torch.manual_seed(0)
    w = (torch.randn(256, 256, device = device) * 4).to(torch.float8_e4m3fn)
    s = _pow2_scales((2, 2), device)
    for kind in (None, "weight_scale_inv"):
        got = F._dequantize_expert_slice(w, s.to(E8M0), torch.bfloat16, quant_kind = kind)
        want = F._dequantize_expert_slice(w, s.clone(), torch.bfloat16, quant_kind = kind)
        assert torch.equal(got, want)


_B = 128


def _quantize(shape, device, e8m0):
    w = torch.randn(*shape, device = device) * 0.05
    E, R, C = shape
    t = w.view(E, R // _B, _B, C // _B, _B)
    s = t.abs().amax(dim = (2, 4)) / 448.0
    if e8m0:
        s = torch.exp2(torch.ceil(torch.log2(s)))
    q = (t / s[:, :, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(shape)
    return q, (s.to(E8M0) if e8m0 else s)


def _reference_dequant(q, s):
    E, R, C = q.shape
    s = s.to(torch.float32)
    return (q.to(torch.float32).view(E, R // _B, _B, C // _B, _B) * s[:, :, None, :, None]).view(E, R, C)


class _BlockExperts(torch.nn.Module):
    def __init__(self, E, H, I, device, e8m0):
        super().__init__()
        self.num_experts = E
        self.block_size = [_B, _B]
        self.act_fn = torch.nn.functional.silu
        self.gate_up_proj, self.gate_up_proj_scale_inv = _quantize((E, 2 * I, H), device, e8m0)
        self.down_proj, self.down_proj_scale_inv = _quantize((E, H, I), device, e8m0)


def _reference_forward(m, x, idx, wts):
    gu = _reference_dequant(m.gate_up_proj, m.gate_up_proj_scale_inv)
    dn = _reference_dequant(m.down_proj, m.down_proj_scale_inv)
    out = torch.zeros(x.shape, dtype = torch.float32, device = x.device)
    for t in range(x.shape[0]):
        for k in range(idx.shape[1]):
            e = idx[t, k]
            g, u = (x[t].float() @ gu[e].T).chunk(2)
            out[t] += wts[t, k] * ((torch.nn.functional.silu(g) * u) @ dn[e].T)
    return out


def _hide_unsloth_fp8_linear(monkeypatch):
    import sys, types
    monkeypatch.setitem(sys.modules, "unsloth.kernels.fp8", types.ModuleType("unsloth.kernels.fp8"))


@needs_cuda
@pytest.mark.parametrize("e8m0", [False, pytest.param(True, marks = needs_e8m0)])
@pytest.mark.parametrize("fp8_linear", ["unsloth", "dequant"])
def test_per_expert_loop_matches_the_dequantized_reference(e8m0, fp8_linear, monkeypatch):
    if fp8_linear == "unsloth":
        pytest.importorskip("unsloth.kernels.fp8")
    else:
        _hide_unsloth_fp8_linear(monkeypatch)
    torch.manual_seed(0)
    m = _BlockExperts(4, 512, 256, "cuda", e8m0)
    x = torch.randn(16, 512, device = "cuda", dtype = torch.bfloat16)
    idx = torch.randint(0, 4, (16, 2), device = "cuda")
    wts = torch.rand(16, 2, device = "cuda")
    ref = _reference_forward(m, x, idx, wts)
    out = F._forward_native_fp8_expert_loop(m, x, idx, wts).float()
    assert torch.isfinite(out).all()
    assert ((out - ref).norm() / ref.norm()).item() < 0.05


@pytest.mark.parametrize("e8m0", [False, pytest.param(True, marks = needs_e8m0)])
def test_per_expert_fallback_agrees_with_the_vectorized_dequant(e8m0):
    """Fallback 2 of _dequantize_full_expert_weights (2-D scale plus block_size) against fallback 1."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q, s = _quantize((2, 256, 256), device, e8m0)
    want = _reference_dequant(q, s)
    assert torch.equal(F._dequantize_full_expert_weights_vectorized(q, s, torch.float32), want)
    for e in range(2):
        got = F._dequantize_expert_slice(q[e], s[e], torch.float32, quant_kind = "weight_scale_inv")
        assert torch.equal(got, want[e])
