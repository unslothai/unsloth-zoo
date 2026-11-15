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

"""float32 gated delta takes IEEE float32 dots: with Triton's default TF32 the decay gradient (dg -> A_log, dt_bias,
in_proj_a) of fla's chunk_gated_delta_rule was wrong on strongly decaying heads (32x the fp64 value on B200)."""

import os
import types

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")

try:
    import unsloth_zoo  # noqa: F401
    from unsloth_zoo.temporary_patches import fla_vendor
except ImportError as _e:
    pytest.skip(f"unsloth_zoo unavailable: {_e}", allow_module_level=True)


class _Tensor:
    def __init__(self, dtype, is_cuda=True):
        self.dtype = dtype
        self.is_cuda = is_cuda


def _install_on(fake_fwd, fake_bwd):
    import sys
    mod = types.ModuleType("fla.ops.gated_delta_rule.chunk")
    mod.chunk_gated_delta_rule_fwd = fake_fwd
    mod.chunk_gated_delta_rule_bwd = fake_bwd
    saved = sys.modules.get(mod.__name__)
    sys.modules[mod.__name__] = mod
    try:
        assert fla_vendor._install_fp32_ieee_dots()
        assert fla_vendor._install_fp32_ieee_dots()   # idempotent: wrapped once
        return mod
    finally:
        if saved is None:
            sys.modules.pop(mod.__name__, None)
        else:
            sys.modules[mod.__name__] = saved


def test_wrapper_sets_ieee_only_for_cuda_float32(monkeypatch):
    import torch
    monkeypatch.delenv("TRITON_F32_DEFAULT", raising=False)
    monkeypatch.delenv("UNSLOTH_FLA_FP32_TF32", raising=False)
    seen = []

    def fwd(q, *args, **kwargs):
        seen.append(os.environ.get("TRITON_F32_DEFAULT"))
        return "out"

    mod = _install_on(fwd, fwd)
    assert getattr(mod.chunk_gated_delta_rule_fwd, "__wrapped__", None) is fwd
    assert mod.chunk_gated_delta_rule_fwd(_Tensor(torch.float32)) == "out"
    assert mod.chunk_gated_delta_rule_bwd(q=_Tensor(torch.float32)) == "out"
    mod.chunk_gated_delta_rule_fwd(_Tensor(torch.bfloat16))
    mod.chunk_gated_delta_rule_fwd(_Tensor(torch.float32, is_cuda=False))
    assert seen == ["ieee", "ieee", None, None]
    assert "TRITON_F32_DEFAULT" not in os.environ   # restored after the call
    monkeypatch.setenv("TRITON_F32_DEFAULT", "tf32x3")   # an explicit choice wins
    mod.chunk_gated_delta_rule_fwd(_Tensor(torch.float32))
    monkeypatch.delenv("TRITON_F32_DEFAULT")
    monkeypatch.setenv("UNSLOTH_FLA_FP32_TF32", "1")     # opt out
    mod.chunk_gated_delta_rule_fwd(_Tensor(torch.float32))
    assert seen[-2:] == ["tf32x3", None]


def _recurrent_reference(q, k, v, g, beta):
    """Gated delta rule, token by token, in float64 (q / k L2-normalised, q scaled by 1/sqrt(K))."""
    import torch
    q = torch.nn.functional.normalize(q, dim=-1) * q.shape[-1] ** -0.5
    k = torch.nn.functional.normalize(k, dim=-1)
    B, T, H, K = q.shape
    S = q.new_zeros(B, H, K, v.shape[-1])
    out = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        delta = (v[:, t] - (S * k[:, t, ..., None]).sum(-2)) * beta[:, t, ..., None]
        S = S + k[:, t, ..., None] * delta[..., None, :]
        out.append((S * q[:, t, ..., None]).sum(-2))
    return torch.stack(out, 1)


def test_float32_decay_gradient_matches_float64():
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("needs a CUDA GPU with TF32 (sm80+)")
    if not fla_vendor._vendored_injection_supported():
        pytest.skip("vendored fla kernels need CUDA + torch>=2.7 + triton>=3.3")
    fla_vendor.patch_vendor_fla()   # what importing unsloth_zoo runs: inject + install the call-site patches
    import fla
    if getattr(fla, fla_vendor._VENDORED_MARK, False) is not True:
        pytest.skip("vendored fla injection failed on this host")
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    gen = torch.Generator(device="cuda").manual_seed(0)
    B, T, H, K, V = 1, 128, 2, 32, 32
    q, k, v = (torch.randn(B, T, H, D, device="cuda", dtype=torch.float64, generator=gen) for D in (K, K, V))
    a = 0.5 * torch.randn(B, T, H, device="cuda", dtype=torch.float64, generator=gen)
    beta = torch.rand(B, T, H, device="cuda", dtype=torch.float64, generator=gen)
    w = torch.randn(B, T, H, V, device="cuda", dtype=torch.float64, generator=gen)
    A = torch.tensor([7.0, 15.0], device="cuda", dtype=torch.float64)   # strong decay, as Qwen3.8 layer 1

    def grads(fn, dtype):
        a_ = a.clone().to(dtype).requires_grad_()
        g = -A.to(dtype) * F.softplus(a_ + 1.0)
        o = fn(*(x.to(dtype) for x in (q, k, v)), g, beta.to(dtype))
        (o.double() * w).sum().backward()
        return o.detach().double(), a_.grad.double()

    o_ref, da_ref = grads(_recurrent_reference, torch.float64)
    o, da = grads(lambda q_, k_, v_, g_, b_: chunk_gated_delta_rule(
        q_, k_, v_, g=g_, beta=b_, use_qk_l2norm_in_kernel=True)[0], torch.float32)
    o_rel = ((o - o_ref).norm() / o_ref.norm()).item()
    da_rel = ((da - da_ref).norm() / da_ref.norm()).item()
    # TF32 dots: o about 1.5e-3, da about 1 to 30 (strong decay); IEEE: o about 1e-6, da about 1e-3.
    assert o_rel < 1e-4 and da_rel < 1e-2, {"o_rel": o_rel, "da_rel": da_rel}


def test_fla_flash_qla_dispatch_defaults_off(monkeypatch):
    # fla >= 0.6 sends grad-free SM120 calls (a no-grad checkpoint forward) to FlashQLA and the recompute to Triton.
    for key in ("FLA_FLASH_QLA", "UNSLOTH_GDN_BACKEND"):
        monkeypatch.delenv(key, raising=False)
    fla_vendor.patch_vendor_fla()
    assert os.environ.get("FLA_FLASH_QLA") == "0"
    monkeypatch.setenv("FLA_FLASH_QLA", "1")   # an explicit choice wins
    fla_vendor.patch_vendor_fla()
    assert os.environ["FLA_FLASH_QLA"] == "1"
    monkeypatch.delenv("FLA_FLASH_QLA")
    monkeypatch.setenv("UNSLOTH_GDN_BACKEND", "flashqla")   # Unsloth's own FlashQLA routing: left alone
    fla_vendor.patch_vendor_fla()
    assert "FLA_FLASH_QLA" not in os.environ
