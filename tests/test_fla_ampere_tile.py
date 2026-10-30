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

"""A100-class 128-wide chunk_o backward tiles for 16-bit inputs (fla PR 1514); fp32 stays at 64 and other tiers are
unchanged; tiers are simulated with Triton kernels swapped for recorders."""

import os

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")

try:
    import unsloth_zoo  # noqa: F401
except ImportError as _e:
    pytest.skip(f"unsloth_zoo unavailable: {_e}", allow_module_level=True)


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _load_vendored_fla():
    from unsloth_zoo.temporary_patches.fla_vendor import (
        _inject_vendored_fla,
        _vendored_injection_supported,
    )

    if not _vendored_injection_supported():
        pytest.skip("vendored fla kernels need CUDA + torch>=2.7 + triton>=3.3")
    injected, _ = _inject_vendored_fla()
    if not injected:
        pytest.skip("vendored fla injection failed on this host")
    import fla.ops.common.chunk_o as chunk_o
    return chunk_o


# Shared-memory tiers as fla's Backend enum defines them (bytes).
_TIERS = {
    "hopper": 232448,  # H100 / B200
    "a100": 166912,    # A100 / A800
    "ada": 101376,     # L4 / RTX 4090 / RTX PRO 6000
    "small": 65536,    # T4
}
_ARCH_BYTES = {"hopper": 232448, "ampere": 166912, "ada": 101376}


class _Recorder:
    def __init__(self, seen, name):
        self.seen, self.name = seen, name

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.seen[self.name] = (kwargs["BK"], kwargs["BV"])
        return launch


def _run_all(chunk_o, tier, dtype, monkeypatch, state_dtype=None):
    import torch

    smem = _TIERS[tier]
    monkeypatch.setattr(
        chunk_o, "check_shared_mem",
        lambda arch="none", tensor_idx=0: smem >= _ARCH_BYTES.get(arch, 102400),
    )
    seen = {}
    for name in ("chunk_bwd_kernel_dv", "chunk_bwd_kernel_dv_local", "chunk_bwd_kernel_dqkwg"):
        monkeypatch.setattr(chunk_o, name, _Recorder(seen, name))

    B, T, H, K, V, BT = 1, 128, 2, 128, 128, 64
    NT = T // BT
    dev = "cuda"
    sdt = state_dtype or dtype
    q = torch.randn(B, T, H, K, device=dev, dtype=dtype)
    k = torch.randn(B, T, H, K, device=dev, dtype=dtype)
    v = torch.randn(B, T, H, V, device=dev, dtype=dtype)
    do = torch.randn(B, T, H, V, device=dev, dtype=dtype)
    h = torch.randn(B, NT, H, K, V, device=dev, dtype=sdt)
    dh = torch.randn(B, NT, H, K, V, device=dev, dtype=sdt)
    w = torch.randn(B, T, H, K, device=dev, dtype=dtype)
    dv = torch.randn(B, T, H, V, device=dev, dtype=dtype)
    A = torch.randn(B, T, H, BT, device=dev, dtype=dtype)
    g = torch.randn(B, T, H, device=dev, dtype=torch.float32)

    def raw(fn):
        return getattr(fn, "__wrapped__", fn)

    raw(chunk_o.chunk_bwd_dv)(q=q, k=k, do=do, dh=dh, g=g, scale=K ** -0.5, chunk_size=BT)
    raw(chunk_o.chunk_bwd_dv_local)(q=q, k=k, do=do, g=g, A=A, scale=K ** -0.5, chunk_size=BT)
    raw(chunk_o.chunk_bwd_dqkwg)(q=q, k=k, v=v, do=do, h=h, dh=dh, w=w, g=g, dv=dv,
                                 scale=K ** -0.5, chunk_size=BT)
    return seen


@pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
def test_a100_16bit_takes_128_wide_tiles(dtype_name, monkeypatch):
    import torch

    chunk_o = _load_vendored_fla()
    seen = _run_all(chunk_o, "a100", getattr(torch, dtype_name), monkeypatch)
    assert seen == {
        "chunk_bwd_kernel_dv": (128, 128),
        "chunk_bwd_kernel_dv_local": (128, 128),
        "chunk_bwd_kernel_dqkwg": (128, 128),
    }


@pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")
def test_a100_fp32_keeps_64_wide_tiles(monkeypatch):
    """fp32 inputs need ~196-229 KB at 128-wide tiles, more than an A100 has."""
    import torch

    chunk_o = _load_vendored_fla()
    seen = _run_all(chunk_o, "a100", torch.float32, monkeypatch)
    assert set(seen.values()) == {(64, 64)}


@pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")
def test_a100_fp32_states_keep_dqkwg_at_64(monkeypatch):
    """Simple GLA keeps fp32 h and dh in backward: dqkwg loads them and stays at 64, while the two dv kernels (which do not load h) take 128."""
    import torch

    chunk_o = _load_vendored_fla()
    seen = _run_all(chunk_o, "a100", torch.bfloat16, monkeypatch, state_dtype=torch.float32)
    assert seen["chunk_bwd_kernel_dqkwg"] == (64, 64)
    assert seen["chunk_bwd_kernel_dv_local"] == (128, 128)
    # chunk_bwd_dv loads dh, so an fp32 dh keeps it at 64 too.
    assert seen["chunk_bwd_kernel_dv"] == (64, 64)


@pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")
@pytest.mark.parametrize("tier,expected", [("hopper", 128), ("ada", 64), ("small", 32)])
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float32"])
def test_other_tiers_unchanged(tier, expected, dtype_name, monkeypatch):
    import torch

    chunk_o = _load_vendored_fla()
    seen = _run_all(chunk_o, tier, getattr(torch, dtype_name), monkeypatch)
    assert set(seen.values()) == {(expected, expected)}


@pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")
def test_gdn_fwd_bwd_matches_fp32_recurrence_with_128_tiles(monkeypatch):
    """End to end through the real kernels with the A100 tier forced; the GDN backward must match an fp32 naive recurrence within bf16 rounding."""
    import torch
    import torch.nn.functional as F

    chunk_o = _load_vendored_fla()
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("bf16 GDN needs sm80+")
    props = torch.cuda.get_device_properties(0)
    smem = getattr(props, "shared_memory_per_block_optin", 0)
    if smem and smem < _TIERS["a100"] - 1024:
        pytest.skip("this GPU cannot hold the 128-wide tiles")
    monkeypatch.setattr(
        chunk_o, "check_shared_mem",
        lambda arch="none", tensor_idx=0: _TIERS["a100"] >= _ARCH_BYTES.get(arch, 102400),
    )
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(0)
    B, T, H, D = 1, 256, 4, 128
    dev, dt = "cuda", torch.bfloat16
    q, k, v = (torch.randn(B, T, H, D, device=dev).to(dt).float() for _ in range(3))
    g = -F.softplus(torch.randn(B, T, H, device=dev)) * 0.5
    beta = torch.sigmoid(torch.randn(B, T, H, device=dev)).to(dt).float()
    do = torch.randn(B, T, H, D, device=dev).to(dt).float()
    scale = D ** -0.5

    xs = [q.to(dt), k.to(dt), v.to(dt), g.clone(), beta.to(dt)]
    xs = [x.requires_grad_() for x in xs]
    o, _ = chunk_gated_delta_rule(*xs, scale=scale, use_qk_l2norm_in_kernel=True)
    o.backward(do.to(dt))

    rs = [x.detach().clone().float().requires_grad_() for x in (q, k, v, g, beta)]
    qn = F.normalize(rs[0], dim=-1, eps=1e-6) * scale
    kn = F.normalize(rs[1], dim=-1, eps=1e-6)
    S = qn.new_zeros(B, H, D, D)
    outs = []
    for t in range(T):
        S = S * rs[3][:, t].exp()[..., None, None]
        u = rs[4][:, t][..., None] * (rs[2][:, t] - torch.einsum("bhk,bhkv->bhv", kn[:, t], S))
        S = S + kn[:, t][..., :, None] * u[..., None, :]
        outs.append(torch.einsum("bhk,bhkv->bhv", qn[:, t], S))
    ro = torch.stack(outs, 1)
    ro.backward(do)

    def rel(a, b):
        return float((a.float() - b.float()).norm() / b.float().norm())

    assert rel(o, ro) < 1e-2
    for name, x, r in zip(("dq", "dk", "dv", "dg", "dbeta"), xs, rs):
        assert rel(x.grad, r.grad) < 2e-2, name
