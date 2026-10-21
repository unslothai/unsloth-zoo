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
"""sonic-moe zero-copy gate_up (concat_layout=("B",)) layout algebra.

The CPU tests pin the layout facts the zero-copy path depends on, emulating quack's
`concat_to_interleave(t, 1 - t.leading_dim)` with `as_strided`. The GPU test runs only
where the kernels-community/sonic-moe build loads (sm90 / sm100 with `kernels`).
"""
import pytest
import torch
import torch.nn.functional as F

from unsloth_zoo.temporary_patches import moe_utils_sonic as S


def _quack_concat_view_b(weight_t):
    """What the kernel reads for B = `weight_t` ([E, K, N]) with "B" in concat_layout:
    the non-contiguous of (N, K) is split as (2, half) with strides (half * s, s)."""
    E, Kdim, N = weight_t.shape
    sE, sK, sN = weight_t.stride()
    if sK == 1:      # K-major: N is the non-contiguous dim -> N interleaved
        half = N // 2
        return torch.as_strided(weight_t, (E, Kdim, half, 2), (sE, sK, sN, half * sN)).reshape(E, Kdim, N)
    half = Kdim // 2  # N-major: K is the non-contiguous dim -> K interleaved
    return torch.as_strided(weight_t, (E, half, 2, N), (sE, sK, half * sK, sN)).reshape(E, Kdim, N)


def test_concat_view_of_k_major_is_the_interleaved_weight():
    E, H, inter = 3, 16, 8
    W = torch.randn(E, 2 * inter, H)            # HF storage [E, 2I, H]
    Wt = W.transpose(-2, -1)                # grouped_mm view [E, H, 2I], K-major
    assert S._is_k_major_gate_up(Wt)
    assert torch.equal(_quack_concat_view_b(Wt), S._interleave_columns(Wt))


def test_concat_view_of_n_major_interleaves_the_contraction_dim():
    # The old probe passed a contiguous [E, H, 2I]: concat then permutes K (= H),
    # which is not the gate/up interleave and is rejected by the eligibility check.
    E, H, inter = 2, 16, 8
    Wn = torch.randn(E, H, 2 * inter)
    assert not S._is_k_major_gate_up(Wn)
    assert not torch.equal(_quack_concat_view_b(Wn), S._interleave_columns(Wn))


def test_dx_concat_on_hf_storage_matches_chunked_dx():
    # dX = d_pre @ W: B is the HF [E, 2I, H] storage (N-major), concat interleaves K = 2I,
    # so an interleaved d_pre gives the same dX as the chunked d_pre on the plain weight.
    E, H, inter, M = 1, 16, 8, 5
    W = torch.randn(E, 2 * inter, H, dtype=torch.float64)
    dpre = torch.randn(M, 2 * inter, dtype=torch.float64)
    dx_ref = dpre @ W[0]
    dx = S._interleave_columns(dpre) @ _quack_concat_view_b(W)[0]
    torch.testing.assert_close(dx, dx_ref)


def test_sqrt2_signature_of_column_order_mismatch():
    # An interleaved preact compared against a chunked reference: rel err ~ sqrt(2).
    torch.manual_seed(0)
    ref = torch.randn(4096, 512, dtype=torch.float64)
    rel = ((S._interleave_columns(ref) - ref).norm() / ref.norm()).item()
    assert abs(rel - 2 ** 0.5) < 0.05


def test_interleaved_lora_up_projection_gives_interleaved_delta():
    H, inter, R, M = 16, 8, 4, 6
    x = torch.randn(M, H, dtype=torch.float64)
    A = torch.randn(H, R, dtype=torch.float64)
    B = torch.randn(R, 2 * inter, dtype=torch.float64, requires_grad=True)
    delta_il = (x @ A) @ S._interleave_columns(B)
    torch.testing.assert_close(delta_il, S._interleave_columns((x @ A) @ B))
    g = torch.randn_like(delta_il)
    (delta_il * g).sum().backward()
    grad_il = B.grad.clone()
    B.grad = None
    g_chunked = torch.cat([g[:, 0::2], g[:, 1::2]], dim=-1)
    ((x @ A) @ B * g_chunked).sum().backward()
    torch.testing.assert_close(grad_il, B.grad)


def test_swiglu_backward_interleaved_matches_autograd():
    M, inter = 7, 12
    pre = torch.randn(M, 2 * inter, dtype=torch.float64, requires_grad=True)
    out = F.silu(pre[:, :inter]) * pre[:, inter:]
    g = torch.randn(M, inter, dtype=torch.float64)
    (out * g).sum().backward()
    d_il = S._swiglu_backward_interleaved(S._interleave_columns(pre.detach()), g)
    torch.testing.assert_close(d_il, S._interleave_columns(pre.grad))


def test_k_major_check_rejects_misaligned_strides():
    W = torch.randn(2, 6, 7, dtype=torch.bfloat16)   # H=7: 14-byte stride, not TMA aligned
    assert not S._is_k_major_gate_up(W.transpose(-2, -1))
    assert not S._is_k_major_gate_up(torch.randn(2, 7, 16).transpose(-2, -1))  # odd 2I
    assert not S._is_k_major_gate_up(torch.randn(2, 24, 16).transpose(-2, -1))  # inter=12, rows not 16B


def _sonic_ready():
    if not torch.cuda.is_available():
        return False
    try:
        return S.sonic_moe_available() and S._concat_layout_supported(S.get_sonic_module())
    except Exception:
        return False


@pytest.mark.skipif(not _sonic_ready(), reason="kernels-community/sonic-moe with concat_layout not available")
@pytest.mark.parametrize("lora", [False, True])
def test_zero_copy_matches_grouped_mm(lora, monkeypatch):
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts
    from unsloth_zoo.temporary_patches import moe_utils as MU

    torch.manual_seed(0)
    dev = torch.device("cuda")
    H, inter, E, K, T, R = 256, 128, 8, 2, 512, 8
    cfg = Qwen3MoeConfig(hidden_size=H, moe_intermediate_size=inter, num_experts=E,
                         num_experts_per_tok=K, hidden_act="silu")
    experts = Qwen3MoeExperts(cfg).to(dev)
    with torch.no_grad():
        experts.gate_up_proj.normal_(0, H ** -0.5)
        experts.down_proj.normal_(0, inter ** -0.5)
    experts = experts.to(torch.bfloat16).requires_grad_(False)
    X0 = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    topv, topi = torch.randn(T, E, device=dev).topk(K, dim=-1)
    W0 = torch.softmax(topv, -1).to(torch.bfloat16)
    lora_w = [torch.randn(E, H, R, device=dev) * H ** -0.5, torch.randn(E, R, 2 * inter, device=dev) * 0.05]

    def run(fn):
        X = X0.clone().requires_grad_(True)
        Wt = W0.clone().requires_grad_(True)
        lw = [t.clone().requires_grad_(True) for t in lora_w]
        if lora:
            experts._unsloth_lora_gate_up_proj = (lw[0], lw[1], 2.0)
        else:
            experts.__dict__.pop("_unsloth_lora_gate_up_proj", None)
        out = fn(experts, X, topi, Wt)
        out.float().square().sum().backward()
        grads = [X.grad, Wt.grad] + ([t.grad for t in lw] if lora else [])
        return [out] + grads

    ref = run(MU.forward_native_grouped_mm)
    monkeypatch.setenv("UNSLOTH_MOE_SONIC_GATED", "1")
    monkeypatch.setenv("UNSLOTH_MOE_SONIC_ZERO_COPY", "1")
    S._INTERLEAVE_CACHE.clear()
    got = run(S.forward_sonic_moe)
    assert S._cached_bytes() == 0, "zero-copy path must not build the interleaved duplicate"
    got2 = run(S.forward_sonic_moe)
    for a, b, c in zip(got, ref, got2):
        rel = ((a.float() - b.float()).norm() / b.float().norm()).item()
        assert rel < 2e-2, rel
        assert torch.equal(a, c), "zero-copy fused path is not bitwise reproducible"


@pytest.mark.skipif(not torch.cuda.is_available() or S.triton is None, reason="needs CUDA + Triton")
def test_triton_swiglu_backward_matches_torch(monkeypatch):
    torch.manual_seed(0)
    pre = torch.randn(3001, 2 * 96, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(3001, 96, device="cuda", dtype=torch.bfloat16)
    got = S._swiglu_backward_interleaved(pre, g)
    monkeypatch.setenv("UNSLOTH_MOE_SONIC_TRITON_DGATED", "0")
    ref = S._swiglu_backward_interleaved(pre, g)
    torch.testing.assert_close(got.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_interleave_cache_is_tied_to_the_source_weight(monkeypatch):
    # data_ptr is reused after a model is freed: a dead or different owner must never hit.
    import gc

    monkeypatch.setenv("UNSLOTH_MOE_SONIC_CACHE_GIB", "1")
    S._INTERLEAVE_CACHE.clear()
    param = torch.randn(2, 8, 6)
    view = param.transpose(-2, -1).transpose(-2, -1)
    first = S._interleaved_gate_up(view)
    assert S._interleaved_gate_up(param.transpose(-2, -1).transpose(-2, -1)) is first
    key = next(iter(S._INTERLEAVE_CACHE))
    other = torch.randn(2, 8, 6)
    entry = S._INTERLEAVE_CACHE[key]
    S._INTERLEAVE_CACHE[key] = entry[:4] + (__import__("weakref").ref(other),)
    assert S._interleaved_gate_up(param.transpose(-2, -1).transpose(-2, -1)) is not first
    del param, view, first
    gc.collect()
    assert not S._INTERLEAVE_CACHE


def test_sonic_availability_probe_runs_outside_dynamo(monkeypatch):
    from unsloth_zoo.temporary_patches import moe_utils as MU

    seen = []
    monkeypatch.setattr(S, "sonic_moe_available", lambda: seen.append(torch.compiler.is_compiling()) or False)
    monkeypatch.delenv("UNSLOTH_DISABLE_MOE_SONIC", raising = False)

    @torch.compile(backend = "eager")
    def f(x):
        return x + int(MU._check_sonic_moe_available())

    f(torch.zeros(1))
    assert seen == [False]
