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

"""ROCm gfx11: float32 gated delta leaves fla's kernels for transformers' torch reference.

On a Radeon 8060S (gfx1151, Triton 3.6 / ROCm 7.13) fla's float32 chunk kernels fault
(chunk_gated_delta_rule_fwd_kernel_h_blockdim64: illegal memory access) or hang
(prepare_wy_repr_bwd_kernel); bfloat16 / float16 are correct. These tests run on CPU:
the arch gate, the routing (float32 only, never on CUDA), and the torch path's numerics
against an fp64 recurrence, forward and backward.
"""

import os
import sys
import types

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")

torch = pytest.importorskip("torch")
try:
    from unsloth_zoo.temporary_patches import fla_vendor
except ImportError as _e:
    pytest.skip(f"unsloth_zoo unavailable: {_e}", allow_module_level = True)


def _fake_torch(hip, archs, available = True):
    def props(i):
        return types.SimpleNamespace(gcnArchName = archs[i])

    return types.SimpleNamespace(
        version = types.SimpleNamespace(hip = "7.13.99004" if hip else None),
        cuda = types.SimpleNamespace(
            is_available = lambda: available,
            device_count = lambda: len(archs),
            get_device_properties = props,
        ),
    )


def test_gfx11_is_detected_on_rocm_only():
    detect = fla_vendor._rocm_has_fp32_faulting_gpu
    for arch in ("gfx1151", "gfx1151:xnack-", "gfx1100", "gfx1103", "gfx1150"):
        assert detect(_fake_torch(True, [arch])) is True, arch
    assert detect(_fake_torch(True, ["gfx942", "gfx1151"])) is True
    for arch in ("gfx1030", "gfx1201", "gfx942", "gfx90a", "gfx1010", ""):
        assert detect(_fake_torch(True, [arch])) is False, arch
    # CUDA builds, no accelerator, no devices, or a torch that raises: never.
    assert detect(_fake_torch(False, ["gfx1151"])) is False
    assert detect(_fake_torch(True, ["gfx1151"], available = False)) is False
    assert detect(_fake_torch(True, [])) is False
    assert detect(types.SimpleNamespace(version = None)) is False


def _recurrent_fp64(q, k, v, g, beta, scale, h0 = None):
    """Gated delta rule, one token at a time in fp64. q/k (B,T,H,K), v (B,T,HV,V), g/beta (B,T,HV)."""
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
    groups = v.shape[2] // q.shape[2]
    q = q.repeat_interleave(groups, 2)
    k = k.repeat_interleave(groups, 2)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim = True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim = True) + 1e-6)
    B, T, H, K = q.shape
    S = torch.zeros(B, H, K, v.shape[-1], dtype = torch.float64) if h0 is None else h0.double()
    out = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        u = beta[:, t][..., None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], S))
        S = S + k[:, t][..., :, None] * u[..., None, :]
        out.append(scale * torch.einsum("bhk,bhkv->bhv", q[:, t], S))
    return torch.stack(out, 1), S


def _inputs(B, T, H, HV, K, V, seed = 0, dtype = torch.float32):
    gen = torch.Generator().manual_seed(seed)
    q, k = torch.randn(B, T, H, K, generator = gen), torch.randn(B, T, H, K, generator = gen)
    v = torch.randn(B, T, HV, V, generator = gen)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, HV, generator = gen)) * 0.5
    beta = torch.rand(B, T, HV, generator = gen)
    return [x.to(dtype).requires_grad_() for x in (q, k, v, g, beta)]


@pytest.mark.parametrize("T", [6, 64, 70])
def test_torch_path_matches_fp64_forward_and_backward(T):
    """Grouped value heads (HV = 2H), a non-default scale, an initial state and the final state."""
    pytest.importorskip("transformers.models.qwen3_next.modeling_qwen3_next")
    B, H, HV, K, V = 2, 2, 4, 16, 8
    leaves = _inputs(B, T, H, HV, K, V)
    h0 = torch.randn(B, HV, K, V, generator = torch.Generator().manual_seed(3)) * 0.1
    scale = 0.3
    o, ht = fla_vendor._rocm_fp32_chunk_gated_delta_rule(
        *leaves, scale = scale, initial_state = h0, output_final_state = True, use_qk_l2norm_in_kernel = True,
    )
    ref_leaves = [x.detach().double().requires_grad_() for x in leaves]
    o_ref, ht_ref = _recurrent_fp64(*ref_leaves, scale, h0)
    assert o.dtype == torch.float32 and o.shape == (B, T, HV, V)
    torch.testing.assert_close(o.double(), o_ref, atol = 2e-5, rtol = 1e-4)
    torch.testing.assert_close(ht.double(), ht_ref, atol = 2e-5, rtol = 1e-4)
    do = torch.randn(o.shape, generator = torch.Generator().manual_seed(5))
    (o * do).sum().backward()
    (o_ref * do.double()).sum().backward()
    for mine, ref in zip(leaves, ref_leaves):
        torch.testing.assert_close(mine.grad.double(), ref.grad, atol = 5e-5, rtol = 1e-3)


def test_torch_path_varlen_and_state_v_first():
    """cu_seqlens: each packed sequence restarts from its own state; V-first state layout."""
    pytest.importorskip("transformers.models.qwen3_next.modeling_qwen3_next")
    H, K, V = 2, 16, 8
    lengths = [5, 0, 67, 3]
    cu = torch.tensor([0, 5, 5, 72, 75])
    q, k, v, g, beta = _inputs(1, sum(lengths), H, H, K, V, seed = 1)
    h0 = torch.randn(len(lengths), H, K, V, generator = torch.Generator().manual_seed(4)) * 0.1
    o, ht = fla_vendor._rocm_fp32_chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state = h0.transpose(-1, -2), output_final_state = True,
        use_qk_l2norm_in_kernel = True, state_v_first = True, cu_seqlens = cu,
    )
    assert o.shape == v.shape and ht.shape == (len(lengths), H, V, K)
    bounds = cu.tolist()
    for i, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        if e == s:  # an empty sequence keeps its initial state
            torch.testing.assert_close(ht[i].transpose(-1, -2), h0[i])
            continue
        o_ref, ht_ref = _recurrent_fp64(
            *(x.detach()[:, s:e] for x in (q, k, v, g, beta)), K ** -0.5, h0[i : i + 1]
        )
        torch.testing.assert_close(o[:, s:e].double(), o_ref, atol = 2e-5, rtol = 1e-4)
        torch.testing.assert_close(ht[i : i + 1].transpose(-1, -2).double(), ht_ref, atol = 2e-5, rtol = 1e-4)


def test_torch_path_honours_the_deprecated_transpose_state_layout():
    """fla's deprecated `transpose_state_layout` means `state_v_first`; passing both is an error."""
    pytest.importorskip("transformers.models.qwen3_next.modeling_qwen3_next")
    H, K, V = 2, 16, 8
    q, k, v, g, beta = _inputs(1, 37, H, H, K, V, seed = 2)
    h0 = torch.randn(1, H, K, V, generator = torch.Generator().manual_seed(5)) * 0.1
    common = dict(output_final_state = True, use_qk_l2norm_in_kernel = True)
    o_new, ht_new = fla_vendor._rocm_fp32_chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state = h0.transpose(-1, -2), state_v_first = True, **common
    )
    o_old, ht_old = fla_vendor._rocm_fp32_chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state = h0.transpose(-1, -2), transpose_state_layout = True, **common
    )
    assert ht_old.shape == (1, H, V, K)
    torch.testing.assert_close(o_old, o_new)
    torch.testing.assert_close(ht_old, ht_new)
    with pytest.raises(ValueError):
        fla_vendor._rocm_fp32_chunk_gated_delta_rule(
            q, k, v, g, beta, initial_state = h0.transpose(-1, -2), state_v_first = True,
            transpose_state_layout = True, **common
        )


def _install_fake_fla(monkeypatch):
    """A live fla with the three names the installer touches, plus a modeling module bound to it."""
    calls = []

    def chunk_gated_delta_rule(q, k, v, g, beta, scale = None, initial_state = None,
                               output_final_state = False, use_qk_l2norm_in_kernel = False, **kwargs):
        calls.append(("fla", q.dtype))
        return "fla"

    def chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state, output_final_state, **kwargs):
        calls.append(("fwd", q.dtype))
        return "fwd"

    def chunk_gated_delta_rule_bwd(q, k, v, g, beta, A, scale, initial_state, do, dht, **kwargs):
        calls.append(("bwd", q.dtype))
        return "bwd"

    package = types.ModuleType("fla.ops.gated_delta_rule")
    chunk = types.ModuleType("fla.ops.gated_delta_rule.chunk")
    package.chunk_gated_delta_rule = chunk.chunk_gated_delta_rule = chunk_gated_delta_rule
    chunk.chunk_gated_delta_rule_fwd = chunk_gated_delta_rule_fwd
    chunk.chunk_gated_delta_rule_bwd = chunk_gated_delta_rule_bwd
    modeling = types.ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")
    modeling.chunk_gated_delta_rule = chunk_gated_delta_rule
    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule", package)
    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule.chunk", chunk)
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_5.modeling_qwen3_5", modeling)
    return package, chunk, modeling, calls


def test_cuda_and_other_archs_leave_fla_untouched(monkeypatch):
    package, chunk, modeling, _ = _install_fake_fla(monkeypatch)
    before = (package.chunk_gated_delta_rule, chunk.chunk_gated_delta_rule_fwd, chunk.chunk_gated_delta_rule_bwd)
    for fake in (_fake_torch(False, ["gfx1151"]), _fake_torch(True, ["gfx942"]), _fake_torch(True, ["gfx1201"])):
        assert fla_vendor._install_rocm_fp32_torch_path(fake) is False
    after = (package.chunk_gated_delta_rule, chunk.chunk_gated_delta_rule_fwd, chunk.chunk_gated_delta_rule_bwd)
    assert all(a is b for a, b in zip(before, after))
    assert modeling.chunk_gated_delta_rule is before[0]


def test_gfx11_routes_only_float32_to_torch(monkeypatch):
    package, chunk, modeling, calls = _install_fake_fla(monkeypatch)
    original = package.chunk_gated_delta_rule
    assert fla_vendor._install_rocm_fp32_torch_path(_fake_torch(True, ["gfx1151"])) is True
    wrapped = package.chunk_gated_delta_rule
    assert wrapped is not original
    assert chunk.chunk_gated_delta_rule is wrapped and modeling.chunk_gated_delta_rule is wrapped
    # Idempotent: a second pass does not wrap again.
    assert fla_vendor._install_rocm_fp32_torch_path(_fake_torch(True, ["gfx1151"])) is True
    assert package.chunk_gated_delta_rule is wrapped

    # CPU tensors stand in for a gfx11 device: the per-call check is the only thing faked.
    monkeypatch.setattr(
        fla_vendor, "_rocm_fp32_on_faulting_gpu",
        lambda *ts: any(isinstance(t, torch.Tensor) and t.dtype == torch.float32 for t in ts),
    )
    seen = []
    monkeypatch.setattr(fla_vendor, "_rocm_fp32_chunk_gated_delta_rule", lambda *a, **k: seen.append(k) or "torch")
    for dtype in (torch.bfloat16, torch.float16):
        q, k, v, g, beta = _inputs(1, 4, 2, 2, 8, 8, dtype = dtype)
        assert wrapped(q, k, v, g = g.float(), beta = beta, use_qk_l2norm_in_kernel = True) == "fla"
    q, k, v, g, beta = _inputs(1, 4, 2, 2, 8, 8)
    assert wrapped(q, k, v, g = g, beta = beta, initial_state = None, output_final_state = True,
                   use_qk_l2norm_in_kernel = True, cu_seqlens = None) == "torch"
    assert [c for c in calls if c[0] == "fla"] == [("fla", torch.bfloat16), ("fla", torch.float16)]
    assert seen[0]["output_final_state"] is True and seen[0]["use_qk_l2norm_in_kernel"] is True

    # Internals refuse float32 before any launch; other dtypes pass through.
    with pytest.raises(RuntimeError, match = "float32"):
        chunk.chunk_gated_delta_rule_fwd(q = q, k = k, v = v, g = g, beta = beta, scale = 1.0,
                                         initial_state = None, output_final_state = False)
    with pytest.raises(RuntimeError, match = "float32"):
        chunk.chunk_gated_delta_rule_bwd(q, k, v, g, beta, None, 1.0, None, v, None)
    qb, kb, vb = (x.detach().bfloat16() for x in (q, k, v))
    assert chunk.chunk_gated_delta_rule_fwd(q = qb, k = kb, v = vb, g = g, beta = beta, scale = 1.0,
                                            initial_state = None, output_final_state = False) == "fwd"


def test_per_call_check_is_false_off_rocm(monkeypatch):
    """A CUDA build answers False before reading any device: NVIDIA routing never changes."""
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    x = torch.zeros(2, dtype = torch.float32)
    assert fla_vendor._rocm_fp32_on_faulting_gpu(x, x, x) is False
