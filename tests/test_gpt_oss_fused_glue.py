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
"""gpt-oss fused MoE glue: dispatch (kill switch, dtype, device), the eager references the GPU self-check compares
against equal the chain grouped_qlora_forward ships, and the combine fallback keeps its bits. CPU; the kernels
themselves are covered on GPU by tests/test_gpt_oss_grouped_qlora.py."""
import os
import types

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth_zoo.temporary_patches import gpt_oss_fused_glue as glue
from unsloth_zoo.temporary_patches import gpt_oss_grouped_qlora as gq
from unsloth_zoo.temporary_patches.moe_utils import combine_permuted_moe_outputs

ALPHA, LIMIT = 1.702, 7.0


def _bits(a, b):
    iv = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(iv), b.view(iv))


def _swiglu(a, alpha, limit):
    # gpt_oss.swiglu_torch_forward's body, uncompiled.
    g = a[..., ::2].to(torch.float32).clamp(max = limit)
    u = a[..., 1::2].to(torch.float32).clamp(min = -limit, max = limit)
    return (g * torch.sigmoid(alpha * g)) * (u + 1)


def _case(M = 29, N = 48, H = 40, E = 4, seed = 0):
    g = torch.Generator().manual_seed(seed)
    eid = torch.randint(0, E, (M,), generator = g).sort().values
    gu = (torch.randn(M, 2 * N, generator = g) * 4).to(torch.bfloat16)
    gu[0, :6] = torch.tensor([LIMIT, -LIMIT, 0.0, -0.0, 2 * LIMIT, -2 * LIMIT]).to(torch.bfloat16)
    d = torch.randn(M, 2 * N, generator = g).to(torch.bfloat16)
    bias = (torch.randn(E, 2 * N, generator = g) * 0.5).to(torch.bfloat16)
    y = torch.randn(M, H, generator = g).to(torch.bfloat16)
    d2 = torch.randn(M, H, generator = g).to(torch.bfloat16)
    b2 = torch.randn(E, H, generator = g).to(torch.bfloat16)
    return eid, gu, d, bias, y, d2, b2


@pytest.mark.parametrize("acc_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lora", [True, False])
def test_eager_references_equal_the_shipped_chain(acc_dtype, lora):
    eid, gu, d, bias, y, d2, b2 = _case()
    scaling = 2.0 if lora else 1.0
    leaves = [gu.clone().requires_grad_(), d.clone().requires_grad_()]
    # grouped_qlora_forward before the fused glue: `gate_up.to(acc) + bias[e].to(acc) + _lora_delta(...).to(acc)`,
    # where _lora_delta = gmm(h, B) * scaling.
    x = leaves[0].to(acc_dtype) + bias[eid].to(acc_dtype)
    if lora:
        x = x + (leaves[1] * scaling).to(acc_dtype)
    shipped = _swiglu(x, ALPHA, LIMIT).to(torch.bfloat16)
    mine = [gu.clone().requires_grad_(), d.clone().requires_grad_()]
    ref = glue._eager_gate_up(mine[0], mine[1] if lora else None, bias, eid, scaling, ALPHA, LIMIT, acc_dtype)
    assert _bits(ref, shipped)
    dy = torch.randn(ref.shape).to(torch.bfloat16)
    want = torch.autograd.grad(shipped, leaves[: 1 + lora], dy)
    got = torch.autograd.grad(ref, mine[: 1 + lora], dy)
    assert all(_bits(a, b) for a, b in zip(got, want))

    yy = [y.clone().requires_grad_(), d2.clone().requires_grad_()]
    out = yy[0].float() + b2[eid].float()
    if lora:
        out = out + (yy[1] * scaling).float()
    mm = [y.clone().requires_grad_(), d2.clone().requires_grad_()]
    ref2 = glue._eager_down(mm[0], mm[1] if lora else None, b2, eid, scaling)
    assert _bits(ref2, out)
    g = torch.randn(out.shape)
    assert all(_bits(a, b) for a, b in zip(torch.autograd.grad(ref2, mm[: 1 + lora], g),
                                            torch.autograd.grad(out, yy[: 1 + lora], g)))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_combine_fallback_keeps_the_shipped_bits(dtype):
    T, k, E, H = 13, 4, 6, 40
    g = torch.Generator().manual_seed(1)
    router = torch.stack([torch.randperm(E, generator = g)[:k] for _ in range(T)])
    flat = router.flatten()
    sorted_idx = flat.argsort(stable = True)
    sorted_tokens = torch.arange(T).repeat_interleave(k)[sorted_idx]
    expert_ids = flat[sorted_idx]
    rw = torch.rand(T, E, generator = g).to(dtype)
    y = torch.randn(T * k, H, generator = g)
    leaves = [y.clone().requires_grad_(), rw.clone().requires_grad_()]
    shipped = combine_permuted_moe_outputs(
        leaves[0] * leaves[1][sorted_tokens, expert_ids, None].to(torch.float32), sorted_idx, T, k, out_dtype = torch.float32)
    mine = [y.clone().requires_grad_(), rw.clone().requires_grad_()]
    got = gq._combine(mine[0], mine[1], sorted_tokens, expert_ids, sorted_idx, T, k)
    assert _bits(got, shipped)
    up = torch.randn(T, H, generator = g)
    assert all(_bits(a, b) for a, b in zip(torch.autograd.grad(got, mine, up), torch.autograd.grad(shipped, leaves, up)))


def _experts(limit = LIMIT):
    return types.SimpleNamespace(alpha = ALPHA, limit = limit)


@pytest.mark.parametrize("env", [None, "1", "0"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("limit", [LIMIT, None])
def test_dispatch_gates(env, dtype, limit, monkeypatch):
    if env is None:
        monkeypatch.delenv("UNSLOTH_GPTOSS_FUSED_GLUE", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_GPTOSS_FUSED_GLUE", env)
    t = torch.zeros(2, 4, dtype = dtype)
    # Real gate on CPU: never fused.
    assert gq._fused_glue(t, torch.bfloat16, _experts(limit)) is None
    assert not glue.fused_glue_enabled(t)
    # With the device / self-check gate forced open, only the kill switch, dtype and limit decide.
    monkeypatch.setattr(glue, "usable", lambda *a: True)
    want = env != "0" and dtype is torch.bfloat16 and limit is not None
    assert (gq._fused_glue(t, torch.bfloat16, _experts(limit)) is glue) == want
    # fp16 accumulate is never fused.
    assert gq._fused_glue(t, torch.float16, _experts(limit)) is None


def test_failed_launch_falls_back_once_and_reraises_oom(monkeypatch):
    monkeypatch.setitem(glue._STATE, "failed", False)
    calls = dict(glue.FUSED_GLUE_CALLS)

    def boom(*a, **k):
        raise RuntimeError("no kernel")

    monkeypatch.setattr(glue, "_launch_gate_up", boom)
    eid, gu, d, bias, *_ = _case()
    with pytest.warns(UserWarning, match = "fused gpt-oss MoE glue failed"):
        assert glue.gate_up_epilogue(gu, d, bias, eid, 1.0, ALPHA, LIMIT, torch.bfloat16) is None
    assert glue._STATE["failed"] and glue.FUSED_GLUE_CALLS["fallback"] == calls["fallback"] + 1
    assert not glue.fused_glue_enabled(gu)

    def oom(*a, **k):
        raise torch.cuda.OutOfMemoryError("oom")

    monkeypatch.setattr(glue, "_launch_gate_up", oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        glue.gate_up_epilogue(gu, d, bias, eid, 1.0, ALPHA, LIMIT, torch.bfloat16)


# ---- Review fixes: the autograd / fallback contract with the launches emulated on CPU (eager, same bits). ----
import contextlib

from torch.utils import checkpoint as _ckpt


def _emu_gate_up(gu, bias, eid, d, scaling, alpha, limit, acc_dtype, write_t2):
    for t in (gu, bias, eid) + ((d,) if d is not None else ()):
        assert t.stride(-1) == 1, "kernel launched on a non unit-stride tensor"
    x = gu.to(acc_dtype) + bias[eid].to(acc_dtype)
    if d is not None:
        x = x + (d * scaling).to(acc_dtype)
    return glue._swiglu_eager(x, alpha, limit), (x.detach() if write_t2 else None)


def _emu_gate_up_bwd(dy, t2, scaling, alpha, limit, has_d):
    x = t2.detach().requires_grad_()
    with torch.enable_grad():
        (dx,) = torch.autograd.grad(glue._swiglu_eager(x, alpha, limit), x, dy)
    dgu = dx.to(torch.bfloat16)
    return dgu, (dgu * scaling if has_d else None)


class _EmuDownKernel:
    def __getitem__(self, grid):
        def launch(y, bias, eid, d, out, M, H, s_y, s_d, s_b, s_o, scaling, HAS_D, **kw):
            out.copy_(glue._eager_down(y, d if HAS_D else None, bias, eid, scaling))
        return launch


@pytest.fixture
def emulated(monkeypatch):
    monkeypatch.setattr(glue, "_down_fwd_kernel", _EmuDownKernel(), raising = False)
    monkeypatch.setitem(glue._STATE, "failed", False)
    monkeypatch.setattr(glue, "_CHECKED", {})
    monkeypatch.setattr(glue, "_launch_gate_up", _emu_gate_up)
    monkeypatch.setattr(glue, "_launch_gate_up_bwd", _emu_gate_up_bwd)
    monkeypatch.setattr(torch.cuda, "device", lambda d: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return monkeypatch


@pytest.mark.parametrize("acc_dtype", [torch.bfloat16, torch.float32])
def test_gate_double_backward_matches_eager(acc_dtype, emulated):
    eid, gu, d, bias, *_ = _case(M = 9, N = 8)
    gu = gu.float().clamp(-6, 6).to(torch.bfloat16)  # keep the second derivative off the clamp edges
    dy = torch.randn(9, 8).to(torch.bfloat16)
    v = [torch.randn(9, 16).to(torch.bfloat16) for _ in range(2)]

    def run(fn):
        leaves = [gu.clone().requires_grad_(), d.clone().requires_grad_()]
        g = torch.autograd.grad(fn(leaves), leaves, dy, create_graph = True)
        s = sum((a.float() * b.float()).sum() for a, b in zip(g, v))
        return g, torch.autograd.grad(s, leaves)

    g_ref, gg_ref = run(lambda L: glue._eager_gate_up(L[0], L[1], bias, eid, 0.5, ALPHA, LIMIT, acc_dtype))
    g_f, gg_f = run(lambda L: glue.gate_up_epilogue(L[0], L[1], bias, eid, 0.5, ALPHA, LIMIT, acc_dtype))
    assert all(_bits(a, b) for a, b in zip(g_f, g_ref))
    assert any(t.abs().sum() > 0 for t in gg_ref)
    assert all(_bits(a, b) for a, b in zip(gg_f, gg_ref))


def test_gate_backward_launch_failure_falls_back_to_eager(emulated):
    def boom(*a, **k):
        raise RuntimeError("bwd compile")

    emulated.setattr(glue, "_launch_gate_up_bwd", boom)
    eid, gu, d, bias, *_ = _case()
    dy = torch.randn(gu.shape[0], gu.shape[1] // 2).to(torch.bfloat16)
    L = [gu.clone().requires_grad_(), d.clone().requires_grad_()]
    R = [gu.clone().requires_grad_(), d.clone().requires_grad_()]
    with pytest.warns(UserWarning, match = "glue failed"):
        got = torch.autograd.grad(glue.gate_up_epilogue(L[0], L[1], bias, eid, 0.5, ALPHA, LIMIT, torch.float32), L, dy)
    want = torch.autograd.grad(glue._eager_gate_up(R[0], R[1], bias, eid, 0.5, ALPHA, LIMIT, torch.float32), R, dy)
    assert all(_bits(a, b) for a, b in zip(got, want))


def _stop():
    cls = getattr(_ckpt, "_StopRecomputationError", None) or getattr(_ckpt, "CheckpointError")
    return cls()


@pytest.mark.parametrize("exc", [_stop, lambda: torch.cuda.OutOfMemoryError("oom")])
def test_wrappers_reraise_control_flow_and_oom(exc, emulated):
    def raiser(*a, **k):
        raise exc()

    eid, gu, d, bias, y, d2, b2 = _case()
    emulated.setattr(glue, "_launch_gate_up", raiser)
    with pytest.raises(type(exc())):
        glue.gate_up_epilogue(gu, d, bias, eid, 1.0, ALPHA, LIMIT, torch.bfloat16)
    emulated.setattr(glue, "_grid", raiser)
    with pytest.raises(type(exc())):
        glue.down_epilogue(y, d2, b2, eid, 1.0)
    assert not glue._STATE["failed"]
    # The self-check propagates them too and caches nothing.
    with pytest.raises(type(exc())):
        glue._self_check(torch.device("cpu"), torch.bfloat16, ALPHA, LIMIT)
    assert glue._CHECKED == {}


def test_self_check_declines_temporary_contexts_without_caching(emulated):
    dev = torch.device("cpu")
    emulated.setattr(torch.compiler, "is_compiling", lambda: True)
    assert glue._self_check(dev, torch.bfloat16, ALPHA, LIMIT) is False
    emulated.setattr(torch.compiler, "is_compiling", lambda: False)

    def no_hooks(*a, **k):
        raise RuntimeError("saved tensor hooks are disabled")

    real = torch.autograd.graph.saved_tensors_hooks
    emulated.setattr(torch.autograd.graph, "saved_tensors_hooks", no_hooks)
    assert glue._self_check(dev, torch.bfloat16, ALPHA, LIMIT) is False
    assert glue._CHECKED == {}
    emulated.setattr(torch.autograd.graph, "saved_tensors_hooks", real)
    before = dict(glue.FUSED_GLUE_CALLS)
    assert glue._self_check(dev, torch.bfloat16, ALPHA, LIMIT) is True
    assert glue.FUSED_GLUE_CALLS == before


def test_failed_self_check_restores_counters(emulated):
    def boom(*a, **k):
        raise RuntimeError("probe launch")

    emulated.setattr(glue, "_grid", boom)  # the down probe fails after the gate probe counted its calls
    before = dict(glue.FUSED_GLUE_CALLS)
    with pytest.warns(UserWarning, match = "self-check failed"):
        assert glue._self_check(torch.device("cpu"), torch.float32, ALPHA, LIMIT) is False
    assert glue.FUSED_GLUE_CALLS == before


def test_wrappers_normalize_strides(emulated):
    eid, gu, d, bias, y, d2, b2 = _case()
    wide = torch.randn(gu.shape[0], 2 * gu.shape[1]).to(torch.bfloat16)
    d_strided = wide[:, ::2]
    d_strided.copy_(d)
    eid_strided = torch.stack([eid, eid], 1)[:, 0]
    assert d_strided.stride(-1) == 2 and eid_strided.stride(0) == 2
    got = glue.gate_up_epilogue(gu, d_strided, bias, eid_strided, 0.5, ALPHA, LIMIT, torch.bfloat16)
    assert _bits(got, glue._eager_gate_up(gu, d, bias, eid, 0.5, ALPHA, LIMIT, torch.bfloat16))
    seen = {}

    def capture(*args):
        seen["args"] = args
        return glue._eager_down(*args)

    emulated.setattr(glue._DownEpilogue, "apply", capture)
    wide2 = torch.randn(y.shape[0], 2 * y.shape[1]).to(torch.bfloat16)
    wide2[:, ::2] = d2
    glue.down_epilogue(y, wide2[:, ::2], b2, eid_strided, 0.5)
    assert all(t.stride(-1) == 1 for t in seen["args"][:4])


def test_wrappers_reject_unsupported_inputs(emulated):
    eid, gu, d, bias, y, d2, b2 = _case()
    calls = glue.FUSED_GLUE_CALLS["fallback"]
    # Odd width (the kernel would drop the last column), trainable bias (would get no grad), mismatched shapes.
    assert glue.gate_up_epilogue(gu[:, :-1], None, bias[:, :-1], eid, 1.0, ALPHA, LIMIT, torch.bfloat16) is None
    assert glue.gate_up_epilogue(gu, d, bias.clone().requires_grad_(), eid, 1.0, ALPHA, LIMIT, torch.bfloat16) is None
    assert glue.gate_up_epilogue(gu, d[:, :4], bias, eid, 1.0, ALPHA, LIMIT, torch.bfloat16) is None
    assert glue.gate_up_epilogue(gu, d, bias.float(), eid, 1.0, ALPHA, LIMIT, torch.bfloat16) is None
    assert glue.down_epilogue(y, d2, b2.clone().requires_grad_(), eid, 1.0) is None
    assert glue.down_epilogue(y, d2, b2, eid[:-1], 1.0) is None
    assert glue.down_epilogue(y.float(), None, b2, eid, 1.0) is None
    assert glue.FUSED_GLUE_CALLS["fallback"] == calls + 7
    assert not glue._STATE["failed"]
    assert not glue.usable(gu, torch.float16, ALPHA, LIMIT)


def test_weighted_unpermute_create_graph_uses_the_differentiable_branch(monkeypatch):
    from unsloth_zoo.temporary_patches import moe_triton_kernels as K
    monkeypatch.setattr(K, "_DISABLED_REASON", None)
    monkeypatch.setattr(K, "moe_triton_kernels_available", lambda device = None: True)

    def no_kernels():
        raise AssertionError("raw kernels under create_graph=True")

    monkeypatch.setattr(K, "_get_kernels", no_kernels)
    T, k, H = 3, 2, 4
    y = torch.randn(T * k, H, requires_grad = True)
    w = torch.rand(T * k, requires_grad = True)
    sidx = torch.randperm(T * k)
    ctx = types.SimpleNamespace(needs_w = True, saved_tensors = (y, sidx, w), y_meta = ((T, H), y.dtype, y.device),
                                top_k = k, round_product = False)
    dout = torch.randn(T, H, requires_grad = True)
    with torch.enable_grad():
        dy, _, dw, *_ = K._WeightedUnpermute.backward(ctx, dout)
    assert dy.requires_grad and dw.requires_grad and K._DISABLED_REASON is None
    (gw,) = torch.autograd.grad(dw.sum(), w, allow_unused = True)
    assert gw is None or torch.equal(gw, torch.zeros_like(w))
    assert torch.autograd.grad(dw.sum(), y)[0].abs().sum() > 0


def test_moe_triton_disable_reraises_oom(monkeypatch):
    from unsloth_zoo.temporary_patches import moe_triton_kernels as K
    monkeypatch.setattr(K, "_DISABLED_REASON", None)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        K._disable("weighted_unpermute", torch.cuda.OutOfMemoryError("oom"))
    assert K._DISABLED_REASON is None
