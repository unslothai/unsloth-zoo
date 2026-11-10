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
