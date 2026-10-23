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

"""Float32 activations into bf16 experts: the Triton MoE path runs post-gate_up GEMMs in bf16."""

import sys
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from unsloth_zoo.temporary_patches import moe_utils
from unsloth_zoo.temporary_patches.moe_utils import forward_triton_grouped_gemm

CALLS = []
TUNED = []


def _triton_like_grouped_gemm(
    X=None,
    W=None,
    m_sizes=None,
    topk=None,
    gather_indices=None,
    permute_x=False,
    permute_y=False,
    is_first_gemm=None,
    **kwargs,
):
    # The Triton kernels: operands rounded to W's dtype, float32 accumulation, output in X's dtype.
    CALLS.append(("gate_up" if is_first_gemm else "down", X.dtype))
    if permute_x:
        X = X[gather_indices // topk]
    outputs, start = [], 0
    for expert_idx, size in enumerate(m_sizes.tolist()):
        end = start + int(size)
        x = X[start:end].to(W.dtype).float()
        outputs.append((x @ W[expert_idx].float().T).to(X.dtype))
        start = end
    Y = torch.cat(outputs, dim = 0)
    if permute_y:
        Y = Y[torch.argsort(gather_indices)]
    return Y


def _autotune(**kw):
    TUNED.append(kw["dtype"])
    return (None, None, None)


@pytest.fixture
def setup(monkeypatch):
    parent = None
    for name in ("unsloth", "unsloth.kernels", "unsloth.kernels.moe", "unsloth.kernels.moe.grouped_gemm"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        if parent is not None:
            monkeypatch.setattr(parent, name.rsplit(".", 1)[1], module, raising = False)
        parent = module
    interface = types.ModuleType("unsloth.kernels.moe.grouped_gemm.interface")
    interface.grouped_gemm = _triton_like_grouped_gemm
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    monkeypatch.setattr(parent, "interface", interface, raising = False)
    autotune_cache = types.ModuleType("unsloth.kernels.moe.autotune_cache")
    autotune_cache.get_or_autotune_moe_kernels = _autotune
    monkeypatch.setitem(sys.modules, autotune_cache.__name__, autotune_cache)
    monkeypatch.setattr(sys.modules["unsloth.kernels.moe"], "autotune_cache", autotune_cache, raising = False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.delenv("UNSLOTH_MOE_TRITON_FP32", raising = False)
    monkeypatch.delenv("UNSLOTH_FORCE_FLOAT32", raising = False)
    CALLS.clear()
    TUNED.clear()
    for key in getattr(moe_utils, "TRITON_MOE_COMPUTE_DTYPE_STATS", {}):
        moe_utils.TRITON_MOE_COMPUTE_DTYPE_STATS[key] = 0
    return monkeypatch


def _experts(dtype, num_experts = 4, hidden = 16, intermediate = 8):
    torch.manual_seed(0)
    experts = nn.Module()
    experts.num_experts = num_experts
    experts.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * intermediate, hidden).to(dtype))
    experts.down_proj = nn.Parameter(torch.randn(num_experts, hidden, intermediate).to(dtype))
    experts.act_fn = F.silu
    return experts


def _inputs(dtype, num_experts = 4, hidden = 16, tokens = 12, top_k = 2):
    torch.manual_seed(1)
    X = torch.randn(tokens, hidden, dtype = dtype, requires_grad = True)
    top_k_index = torch.randint(0, num_experts, (tokens, top_k))
    top_k_weights = torch.softmax(torch.randn(tokens, top_k), dim = -1)
    return X, top_k_index, top_k_weights


def _run(experts, dtype):
    X, idx, w = _inputs(dtype)
    out = forward_triton_grouped_gemm(experts, X, idx, w)
    out.float().square().sum().backward()
    return out, X.grad, experts.gate_up_proj.grad, experts.down_proj.grad


def test_float32_activations_run_both_gemms_in_bf16(setup):
    out, dX, dgu, ddn = _run(_experts(torch.bfloat16), torch.float32)
    assert CALLS == [("gate_up", torch.bfloat16), ("down", torch.bfloat16)]
    # The cast is autograd-tracked: dX and the routed output stay float32.
    assert out.dtype == torch.float32 and dX.dtype == torch.float32
    assert dgu.dtype == torch.bfloat16 and ddn.dtype == torch.bfloat16
    assert TUNED == [torch.bfloat16, torch.bfloat16]
    assert moe_utils.TRITON_MOE_COMPUTE_DTYPE_STATS == {"cast": 1, "kept_fp32": 0}


def test_kill_switch_keeps_float32(setup):
    setup.setenv("UNSLOTH_MOE_TRITON_FP32", "1")
    out, dX, _, _ = _run(_experts(torch.bfloat16), torch.float32)
    assert CALLS == [("gate_up", torch.float32), ("down", torch.float32)]
    assert out.dtype == torch.float32 and dX.dtype == torch.float32
    assert TUNED == [torch.float32, torch.float32]
    assert moe_utils.TRITON_MOE_COMPUTE_DTYPE_STATS == {"cast": 0, "kept_fp32": 1}


def test_bf16_matches_transformers_under_autocast(setup):
    experts = _experts(torch.bfloat16)
    X, idx, w = _inputs(torch.float32)
    ours = forward_triton_grouped_gemm(experts, X, idx, w)
    ref = torch.zeros_like(X)
    for t in range(X.shape[0]):
        for k in range(idx.shape[1]):
            e = int(idx[t, k])
            gate, up = (X[t].to(torch.bfloat16) @ experts.gate_up_proj[e].T).chunk(2, dim = -1)
            h = F.silu(gate) * up
            ref[t] += (h @ experts.down_proj[e].T).float() * w[t, k]
    torch.testing.assert_close(ours, ref, rtol = 2e-2, atol = 2e-2)


@pytest.mark.parametrize(
    "weight_dtype,act_dtype,env",
    [
        (torch.bfloat16, torch.bfloat16, None),
        (torch.float16, torch.float32, None),
        (torch.float32, torch.float32, None),
        (torch.bfloat16, torch.float32, "UNSLOTH_FORCE_FLOAT32"),
    ],
)
def test_other_dtypes_unchanged(setup, weight_dtype, act_dtype, env):
    if env:
        setup.setenv(env, "1")
    on = _run(_experts(weight_dtype), act_dtype)
    calls_on = list(CALLS)
    CALLS.clear()
    setup.setenv("UNSLOTH_MOE_TRITON_FP32", "1")
    off = _run(_experts(weight_dtype), act_dtype)
    assert calls_on == CALLS == [("gate_up", act_dtype), ("down", act_dtype)]
    for a, b in zip(on, off):
        assert a.dtype == b.dtype and torch.equal(a, b)
    assert moe_utils.TRITON_MOE_COMPUTE_DTYPE_STATS["cast"] == 0
