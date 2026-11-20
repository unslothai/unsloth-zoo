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
"""The fused SwiGLU first-use self-check inside a non-reentrant gradient checkpoint (CPU + GPU)."""

import contextlib

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from unsloth_zoo.temporary_patches import moe_fused_swiglu as sw


def _clear_self_check_cache():
    for k in [k for k in sw._CACHE if isinstance(k, tuple) and k and k[0] == "self_check"]:
        del sw._CACHE[k]


def _autograd_bwd(grad, gate_up, act, interleaved, alpha, limit, eager_rounding):
    x = gate_up.clone().requires_grad_()
    y = sw.moe_swiglu_reference(x, act, interleaved, alpha, limit, eager_rounding)
    return torch.autograd.grad(y, x, grad)[0]


def _cpu_probe(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda d: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_self_check_inside_checkpoint_cpu(monkeypatch, use_reentrant):
    # The probe's kernels replaced by the eager expression: only its autograd interplay is under test.
    monkeypatch.setattr(sw, "_launch_fwd", sw.moe_swiglu_reference)
    monkeypatch.setattr(sw, "_launch_bwd", _autograd_bwd)
    _cpu_probe(monkeypatch)
    _clear_self_check_cache()
    W = torch.randn(64, 16, requires_grad = True)
    runs, results = [0], []

    def body(x):
        runs[0] += 1
        gu = x @ W.t()
        results.append(sw.fused_swiglu_self_check(torch.device("cpu"), torch.float32, sw.ACT_SILU, False))
        g, u = gu.chunk(2, dim = -1)
        return torch.nn.functional.silu(g) * u

    x = torch.randn(8, 16, requires_grad = True)
    y = checkpoint(body, x, use_reentrant = use_reentrant)
    assert runs[0] == 1, "the probe's grad recomputed the checkpointed body during the forward"
    y.sum().backward()
    assert results[0] is True
    assert W.grad is not None and x.grad is not None
    W2 = W.detach().clone().requires_grad_()
    x2 = x.detach().clone().requires_grad_()
    g, u = (x2 @ W2.t()).chunk(2, dim = -1)
    (torch.nn.functional.silu(g) * u).sum().backward()
    assert torch.equal(W.grad, W2.grad) and torch.equal(x.grad, x2.grad)
    _clear_self_check_cache()


def test_self_check_hooks_disabled_is_not_cached(monkeypatch):
    _cpu_probe(monkeypatch)
    _clear_self_check_cache()
    with torch.autograd.graph.disable_saved_tensors_hooks("no hooks here"):
        assert sw.fused_swiglu_self_check(torch.device("cpu"), torch.float32, sw.ACT_SILU, False) is False
    assert not any(isinstance(k, tuple) and k and k[0] == "self_check" for k in sw._CACHE)


def test_bf16_gate_is_per_operand_device(monkeypatch):
    # A visible sm75 card must not disable bf16 fusion on an sm90 operand device, and vice versa.
    import types

    caps = {0: (7, 5), 1: (9, 0)}
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i = None: caps[i])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(sw, "_CACHE", {})
    on = lambda d: types.SimpleNamespace(device = torch.device(d))
    assert sw._bf16_capable(on("cuda:1")) is True
    assert sw._bf16_capable(on("cuda:0")) is False
    assert sw._bf16_capable(on("cuda")) is True  # bare "cuda" = the current device


def _gpu_dtypes():
    if not torch.cuda.is_available():
        return []
    out = [torch.float32, torch.float16]
    if torch.cuda.get_device_capability()[0] >= 8:
        out.append(torch.bfloat16)
    return out


class Qwen3MoeExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act_fn = torch.nn.SiLU()


class GptOssExperts(torch.nn.Module):
    alpha, limit = 1.702, 7.0


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason = "NVIDIA CUDA required")
@pytest.mark.parametrize("dtype", _gpu_dtypes())
@pytest.mark.parametrize("expert_cls", [Qwen3MoeExperts, GptOssExperts])
@pytest.mark.parametrize("interleaved", [False, True])
def test_fused_moe_activation_non_reentrant_checkpoint(dtype, expert_cls, interleaved):
    from unsloth_zoo.temporary_patches import moe_utils as mu

    if not sw.fused_swiglu_supported(torch.empty(2, device = "cuda", dtype = dtype)):
        pytest.skip("fused SwiGLU unsupported here")
    _clear_self_check_cache()
    mod = expert_cls()
    torch.manual_seed(0)
    W0 = torch.randn(128, 32, device = "cuda", dtype = dtype)
    x0 = torch.randn(16, 32, device = "cuda", dtype = dtype)
    gy = torch.randn(16, 64, device = "cuda", dtype = dtype)

    def run(fused):
        W = W0.clone().requires_grad_()
        x = x0.clone().requires_grad_()
        runs = [0]

        def body(x):
            runs[0] += 1
            gu = x @ W.t()
            out = mu._fused_moe_activation(mod, gu, interleaved) if fused else None
            if out is None:
                act = sw.ACT_GPTOSS if expert_cls is GptOssExperts else sw.ACT_SILU
                out = sw.moe_swiglu_reference(gu, act, interleaved)
            return out

        before = list(sw.FUSED_SWIGLU_CALLS)
        y = checkpoint(body, x, use_reentrant = False)
        assert runs[0] == 1
        y.backward(gy)
        calls = [a - b for a, b in zip(sw.FUSED_SWIGLU_CALLS, before)]
        return y.detach(), x.grad, W.grad, calls

    for step in range(2):
        y_f, gx_f, gw_f, calls = run(True)
        # forward + recompute, then backward: the fused path stays engaged on every call.
        assert calls == [2, 1], (step, calls)
        y_e, gx_e, gw_e, _ = run(False)
        assert torch.equal(y_f, y_e) and torch.equal(gx_f, gx_e) and torch.equal(gw_f, gw_e)
    assert [v for k, v in sw._CACHE.items() if isinstance(k, tuple) and k[0] == "self_check"] == [True]
    _clear_self_check_cache()
