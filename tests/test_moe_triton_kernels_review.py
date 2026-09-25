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

"""Triton MoE kernels: saved tensors, cold fullgraph compile, gategrad cache."""
import os

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")


def _inputs(T = 64, top_k = 4, H = 96, dtype = torch.bfloat16, w_requires_grad = True):
    g = torch.Generator(device = "cpu").manual_seed(0)
    y = torch.randn(T * top_k, H, generator = g).to("cuda", dtype).requires_grad_(True)
    idx = torch.randperm(T * top_k, generator = g).to("cuda")
    w = torch.rand(T * top_k, generator = g).to("cuda", torch.float32)
    w = w.requires_grad_(w_requires_grad)
    return y, idx, w


@cuda
def test_detached_routing_weights_do_not_pin_the_expert_output():
    from unsloth_zoo.temporary_patches.moe_triton_kernels import _WeightedUnpermute, moe_triton_kernels_available
    if not moe_triton_kernels_available(torch.device("cuda")):
        pytest.skip("Triton MoE kernels unavailable here")
    y, idx, w = _inputs(dtype = torch.float32, w_requires_grad = False)
    out = _WeightedUnpermute.apply(y, idx, w, 64, 4, None)
    node = out.grad_fn
    saved = [t for t in (getattr(node, "saved_tensors", None) or ()) if isinstance(t, torch.Tensor)]
    assert not any(t.data_ptr() == y.data_ptr() for t in saved), "y was saved although no weight gradient is needed"
    out.sum().backward()
    torch.testing.assert_close(y.grad.float(), w.unsqueeze(-1).expand_as(y).float(), rtol = 0, atol = 0)


@cuda
def test_trainable_routing_weights_still_get_their_gradient():
    from unsloth_zoo.temporary_patches.moe_triton_kernels import weighted_unpermute, moe_triton_kernels_available
    if not moe_triton_kernels_available(torch.device("cuda")):
        pytest.skip("Triton MoE kernels unavailable here")
    y, idx, w = _inputs(dtype = torch.float32)
    out = weighted_unpermute(y, idx, w, 64, 4)
    assert out is not None
    out.sum().backward()
    torch.testing.assert_close(w.grad, y.detach().sum(-1), rtol = 1e-5, atol = 1e-5)


_COLD = r"""
import torch
import unsloth_zoo.temporary_patches.moe_triton_kernels as k
if not k.moe_triton_kernels_available(torch.device("cuda")):
    print("SKIP"); raise SystemExit(0)
assert k._K is not None, "kernels must be defined at import, not on first use"
g = torch.Generator(device = "cpu").manual_seed(0)
y = torch.randn(256, 96, generator = g).cuda().requires_grad_(True)
idx = torch.randperm(256, generator = g).cuda()
w = torch.rand(256, generator = g).cuda()
def f(y, idx, w):
    return k._WeightedUnpermute.apply(y, idx, w, 64, 4, None)
out = torch.compile(f, fullgraph = True)(y, idx, w)      # first ever call: cold
ref = torch.zeros(64, 96, device = "cuda")
ref.index_add_(0, idx // 4, y * w.unsqueeze(-1))
torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-5)
print("COLD_OK")
"""


@cuda
def test_cold_fullgraph_compile_does_not_trace_the_kernel_definitions():
    """Needs a fresh interpreter: any earlier use in this process would have built the kernels already."""
    import subprocess, sys
    out = subprocess.run([sys.executable, "-c", _COLD], capture_output = True, text = True,
                         env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    if "SKIP" in out.stdout:
        pytest.skip("Triton MoE kernels unavailable here")
    assert "COLD_OK" in out.stdout, out.stderr[-3000:]


def test_gategrad_switch_is_cached_as_documented(monkeypatch):
    from unsloth_zoo.temporary_patches import moe_utils
    assert hasattr(moe_utils._moe_gategrad_enabled, "cache_clear"), "_moe_gategrad_enabled lost its lru_cache"
    moe_utils._moe_gategrad_enabled.cache_clear()
    monkeypatch.setenv("UNSLOTH_MOE_GATEGRAD", "0")
    assert moe_utils._moe_gategrad_enabled() is False
    monkeypatch.setenv("UNSLOTH_MOE_GATEGRAD", "1")
    assert moe_utils._moe_gategrad_enabled() is False   # cached: documented behaviour
    moe_utils._moe_gategrad_enabled.cache_clear()
    assert moe_utils._moe_gategrad_enabled() is True
