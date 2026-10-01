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

"""Guards: gpt-oss GRPO `target_frame.early_stop is set` (MoE combine swallowed checkpoint early stop)."""
import pytest
import torch
from torch.utils import checkpoint

import unsloth_zoo.temporary_patches.moe_triton_kernels as K

_STOP = getattr(checkpoint, "_StopRecomputationError", None)


@pytest.fixture
def kernels(monkeypatch):
    monkeypatch.setattr(K, "_DISABLED_REASON", None)
    monkeypatch.setattr(K, "moe_triton_kernels_available", lambda device = None: True)
    return K


def _cpu_args():
    return torch.randn(8, 4), torch.arange(8), torch.rand(8), 4, 2


@pytest.mark.skipif(_STOP is None, reason = "this torch has no checkpoint early stop")
def test_checkpoint_early_stop_signal_propagates(kernels, monkeypatch):
    def stop(*args):
        raise _STOP()
    monkeypatch.setattr(kernels._WeightedUnpermute, "apply", stop)
    with pytest.raises(_STOP):
        kernels.weighted_unpermute(*_cpu_args())
    assert kernels._DISABLED_REASON is None


def test_kernel_failure_still_falls_back(kernels, monkeypatch):
    def fail(*args):
        raise RuntimeError("launch failed")
    monkeypatch.setattr(kernels._WeightedUnpermute, "apply", fail)
    assert kernels.weighted_unpermute(*_cpu_args()) is None
    assert "launch failed" in kernels._DISABLED_REASON


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_combine_under_non_reentrant_checkpoint_matches_plain_backward(monkeypatch):
    monkeypatch.setattr(K, "_DISABLED_REASON", None)
    if not K.moe_triton_kernels_available(torch.device("cuda")):
        pytest.skip("Triton MoE kernels unavailable here")
    T, top_k, H = 64, 4, 128
    g = torch.Generator(device = "cpu").manual_seed(0)
    x = torch.randn(T * top_k, H, generator = g).to("cuda", torch.bfloat16)
    up = torch.randn(H, H, generator = g).to("cuda", torch.bfloat16)
    idx = torch.randperm(T * top_k, generator = g).to("cuda")
    w = torch.rand(T * top_k, generator = g).to("cuda", torch.bfloat16)

    def region(x, up, w):
        # The combine's save_for_backward is the region's last save, where early stop fires.
        return K.weighted_unpermute(x @ up, idx, w, T, top_k, out_dtype = torch.bfloat16)

    def grads(checkpointed):
        leaves = [t.clone().requires_grad_(True) for t in (x, up, w)]
        out = checkpoint.checkpoint(region, *leaves, use_reentrant = False) if checkpointed else region(*leaves)
        out.float().pow(2).sum().backward()
        return [t.grad for t in leaves]

    for got, ref in zip(grads(True), grads(False)):
        torch.testing.assert_close(got, ref, rtol = 0, atol = 0)
    assert K._DISABLED_REASON is None
