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

"""CPU chunked CE, fp32 head under bf16 / fp16 autocast: one head cast per call, not per chunk, same bits."""

from __future__ import annotations

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from unsloth_zoo.cpu import ops


class _HeadCasts(TorchDispatchMode):

    def __init__(self, w):
        super().__init__()
        self.ptr, self.numel, self.elements = w.untyped_storage().data_ptr(), w.numel(), 0

    @property
    def count(self):
        return self.elements / self.numel

    def __torch_dispatch__(self, func, types, args = (), kwargs = None):
        kwargs = kwargs or {}
        if func is torch.ops.aten._to_copy.default and kwargs.get("dtype") in (torch.bfloat16, torch.float16) \
                and args[0].untyped_storage().data_ptr() == self.ptr:
            self.elements += args[0].numel()
        return func(*args, **kwargs)


def _case(hidden_dtype, seed = 0, N = 300, H = 64, V = 2003):
    torch.manual_seed(seed)
    w = torch.randn(V, H) * 0.05
    h = torch.randn(N, H).to(hidden_dtype)
    y = torch.randint(0, V, (N,))
    y[::7] = -100
    return h, w, y


@pytest.mark.parametrize("amp_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("need_weight_grad", [False, True])
def test_fp32_head_cast_once_same_bits(amp_dtype, hidden_dtype, need_weight_grad):
    h, w, y = _case(hidden_dtype)
    rows = 64
    exact = amp_dtype == torch.float16
    assert -(-h.shape[0] // rows) > 1
    with _HeadCasts(w) as casts:
        new = ops._chunked_ce_forward_impl(h, w, y, rows, -100, True, need_weight_grad, exact, True, amp_dtype)
    assert casts.count == 1, casts.count
    # Reference: the per-chunk casts autocast does when the loop gets the fp32 weight itself.
    with torch.autocast("cpu", dtype = amp_dtype):
        with _HeadCasts(w) as casts:
            old = ops._chunked_ce_forward_loop(h, w, y, rows, -100, True, need_weight_grad, exact)
    assert casts.count > 1
    for a, b in zip(new, old):
        assert torch.equal(a, b)


@pytest.mark.parametrize("amp_dtype", [torch.bfloat16, torch.float16])
def test_autograd_loss_and_grads_match_reference(amp_dtype):
    h, w, y = _case(torch.bfloat16, seed = 1)
    hh, ww = h.clone().requires_grad_(), w.clone().requires_grad_()
    with torch.autocast("cpu", dtype = amp_dtype):
        loss = ops.chunked_linear_cross_entropy(hh, ww, y.unsqueeze(0), shift_labels = y.unsqueeze(0), chunk_rows = 50)
    loss.backward()
    assert ww.grad.dtype == torch.float32 and hh.grad.dtype == torch.bfloat16
    h64, w64 = h.double().requires_grad_(), w.double().requires_grad_()
    ref = torch.nn.functional.cross_entropy(h64 @ w64.t(), y, ignore_index = -100)
    ref.backward()
    torch.testing.assert_close(loss.double(), ref.detach(), rtol = 2e-2, atol = 2e-2)
    torch.testing.assert_close(hh.grad.double(), h64.grad, rtol = 5e-2, atol = 2e-3)
    torch.testing.assert_close(ww.grad.double(), w64.grad, rtol = 5e-2, atol = 2e-3)


def test_no_cast_without_autocast_or_for_a_low_precision_head():
    h, w, y = _case(torch.float32, seed = 2)
    with _HeadCasts(w) as casts:
        ops._chunked_ce_forward_impl(h, w, y, 64, -100, True, False, False, False, torch.bfloat16)
    assert casts.count == 0
    wb = w.bfloat16()
    with _HeadCasts(wb) as casts:
        ops._chunked_ce_forward_impl(h.bfloat16(), wb, y, 64, -100, True, False, False, True, torch.bfloat16)
    assert casts.count == 0
