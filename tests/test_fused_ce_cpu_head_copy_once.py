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

"""Fused CE off CUDA, half inputs: the dH GEMMs multiply an fp32 copy of the head made once per call, not once per
row chunk; loss and hidden gradient still match an fp64 reference."""

from __future__ import annotations

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from unsloth_zoo.fused_losses import fused_linear as fl


class _HeadCopies(TorchDispatchMode):
    # fp32 copies read from the head's storage (whole head or a vocab slice), in head-sized units.
    def __init__(self, w):
        super().__init__()
        self.ptr, self.numel, self.elements = w.untyped_storage().data_ptr(), w.numel(), 0

    @property
    def count(self):
        return self.elements / self.numel

    def __torch_dispatch__(self, func, types, args = (), kwargs = None):
        kwargs = kwargs or {}
        if func is torch.ops.aten._to_copy.default and kwargs.get("dtype") == torch.float32 \
                and args[0].untyped_storage().data_ptr() == self.ptr:
            self.elements += args[0].numel()
        return func(*args, **kwargs)


def _impl_args(h, w, t, rows, need_h = True):
    return (h, w, None, t, torch.tensor(float((t != -100).sum())), torch.ones(()), -100, 0.0, 0.0, 0.0, 0.0, None,
            h.dtype, need_h, False, False, False, rows)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [64, 100])
def test_head_copied_to_fp32_once_per_call(dtype, rows):
    torch.manual_seed(0)
    N, H, V = 512, 64, 1003  # V not a multiple of 8: the aligned split takes two GEMMs per chunk
    h = (torch.randn(N, H) * 0.5).to(dtype)
    w = (torch.randn(V, H) * 0.5).to(dtype)
    t = torch.randint(0, V, (N,))
    t[::7] = -100
    with _HeadCopies(w) as copies:
        loss, grad_h, _, _ = fl._ce_impl(*_impl_args(h, w, t, rows))
    n_chunks = -(-int((t != -100).sum()) // rows)
    assert n_chunks > 1
    assert copies.count == 1, (copies.count, n_chunks)

    # fp64 reference of the same mean CE and its hidden gradient.
    h64 = h.double().requires_grad_(True)
    keep = t != -100
    ref = torch.nn.functional.cross_entropy(h64 @ w.double().t(), t, ignore_index = -100, reduction = "sum") / keep.sum()
    (g64,) = torch.autograd.grad(ref, h64)
    torch.testing.assert_close(loss.double(), ref.detach(), rtol = 1e-2, atol = 1e-2)
    torch.testing.assert_close(grad_h.double(), g64, rtol = 2e-2, atol = 2e-3)


def test_no_copy_without_hidden_grad_or_with_one_chunk():
    torch.manual_seed(1)
    h = torch.randn(64, 32).bfloat16()
    w = torch.randn(257, 32).bfloat16()
    t = torch.randint(0, 257, (64,))
    with _HeadCopies(w) as copies:
        fl._ce_impl(*_impl_args(h, w, t, 16, need_h = False))
    assert copies.count == 0
    with _HeadCopies(w) as copies:
        fl._ce_impl(*_impl_args(h, w, t, 64))
    assert copies.count <= 1
