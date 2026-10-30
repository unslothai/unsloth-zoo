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
"""TiledMLP backward with an MLP whose backward writes dX into its input (CPU only).

Unsloth's fused LoRA MLP computes dX with `torch.matmul(..., out = X)`. Under per-block compile the
tiled MLP input is a compiled region's output: a non-view tensor that requires grad. TiledMLP's
backward then made no_grad views of it, and AccumulateGrad raised "A view was created in no_grad
mode and its base ... modified inplace with grad mode enabled".
"""

import pytest
import torch

from unsloth_zoo.tiled_mlp import patch_mlp


class _InplaceDXLinear(torch.autograd.Function):
    """y = x @ W.T; backward writes dX into x's storage, like fast_lora.LoRA_MLP (inplace = True)."""

    @staticmethod
    def forward(ctx, X, W):
        ctx.save_for_backward(X, W)
        return X @ W.t()

    @staticmethod
    def backward(ctx, dY):
        X, W = ctx.saved_tensors
        dW = dY.reshape(-1, dY.shape[-1]).t() @ X.reshape(-1, X.shape[-1])
        dX = torch.matmul(dY, W, out = X)
        return dX, dW
pass


class _InplaceMLP(torch.nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden, hidden) / hidden**0.5)

    def forward(self, x):
        return _InplaceDXLinear.apply(x, self.weight)
pass


def _run(tiled, bsz, qlen, hidden):
    torch.manual_seed(0)
    mlp = _InplaceMLP(hidden)
    if tiled:
        patch_mlp(mlp, target_arctic = True)
    leaf = torch.randn(bsz, qlen, hidden, requires_grad = True)
    # Non-view, requires grad, has a grad_fn: what a compiled decoder block hands the MLP.
    x = leaf * 1.5
    assert x._base is None and x.grad_fn is not None
    out = mlp(x)
    (out.float() * torch.linspace(-1, 1, hidden)).sum().backward()
    return out.detach(), leaf.grad, mlp.weight.grad
pass


@pytest.mark.parametrize(("bsz", "qlen"), [(1, 32), (2, 24)])
def test_tiled_backward_with_inplace_dx_matches_untiled(bsz, qlen):
    hidden = 8  # arctic chunk size = hd, so several shards
    ref_out, ref_dx, ref_dw = _run(False, bsz, qlen, hidden)
    out, dx, dw = _run(True, bsz, qlen, hidden)
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(dx, ref_dx)
    torch.testing.assert_close(dw, ref_dw)
pass
