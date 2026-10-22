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

"""moe_triton_kernels_available() must not write its global (graph break) inside a traced autograd.Function."""

import pytest
import torch

M = pytest.importorskip("unsloth_zoo.temporary_patches.moe_triton_kernels")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")


class _WithProvider(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, provider):
        ctx.provider = provider
        with torch.no_grad():
            w = provider()
        return x * w

    @staticmethod
    def backward(ctx, g):
        return g * ctx.provider(), None


def _body(t):
    provider = lambda: torch.full(  # noqa: E731
        (4,), 2.0 if M.moe_triton_kernels_available(t.device) else 3.0, device = t.device
    )
    return _WithProvider.apply(t, provider)


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("grad", [True, False])
def test_first_probe_inside_traced_autograd_function_is_fullgraph(monkeypatch, grad, checkpointed):
    monkeypatch.setattr(M, "_TRITON", None)
    monkeypatch.setattr(M, "_DISABLED_REASON", None)
    torch._dynamo.reset()

    def fn(t):
        if checkpointed:
            return torch.utils.checkpoint.checkpoint(_body, t, use_reentrant = False)
        return _body(t)

    x = torch.ones(4, device = "cuda", requires_grad = grad)
    with torch.enable_grad() if grad else torch.no_grad():
        out = torch.compile(fn, backend = "eager", fullgraph = True)(x)
    assert M._TRITON is None  # nothing written while tracing
    expected = M.moe_triton_kernels_available(x.device)  # the eager probe still caches
    assert M._TRITON is expected
    assert torch.equal(out.detach(), torch.full((4,), 2.0 if expected else 3.0, device = "cuda"))
