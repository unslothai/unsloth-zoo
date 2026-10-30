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

"""unsloth_gradient_checkpoint stays opaque to Dynamo: under a caller's torch.compile it is one graph break, not a
traced region that breaks again inside (the decorator once moved to a helper, and breaks went from 1 to 3)."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from _sequence_tiling_models import GatedMLP, H  # noqa: E402
from unsloth_zoo.gradient_checkpointing import unsloth_gradient_checkpoint  # noqa: E402


def test_user_compile_around_unsloth_gradient_checkpoint_breaks_once():
    from torch._dynamo.testing import CompileCounterWithBackend
    from torch._dynamo.utils import counters
    m = GatedMLP()

    def f(x):
        h = x * 2
        return unsloth_gradient_checkpoint(m, h) + 1

    torch._dynamo.reset()
    counters.clear()
    cnt = CompileCounterWithBackend("aot_eager")
    x = torch.randn(2, 8, H, requires_grad = True)
    y = torch.compile(f, backend = cnt)(x)
    y.sum().backward()
    breaks = sum(counters["graph_break"].values())
    assert breaks <= 1, dict(counters["graph_break"])
    xr = x.detach().clone().requires_grad_()
    torch.testing.assert_close(y, f(xr))
