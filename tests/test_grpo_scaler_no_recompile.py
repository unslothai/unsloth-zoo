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

"""A changing fp16 GradScaler scale must not recompile the GRPO chunk loop.

UnslothEfficientGRPO read GradScaler.get_scale(), a Python float, and passed it
straight into the compiled accumulate_chunk, where grad_and_value made dynamo
guard on its value (``scaling == 65536.0``). Each overflow halving and each
growth step built a new graph, and after the recompile limit the chunk loop
ran eager.
"""

import pytest
import torch
import torch._dynamo
from torch._dynamo.utils import counters

import unsloth_zoo.rl_replacements as rl


class _Scaler:
    def __init__(self, scale):
        self.scale = scale

    def get_scale(self):
        return self.scale


def _run(scale, dtype = torch.float32, seed = 0, bsz = 4, qlen = 16):
    g = torch.Generator().manual_seed(seed)
    new = (torch.randn(bsz, qlen, generator = g) * 0.1 - 1.0).to(dtype).requires_grad_(True)
    old = (new.detach().float() + torch.randn(bsz, qlen, generator = g) * 0.01).to(dtype)
    ref = (new.detach().float() + torch.randn(bsz, qlen, generator = g) * 0.01).to(dtype)
    input_ids = torch.randint(0, 100, (bsz, qlen), generator = g)
    mask = torch.ones(bsz, qlen, dtype = torch.int64)
    mask[:, -3:] = 0
    advantages = torch.randn(bsz, generator = g)
    scaler = None if scale is None else (scale if hasattr(scale, "get_scale") else _Scaler(scale))
    kwargs = dict(loss_type = "grpo", max_completion_length = qlen)
    loss, *_ = rl.UnslothEfficientGRPO.apply(
        new, old, ref, None, None, input_ids, mask, advantages, 0.04, scaler, 1, kwargs, None,
    )
    loss.backward()
    return loss.detach(), new.grad


def _unique_graphs():
    return counters["stats"].get("unique_graphs", 0)


@pytest.fixture(autouse = True)
def _fresh_dynamo():
    torch._dynamo.reset()
    counters.clear()
    yield
    torch._dynamo.reset()


def test_scale_changes_reuse_one_graph():
    _run(65536.0)
    first = _unique_graphs()
    assert first >= 1
    for scale in (32768.0, 16384.0, 8192.0, 131072.0, 3.0):
        _run(scale)
    assert _unique_graphs() == first, dict(counters["stats"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_scaled_gradient_is_unchanged(dtype):
    # GradScaler scales are powers of two, so the gradient is exactly scale times the
    # scaling = 1 gradient, and the returned loss is the unscaled float32 one.
    loss_1, grad_1 = _run(1.0, dtype, seed = 1)
    assert loss_1.dtype == torch.float32 and grad_1.dtype == dtype
    for scale in (1024.0, 0.5):
        loss_s, grad_s = _run(scale, dtype, seed = 1)
        assert torch.equal(loss_s, loss_1)
        assert grad_s.dtype == dtype
        assert torch.equal(grad_s / scale, grad_1)


def test_tensor_and_none_scale_still_work():
    loss_n, grad_n = _run(None, seed = 2)
    loss_t, grad_t = _run(_Scaler(torch.tensor(1.0)), seed = 2)
    torch.testing.assert_close(loss_t, loss_n)
    torch.testing.assert_close(grad_t, grad_n)
