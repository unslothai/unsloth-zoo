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

import pytest
import torch


def _patched_activation():
    from unsloth_zoo.temporary_patches.misc import patch_relu_squared_activation_dtype
    patch_relu_squared_activation_dtype()
    from transformers.activations import ReLUSquaredActivation, ACT2FN
    assert isinstance(ACT2FN["relu2"], ReLUSquaredActivation)
    return ACT2FN["relu2"]


def test_patch_is_idempotent_and_marks_the_class():
    act = _patched_activation()
    from unsloth_zoo.temporary_patches.misc import patch_relu_squared_activation_dtype
    forward = type(act).forward
    patch_relu_squared_activation_dtype()
    assert type(act).forward is forward
    assert getattr(type(act), "_unsloth_dtype_patched", False) is True


def test_keeps_dtype_outside_autocast_and_matches_square():
    act = _patched_activation()
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        x = torch.randn(64, 32, dtype = dtype) * 4
        out = act(x)
        assert out.dtype == dtype
        assert torch.equal(out, torch.square(torch.relu(x)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "autocast on CUDA")
def test_bf16_under_autocast_stays_bf16_and_equals_the_rounded_reference():
    act = _patched_activation()
    x = (torch.randn(2048, 512, device = "cuda", dtype = torch.bfloat16) * 4).requires_grad_()
    with torch.autocast("cuda", dtype = torch.bfloat16):
        out = act(x)
    assert out.dtype == torch.bfloat16
    reference = torch.square(torch.relu(x.detach().float())).to(torch.bfloat16)
    assert torch.equal(out.detach(), reference)
    out.float().sum().backward()
    assert torch.equal(x.grad, (2 * torch.relu(x.detach())).to(x.grad.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "autocast on CUDA")
def test_nemotron_h_style_combine_no_longer_fails_under_autocast():
    act = _patched_activation()
    up = torch.nn.Linear(32, 64, bias = False).cuda().to(torch.bfloat16)
    down = torch.nn.Linear(64, 32, bias = False).cuda().to(torch.bfloat16)
    x = torch.randn(16, 32, device = "cuda", dtype = torch.bfloat16)
    router_w = torch.randn(16, 1, device = "cuda", dtype = torch.bfloat16)
    with torch.autocast("cuda", dtype = torch.bfloat16):
        out = torch.zeros_like(x, dtype = router_w.dtype)
        expert = down(act(up(x))) * router_w
        out.index_add_(0, torch.arange(16, device = "cuda"), expert)
    assert out.dtype == torch.bfloat16
