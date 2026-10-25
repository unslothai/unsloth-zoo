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

"""fp16 full finetuning of an MoE on a GPU without bfloat16 (T4, V100), grouped_mm backend.

The expert stacks are float32 masters and the Trainer runs fp16 autocast, so fp16
activations reach torch._grouped_mm, which autocast does not cast:
"expected mat1 and mat2 to have the same dtype, but got: c10::Half != float".
Runs on CPU; equal dtypes (LoRA, bf16) must stay bit-for-bit unchanged.
"""

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils


@pytest.fixture(autouse = True)
def native_grouped_mm(monkeypatch):
    """Take the real torch._grouped_mm on CPU (the probe says no there and the per-group
    matmul fallback would be cast by CPU autocast, hiding the GPU failure)."""
    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("torch._grouped_mm unavailable")
    monkeypatch.setattr(moe_utils, "_grouped_mm_supported_on", lambda device: True)
    monkeypatch.setattr(moe_utils, "_transposed_view_safe_on", lambda device: True)
    monkeypatch.setattr(moe_utils, "_triton_grouped_mm_wanted", lambda inputs, weight: False)


def _case(x_dtype, w_dtype, seed = 0):
    torch.manual_seed(seed)
    experts, hidden, out = 3, 16, 24
    x = torch.randn(12, hidden).to(x_dtype)
    w = torch.randn(experts, hidden, out).mul_(0.1).to(w_dtype).requires_grad_(True)
    offsets = torch.tensor([4, 8, 12], dtype = torch.int32)
    return x, w, offsets


def _reference(x, w, offsets, dtype):
    starts = [0] + offsets.tolist()[:-1]
    return torch.cat([x[s:e].to(dtype) @ w[i].to(dtype) for i, (s, e) in enumerate(zip(starts, offsets.tolist()))])


def _cpu_half_autocast_or_skip():
    try:
        with torch.autocast("cpu", dtype = torch.float16):
            pass
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"cpu fp16 autocast unavailable: {exc}")


def test_fp16_activation_meets_fp32_master_under_fp16_autocast():
    _cpu_half_autocast_or_skip()
    x, w, offsets = _case(torch.float16, torch.float32)
    with torch.autocast("cpu", dtype = torch.float16):
        out = moe_utils._grouped_mm_with_backward_fix(x, w, offsets)
    # Autocast semantics, as F.linear: the GEMM runs in fp16.
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), _reference(x, w.detach(), offsets, torch.float16).float(), rtol = 2e-2, atol = 2e-2)
    out.float().sum().backward()
    # The fp32 master still receives its gradient, in float32.
    assert w.grad is not None and w.grad.dtype == torch.float32 and torch.isfinite(w.grad).all()


def test_mixed_dtypes_without_autocast_promote_to_the_weight():
    x, w, offsets = _case(torch.float16, torch.float32)
    out = moe_utils._grouped_mm_with_backward_fix(x, w, offsets)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, _reference(x, w.detach(), offsets, torch.float32))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_matching_dtypes_are_bitwise_unchanged(dtype):
    x, w, offsets = _case(dtype, dtype)
    expected = moe_utils._grouped_mm_eager(x, w.detach(), offsets)
    out = moe_utils._grouped_mm_with_backward_fix(x, w.detach(), offsets)
    assert out.dtype == dtype
    assert torch.equal(out, expected)
