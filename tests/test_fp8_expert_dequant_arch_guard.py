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

"""FP8 MoE expert dequant must not launch the fp8e4nv Triton kernel on a GPU
that cannot compile it.

unsloth.kernels.fp8.weight_dequant_block loads float8_e4m3fn (triton fp8e4nv).
Triton rejects that dtype on pre-sm89 CUDA and on ROCm archs without OCP fp8
(gfx90a, gfx11xx such as Strix Halo gfx1151), and ROCm reports a gfx-derived
capability (gfx1151 -> 11.5), so transformers keeps the fp8 weights there. The
unsloth helper must consult unsloth's own guard and decline, so the caller's
vectorized torch fallback dequantizes with identical values.
"""

from __future__ import annotations

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils_fp8 as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _fp8_module():
    try:
        from unsloth.kernels import fp8
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"unsloth.kernels.fp8 is not importable ({type(e).__name__}: {e})")
    if not hasattr(fp8, "_fp8_kernel_unsupported"):
        pytest.skip("unsloth predates the fp8 arch guard")
    return fp8


def _expert_stack():
    torch.manual_seed(0)
    w = (torch.randn(2, 256, 256, device=DEV) * 0.4).to(torch.float8_e4m3fn)
    s = torch.rand(2, 2, 2, device=DEV, dtype=torch.float32) + 0.5
    return w, s


def test_unsupported_device_declines_and_fallback_dequantizes(monkeypatch):
    fp8 = _fp8_module()

    def _no_kernel(*a, **k):
        raise AssertionError("fp8e4nv triton kernel launched on a device that cannot compile it")

    monkeypatch.setattr(fp8, "weight_dequant_block", _no_kernel)
    monkeypatch.setattr(fp8, "_fp8_kernel_unsupported", lambda *a, **k: True)
    w, s = _expert_stack()
    assert F._dequantize_full_expert_weights_unsloth(w, s, torch.bfloat16) is None

    out = F._dequantize_full_expert_weights(w, s, torch.bfloat16)
    assert out is not None
    # Exactly the existing vectorized fallback, and the right values to bf16 precision.
    assert torch.equal(out, F._dequantize_full_expert_weights_vectorized(w, s, torch.bfloat16))
    ref = w.float() * s.repeat_interleave(128, 1).repeat_interleave(128, 2)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


def test_supported_device_keeps_the_triton_kernel(monkeypatch):
    fp8 = _fp8_module()
    calls = []

    def _stub(x, s, block_size=128, dtype=torch.bfloat16):
        calls.append(block_size)
        return torch.zeros(x.shape, dtype=dtype, device=x.device)

    monkeypatch.setattr(fp8, "weight_dequant_block", _stub)
    monkeypatch.setattr(fp8, "_fp8_kernel_unsupported", lambda *a, **k: False)
    w, s = _expert_stack()
    assert F._dequantize_full_expert_weights_unsloth(w, s, torch.bfloat16) is not None
    assert calls == [128]
