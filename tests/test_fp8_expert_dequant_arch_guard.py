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

"""FP8 expert dequant must skip the fp8e4nv Triton kernel where unsloth's guard says it cannot compile
(pre-sm89 CUDA; AMD archs without fp8e4nv on triton < 3.5), falling back to torch dequant."""

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


def _block_scaled(E, M, N, device):
    g = torch.Generator(device = "cpu").manual_seed(0)
    w = (torch.randn(E, M, N, generator = g) * 0.5).to(torch.float8_e4m3fn).to(device)
    s = (torch.rand(E, M // 128, N // 128, generator = g) + 0.5).to(device)
    return w, s


def test_vectorized_block_dequant_in_chunks_matches_one_shot(monkeypatch):
    w, s = _block_scaled(5, 256, 384, "cpu")
    ref = w.to(torch.bfloat16) * s.to(torch.bfloat16).repeat_interleave(128, 1).repeat_interleave(128, 2)
    monkeypatch.setattr(F, "_VECTORIZED_DEQUANT_CHUNK_BYTES", 2 * 2 * 256 * 384 * 2)  # 2 experts per chunk, 2 temporaries each
    torch.testing.assert_close(F._dequantize_full_expert_weights_vectorized(w, s, torch.bfloat16), ref, rtol = 0, atol = 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA for the peak memory check")
def test_vectorized_block_dequant_peak_stays_near_the_result(monkeypatch):
    # Whole-stack version peaked at ~3x the result size.
    w, s = _block_scaled(32, 1024, 1024, "cuda")
    monkeypatch.setattr(F, "_VECTORIZED_DEQUANT_CHUNK_BYTES", 4 * 1024 * 1024 * 2)
    result_bytes = 32 * 1024 * 1024 * 2
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = F._dequantize_full_expert_weights_vectorized(w, s, torch.bfloat16)
    torch.cuda.synchronize()
    assert out.shape == (32, 1024, 1024)
    assert torch.cuda.max_memory_allocated() - base < 1.5 * result_bytes
    # Both chunk temporaries together stay within the budget (small repeat_interleave intermediates aside).
    assert torch.cuda.max_memory_allocated() - base - result_bytes <= 1.25 * F._VECTORIZED_DEQUANT_CHUNK_BYTES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_real_device_full_expert_dequant_matches_reference():
    # Unmocked: whichever path the guard picks on this GPU must run and match the float32 reference.
    _fp8_module()
    w, s = _expert_stack()
    out = F._dequantize_full_expert_weights(w, s, torch.bfloat16)
    torch.cuda.synchronize()
    ref = w.float() * s.repeat_interleave(128, 1).repeat_interleave(128, 2)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


def test_block_scaled_dequant_converts_each_expert_once(monkeypatch):
    w, s = _block_scaled(4, 256, 256, "cpu")
    seen = []
    real_to = torch.Tensor.to

    def counting_to(self, *args, **kwargs):
        if self.dtype == torch.float8_e4m3fn:
            seen.append(self.shape[0])
        return real_to(self, *args, **kwargs)

    real_copy = torch.Tensor.copy_

    def counting_copy(self, src, *args, **kwargs):
        if src.dtype == torch.float8_e4m3fn:
            seen.append(src.shape[0])
        return real_copy(self, src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", counting_to)
    monkeypatch.setattr(torch.Tensor, "copy_", counting_copy)
    F._dequantize_full_expert_weights_vectorized(w, s, torch.bfloat16)
    monkeypatch.undo()
    assert sum(seen) == 4, seen


def test_block_dequant_accepts_scales_that_require_grad():
    # FP8Experts holds its scales as Parameters; the dequant must run under autograd like main's mul_.
    w, s = _block_scaled(3, 256, 256, "cpu")
    out = F._dequantize_full_expert_weights_vectorized(w, torch.nn.Parameter(s), torch.bfloat16)
    ref = w.to(torch.bfloat16) * s.to(torch.bfloat16).repeat_interleave(128, 1).repeat_interleave(128, 2)
    torch.testing.assert_close(out.detach(), ref, rtol = 0, atol = 0)
