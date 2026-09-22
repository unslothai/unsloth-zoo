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

"""A coarse FP8 expert scale derives a Triton tile larger than Triton allows.

weight_dequant_block takes one BLOCK_SIZE and materialises a
BLOCK_SIZE x BLOCK_SIZE tile, so BLOCK_SIZE is legal only up to 1024. The
block size is derived from the scale granularity, and a per-expert per-tensor
scale (p == q == 1) derives BLOCK_SIZE = M. At M = 4096 that is a 16777216
element tile and the kernel fails to compile, which is how
mistralai/Mistral-Small-4-119B-2603 raised
"numel (16777216) exceeds triton maximum tensor numel (1048576)".

The Triton helper must decline those layouts so the caller's vectorized
fallback answers. Finer scales must keep taking the Triton path unchanged.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils_fp8 as F

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _weight(E, M, N, device):
    return torch.zeros(E, M, N, device=device).to(torch.float8_e4m3fn)


@pytest.mark.parametrize("M,N", [(2048, 2048), (4096, 4096), (1088, 1088)])
def test_per_tensor_expert_scale_declines_instead_of_failing_to_compile(M, N):
    """p == q == 1 derives BLOCK_SIZE = M; above 1024 the tile is illegal.

    Decided before any kernel launch, so this needs no GPU. 1088 is the first
    multiple of 64 past the limit, which pins the boundary rather than a round
    number well clear of it.
    """
    limit = F._triton_max_tensor_numel()
    assert M * N > limit
    w = _weight(2, M, N, "cpu")
    s = torch.ones(2, 1, 1)
    assert F._dequantize_full_expert_weights_unsloth(w, s, torch.bfloat16) is None, (
        f"a {M}x{N} tile exceeds Triton's {limit} element limit but the Triton "
        f"path still claimed it"
    )


@requires_cuda
def test_a_tile_exactly_at_the_limit_is_still_taken():
    """1024 x 1024 is exactly the cap, so it is legal and must not be declined.
    The guard is `>`, and an off-by-one to `>=` would silently cost the fast
    path here."""
    M = N = 1024
    torch.manual_seed(0)
    w = (torch.randn(2, M, N, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    s = torch.full((2, 1, 1), 0.05, device="cuda", dtype=torch.float32)
    assert M * N == F._TRITON_MAX_TENSOR_NUMEL
    out = F._dequantize_full_expert_weights_unsloth(w, s, torch.float32)
    assert out is not None, "a legal tile exactly at Triton's limit was declined"
    assert torch.allclose(out, w.to(torch.float32) * 0.05, rtol=1e-6, atol=1e-8)


def test_the_limit_matches_tritons_own():
    """Read out of Triton, not asserted against our own literal, which would be
    a tautology. This is what answers the cap on a non-CUDA Triton build: the
    check lives in the Python frontend (triton/_utils.validate_block_shape), so
    a ROCm wheel reports the same value, and if one ever did not, this fails
    here rather than at kernel compile time on that hardware."""
    triton_language = pytest.importorskip("triton.language")
    theirs = getattr(triton_language, "TRITON_MAX_TENSOR_NUMEL", None)
    # Not `is None`: with no GPU visible, unsloth's import_fixes installs a Triton
    # stub whose every attribute is a placeholder object, so the name resolves to
    # something that is not a number. Comparing against it would fail here and,
    # worse, `tile > placeholder` in the resolver would raise TypeError on exactly
    # the machines that have no Triton, which is why it checks the type too.
    if type(theirs) is not int:
        pytest.skip("Triton is stubbed or does not export TRITON_MAX_TENSOR_NUMEL")
    assert F._triton_max_tensor_numel() == theirs


def test_the_fallback_is_used_when_triton_cannot_be_read(monkeypatch):
    """Absent or broken Triton must leave the literal in place rather than
    raising: the zoo imports on machines with no Triton at all (macOS arm64)."""
    monkeypatch.setattr(F, "_TRITON_MAX_TENSOR_NUMEL_RESOLVED", None)

    real_import_module = importlib.import_module

    def _no_triton(name, *args, **kwargs):
        if name.startswith("triton"):
            raise ImportError("no triton here")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _no_triton)
    assert F._triton_max_tensor_numel() == F._TRITON_MAX_TENSOR_NUMEL == 2 ** 20


@requires_cuda
@pytest.mark.parametrize("block", [128, 256, 512, 1024])
def test_block_scales_still_take_the_triton_path(block):
    """Every scale granularity that fits the tile limit must still reach the
    Triton kernel, so the guard cannot have narrowed the fast path. Returning
    None here would be a silent demotion to the slower fallback."""
    M = N = 2048
    torch.manual_seed(0)
    w = (torch.randn(2, M, N, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    s = torch.rand(2, M // block, N // block, device="cuda") * 0.1 + 0.01
    assert block * block <= F._TRITON_MAX_TENSOR_NUMEL

    out = F._dequantize_full_expert_weights_unsloth(w, s, torch.float32)
    assert out is not None, (
        f"a {block}-element block scale no longer reaches the Triton path"
    )
    vec = F._dequantize_full_expert_weights_vectorized(w, s, torch.float32)
    assert torch.allclose(out, vec, rtol=1e-5, atol=1e-6)


@requires_cuda
@pytest.mark.parametrize("M,N", [(4096, 4096), (2048, 4096)])
def test_caller_falls_back_and_returns_correct_values(M, N):
    """End to end: the public entry point must answer, and answer correctly."""
    torch.manual_seed(0)
    E = 2
    ref_hi = (torch.randn(E, M, N, device="cuda") * 0.1)
    w = ref_hi.to(torch.float8_e4m3fn)
    s = torch.full((E, 1, 1), 0.05, device="cuda", dtype=torch.float32)

    assert F._dequantize_full_expert_weights_unsloth(w, s, torch.bfloat16) is None

    out = F._dequantize_full_expert_weights(w, s, torch.bfloat16)
    assert out is not None, "no path answered a per-expert per-tensor FP8 scale"
    assert tuple(out.shape) == (E, M, N) and out.dtype == torch.bfloat16
    # The fallback multiplies in the target dtype, so it rounds once where a
    # float32 reference rounds twice; compare at bfloat16's own resolution
    # rather than pinning one multiplication order.
    expected = w.to(torch.float32) * s.to(torch.float32)
    assert torch.allclose(out.to(torch.float32), expected, rtol=8e-3, atol=1e-8), (
        f"max abs error {(out.to(torch.float32) - expected).abs().max().item()}"
    )


@requires_cuda
def test_finer_scales_still_go_through_triton_and_agree():
    """The Triton path must still be taken for a normal 128-block scale, and
    must agree with the vectorized maths."""
    torch.manual_seed(0)
    E, M, N, block = 2, 1024, 1024, 128
    w = (torch.randn(E, M, N, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    s = torch.rand(E, M // block, N // block, device="cuda") * 0.1 + 0.01

    triton_out = F._dequantize_full_expert_weights_unsloth(w, s, torch.float32)
    assert triton_out is not None, (
        "a 128-block scale no longer reaches the Triton path; the guard narrowed it"
    )
    vec_out = F._dequantize_full_expert_weights_vectorized(w, s, torch.float32)
    assert torch.allclose(triton_out, vec_out, rtol=1e-5, atol=1e-6)


def test_a_per_expert_scalar_scale_is_broadcast_not_materialized(monkeypatch):
    """The fallback this guard routes to must not expand an (E, 1, 1) scale to
    (E, M, N) first. At the cited Mistral layout that second full-size tensor
    measured an extra 4.00 GiB, which is how an OOM replaces the crash the
    guard was added to avoid.

    Asserted on the call rather than on peak memory: the caching allocator
    reuses freed blocks, so a small fixture can expand the scale and still show
    no rise, which is how the first version of this test passed with the fix
    reverted.
    """
    E, M, N = 4, 256, 512
    w = torch.zeros(E, M, N).to(torch.float8_e4m3fn)
    s = torch.full((E, 1, 1), 0.05)

    calls = []
    real = torch.Tensor.repeat_interleave
    monkeypatch.setattr(
        torch.Tensor, "repeat_interleave",
        lambda self, *a, **k: (calls.append(tuple(self.shape)), real(self, *a, **k))[1],
    )
    out = F._dequantize_full_expert_weights_vectorized(w, s, torch.bfloat16)

    assert out is not None and tuple(out.shape) == (E, M, N)
    assert not calls, (
        f"the (E, 1, 1) scale was expanded via repeat_interleave{calls}; it "
        f"broadcasts over (E, M, N) on its own"
    )


@requires_cuda
def test_broadcast_and_expansion_agree_exactly():
    """The shortcut must be a shortcut, not a different multiplication."""
    torch.manual_seed(0)
    E, M, N = 4, 256, 512
    w = (torch.randn(E, M, N, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    s = (torch.rand(E, 1, 1, device="cuda") * 0.1 + 0.01)

    got = F._dequantize_full_expert_weights_vectorized(w, s, torch.bfloat16)
    expanded = (
        s.to(torch.bfloat16)
        .repeat_interleave(M, dim=1)[:, :M, :]
        .repeat_interleave(N, dim=2)[:, :, :N]
    )
    assert torch.equal(got, w.to(torch.bfloat16) * expanded)
