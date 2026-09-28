# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Tier 0 LoRA merge correctness tests for unsloth_zoo/saving_utils.py.

Run on Linux+CUDA/XPU without an MLX shim. Cover _active_merge_device(), _merge_lora
(base merge, vocab-resize, non-finite guard), and the 5 MoE expert-merge variants
against a numpy reference.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unsloth_zoo.device_type import DEVICE_TYPE_TORCH
from unsloth_zoo.saving_utils import (
    LoraStats,
    _active_merge_device,
    _merge_lora,
    _merge_moe_down_proj_expert,
    _merge_moe_fused_down_proj_expert,
    _merge_moe_fused_gate_up_expert,
    _merge_moe_gate_expert,
    _merge_moe_up_expert,
)

# conftest sets UNSLOTH_ALLOW_CPU=1, so DEVICE_TYPE_TORCH says "cuda" even with no GPU: probe torch.
gpu_available = (
    (hasattr(torch, "cuda") and torch.cuda.is_available())
    or (hasattr(torch, "xpu") and torch.xpu.is_available())
)

SEED = 1234


def _ls(lora_A: torch.Tensor, lora_B: torch.Tensor, alpha: float) -> LoraStats:
    return LoraStats(module=None, lora_A=lora_A, lora_B=lora_B, alpha=alpha)


# A fused MoE expert LoRA is two ordinary Linear weights, and PEFT does NOT flatten them the
# same way. `lora_A` is `(num_experts * rank, in)` with the expert index SLOWEST, so expert `e`
# is the contiguous row block `e*rank : (e+1)*rank`. `lora_B` is `(out, num_experts * rank)`
# with the expert index FASTEST: `ParamWrapper.get_delta_factors` does
# `lora_B.reshape(out, -1, num_experts).permute(2, 0, 1)`, so expert `e` is plane `e` of that
# reshape, i.e. columns `e::num_experts`. Both are written out below rather than imported from
# `unsloth_zoo.temporary_patches.moe_utils`: an expectation that called the helper the merge
# calls would agree with any packing the helper happened to implement and could never fail.

def _peft_expert_lora_b(lora_B: torch.Tensor, expert_idx: int, num_experts: int) -> torch.Tensor:
    """Expert `expert_idx`'s `(out, rank)` block of a fused `lora_B`, as PEFT packs it."""
    return lora_B.reshape(lora_B.shape[0], -1, num_experts)[:, :, expert_idx]


def _expert_slowest_lora_b(lora_B: torch.Tensor, expert_idx: int, rank_per: int) -> torch.Tensor:
    """The pre-fix reading: one contiguous rank-wide column block per expert.

    Valid for every `(num_experts, rank)` since `num_experts * rank == rank * num_experts`, so
    it never raises and the norm of the delta barely moves. Kept only so a test can assert the
    merge does NOT produce it."""
    return lora_B[:, expert_idx * rank_per:(expert_idx + 1) * rank_per]


# 1. _active_merge_device — recent fix that replaced the W-based helper.

def test_active_merge_device_returns_string_on_gpu():
    if not gpu_available:
        pytest.skip("requires CUDA or XPU")
    assert _active_merge_device() == DEVICE_TYPE_TORCH


def test_active_merge_device_takes_no_args():
    """The post-fix helper takes no arguments (was previously _active_merge_device(W))."""
    import inspect
    sig = inspect.signature(_active_merge_device)
    assert len(sig.parameters) == 0, (
        f"expected no params; previous bug: helper took W and leaked W.device.index "
        f"across device types. got params: {sig.parameters}"
    )


# 2. _merge_lora — basic correctness against a numpy reference.

def _ref_merge_lora(W: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor,
                    alpha: float) -> torch.Tensor:
    """Reference: float32 result of W + alpha * (lora_B @ lora_A)."""
    return (W.to(torch.float32) +
            alpha * (lora_B.to(torch.float32) @ lora_A.to(torch.float32)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_merge_lora_standard(dtype):
    torch.manual_seed(SEED)
    out_dim, in_dim, rank = 64, 32, 8
    alpha = 16.0
    W = torch.randn(out_dim, in_dim, dtype=dtype)
    lora_A = torch.randn(rank, in_dim, dtype=dtype) * 0.05
    lora_B = torch.randn(out_dim, rank, dtype=dtype) * 0.05
    expected = _ref_merge_lora(W, lora_A, lora_B, alpha)

    out = _merge_lora(W.clone(), _ls(lora_A, lora_B, alpha), name="test_layer")

    assert out.shape == expected.shape
    # _merge_lora returns the float32 in-place addmm result.
    assert out.dtype == torch.float32
    # bf16/fp16 inputs give reduced precision; pick the looser tolerance.
    tol = {torch.float32: 1e-5, torch.bfloat16: 5e-2, torch.float16: 5e-3}[dtype]
    torch.testing.assert_close(out.cpu().to(torch.float32),
                               expected.to(torch.float32), atol=tol, rtol=tol)


def test_merge_lora_moves_cpu_inputs_to_active_device():
    """W on CPU should land on the active device after _merge_lora.

    The W-based helper returned an indexless torch.device('cuda') for CPU W
    (unreliable on multi-GPU); the fix returns the string 'cuda' instead.
    """
    if not gpu_available:
        pytest.skip("requires CUDA or XPU")
    torch.manual_seed(SEED)
    W = torch.randn(64, 32, dtype=torch.bfloat16)
    lora_A = torch.randn(8, 32, dtype=torch.bfloat16) * 0.05
    lora_B = torch.randn(64, 8, dtype=torch.bfloat16) * 0.05
    out = _merge_lora(W.clone(), _ls(lora_A, lora_B, alpha=16.0), name="cpu_input")
    assert out.device.type == DEVICE_TYPE_TORCH, (
        f"expected merge result on {DEVICE_TYPE_TORCH} after _active_merge_device(), got {out.device}"
    )


def test_merge_lora_vocab_resize():
    """lora_B taller than W: merge zero-pads W (vocab-grow / added-tokens path)."""
    torch.manual_seed(SEED)
    old_vocab, new_vocab, dim, rank = 100, 128, 32, 8
    alpha = 16.0
    W = torch.randn(old_vocab, dim, dtype=torch.bfloat16)
    lora_A = torch.randn(rank, dim, dtype=torch.bfloat16) * 0.05
    lora_B = torch.randn(new_vocab, rank, dtype=torch.bfloat16) * 0.05

    out = _merge_lora(W.clone(), _ls(lora_A, lora_B, alpha), name="vocab_resize")

    assert out.shape == (new_vocab, dim)
    assert out.dtype == torch.float32
    expected_old = (W.to(torch.float32) +
                    alpha * (lora_B[:old_vocab].to(torch.float32) @ lora_A.to(torch.float32)))
    expected_new = alpha * (lora_B[old_vocab:].to(torch.float32) @ lora_A.to(torch.float32))
    torch.testing.assert_close(out[:old_vocab].cpu(), expected_old, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out[old_vocab:].cpu(), expected_new, atol=5e-2, rtol=5e-2)


def test_merge_lora_raises_on_nonfinite():
    torch.manual_seed(SEED)
    W = torch.full((8, 4), float("inf"), dtype=torch.float32)
    lora_A = torch.zeros(2, 4, dtype=torch.float32)
    lora_B = torch.zeros(8, 2, dtype=torch.float32)
    with pytest.raises(ValueError, match="infinite elements"):
        _merge_lora(W, _ls(lora_A, lora_B, alpha=1.0), name="bad_layer")


def test_merge_lora_returns_W_when_lora_missing():
    W = torch.randn(8, 4)
    out = _merge_lora(W, _ls(None, None, alpha=1.0), name="no_lora")
    assert out is W


# 3. _merge_moe_gate_expert — first half of A is gate_proj.

def test_merge_moe_gate_expert():
    """gate_W shape (inter_dim, hidden_dim).  delta = (B @ gate_a).T."""
    torch.manual_seed(SEED)
    # num_experts != rank_per so a confusion of the two axes cannot pass unnoticed, and the
    # factors are scaled so the delta is large against the atol below: at 0.05 the whole delta
    # was smaller than the tolerance and this test passed under either lora_B packing.
    num_experts, rank_per, inter_dim, hidden_dim = 4, 3, 8, 12
    total_rank = num_experts * rank_per
    two_inter = 2 * inter_dim
    alpha = 8.0

    gate_W = torch.randn(inter_dim, hidden_dim, dtype=torch.bfloat16)
    lora_A = torch.randn(total_rank, two_inter, dtype=torch.bfloat16) * 0.5
    lora_B = torch.randn(hidden_dim, total_rank, dtype=torch.bfloat16) * 0.5
    expert_idx = 1
    out = _merge_moe_gate_expert(gate_W.clone(), _ls(lora_A, lora_B, alpha),
                                 expert_idx=expert_idx, num_experts=num_experts,
                                 output_dtype=torch.bfloat16)

    s, e = expert_idx * rank_per, (expert_idx + 1) * rank_per
    a_slice = lora_A[s:e].to(torch.float32)
    b_slice = _peft_expert_lora_b(lora_B, expert_idx, num_experts).to(torch.float32)
    gate_a = a_slice[:, :inter_dim]
    gate_delta = b_slice @ gate_a  # (H, I)
    expected = gate_W.to(torch.float32) + alpha * gate_delta.T  # (I, H)

    assert out.shape == gate_W.shape
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.to(torch.float32).cpu(), expected, atol=1e-1, rtol=1e-1)


# 4. _merge_moe_up_expert — second half of A is up_proj.

def test_merge_moe_up_expert():
    torch.manual_seed(SEED)
    # See test_merge_moe_gate_expert for why num_experts != rank_per and why the factors are
    # not scaled by 0.05 any more.
    num_experts, rank_per, inter_dim, hidden_dim = 4, 3, 8, 12
    total_rank = num_experts * rank_per
    two_inter = 2 * inter_dim
    alpha = 8.0

    up_W = torch.randn(inter_dim, hidden_dim, dtype=torch.bfloat16)
    lora_A = torch.randn(total_rank, two_inter, dtype=torch.bfloat16) * 0.5
    lora_B = torch.randn(hidden_dim, total_rank, dtype=torch.bfloat16) * 0.5
    expert_idx = 2
    out = _merge_moe_up_expert(up_W.clone(), _ls(lora_A, lora_B, alpha),
                               expert_idx=expert_idx, num_experts=num_experts,
                               output_dtype=torch.bfloat16)

    s, e = expert_idx * rank_per, (expert_idx + 1) * rank_per
    a_slice = lora_A[s:e].to(torch.float32)
    b_slice = _peft_expert_lora_b(lora_B, expert_idx, num_experts).to(torch.float32)
    up_a = a_slice[:, inter_dim:]
    up_delta = b_slice @ up_a
    expected = up_W.to(torch.float32) + alpha * up_delta.T

    torch.testing.assert_close(out.to(torch.float32).cpu(), expected, atol=1e-1, rtol=1e-1)


# 5. _merge_moe_down_proj_expert — full A slice (no halving).

def test_merge_moe_down_proj_expert():
    """down_W shape (H, I).  A: (total_rank, H).  B: (I, total_rank).  delta = (B @ A).T = (H, I)."""
    torch.manual_seed(SEED)
    num_experts, rank_per = 4, 3
    total_rank = num_experts * rank_per
    H, I = 12, 8  # hidden_dim, intermediate_dim
    alpha = 8.0

    down_W = torch.randn(H, I, dtype=torch.bfloat16)
    lora_A = torch.randn(total_rank, H, dtype=torch.bfloat16) * 0.5  # A.shape[1] = H = out_dim
    lora_B = torch.randn(I, total_rank, dtype=torch.bfloat16) * 0.5  # B.shape[0] = I = in_dim
    expert_idx = 3
    out = _merge_moe_down_proj_expert(down_W.clone(), _ls(lora_A, lora_B, alpha),
                                      expert_idx=expert_idx, num_experts=num_experts,
                                      output_dtype=torch.bfloat16)

    s, e = expert_idx * rank_per, (expert_idx + 1) * rank_per
    a_slice = lora_A[s:e].to(torch.float32)         # (R, H)
    b_slice = _peft_expert_lora_b(lora_B, expert_idx, num_experts).to(torch.float32)  # (I, R)
    delta = b_slice @ a_slice                       # (I, H)
    expected = down_W.to(torch.float32) + alpha * delta.T  # (H, I)

    torch.testing.assert_close(out.to(torch.float32).cpu(), expected, atol=1e-1, rtol=1e-1)


# 6. _merge_moe_fused_gate_up_expert — 3D fused tensor across all experts.

@pytest.mark.parametrize("is_transposed", [True, False])
def test_merge_moe_fused_gate_up_expert(is_transposed):
    """Both transposed (GPT-OSS) and standard (Gemma4) layouts."""
    torch.manual_seed(SEED)
    num_experts, rank_per = 4, 3
    total_rank = num_experts * rank_per
    inter_dim, hidden_dim = 8, 12
    two_inter = 2 * inter_dim
    alpha = 8.0

    if is_transposed:
        # GPT-OSS: (E, H, 2*I), A (E*R, H), B (2*I, E*R)
        gate_up_W = torch.randn(num_experts, hidden_dim, two_inter, dtype=torch.bfloat16)
        lora_A = torch.randn(total_rank, hidden_dim, dtype=torch.bfloat16) * 0.05
        lora_B = torch.randn(two_inter, total_rank, dtype=torch.bfloat16) * 0.05
    else:
        # Gemma4: (E, 2*I, H), A (E*R, H), B (2*I, E*R)
        gate_up_W = torch.randn(num_experts, two_inter, hidden_dim, dtype=torch.bfloat16)
        lora_A = torch.randn(total_rank, hidden_dim, dtype=torch.bfloat16) * 0.05
        lora_B = torch.randn(two_inter, total_rank, dtype=torch.bfloat16) * 0.05

    out = _merge_moe_fused_gate_up_expert(
        gate_up_W.clone(), _ls(lora_A, lora_B, alpha),
        output_dtype=torch.bfloat16, is_transposed=is_transposed,
    )
    expected = gate_up_W.to(torch.float32).clone()
    for ei in range(num_experts):
        s, e = ei * rank_per, (ei + 1) * rank_per
        b_e = _peft_expert_lora_b(lora_B, ei, num_experts).to(torch.float32)
        delta = b_e @ lora_A[s:e].to(torch.float32)
        expected[ei] = expected[ei] + alpha * (delta.T if is_transposed else delta)

    torch.testing.assert_close(out.to(torch.float32).cpu(), expected, atol=1e-1, rtol=1e-1)


# 7. _merge_moe_fused_down_proj_expert — 3D fused tensor.

@pytest.mark.parametrize("is_transposed", [True, False])
def test_merge_moe_fused_down_proj_expert(is_transposed):
    """Fused 3D down weight (E, dim1, dim2).

    The function uses a heuristic to detect layout based on which of dim1/dim2
    matches A's column count (dim_A) vs B's row count (dim_B):
      - is_transposed=True (use_transpose=True): A.shape[1] == down_W.shape[1]
      - is_transposed=False: A.shape[1] == down_W.shape[2]
    delta = b_slice @ a_slice has shape (dim_B, dim_A).
    Then merged[ei] += delta.T if use_transpose else delta.
    """
    torch.manual_seed(SEED)
    num_experts, rank_per = 4, 3
    total_rank = num_experts * rank_per
    H, I = 12, 8
    alpha = 8.0

    if is_transposed:
        # dim_A == dim1, dim_B == dim2.  Pick: down_W (E, H, I), A (R, H), B (I, R)
        # delta = (I, H), delta.T = (H, I) which fits down_W[ei]
        down_W = torch.randn(num_experts, H, I, dtype=torch.bfloat16)
        lora_A = torch.randn(total_rank, H, dtype=torch.bfloat16) * 0.05  # dim_A=H=dim1
        lora_B = torch.randn(I, total_rank, dtype=torch.bfloat16) * 0.05  # dim_B=I=dim2
    else:
        # dim_A == dim2, dim_B == dim1.  Pick: down_W (E, H, I), A (R, I), B (H, R)
        # delta = (H, I) which fits down_W[ei] directly
        down_W = torch.randn(num_experts, H, I, dtype=torch.bfloat16)
        lora_A = torch.randn(total_rank, I, dtype=torch.bfloat16) * 0.05  # dim_A=I=dim2
        lora_B = torch.randn(H, total_rank, dtype=torch.bfloat16) * 0.05  # dim_B=H=dim1

    out = _merge_moe_fused_down_proj_expert(
        down_W.clone(), _ls(lora_A, lora_B, alpha),
        output_dtype=torch.bfloat16, is_transposed=is_transposed,
    )
    expected = down_W.to(torch.float32).clone()
    for ei in range(num_experts):
        s, e = ei * rank_per, (ei + 1) * rank_per
        b_e = _peft_expert_lora_b(lora_B, ei, num_experts).to(torch.float32)
        delta = b_e @ lora_A[s:e].to(torch.float32)
        expected[ei] = expected[ei] + alpha * (delta.T if is_transposed else delta)

    torch.testing.assert_close(out.to(torch.float32).cpu(), expected, atol=1e-1, rtol=1e-1)


# 8. PEFT 0.19+ standard layout (#5410).

def test_merge_moe_gate_expert_standard_layout():
    torch.manual_seed(SEED)
    num_experts, rank_per, inter_dim, hidden_dim = 4, 3, 8, 12
    total_rank = num_experts * rank_per
    alpha = 8.0

    gate_W = torch.randn(inter_dim, hidden_dim, dtype=torch.bfloat16)
    lora_A = torch.randn(total_rank, hidden_dim,  dtype=torch.bfloat16) * 0.5
    lora_B = torch.randn(2 * inter_dim, total_rank, dtype=torch.bfloat16) * 0.5
    expert_idx = 1

    out = _merge_moe_gate_expert(
        gate_W.clone(), _ls(lora_A, lora_B, alpha),
        expert_idx=expert_idx, num_experts=num_experts,
        output_dtype=torch.bfloat16,
    )
    s, e = expert_idx * rank_per, (expert_idx + 1) * rank_per
    a_slice = lora_A[s:e].to(torch.float32)
    b_slice = _peft_expert_lora_b(lora_B, expert_idx, num_experts).to(torch.float32)
    delta = b_slice[:inter_dim, :] @ a_slice
    expected = gate_W.to(torch.float32) + alpha * delta
    torch.testing.assert_close(out.to(torch.float32).cpu(), expected, atol=1e-1, rtol=1e-1)


def test_merge_moe_up_expert_standard_layout():
    torch.manual_seed(SEED)
    num_experts, rank_per, inter_dim, hidden_dim = 4, 3, 8, 12
    total_rank = num_experts * rank_per
    alpha = 8.0

    up_W = torch.randn(inter_dim, hidden_dim, dtype=torch.bfloat16)
    lora_A = torch.randn(total_rank, hidden_dim,  dtype=torch.bfloat16) * 0.5
    lora_B = torch.randn(2 * inter_dim, total_rank, dtype=torch.bfloat16) * 0.5
    expert_idx = 2

    out = _merge_moe_up_expert(
        up_W.clone(), _ls(lora_A, lora_B, alpha),
        expert_idx=expert_idx, num_experts=num_experts,
        output_dtype=torch.bfloat16,
    )
    s, e = expert_idx * rank_per, (expert_idx + 1) * rank_per
    a_slice = lora_A[s:e].to(torch.float32)
    b_slice = _peft_expert_lora_b(lora_B, expert_idx, num_experts).to(torch.float32)
    delta = b_slice[inter_dim:, :] @ a_slice
    expected = up_W.to(torch.float32) + alpha * delta
    torch.testing.assert_close(out.to(torch.float32).cpu(), expected, atol=1e-1, rtol=1e-1)


def test_merge_moe_down_proj_expert_standard_layout():
    torch.manual_seed(SEED)
    num_experts, rank_per = 4, 3
    total_rank = num_experts * rank_per
    H, I = 12, 8
    alpha = 8.0

    down_W = torch.randn(H, I, dtype=torch.bfloat16)
    lora_A = torch.randn(total_rank, I, dtype=torch.bfloat16) * 0.5
    lora_B = torch.randn(H, total_rank, dtype=torch.bfloat16) * 0.5
    expert_idx = 3

    out = _merge_moe_down_proj_expert(
        down_W.clone(), _ls(lora_A, lora_B, alpha),
        expert_idx=expert_idx, num_experts=num_experts,
        output_dtype=torch.bfloat16,
    )
    s, e = expert_idx * rank_per, (expert_idx + 1) * rank_per
    a_slice = lora_A[s:e].to(torch.float32)
    b_slice = _peft_expert_lora_b(lora_B, expert_idx, num_experts).to(torch.float32)
    delta = b_slice @ a_slice
    expected = down_W.to(torch.float32) + alpha * delta
    torch.testing.assert_close(out.to(torch.float32).cpu(), expected, atol=1e-1, rtol=1e-1)


# 8b. The packing itself: the merge must reproduce PEFT's reading and must NOT reproduce the
#     pre-fix one, and the legacy override has to still reach an adapter trained before it.

def test_moe_expert_merge_rejects_the_expert_slowest_reading():
    """A merge that read lora_B expert-slowest would still produce a correctly shaped, similarly
    sized delta, so only an explicit mismatch assertion catches a swap back. Run in float32 with
    a tight tolerance so "matches PEFT" and "does not match the old reading" are both decided by
    the packing and not by bf16 rounding."""
    torch.manual_seed(SEED)
    num_experts, rank_per = 4, 3
    total_rank = num_experts * rank_per
    inter_dim, hidden_dim = 8, 12
    two_inter = 2 * inter_dim
    alpha = 8.0

    gate_up_W = torch.randn(num_experts, two_inter, hidden_dim, dtype=torch.float32)
    lora_A = torch.randn(total_rank, hidden_dim, dtype=torch.float32) * 0.5
    lora_B = torch.randn(two_inter, total_rank, dtype=torch.float32) * 0.5

    out = _merge_moe_fused_gate_up_expert(
        gate_up_W.clone(), _ls(lora_A, lora_B, alpha),
        output_dtype=torch.float32, is_transposed=False,
    ).to(torch.float32).cpu()

    peft = gate_up_W.clone()
    old  = gate_up_W.clone()
    for ei in range(num_experts):
        a_e = lora_A[ei * rank_per:(ei + 1) * rank_per]
        peft[ei] += alpha * (_peft_expert_lora_b(lora_B, ei, num_experts) @ a_e)
        old[ei]  += alpha * (_expert_slowest_lora_b(lora_B, ei, rank_per) @ a_e)

    torch.testing.assert_close(out, peft, atol=1e-4, rtol=1e-4)

    # The two readings must be far apart for this to mean anything: E and rank are both > 1, so
    # they are, and the delta is large enough to see. Guard the fixture, then the merge.
    separation = (peft - old).abs().max().item()
    assert separation > 1.0, (
        f"fixture is degenerate: the two lora_B readings differ by only {separation:.2e}, so "
        "this test would pass under either of them"
    )
    assert (out - old).abs().max().item() > 1.0, (
        "the merge reproduced the pre-fix expert-slowest reading of lora_B"
    )


@pytest.mark.parametrize("layout", ["rank_major", "grouped_by_expert"])
def test_moe_expert_merge_follows_the_layout_override(monkeypatch, layout):
    """`UNSLOTH_MOE_LORA_B_LAYOUT=grouped_by_expert` is the only way to read back an adapter
    trained by a pre-fix Unsloth, so the per-expert merge path has to honour it, not just the
    fused one. Both values are covered here so neither reading can be dropped silently."""
    monkeypatch.setenv("UNSLOTH_MOE_LORA_B_LAYOUT", layout)
    torch.manual_seed(SEED)
    num_experts, rank_per, inter_dim, hidden_dim = 4, 3, 8, 12
    total_rank = num_experts * rank_per
    alpha = 8.0

    gate_W = torch.randn(inter_dim, hidden_dim, dtype=torch.float32)
    lora_A = torch.randn(total_rank, hidden_dim, dtype=torch.float32) * 0.5
    lora_B = torch.randn(2 * inter_dim, total_rank, dtype=torch.float32) * 0.5
    expert_idx = 1

    b_slice = (_peft_expert_lora_b(lora_B, expert_idx, num_experts) if layout == "rank_major"
               else _expert_slowest_lora_b(lora_B, expert_idx, rank_per))
    a_slice = lora_A[expert_idx * rank_per:(expert_idx + 1) * rank_per]

    # The override has to reach all three per-expert helpers. gate and up split the same
    # lora_B in half; down_proj takes the whole slice against its own weight.
    for merge, base_W, expected_delta in (
        (_merge_moe_gate_expert, gate_W, b_slice[:inter_dim, :] @ a_slice),
        (_merge_moe_up_expert,   gate_W, b_slice[inter_dim:, :] @ a_slice),
    ):
        out = merge(
            base_W.clone(), _ls(lora_A, lora_B, alpha),
            expert_idx=expert_idx, num_experts=num_experts,
            output_dtype=torch.float32,
        ).to(torch.float32).cpu()
        torch.testing.assert_close(
            out, base_W + alpha * expected_delta, atol=1e-4, rtol=1e-4,
        )

    down_W = torch.randn(hidden_dim, inter_dim, dtype=torch.float32)
    down_A = torch.randn(total_rank, inter_dim, dtype=torch.float32) * 0.5
    down_B = torch.randn(hidden_dim, total_rank, dtype=torch.float32) * 0.5
    down_b = (_peft_expert_lora_b(down_B, expert_idx, num_experts) if layout == "rank_major"
              else _expert_slowest_lora_b(down_B, expert_idx, rank_per))
    down_out = _merge_moe_down_proj_expert(
        down_W.clone(), _ls(down_A, down_B, alpha),
        expert_idx=expert_idx, num_experts=num_experts,
        output_dtype=torch.float32,
    ).to(torch.float32).cpu()
    down_a = down_A[expert_idx * rank_per:(expert_idx + 1) * rank_per]
    torch.testing.assert_close(
        down_out, down_W + alpha * (down_b @ down_a), atol=1e-4, rtol=1e-4,
    )

    # Not vacuous: the other reading would have failed every assertion above.
    other = (_expert_slowest_lora_b(lora_B, expert_idx, rank_per) if layout == "rank_major"
             else _peft_expert_lora_b(lora_B, expert_idx, num_experts))
    assert (other - b_slice).abs().max().item() > 1e-3


# 9. Layout detection + fallback (#5410).

def test_detect_moe_lora_layout_classifies_both_conventions():
    from unsloth_zoo.saving_utils import _detect_moe_lora_layout
    num_experts, r, out_dim, in_dim = 4, 4, 16, 12
    total_rank = num_experts * r
    A_swap = torch.empty(total_rank, out_dim)
    B_swap = torch.empty(in_dim,     total_rank)
    assert _detect_moe_lora_layout(A_swap, B_swap, num_experts, out_dim, in_dim) == ("swapped", r)
    A_std = torch.empty(total_rank, in_dim)
    B_std = torch.empty(out_dim,    total_rank)
    assert _detect_moe_lora_layout(A_std, B_std, num_experts, out_dim, in_dim) == ("standard", r)
    A_bad = torch.empty(total_rank, out_dim + 1)
    B_bad = torch.empty(in_dim,     total_rank)
    assert _detect_moe_lora_layout(A_bad, B_bad, num_experts, out_dim, in_dim)[0] == "unknown"
    assert _detect_moe_lora_layout(A_swap, B_swap, num_experts + 1, out_dim, in_dim)[0] == "unknown"


def test_moe_merge_fallback_counter_records_bad_layout():
    from unsloth_zoo.saving_utils import _MOE_MERGE_STATE, _reset_moe_merge_state
    _reset_moe_merge_state()
    num_experts, rank_per, inter_dim, hidden_dim = 4, 4, 8, 12
    total_rank = num_experts * rank_per
    gate_W = torch.randn(inter_dim, hidden_dim, dtype=torch.bfloat16)
    lora_A = torch.randn(total_rank, hidden_dim + 7, dtype=torch.bfloat16)
    lora_B = torch.randn(hidden_dim, total_rank,    dtype=torch.bfloat16)
    out = _merge_moe_gate_expert(
        gate_W.clone(), _ls(lora_A, lora_B, 1.0),
        expert_idx=0, num_experts=num_experts, output_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(out.cpu(), gate_W)
    assert _MOE_MERGE_STATE["fallback"] >= 1
    assert _MOE_MERGE_STATE["first_error"] is not None
    assert _MOE_MERGE_STATE["first_error"]["role"] == "gate"
    _reset_moe_merge_state()


def test_resolve_num_experts_walks_base_layer_chain():
    from unsloth_zoo.saving_utils import _resolve_num_experts_from_lora_stats

    class Inner:
        num_experts = 128

    class Outer:
        base_layer = Inner()

    stats_inner_only = LoraStats(module=Inner(), lora_A=None, lora_B=None, alpha=1.0)
    assert _resolve_num_experts_from_lora_stats(stats_inner_only, fallback=-1) == 128

    stats_via_base_layer = LoraStats(module=Outer(), lora_A=None, lora_B=None, alpha=1.0)
    assert _resolve_num_experts_from_lora_stats(stats_via_base_layer, fallback=-1) == 128

    stats_none = LoraStats(module=None, lora_A=None, lora_B=None, alpha=1.0)
    assert _resolve_num_experts_from_lora_stats(stats_none, fallback=17) == 17
