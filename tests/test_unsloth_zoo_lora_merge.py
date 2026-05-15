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

"""
Tier 0 LoRA merge correctness tests for unsloth_zoo/saving_utils.py.

These run on Linux+CUDA without any MLX shim.  They exercise:

1. _active_merge_device() returns the active accelerator family string
   (cuda on a CUDA host).  This is the recently-pushed fix that replaced
   the W-based helper which leaked device indices across device types.
2. _merge_lora computes  W + alpha * lora_B @ lora_A  with the right
   shapes, dtypes, and device placement.
3. _merge_lora handles the vocab-resize case (lora_B taller than W).
4. _merge_lora raises on non-finite values.
5. The 5 MoE expert-merge variants compute the correct per-expert
   updates against a numpy reference.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

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


SEED = 1234


def _ls(lora_A: torch.Tensor, lora_B: torch.Tensor, alpha: float) -> LoraStats:
    return LoraStats(module=None, lora_A=lora_A, lora_B=lora_B, alpha=alpha)


# ---------------------------------------------------------------------------
# 1. _active_merge_device — recent fix that replaced the W-based helper.
# ---------------------------------------------------------------------------

def test_active_merge_device_returns_string_on_cuda_host():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    assert _active_merge_device() == "cuda"


def test_active_merge_device_takes_no_args():
    """The post-fix helper takes no arguments (was previously _active_merge_device(W))."""
    import inspect
    sig = inspect.signature(_active_merge_device)
    assert len(sig.parameters) == 0, (
        f"expected no params; previous bug: helper took W and leaked W.device.index "
        f"across device types. got params: {sig.parameters}"
    )


# ---------------------------------------------------------------------------
# 2. _merge_lora — basic correctness against a numpy reference.
# ---------------------------------------------------------------------------

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

    Pre-fix: the W-based helper returned torch.device('cuda') (no index)
    when W was on CPU, which delegates to current_device() — mostly
    correct on single-GPU but unreliable on multi-GPU.
    Post-fix: returns the string 'cuda', .to('cuda') uses current_device
    consistently.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(SEED)
    W = torch.randn(64, 32, dtype=torch.bfloat16)
    lora_A = torch.randn(8, 32, dtype=torch.bfloat16) * 0.05
    lora_B = torch.randn(64, 8, dtype=torch.bfloat16) * 0.05
    out = _merge_lora(W.clone(), _ls(lora_A, lora_B, alpha=16.0), name="cpu_input")
    assert out.is_cuda, "expected merge result on CUDA after _active_merge_device()"


def test_merge_lora_vocab_resize():
    """When lora_B has more rows than W, the merge expands W with zero-padding.

    This path is used when fine-tuning grows the vocab (added tokens).
    """
    torch.manual_seed(SEED)
    old_vocab, new_vocab, dim, rank = 100, 128, 32, 8
    alpha = 16.0
    W = torch.randn(old_vocab, dim, dtype=torch.bfloat16)
    lora_A = torch.randn(rank, dim, dtype=torch.bfloat16) * 0.05
    lora_B = torch.randn(new_vocab, rank, dtype=torch.bfloat16) * 0.05

    out = _merge_lora(W.clone(), _ls(lora_A, lora_B, alpha), name="vocab_resize")

    assert out.shape == (new_vocab, dim)
    assert out.dtype == torch.float32
    # The first old_vocab rows: original W + alpha * lora_B[:old_vocab] @ lora_A
    expected_old = (W.to(torch.float32) +
                    alpha * (lora_B[:old_vocab].to(torch.float32) @ lora_A.to(torch.float32)))
    # New rows: zero base + alpha * lora_B[old_vocab:] @ lora_A
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


# ---------------------------------------------------------------------------
# 3. _merge_moe_gate_expert — first half of A is gate_proj.
# ---------------------------------------------------------------------------

def test_merge_moe_gate_expert():
    """gate_W shape (inter_dim, hidden_dim).  delta = (B @ gate_a).T."""
    torch.manual_seed(SEED)
    num_experts, rank_per, inter_dim, hidden_dim = 4, 4, 8, 12
    total_rank = num_experts * rank_per
    two_inter = 2 * inter_dim
    alpha = 8.0

    gate_W = torch.randn(inter_dim, hidden_dim, dtype=torch.bfloat16)
    lora_A = torch.randn(total_rank, two_inter, dtype=torch.bfloat16) * 0.05
    lora_B = torch.randn(hidden_dim, total_rank, dtype=torch.bfloat16) * 0.05
    expert_idx = 1
    out = _merge_moe_gate_expert(gate_W.clone(), _ls(lora_A, lora_B, alpha),
                                 expert_idx=expert_idx, num_experts=num_experts,
                                 output_dtype=torch.bfloat16)

    # Reference: a_slice = lora_A[r:r*2], gate_a = a_slice[:, :inter_dim]
    s, e = expert_idx * rank_per, (expert_idx + 1) * rank_per
    a_slice = lora_A[s:e].to(torch.float32)
    b_slice = lora_B[:, s:e].to(torch.float32)
    gate_a = a_slice[:, :inter_dim]
    gate_delta = b_slice @ gate_a  # (H, I)
    expected = gate_W.to(torch.float32) + alpha * gate_delta.T  # (I, H)

    assert out.shape == gate_W.shape
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.to(torch.float32).cpu(), expected, atol=1e-1, rtol=1e-1)


# ---------------------------------------------------------------------------
# 4. _merge_moe_up_expert — second half of A is up_proj.
# ---------------------------------------------------------------------------

def test_merge_moe_up_expert():
    torch.manual_seed(SEED)
    num_experts, rank_per, inter_dim, hidden_dim = 4, 4, 8, 12
    total_rank = num_experts * rank_per
    two_inter = 2 * inter_dim
    alpha = 8.0

    up_W = torch.randn(inter_dim, hidden_dim, dtype=torch.bfloat16)
    lora_A = torch.randn(total_rank, two_inter, dtype=torch.bfloat16) * 0.05
    lora_B = torch.randn(hidden_dim, total_rank, dtype=torch.bfloat16) * 0.05
    expert_idx = 2
    out = _merge_moe_up_expert(up_W.clone(), _ls(lora_A, lora_B, alpha),
                               expert_idx=expert_idx, num_experts=num_experts,
                               output_dtype=torch.bfloat16)

    s, e = expert_idx * rank_per, (expert_idx + 1) * rank_per
    a_slice = lora_A[s:e].to(torch.float32)
    b_slice = lora_B[:, s:e].to(torch.float32)
    up_a = a_slice[:, inter_dim:]  # second half
    up_delta = b_slice @ up_a
    expected = up_W.to(torch.float32) + alpha * up_delta.T

    torch.testing.assert_close(out.to(torch.float32).cpu(), expected, atol=1e-1, rtol=1e-1)


# ---------------------------------------------------------------------------
# 5. _merge_moe_down_proj_expert — full A slice (no halving).
# ---------------------------------------------------------------------------

def test_merge_moe_down_proj_expert():
    """down_W shape (H, I).  A: (total_rank, H).  B: (I, total_rank).  delta = (B @ A).T = (H, I)."""
    torch.manual_seed(SEED)
    num_experts, rank_per = 4, 4
    total_rank = num_experts * rank_per
    H, I = 12, 8  # hidden_dim, intermediate_dim
    alpha = 8.0

    down_W = torch.randn(H, I, dtype=torch.bfloat16)
    lora_A = torch.randn(total_rank, H, dtype=torch.bfloat16) * 0.05  # A.shape[1] = H = out_dim
    lora_B = torch.randn(I, total_rank, dtype=torch.bfloat16) * 0.05  # B.shape[0] = I = in_dim
    expert_idx = 3
    out = _merge_moe_down_proj_expert(down_W.clone(), _ls(lora_A, lora_B, alpha),
                                      expert_idx=expert_idx, num_experts=num_experts,
                                      output_dtype=torch.bfloat16)

    s, e = expert_idx * rank_per, (expert_idx + 1) * rank_per
    a_slice = lora_A[s:e].to(torch.float32)         # (R, H)
    b_slice = lora_B[:, s:e].to(torch.float32)      # (I, R)
    delta = b_slice @ a_slice                       # (I, H)
    expected = down_W.to(torch.float32) + alpha * delta.T  # (H, I)

    torch.testing.assert_close(out.to(torch.float32).cpu(), expected, atol=1e-1, rtol=1e-1)


# ---------------------------------------------------------------------------
# 6. _merge_moe_fused_gate_up_expert — 3D fused tensor across all experts.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("is_transposed", [True, False])
def test_merge_moe_fused_gate_up_expert(is_transposed):
    """Both transposed (GPT-OSS) and standard (Gemma4) layouts."""
    torch.manual_seed(SEED)
    num_experts, rank_per = 4, 4
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
        delta = lora_B[:, s:e].to(torch.float32) @ lora_A[s:e].to(torch.float32)
        expected[ei] = expected[ei] + alpha * (delta.T if is_transposed else delta)

    torch.testing.assert_close(out.to(torch.float32).cpu(), expected, atol=1e-1, rtol=1e-1)


# ---------------------------------------------------------------------------
# 7. _merge_moe_fused_down_proj_expert — 3D fused tensor.
# ---------------------------------------------------------------------------

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
    num_experts, rank_per = 4, 4
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
        delta = lora_B[:, s:e].to(torch.float32) @ lora_A[s:e].to(torch.float32)
        expected[ei] = expected[ei] + alpha * (delta.T if is_transposed else delta)

    torch.testing.assert_close(out.to(torch.float32).cpu(), expected, atol=1e-1, rtol=1e-1)


# 8. PEFT 0.19+ standard layout (#5410).

def test_merge_moe_gate_expert_standard_layout():
    torch.manual_seed(SEED)
    num_experts, rank_per, inter_dim, hidden_dim = 4, 4, 8, 12
    total_rank = num_experts * rank_per
    alpha = 8.0

    gate_W = torch.randn(inter_dim, hidden_dim, dtype=torch.bfloat16)
    lora_A = torch.randn(total_rank, hidden_dim,  dtype=torch.bfloat16) * 0.05
    lora_B = torch.randn(2 * inter_dim, total_rank, dtype=torch.bfloat16) * 0.05
    expert_idx = 1

    out = _merge_moe_gate_expert(
        gate_W.clone(), _ls(lora_A, lora_B, alpha),
        expert_idx=expert_idx, num_experts=num_experts,
        output_dtype=torch.bfloat16,
    )
    s, e = expert_idx * rank_per, (expert_idx + 1) * rank_per
    a_slice = lora_A[s:e].to(torch.float32)
    b_slice = lora_B[:, s:e].to(torch.float32)
    delta = b_slice[:inter_dim, :] @ a_slice
    expected = gate_W.to(torch.float32) + alpha * delta
    torch.testing.assert_close(out.to(torch.float32).cpu(), expected, atol=1e-1, rtol=1e-1)


def test_merge_moe_up_expert_standard_layout():
    torch.manual_seed(SEED)
    num_experts, rank_per, inter_dim, hidden_dim = 4, 4, 8, 12
    total_rank = num_experts * rank_per
    alpha = 8.0

    up_W = torch.randn(inter_dim, hidden_dim, dtype=torch.bfloat16)
    lora_A = torch.randn(total_rank, hidden_dim,  dtype=torch.bfloat16) * 0.05
    lora_B = torch.randn(2 * inter_dim, total_rank, dtype=torch.bfloat16) * 0.05
    expert_idx = 2

    out = _merge_moe_up_expert(
        up_W.clone(), _ls(lora_A, lora_B, alpha),
        expert_idx=expert_idx, num_experts=num_experts,
        output_dtype=torch.bfloat16,
    )
    s, e = expert_idx * rank_per, (expert_idx + 1) * rank_per
    a_slice = lora_A[s:e].to(torch.float32)
    b_slice = lora_B[:, s:e].to(torch.float32)
    delta = b_slice[inter_dim:, :] @ a_slice
    expected = up_W.to(torch.float32) + alpha * delta
    torch.testing.assert_close(out.to(torch.float32).cpu(), expected, atol=1e-1, rtol=1e-1)


def test_merge_moe_down_proj_expert_standard_layout():
    torch.manual_seed(SEED)
    num_experts, rank_per = 4, 4
    total_rank = num_experts * rank_per
    H, I = 12, 8
    alpha = 8.0

    down_W = torch.randn(H, I, dtype=torch.bfloat16)
    lora_A = torch.randn(total_rank, I, dtype=torch.bfloat16) * 0.05
    lora_B = torch.randn(H, total_rank, dtype=torch.bfloat16) * 0.05
    expert_idx = 3

    out = _merge_moe_down_proj_expert(
        down_W.clone(), _ls(lora_A, lora_B, alpha),
        expert_idx=expert_idx, num_experts=num_experts,
        output_dtype=torch.bfloat16,
    )
    s, e = expert_idx * rank_per, (expert_idx + 1) * rank_per
    a_slice = lora_A[s:e].to(torch.float32)
    b_slice = lora_B[:, s:e].to(torch.float32)
    delta = b_slice @ a_slice
    expected = down_W.to(torch.float32) + alpha * delta
    torch.testing.assert_close(out.to(torch.float32).cpu(), expected, atol=1e-1, rtol=1e-1)


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
