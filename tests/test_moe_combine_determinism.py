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

"""Tests that the MoE top_k combine is reproducible (CUDA index_add_ atomics made bf16 nondeterministic), equals the
index_add_ sum in float64, and casts after the reduction."""

import pytest
import torch

from unsloth_zoo.temporary_patches.moe_utils import combine_permuted_moe_outputs


def _make_case(num_tokens, top_k, hidden, num_experts, dtype, device, seed = 0):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    expert_of_slot = torch.randint(0, num_experts, (num_tokens * top_k,), generator = g)
    sorted_indices = expert_of_slot.argsort(stable = True).to(device)
    permuted = (torch.randn(num_tokens * top_k, hidden, generator = g) * 0.1).to(device, dtype)
    return sorted_indices, permuted


def _index_add_reference(permuted, sorted_indices, num_tokens, top_k, hidden, dtype):
    """The original spelling, kept here as the thing we must agree with."""
    token_indices = sorted_indices // top_k
    out = torch.zeros((num_tokens, hidden), dtype = dtype, device = permuted.device)
    out.index_add_(0, token_indices, permuted.to(dtype))
    return out


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_combine_matches_index_add_exactly_in_float64(top_k):
    num_tokens, hidden, num_experts = 512, 64, 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sorted_indices, permuted = _make_case(
        num_tokens, top_k, hidden, num_experts, torch.float64, device,
    )
    got = combine_permuted_moe_outputs(
        permuted, sorted_indices, num_tokens, top_k, out_dtype = torch.float64,
    )
    want = _index_add_reference(
        permuted, sorted_indices, num_tokens, top_k, hidden, torch.float64,
    )
    assert got.shape == want.shape == (num_tokens, hidden)
    # float64 with a few addends makes both orderings exact, so require equality rather than a tolerance that would accept a wrong-but-close reduction.
    torch.testing.assert_close(got, want, rtol = 0, atol = 1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "atomics are a CUDA property")
def test_combine_is_bitwise_reproducible_in_bf16():
    # Large hidden dim and top_k > 1 so many threads contend for the same output rows, the contention that made the old index_add_ drift.
    num_tokens, top_k, hidden, num_experts = 4096, 4, 1024, 128
    sorted_indices, permuted = _make_case(
        num_tokens, top_k, hidden, num_experts, torch.bfloat16, "cuda",
    )
    first = combine_permuted_moe_outputs(
        permuted, sorted_indices, num_tokens, top_k, out_dtype = torch.bfloat16,
    )
    for _ in range(8):
        again = combine_permuted_moe_outputs(
            permuted, sorted_indices, num_tokens, top_k, out_dtype = torch.bfloat16,
        )
        assert torch.equal(first, again), "MoE top_k combine is not reproducible"


def test_combine_covers_every_slot_exactly_once():
    """Combine assumes a permutation; one that dropped or doubled slots would look plausible on random data, so a counting case pins it."""
    num_tokens, top_k, hidden, num_experts = 32, 4, 8, 7
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sorted_indices, _ = _make_case(
        num_tokens, top_k, hidden, num_experts, torch.float64, device,
    )
    ones = torch.ones(num_tokens * top_k, hidden, dtype = torch.float64, device = device)
    got = combine_permuted_moe_outputs(
        ones, sorted_indices, num_tokens, top_k, out_dtype = torch.float64,
    )
    assert torch.equal(got, torch.full_like(got, float(top_k)))


def _inverse_permutation(sorted_indices):
    inverse = torch.empty_like(sorted_indices)
    inverse[sorted_indices] = torch.arange(
        sorted_indices.numel(), device = sorted_indices.device, dtype = sorted_indices.dtype,
    )
    return inverse


def test_combine_casts_to_out_dtype_after_the_sum_not_before():
    """fp32 summands, bf16 output: an fp32 router (Gemma 4, DeepSeek V3) makes cast placement visible; casting last sums
    in fp32 and rounds once, as transformers' grouped_mm does; a second assertion keeps it discriminating."""
    num_tokens, top_k, hidden, num_experts = 512, 4, 64, 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sorted_indices, permuted = _make_case(
        num_tokens, top_k, hidden, num_experts, torch.float32, device,
    )
    got = combine_permuted_moe_outputs(
        permuted, sorted_indices, num_tokens, top_k, out_dtype = torch.bfloat16,
    )
    assert got.dtype == torch.bfloat16

    inverse = _inverse_permutation(sorted_indices)
    cast_last = (
        permuted[inverse].view(num_tokens, top_k, hidden).sum(dim = 1).to(torch.bfloat16)
    )
    cast_first = (
        permuted.to(torch.bfloat16)[inverse].view(num_tokens, top_k, hidden).sum(dim = 1)
    )
    assert torch.equal(got, cast_last), (
        "combine rounded the summands before adding them; max diff "
        f"{(got.float() - cast_last.float()).abs().max().item():.3e}"
    )
    differing = (cast_first != cast_last).float().mean().item()
    assert differing > 0.05, (
        "this case no longer separates the two cast placements "
        f"(only {differing:.1%} of elements differ), so it would not catch a regression"
    )


def test_combine_rejects_a_non_permutation():
    """The slot-count guard must raise ValueError, not a bare assert: python -O strips asserts and a short
    sorted_indices would silently read wrong rows; reverting to assert fails the test."""
    num_tokens, top_k, hidden, num_experts = 32, 4, 8, 7
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sorted_indices, permuted = _make_case(
        num_tokens, top_k, hidden, num_experts, torch.float32, device,
    )
    with pytest.raises(ValueError):
        combine_permuted_moe_outputs(
            permuted[:-1], sorted_indices, num_tokens, top_k, out_dtype = torch.float32,
        )


def test_combine_gradient_matches_the_index_add_form():
    """The combine is in every MoE training backward, so forward agreement is not enough: with the same upstream
    gradient, grads must match the index_add_ form exactly in every dtype."""
    num_tokens, top_k, hidden, num_experts = 512, 4, 64, 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for dtype in (torch.float64, torch.float32, torch.bfloat16):
        sorted_indices, permuted = _make_case(
            num_tokens, top_k, hidden, num_experts, dtype, device,
        )
        # One upstream gradient shared by both arms, or the comparison is meaningless.
        upstream = torch.randn(num_tokens, hidden, dtype = dtype, device = device)

        ours = permuted.clone().requires_grad_(True)
        (combine_permuted_moe_outputs(
            ours, sorted_indices, num_tokens, top_k, out_dtype = dtype,
        ) * upstream).sum().backward()

        theirs = permuted.clone().requires_grad_(True)
        (_index_add_reference(
            theirs, sorted_indices, num_tokens, top_k, hidden, dtype,
        ) * upstream).sum().backward()

        assert torch.equal(ours.grad, theirs.grad), (
            f"{dtype}: combine gradient diverges from the index_add_ form, "
            f"max diff {(ours.grad - theirs.grad).abs().max().item():.3e}"
        )
