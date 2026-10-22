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

"""The MoE top_k combine must be reproducible, and must still be the right sum.

`forward_native_grouped_mm` used to finish with
`zeros(num_tokens, hidden).index_add_(0, sorted_indices // top_k, permuted)`.
Each token index appears `top_k` times in that single call, so on CUDA it takes
index_add_'s atomicAdd path, which fixes no accumulation order; in bf16 that was
enough to give 40 distinct losses, spread over 0.0284 nats, from 40 repeats of one
identical Gemma-4 MoE forward on one loaded model with no optimizer step.

Three things are asserted here, because any one alone is easy to pass by accident:
  1. equivalence - the replacement computes the same sum as the index_add_ form
     (checked in float64, where both are exact);
  2. reproducibility - repeated calls on identical input are bitwise identical
     in bf16, with a duplicate-heavy index pattern;
  3. cast placement - `out_dtype` is applied after the reduction, not before it.
     That only has an observable effect when the routed slots arrive wider than
     `out_dtype`, so it needs a case built specifically for it: every other test
     here feeds `permuted` already in `out_dtype`, where the cast is a no-op and
     could be moved anywhere without a single assertion noticing.
"""

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
    # float64 + a handful of addends: both orderings are exact, so require equality
    # rather than a tolerance that would also accept a wrong-but-close reduction.
    torch.testing.assert_close(got, want, rtol = 0, atol = 1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "atomics are a CUDA property")
def test_combine_is_bitwise_reproducible_in_bf16():
    # Large hidden dim and top_k > 1 so many threads contend for the same output
    # rows; that contention is exactly what made the old index_add_ drift.
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
    """A permutation is assumed; a combine that dropped or doubled slots would
    still look plausible on random data, so pin it with a counting case."""
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
    """fp32 summands, bf16 output: the one shape where cast placement is visible.

    An fp32 router (Gemma 4, DeepSeek V3) promotes the routing-weight multiply, so the
    routed slots reach the combine in fp32 while the model runs in bf16. Casting first
    rounds each of the top_k summands to bf16 and adds them in bf16; casting last adds
    in fp32 and rounds once, which is what transformers' own grouped_mm path does. The
    second assertion below exists so this test cannot quietly stop discriminating.
    """
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
    """The slot-count guard must be a raised exception, not a bare `assert`.

    `python -O` strips `assert`, and a build with the guard stripped is precisely the
    one where a short `sorted_indices` would silently read the wrong rows. Requiring
    ValueError also fails this test if the guard is reverted to an assert, which would
    raise AssertionError instead.
    """
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
    """The combine sits in the backward path of every MoE training step, so agreeing
    on the forward is not enough: a reduction that summed the right values through a
    different graph would pass every test above and still train a different model.

    Both spellings are gathers in reverse, so with the same upstream gradient this is
    an exact equality rather than a tolerance, in every dtype.
    """
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
