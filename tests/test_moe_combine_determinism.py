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
index_add_'s atomicAdd path, which fixes no accumulation order; in bf16 the
run-to-run drift was large enough to move a Gemma-4-26B-A4B forward loss by
0.44 nats between two identical forwards on one loaded model.

Two things are asserted here, because either alone is easy to pass by accident:
  1. equivalence - the replacement computes the same sum as the index_add_ form
     (checked in float64, where both are exact);
  2. reproducibility - repeated calls on identical input are bitwise identical
     in bf16, with a duplicate-heavy index pattern.
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
