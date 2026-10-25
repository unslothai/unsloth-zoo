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

"""Selective-dequant backward: a transposed view (no copy) where the slice reads views exactly, else the copy; dX bitwise."""
import pytest
import torch

import unsloth_zoo.temporary_patches.moe_utils as mu

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16


def _case(chunks_split):
    gen = torch.Generator().manual_seed(0)
    E, K, N = 8, 64, 96
    counts = torch.tensor([5, 0, 7, 3, 0, 6, 2, 0])
    W = (torch.randn(E, K, N, generator = gen) * 0.1).to(DEVICE, DT)
    x = torch.randn(int(counts.sum()), K, generator = gen).to(DEVICE, DT)
    dy = torch.randn(x.shape[0], N, generator = gen).to(DEVICE, DT)
    routed = [e for e in range(E) if counts[e] > 0]
    ends = counts.cumsum(0)
    chunks = []
    groups = [routed[:chunks_split], routed[chunks_split:]] if chunks_split else [routed]
    for ids in groups:
        lo = int(ends[ids[0]] - counts[ids[0]])
        hi = int(ends[ids[-1]])
        offs = torch.tensor([int(ends[e]) - lo for e in ids], dtype = torch.int32, device = DEVICE)
        chunks.append((torch.tensor(ids, device = DEVICE), lo, hi, offs))
    plan = mu._SelectiveDequantPlan(E, x.shape[0], chunks)
    provided = []

    def provider(experts):
        weight = W.index_select(0, experts)
        provided.append(weight.untyped_storage().data_ptr())
        return weight

    return x, dy, W, plan, provider, provided


def _backward_weights(chunks_split, monkeypatch, reads_views):
    x, dy, W, plan, provider, provided = _case(chunks_split)
    monkeypatch.setattr(mu, "_slice_reads_views", lambda *args: reads_views)
    seen = []
    real = mu._grouped_mm_slice

    def spy(inputs, weight, offsets, plan):
        seen.append((weight.untyped_storage().data_ptr(), weight.is_contiguous()))
        return real(inputs, weight, offsets, plan)

    monkeypatch.setattr(mu, "_grouped_mm_slice", spy)
    xg = x.clone().requires_grad_(True)
    mu._GroupedMMSelectiveRecompute.apply(xg, provider, plan).backward(dy)
    n = len(plan.chunks)
    return seen[n:], provided[n:]


@pytest.mark.parametrize("chunks_split", [0, 2])
def test_backward_passes_the_planned_experts_as_a_view(chunks_split, monkeypatch):
    seen, provided = _backward_weights(chunks_split, monkeypatch, True)
    assert [p for p, _ in seen] == provided   # same storage: no copy
    assert not any(contiguous for _, contiguous in seen)


@pytest.mark.parametrize("chunks_split", [0, 2])
def test_per_group_fallback_keeps_the_contiguous_copy(chunks_split, monkeypatch):
    seen, provided = _backward_weights(chunks_split, monkeypatch, False)
    assert all(contiguous for _, contiguous in seen)
    assert not set(p for p, _ in seen) & set(provided)


@pytest.mark.parametrize("chunks_split", [0, 2])
def test_view_backward_matches_the_contiguous_copy_bitwise(chunks_split):
    x, dy, W, plan, provider, _ = _case(chunks_split)
    xg = x.clone().requires_grad_(True)
    mu._GroupedMMSelectiveRecompute.apply(xg, provider, plan).backward(dy)

    def copied(experts):
        return provider(experts).transpose(-2, -1).contiguous()

    with torch.no_grad():
        want = mu._selective_grouped_mm(dy, copied, plan, False)
    assert torch.equal(xg.grad, want)
