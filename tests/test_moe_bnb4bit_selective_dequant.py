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

"""UNSLOTH_MOE_SELECTIVE_DEQUANT=1 / UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS=N: the bnb 4-bit recompute
path dequantizes only routed experts (or a few at a time) and matches the full-stack path bit
for bit in the output, the input / routing-weight grads and the expert LoRA grads."""
import os

import pytest
import torch
import torch.nn as nn

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

bnb = pytest.importorskip("bitsandbytes")
from bitsandbytes.nn import Params4bit

import unsloth_zoo.temporary_patches.moe_utils as mu

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
E, H, I, TOP_K, RANK = 16, 128, 64, 2, 4


def _quantized(shape, nested):
    w = torch.randn(*shape, dtype = torch.bfloat16) * 0.05
    p = Params4bit(w, requires_grad = False, quant_type = "nf4", compress_statistics = nested).to(DEVICE)
    p._original_shape = torch.Size(shape)
    return p


class _Experts(nn.Module):
    def __init__(self, nested):
        super().__init__()
        self.num_experts = E
        # gate_up: 256 absmax blocks per expert (nested blocks align); down: 128 (they straddle).
        self.gate_up_proj = _quantized((E, 2 * I, H), nested)
        self.down_proj = _quantized((E, H, I), nested)
        self.act_fn = nn.SiLU()


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    for key in ("UNSLOTH_MOE_SELECTIVE_DEQUANT", "UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS", "UNSLOTH_MOE_RECOMPUTE"):
        monkeypatch.delenv(key, raising = False)
    if DEVICE == "cpu":
        monkeypatch.setattr(mu, "_check_torch_grouped_mm_supported", lambda: True)
        monkeypatch.setattr(mu, "_transposed_view_grouped_mm_is_safe", lambda: True)
    for key in mu.SELECTIVE_DEQUANT_STATS:
        mu.SELECTIVE_DEQUANT_STATS[key] = 0


@pytest.mark.parametrize("nested", [False, True])
def test_subset_dequant_matches_full_stack(nested, monkeypatch):
    torch.manual_seed(0)
    experts = _Experts(nested)
    for param in (experts.gate_up_proj, experts.down_proj):
        full = mu._get_base_weight(param, torch.bfloat16)
        for ids in (torch.tensor([7], device = DEVICE), torch.tensor([1, 4, 5, 11, 15], device = DEVICE), slice(2, 9)):
            sub = mu._get_base_weight(param, torch.bfloat16, experts = ids)
            assert torch.equal(sub, full[ids])
        monkeypatch.setattr(mu, "_SELECTIVE_GATHER_MIN", 1)   # several pieces decoded into one output
        ids = torch.tensor([0, 2, 3, 6, 8, 9, 10, 12, 13, 15], device = DEVICE)
        assert torch.equal(mu._get_base_weight(param, torch.bfloat16, experts = ids), full[ids])
        monkeypatch.setattr(mu, "_SELECTIVE_GATHER_MIN", 32)


def _run(experts, hidden, top_k_index, top_k_weights, lora, monkeypatch):
    stash = {
        name: (first.detach().clone().requires_grad_(), second.detach().clone().requires_grad_(), 2.0)
        for name, (first, second) in lora.items()
    }
    monkeypatch.setattr(mu, "take_moe_lora_stash", lambda module, name: stash.get(name))
    hidden = hidden.detach().clone().requires_grad_()
    weights = top_k_weights.detach().clone().requires_grad_()
    out = mu.forward_native_grouped_mm(experts, hidden, top_k_index, weights)
    out.float().square().sum().backward()
    grads = [hidden.grad, weights.grad] + [t.grad for name in sorted(stash) for t in stash[name][:2]]
    return out.detach(), grads


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("mode", ["selective", "chunked", "selective_chunked"])
def test_forward_and_grads_match_full_stack(nested, mode, monkeypatch):
    torch.manual_seed(0)
    experts = _Experts(nested)
    T = 24
    hidden = torch.randn(T, H, dtype = torch.bfloat16, device = DEVICE)
    # Route to 6 of the 16 experts, with gaps, so the compact map is not a prefix.
    pool = torch.tensor([0, 3, 4, 9, 10, 14], device = DEVICE)
    top_k_index = pool[torch.stack([torch.randperm(6, device = DEVICE)[:TOP_K] for _ in range(T)])]
    top_k_weights = torch.rand(T, TOP_K, dtype = torch.bfloat16, device = DEVICE)
    lora = {
        "gate_up_proj": (torch.randn(E, H, RANK, device = DEVICE) * 0.1, torch.randn(E, RANK, 2 * I, device = DEVICE) * 0.1),
        "down_proj": (torch.randn(E, I, RANK, device = DEVICE) * 0.1, torch.randn(E, RANK, H, device = DEVICE) * 0.1),
    }

    base_out, base_grads = _run(experts, hidden, top_k_index, top_k_weights, lora, monkeypatch)
    assert mu.SELECTIVE_DEQUANT_STATS["dequant_calls"] == 0   # off by default: the old path

    if mode in ("selective", "selective_chunked"):
        monkeypatch.setenv("UNSLOTH_MOE_SELECTIVE_DEQUANT", "1")
    if mode in ("chunked", "selective_chunked"):
        monkeypatch.setenv("UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS", "4")
    out, grads = _run(experts, hidden, top_k_index, top_k_weights, lora, monkeypatch)

    assert torch.equal(out, base_out)
    for got, want in zip(grads, base_grads):
        assert torch.equal(got, want)
    # Unrouted experts get exactly zero LoRA grad on both paths.
    unused = torch.ones(E, dtype = torch.bool, device = DEVICE)
    unused[pool] = False
    for g in grads[2:]:
        assert g[unused].abs().sum().item() == 0

    stats = mu.SELECTIVE_DEQUANT_STATS
    assert stats["plans"] == 1
    # gate_up + down, each read in forward and again in backward.
    per_read = 6 if mode != "chunked" else E
    assert stats["experts_dequantized"] == 4 * per_read
    assert stats["experts_total"] == stats["dequant_calls"] * E
    if mode == "selective":
        assert stats["dequant_calls"] == 4
    else:
        assert stats["dequant_calls"] == 4 * -(-per_read // 4)


def test_nearly_all_experts_routed_keeps_the_full_stack_path(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_SELECTIVE_DEQUANT", "1")
    experts = _Experts(False)
    sources = (experts.gate_up_proj, experts.down_proj)
    counts = torch.ones(E, dtype = torch.int32, device = DEVICE)
    assert mu._selective_dequant_plan(sources, counts, E) is None
    counts[3] = 0   # 15 of 16 routed: above 7/8
    assert mu._selective_dequant_plan(sources, counts, E - 1) is None
    counts[4] = 0   # 14 of 16
    plan = mu._selective_dequant_plan(sources, counts, E - 2)
    assert plan is not None and plan.chunks[0][0].numel() == E - 2
    monkeypatch.setenv("UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS", "8")
    plan = mu._selective_dequant_plan(sources, torch.ones(E, dtype = torch.int32, device = DEVICE), E)
    assert [c[0] for c in plan.chunks] == [slice(0, 8), slice(8, 16)]


def test_plan_declines_unsplittable_or_plain_weights(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_SELECTIVE_DEQUANT", "1")
    counts = torch.tensor([2] + [0] * (E - 1), dtype = torch.int32, device = DEVICE)
    plain = torch.zeros(E, 2 * I, H, dtype = torch.bfloat16, device = DEVICE)
    assert mu._selective_dequant_plan((plain,), counts, 2) is None
    odd = _quantized((E, 3, 7), False)   # 21 weights per expert: blocks straddle experts
    assert mu._selective_dequant_plan((odd,), counts, 2) is None
