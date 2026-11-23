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

"""UNSLOTH_MOE_SELECTIVE_DEQUANT=1 and UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS=N dequantize only routed experts in bnb 4-bit
recompute; outputs and grads match the full-stack path bit for bit."""
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
    # These pin the dequant path they test: the fused NF4 GEMM (UNSLOTH_MOE_NF4_FUSED) dequantizes nothing.
    for key in ("UNSLOTH_MOE_SELECTIVE_DEQUANT", "UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS", "UNSLOTH_MOE_RECOMPUTE", "UNSLOTH_MOE_NF4_FUSED"):
        monkeypatch.delenv(key, raising = False)
    if DEVICE == "cpu":
        monkeypatch.setattr(mu, "_check_torch_grouped_mm_supported", lambda: True)
        monkeypatch.setattr(mu, "_transposed_view_grouped_mm_is_safe", lambda: True)
    for key in mu.SELECTIVE_DEQUANT_STATS:
        mu.SELECTIVE_DEQUANT_STATS[key] = 0
    monkeypatch.setattr(mu, "_AUTO_DEQUANT_TIGHT", set(), raising = False)


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


def test_chunked_without_selective_skips_empty_chunks(monkeypatch):
    # UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS alone covers every expert; chunks no token reaches are skipped rather than dequantized and multiplied by zero rows.
    torch.manual_seed(0)
    experts = _Experts(False)
    T = 16
    hidden = torch.randn(T, H, dtype = torch.bfloat16, device = DEVICE)
    pool = torch.tensor([0, 2, 5, 7], device = DEVICE)   # only chunks [0, 4) and [4, 8)
    top_k_index = pool[torch.stack([torch.randperm(4, device = DEVICE)[:TOP_K] for _ in range(T)])]
    top_k_weights = torch.rand(T, TOP_K, dtype = torch.bfloat16, device = DEVICE)
    lora = {
        "gate_up_proj": (torch.randn(E, H, RANK, device = DEVICE) * 0.1, torch.randn(E, RANK, 2 * I, device = DEVICE) * 0.1),
        "down_proj": (torch.randn(E, I, RANK, device = DEVICE) * 0.1, torch.randn(E, RANK, H, device = DEVICE) * 0.1),
    }
    base_out, base_grads = _run(experts, hidden, top_k_index, top_k_weights, lora, monkeypatch)
    monkeypatch.setenv("UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS", "4")
    out, grads = _run(experts, hidden, top_k_index, top_k_weights, lora, monkeypatch)
    assert torch.equal(out, base_out)
    for got, want in zip(grads, base_grads):
        assert torch.equal(got, want)
    stats = mu.SELECTIVE_DEQUANT_STATS
    # gate_up + down, forward and backward, two non-empty chunks of 4 experts each.
    assert stats["dequant_calls"] == 4 * 2
    assert stats["experts_dequantized"] == 4 * 8

    sources = (experts.gate_up_proj, experts.down_proj)
    assert mu._selective_dequant_plan(sources, torch.zeros(E, dtype = torch.int32, device = DEVICE), 0) is None


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("T", [1, 3, 5])
def test_default_capped_plan_is_sync_free_and_bit_exact(nested, T, monkeypatch):
    torch.manual_seed(T)
    experts = _Experts(nested)
    hidden = torch.randn(T, H, dtype = torch.bfloat16, device = DEVICE)
    pool = torch.tensor([2, 7, 11, 12, 15], device = DEVICE)
    top_k_index = pool[torch.stack([torch.randperm(5, device = DEVICE)[:TOP_K] for _ in range(T)])]
    top_k_weights = torch.rand(T, TOP_K, dtype = torch.bfloat16, device = DEVICE)
    lora = {
        "gate_up_proj": (torch.randn(E, H, RANK, device = DEVICE) * 0.1, torch.randn(E, RANK, 2 * I, device = DEVICE) * 0.1),
        "down_proj": (torch.randn(E, I, RANK, device = DEVICE) * 0.1, torch.randn(E, RANK, H, device = DEVICE) * 0.1),
    }
    monkeypatch.setenv("UNSLOTH_MOE_SELECTIVE_DEQUANT", "0")
    base_out, base_grads = _run(experts, hidden, top_k_index, top_k_weights, lora, monkeypatch)
    assert mu.SELECTIVE_DEQUANT_STATS["dequant_calls"] == 0
    monkeypatch.delenv("UNSLOTH_MOE_SELECTIVE_DEQUANT")

    host_reads = []
    item, tolist = torch.Tensor.item, torch.Tensor.tolist
    monkeypatch.setattr(torch.Tensor, "item", lambda self: host_reads.append("item") or item(self))
    monkeypatch.setattr(torch.Tensor, "tolist", lambda self: host_reads.append("tolist") or tolist(self))
    out, grads = _run(experts, hidden, top_k_index, top_k_weights, lora, monkeypatch)
    monkeypatch.setattr(torch.Tensor, "item", item)
    monkeypatch.setattr(torch.Tensor, "tolist", tolist)

    assert host_reads == []
    assert torch.equal(out, base_out)
    for got, want in zip(grads, base_grads):
        assert torch.equal(got, want)
    stats = mu.SELECTIVE_DEQUANT_STATS
    assert stats["plans"] == 1
    assert stats["dequant_calls"] == 4
    assert stats["experts_dequantized"] == 4 * T * TOP_K


def test_default_capped_plan_declines_when_rows_cover_the_stack(monkeypatch):
    experts = _Experts(False)
    sources = (experts.gate_up_proj, experts.down_proj)
    counts = torch.zeros(E, dtype = torch.int32, device = DEVICE)
    counts[:2] = 7
    assert mu._selective_dequant_plan(sources, counts, 14) is not None   # 14 of 16: exactly 7/8
    counts[2] = 1
    assert mu._selective_dequant_plan(sources, counts, 15) is None
    assert mu._selective_dequant_plan(sources, torch.zeros(E, dtype = torch.int32, device = DEVICE), 0) is None
    monkeypatch.setenv("UNSLOTH_MOE_SELECTIVE_DEQUANT", "0")
    assert mu._selective_dequant_plan(sources, counts, 2) is None
    plain = torch.zeros(E, 2 * I, H, dtype = torch.bfloat16, device = DEVICE)
    monkeypatch.delenv("UNSLOTH_MOE_SELECTIVE_DEQUANT")
    assert mu._selective_dequant_plan((plain,), counts, 2) is None
    counts = torch.zeros(E, dtype = torch.int32, device = DEVICE)
    counts[5], counts[9] = 2, 1   # unrouted expert pads
    ids, row_lo, row_hi, offsets = mu._selective_dequant_plan(sources, counts, 3).chunks[0]
    assert (row_lo, row_hi) == (0, 3) and ids.tolist() == [5, 9, 0] and offsets.tolist() == [2, 3, 3]


def _routed_all(T, monkeypatch):
    torch.manual_seed(1)
    hidden = torch.randn(T, H, dtype = torch.bfloat16, device = DEVICE)
    top_k_index = torch.stack([torch.randperm(E, device = DEVICE)[:TOP_K] for _ in range(T)])
    top_k_weights = torch.rand(T, TOP_K, dtype = torch.bfloat16, device = DEVICE)
    lora = {
        "gate_up_proj": (torch.randn(E, H, RANK, device = DEVICE) * 0.1, torch.randn(E, RANK, 2 * I, device = DEVICE) * 0.1),
        "down_proj": (torch.randn(E, I, RANK, device = DEVICE) * 0.1, torch.randn(E, RANK, H, device = DEVICE) * 0.1),
    }
    return hidden, top_k_index, top_k_weights, lora


def test_scratch_cap_chunks_the_whole_stack_bit_exactly(monkeypatch):
    """UNSLOTH_MOE_DEQUANT_MAX_MB bounds one projection's dequantized stack: every expert is routed (the auto plan
    declines), yet the stack is read in chunks and the result is the whole-stack one bit for bit."""
    monkeypatch.delenv("UNSLOTH_MOE_DEQUANT_MAX_MB", raising = False)
    experts = _Experts(True)
    hidden, top_k_index, top_k_weights, lora = _routed_all(64, monkeypatch)
    base_out, base_grads = _run(experts, hidden, top_k_index, top_k_weights, lora, monkeypatch)
    assert mu.SELECTIVE_DEQUANT_STATS["dequant_calls"] == 0
    per_expert = 2 * I * H * 2   # gate_up, bf16: the larger stack
    monkeypatch.setenv("UNSLOTH_MOE_DEQUANT_MAX_MB", str(5 * per_expert / 2**20))
    out, grads = _run(experts, hidden, top_k_index, top_k_weights, lora, monkeypatch)
    assert torch.equal(out, base_out)
    for got, want in zip(grads, base_grads):
        assert torch.equal(got, want)
    stats = mu.SELECTIVE_DEQUANT_STATS
    assert stats["plans"] == 1
    assert stats["dequant_calls"] == 4 * -(-E // 5)   # 5 experts per chunk, gate_up + down, forward + backward
    # An explicit chunk setting wins, 0 included.
    for key in stats:
        stats[key] = 0
    monkeypatch.setenv("UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS", "0")
    _run(experts, hidden, top_k_index, top_k_weights, lora, monkeypatch)
    assert stats["dequant_calls"] == 0


def _headroom(monkeypatch, gib, calls = None):
    def fake(index):
        if calls is not None:
            calls.append(index)
        free = int(gib * 2**30)
        return free, lambda: free
    monkeypatch.setattr(mu, "_device_headroom_bytes", fake, raising = False)


class _Flagged:
    _unsloth_dequant_scratch_fraction = 0.025


MINI = [(128, (2304, 2304)), (128, (2304, 1152))]                      # Inkling-2-Mini gate_up, down


def test_scratch_fraction_follows_the_experts_class_and_the_card(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MOE_DEQUANT_MAX_MB", raising = False)
    cuda0 = torch.device("cuda", 0)
    monkeypatch.setitem(mu._DEVICE_TOTAL_BYTES, 0, int(14.56 * 2**30))   # a T4
    _headroom(monkeypatch, 1.0)                                         # memory is short
    cap = 0.025 * 14.56 * 2**30
    assert mu._auto_dequant_chunk_experts(MINI, torch.float16, object(), cuda0) == 0     # unflagged: whole stack
    assert mu._auto_dequant_chunk_experts(MINI, torch.float16, _Flagged(), torch.device("cpu")) == 0
    chunk = mu._auto_dequant_chunk_experts(MINI, torch.float16, _Flagged(), cuda0)
    assert chunk == int(cap // (2304 * 2304 * 2)) == 36   # 4 chunks
    # Stacks up to twice the cap stay whole without reading free memory: Mini's down stack alone (0.63 GiB), or Mini
    # on a 40 GB card.
    mu._AUTO_DEQUANT_TIGHT.clear()
    calls = []
    _headroom(monkeypatch, 0.0, calls)
    assert mu._auto_dequant_chunk_experts(MINI[1:], torch.float16, _Flagged(), cuda0) == 0
    monkeypatch.setitem(mu._DEVICE_TOTAL_BYTES, 0, int(39.4 * 2**30))
    assert mu._auto_dequant_chunk_experts(MINI, torch.float16, _Flagged(), cuda0) == 0
    assert calls == []
    small = [(384, (4608, 4608)), (384, (4608, 2304))]                 # Inkling-2-Small: 15.2 GiB gate_up stack
    monkeypatch.setitem(mu._DEVICE_TOTAL_BYTES, 0, int(79.2 * 2**30))
    _headroom(monkeypatch, 20.0)
    assert mu._auto_dequant_chunk_experts(small, torch.float16, _Flagged(), cuda0) == int(0.025 * 79.2 * 2**30 // (4608 * 4608 * 2))


def test_auto_chunking_waits_for_memory_need(monkeypatch):
    """A 24 GB card holds Mini's 1.27 GiB stack while memory is free: no chunking, so no host sync per MoE call. Once
    the stack does not fit, that device chunks from then on; other devices are unaffected."""
    monkeypatch.delenv("UNSLOTH_MOE_DEQUANT_MAX_MB", raising = False)
    for index in (0, 1):
        monkeypatch.setitem(mu._DEVICE_TOTAL_BYTES, index, int(21.96 * 2**30))   # L4
    cuda0, cuda1 = torch.device("cuda", 0), torch.device("cuda", 1)
    _headroom(monkeypatch, 8.0)
    assert mu._auto_dequant_chunk_experts(MINI, torch.float16, _Flagged(), cuda0) == 0
    # Twice the scratch plus one cap of slack (2.54 + 0.55 GiB) is the bar.
    _headroom(monkeypatch, 3.2)
    assert mu._auto_dequant_chunk_experts(MINI, torch.float16, _Flagged(), cuda0) == 0
    _headroom(monkeypatch, 2.5)
    chunk = mu._auto_dequant_chunk_experts(MINI, torch.float16, _Flagged(), cuda0)
    assert chunk == int(0.025 * 21.96 * 2**30 // (2304 * 2304 * 2)) == 55
    calls = []
    _headroom(monkeypatch, 8.0, calls)
    assert mu._auto_dequant_chunk_experts(MINI, torch.float16, _Flagged(), cuda0) == 55   # stays chunked
    assert calls == []
    assert mu._auto_dequant_chunk_experts(MINI, torch.float16, _Flagged(), cuda1) == 0
    # The unchunked plan's real scratch decides: a sync-free plan over 16 of 128 experts fits anywhere.
    mu._AUTO_DEQUANT_TIGHT.clear()
    _headroom(monkeypatch, 0.5)
    assert mu._auto_dequant_chunk_experts(MINI, torch.float16, _Flagged(), cuda0, experts_needed = 16) == 0


def test_unflagged_models_skip_the_scratch_cap_prep(monkeypatch):
    """An experts class without _unsloth_dequant_scratch_fraction (and no UNSLOTH_MOE_DEQUANT_MAX_MB) does no
    scratch-cap work in the plan: no 4-bit split of its sources, no total-memory read."""
    monkeypatch.delenv("UNSLOTH_MOE_DEQUANT_MAX_MB", raising = False)
    experts = _Experts(False)
    sources = (experts.gate_up_proj, experts.down_proj)
    calls = []
    real = mu._bnb4bit_expert_split
    monkeypatch.setattr(mu, "_bnb4bit_expert_split", lambda param: calls.append(1) or real(param))
    counts = torch.ones(E, dtype = torch.int32, device = DEVICE)
    assert mu._selective_dequant_plan(sources, counts, E, experts_module = experts) is None   # every expert routed
    assert calls == []


def test_scratch_cap_keeps_a_sparse_plan_that_already_fits(monkeypatch):
    """Two routed rows under a five-expert cap: the sync-free plan reads two experts, inside the cap, so it is kept
    rather than replaced by a chunked whole-stack read (two chunks, six experts and a host sync)."""
    experts = _Experts(True)
    sources = (experts.gate_up_proj, experts.down_proj)
    per_expert = 2 * I * H * 2
    monkeypatch.setenv("UNSLOTH_MOE_DEQUANT_MAX_MB", str(5 * per_expert / 2**20))
    counts = torch.zeros(E, dtype = torch.int32, device = DEVICE)
    counts[1], counts[9] = 1, 1
    plan = mu._selective_dequant_plan(sources, counts, 2, experts_module = experts)
    assert plan is not None and len(plan.chunks) == 1
    ids, row_lo, row_hi, _ = plan.chunks[0]
    assert isinstance(ids, torch.Tensor) and ids.tolist() == [1, 9] and (row_lo, row_hi) == (0, 2)
    # A plan over the cap is still chunked.
    counts = torch.ones(E, dtype = torch.int32, device = DEVICE)
    plan = mu._selective_dequant_plan(sources, counts, E, experts_module = experts)
    assert [c[0] for c in plan.chunks] == [slice(0, 5), slice(5, 10), slice(10, 15), slice(15, 16)]
