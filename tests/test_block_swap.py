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

"""Tests unsloth_zoo.block_swap. The scheduling is driven through fake blocks so it
runs on a CPU-only runner; the one case that needs a real Params4bit round trip
skips without a GPU.

The module is loaded from its file, not through the package: ``import unsloth_zoo``
refuses to run without unsloth installed, and the conftest stubs torch.cuda for
import, neither of which this module needs or should be exercised through."""

import importlib.util
import os

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "block_swap_under_test", os.path.join(_HERE, "unsloth_zoo", "block_swap.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BlockSwap, find_decoder_layers, swap_indices = _mod.BlockSwap, _mod.find_decoder_layers, _mod.swap_indices
auto_swap_indices, estimate_training_reserve_bytes = _mod.auto_swap_indices, _mod.estimate_training_reserve_bytes


class _FakeBlock:
    """Just enough of _Block for the scheduler: residency, a slot, a signature."""

    def __init__(self, sig = "a"):
        self.sig = sig
        self.home = None
        self.resident = True
        self.slot = None
        self.pending = False
        self.fetches = 0

    def evict(self):
        if not self.resident:
            return None
        self.resident = False
        self.pending = False
        freed, self.slot = self.slot, None
        return freed

    def prefetch(self, slot):
        if self.resident:
            return
        self.fetches += 1
        self.slot = slot
        self.resident = True

    def wait(self):
        pass


def _scheduler(n, depth = 2, sigs = None, start = 0):
    """A BlockSwap with fake blocks and pools, no CUDA touched."""
    sw = BlockSwap.__new__(BlockSwap)
    sw.depth = depth
    sw.handles = []
    sigs = sigs or ["a"] * n
    sw.blocks = [_FakeBlock(s) for s in sigs]
    sw.free = {s: [object() for _ in range(depth + 1)] for s in set(sigs)}
    sw.start = start
    sw.indices = list(range(start, start + n))
    sw.pos = {li: k for k, li in enumerate(sw.indices)}
    sw._input_hooked = [False] * n
    sw._grew = False
    sw._new_slot = lambda sig: object()
    for b in sw.blocks:
        sw._release(b)
    sw._arm(forward = True)
    return sw


def _live(sw, sig = "a"):
    return sum(1 for b in sw.blocks if b.sig == sig and b.resident)


def test_zero_layers_installs_nothing():
    layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
    sw = BlockSwap(layers, 0)
    assert sw.blocks == [] and sw.handles == []
    assert all(len(l._forward_pre_hooks) == 0 for l in layers)
    assert all(len(l._forward_hooks) == 0 for l in layers)


def test_forward_arm_prefetches_leading_blocks():
    sw = _scheduler(8, depth = 2)
    assert [b.resident for b in sw.blocks] == [True, True] + [False] * 6


def test_pool_never_exceeds_depth_plus_one_live():
    sw = _scheduler(8, depth = 2)
    with torch.no_grad():
        for i in range(8):
            sw._pre(i)(None, (), {})
            assert _live(sw) <= sw.depth + 1, f"step {i}: {_live(sw)} live"
            sw._post(i)(None, None, None)
    # The last block's eviction starts fetching the next inference forward's first blocks.
    assert [b.resident for b in sw.blocks] == [True] * sw.depth + [False] * (len(sw.blocks) - sw.depth)


def test_prefetch_direction_flips_with_grad_mode():
    sw = _scheduler(8, depth = 2)
    for b in sw.blocks:
        sw._release(b)
    # grad off: entering block 3 looks ahead to 5.
    with torch.no_grad():
        sw._pre(3)(None, (), {})
    assert sw.blocks[3].resident and sw.blocks[5].resident and not sw.blocks[1].resident
    for b in sw.blocks:
        sw._release(b)
    # grad on: the recompute sweep runs in reverse, so entering 3 looks back to 1.
    with torch.enable_grad():
        sw._pre(3)(None, (), {})
    assert sw.blocks[3].resident and sw.blocks[1].resident and not sw.blocks[5].resident


def test_post_hook_evicts_only_under_no_grad():
    # Not block 0: its backward hook re-arms and refetches it.
    sw = _scheduler(6, depth = 3)
    for b in sw.blocks:
        sw._release(b)
    with torch.no_grad():
        sw._pre(2)(None, (), {})
        assert sw.blocks[2].resident
        sw._post(2)(None, None, None)
    assert not sw.blocks[2].resident, "under no_grad the post hook evicts"
    for b in sw.blocks:
        sw._release(b)
    with torch.enable_grad():
        x = torch.ones(1, requires_grad = True)
        sw._pre(2)(None, (x,), {})
        assert sw.blocks[2].resident
        sw._post(2)(None, None, x * 2)
    assert sw.blocks[2].resident, "with grad on, eviction waits for the backward hook"
    sw._bwd(2)(None)
    assert not sw.blocks[2].resident


def test_backward_hook_on_block_zero_arms_next_step():
    sw = _scheduler(6, depth = 2)
    for b in sw.blocks:
        sw._release(b)
    assert _live(sw) == 0
    sw._bwd(3)(None)
    assert _live(sw) == 0, "only block 0's backward marks the step boundary"
    sw._bwd(0)(None)
    assert [b.resident for b in sw.blocks[:2]] == [True, True]


def test_each_signature_gets_its_own_pool():
    sigs = ["a", "a", "b", "a", "b", "b"]
    sw = _scheduler(6, depth = 1, sigs = sigs)
    assert set(sw.free) == {"a", "b"}
    with torch.no_grad():
        for i in range(6):
            sw._pre(i)(None, (), {})
            assert _live(sw, "a") <= 2 and _live(sw, "b") <= 2
            sw._post(i)(None, None, None)


def test_slot_returns_to_the_pool_it_came_from():
    sw = _scheduler(4, depth = 1, sigs = ["a", "b", "a", "b"])
    for b in sw.blocks:
        sw._release(b)
    n_a, n_b = len(sw.free["a"]), len(sw.free["b"])
    sw._fetch(sw.blocks[1])
    assert len(sw.free["b"]) == n_b - 1 and len(sw.free["a"]) == n_a
    sw._release(sw.blocks[1])
    assert len(sw.free["b"]) == n_b and len(sw.free["a"]) == n_a


def test_enter_leave_walks_a_decode_step():
    # enter/leave take full-list indices; the swapped tail starts at 4.
    sw = _scheduler(6, depth = 2, start = 4)
    for b in sw.blocks:
        sw._release(b)
    with torch.no_grad():
        for idx in range(10):
            sw.enter(idx)
            if idx >= 4:
                assert sw.blocks[idx - 4].resident
            assert _live(sw) <= sw.depth + 1
            sw.leave(idx)
    # Leaving the last swapped layer arms the next step's leading fetches.
    assert [b.resident for b in sw.blocks] == [True, True] + [False] * 4


def test_enter_leave_ignore_layers_on_the_card():
    sw = _scheduler(4, depth = 2, start = 10)
    before = [b.resident for b in sw.blocks]
    sw.enter(3)
    sw.leave(3)
    assert [b.resident for b in sw.blocks] == before


def test_interrupted_step_never_evicts_the_block_about_to_run():
    sw = _scheduler(8, depth = 2)
    for i in range(8):
        with torch.no_grad():
            sw._pre(i)(None, (), {})
            sw._post(i)(None, None, None)
    # Recompute of the top blocks starts, then backward dies: their slots stay held.
    with torch.enable_grad():
        sw._pre(7)(None, (), {})
        sw._pre(6)(None, (), {})
    assert not sw.free["a"]
    with torch.no_grad():
        for i in range(8):
            x = torch.ones(1, requires_grad = True)
            sw._pre(i)(None, (x,), {})
            assert sw.blocks[i].resident, i
            sw._post(i)(None, None, x * 2)


def test_grad_on_forward_order_never_reuses_a_slot_backward_still_reads():
    # Plain forward (or several layers in one reentrant checkpoint): every block keeps its weights for backward.
    sw = _scheduler(8, depth = 2)
    slots = []
    with torch.enable_grad():
        for i in range(8):
            x = torch.ones(1, requires_grad = True)
            sw._pre(i)(None, (x,), {})
            assert sw.blocks[i].resident, i
            sw._post(i)(None, None, x * 2)
            slots.append(sw.blocks[i].slot)
    assert all(b.resident and b.pending for b in sw.blocks)
    assert len({id(s) for s in slots}) == 8, "a slot was handed to two blocks autograd still reads"
    for i in reversed(range(8)):
        sw._bwd(i)(None)
    assert not any(b.pending for b in sw.blocks)


def test_non_reentrant_checkpoint_forward_evicts_like_no_grad():
    import torch.utils.checkpoint as cp
    sw = _scheduler(6, depth = 2)
    x = torch.ones(1, requires_grad = True)

    def run(i):
        def f(t):
            sw._pre(i)(None, (), {})
            sw._post(i)(None, None, None)
            return t * 2
        return f

    h = x
    for i in range(6):
        h = cp.checkpoint(run(i), h, use_reentrant = False)
        assert not sw.blocks[i].resident and not sw.blocks[i].pending, i
    assert sw._grew is False


def test_enter_leave_are_no_ops_after_remove():
    # The fast decode loop keeps its reference after remove(); an emptied pool must not be indexed.
    sw = _scheduler(4, depth = 2)
    sw.streams = {}
    for b in sw.blocks:
        b.params, b.host, b.devices = [], [], []
    sw.remove()
    for i in range(4):
        sw.enter(i)
        sw.leave(i)
    sw.reset()


def test_training_forward_keeps_the_blocks_recompute_needs_first():
    class M:
        training = True
    sw = _scheduler(6, depth = 2)
    with torch.no_grad():
        for i in range(6):
            sw._pre(i)(M, (), {})
            sw._post(i)(M, None, None)
    assert [b.resident for b in sw.blocks] == [False] * 4 + [True, True]
    sw = _scheduler(6, depth = 2)
    M.training = False
    with torch.no_grad():
        for i in range(6):
            sw._pre(i)(M, (), {})
            sw._post(i)(M, None, None)
    assert not any(b.resident for b in sw.blocks[2:])


def test_find_decoder_layers_through_common_wrappers():
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(2, 2)])

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    class Peft(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = Wrapped()

    for m in (Inner(), Wrapped(), Peft()):
        assert len(find_decoder_layers(m)) == 1


def test_state_dict_substitutes_host_copies_for_evicted_blocks():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return
    torch.manual_seed(0)
    layers = nn.ModuleList([nn.Linear(8, 8, bias = False) for _ in range(4)]).to("cuda")
    ref = {k: v.clone() for k, v in layers.state_dict().items()}
    for p in layers.parameters():
        p.requires_grad_(False)
    sw = BlockSwap(layers, 4, prefetch_depth = 1)
    sd = layers.state_dict()
    for k, v in ref.items():
        assert sd[k].shape == v.shape, k
        assert torch.equal(sd[k].to(v.device), v), k
    sw.remove()


def test_state_dict_follows_weights_renamed_after_install():
    # from_pretrained(offload_layers = N) installs before PEFT wraps each Linear in a base_layer.
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    class Wrap(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base_layer = base

    torch.manual_seed(0)
    layers = nn.ModuleList([nn.Sequential(nn.Linear(8, 8, bias = False)) for _ in range(4)]).to("cuda")
    for p in layers.parameters():
        p.requires_grad_(False)
    sw = BlockSwap(layers, 4, prefetch_depth = 1)
    for layer in layers:
        layer[0] = Wrap(layer[0])
    ref = [layer[0].base_layer.weight for layer in layers]
    sd = layers.state_dict()
    for i, b in enumerate(sw.blocks):
        v = sd[f"{i}.0.base_layer.weight"]
        assert v.numel() == 64, i
        assert torch.equal(v, b.host[0]), i
    sw.remove()
    for i, w in enumerate(ref):
        assert w.numel() == 64, i


def test_params4bit_round_trip_is_bitwise():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return
    try:
        import bitsandbytes as bnb
    except ImportError:
        print("[SKIP] bitsandbytes not installed")
        return
    torch.manual_seed(0)
    lin = bnb.nn.Linear4bit(256, 256, bias = False, compute_dtype = torch.bfloat16,
                            quant_type = "nf4").to("cuda")
    x = torch.randn(4, 256, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        ref = lin(x).clone()
    layers = nn.ModuleList([lin])
    for p in lin.parameters():
        p.requires_grad_(False)
    sw = BlockSwap(layers, 1, prefetch_depth = 1)
    with torch.no_grad():
        out = lin(x)
    assert torch.equal(ref, out)
    sw.remove()
    with torch.no_grad():
        after = lin(x)
    assert torch.equal(ref, after)
    assert all(len(l._forward_pre_hooks) == 0 for l in layers)


def test_host_loaded_layer_is_packed_and_its_original_released():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return
    torch.manual_seed(0)
    lin = nn.Linear(64, 64, bias = False).requires_grad_(False)
    host = lin.weight.data.clone()
    lin.weight.data = host
    x = torch.randn(2, 64, device = "cuda")
    ref = x @ host.to("cuda").t()
    layers = nn.ModuleList([nn.Linear(64, 64).cuda(), lin])
    sw = BlockSwap(layers, 1, prefetch_depth = 1, device = "cuda")
    packed = sw.blocks[0].host[0]
    assert torch.equal(packed, host) and packed.data_ptr() != host.data_ptr()
    assert lin.weight.data_ptr() != host.data_ptr(), "the pageable original is not kept alongside the packed copy"
    with torch.no_grad():
        sw.enter(1)
        out = lin(x)
        sw.leave(1)
    assert torch.equal(ref, out)
    assert lin.weight.device.type == "cuda"


def _linear_tail():
    # 4.5 MiB blobs: torch's pinned allocator would hold 8 MiB for each.
    return nn.ModuleList([nn.Linear(1536, 1536, bias = False).cuda().requires_grad_(False) for _ in range(4)])


def test_host_copies_are_pinned_in_place_at_their_exact_size():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return
    if os.environ.get("UNSLOTH_DISABLE_PINNED_MEMORY", "0") == "1":
        print("[SKIP] pinned memory disabled")
        return
    layers = _linear_tail()
    x = torch.randn(2, 1536, device = "cuda")
    with torch.no_grad():
        ref = layers[3](x).clone()
    sw = BlockSwap(layers, 3, prefetch_depth = 1)
    need = sw.host_bytes()
    assert need == 3 * 1536 * 1536 * 4
    arena = sw._chunks[0]
    assert arena.numel() == need and sw.pinned_bytes == need, "pinned exactly, no power-of-two rounding"
    for b in sw.blocks:
        assert b.host[0].is_pinned()
        assert arena.data_ptr() <= b.host[0].data_ptr() < arena.data_ptr() + need
    with torch.no_grad():
        sw.enter(3)
        out = layers[3](x)
        sw.leave(3)
    assert torch.equal(ref, out)
    sw.remove()
    assert not arena.is_pinned(), "remove() unregisters the arena"


def test_partial_pinning_is_announced(capsys):
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return
    budget = _mod._pin_budget
    _mod._pin_budget = lambda: 0
    try:
        sw = BlockSwap(_linear_tail(), 3, prefetch_depth = 1)
    finally:
        _mod._pin_budget = budget
    assert sw.pinned_bytes == 0
    assert "offload_layers pinned 0.0 of" in capsys.readouterr().out
    sw.remove()


def test_chunked_fallback_when_registration_is_unavailable():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return
    orig = BlockSwap._pack_registered
    BlockSwap._pack_registered = lambda self: False
    try:
        sw = BlockSwap(_linear_tail(), 3, prefetch_depth = 1)
    finally:
        BlockSwap._pack_registered = orig
    need = sw.host_bytes()
    assert len(sw._chunks) == 1 and sw._chunks[0].numel() < 2 * need
    lo = sw._chunks[0].data_ptr()
    hi = lo + sw._chunks[0].numel()
    for b in sw.blocks:
        r = b.host_buf[b.devices[0]]
        assert lo <= r.data_ptr() and r.data_ptr() + r.numel() <= hi, "every block lives inside the shared chunk"
    sw.remove()




def test_swap_indices_spread_and_tail():
    assert swap_indices(8, 3, "tail") == [5, 6, 7]
    assert swap_indices(32, 8) == [3, 7, 11, 15, 19, 23, 27, 31]
    assert swap_indices(5, 5) == [0, 1, 2, 3, 4]
    assert swap_indices(5, 0) == [] and swap_indices(5, 9) == [0, 1, 2, 3, 4]
    for total in range(1, 40):
        for n in range(total + 1):
            idx = swap_indices(total, n)
            assert len(set(idx)) == n and all(0 <= i < total for i in idx) and idx == sorted(idx)


def test_find_decoder_layers_without_a_layers_attribute():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 16)

    class GPT2Like(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([Block() for _ in range(4)])
            self.transformer.heads = nn.ModuleList([nn.Linear(16, 2) for _ in range(2)])

    m = GPT2Like()
    assert find_decoder_layers(m) is m.transformer.h

    class Seq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.block = nn.ModuleList([Block() for _ in range(2)])
            self.decoder = nn.Module()
            self.decoder.block = nn.ModuleList([Block() for _ in range(3)])

    m = Seq2Seq()
    assert find_decoder_layers(m) is m.decoder.block


class _XBlock(nn.Module):
    """Self + cross term: `enc` reaches every block, like cross-attention states."""

    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, d, bias = False).requires_grad_(False)
        self.c = nn.Linear(d, d, bias = False).requires_grad_(False)
        self.lora = nn.Parameter(torch.randn(d, d) * 0.01)

    def forward(self, x, enc):
        return x + torch.tanh(self.w(x) + x @ self.lora + self.c(enc))


def _xrun(blocks, mode, swap, idx = None):
    import torch.utils.checkpoint as cp
    torch.manual_seed(0)
    x = torch.randn(4, 256, device = "cuda", requires_grad = True)
    enc = torch.randn(4, 256, device = "cuda", requires_grad = True)
    sw = BlockSwap(blocks, idx if idx is not None else 6, prefetch_depth = 1, device = "cuda") if swap else None
    for b in blocks:
        b.lora.grad = None
    h = x
    for b in blocks:
        if mode == "none":
            h = b(h, enc)
        else:
            h = cp.checkpoint(b, h, enc, use_reentrant = mode == "reentrant")
    h.square().sum().backward()
    torch.cuda.synchronize()
    out = [x.grad.clone(), enc.grad.clone()] + [b.lora.grad.clone() for b in blocks]
    live = sum(blk.resident for blk in sw.blocks) if sw is not None else None
    if sw is not None:
        sw.remove()
    return out, live


def test_cross_fed_tensor_grads_are_bitwise_with_swap():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    blocks = nn.ModuleList([_XBlock(256) for _ in range(8)]).cuda()
    for mode in ("reentrant", "non_reentrant", "none"):
        ref, _ = _xrun(blocks, mode, False)
        for idx in (None, [1, 3, 5, 7], [0, 2, 4, 6]):
            got, live = _xrun(blocks, mode, True, idx)
            assert all(torch.equal(a, b) for a, b in zip(ref, got)), (mode, idx)
            # Released after backward except the next step's first blocks.
            assert live == 1, (mode, idx, live)


def test_block_without_grad_input_is_released_after_backward():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    blocks = nn.ModuleList([_XBlock(256) for _ in range(4)]).cuda()
    sw = BlockSwap(blocks, 4, prefetch_depth = 1, device = "cuda")
    x = torch.randn(4, 256, device = "cuda")  # no grad: block 0's input hook cannot fire
    enc = torch.randn(4, 256, device = "cuda")
    h = x
    for b in blocks:
        h = b(h, enc)
    assert sw.blocks[0].pending
    h.sum().backward()
    torch.cuda.synchronize()
    assert not any(b.pending for b in sw.blocks)
    assert blocks[0].lora.grad is not None
    sw.remove()


def test_shared_parameter_stays_on_the_card():
    if not torch.cuda.is_available():
        return
    shared = nn.Linear(64, 64, bias = False).cuda().requires_grad_(False)
    layers = nn.ModuleList([nn.Sequential(nn.Linear(64, 64, bias = False), shared) for _ in range(3)])
    layers = layers.cuda().requires_grad_(False)
    x = torch.randn(2, 64, device = "cuda")
    with torch.no_grad():
        ref = [layer(x) for layer in layers]
    sw = BlockSwap(layers, 2, prefetch_depth = 1, device = "cuda")
    assert shared.weight.device.type == "cuda" and shared.weight.numel() == 64 * 64
    assert all(id(shared.weight) not in b.index for b in sw.blocks)
    with torch.no_grad():
        for _ in range(2):
            assert all(torch.equal(layer(x), r) for layer, r in zip(layers, ref))
    sw.remove()


def test_enter_leave_follow_explicit_indices():
    sw = _scheduler(3, depth = 1)
    sw.indices, sw.pos = [1, 4, 7], {1: 0, 4: 1, 7: 2}
    for b in sw.blocks:
        sw._release(b)
    sw.enter(0)
    assert not any(b.resident for b in sw.blocks)
    sw.enter(4)
    assert sw.blocks[1].resident and sw.blocks[2].resident
    sw.leave(4)
    assert not sw.blocks[1].resident


def test_reserve_estimate_scales_with_tokens():
    from types import SimpleNamespace
    cfg = SimpleNamespace(hidden_size = 4096, vocab_size = 128256)
    one = estimate_training_reserve_bytes(cfg, 2048, safety_bytes = 0, fragmentation = 0)
    four = estimate_training_reserve_bytes(cfg, 2048, batch_size = 4, safety_bytes = 0, fragmentation = 0)
    # Activations scale with tokens; logits are capped at 2048 rows.
    assert four - one == 3 * 2048 * 4096 * 48
    assert estimate_training_reserve_bytes(cfg, 2048, extra_bytes = 7, safety_bytes = 0, fragmentation = 0) == one + 7
    assert estimate_training_reserve_bytes(cfg, 2048, safety_bytes = 0) == one + one // 16


def test_pool_bytes_gives_equal_size_blocks_of_different_shapes_a_pool_each():
    a = [((64, 64), torch.bfloat16)]
    b = [((32, 128), torch.bfloat16)]
    sigs = [_mod._layer_signature([("w", torch.empty(s, dtype = d, device = "meta"))]) for (s, d), in (a, b, a, b)]
    assert sigs[0] != sigs[1]
    assert _mod._pool_bytes([10] * 4, 2, sigs) == 4 * 10
    assert _mod._pool_bytes([10] * 4, 2, [sigs[0]] * 4) == 3 * 10


def test_auto_swap_indices_takes_only_the_shortfall():
    if not torch.cuda.is_available():
        return
    layers = nn.ModuleList([nn.Linear(256, 256, bias = False) for _ in range(12)]).cuda().requires_grad_(False)
    layer = 256 * 256 * 4
    dev = layers[0].weight.device
    assert auto_swap_indices(layers, 10 * layer, 2, free_bytes = {dev: 10 * layer}) == ([], 0)
    # One layer short: four must go, since the pool keeps three slots on the card.
    idx, left = auto_swap_indices(layers, 10 * layer, 2, free_bytes = {dev: 9 * layer})
    assert len(idx) == 4 and left == 0 and idx == sorted(idx)
    idx, left = auto_swap_indices(layers, 100 * layer, 2, free_bytes = {dev: 0})
    assert len(idx) == 11 and left > 0


def test_pool_bytes_rounds_depth_up_like_the_runtime():
    assert _mod._pool_bytes([10] * 4, 0) == _mod._pool_bytes([10] * 4, 1) == 2 * 10


class _SharedBlock(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.own = nn.Linear(256, 256, bias = False)
        self.shared = shared


def test_auto_swap_indices_counts_no_savings_for_shared_weights():
    if not torch.cuda.is_available():
        return
    shared = nn.Linear(256, 2048, bias = False)
    layers = nn.ModuleList([_SharedBlock(shared) for _ in range(12)]).cuda().requires_grad_(False)
    own = 256 * 256 * 4
    dev = layers[0].own.weight.device
    # Two own-layers short: five go (three stay as the pool); the shared table frees nothing.
    idx, left = auto_swap_indices(layers, 10 * own, 2, free_bytes = {dev: 8 * own})
    assert len(idx) == 5 and left == 0


def test_block_on_another_card_gets_its_inputs_moved():
    if torch.cuda.device_count() < 2:
        return
    torch.manual_seed(0)
    blocks = nn.ModuleList([_XBlock(64).to("cuda:0"), _XBlock(64).to("cuda:1")])
    x = torch.randn(2, 64, device = "cuda:0", requires_grad = True)
    enc = torch.randn(2, 64, device = "cuda:0")
    ref = blocks[1](blocks[0](x, enc).to("cuda:1"), enc.to("cuda:1"))
    sw = BlockSwap(blocks, [1], prefetch_depth = 1)
    # Fast decode loops skip the pre-hook, so they must see that inputs need moving.
    assert sw.spans_devices
    out = blocks[1](blocks[0](x, enc), enc)
    assert out.device.index == 1 and torch.equal(out, ref)
    sw.remove()


def test_one_card_swap_keeps_fast_decode():
    if not torch.cuda.is_available():
        return
    blocks = nn.ModuleList([_XBlock(64) for _ in range(4)]).cuda().requires_grad_(False)
    sw = BlockSwap(blocks, [1, 3], prefetch_depth = 1)
    assert not sw.spans_devices
    # An embedding on another card shows up as hidden states off this device.
    assert sw.layer_device == blocks[0].w.weight.device
    sw.remove()
    assert not BlockSwap(blocks, [], prefetch_depth = 1).spans_devices


def test_hooks_are_opaque_to_torch_compile():
    # Stream copies and `.data` swaps cannot be traced: a compiled caller must break around them.
    layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
    sw = _scheduler(3)
    for hook in (sw._pre(0), sw._post(0), sw._bwd(0), BlockSwap.enter, BlockSwap.leave):
        assert getattr(hook, "_torchdynamo_disable", False) or hasattr(hook, "__wrapped__")


def test_compiled_layers_match_eager_with_swap():
    if not torch.cuda.is_available():
        return
    from torch import _dynamo
    torch.manual_seed(0)
    blocks = nn.ModuleList([_XBlock(64) for _ in range(4)]).cuda()
    for p in blocks.parameters():
        p.requires_grad_(False)
    x = torch.randn(2, 64, device = "cuda")
    enc = torch.randn(2, 64, device = "cuda")
    with torch.no_grad():
        ref = x
        for b in blocks:
            ref = b(ref, enc)
    _dynamo.reset()
    sw = BlockSwap(blocks, [1, 3], prefetch_depth = 1)
    for b in blocks:
        b.compile()
    with torch.no_grad():
        for _ in range(3):
            out = x
            for b in blocks:
                out = b(out, enc)
            assert torch.equal(out, ref)
    sw.remove()
    _dynamo.reset()


class _Stack(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4)) for _ in range(n)])


def test_host_load_moves_a_layer_only_once_all_its_weights_landed():
    with torch.device("meta"):
        model = _Stack(4)
    state = _mod._HostLoad(2, "tail")
    layer = model.model.layers[3]
    names = [f"model.layers.3.{n}" for n, _ in layer.named_parameters()]
    for k, (n, p) in enumerate(list(layer.named_parameters())):
        mod_name, _, attr = n.rpartition(".")
        setattr(layer.get_submodule(mod_name), attr, nn.Parameter(torch.ones(p.shape)))
        state.on_param(model, names[k])
        assert (3 in state.done) == (k == len(names) - 1)
    assert state.indices == [2, 3] and 2 not in state.done
    assert not any(p.requires_grad for p in layer.parameters())
    state.on_param(model, "model.layers.1.0.weight")
    assert state.done == {3}


def _core_model_loading_or_none():
    # Same lookup load_layers_to_host does. Not find_spec: on transformers 4 unsloth registers a
    # stand-in transformers.core_model_loading with __spec__ None, and find_spec raises ValueError
    # on it, which failed this test on the HF 4.57 leg while the code under test was fine.
    try:
        return importlib.import_module("transformers.core_model_loading")
    except ImportError:
        return None


def test_load_layers_to_host_restores_what_it_patched():
    core = _core_model_loading_or_none()
    mu = importlib.import_module("transformers.modeling_utils")
    before = (getattr(core, "set_param_for_module", None), getattr(mu, "caching_allocator_warmup", None),
              os.environ.get("HF_DEACTIVATE_ASYNC_LOAD"))
    with _mod.load_layers_to_host(1):
        assert os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == "1"
    after = (getattr(core, "set_param_for_module", None), getattr(mu, "caching_allocator_warmup", None),
             os.environ.get("HF_DEACTIVATE_ASYNC_LOAD"))
    assert before == after


def test_load_layers_to_host_matches_a_normal_load(tmp_path):
    if not torch.cuda.is_available():
        return
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size = 64, intermediate_size = 128, num_hidden_layers = 6, num_attention_heads = 4,
                      num_key_value_heads = 2, vocab_size = 256, max_position_embeddings = 64)
    LlamaForCausalLM(cfg).save_pretrained(tmp_path)
    ids = torch.randint(0, 256, (1, 16), device = "cuda")
    ref = LlamaForCausalLM.from_pretrained(tmp_path, device_map = {"": 0})
    with torch.no_grad():
        want = ref(input_ids = ids).logits
    with _mod.load_layers_to_host(3) as state:
        model = LlamaForCausalLM.from_pretrained(tmp_path, device_map = {"": 0})
    assert state.indices == [1, 3, 5]
    for i, layer in enumerate(model.model.layers):
        devices = {p.device.type for p in layer.parameters()}
        assert devices == ({"cpu"} if i in state.indices else {"cuda"})
    sw = BlockSwap(state.layers, state.indices)
    with torch.no_grad():
        assert torch.equal(model(input_ids = ids).logits, want)
    sw.remove()


def test_gpt_oss_fast_decode_checks_where_hidden_states_arrive():
    # The fast loop skips the pre-hook: an embedding on another card must send decode down the hooked path.
    src = open(os.path.join(_HERE, "unsloth_zoo", "temporary_patches", "gpt_oss.py"), encoding = "utf-8").read()
    gate = src[src.index('block_swap = getattr(self.layers, "_unsloth_block_swap", None)'):]
    gate = gate[:gate.index("torch.compiler.cudagraph_mark_step_begin()")]
    assert "layer_device" in gate and "hidden_states.device" in gate and "spans_devices" in gate


class _PerLayerTableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        self.embed_tokens = nn.Embedding(8, 4)
        self.per_layer = nn.Embedding(64, 16)

    def get_input_embeddings(self):
        return self.embed_tokens


def test_transformers_4_load_streams_tables_shard_by_shard(monkeypatch):
    # No core_model_loading: the per-shard path runs, and the plan counted the tables off the card already.
    import sys
    monkeypatch.setitem(sys.modules, "transformers.core_model_loading", None)
    monkeypatch.setattr(_mod, "EXTRA_EMBEDDING_MIN_BYTES", 64 * 16 * 4)
    mu = importlib.import_module("transformers.modeling_utils")
    model = _PerLayerTableModel()
    monkeypatch.setattr(mu, "_load_state_dict_into_meta_model", lambda model, *a, **k: None, raising = False)
    with _mod.load_layers_to_host(1, embeddings = True) as state:
        mu._load_state_dict_into_meta_model(model)
        assert state.embeddings == [model.per_layer]


def test_reserve_reads_the_decoder_width_when_the_text_config_is_the_outer_one():
    import types
    decoder = types.SimpleNamespace(hidden_size = 64, vocab_size = 100)
    # T5Gemma on transformers 4.56: get_text_config() hands back the outer config, which has no width.
    outer = types.SimpleNamespace(decoder = decoder, vocab_size = 100)
    outer.get_text_config = lambda: outer
    got = estimate_training_reserve_bytes(outer, 128, safety_bytes = 0, fragmentation = 0)
    assert got == estimate_training_reserve_bytes(decoder, 128, safety_bytes = 0, fragmentation = 0) > 0


def test_lora_count_includes_gpt2_conv1d_projections():
    from transformers.pytorch_utils import Conv1D
    block = nn.Module()
    block.c_attn = Conv1D(3 * 64, 64)
    block.c_proj = Conv1D(64, 64)
    assert _mod.lora_param_count([block], r = 16) == 16 * ((64 + 192) + (64 + 64))


def test_host_arena_is_registered_portable(monkeypatch):
    # Blocks may fetch onto several cards: the arena must count as pinned in every device's context.
    import types
    calls = []
    fake = types.SimpleNamespace(cudaHostRegister = lambda ptr, n, flags: calls.append(flags) or 0,
                                 cudaHostUnregister = lambda ptr: 0)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: fake)
    reg = _mod._Registered(torch.empty(1024, dtype = torch.uint8), 1024)
    assert calls == [1]  # cudaHostRegisterPortable
    del reg
