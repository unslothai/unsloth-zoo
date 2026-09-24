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
BlockSwap, find_decoder_layers = _mod.BlockSwap, _mod.find_decoder_layers


class _FakeBlock:
    """Just enough of _Block for the scheduler: residency, a slot, a signature."""

    def __init__(self, sig = "a"):
        self.sig = sig
        self.resident = True
        self.slot = None
        self.fetches = 0

    def evict(self):
        if not self.resident:
            return None
        self.resident = False
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


def _scheduler(n, depth = 2, sigs = None):
    """A BlockSwap with fake blocks and pools, no CUDA touched."""
    sw = BlockSwap.__new__(BlockSwap)
    sw.depth = depth
    sw.handles = []
    sigs = sigs or ["a"] * n
    sw.blocks = [_FakeBlock(s) for s in sigs]
    sw.free = {s: [object() for _ in range(depth + 1)] for s in set(sigs)}
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
    # Walk the forward sweep under no_grad: pre fetches ahead, post evicts.
    with torch.no_grad():
        for i in range(8):
            sw._pre(i)(None, None)
            assert _live(sw) <= sw.depth + 1, f"step {i}: {_live(sw)} live"
            sw._post(i)(None, None, None)
    assert _live(sw) == 0


def test_prefetch_direction_flips_with_grad_mode():
    sw = _scheduler(8, depth = 2)
    for b in sw.blocks:
        sw._release(b)
    # grad off: entering block 3 looks ahead to 5.
    with torch.no_grad():
        sw._pre(3)(None, None)
    assert sw.blocks[3].resident and sw.blocks[5].resident and not sw.blocks[1].resident
    for b in sw.blocks:
        sw._release(b)
    # grad on: the recompute sweep runs in reverse, so entering 3 looks back to 1.
    with torch.enable_grad():
        sw._pre(3)(None, None)
    assert sw.blocks[3].resident and sw.blocks[1].resident and not sw.blocks[5].resident


def test_post_hook_evicts_only_under_no_grad():
    # Block 2, not 0: block 0's backward hook also re-arms the next step, which
    # would fetch it straight back and hide what this test is checking. Depth 3
    # so the look-ahead and look-behind fetches never contend for block 2's slot.
    sw = _scheduler(6, depth = 3)
    for b in sw.blocks:
        sw._release(b)
    with torch.no_grad():
        sw._pre(2)(None, None)
        assert sw.blocks[2].resident
        sw._post(2)(None, None, None)
    assert not sw.blocks[2].resident, "under no_grad the post hook evicts"
    for b in sw.blocks:
        sw._release(b)
    with torch.enable_grad():
        sw._pre(2)(None, None)
        assert sw.blocks[2].resident
        sw._post(2)(None, None, None)
    assert sw.blocks[2].resident, "with grad on, eviction waits for the backward hook"
    sw._bwd(2)(None, None, None)
    assert not sw.blocks[2].resident


def test_backward_hook_on_block_zero_arms_next_step():
    sw = _scheduler(6, depth = 2)
    for b in sw.blocks:
        sw._release(b)
    assert _live(sw) == 0
    sw._bwd(3)(None, None, None)
    assert _live(sw) == 0, "only block 0's backward marks the step boundary"
    sw._bwd(0)(None, None, None)
    assert [b.resident for b in sw.blocks[:2]] == [True, True]


def test_each_signature_gets_its_own_pool():
    sigs = ["a", "a", "b", "a", "b", "b"]
    sw = _scheduler(6, depth = 1, sigs = sigs)
    assert set(sw.free) == {"a", "b"}
    with torch.no_grad():
        for i in range(6):
            sw._pre(i)(None, None)
            assert _live(sw, "a") <= 2 and _live(sw, "b") <= 2
            sw._post(i)(None, None, None)


def test_slot_returns_to_the_pool_it_came_from():
    sw = _scheduler(4, depth = 1, sigs = ["a", "b", "a", "b"])
    for b in sw.blocks:
        sw._release(b)
    n_a, n_b = len(sw.free["a"]), len(sw.free["b"])
    sw._fetch(sw.blocks[1])          # a "b" block
    assert len(sw.free["b"]) == n_b - 1 and len(sw.free["a"]) == n_a
    sw._release(sw.blocks[1])
    assert len(sw.free["b"]) == n_b and len(sw.free["a"]) == n_a


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
