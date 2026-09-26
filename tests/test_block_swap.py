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


def _scheduler(n, depth = 2, sigs = None, start = 0):
    """A BlockSwap with fake blocks and pools, no CUDA touched."""
    sw = BlockSwap.__new__(BlockSwap)
    sw.depth = depth
    sw.handles = []
    sigs = sigs or ["a"] * n
    sw.blocks = [_FakeBlock(s) for s in sigs]
    sw.free = {s: [object() for _ in range(depth + 1)] for s in set(sigs)}
    sw.start = start
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
    # Not block 0: its backward hook re-arms and refetches it.
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


def test_enter_leave_walks_a_decode_step():
    # The fast decode loop reads weights without calling the layer. enter/leave
    # take indices into the full layer list; the swapped tail starts at 4 here.
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
            sw._pre(i)(None, None)
            sw._post(i)(None, None, None)
    # Recompute of the top blocks starts, then backward dies: their slots stay held.
    with torch.enable_grad():
        sw._pre(7)(None, None)
        sw._pre(6)(None, None)
    assert not sw.free["a"]
    with torch.no_grad():
        for i in range(8):
            sw._pre(i)(None, None)
            assert sw.blocks[i].resident, i
            sw._post(i)(None, None, None)


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
    # from_pretrained(block_swap_layers = N) installs before PEFT wraps each Linear in a base_layer.
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


def test_host_loaded_layer_is_adopted_not_copied():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return
    torch.manual_seed(0)
    lin = nn.Linear(64, 64, bias = False).requires_grad_(False)   # built on the host
    host = lin.weight.data.pin_memory()
    lin.weight.data = host
    x = torch.randn(2, 64, device = "cuda")
    ref = x @ host.to("cuda").t()
    layers = nn.ModuleList([nn.Linear(64, 64).cuda(), lin])
    sw = BlockSwap(layers, 1, prefetch_depth = 1, device = "cuda")
    assert sw.blocks[0].host[0].data_ptr() == host.data_ptr(), "an already-pinned host weight is used as is"
    with torch.no_grad():
        sw.enter(1)
        out = lin(x)
        sw.leave(1)
    assert torch.equal(ref, out)
    assert lin.weight.device.type == "cuda"
