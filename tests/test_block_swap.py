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

"""Tests unsloth_zoo.block_swap with fake blocks on a CPU-only runner (one real Params4bit case skips without a GPU);
loaded from file since import unsloth_zoo needs unsloth, and conftest stubs torch.cuda."""

import importlib.util
import os

import types

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


def test_shaped_gives_evicted_expert_weights_their_shape_without_memory():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(4, 16, 8), requires_grad = False)

    layers = nn.ModuleList([Experts() for _ in range(3)]).to("cuda")
    sw = BlockSwap(layers, 3, prefetch_depth = 1)
    for b in sw.blocks:
        sw._release(b)
    w = layers[0].gate_up_proj
    assert w.numel() == 0
    with sw.shaped():
        assert w.shape == (4, 16, 8) and w.device.type == "cuda"
        assert w.untyped_storage().nbytes() == w.element_size()
        try:
            from peft.tuners.lora.layer import ParamWrapper
        except ImportError:
            ParamWrapper = None
        if ParamWrapper is not None:
            import inspect
            kwargs = {}
            if "config" in inspect.signature(ParamWrapper.__init__).parameters:
                from peft import LoraConfig
                kwargs["config"] = LoraConfig(r = 2, lora_alpha = 2)
            pw = ParamWrapper(layers[0], "default", parameter_name = "gate_up_proj", r = 2, lora_alpha = 2, **kwargs)
            assert pw.num_experts == 4
    assert w.numel() == 0
    sw.remove()
    assert w.shape == (4, 16, 8)


def test_shaped_restores_placeholders_set_before_a_failure():
    import types, pytest

    class FailingSignature:
        def __getitem__(self, index):
            raise RuntimeError("shape lookup failed")

    a, b = nn.Parameter(torch.empty(0), requires_grad = False), nn.Parameter(torch.empty(0), requires_grad = False)
    blk = types.SimpleNamespace(resident = False, params = [a, b], empties = [a.data, b.data],
                                sig = [((4, 16, 8),), FailingSignature()])
    with pytest.raises(RuntimeError, match = "shape lookup failed"):
        with BlockSwap.shaped(types.SimpleNamespace(blocks = [blk])):
            pass
    assert a.shape == (0,) and b.shape == (0,)


def test_shaped_leaves_a_block_fetched_inside_the_context():
    import types
    w = nn.Parameter(torch.empty(0), requires_grad = False)
    blk = types.SimpleNamespace(resident = False, params = [w], empties = [w.data], sig = [((4, 8),)])
    with BlockSwap.shaped(types.SimpleNamespace(blocks = [blk])):
        w.data, blk.resident = torch.ones(4, 8), True
    assert w.shape == (4, 8) and w.sum() == 32


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
        __import__("pytest").skip("needs CUDA")
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
        __import__("pytest").skip("needs CUDA")
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
        __import__("pytest").skip("needs CUDA")
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


def test_layer_accelerate_left_on_meta_is_refused_before_anything_is_hooked():
    import pytest
    # A model that did not fit at load: accelerate keeps the last layers' weights on meta behind its hooks.
    layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)] + [nn.Linear(8, 8, device = "meta") for _ in range(2)])
    layers.requires_grad_(False)
    with pytest.raises(ValueError, match = "offload_layers to from_pretrained"):
        BlockSwap(layers, [1, 3], device = "cuda")
    assert all(len(l._forward_pre_hooks) == 0 and len(l._forward_hooks) == 0 for l in layers)
    assert layers[1].weight.device.type == "cpu" and layers[1].weight.numel() == 64


def test_shared_weight_left_on_meta_is_not_refused():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("BlockSwap needs a CUDA device")
    # Block Swap never manages a weight two layers share, so accelerate keeps serving it.
    shared = nn.Linear(8, 8, bias = False, device = "meta")
    layers = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), shared) for _ in range(3)]).requires_grad_(False)
    sw = BlockSwap(layers, [0, 2], prefetch_depth = 1, device = "cuda")
    assert shared.weight.device.type == "meta"
    assert all(id(shared.weight) not in b.index for b in sw.blocks)
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
        __import__("pytest").skip("needs CUDA")
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
        __import__("pytest").skip("needs CUDA")
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
        __import__("pytest").skip("needs CUDA")
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
        __import__("pytest").skip("needs CUDA")
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
    # Same lookup load_layers_to_host uses, not find_spec: on transformers 4 unsloth registers a stand-in
    # transformers.core_model_loading with __spec__ None, and find_spec raises ValueError on it.
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
        __import__("pytest").skip("needs CUDA")
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


def test_harvest_copy_keeps_samples_still_in_flight():
    import types
    # CPU: fake events. A refetch queued before the last copy finished must not lose either sample.
    class Ev:
        def __init__(self, done, ms = 0.0):
            self.done, self.ms = done, ms
        def query(self):
            return self.done
        def elapsed_time(self, end):
            return end.ms

    old = ({"d": Ev(True)}, {"d": Ev(False, 2.0)})
    b = types.SimpleNamespace(copy_pending = [old], copy_open = True, t_copy = {"d": Ev(True)},
                              events = {"d": Ev(True, 3.0)}, nbytes = lambda: 10)
    sw = types.SimpleNamespace(_tot = {"copy_ms": 0.0, "copies": 0, "copy_bytes": 0})
    BlockSwap._harvest_copy(sw, b)
    assert sw._tot == {"copy_ms": 3.0, "copies": 1, "copy_bytes": 10} and b.copy_pending == [old]
    old[1]["d"].done = True
    BlockSwap._harvest_copy(sw, b)
    assert sw._tot == {"copy_ms": 5.0, "copies": 2, "copy_bytes": 20} and b.copy_pending == []


def test_stats_reports_placement_and_timings_after_a_step():
    if not torch.cuda.is_available():
        __import__("pytest").skip("needs CUDA")
    import torch.utils.checkpoint as cp
    torch.manual_seed(0)
    blocks = nn.ModuleList([_XBlock(256) for _ in range(8)]).cuda()
    sw = BlockSwap(blocks, [1, 3, 5, 7], prefetch_depth = 1, device = "cuda")
    # Step 1 is warmup and stays out of the window, so run two.
    for _ in range(2):
        x = torch.randn(4, 256, device = "cuda", requires_grad = True)
        enc = torch.randn(4, 256, device = "cuda", requires_grad = True)
        h = x
        for b in blocks:
            h = cp.checkpoint(b, h, enc, use_reentrant = False)
        h.square().sum().backward()
    torch.cuda.synchronize()
    s = sw.stats()
    assert s["total_layers"] == 8 and s["swapped"] == [1, 3, 5, 7]
    assert set(s["state"]) == {1, 3, 5, 7} and set(s["state"].values()) <= {"gpu", "host", "copying", "held"}
    # After backward only the next step's first block is armed.
    assert list(s["state"].values()).count("host") == 3
    assert s["copies"] > 0 and s["copy_ms"] > 0 and s["compute_ms"] > 0 and s["layers"] > 0
    assert s["copy_bytes"] == s["copies"] * sw.blocks[0].nbytes()
    # The window resets: nothing ran since.
    assert sw.stats()["copies"] == 0
    sw.remove()


def test_pre_keeps_an_in_flight_timing_sample_for_harvest():
    if not torch.cuda.is_available():
        __import__("pytest").skip("needs CUDA")

    class Ev:
        done, recorded = False, 0

        def record(self, stream):
            self.recorded += 1

        def query(self):
            return self.done

        def elapsed_time(self, other):
            return 1.0

    blocks = nn.ModuleList([_XBlock(256) for _ in range(4)]).cuda()
    sw = BlockSwap(blocks, [1, 3], prefetch_depth = "auto", device = "cuda")
    b = sw.blocks[0]
    marks = [Ev(), Ev(), Ev()]
    b.marks, b.mark_open = marks, 2
    # The backward recompute reaches the block before its forward sample finished on the card.
    sw._pre(0)(blocks[1], (), {})
    assert b.mark_open == 2 and b.marks is marks and all(m.recorded == 0 for m in marks)
    marks[2].done = True
    layers = sw._tot["layers"]
    sw._harvest()
    assert sw._tot["layers"] == layers + 1
    sw.remove()


def _auto(n = 8, depth = 1, sample = 1):
    _mod._AUTO_SAMPLE = sample
    sw = _scheduler(n, depth = depth)
    sw._auto, sw._settled, sw._steps, sw._last_stall = True, False, 1, None
    sw._tot = dict.fromkeys(_mod._STAT_KEYS, 0.0)
    sw._seen, sw._adapt_seen = dict(sw._tot), dict(sw._tot)
    sw._harvest = lambda: None
    sw._grow_pool = lambda: True
    sw._shrink_pool = lambda: None
    return sw


def _step(sw, copy_ms, compute_ms, stall_ms, layers = 8):
    for k, v in (("copies", layers), ("copy_ms", copy_ms * layers), ("compute_ms", compute_ms * layers),
                 ("stall_ms", stall_ms * layers), ("layers", layers)):
        sw._tot[k] += v
    sw._steps += 1
    sw._adapt_depth()


def test_auto_depth_deepens_while_each_slot_cuts_the_stall():
    sw = _auto()
    _step(sw, 5, 9, 2)
    assert sw.depth == 2 and not sw._settled
    _step(sw, 5, 9, 1)
    assert sw.depth == 3 and not sw._settled
    _step(sw, 5, 9, 0)
    assert sw.depth == 3 and sw._settled


def test_auto_depth_gives_back_a_slot_that_did_not_help():
    sw = _auto()
    shrunk = []
    sw._shrink_pool = lambda: shrunk.append(sw.depth)
    _step(sw, 5, 9, 2)
    _step(sw, 5, 9, 2)
    assert sw.depth == 1 and sw._settled and shrunk == [2]


def test_auto_depth_tries_one_slot_on_a_slow_bus_and_gives_it_back():
    sw = _auto()
    shrunk = []
    sw._shrink_pool = lambda: shrunk.append(sw.depth)
    _step(sw, 20, 9, 11)
    _step(sw, 20, 9, 11)
    # No depth hides a copy longer than a layer's compute, so the stall stays and the slot goes back.
    assert sw.depth == 1 and sw._settled and shrunk == [2]


def test_auto_depth_judges_each_depth_over_several_steps():
    sw = _auto(sample = 2)
    _step(sw, 5, 9, 11)  # one step alone decides nothing
    assert sw.depth == 1 and not sw._settled
    _step(sw, 1, 9, 0.5)
    assert sw.depth == 2 and not sw._settled
    _mod._AUTO_SAMPLE = 2


def test_auto_depth_stops_when_the_card_has_no_room_for_a_slot():
    sw = _auto()
    sw._grow_pool = lambda: False
    _step(sw, 5, 9, 2)
    assert sw.depth == 1 and sw._settled


def test_auto_depth_starts_at_one_and_settles_with_its_pool():
    if not torch.cuda.is_available():
        __import__("pytest").skip("needs CUDA")
    blocks = nn.ModuleList([_XBlock(256) for _ in range(8)]).cuda()
    sw = BlockSwap(blocks, 6, prefetch_depth = "auto", device = "cuda")
    assert sw.depth == 1 and sw._auto and not sw._settled
    assert sum(len(v) for v in sw.free.values()) + sw.resident_count() == 2
    sw.remove()


def test_grow_pool_adds_a_slot_only_with_room_for_the_next_step(monkeypatch):
    if not torch.cuda.is_available():
        __import__("pytest").skip("needs CUDA")
    blocks = nn.ModuleList([_XBlock(256) for _ in range(8)]).cuda()
    sw = BlockSwap(blocks, 6, prefetch_depth = "auto", device = "cuda")
    slots = lambda: sum(len(v) for v in sw.free.values()) + sum(b.slot is not None for b in sw.blocks)
    before = slots()
    assert sw._grow_pool() and slots() == before + 1
    monkeypatch.setattr(_mod, "_free_device_bytes", lambda dv: 0)
    assert not sw._grow_pool() and slots() == before + 1
    sw.remove()


def test_grow_pool_ignores_the_load_time_peak(monkeypatch):
    if not torch.cuda.is_available():
        __import__("pytest").skip("needs CUDA")
    # The process peak still holds the model load (already there at install); only the swap's own last step
    # counts as headroom.
    blocks = nn.ModuleList([_XBlock(256) for _ in range(8)]).cuda()
    now = torch.cuda.memory_allocated()
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: now + (64 << 30))
    monkeypatch.setattr(_mod, "_WORKSPACE_MIN_BYTES", 0)
    monkeypatch.setattr(_mod, "_WORKSPACE_FRACTION", 0.0)
    resets = []
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda dv = None: resets.append(dv))
    sw = BlockSwap(blocks, 6, prefetch_depth = "auto", device = "cuda")
    # Install opened a fresh peak window: the allocator's peak now starts at what is allocated.
    assert resets and sw._hwm_fresh
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: torch.cuda.memory_allocated())
    monkeypatch.setattr(_mod, "_free_device_bytes", lambda dv: 1 << 30)
    assert sw._grow_pool()
    sw._last_peak = {torch.device("cuda").index or 0: now + (2 << 30)}
    assert not sw._grow_pool()
    # A temporary inside a layer that no hook saw still shows in the fresh window's peak.
    sw._last_peak = {}
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: torch.cuda.memory_allocated() + (2 << 30))
    assert not sw._grow_pool()
    sw.remove()


_GiB = 1 << 30


@__import__("pytest").fixture(autouse = True)
def _no_cgroup_cap(monkeypatch):
    # The host-fit tests fake psutil; keep memory_budget from capping that by this machine's real cgroup.
    monkeypatch.setenv("UNSLOTH_HOST_BUDGET_CGROUP", "0")


def _fake_psutil(available, total = 32 * _GiB):
    import types
    vm = types.SimpleNamespace(available = available, total = total)
    return types.SimpleNamespace(virtual_memory = lambda: vm)


def test_host_fit_passes_when_the_layers_fit(monkeypatch):
    import sys
    # 32 GiB box: reserve is max(4 GiB, 15%) = 4.8 GiB, so 20 GiB free leaves ~15.2 GiB for layers.
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(20 * _GiB))
    _mod._check_host_fit([_GiB // 2] * 30)


def test_host_fit_refuses_with_the_counts_when_the_layers_do_not_fit(monkeypatch):
    import sys
    import pytest
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(20 * _GiB))
    with pytest.raises(ValueError) as e:
        _mod._check_host_fit([_GiB // 2] * 63)
    msg = str(e.value)
    assert "63 layers" in msg and "31.5 GiB" in msg and "15.2 GiB" in msg and "At most 30 layers" in msg


def test_host_fit_counts_weights_already_in_host_ram_as_available(monkeypatch):
    import sys
    # The host-load guard's running total: layers it already moved are not copied again.
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(6 * _GiB))
    _mod._check_host_fit([_GiB // 2] * 20, on_host = 10 * _GiB)


def test_block_swap_refuses_host_loaded_layers_without_room_to_pack(monkeypatch):
    import sys, pytest
    if not torch.cuda.is_available():
        __import__("pytest").skip("needs CUDA")
    # The chunked fallback copies host-loaded layers before their originals go: both are resident at once.
    monkeypatch.setenv("UNSLOTH_DISABLE_PINNED_MEMORY", "1")
    blocks = nn.ModuleList([_XBlock(256) for _ in range(8)])
    blocks[:2].cuda(); blocks[2:].cpu()
    need = sum(p.numel() * p.element_size() for b in blocks[2:] for p in b.parameters())
    monkeypatch.setattr(_mod, "_host_reserve", lambda total: 0)
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(need // 2))
    with pytest.raises(ValueError, match = "offload_layers asked for 6 layers"):
        BlockSwap(blocks, 6, device = "cuda")


def test_block_swap_credits_host_loaded_layers_when_the_arena_registers_in_place(monkeypatch):
    import sys, os
    if not torch.cuda.is_available() or os.name == "nt" or not _mod._can_register():
        __import__("pytest").skip("needs CUDA")
    # Registered in place, block by block: only about two blocks are ever held twice.
    blocks = nn.ModuleList([_XBlock(256) for _ in range(8)])
    blocks[:2].cuda(); blocks[2:].cpu()
    need = sum(p.numel() * p.element_size() for b in blocks[2:] for p in b.parameters())
    monkeypatch.setattr(_mod, "_host_reserve", lambda total: 0)
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(need // 2))
    BlockSwap(blocks, 6, device = "cuda").remove()


def test_host_fit_splits_free_ram_between_local_ranks(monkeypatch):
    import sys, pytest
    # 20 GiB free less a 4.8 GiB reserve is 15.2 GiB: 10 GiB fits one rank, not each of two.
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(20 * _GiB))
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising = False)
    _mod._check_host_fit([_GiB // 2] * 20)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    with pytest.raises(ValueError, match = "At most 15 layers"):
        _mod._check_host_fit([_GiB // 2] * 20)


def test_host_fit_does_not_hold_back_ram_for_a_rank_that_already_loaded(monkeypatch, tmp_path):
    import sys, pytest
    # Two local ranks need 8 GiB each. Before either allocates, 16 GiB spare holds both; once rank 1 has loaded,
    # available RAM already shows its 8 GiB and rank 0 may take the remaining 8 GiB in full.
    monkeypatch.setattr(_mod, "_host_reserve", lambda total: 0)
    monkeypatch.setattr(_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("MASTER_PORT", "29511")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(_mod, "_HOST_SESSION", 0)
    _mod._begin_host_session()
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(8 * _GiB))
    monkeypatch.setattr(_mod, "_host_memory", lambda: (64 * _GiB, 8 * _GiB))
    # Rank 1 has not reached this load: it is assumed to need as much as this rank.
    with pytest.raises(ValueError):
        _mod._check_host_fit([_GiB] * 8)
    ledger = _mod._rank_ledger()
    assert ledger is not None and ledger[1:] == (0, 2)
    (tmp_path / os.path.basename(ledger[0]) / "1").write_text(f"{_mod._HOST_SESSION} 0")
    _mod._check_host_fit([_GiB] * 8)
    # Rank 1 still has 4 GiB to go: that much stays held back.
    (tmp_path / os.path.basename(ledger[0]) / "1").write_text(f"{_mod._HOST_SESSION} {4 * _GiB}")
    with pytest.raises(ValueError):
        _mod._check_host_fit([_GiB] * 8)
    _mod._check_host_fit([_GiB] * 4)
    # A rank still on the previous load is assumed to need this rank's bytes, whatever it published then.
    (tmp_path / os.path.basename(ledger[0]) / "1").write_text(f"{_mod._HOST_SESSION - 1} 0")
    monkeypatch.setattr(_mod, "_host_memory", lambda: (64 * _GiB, 12 * _GiB))
    with pytest.raises(ValueError):
        _mod._check_host_fit([_GiB] * 8)
    _mod._check_host_fit([_GiB] * 6)
    # This rank published what it still needs, and 0 once the session ends.
    mine = tmp_path / os.path.basename(ledger[0]) / "0"
    assert mine.read_text().split() == [str(_mod._HOST_SESSION), str(6 * _GiB)]
    _mod._end_host_session()
    assert mine.read_text().split() == [str(_mod._HOST_SESSION), "0"]


def test_auto_depth_deepens_when_every_block_already_has_a_slot():
    # Signatures A/B/A/B at depth 1: both pools already hold both of their blocks, so depth 2 needs no new slot.
    blocks = [types.SimpleNamespace(sig = s) for s in "ABAB"]
    fake = types.SimpleNamespace(blocks = blocks, free = {"A": [object()], "B": [object()]}, depth = 1)
    assert _mod.BlockSwap._grow_pool(fake) is True
    fake.depth = 3
    assert _mod.BlockSwap._grow_pool(fake) is False


def test_peaks_are_sampled_on_every_card_a_block_uses(monkeypatch):
    seen = []
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda dv: seen.append(torch.device(dv).index) or 100)
    fake = types.SimpleNamespace(_step_peak = None)
    fake._note_peak = lambda dv, extra = 0: _mod.BlockSwap._note_peak(fake, dv, extra)
    block = types.SimpleNamespace(sizes = {torch.device("cuda", 0): 1, torch.device("cuda", 1): 1},
                                  home = torch.device("cuda", 0))
    _mod.BlockSwap._note_block_peak(fake, block)
    assert sorted(seen) == [0, 1] and set(fake._step_peak) == {0, 1}


def test_host_fit_counts_the_most_layers_that_fit(monkeypatch):
    import sys, pytest
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(4 * _GiB))
    monkeypatch.setattr(_mod, "_host_reserve", lambda total: 0)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising = False)
    with pytest.raises(ValueError, match = "At most 2 layers"):
        _mod._check_host_fit([5 * _GiB, 3 * _GiB, _GiB])


def test_host_fit_uses_the_cgroup_capped_host_memory(monkeypatch):
    import pytest
    # memory_budget.host_memory_bytes caps psutil by an enforcing cgroup: a container on a big host reports its
    # own 16 GiB limit and 8 GiB headroom. Reserve max(4 GiB, 15% of 16 GiB) = 4 GiB leaves 4 GiB for layers.
    monkeypatch.setattr(_mod, "_host_memory", lambda: (16 * _GiB, 8 * _GiB))
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising = False)
    with pytest.raises(ValueError, match = "At most 8 layers"):
        _mod._check_host_fit([_GiB // 2] * 20)
    assert _mod._pin_budget() == 4 * _GiB
    _mod._check_host_fit([_GiB // 2] * 8)


def test_host_fit_is_skipped_without_psutil(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "psutil", None)
    _mod._check_host_fit([_GiB] * 1000)


def _host_load(n_layers, n, width = 64, meta = False):
    # meta = True: layers still to stream are on meta, as during a load; _land(layer) materializes one on CPU.
    layers = nn.ModuleList([nn.Linear(width, width, bias = False, device = "meta" if meta else None)
                            for _ in range(n_layers)])
    state = _mod._HostLoad(n, "tail")
    state.layers, state.indices = layers, swap_indices(n_layers, n, "tail")
    return state, layers


def _land(layer):
    layer.to_empty(device = "cpu")


def test_host_load_refuses_before_moving_any_layer(monkeypatch):
    import sys
    import pytest
    # Each 64x64 fp32 layer is 16 KiB and the CPU-built one counts as already on host: room for 2 of 3.
    total = 64 * _GiB
    reserve = int(total * _mod._PIN_RESERVE_FRACTION)
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(reserve + 64 * 64 * 4 + 100, total))
    state, layers = _host_load(4, 3, meta = True)
    state.shape_bytes = {i: 64 * 64 * 4 for i in state.indices}
    _land(layers[1])
    with pytest.raises(ValueError) as e:
        state.evict(1)
    assert "3 layers" in str(e.value) and "At most 2 layers" in str(e.value)
    assert not state.done and layers[1].weight.requires_grad


def test_host_load_moves_layers_that_fit(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(20 * _GiB))
    state, layers = _host_load(4, 3)
    for i in state.indices:
        state.evict(i)
    assert state.done == {1, 2, 3} and len(state.done_bytes) == 3
    assert not layers[1].weight.requires_grad


def test_host_load_counts_layers_already_moved_as_used(monkeypatch):
    import sys
    import pytest
    total = 64 * _GiB
    reserve = int(total * _mod._PIN_RESERVE_FRACTION)
    state, layers = _host_load(4, 3, meta = True)
    state.shape_bytes = {i: 64 * 64 * 4 for i in state.indices}
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(20 * _GiB, total))
    _land(layers[1])
    state.evict(1)
    # RAM dropped after the first layer landed: the next two no longer fit, and the error counts all three.
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(reserve + 100, total))
    _land(layers[2])
    with pytest.raises(ValueError, match = "3 layers"):
        state.evict(2)
    assert state.done == {1}


def test_host_load_counts_streamed_embeddings_still_to_come(monkeypatch):
    import sys
    import pytest
    # Three 16 KiB layers fit, but a 64 MiB streamed table still on meta does not: refuse before any layer moves.
    total = 64 * _GiB
    reserve = int(total * _mod._PIN_RESERVE_FRACTION)
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(reserve + 10 * 64 * 64 * 4, total))
    state, layers = _host_load(4, 3)
    state.embedding_prefixes = {"emb.": nn.Embedding(1 << 18, 64, device = "meta")}
    with pytest.raises(ValueError, match = "At most"):
        state.evict(1)
    assert not state.done


def test_host_load_refuses_an_embedding_that_does_not_fit(monkeypatch):
    if not torch.cuda.is_available():
        __import__("pytest").skip("needs CUDA")
    import sys
    import pytest
    total = 64 * _GiB
    reserve = int(total * _mod._PIN_RESERVE_FRACTION)
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(reserve + (1 << 20), total))
    state, layers = _host_load(4, 0)
    emb = nn.Embedding(1 << 16, 64, device = "cuda")  # 16 MiB
    state.embedding_prefixes = {"emb.": emb}
    with pytest.raises(ValueError, match = "At most"):
        state.evict_embedding(emb)
    assert emb.weight.device.type == "cuda" and not state.embeddings


def test_host_load_is_unguarded_without_psutil(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "psutil", None)
    state, layers = _host_load(4, 3)
    for i in state.indices:
        state.evict(i)
    assert state.done == {1, 2, 3}


def test_block_swap_refuses_before_moving_anything(monkeypatch):
    if not torch.cuda.is_available():
        __import__("pytest").skip("needs CUDA")
    import sys
    import pytest
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(0))
    layers = _linear_tail()
    ptrs = [l.weight.data_ptr() for l in layers]
    with pytest.raises(ValueError, match = "At most 0 layers"):
        BlockSwap(layers, 3, prefetch_depth = 1, device = "cuda")
    assert [l.weight.data_ptr() for l in layers] == ptrs
    assert all(l.weight.device.type == "cuda" and not l._forward_pre_hooks for l in layers)


def test_pin_budget_is_split_across_the_ranks_on_this_host(monkeypatch):
    import sys
    import types
    gib = 1 << 30
    vm = types.SimpleNamespace(available = 64 * gib, total = 128 * gib)
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(virtual_memory = lambda: vm))
    spare = 64 * gib - max(_mod._PIN_RESERVE_MIN_BYTES, int(128 * gib * _mod._PIN_RESERVE_FRACTION))
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising = False)
    assert _mod._pin_budget() == spare
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    assert _mod._pin_budget() == spare // 4
    for bad in ("0", "-2", "x"):
        monkeypatch.setenv("LOCAL_WORLD_SIZE", bad)
        assert _mod._pin_budget() == spare


def test_chunked_fallback_size_matches_what_pack_allocates(monkeypatch):
    # The host-fit check charges the chunked fallback's rounding: it must be the bytes _pack really takes.
    monkeypatch.setenv("UNSLOTH_DISABLE_PINNED_MEMORY", "1")
    monkeypatch.setattr(torch.cuda, "Event", lambda *a, **k: object())
    monkeypatch.setattr(_mod, "_CHUNK_BYTES", 1 << 16)
    cpu = torch.device("cpu")
    layers = [nn.Linear(n, 96, bias = False).requires_grad_(False) for n in (40, 200, 72, 72, 130)]
    sw = BlockSwap.__new__(BlockSwap)
    sw.blocks = [_mod._Block(layer, {cpu: None}, cpu) for layer in layers]
    sw.free, sw._chunks, sw.pinned_bytes = {}, [], 0
    expected = _mod._chunked_bytes([b.sizes for b in sw.blocks])
    sw._pack()
    assert sum(c.numel() for c in sw._chunks) == expected > sum(b.nbytes() for b in sw.blocks)


def test_host_fit_charges_the_chunked_fallback_rounding(monkeypatch):
    import sys, pytest
    # One 17 GiB layer packs into a 32 GiB chunk when it cannot be registered in place.
    monkeypatch.setattr(_mod, "_host_reserve", lambda total: 0)
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(21 * _GiB))
    size = 17 * _GiB
    slack = _mod._chunked_bytes([{"cuda": size}]) - size
    assert slack == 15 * _GiB
    _mod._check_host_fit([size])
    with pytest.raises(ValueError):
        _mod._check_host_fit([size], slack = slack)


def test_auto_depth_without_timing_is_the_fixed_default(monkeypatch):
    monkeypatch.setattr(_mod, "_TIMING", False)
    sw = BlockSwap(nn.ModuleList([nn.Linear(4, 4)]), 0, prefetch_depth = "auto")
    assert (sw._auto, sw._settled, sw.depth) == (False, True, 2)
    monkeypatch.setattr(_mod, "_TIMING", True)
    sw = BlockSwap(nn.ModuleList([nn.Linear(4, 4)]), 0, prefetch_depth = "auto")
    assert (sw._auto, sw._settled, sw.depth) == (True, False, 1)


class _HeadModel(nn.Module):
    """Swapped layers, then a large head and the loss inside forward, as HF models do: the step's peak comes
    after the last layer ran, and the logits' grad is gone before any layer's backward."""

    def __init__(self, d, n, vocab):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_XBlock(d) for _ in range(n)])
        self.head = nn.Linear(d, vocab, bias = False).requires_grad_(False)

    def forward(self, x):
        h = x
        for layer in self.model.layers:
            h = layer(h, x)
        logits = self.head(h)
        return types.SimpleNamespace(loss = logits.logsumexp(-1).mean(), logits = logits)


def test_auto_depth_peak_includes_the_model_output():
    if not torch.cuda.is_available():
        __import__("pytest").skip("needs CUDA")
    torch.manual_seed(0)
    model = _HeadModel(256, 8, 1 << 15).cuda()
    # Swapped layers end early (explicit indices), so no swapped layer's hook runs while the head output is alive.
    sw = BlockSwap(find_decoder_layers(model), [0, 1], prefetch_depth = "auto", device = "cuda")
    x = torch.randn(1024, 256, device = "cuda", requires_grad = True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    model(x).loss.backward()
    true_peak = torch.cuda.max_memory_allocated()
    k = torch.device("cuda").index or 0
    # One step done: its peak is the headroom _grow_pool keeps. Without the output hook it stops at the last
    # swapped layer, a fifth of the true peak here; hooks see op boundaries, not the loss op's own temporaries.
    assert sw._steps == 1 and sw._last_peak.get(k, 0) >= 0.75 * true_peak, (sw._last_peak, true_peak)
    sw.remove()


class _OwnedLayers(nn.Module):
    # A model owning `layers`, so BlockSwap(model = ...) hooks its output like a real one.
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, h, ckpt = False):
        from torch.utils.checkpoint import checkpoint
        for layer in self.layers:
            h = checkpoint(layer, h, use_reentrant = True) if ckpt else layer(h)
        return h


def test_auto_depth_peak_includes_unswapped_layers_backward():
    if not torch.cuda.is_available():
        __import__("pytest").skip("needs CUDA")

    class Head(nn.Module):
        # Trainable and unswapped: its 256 MiB weight grad lands after the first swapped layer's backward hook.
        def __init__(self):
            super().__init__()
            self.big = nn.Parameter(torch.zeros(1 << 26))

        def forward(self, x):
            return x + self.big[:1].sum() * 0

    torch.manual_seed(0)
    layers = nn.ModuleList([Head()] + [nn.Linear(256, 256) for _ in range(3)]).cuda()
    for layer in layers[1:]:
        layer.requires_grad_(False)
    model = _OwnedLayers(layers)
    sw = BlockSwap(layers, [1, 2, 3], prefetch_depth = "auto", device = "cuda", model = model)
    x = torch.randn(64, 256, device = "cuda", requires_grad = True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    model(x).sum().backward()
    true_peak = torch.cuda.max_memory_allocated()
    k = torch.device("cuda").index or 0
    # Judged after the whole backward: the head's grad is in the step's peak, not left for the next step.
    assert sw._steps == 1 and sw._last_peak.get(k, 0) >= 0.9 * true_peak, (sw._last_peak, true_peak)
    sw.remove()


def test_auto_depth_counts_steps_when_the_first_swapped_block_gets_no_grad():
    if not torch.cuda.is_available():
        __import__("pytest").skip("needs CUDA")
    torch.manual_seed(0)
    # Frozen prefix: layers 0 and 1 see no grad-requiring input, only layer 2 onward trains (its bias).
    layers = nn.ModuleList([nn.Linear(256, 256) for _ in range(4)]).cuda()
    for li, layer in enumerate(layers):
        layer.weight.requires_grad_(False)
        layer.bias.requires_grad_(li >= 2)
    model = _OwnedLayers(layers)
    sw = BlockSwap(layers, [1, 2, 3], prefetch_depth = "auto", device = "cuda", model = model)
    for _ in range(3):
        model(torch.randn(64, 256, device = "cuda")).sum().backward()
    # Each backward counts until auto settles; before the fix not one did (the frozen first block never hooks).
    assert sw._steps >= 2 and (sw._settled or sw._steps == 3), (sw._steps, sw._settled)
    sw.remove()


def test_auto_depth_ends_the_step_from_the_loss_when_logits_are_not_in_the_graph():
    import types
    if not torch.cuda.is_available():
        __import__("pytest").skip("needs CUDA")
    torch.manual_seed(0)

    class FusedCE(nn.Module):
        # Unsloth's fused cross entropy: the loss is in the graph, the returned logits are not.
        def __init__(self, layers):
            super().__init__()
            self.layers = layers

        def forward(self, h):
            for layer in self.layers:
                h = layer(h)
            return types.SimpleNamespace(loss = h.square().mean(), logits = torch.empty(0, device = h.device))

    layers = nn.ModuleList([nn.Linear(256, 256) for _ in range(4)]).cuda()
    for layer in layers:
        layer.weight.requires_grad_(False)
    model = FusedCE(layers)
    sw = BlockSwap(layers, [1, 2, 3], prefetch_depth = "auto", device = "cuda", model = model)
    for _ in range(2):
        model(torch.randn(64, 256, device = "cuda", requires_grad = True)).loss.backward()
    assert sw._steps == 2, sw._steps
    sw.remove()


def test_auto_depth_ends_the_step_from_the_loss_when_the_logits_are_a_side_branch():
    import types
    if not torch.cuda.is_available():
        __import__("pytest").skip("needs CUDA")
    torch.manual_seed(0)

    class ReturnLogits(nn.Module):
        # Fused cross entropy with UNSLOTH_RETURN_LOGITS=1: the logits carry a grad_fn but are not the loss's
        # ancestors, so backward from the loss never reaches their node.
        def __init__(self, layers):
            super().__init__()
            self.layers, self.head = layers, nn.Linear(256, 512).cuda()

        def forward(self, h):
            for layer in self.layers:
                h = layer(h)
            return types.SimpleNamespace(loss = h.square().mean(), logits = self.head(h))

    layers = nn.ModuleList([nn.Linear(256, 256) for _ in range(4)]).cuda()
    for layer in layers:
        layer.weight.requires_grad_(False)
    model = ReturnLogits(layers)
    sw = BlockSwap(layers, [0, 1, 2, 3], prefetch_depth = "auto", device = "cuda", model = model)
    for _ in range(2):
        out = model(torch.randn(64, 256, device = "cuda"))
        assert out.logits.grad_fn is not None
        out.loss.backward()
    assert sw._steps == 2, sw._steps
    sw.remove()


def test_auto_depth_counts_one_step_per_backward_under_reentrant_checkpointing():
    if not torch.cuda.is_available():
        __import__("pytest").skip("needs CUDA")
    torch.manual_seed(0)
    # Unsloth checkpoints with use_reentrant = True: each layer's backward is its own inner graph.
    layers = nn.ModuleList([nn.Linear(256, 256) for _ in range(4)]).cuda()
    for layer in layers:
        layer.weight.requires_grad_(False)
    model = _OwnedLayers(layers)
    sw = BlockSwap(layers, [1, 2, 3], prefetch_depth = "auto", device = "cuda", model = model)
    for _ in range(3):
        x = torch.randn(64, 256, device = "cuda", requires_grad = True)
        model(x, ckpt = True).sum().backward()
    assert sw._steps == 3, sw._steps
    sw.remove()


def test_output_peak_reads_the_logits_from_a_tuple_with_the_loss_first():
    import types
    logits, loss = torch.zeros(4, 8), torch.zeros(())
    assert _mod._output_logits((loss, logits)) is logits
    assert _mod._output_logits((logits,)) is logits
    assert _mod._output_logits(types.SimpleNamespace(logits = logits)) is logits


def test_find_decoder_layers_remembers_the_model_without_holding_it():
    import gc
    model = _HeadModel(8, 2, 16)
    layers = find_decoder_layers(model)
    assert _mod._LAYER_OWNERS[layers]() is model
    del model
    gc.collect()
    assert _mod._LAYER_OWNERS.get(layers) is None or _mod._LAYER_OWNERS[layers]() is None


def test_host_load_sizes_each_layer_still_to_come(monkeypatch):
    import sys
    # A 256 KiB layer then two 16 KiB ones: 288 KiB in all, not three times the first.
    total = 64 * _GiB
    reserve = int(total * _mod._PIN_RESERVE_FRACTION)
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(reserve + 300 * 1024, total))
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(w, w, bias = False).to("meta") for w in (64, 256, 64, 64)])
    state = _mod._HostLoad(3, "tail")
    state.bind(model)
    # Loaded, as each would be when its first weight lands (CUDA, so none counts as already on host).
    for i in (1, 2, 3):
        model.layers[i].to_empty(device = "cuda" if torch.cuda.is_available() else "cpu")
    for i in (1, 2, 3):
        state.evict(i)
    assert state.done == {1, 2, 3}


def test_host_load_reads_layer_sizes_before_the_first_weight_lands(monkeypatch):
    # transformers 5: set_param_for_module materializes the weight; bind() must have read the meta sizes first.
    import sys, types
    seen = {}

    def set_param_for_module(model, target_name, *a, **k):
        seen.setdefault("bound", state.shape_bytes.get(1))
        model.layers[1].to_empty(device = "cpu")

    core = types.ModuleType("transformers.core_model_loading")
    core.set_param_for_module = set_param_for_module
    monkeypatch.setitem(sys.modules, "transformers.core_model_loading", core)
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(64, 64, bias = False).to("meta") for _ in range(2)])
    with _mod.load_layers_to_host(1, placement = "tail") as state:
        core.set_param_for_module(model, "layers.1.weight")
    assert seen["bound"] == 64 * 64 * 4


def test_auto_depth_peak_with_a_layer_list_passed_directly():
    if not torch.cuda.is_available():
        __import__("pytest").skip("needs CUDA")
    model = _HeadModel(256, 4, 1 << 12).cuda()
    layers = nn.ModuleList(list(model.model.layers))  # not from find_decoder_layers
    without = BlockSwap(layers, [0], prefetch_depth = "auto", device = "cuda")
    assert len(without.handles) == 3
    without.remove()
    sw = BlockSwap(layers, [0], prefetch_depth = "auto", device = "cuda", model = model)
    assert len(sw.handles) == 4
    sw.remove()


def test_host_load_embedding_check_counts_the_other_tables_to_come(monkeypatch):
    import sys, pytest
    if not torch.cuda.is_available():
        __import__("pytest").skip("needs CUDA")
    total = 64 * _GiB
    reserve = int(total * _mod._PIN_RESERVE_FRACTION)
    # Room for one 1 MiB table, not two: refuse at the first, before anything moved.
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(reserve + 1536 * 1024, total))
    state = _mod._HostLoad(0, "tail", embeddings = True)
    a, b = nn.Embedding(1024, 256).cuda(), nn.Embedding(1024, 256).to("meta")
    state.layers, state.embedding_prefixes = nn.ModuleList(), {"a.": a, "b.": b}
    with pytest.raises(ValueError):
        state.evict_embedding(a)
    assert state.embeddings == []

class _Ple(nn.Module):

    def __init__(self, table):
        super().__init__()
        self.mlp = nn.Linear(64, 64, bias = False)
        if table:
            self.ple = nn.Module()
            self.ple.ple_embedding = nn.Module()
            self.ple.ple_embedding.ngram_embedding = nn.Embedding(4096, 64)


class _PleModel(nn.Module):
    _no_placement_params = ["ple.ple_embedding.ngram_embedding.weight"]

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Ple(i == 1) for i in range(3)]).requires_grad_(False)


def _cpu_block(layer, monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", lambda *a, **k: object())
    cpu = torch.device("cpu")
    return _mod._Block(layer, {cpu: None}, cpu)


def test_no_placement_table_is_never_packed_into_a_slot(monkeypatch):
    monkeypatch.delenv("UNSLOTH_PLACE_NO_PLACEMENT_PARAMS", raising = False)
    model = _PleModel()
    layers = find_decoder_layers(model)
    table = model.layers[1].ple.ple_embedding.ngram_embedding.weight
    assert [n for n, _ in _mod._swappable(layers[1])] == ["mlp.weight"]
    block = _cpu_block(layers[1], monkeypatch)
    assert all(p is not table for p in block.params)
    assert block.nbytes() == 64 * 64 * 4
    assert _mod._layer_signature(_mod._swappable(layers[1])) == _mod._layer_signature(_mod._swappable(layers[0]))


def test_no_placement_table_stays_out_after_the_load_replaces_its_parameter(monkeypatch):
    # A load to host replaces each Parameter after find_decoder_layers ran.
    monkeypatch.delenv("UNSLOTH_PLACE_NO_PLACEMENT_PARAMS", raising = False)
    model = _PleModel()
    layers = find_decoder_layers(model)
    emb = model.layers[1].ple.ple_embedding.ngram_embedding
    emb.weight = nn.Parameter(torch.zeros_like(emb.weight), requires_grad = False)
    assert [n for n, _ in _mod._swappable(layers[1])] == ["mlp.weight"]


def test_placing_no_placement_params_makes_them_swappable_again(monkeypatch):
    monkeypatch.setenv("UNSLOTH_PLACE_NO_PLACEMENT_PARAMS", "1")
    layers = find_decoder_layers(_PleModel())
    assert len(_mod._swappable(layers[1])) == 2


def test_auto_swap_indices_does_not_count_a_host_kept_table():
    if not torch.cuda.is_available():
        __import__("pytest").skip("needs CUDA")

    class Model(_PleModel):
        def __init__(self):
            nn.Module.__init__(self)
            self.layers = nn.ModuleList([_Ple(i == 7) for i in range(8)]).requires_grad_(False)

    model = Model()
    model.layers.cuda()
    model.layers[7].ple.cpu()
    layers = find_decoder_layers(model)
    dev = layers[0].mlp.weight.device
    layer = 64 * 64 * 4
    # Same signature as its neighbours, so it shares their pool.
    idx, left = auto_swap_indices(layers, 2 * layer, 1, free_bytes = {dev: 0})
    assert idx == [1, 3, 5, 7] and left == 0


class _FrozenInputHead(nn.Module):
    """First swapped layer gets an input that needs no grad (no checkpointing): its release is deferred to the
    end of backward, after the output hook already queued the step end."""

    def __init__(self, d, n):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_XBlock(d) for _ in range(n)])

    def forward(self, x):
        h = x
        for layer in self.model.layers:
            h = layer(h, x)
        return types.SimpleNamespace(loss = h.square().mean(), logits = h)


def test_auto_depth_ends_each_step_once_with_a_deferred_first_release():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("BlockSwap needs a CUDA device")
    torch.manual_seed(0)
    model = _FrozenInputHead(64, 4).cuda()
    sw = BlockSwap(find_decoder_layers(model), [0, 1], prefetch_depth = "auto", device = "cuda")
    for step in range(3):
        x = torch.randn(8, 64, device = "cuda")   # requires_grad = False
        model(x).loss.backward()
        assert sw._steps == step + 1, (step, sw._steps)
    sw.remove()


def test_auto_depth_skips_peaks_for_a_block_without_swappable_weights():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("BlockSwap needs a CUDA device")
    torch.manual_seed(0)
    model = _FrozenInputHead(64, 4).cuda()
    model.model.layers[1].requires_grad_(True)   # entirely trainable: nothing to swap, no home card
    sw = BlockSwap(find_decoder_layers(model), [0, 1], prefetch_depth = "auto", device = "cuda")
    assert sw.blocks[1].home is None
    x = torch.randn(8, 64, device = "cuda", requires_grad = True)
    model(x).loss.backward()
    assert sw._steps == 1
    sw.remove()


def test_chunked_pack_credits_host_loaded_originals(monkeypatch):
    import sys, pytest
    if not torch.cuda.is_available():
        pytest.skip("BlockSwap needs a CUDA device")
    # The chunked fallback releases each original right after packing its block: on top of the host-loaded
    # originals it holds one chunk and one block, not a second copy of everything.
    monkeypatch.setenv("UNSLOTH_DISABLE_PINNED_MEMORY", "1")
    monkeypatch.setattr(_mod, "_can_register", lambda: False)
    blocks = nn.ModuleList([_XBlock(256) for _ in range(8)])
    blocks[:2].cuda(); blocks[2:].cpu()
    block = sum(p.numel() * p.element_size() for p in blocks[2].parameters() if not p.requires_grad)
    need = block * 6
    monkeypatch.setattr(_mod, "_CHUNK_BYTES", block)
    monkeypatch.setattr(_mod, "_host_reserve", lambda total: 0)
    # Room for the chunk + one block + allocator rounding, far less than a second copy of all six.
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(4 * block))
    BlockSwap(blocks, 6, device = "cuda").remove()


def _bhl_layer(idx):
    return nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64))


def _bhl_tensors(n, reads):
    def t(key, shape):
        def load():
            reads.append(key)
            return torch.randn(*shape)
        return load
    out = {}
    for i in range(n):
        out[f"model.layers.{i}.0.weight"] = t(f"{i}.w", (64, 64))
        out[f"model.layers.{i}.0.bias"] = t(f"{i}.b", (64,))
        out[f"model.layers.{i}.1.weight"] = t(f"{i}.ln", (64,))
        out[f"model.layers.{i}.1.bias"] = t(f"{i}.lnb", (64,))
    return out


def test_build_host_layers_refuses_before_reading_anything(monkeypatch):
    import sys, pytest
    pytest.importorskip("bitsandbytes")
    reads = []
    monkeypatch.setattr(_mod, "_host_reserve", lambda total: 0)
    # Each layer is 64*64*4 + 3*64*4 bytes in fp32: room for 2 of the 4.
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(2 * (64 * 64 * 4 + 3 * 64 * 4) + 100))
    with pytest.raises(ValueError, match = "At most 2 layers"):
        _mod.build_host_layers(_bhl_layer, 0, 4, _bhl_tensors(4, reads), "cpu", torch.float32)
    assert reads == []


def test_build_host_layers_builds_when_it_fits(monkeypatch):
    import sys, pytest
    pytest.importorskip("bitsandbytes")
    reads = []
    monkeypatch.setattr(_mod, "_host_reserve", lambda total: 0)
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(1 << 30))
    out = _mod.build_host_layers(_bhl_layer, 0, 2, _bhl_tensors(2, reads), "cpu", torch.float32)
    assert len(out) == 2 and out[0][0].weight.device.type == "cpu" and len(reads) == 8


def test_pool_bytes_accepts_auto_like_the_constructor(monkeypatch):
    monkeypatch.setattr(_mod, "_TIMING", True)
    assert _mod._pool_bytes([10, 10, 10], "auto") == _mod._pool_bytes([10, 10, 10], 1)
    monkeypatch.setattr(_mod, "_TIMING", False)
    assert _mod._pool_bytes([10, 10, 10], "auto") == _mod._pool_bytes([10, 10, 10], 2)


def test_host_reserve_is_at_most_a_quarter_of_a_small_host():
    assert _mod._host_reserve(2 * _GiB) == _GiB // 2
    assert _mod._host_reserve(32 * _GiB) == int(32 * _GiB * 0.15)
    assert _mod._host_reserve(16 * _GiB) == 4 * _GiB


def test_host_load_skips_host_kept_tables_and_tables_inside_selected_layers():
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8, bias = False)
            self.table = nn.Embedding(1 << 20, 64)   # 256 MiB: an extra embedding by size

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(16, 8)
            self.model.layers = nn.ModuleList([Layer() for _ in range(2)])
            self._no_placement_params = ["table.weight"]

        def get_input_embeddings(self):
            return self.model.embed_tokens

    with torch.device("meta"):
        m = Model()
    state = _mod._HostLoad([0, 1], "tail", embeddings = True)
    state.bind(m)
    # The tables live in selected layers (charged with them, not again as streamed tables) and are host-kept.
    assert state.embedding_prefixes == {}
    assert state.shape_bytes == {0: 8 * 8 * 4, 1: 8 * 8 * 4}


def test_unindexed_cuda_device_is_canonical():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("BlockSwap needs a CUDA device")
    blocks = nn.ModuleList([_XBlock(16) for _ in range(3)]).cuda()
    sw = BlockSwap(blocks, [1], device = "cuda")
    assert sw.device.index == torch.cuda.current_device() and not sw.spans_devices
    sw.remove()


def test_pool_growth_keeps_the_optimizer_reserve():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    blocks = nn.ModuleList([_XBlock(256) for _ in range(8)]).cuda()
    sw = BlockSwap(blocks, 6, prefetch_depth = "auto", device = "cuda")
    k = torch.cuda.current_device()
    # Each block's trainable lora (256 x 256 fp32) is reserved for AdamW: 12 bytes per param, all 8 blocks.
    assert sw._optimizer_reserve.get(k) == 8 * 256 * 256 * 12
    free = _mod._free_device_bytes(torch.device("cuda", k))
    sw._optimizer_reserve = {k: free}
    sw._last_peak = {}
    assert sw._grow_pool() is False
    sw.remove()


def test_cuda_index_of_cpu_and_bare_devices():
    assert _mod._cuda_index("cpu") is None
    if torch.cuda.is_available():
        assert _mod._cuda_index("cuda") == torch.cuda.current_device() == _mod._cuda_index(torch.device("cuda", torch.cuda.current_device()))


def test_host_load_charges_a_param_shared_by_selected_layers_once():
    shared = nn.Linear(64, 64, bias = False, device = "meta")

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.own = nn.Linear(8, 8, bias = False)
            self.shared = shared

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Layer() for _ in range(4)])

    with torch.device("meta"):
        m = Model()
    state = _mod._HostLoad([0, 1, 2, 3], "tail")
    state.bind(m)
    own, big = 8 * 8 * 4, 64 * 64 * 4
    assert state.shape_bytes == {0: own + big, 1: own, 2: own, 3: own}


def test_pool_growth_counts_the_allocator_high_water_mark(monkeypatch):
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    blocks = nn.ModuleList([_XBlock(256) for _ in range(8)]).cuda()
    sw = BlockSwap(blocks, 6, prefetch_depth = "auto", device = "cuda")
    k = torch.cuda.current_device()
    sw._last_peak, sw._optimizer_reserve = {}, {}
    total = torch.cuda.get_device_properties(k).total_memory
    # A temporary inside a layer pushed the high-water mark to the whole card: no room left to grow.
    sw._hwm_base = {k: 0}
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dv = None: total)
    assert sw._grow_pool() is False
    sw.remove()


def test_file_backed_cpu_storage_is_not_credited(tmp_path):
    import sys, pytest
    if not sys.platform.startswith("linux"):
        pytest.skip("/proc/self/maps is Linux only")
    path = tmp_path / "w.bin"
    path.write_bytes(b"\0" * (1 << 20))
    mapped = torch.from_file(str(path), shared = True, size = 1 << 20, dtype = torch.uint8)
    ranges = _mod._file_backed_ranges()
    assert _mod._in_ranges(mapped.untyped_storage().data_ptr(), ranges)
    anon = torch.empty(1 << 20, dtype = torch.uint8)
    assert not _mod._in_ranges(anon.untyped_storage().data_ptr(), ranges)


def test_layers_to_come_uses_real_bytes_and_the_smallest_ratio():
    class Layer(nn.Module):
        def __init__(self, device):
            super().__init__()
            self.w = nn.Linear(64, 64, bias = False, device = device)

    meta = 64 * 64 * 4
    state = _mod._HostLoad([0, 1, 2, 3], "tail")
    state.layers = nn.ModuleList([Layer("cpu"), Layer("meta"), Layer("meta"), Layer("cpu")])
    state.indices = [0, 1, 2, 3]
    state.shape_bytes = {i: meta for i in range(4)}
    # Layer 0 moved at full size (a skip module kept it 16-bit); the layer being loaded now is 4-bit (1/4).
    state.done, state.ratios = {0}, {0: 1.0}
    out = state._layers_to_come(meta // 4, 1)
    # Layer 1 and 2 (meta) take the smallest ratio seen, layer 3 is materialized: its own bytes.
    assert out == [meta // 4, meta // 4, meta]


def test_layers_to_come_counts_a_shared_materialized_weight_once():
    shared = nn.Linear(64, 64, bias = False)
    layers = nn.ModuleList([nn.Sequential(shared, nn.Linear(64, 64, bias = False)) for _ in range(4)])
    state = _mod._HostLoad([0, 1, 2, 3], "tail")
    state.layers, state.indices = layers, [0, 1, 2, 3]
    state.shape_bytes = {i: 2 * 64 * 64 * 4 for i in range(4)}
    one = 64 * 64 * 4
    assert state._layers_to_come(2 * one, 0) == [2 * one, one, one, one]
    # Already moved with an earlier layer: not counted again.
    state.moved_storages = {shared.weight.untyped_storage().data_ptr()}
    assert state._layers_to_come(2 * one, 0) == [one] * 4


def test_build_host_layers_publishes_what_is_left_after_each_layer(monkeypatch):
    import sys, types as _t
    published = []
    monkeypatch.setattr(_mod, "_publish_host_pending", lambda n: published.append(n))
    monkeypatch.setattr(_mod, "_check_host_fit", lambda sizes, **k: published.append(("check", sum(sizes))))
    monkeypatch.setattr(_mod, "_host_layer_bytes", lambda make_layer, idx, *a: 100 * (idx + 1))
    monkeypatch.setitem(sys.modules, "bitsandbytes", _t.SimpleNamespace(nn = None))
    _mod.build_host_layers(lambda idx: nn.Identity(), 0, 3, {}, "cpu", torch.float32)
    assert published == [("check", 600), 500, 300, 0]


def test_pool_growth_reads_the_optimizer_reserve_of_adapters_added_after_install():
    fake = types.SimpleNamespace(_reserve_owner = None, _reserve_layers = nn.ModuleList([nn.Linear(4, 4)]))
    assert _mod.BlockSwap._optimizer_bytes(fake) == {}   # nothing trainable on a card yet
    if torch.cuda.is_available():
        fake._reserve_layers.append(nn.Linear(8, 8, bias = False).cuda())
        assert _mod.BlockSwap._optimizer_bytes(fake) == {torch.cuda.current_device(): 64 * 12}


def test_layers_already_on_cpu_are_credited_by_the_streaming_guard(monkeypatch):
    import sys
    # Four 1 GiB-ish layers the device map already put on CPU; little spare RAM: evicting them copies nothing.
    layers = nn.ModuleList([nn.Linear(256, 256, bias = False) for _ in range(4)])
    one = 256 * 256 * 4
    state = _mod._HostLoad([0, 1, 2, 3], "tail")
    state.layers, state.indices = layers, [0, 1, 2, 3]
    state.shape_bytes = {i: one for i in range(4)}
    monkeypatch.setattr(_mod, "_host_reserve", lambda total: 0)
    monkeypatch.setattr(_mod, "_host_memory", lambda: (64 * _GiB, one // 2))
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising = False)
    for i in range(4):
        state.evict(i)
    assert state.done == {0, 1, 2, 3}


def test_streaming_guard_updates_the_rank_ledger_after_each_eviction(monkeypatch):
    import sys
    published = []
    monkeypatch.setattr(_mod, "_publish_host_pending", lambda n: published.append(n))
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(20 * _GiB, 64 * _GiB))
    state, layers = _host_load(4, 3, meta = True)
    state.shape_bytes = {i: 64 * 64 * 4 for i in state.indices}
    one = 64 * 64 * 4
    for i in state.indices:
        _land(layers[i])
        state.evict(i)
    # Layers that landed on CPU copy nothing; what each check published is the rest still to allocate.
    assert published and all(n >= 0 for n in published) and published[-1] == 0


def test_embedding_table_already_on_cpu_is_credited(monkeypatch):
    import sys
    one = 64 * 64 * 4
    state, layers = _host_load(4, 3, meta = True)
    state.shape_bytes = {i: one for i in state.indices}
    state.embedding_prefixes = {"emb.": nn.Embedding(1 << 16, 64)}   # 16 MiB, already on CPU
    monkeypatch.setattr(_mod, "_host_reserve", lambda total: 0)
    monkeypatch.setattr(_mod, "_host_memory", lambda: (64 * _GiB, 4 * one))
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising = False)
    _land(layers[1])
    state.evict(1)
    assert state.done == {1}


def test_layers_to_come_counts_a_partly_loaded_layer_by_what_landed():
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(64, 64, bias = False, device = "meta")
            self.b = nn.Linear(64, 64, bias = False, device = "meta")

    one = 64 * 64 * 4
    layers = nn.ModuleList([Layer(), Layer()])
    state = _mod._HostLoad([0, 1], "tail")
    state.layers, state.indices = layers, [0, 1]
    state.shape_bytes, state.owned_names = {0: 2 * one, 1: 2 * one}, {0: {"a.weight", "b.weight"}, 1: {"a.weight", "b.weight"}}
    layers[1].a.to_empty(device = "cpu")   # half of layer 1 landed on CPU
    out = state._layers_to_come(2 * one, 0)
    assert out[1] == 2 * one and state.cpu_to_come == one   # not charged twice, the landed half is on host


def test_storage_ratio_ignores_a_shared_weight_the_layer_loaded_first():
    shared = nn.Linear(256, 256, bias = False, device = "meta")
    layers = nn.ModuleList([nn.Sequential(shared, nn.Linear(16, 16, bias = False, device = "meta")) for _ in range(2)])
    state = _mod._HostLoad([0, 1], "tail")
    state.layers, state.indices = layers, [0, 1]
    seen, state.shape_bytes, state.owned_names = set(), {}, {}
    for i in (0, 1):
        own = [(n, p) for n, p in _mod._counted_named(layers[i]) if id(p) not in seen]
        seen.update(id(p) for _, p in own)
        state.shape_bytes[i], state.owned_names[i] = sum(p.nbytes for _, p in own), {n for n, _ in own}
    layers[1].to_empty(device = "cpu")    # layer 1 (which does not own the shared weight) finishes first
    import sys
    sys.modules.setdefault("psutil", __import__("psutil"))
    state.evict(1)
    assert state.ratios[1] == 1.0


def test_install_packing_lowers_the_published_reservation_as_it_goes():
    # Eight 1 GiB GPU blocks, nothing to release: the reservation falls by what each packed block committed.
    walk = _mod._HostWalk([(k + 1) * _GiB for k in range(8)], [0] * 8, _GiB)
    assert walk.pending(-1) == 8 * _GiB
    assert [walk.pending(k) for k in (0, 3, 7)] == [7 * _GiB, 4 * _GiB, 0]


def test_packing_the_cpu_prefix_first_keeps_the_gpu_blocks_reserved():
    # Ten 1 GiB blocks, the first five already on the CPU. Once the CPU five are packed (and their originals
    # released), the five GPU blocks still need 5 GiB, not a prorated share of the first estimate.
    walk = _mod._HostWalk([(k + 1) * _GiB for k in range(10)], [_GiB] * 5 + [0] * 5, _GiB)
    pend = [walk.pending(k) for k in range(-1, 10)]
    assert pend[5] == 5 * _GiB + _GiB and pend[-1] == 0     # the five GPU blocks plus one block of margin
    assert all(a >= b for a, b in zip(pend, pend[1:]))
    # GPU blocks first: five committed, then each CPU block's copy is held next to its original only until it is
    # released, so the walk peaks at six blocks (plus the margin).
    walk = _mod._HostWalk([(k + 1) * _GiB for k in range(10)], [0] * 5 + [_GiB] * 5, _GiB)
    assert walk.pending(-1) == 7 * _GiB and walk.pending(4) == 2 * _GiB and walk.pending(9) == 0


def test_a_large_storage_released_last_is_not_credited_up_front():
    # Ten 1 GiB GPU blocks then a 1 GiB CPU view of a 10 GiB storage: packing commits 11 GiB before that storage
    # goes, so the guard must plan for 11 GiB, not 11 - 10 + headroom.
    walk = _mod._HostWalk([(k + 1) * _GiB for k in range(11)], [0] * 10 + [10 * _GiB], _GiB)
    assert walk.pending(-1) == 11 * _GiB


def test_committed_chunk_padding_is_not_reserved_again():
    # 32 blocks of 3 GiB in 4 GiB chunks: 128 GiB in all. With half packed, 64 GiB is committed (padding
    # included), so 64 GiB is still to come, not 128 - 48.
    commits = _mod._chunked_commits([{"cuda": 3 * _GiB}] * 32)
    assert commits[-1] == 128 * _GiB and commits[15] == 64 * _GiB
    walk = _mod._HostWalk(commits, [0] * 32, 3 * _GiB)
    assert walk.pending(15) == 64 * _GiB and walk.pending(31) == 0


def test_host_fit_waits_for_a_peer_that_is_mid_copy(monkeypatch, tmp_path):
    # Rank 1 has copied 4 of its 8 GiB (free RAM already shows it) but has not lowered its reservation yet.
    monkeypatch.setattr(_mod, "_host_reserve", lambda total: 0)
    monkeypatch.setattr(_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("MASTER_PORT", "29512")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(_mod, "_HOST_SESSION", 0)
    monkeypatch.setattr(_mod, "_PEER_RETRY_S", 0.0)
    _mod._begin_host_session()
    ledger = _mod._rank_ledger()
    peer = tmp_path / os.path.basename(ledger[0]) / "1"
    peer.write_text(f"{_mod._HOST_SESSION} {8 * _GiB}")
    monkeypatch.setattr(_mod, "_host_memory", lambda: (64 * _GiB, 12 * _GiB))
    sleeps = []

    def progress(_):
        sleeps.append(1)
        peer.write_text(f"{_mod._HOST_SESSION} {4 * _GiB}")   # the peer publishes its progress
    monkeypatch.setattr(_mod.time, "sleep", progress)
    _mod._check_host_fit([_GiB] * 8)
    assert sleeps


def test_an_early_stopped_recompute_keeps_its_wait_sample():
    class Ev:
        def __init__(self, t): self.t = t
        def query(self): return True
        def elapsed_time(self, other): return other.t - self.t
    fake = types.SimpleNamespace(_tot = {"stall_ms": 0.0, "compute_ms": 0.0, "layers": 0})
    block = types.SimpleNamespace(marks = (Ev(0.0), Ev(3.0), Ev(9.0)), mark_open = 1)
    _mod.BlockSwap._harvest_marks(fake, block)
    assert fake._tot["stall_ms"] == 3.0 and block.mark_open == 0


def test_registered_arena_is_not_prefilled_under_deterministic_algorithms(monkeypatch):
    import torch.utils.deterministic as det
    seen = []
    real_empty = torch.empty

    def spy(*a, **k):
        seen.append(det.fill_uninitialized_memory)
        return real_empty(*a, **k)
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        monkeypatch.setattr(torch, "empty", spy)
        _mod._lazy_host_empty(1 << 20)
    finally:
        torch.use_deterministic_algorithms(prev)
    assert seen == [False] and det.fill_uninitialized_memory is True


def test_shared_cpu_storage_is_credited_once():
    shared = nn.Linear(64, 64, bias = False)            # already on CPU, shared by layers 0 and 1
    layers = nn.ModuleList([nn.Sequential(shared, nn.Linear(8, 8, bias = False)) for _ in range(2)])
    state = _mod._HostLoad([0, 1], "tail")
    state.layers, state.indices = layers, [0, 1]
    state.shape_bytes = {0: 64 * 64 * 4 + 256, 1: 256}
    state.owned_names = {0: {"0.weight", "1.weight"}, 1: {"1.weight"}}
    state._layers_to_come(256, 1)                      # layer 1 finished loading first
    assert state.cpu_to_come == 256                    # layer 0's own weight only: the shared one is layer 1's credit


def test_an_in_flight_wait_only_sample_is_kept():
    class Ev:
        def __init__(self, t, done): self.t, self.done = t, done
        def query(self): return self.done
        def elapsed_time(self, other): return other.t - self.t
    fake = types.SimpleNamespace(_tot = {"stall_ms": 0.0, "compute_ms": 0.0, "layers": 0})
    block = types.SimpleNamespace(marks = (Ev(0.0, True), Ev(2.0, False), Ev(0.0, False)), mark_open = 3)
    fake.blocks = [block]
    fake._harvest_copy = lambda b: None
    fake._harvest_marks = lambda b: _mod.BlockSwap._harvest_marks(fake, b)
    _mod.BlockSwap._harvest(fake)
    assert block.mark_open == 3 and fake._tot["stall_ms"] == 0.0   # not dropped while in flight
    block.marks[1].done = True
    _mod.BlockSwap._harvest(fake)
    assert block.mark_open == 0 and fake._tot["stall_ms"] == 2.0


def test_block_swap_gives_no_credit_for_a_storage_several_layers_view(monkeypatch):
    import sys, os, pytest
    if not torch.cuda.is_available() or os.name == "nt" or not _mod._can_register():
        pytest.skip("needs CUDA")
    # Six host layers whose weights are views into one storage: it stays resident until the last is packed.
    blocks = nn.ModuleList([_XBlock(256) for _ in range(8)])
    blocks[:2].cuda()
    big = torch.zeros(6 * 2 * 256 * 256)
    for k, b in enumerate(blocks[2:]):
        for j, lin in enumerate((b.w, b.c)):
            off = (2 * k + j) * 256 * 256
            lin.weight = nn.Parameter(big[off:off + 256 * 256].view(256, 256), requires_grad = False)
        b.lora.data = b.lora.data.cpu()
    need = sum(p.numel() * p.element_size() for b in blocks[2:] for p in b.parameters())
    monkeypatch.setattr(_mod, "_host_reserve", lambda total: 0)
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(need // 2))
    with pytest.raises(ValueError, match = "offload_layers asked for 6 layers"):
        BlockSwap(blocks, 6, device = "cuda")


def test_deferred_first_release_re_arms_the_next_forward():
    armed = []
    fake = types.SimpleNamespace(_auto = False, _settled = True, blocks = [object()], _step_queued = True)
    fake._release = lambda b: None
    fake._arm = lambda forward: armed.append(forward)
    fake._end_step = lambda: armed.append("end")
    _mod.BlockSwap._bwd(fake, 0)(None)
    assert armed == [True]


def test_a_rank_with_nothing_to_swap_publishes_a_zero_reservation(monkeypatch):
    published = []
    monkeypatch.setattr(_mod, "_publish_host_pending", lambda n: published.append((_mod._HOST_SESSION, n)))
    monkeypatch.setattr(_mod, "_HOST_SESSION", 0)
    BlockSwap(nn.ModuleList([nn.Linear(4, 4) for _ in range(3)]), 0, device = "cpu")
    assert published == [(1, 0)]


def test_a_peer_still_loading_keeps_its_published_reservation(monkeypatch, tmp_path):
    monkeypatch.setattr(_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("MASTER_PORT", "29513")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(_mod, "_HOST_SESSION", 2)
    ledger = _mod._rank_ledger()
    (tmp_path / os.path.basename(ledger[0]) / "1").write_text(f"1 {12 * _GiB}")   # peer: still loading, 12 GiB to go
    assert _mod._others_host_pending(4 * _GiB) == 12 * _GiB
    (tmp_path / os.path.basename(ledger[0]) / "1").write_text("1 0")              # peer done with that session
    assert _mod._others_host_pending(4 * _GiB) == 4 * _GiB


def test_block_swap_gives_no_credit_for_a_storage_a_buffer_still_views(monkeypatch):
    import sys, os, pytest
    if not torch.cuda.is_available() or os.name == "nt" or not _mod._can_register():
        pytest.skip("needs CUDA")
    blocks = nn.ModuleList([_XBlock(256) for _ in range(8)])
    blocks[:2].cuda()
    for b in blocks[2:]:
        b.lora.data = b.lora.data.cpu()
        b.register_buffer("alias", b.w.weight.data.view(-1), persistent = False)   # keeps w's storage alive
        b.register_buffer("alias_c", b.c.weight.data.view(-1), persistent = False)
    need = sum(p.numel() * p.element_size() for b in blocks[2:] for p in b.parameters())
    monkeypatch.setattr(_mod, "_host_reserve", lambda total: 0)
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(need // 2))
    with pytest.raises(ValueError, match = "offload_layers asked for 6 layers"):
        BlockSwap(blocks, 6, device = "cuda")


def test_block_swap_gives_no_credit_for_a_storage_the_inferred_owner_still_views(monkeypatch):
    # The usual call, BlockSwap(find_decoder_layers(model), ...): an alias held by the model outside the decoder
    # list keeps the storage alive, so it must be found through the remembered owner, not only the layers.
    import sys, os, weakref, pytest
    if not torch.cuda.is_available() or os.name == "nt" or not _mod._can_register():
        pytest.skip("needs CUDA")
    blocks = nn.ModuleList([_XBlock(256) for _ in range(8)])
    blocks[:2].cuda()
    owner = nn.Module()
    owner.layers = blocks
    for i, b in enumerate(blocks[2:]):
        b.lora.data = b.lora.data.cpu()
        owner.register_buffer(f"alias_w{i}", b.w.weight.data.view(-1), persistent = False)
        owner.register_buffer(f"alias_c{i}", b.c.weight.data.view(-1), persistent = False)
    monkeypatch.setitem(_mod._LAYER_OWNERS, blocks, weakref.ref(owner))
    need = sum(p.numel() * p.element_size() for b in blocks[2:] for p in b.parameters())
    monkeypatch.setattr(_mod, "_host_reserve", lambda total: 0)
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(need // 2))
    with pytest.raises(ValueError, match = "offload_layers asked for 6 layers"):
        BlockSwap(blocks, 6, device = "cuda")


def test_install_holds_no_reference_to_a_credited_storage_while_packing(monkeypatch):
    # The credit assumes a CPU original's storage is freed as its block is packed: install must not keep it alive.
    import os, weakref, numpy as np, pytest
    if not torch.cuda.is_available() or os.name == "nt" or not _mod._can_register():
        pytest.skip("needs CUDA")
    blocks = nn.ModuleList([_XBlock(256) for _ in range(4)]).cuda()
    big = np.zeros(8 * 256 * 256, dtype = np.float32)       # a larger storage the last block's weight views
    gone = weakref.ref(big)
    blocks[3].w.weight.data = torch.from_numpy(big)[:256 * 256].view(256, 256)
    blocks[3].c.weight.data = blocks[3].c.weight.data.cpu()
    blocks[3].lora.data = blocks[3].lora.data.cpu()
    del big
    seen = {}
    pack = _mod.BlockSwap._pack

    def spy(self):
        pack(self)
        import gc
        gc.collect()
        seen["freed"] = gone() is None
    monkeypatch.setattr(_mod.BlockSwap, "_pack", spy)
    sw = BlockSwap(blocks, [3], device = "cuda")
    assert sw._host_credit[0] >= 8 * 256 * 256 * 4
    assert seen["freed"], "the credited storage was still alive after packing"
    sw.remove()


def test_remove_drops_the_reserve_references():
    fake = types.SimpleNamespace(handles = [], free = {}, blocks = [], _chunks = [], streams = {},
                                 _reserve_layers = nn.ModuleList([nn.Linear(2, 2)]), _reserve_owner = object())
    _mod.BlockSwap.remove(fake)
    assert fake._reserve_layers == () and fake._reserve_owner is None
