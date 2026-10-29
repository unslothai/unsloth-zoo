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

# Routed NF4 decode epochs: within one epoch (one-token steps on one cache) a layer's tables are fully checked
# once and then only sentinel-checked; a new cache, a registered module / parameter, a move or a check without
# an epoch starts over, so swaps, merges and moves between generations still rebuild the tables.
import gc
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
import unsloth_zoo.temporary_patches.gpt_oss_routed as GR  # noqa: E402


class _Cache:  # stand-in for a transformers cache object: only its identity matters
    pass


def test_epoch_follows_the_cache_object(monkeypatch):
    monkeypatch.setattr(GR, "_EPOCH_ENABLED", True)
    a, b = _Cache(), _Cache()
    ea = GR.routed_decode_epoch(a)
    assert GR.routed_decode_epoch(a) == ea  # next one-token step on the same cache
    eb = GR.routed_decode_epoch(b)
    assert eb != ea
    assert GR.routed_decode_epoch(a) not in (ea, eb)  # back to a: a new epoch
    assert GR.routed_decode_epoch(None) is None
    ec = GR.routed_decode_epoch(a)
    GR._bump_epoch()  # what a module / parameter registration does
    assert GR.routed_decode_epoch(a) != ec
    del a, b
    gc.collect()
    monkeypatch.setattr(GR, "_EPOCH_ENABLED", False)
    assert GR.routed_decode_epoch(_Cache()) is None


try:
    import bitsandbytes  # noqa: F401
    HAS_BNB = True
except Exception:
    HAS_BNB = False
needs_cuda = pytest.mark.skipif(not (torch.cuda.is_available() and HAS_BNB), reason = "routed NF4 tables need CUDA")
if torch.cuda.is_available() and HAS_BNB:
    from test_gpt_oss_routed_nf4 import DT, E, H, I, _Experts, _linear4bit, _reference, _routes_through  # noqa: E402


def _full_checks(fn):
    before = dict(GR.ROUTED_EPOCH_STATS)
    out = fn()
    return out, GR.ROUTED_EPOCH_STATS["full"] - before["full"], GR.ROUTED_EPOCH_STATS["sentinel"] - before["sentinel"]


@needs_cuda
def test_one_full_check_per_epoch():
    ex = _Experts(True).eval()
    cache = _Cache()
    with torch.no_grad():
        epoch = GR.routed_decode_epoch(cache)
        state = GR.prepare_routed_experts(ex, epoch)  # built (and stamped) now
        assert isinstance(state, dict)
        for _ in range(5):
            got, full, sentinel = _full_checks(lambda: GR.prepare_routed_experts(ex, GR.routed_decode_epoch(cache)))
            assert got is state and full == 0 and sentinel == 1
        # A new cache (the next generate()): one full check, then sentinel checks again.
        cache2 = _Cache()
        got, full, _ = _full_checks(lambda: GR.prepare_routed_experts(ex, GR.routed_decode_epoch(cache2)))
        assert got is state and full == 1
        got, full, sentinel = _full_checks(lambda: GR.prepare_routed_experts(ex, GR.routed_decode_epoch(cache2)))
        assert full == 0 and sentinel == 1
        # A check without an epoch (the experts' eval forward, as in prefill) clears the stamp.
        GR.prepare_routed_experts(ex)
        _, full, _ = _full_checks(lambda: GR.prepare_routed_experts(ex, GR.routed_decode_epoch(cache2)))
        assert full == 1


def _swap(ex, proj, what, mid):
    shape = (H, 2 * I) if proj == "gate_up_projs" else (I, H)
    new = _linear4bit(*shape, True, 999)
    projs = getattr(ex, proj)
    if what == "module":
        projs[mid] = new
    elif what == "weight":
        projs[mid].weight = new.weight
    elif what == "requantize":
        projs[mid].weight.data = new.weight.data
        projs[mid].weight.quant_state = new.weight.quant_state
    elif what == "offset":
        qs = projs[mid].weight.quant_state
        qs.offset = qs.offset + 0.05
    return new


def _routed_matches_reference(ex, mid, epoch):
    T = 4
    x = torch.randn(1, T, H, device = "cuda", dtype = DT)
    idx, w = _routes_through(mid, T)
    with torch.no_grad():
        state = GR.prepare_routed_experts(ex, epoch)
        got = GR.routed_experts_forward(ex, x, idx, w, _state = state)
    ref = _reference(ex, x, idx, w)
    assert got is not None
    return (got.double() - ref).abs().max().item() <= 2 ** -8 * ref.abs().max().item() + 1e-3


@needs_cuda
@pytest.mark.parametrize("proj", ["gate_up_projs", "down_projs"])
@pytest.mark.parametrize("what", ["module", "weight", "requantize", "offset"])
def test_swap_between_generations_rebuilds(proj, what):
    # The next generate() brings a new cache, so a replaced middle expert is caught by the full check.
    ex = _Experts(True).eval()
    mid = E // 2
    cache = _Cache()
    with torch.no_grad():
        epoch = GR.routed_decode_epoch(cache)
        GR.prepare_routed_experts(ex, epoch)
        GR.prepare_routed_experts(ex, GR.routed_decode_epoch(cache))
    new = _swap(ex, proj, what, mid)
    del new
    torch.cuda.empty_cache()
    junk = torch.full((1 << 22,), float("nan"), device = "cuda")  # reuse freed blocks
    assert _routed_matches_reference(ex, mid, GR.routed_decode_epoch(_Cache()))
    del junk


@needs_cuda
@pytest.mark.parametrize("proj", ["gate_up_projs", "down_projs"])
@pytest.mark.parametrize("what", ["module", "weight"])
def test_registered_replacement_inside_an_epoch_rebuilds(proj, what):
    # setattr of a Module / Parameter (how PEFT merges and unloads replace layers) starts a new epoch, even on the
    # same cache between two one-token steps.
    ex = _Experts(True).eval()
    mid = E // 2
    cache = _Cache()
    with torch.no_grad():
        GR.prepare_routed_experts(ex, GR.routed_decode_epoch(cache))
    new = _swap(ex, proj, what, mid)
    del new
    torch.cuda.empty_cache()
    junk = torch.full((1 << 22,), float("nan"), device = "cuda")
    assert _routed_matches_reference(ex, mid, GR.routed_decode_epoch(cache))
    del junk


@needs_cuda
def test_moved_or_requantized_first_expert_inside_an_epoch_rebuilds():
    # .data / quant_state surgery registers nothing; the sentinel watches the first expert of each list, which a
    # move (.to / .cuda) or a requantization of all experts always changes.
    ex = _Experts(True).eval()
    cache = _Cache()
    with torch.no_grad():
        GR.prepare_routed_experts(ex, GR.routed_decode_epoch(cache))
    for proj in ("gate_up_projs", "down_projs"):
        new = _swap(ex, proj, "requantize", 0)
        del new
    torch.cuda.empty_cache()
    junk = torch.full((1 << 22,), float("nan"), device = "cuda")
    _, full, _ = _full_checks(lambda: GR.prepare_routed_experts(ex, GR.routed_decode_epoch(cache)))
    assert full == 1
    assert _routed_matches_reference(ex, 0, GR.routed_decode_epoch(cache))
    del junk


@needs_cuda
def test_kill_switch_checks_every_call(monkeypatch):
    monkeypatch.setattr(GR, "_EPOCH_ENABLED", False)
    ex = _Experts(True).eval()
    cache = _Cache()
    with torch.no_grad():
        GR.prepare_routed_experts(ex, GR.routed_decode_epoch(cache))
        _, full, sentinel = _full_checks(lambda: GR.prepare_routed_experts(ex, GR.routed_decode_epoch(cache)))
    assert full == 1 and sentinel == 0


def test_generate_and_cache_reset_start_a_new_epoch(monkeypatch):
    # A static cache is reset and reused by the next generate(): its identity is no generation boundary.
    monkeypatch.setattr(GR, "_EPOCH_ENABLED", True)
    cache_utils = pytest.importorskip("transformers.cache_utils")
    from transformers.generation.utils import GenerationMixin
    cache = cache_utils.Cache(layers = [cache_utils.StaticLayer(8)])
    e = GR.routed_decode_epoch(cache)
    assert GR.routed_decode_epoch(cache) == e
    cache.reset()
    e2 = GR.routed_decode_epoch(cache)
    assert e2 != e
    with pytest.raises(Exception):
        GenerationMixin.generate(object())  # the wrapper starts the epoch before generate runs
    assert GR.routed_decode_epoch(cache) != e2


@needs_cuda
@pytest.mark.parametrize("proj", ["gate_up_projs", "down_projs"])
@pytest.mark.parametrize("what", ["module", "weight"])
def test_registration_after_the_steps_epoch_capture_rebuilds(proj, what):
    # The decode loop captures the epoch once per token, before its layers run: a replacement registered after
    # that capture (a hook inside the step) must not be served from the old tables.
    ex = _Experts(True).eval()
    mid = E // 2
    cache = _Cache()
    with torch.no_grad():
        GR.prepare_routed_experts(ex, GR.routed_decode_epoch(cache))
        GR.prepare_routed_experts(ex, GR.routed_decode_epoch(cache))
    captured = GR.routed_decode_epoch(cache)
    new = _swap(ex, proj, what, mid)
    del new
    torch.cuda.empty_cache()
    junk = torch.full((1 << 22,), float("nan"), device = "cuda")
    assert _routed_matches_reference(ex, mid, captured)
    del junk


@needs_cuda
@pytest.mark.parametrize("proj", ["gate_up_projs", "down_projs"])
def test_in_place_change_before_a_cache_reset_rebuilds(proj):
    # quant_state surgery in place registers nothing; the next generation (here: the reused cache's reset) checks
    # every expert again.
    cache_utils = pytest.importorskip("transformers.cache_utils")
    ex = _Experts(True).eval()
    mid = E // 2
    cache = cache_utils.Cache(layers = [cache_utils.StaticLayer(8)])
    with torch.no_grad():
        GR.prepare_routed_experts(ex, GR.routed_decode_epoch(cache))
        GR.prepare_routed_experts(ex, GR.routed_decode_epoch(cache))
        getattr(ex, proj)[mid].weight.quant_state.offset.add_(0.25)
    cache.reset()
    assert _routed_matches_reference(ex, mid, GR.routed_decode_epoch(cache))


@needs_cuda
@pytest.mark.parametrize("proj", ["gate_up_projs", "down_projs"])
def test_moved_middle_expert_inside_an_epoch_rebuilds(proj):
    # Module.cpu() / .cuda() frees and reallocates storage without registering anything.
    ex = _Experts(True).eval()
    mid = E // 2
    cache = _Cache()
    with torch.no_grad():
        GR.prepare_routed_experts(ex, GR.routed_decode_epoch(cache))
        GR.prepare_routed_experts(ex, GR.routed_decode_epoch(cache))
    getattr(ex, proj)[mid].cpu()
    torch.cuda.empty_cache()
    junk = torch.full((1 << 24,), float("nan"), device = "cuda")
    getattr(ex, proj)[mid].cuda()
    assert _routed_matches_reference(ex, mid, GR.routed_decode_epoch(cache))
    del junk


@needs_cuda
def test_without_move_tracking_every_call_is_fully_checked(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_FAST_READY", "0")
    ex = _Experts(True).eval()
    cache = _Cache()
    with torch.no_grad():
        GR.prepare_routed_experts(ex, GR.routed_decode_epoch(cache))
        _, full, sentinel = _full_checks(lambda: GR.prepare_routed_experts(ex, GR.routed_decode_epoch(cache)))
    assert full == 1 and sentinel == 0
