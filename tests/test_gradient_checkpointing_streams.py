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

"""Smart gradient checkpointing (CPU offload) when training runs on a non-default stream.

The offload copies run on a side stream and must be ordered against the stream
that produces and consumes each activation. That stream is whatever is current
when forward / backward runs, not the device's default stream: a caller that
trains under `torch.cuda.stream(s)` produces every activation on `s`.

The stress test runs a checkpointed stack forward + backward on side streams
(switching stream every step, fresh inputs every step) and compares every
gradient with the same stack run without checkpointing. A `torch.cuda._sleep`
inside each layer keeps the compute stream behind the host so an unordered copy
reads the activation before it is written.

The same harness checks the opt-in backward prefetch (UNSLOTH_GC_PREFETCH=1):
gradients must be bit-equal with it on and off, and it must actually engage.
"""

import pytest
import torch

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
# Pre-Ampere GPUs have no BF16 compute; the production initializer falls back to FP16 the same way.
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

HIDDEN = 1024
SEQ = 2048          # (1, 2048, 1024) bf16 = 4 MB per activation, above the 2 MB offload cutoff
N_LAYERS = 6
SLEEP_CYCLES = 2_000_000


def _gc():
    import unsloth_zoo.gradient_checkpointing as G
    return G


def _make_weights(seed):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    w_in = (torch.randn(HIDDEN, HIDDEN, device = "cuda", generator = g) / HIDDEN ** 0.5).to(DTYPE)
    ws = [(torch.randn(HIDDEN, HIDDEN, device = "cuda", generator = g) / HIDDEN ** 0.5).to(DTYPE)
          for _ in range(N_LAYERS)]
    w_in.requires_grad_(True)
    for w in ws: w.requires_grad_(True)
    return w_in, ws


def _layer(w, sleep):
    def f(h):
        if sleep: torch.cuda._sleep(SLEEP_CYCLES)
        return h + torch.tanh(h @ w)
    return f


def _step(x, w_in, ws, checkpointed, sleep):
    G = _gc()
    h = x @ w_in
    for w in ws:
        fn = _layer(w, sleep)
        if checkpointed:
            h = G.UnslothCheckpointFunction.apply(fn, True, h)
        else:
            h = fn(h)
    loss = h.float().square().mean()
    loss.backward()
    return loss.detach()


def _grads(w_in, ws):
    out = [w_in.grad.detach().clone()] + [w.grad.detach().clone() for w in ws]
    w_in.grad = None
    for w in ws: w.grad = None
    return out


def _run(streams, inputs, checkpointed, sleep):
    w_in, ws = _make_weights(1234)
    results = []
    for x, s in zip(inputs, streams):
        cur = torch.cuda.current_stream()
        if s is None:
            loss = _step(x, w_in, ws, checkpointed, sleep)
        else:
            s.wait_stream(cur)
            with torch.cuda.stream(s):
                loss = _step(x, w_in, ws, checkpointed, sleep)
            cur.wait_stream(s)
        torch.cuda.synchronize()
        results.append((loss.clone(), _grads(w_in, ws)))
    return results


def _inputs(n):
    g = torch.Generator(device = "cuda").manual_seed(7)
    return [torch.randn(1, SEQ, HIDDEN, device = "cuda", generator = g).to(DTYPE) for _ in range(n)]


@pytest.fixture
def smart_gc(request, monkeypatch):
    prefetch = getattr(request, "param", False)
    monkeypatch.setenv("UNSLOTH_GC_PREFETCH", "1" if prefetch else "0")
    G = _gc()
    G.initialize_unsloth_gradient_checkpointing(DTYPE)
    stats = getattr(G, "GC_PREFETCH_STATS", None)
    if stats is not None:
        for k in stats: stats[k] = 0
    yield G
    monkeypatch.setenv("UNSLOTH_GC_PREFETCH", "0")
    G.initialize_unsloth_gradient_checkpointing(DTYPE)


def _compare(got, ref):
    worst = 0.0
    for (l_got, g_got), (l_ref, g_ref) in zip(got, ref):
        worst = max(worst, (l_got - l_ref).abs().item())
        for a, b in zip(g_got, g_ref):
            worst = max(worst, (a.float() - b.float()).abs().max().item())
    return worst


@needs_cuda
@pytest.mark.parametrize("smart_gc", [False, True], ids = ["demand_reload", "prefetch"], indirect = True)
@pytest.mark.parametrize("sleep", [True, False], ids = ["lagging_stream", "plain"])
def test_side_stream_grads_match_plain_backward(smart_gc, sleep):
    n_steps = 8
    inputs = _inputs(n_steps)
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    # Step 0 is the first pass, step 1 turns double buffering on, then streams keep changing.
    streams = [s1, s1, s2, s1, None, s2, s1, s2]
    ref = _run([None] * n_steps, inputs, checkpointed = False, sleep = False)
    got = _run(streams, inputs, checkpointed = True, sleep = sleep)
    assert smart_gc.CPU_INDEX is not None and smart_gc.CPU_INDEX >= N_LAYERS - 1, \
        "the offload path did not run"
    worst = _compare(got, ref)
    assert worst == 0.0, f"checkpointed grads on a side stream differ from plain backward by {worst}"


@needs_cuda
@pytest.mark.parametrize("smart_gc", [False, True], ids = ["demand_reload", "prefetch"], indirect = True)
def test_default_stream_grads_match_plain_backward(smart_gc):
    n_steps = 4
    inputs = _inputs(n_steps)
    ref = _run([None] * n_steps, inputs, checkpointed = False, sleep = False)
    got = _run([None] * n_steps, inputs, checkpointed = True, sleep = True)
    assert _compare(got, ref) == 0.0


@needs_cuda
@pytest.mark.parametrize("smart_gc", [True], ids = ["prefetch"], indirect = True)
def test_prefetch_engages_and_is_consumed_once(smart_gc):
    n_steps = 4
    _run([None] * n_steps, _inputs(n_steps), checkpointed = True, sleep = False)
    stats = smart_gc.GC_PREFETCH_STATS
    assert smart_gc.USE_DOUBLE_BUFFER, "double buffering did not turn on"
    # Steps 1..3 offload N_LAYERS - 1 inputs (the last layer stays resident) and every
    # backward but the lowest one prefetches the next: N_LAYERS - 2 per step.
    assert stats["issued"] == (n_steps - 1) * (N_LAYERS - 2), stats
    assert stats["consumed"] == stats["issued"], stats
    assert stats["dropped"] == 0, stats


def test_prefetch_descriptor_is_consumed_once_and_stale_ones_dropped(monkeypatch):
    G = _gc()
    monkeypatch.setattr(G, "PREFETCHED", {})
    monkeypatch.setattr(G, "GC_PREFETCH_STATS", {"issued": 0, "consumed": 0, "dropped": 0})
    G.PREFETCHED[0] = (3, 5, 1)
    assert G._take_prefetched(0, 3, 5, 1)
    assert not G._take_prefetched(0, 3, 5, 1), "a prefetch must be consumed once"
    G.PREFETCHED[0] = (3, 5, 1)
    assert not G._take_prefetched(0, 4, 5, 1), "a descriptor from an older graph must not be used"
    G.PREFETCHED[0] = (4, 5, 1)
    assert not G._take_prefetched(0, 4, 4, 0), "a descriptor for another activation must not be used"
    assert G.PREFETCHED == {}
    assert G.GC_PREFETCH_STATS == {"issued": 0, "consumed": 1, "dropped": 2}


def test_prefetch_refuses_without_double_buffer(monkeypatch):
    G = _gc()
    monkeypatch.setattr(G, "GC_PREFETCH", True)
    monkeypatch.setattr(G, "USE_DOUBLE_BUFFER", False, raising = False)
    monkeypatch.setattr(G, "PREFETCHED", {})
    monkeypatch.setattr(G, "OFFLOAD_SLOT_META", [(1, 0, 0, 16, (4,), torch.float32)])
    assert not G._prefetch_offloaded_input(0, 0, 1, 1)
    monkeypatch.setattr(G, "GC_PREFETCH", False)
    monkeypatch.setattr(G, "USE_DOUBLE_BUFFER", True)
    assert not G._prefetch_offloaded_input(0, 0, 1, 1)
    assert G.PREFETCHED == {}
