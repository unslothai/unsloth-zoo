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

"""Lifecycle of the non-reentrant activation offloader across odd training schedules.

Every scenario compares gradients bit for bit with plain non-reentrant checkpointing and checks that a graph which is
finished or abandoned releases its records, host copies and pinned charge. The "fake" backend drives the real
bookkeeping on CPU (host copies are clones, streams and events are inert); the "cuda" backend is the real class on a
GPU with real side streams and runs only where a GPU exists.
"""
import contextlib
import gc
import weakref

import pytest
import torch
import torch.utils.checkpoint as ckpt


class _FakeStream:
    def __init__(self, name): self.name = name
    def wait_event(self, event): pass
    def wait_stream(self, other): pass
    def __repr__(self): return self.name


class _FakeEvent:
    def record(self, stream = None): pass
    def query(self): return True


_COMPUTE = _FakeStream("compute")


@pytest.fixture
def G(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ALLOW_CPU", "1")
    for name in ("UNSLOTH_GC_OFFLOAD_BACKEND", "UNSLOTH_GC_STREAM_KEEP_LAST", "UNSLOTH_GC_MAX_INFLIGHT_COPIES",
                 "UNSLOTH_GC_PINNED_MAX_GB"):
        monkeypatch.delenv(name, raising = False)
    # The scenarios use small tensors: the size floor has its own tests below.
    monkeypatch.setenv("UNSLOTH_GC_OFFLOAD_MIN_BYTES", "0")
    try:
        from unsloth_zoo import gradient_checkpointing as G
    except ImportError as e:
        pytest.skip(f"unsloth_zoo import unavailable: {e}")
    ledger = {"live": 0}

    def reserve(nbytes):
        ledger["live"] += nbytes
        return True

    def release(nbytes):
        ledger["live"] -= nbytes

    monkeypatch.setattr(G, "_reserve_pinned", reserve)
    monkeypatch.setattr(G, "_release_pinned", release)
    monkeypatch.setattr(G, "PINNED_MEMORY_AVAILABLE", True)
    G._test_ledger = ledger
    return G


def _fake_offloader_class(G):
    class FakeDeviceOffloader(G.NonReentrantActivationOffloader):
        """CPU tensors play the device; host copies are clones so a dropped one is observable."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.host_refs = []

        def _on_host(self, x): return x.numel() == 0  # torch's checkpoint sentinel
        def _streamable(self, x): return True
        def _new_stream(self, device_index): return _FakeStream("side")
        def _current_stream(self, device_index): return _COMPUTE
        def _stream_ctx(self, stream): return contextlib.nullcontext()
        def _new_event(self): return _FakeEvent()
        def _record_stream(self, x, stream): pass

        def _host_copy(self, x, non_blocking = True):
            host = x.detach().clone()
            self.host_refs.append(weakref.ref(host))
            return host

        def _device_copy(self, host, device): return host.clone()
        def _empty_on_device(self, host, device): return torch.empty_like(host)

    return FakeDeviceOffloader


BACKENDS = ["fake", pytest.param("cuda", marks = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device"))]
MODES = ["default", "side1", "stream0", "stream1"]


def _configure(monkeypatch, mode):
    if mode.startswith("side"):
        monkeypatch.setenv("UNSLOTH_GC_MAX_INFLIGHT_COPIES", mode[-1])
    else:
        monkeypatch.setenv("UNSLOTH_GC_MAX_INFLIGHT_COPIES", "0")
    if mode.startswith("stream"):
        monkeypatch.setenv("UNSLOTH_GC_STREAM_KEEP_LAST", mode[-1])
    return mode.startswith("stream")


@pytest.fixture
def env(G, monkeypatch, request):
    backend, mode = request.param
    stream = _configure(monkeypatch, mode)
    cls = _fake_offloader_class(G) if backend == "fake" else G.NonReentrantActivationOffloader
    device = "cpu" if backend == "fake" else "cuda"
    if hasattr(G, "_new_activation_offloader"):
        monkeypatch.setattr(G, "_new_activation_offloader", lambda: cls(stream_mode = stream))
    return {"G": G, "cls": cls, "stream": stream, "device": device, "backend": backend, "mode": mode,
            "new": lambda: cls(stream_mode = stream)}


def _params(backends = BACKENDS, modes = MODES):
    out = []
    for b in backends:
        values = b.values if hasattr(b, "values") else (b,)
        marks = b.marks if hasattr(b, "marks") else ()
        for m in modes:
            out.append(pytest.param((values[0], m), id = f"{values[0]}-{m}", marks = marks))
    return out


class _Block(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = torch.nn.Linear(d, 2 * d)
        self.b = torch.nn.Linear(2 * d, d)

    def forward(self, h):
        return h + self.b(torch.nn.functional.gelu(self.a(h)))


class _Stack(torch.nn.Module):
    """A model whose checkpoints offload through `hooks()` (None = plain non-reentrant checkpoint)."""
    def __init__(self, d = 16, n = 4, seed = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.layers = torch.nn.ModuleList([_Block(d) for _ in range(n)])
        self.hooks = None

    def forward(self, h):
        for layer in self.layers:
            if self.hooks is None:
                h = ckpt.checkpoint(layer, h, use_reentrant = False)
            else:
                with self.hooks():
                    h = ckpt.checkpoint(layer, h, use_reentrant = False)
        return h


def _grads(model, *inputs):
    out = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
    out += [x.grad.clone() if x.grad is not None else None for x in inputs]
    model.zero_grad(set_to_none = True)
    for x in inputs: x.grad = None
    return out


def _assert_same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert (x is None) == (y is None)
        if x is not None: assert torch.equal(x, y)


def _input(device, *shape, seed = 1):
    torch.manual_seed(seed)
    return torch.randn(*shape, device = device, requires_grad = True)


class _Spy:
    """Labels every record pack() returns with the current phase, and checks on every unpack that the prefetches it
    issued only target records of the same forward as the record being unpacked."""
    def __init__(self, offloader, phase = None):
        self.phase, self.records, self.violations = phase, {}, []
        self.labels = weakref.WeakKeyDictionary()
        pack, unpack = offloader.pack, offloader.unpack

        def spy_pack(x):
            packed = pack(x)
            if type(packed) is not tuple:
                self.records.setdefault(self.phase, []).append(weakref.ref(packed))
                self.labels[packed] = self.phase
            return packed

        def spy_unpack(packed):
            if type(packed) is tuple: return unpack(packed)
            before = {id(r) for r in self._waiting()}
            out = unpack(packed)
            for r in self._waiting():
                if id(r) not in before and self.labels.get(r) != self.labels.get(packed):
                    self.violations.append((self.labels.get(packed), self.labels.get(r)))
            return out

        offloader.pack, offloader.unpack = spy_pack, spy_unpack

    def _waiting(self):
        return [r for refs in self.records.values() for r in _live(refs) if r.ready is not None]


def _spy_records(offloader):
    return _Spy(offloader)


def _live(refs):
    return [r() for r in refs if r() is not None]


def _settle(env):
    gc.collect()
    if env["backend"] == "cuda": torch.cuda.synchronize()


def _owned_hooks(env, model):
    """Wire `model` to its own offloader (forward identity from its forward pre-hook)."""
    G = env["G"]
    model.hooks = lambda: G.non_reentrant_offload_hooks(model)
    return G.activation_offloader_for(model)


# 1. forward -> backward, every mode, three steps (the later ones reuse the cached state).
@pytest.mark.parametrize("env", _params(), indirect = True)
def test_forward_backward_matches_plain_checkpoint(env):
    model = _Stack().to(env["device"])
    x = _input(env["device"], 2, 5, 16)
    model(x).square().mean().backward()
    ref = _grads(model, x)
    offloader = env["new"]()
    model.hooks = offloader.hooks
    for _ in range(3):
        model(x).square().mean().backward()
        _assert_same(_grads(model, x), ref)
    _settle(env)
    assert env["G"]._test_ledger["live"] == 0
    if env["backend"] == "fake":
        assert offloader.host_refs and not _live(offloader.host_refs)


# 2. two forwards, then one combined backward, or two separate backwards in either order.
@pytest.mark.parametrize("env", _params(), indirect = True)
@pytest.mark.parametrize("schedule", ["combined", "in_order", "reversed"])
def test_two_forwards_then_backward(env, schedule):
    model = _Stack().to(env["device"])
    x1, x2 = _input(env["device"], 2, 5, 16, seed = 1), _input(env["device"], 2, 7, 16, seed = 2)

    def run():
        l1 = model(x1).square().mean()
        l2 = model(x2).square().mean()
        if schedule == "combined":
            (l1 + l2).backward()
        elif schedule == "in_order":
            l1.backward(); l2.backward()
        else:
            l2.backward(); l1.backward()
        return _grads(model, x1, x2)

    ref = run()
    offloader = env["new"]()
    model.hooks = offloader.hooks
    _assert_same(run(), ref)
    _settle(env)
    assert env["G"]._test_ledger["live"] == 0


# 2b. pipelined microbatches on one offloader: fwd 1, fwd 2, bwd 1, fwd 3, bwd 2, bwd 3. The forward that starts
# after a backward must not hand microbatch 2's backward a prefetch order that points into microbatch 3.
@pytest.mark.parametrize("env", _params(), indirect = True)
def test_pipelined_microbatches(env):
    model = _Stack(n = 5).to(env["device"])
    xs = [_input(env["device"], 2, 5 + i, 16, seed = i) for i in range(3)]

    def run(spy = None):
        def fwd(i):
            if spy is not None: spy.phase = f"mb{i}"
            return model(xs[i]).square().mean()
        l0, l1 = fwd(0), fwd(1)
        l0.backward()
        l2 = fwd(2)
        l1.backward()
        l2.backward()
        return _grads(model, *xs)

    ref = run()
    offloader = env["new"]()
    model.hooks = offloader.hooks
    spy = _Spy(offloader)
    _assert_same(run(spy), ref)
    assert not spy.violations
    _settle(env)
    assert env["G"]._test_ledger["live"] == 0


# 3. interleaved models (policy and reference both with grad): fwd A, fwd B, bwd in both orders.
@pytest.mark.parametrize("env", _params(), indirect = True)
@pytest.mark.parametrize("order", ["AB", "BA"])
def test_interleaved_models_do_not_touch_each_other(env, order):
    a, b = _Stack(seed = 0).to(env["device"]), _Stack(d = 16, n = 6, seed = 3).to(env["device"])
    xa, xb = _input(env["device"], 2, 5, 16, seed = 1), _input(env["device"], 3, 4, 16, seed = 2)

    def run():
        la, lb = a(xa).square().mean(), b(xb).square().mean()
        for which in order:
            (la if which == "A" else lb).backward()
        return _grads(a, xa) + _grads(b, xb)

    ref = run()
    off_a, off_b = _owned_hooks(env, a), _owned_hooks(env, b)
    assert off_a is not off_b
    spy_a, spy_b = _Spy(off_a, "A"), _Spy(off_b, "B")
    la, lb = a(xa).square().mean(), b(xb).square().mean()
    first, second = (la, lb) if order == "AB" else (lb, la)
    first.backward()
    if order == "BA":
        # B's backward must not have prefetched or moved any of A's still pending records.
        assert not [r for r in _live(spy_a.records["A"]) if r.ready is not None or r.unpacked]
        assert off_a.stats["prefetch_hits"] == 0
    second.backward()
    _assert_same(_grads(a, xa) + _grads(b, xb), ref)
    assert not spy_a.violations and not spy_b.violations
    _settle(env)
    assert env["G"]._test_ledger["live"] == 0


# 4. abandoned graphs: records, host copies and pinned charge are released, and metadata stays bounded.
@pytest.mark.parametrize("env", _params(), indirect = True)
def test_abandoned_graphs_release_everything(env):
    model = _Stack().to(env["device"])
    x = _input(env["device"], 2, 5, 16)
    offloader = env["new"]()
    model.hooks = offloader.hooks
    spy = _Spy(offloader, "dropped")
    for _ in range(40):
        loss = model(x).square().mean()
        del loss
    _settle(env)
    assert not _live(spy.records["dropped"])
    assert env["G"]._test_ledger["live"] == 0
    if env["backend"] == "fake":
        assert not _live(offloader.host_refs)
    # 40 dropped forwards of 4 checkpoints must not leave 160 dead entries behind.
    assert len(offloader.order) <= len(model.layers)
    # And training afterwards is still exact.
    model.hooks = None
    model(x).square().mean().backward()
    ref = _grads(model, x)
    model.hooks = offloader.hooks
    model(x).square().mean().backward()
    _assert_same(_grads(model, x), ref)


# 5. an evaluation or generation forward between a training forward and its backward.
@pytest.mark.parametrize("env", _params(), indirect = True)
@pytest.mark.parametrize("eval_mode", ["no_grad", "grad_enabled", "inference_mode"])
def test_eval_forward_between_forward_and_backward(env, eval_mode):
    model = _Stack().to(env["device"])
    x, e = _input(env["device"], 2, 5, 16, seed = 1), _input(env["device"], 1, 9, 16, seed = 4)
    model(x).square().mean().backward()
    ref = _grads(model, x)
    offloader = _owned_hooks(env, model)
    spy = _Spy(offloader, "train")
    loss = model(x).square().mean()
    spy.phase = "eval"
    ctx = {"no_grad": torch.no_grad, "grad_enabled": contextlib.nullcontext, "inference_mode": torch.inference_mode}[eval_mode]
    with ctx():
        out = model(e)
    loss.backward()
    _assert_same(_grads(model, x), ref)
    # The training backward never prefetched the evaluation graph's records.
    assert not [r for r in _live(spy.records.get("eval", [])) if r.ready is not None]
    assert not spy.violations
    del out
    _settle(env)
    assert env["G"]._test_ledger["live"] == 0


# 6. retain_graph: two backward passes through the same graph.
@pytest.mark.parametrize("env", _params(), indirect = True)
def test_retain_graph_double_backward(env):
    model = _Stack().to(env["device"])
    x = _input(env["device"], 2, 5, 16)

    def run():
        loss = model(x).square().mean()
        loss.backward(retain_graph = True)
        first = _grads(model, x)
        loss.backward()
        return first + _grads(model, x)

    ref = run()
    offloader = env["new"]()
    model.hooks = offloader.hooks
    _assert_same(run(), ref)
    _settle(env)
    assert env["G"]._test_ledger["live"] == 0


# 7. torch.autograd.grad instead of backward, including a partial request.
@pytest.mark.parametrize("env", _params(), indirect = True)
def test_autograd_grad(env):
    model = _Stack().to(env["device"])
    x = _input(env["device"], 2, 5, 16)
    wanted = [x] + list(model.layers[-1].parameters())

    def run():
        return list(torch.autograd.grad(model(x).square().mean(), wanted))

    ref = run()
    offloader = env["new"]()
    model.hooks = offloader.hooks
    _assert_same(run(), ref)
    _settle(env)
    assert env["G"]._test_ledger["live"] == 0


class _Nested(torch.nn.Module):
    """An outer checkpoint whose function checkpoints two inner blocks, each through `hooks`."""
    def __init__(self, d = 16):
        super().__init__()
        self.inner = torch.nn.ModuleList([_Block(d), _Block(d)])
        self.hooks = None

    def forward(self, h):
        for block in self.inner:
            if self.hooks is None:
                h = ckpt.checkpoint(block, h, use_reentrant = False)
            else:
                with self.hooks():
                    h = ckpt.checkpoint(block, h, use_reentrant = False)
        return h


# 8. nested checkpoints: the outer recompute re-enters the inner checkpoints during backward.
@pytest.mark.parametrize("env", _params(), indirect = True)
def test_nested_checkpoints(env):
    torch.manual_seed(0)
    outers = torch.nn.ModuleList([_Nested() for _ in range(3)]).to(env["device"])
    x = _input(env["device"], 2, 5, 16)
    offloader = env["new"]()

    def run(hooks):
        h = x
        for outer in outers:
            outer.hooks = hooks
            if hooks is None:
                h = ckpt.checkpoint(outer, h, use_reentrant = False)
            else:
                with hooks():
                    h = ckpt.checkpoint(outer, h, use_reentrant = False)
        h.square().mean().backward()
        return _grads(outers, x)

    ref = run(None)
    spy = _Spy(offloader, "forward")
    hooks = offloader.hooks
    spy_unpack = offloader.unpack

    def unpack(packed):
        spy.phase = "recompute"  # anything packed from here on is packed by a recompute
        return spy_unpack(packed)

    offloader.unpack = unpack
    _assert_same(run(hooks), ref)
    # The recompute's packs must not hijack the outer forward's prefetch order.
    assert not spy.violations
    _settle(env)
    assert not _live(spy.records.get("forward", []))
    assert not _live(spy.records.get("recompute", []))
    assert env["G"]._test_ledger["live"] == 0


class _TwoInput(torch.nn.Module):
    def __init__(self, d = 16):
        super().__init__()
        self.a = torch.nn.Linear(d, d)
        self.b = torch.nn.Linear(d, d)

    def forward(self, h, g):
        return torch.tanh(self.a(h) + self.b(g)), g * 1.5


# 9. checkpoints with several saved inputs.
@pytest.mark.parametrize("env", _params(), indirect = True)
def test_multi_input_checkpoints(env):
    torch.manual_seed(0)
    layers = torch.nn.ModuleList([_TwoInput() for _ in range(4)]).to(env["device"])
    h0, g0 = _input(env["device"], 2, 5, 16, seed = 1), _input(env["device"], 2, 5, 16, seed = 2)

    def run(hooks):
        h, g = h0, g0
        for layer in layers:
            if hooks is None:
                h, g = ckpt.checkpoint(layer, h, g, use_reentrant = False)
            else:
                with hooks():
                    h, g = ckpt.checkpoint(layer, h, g, use_reentrant = False)
        (h.square().mean() + g.mean()).backward()
        return _grads(layers, h0, g0)

    ref = run(None)
    offloader = env["new"]()
    _assert_same(run(offloader.hooks), ref)
    if env["stream"]:
        assert offloader.stats["prefetch_hits"] == offloader.stats["prefetched"] > 0
    _settle(env)
    assert env["G"]._test_ledger["live"] == 0


class _Boom(RuntimeError):
    pass


# 10. an exception in the middle of a forward: partial state is released and the next step is exact.
@pytest.mark.parametrize("env", _params(), indirect = True)
def test_exception_mid_forward(env):
    model = _Stack(n = 5).to(env["device"])
    x = _input(env["device"], 2, 5, 16)
    model(x).square().mean().backward()
    ref = _grads(model, x)
    offloader = env["new"]()
    model.hooks = offloader.hooks
    spy = _Spy(offloader, "failed")
    calls = {"n": 0}
    original_forward = model.layers[3].forward

    def exploding(h):
        calls["n"] += 1
        if calls["n"] == 1: raise _Boom("injected")
        return original_forward(h)

    model.layers[3].forward = exploding
    with pytest.raises(_Boom):
        model(x).square().mean().backward()
    _settle(env)
    assert spy.records["failed"] and not _live(spy.records["failed"])
    assert env["G"]._test_ledger["live"] == 0
    spy.phase = "after"
    model(x).square().mean().backward()
    _assert_same(_grads(model, x), ref)


# 11. varying shapes step to step.
@pytest.mark.parametrize("env", _params(), indirect = True)
def test_varying_shapes(env):
    model = _Stack().to(env["device"])
    offloader = env["new"]()
    for step, (batch, seq) in enumerate([(2, 5), (1, 13), (3, 2), (2, 5)]):
        x = _input(env["device"], batch, seq, 16, seed = step)
        model.hooks = None
        model(x).square().mean().backward()
        ref = _grads(model, x)
        model.hooks = offloader.hooks
        model(x).square().mean().backward()
        _assert_same(_grads(model, x), ref)
    _settle(env)
    assert env["G"]._test_ledger["live"] == 0


# ---- Eligibility, counters, ownership.

@pytest.fixture
def fake(G, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GC_MAX_INFLIGHT_COPIES", "0")
    return _fake_offloader_class(G)


def test_size_floor_is_off_by_default(G, fake, monkeypatch):
    monkeypatch.delenv("UNSLOTH_GC_OFFLOAD_MIN_BYTES", raising = False)
    offloader = fake(stream_mode = False)
    assert offloader.min_bytes == 0
    record = offloader.pack(torch.randn(1000, requires_grad = True))  # 4 KB still offloaded
    assert type(record) is not tuple and record.host is not None
    assert offloader.stats["kept_small"] == 0 and offloader.stats["offloaded"] == 1


def test_small_tensors_stay_on_device_with_a_floor(G, fake, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GC_OFFLOAD_MIN_BYTES", str(1 << 20))
    offloader = fake(stream_mode = False)
    assert offloader.min_bytes == 1 << 20
    small = torch.randn(1000, requires_grad = True)                  # 4 KB
    packed = offloader.pack(small)
    assert type(packed) is tuple and packed[1] is small
    big = torch.randn(1 << 18, requires_grad = True)                 # exactly 1 MiB
    record = offloader.pack(big)
    assert type(record) is not tuple and record.host is not None
    assert offloader.stats["kept_small"] == 1 and offloader.stats["kept_small_bytes"] == 4000
    assert offloader.stats["offloaded"] == 1 and offloader.stats["offloaded_bytes"] == 1 << 20
    assert torch.equal(offloader.unpack(record), big.detach())
    assert offloader.stats["restored_bytes"] == 1 << 20


def test_min_bytes_env_override(G, fake, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GC_OFFLOAD_MIN_BYTES", "0")
    assert fake(stream_mode = False).min_bytes == 0
    monkeypatch.setenv("UNSLOTH_GC_OFFLOAD_MIN_BYTES", "4096")
    offloader = fake(stream_mode = False)
    assert type(offloader.pack(torch.randn(1024, requires_grad = True))) is not tuple  # 4096 bytes: offloaded
    assert type(offloader.pack(torch.randn(1023, requires_grad = True))) is tuple
    monkeypatch.setenv("UNSLOTH_GC_OFFLOAD_MIN_BYTES", "not a number")
    assert fake(stream_mode = False).min_bytes == 0


@pytest.mark.parametrize("kind", ["transposed", "expanded", "strided_slice", "prefix_view", "offset_view"])
def test_non_contiguous_and_aliased_tensors_stay_on_device(G, fake, kind):
    offloader = fake(stream_mode = False)
    base = torch.randn(64, 64, requires_grad = True)
    x = {
        "transposed": lambda: base.t(),
        "expanded": lambda: base[:1].expand(64, 64),
        "strided_slice": lambda: base[:, ::2],
        "prefix_view": lambda: base.view(-1)[:1024],
        "offset_view": lambda: base.view(-1)[1024:],
    }[kind]()
    packed = offloader.pack(x)
    assert type(packed) is tuple and packed[1] is x
    reason = "non_contiguous" if kind in ("transposed", "expanded", "strided_slice") else "aliased"
    assert offloader.stats["kept_" + reason] == 1
    assert offloader.stats["kept_" + reason + "_bytes"] == x.numel() * x.element_size()
    assert offloader.stats["offloaded"] == 0 and not offloader.host_refs
    # A contiguous tensor that owns its whole storage is still offloaded.
    whole = base.detach().clone().requires_grad_()
    assert type(offloader.pack(whole)) is not tuple


@pytest.mark.parametrize("stream", [False, True])
def test_byte_counters_balance_over_a_step(G, fake, monkeypatch, stream):
    monkeypatch.setenv("UNSLOTH_GC_STREAM_KEEP_LAST", "0")
    model = _Stack(n = 5)
    x = _input("cpu", 2, 5, 16)
    offloader = fake(stream_mode = stream)
    model.hooks = offloader.hooks
    G.get_activation_offload_stats(reset = True)
    model(x).square().mean().backward()
    stats = offloader.stats
    per_input = x.numel() * x.element_size()
    assert stats["offloaded"] == stats["restored"] == 5
    assert stats["offloaded_bytes"] == stats["restored_bytes"] == 5 * per_input
    if stream:
        assert stats["prefetch_hits"] == stats["prefetched"] == 4
        assert stats["prefetched_bytes"] == 4 * per_input
    total = G.get_activation_offload_stats(reset = True)
    assert total["offloaded_bytes"] >= 5 * per_input
    assert offloader.stats["offloaded_bytes"] == 0  # reset


def test_each_owner_gets_its_own_offloader_and_the_default_is_shared(G, fake, monkeypatch):
    monkeypatch.setattr(G, "_new_activation_offloader", lambda: fake(stream_mode = True))
    monkeypatch.setattr(G, "NONREENTRANT_OFFLOADER", None)
    a, b = _Stack(), _Stack()
    assert G.activation_offloader_for(a) is G.activation_offloader_for(a)
    assert G.activation_offloader_for(a) is not G.activation_offloader_for(b)
    assert G.activation_offloader_for(a).owned
    # No owner: one process-wide offloader, as before.
    with G.non_reentrant_offload_hooks():
        pass
    shared = G.NONREENTRANT_OFFLOADER
    with G.non_reentrant_offload_hooks():
        pass
    assert G.NONREENTRANT_OFFLOADER is shared and not shared.owned
    # The owner's offloader goes away with the owner.
    ref = weakref.ref(G.activation_offloader_for(b))
    del b
    gc.collect()
    assert ref() is None


def test_owner_forward_pre_hook_separates_pending_forwards(G, fake, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GC_STREAM_KEEP_LAST", "0")
    monkeypatch.setattr(G, "_new_activation_offloader", lambda: fake(stream_mode = True))
    model = _Stack(n = 4)
    offloader = _owned_hooks({"G": G}, model)
    x1, x2 = _input("cpu", 2, 5, 16, seed = 1), _input("cpu", 2, 5, 16, seed = 2)
    spy = _Spy(offloader, "first")
    l1 = model(x1).square().mean()
    spy.phase = "second"
    l2 = model(x2).square().mean()
    first = [r() for r in spy.records["first"]]
    second = [r() for r in spy.records["second"]]
    assert len({id(r.order) for r in first}) == 1 and len({id(r.order) for r in second}) == 1
    assert first[0].order is not second[0].order
    assert [r.index for r in second] == list(range(4))
    l2.backward()
    # The second forward's backward ended at its own first checkpoint: nothing of the first one was reloaded.
    assert not [r for r in first if r.ready is not None or r.unpacked]
    l1.backward()
    assert not spy.violations
    assert offloader.stats["prefetch_hits"] == offloader.stats["prefetched"] == 6
