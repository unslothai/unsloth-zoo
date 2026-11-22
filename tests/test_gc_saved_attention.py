# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""UNSLOTH_GC_SAVE_ATTENTION=1 reuses the first pass's attention outputs in checkpoint replay; CPU
only, where SDPA's CPU flash kernel returns the logsumexp the backward needs."""
import sys

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

G = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
from unsloth_zoo import saved_attention as SA  # noqa: E402

B, S, H, KV, D = 2, 64, 4, 4, 16
DM = H * D


class _KernelCounter(TorchDispatchMode):
    """Counts attention forward kernels that really run (a reused output never reaches this mode)."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def __torch_dispatch__(self, func, types, args = (), kwargs = None):
        if func is torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default:
            self.n += 1
        return func(*args, **(kwargs or {}))


def _sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal = True)


def _make(n_layers = 3, kv = KV, seed = 0):
    g = torch.Generator().manual_seed(seed)
    params = []
    for _ in range(n_layers):
        params.append([
            (torch.randn(DM, H * D, generator = g) * 0.1).requires_grad_(),
            (torch.randn(DM, kv * D, generator = g) * 0.1).requires_grad_(),
            (torch.randn(DM, kv * D, generator = g) * 0.1).requires_grad_(),
            (torch.randn(H * D, DM, generator = g) * 0.1).requires_grad_(),
        ])
    x = torch.randn(B, S, DM, generator = g)
    return params, x


def _layer(weights, attention, call):
    wq, wk, wv, wo = weights
    kv = wk.shape[1] // D

    def forward(h):
        q = (h @ wq).view(B, S, H, D).transpose(1, 2)
        k = (h @ wk).view(B, S, kv, D).transpose(1, 2)
        v = (h @ wv).view(B, S, kv, D).transpose(1, 2)
        if kv != H:
            k = k.repeat_interleave(H // kv, dim = 1)
            v = v.repeat_interleave(H // kv, dim = 1)
        a = call(attention, q, k, v)
        return h + a.transpose(1, 2).reshape(B, S, H * D) @ wo
    return forward


def _run(params, x, ckpt, attention = _sdpa, call = None, steps = 1):
    if call is None:
        call = lambda fn, q, k, v: SA.saved_attention_call(fn, q, k, v)  # noqa: E731
    counter = _KernelCounter()
    grads = None
    with counter:
        for _ in range(steps):
            for w in params:
                for p in w:
                    p.grad = None
            h = x.clone().requires_grad_()
            for w in params:
                f = _layer(w, attention, call)
                h = G.UnslothCheckpointFunction.apply(f, True, h) if ckpt else f(h)
            loss = h.float().pow(2).mean()
            loss.backward()
            grads = [p.grad.clone() for w in params for p in w]
    return loss.detach(), grads, counter.n


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv("UNSLOTH_GC_SAVE_ATTENTION", raising = False)
    monkeypatch.setenv("UNSLOTH_DISABLE_PINNED_MEMORY", "1")
    monkeypatch.setenv("UNSLOTH_DISABLE_DOUBLE_BUFFER", "1")
    G._double_buffer_disabled.cache_clear()
    G.initialize_unsloth_gradient_checkpointing(torch.float32)
    # Hidden states stay on the device: this file tests the attention replay, not the host offload.
    monkeypatch.setattr(G, "MINIMUM_SIZE", 1 << 62)
    SA.SAVED_ATTENTION_STATS.clear()
    yield
    SA.SAVED_ATTENTION_STATS.clear()


def _close(a, b):
    la, ga = a[:2]
    lb, gb = b[:2]
    torch.testing.assert_close(la, lb, rtol = 1e-5, atol = 1e-6)
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, rtol = 1e-4, atol = 1e-6)


def test_replay_skips_the_attention_forward_and_matches(monkeypatch):
    params, x = _make()
    plain = _run(params, x, ckpt = False)
    full = _run(params, x, ckpt = True)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    saved = _run(params, x, ckpt = True)
    _close(full, plain)
    _close(saved, full)
    # Today's checkpointing runs every attention forward twice; saving runs it once.
    assert plain[2] == 3 and full[2] == 6 and saved[2] == 3
    assert SA.SAVED_ATTENTION_STATS["stored"] == 3
    assert SA.SAVED_ATTENTION_STATS["reused"] == 3
    assert SA.SAVED_ATTENTION_STATS["mismatch"] == 0 and SA.SAVED_ATTENTION_STATS["unmatched"] == 0


def test_off_by_default_changes_nothing():
    params, x = _make()
    full = _run(params, x, ckpt = True)
    assert full[2] == 6
    assert sum(SA.SAVED_ATTENTION_STATS.values()) == 0


def test_several_steps_reuse_each_layer_once(monkeypatch):
    params, x = _make()
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    saved = _run(params, x, ckpt = True, steps = 3)
    monkeypatch.delenv("UNSLOTH_GC_SAVE_ATTENTION")
    full = _run(params, x, ckpt = True, steps = 3)
    _close(saved, full)
    assert saved[2] == 9
    assert SA.SAVED_ATTENTION_STATS["reused"] == 9


def test_a_replay_that_differs_recomputes(monkeypatch):
    """The replay runs other shapes than the first pass: nothing stale is handed back."""
    params, x = _make(n_layers = 1)
    state = {"pass": 0}

    def attention(q, k, v):
        state["pass"] += 1
        if state["pass"] == 2:
            out = torch.cat([_sdpa(q[:1], k[:1], v[:1]), _sdpa(q[1:], k[1:], v[1:])], 0)
            return out
        return _sdpa(q, k, v)

    full = _run(params, x, ckpt = False, attention = _sdpa)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    saved = _run(params, x, ckpt = True, attention = attention)
    _close(saved, full)
    assert SA.SAVED_ATTENTION_STATS["mismatch"] == 1
    assert SA.SAVED_ATTENTION_STATS["reused"] == 0


def test_math_backend_is_recomputed(monkeypatch):
    """A call with no recognised kernel (SDPA math with an explicit float mask) stores nothing."""
    params, x = _make(n_layers = 2)
    mask = torch.zeros(S, S).masked_fill(torch.ones(S, S, dtype = torch.bool).triu(1), float("-inf"))

    def attention(q, k, v):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            return F.scaled_dot_product_attention(q, k, v, attn_mask = mask)

    full = _run(params, x, ckpt = True, attention = attention)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    saved = _run(params, x, ckpt = True, attention = attention)
    _close(saved, full)
    assert SA.SAVED_ATTENTION_STATS["no_kernel"] == 2
    assert SA.SAVED_ATTENTION_STATS["reused"] == 0


def test_function_level_keeps_only_the_output(monkeypatch):
    """saved_function_call (FlexAttention's route): the kept graph addresses Q / K / V through placeholders."""
    params, x = _make()
    call = lambda fn, q, k, v: SA.saved_function_call(fn, 3, q, k, v)  # noqa: E731
    full = _run(params, x, ckpt = True)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    saved = _run(params, x, ckpt = True, call = call, steps = 2)
    _close(saved, full)
    assert saved[2] == 3 * 2
    assert SA.SAVED_ATTENTION_STATS["stored"] == 6 and SA.SAVED_ATTENTION_STATS["reused"] == 6
    assert SA.SAVED_ATTENTION_STATS["too_big"] == 0


def test_function_level_refuses_a_graph_that_keeps_expanded_inputs(monkeypatch):
    """A function that copies its inputs before the kernel would keep those copies: recompute it."""
    params, x = _make(n_layers = 1)

    def attention(q, k, v):
        return _sdpa(q.contiguous() * 1.0, (k * 1.0).repeat(1, 1, 1, 1), (v * 1.0).repeat(1, 1, 1, 1))

    call = lambda fn, q, k, v: SA.saved_function_call(fn, 3, q, k, v)  # noqa: E731
    monkeypatch.setattr(SA, "_FUNCTION_SLACK_BYTES", 0)
    full = _run(params, x, ckpt = True, attention = attention)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    saved = _run(params, x, ckpt = True, attention = attention, call = call)
    _close(saved, full)
    assert SA.SAVED_ATTENTION_STATS["too_big"] == 1
    assert SA.SAVED_ATTENTION_STATS["reused"] == 0


def test_outside_a_checkpoint_is_a_plain_call(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    q = torch.randn(1, 2, 8, 4, requires_grad = True)
    out = SA.saved_attention_call(_sdpa, q, q, q)
    out.sum().backward()
    assert q.grad is not None
    assert sum(SA.SAVED_ATTENTION_STATS.values()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA (flash / memory-efficient SDPA, pinned host memory)")
@pytest.mark.parametrize("setting", ["1", "offload"])
def test_cuda_kernels_match(monkeypatch, setting):
    """CUDA flash attention (bf16) and memory-efficient attention (fp32): GPU-kept and host-offloaded outputs."""
    monkeypatch.setattr(G, "MINIMUM_SIZE", 1 << 62)
    monkeypatch.setattr(SA, "_OFFLOAD_MIN_BYTES", 0)
    for dtype in (torch.bfloat16, torch.float32):
        params, x = _make()
        params = [[p.detach().to("cuda", dtype).requires_grad_() for p in w] for w in params]
        x = x.to("cuda", dtype)
        monkeypatch.delenv("UNSLOTH_GC_SAVE_ATTENTION", raising = False)
        full = _run(params, x, ckpt = True)
        SA.SAVED_ATTENTION_STATS.clear()
        monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", setting)
        saved = _run(params, x, ckpt = True, steps = 2)
        torch.testing.assert_close(saved[0], full[0])
        for a, b in zip(saved[1], full[1]):
            torch.testing.assert_close(a, b)
        assert SA.SAVED_ATTENTION_STATS["reused"] == 6, dict(SA.SAVED_ATTENTION_STATS)
        assert SA.SAVED_ATTENTION_STATS["mismatch"] == 0
        if setting == "offload":
            assert SA.SAVED_ATTENTION_STATS["offloaded_bytes"] > 0
            assert SA.SAVED_ATTENTION_STATS["prefetched"] >= 2


def test_function_level_with_q_k_v_split_from_one_buffer(monkeypatch):
    """Q / K / V as views of one fused projection: each saved view maps back to its own input."""
    params, x = _make(n_layers = 2)

    def fused(weights, attention, call):
        wq, wk, wv, wo = weights
        w = torch.cat([wq, wk, wv], 1)

        def forward(h):
            qkv = (h @ w).view(B, S, 3, H, D)
            q, k, v = qkv.unbind(2)
            a = call(attention, q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            return h + a.transpose(1, 2).reshape(B, S, H * D) @ wo
        return forward

    call = lambda fn, q, k, v: SA.saved_function_call(fn, 3, q, k, v)  # noqa: E731
    monkeypatch.setattr(sys.modules[__name__], "_layer", fused)
    full = _run(params, x, ckpt = True)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    saved = _run(params, x, ckpt = True, call = call)
    _close(saved, full)
    assert SA.SAVED_ATTENTION_STATS["reused"] == 2


def test_rocm_keeps_the_mode_off(monkeypatch):
    from unsloth_zoo import saved_attention as sa
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    monkeypatch.setattr(torch.version, "hip", "6.4", raising = False)
    assert sa.save_attention_enabled() is False
    assert sa.new_saved_attention_frame([torch.zeros(1, requires_grad = True)]) is None
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    assert sa.save_attention_enabled() is True


def _math_attention(q, k, v, keep):
    scores = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    causal = torch.ones(q.shape[-2], k.shape[-2], dtype = torch.bool, device = q.device).triu(1)
    p = scores.masked_fill(causal, float("-inf")).softmax(-1)
    return (p * keep) @ v


# Dispatcher-visible dropout attention; the "flash_attn" namespace marks it as an attention kernel.
@torch.library.custom_op("flash_attn_unsloth_test::dropout_attention_fwd", mutates_args = ())
def _dropout_attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float) -> tuple[torch.Tensor, torch.Tensor]:
    shape = q.shape[:-1] + (k.shape[-2],)
    keep = (torch.rand(shape, dtype = q.dtype, device = q.device) >= dropout_p).to(q.dtype) / (1 - dropout_p)
    return _math_attention(q, k, v, keep), keep


@_dropout_attention_fwd.register_fake
def _(q, k, v, dropout_p):
    return q.new_empty(q.shape), q.new_empty(q.shape[:-1] + (k.shape[-2],))


def _dropout_attention_setup(ctx, inputs, output):
    q, k, v, _ = inputs
    ctx.save_for_backward(q, k, v, output[1])


def _dropout_attention_backward(ctx, grad_out, grad_keep):
    q, k, v, keep = ctx.saved_tensors
    with torch.enable_grad():
        q, k, v = (t.detach().requires_grad_() for t in (q, k, v))
        out = _math_attention(q, k, v, keep)
    return (*torch.autograd.grad(out, (q, k, v), grad_out), None)


_dropout_attention_fwd.register_autograd(_dropout_attention_backward, setup_context = _dropout_attention_setup)


def _random_layer(attention_dropout):
    """Attention with dropout, then LoRA-style dropout on its output in the same checkpointed layer."""
    def make(weights, attention, call):
        wq, wk, wv, wo = weights

        def forward(h):
            q = (h @ wq).view(B, S, H, D).transpose(1, 2)
            k = (h @ wk).view(B, S, KV, D).transpose(1, 2)
            v = (h @ wv).view(B, S, KV, D).transpose(1, 2)
            a = call(attention_dropout, q, k, v)
            a = F.dropout(a.transpose(1, 2).reshape(B, S, H * D), p = 0.3, training = True)
            return h + a @ wo
        return forward
    return make


def _seeded_run(params, x, **kwargs):
    torch.manual_seed(1234)
    return _run(params, x, **kwargs)


@pytest.mark.parametrize("level", ["kernel", "function"])
def test_random_attention_is_recomputed_so_later_dropout_matches(monkeypatch, level):
    """A kept output would skip the kernel's random draws in the replay, shifting the LoRA dropout mask."""
    params, x = _make(n_layers = 2)
    if level == "kernel":
        attention = lambda q, k, v: _dropout_attention_fwd(q, k, v, 0.2)[0]  # noqa: E731
        call = None
    else:
        attention = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, dropout_p = 0.2, is_causal = True)  # noqa: E731
        call = lambda fn, q, k, v: SA.saved_function_call(fn, 3, q, k, v)  # noqa: E731
    monkeypatch.setattr(sys.modules[__name__], "_layer", _random_layer(attention))
    plain = _seeded_run(params, x, ckpt = False, call = call)
    full = _seeded_run(params, x, ckpt = True, call = call)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    saved = _seeded_run(params, x, ckpt = True, call = call)
    _close(full, plain)
    _close(saved, full)
    assert SA.SAVED_ATTENTION_STATS["random"] == 2
    assert SA.SAVED_ATTENTION_STATS["stored"] == 0 and SA.SAVED_ATTENTION_STATS["reused"] == 0


def _entry_storages(entry):
    """Storage bytes reachable from a function entry: its fields and its kept autograd graph."""
    from torch.utils._pytree import tree_flatten as flatten
    found = {}

    def add(t):
        if isinstance(t, torch.Tensor):
            storage = t.untyped_storage()
            found[storage.data_ptr()] = storage.nbytes()

    for name in type(entry).__slots__:
        for leaf in flatten(getattr(entry, name, None))[0]:
            add(leaf)
    stack = [o.grad_fn for o in entry.outputs if o.grad_fn is not None]
    seen = set()
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        add(getattr(node, "variable", None))
        for name in dir(node):
            if name.startswith("_saved_"):
                try:
                    add(getattr(node, name))
                except Exception:
                    pass  # a placeholder for Q / K / V: unpacks only during backward
        stack.extend(fn for fn, _ in node.next_functions)
    return found


def test_function_entry_holds_only_output_and_logsumexp(monkeypatch):
    from torch.multiprocessing.reductions import StorageWeakRef

    params, x = _make(n_layers = 2)
    entries = []
    originals = []

    class Recording(SA._FunctionEntry):
        __slots__ = ()

        def __init__(self):
            super().__init__()
            entries.append(self)

    def attention(q, k, v):
        if not entries or len(originals) < len(entries):
            originals.append(StorageWeakRef(q.untyped_storage()))
        return _sdpa(q, k, v)

    call = lambda fn, q, k, v: SA.saved_function_call(fn, 3, q, k, v)  # noqa: E731
    full = _run(params, x, ckpt = True, attention = attention)
    entries.clear(); originals.clear()
    monkeypatch.setattr(SA, "_FunctionEntry", Recording)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    for w in params:
        for p in w:
            p.grad = None
    h = x.clone().requires_grad_()
    for w in params:
        h = G.UnslothCheckpointFunction.apply(_layer(w, attention, call), True, h)
    assert len(entries) == 2 and SA.SAVED_ATTENTION_STATS["stored"] == 2
    output_bytes = B * H * S * D * 4
    logsumexp_bytes = B * H * S * 4
    for entry, original in zip(entries, originals):
        held = _entry_storages(entry)
        assert sum(held.values()) <= output_bytes + logsumexp_bytes + 1024, held
        # The first pass's Q storage (the q projection) is free once the layer's forward returns.
        assert original.expired()
    h.float().pow(2).mean().backward()
    grads = [p.grad.clone() for w in params for p in w]
    for a, b in zip(grads, full[1]):
        torch.testing.assert_close(a, b, rtol = 1e-4, atol = 1e-6)
    assert SA.SAVED_ATTENTION_STATS["reused"] == 2


def test_offload_mode_keeps_no_function_entries(monkeypatch):
    """Function entries are not offloaded: offload mode recomputes them instead of keeping them on the device."""
    params, x = _make()
    call = lambda fn, q, k, v: SA.saved_function_call(fn, 3, q, k, v)  # noqa: E731
    full = _run(params, x, ckpt = True)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "offload")
    saved = _run(params, x, ckpt = True, call = call)
    _close(saved, full)
    assert saved[2] == 6
    assert SA.SAVED_ATTENTION_STATS["stored"] == 0 and SA.SAVED_ATTENTION_STATS["reused"] == 0
    assert SA.SAVED_ATTENTION_STATS["not_offloaded"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA (flash / memory-efficient attention with dropout)")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cuda_dropout_attention_is_recomputed(monkeypatch, dtype):
    params, x = _make(n_layers = 2)
    params = [[p.detach().to("cuda", dtype).requires_grad_() for p in w] for w in params]
    x = x.to("cuda", dtype)
    attention = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, dropout_p = 0.2, is_causal = True)  # noqa: E731
    monkeypatch.setattr(sys.modules[__name__], "_layer", _random_layer(attention))
    full = _seeded_run(params, x, ckpt = True)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    saved = _seeded_run(params, x, ckpt = True)
    torch.testing.assert_close(saved[0], full[0])
    for a, b in zip(saved[1], full[1]):
        torch.testing.assert_close(a, b)
    assert SA.SAVED_ATTENTION_STATS["random"] == 2, dict(SA.SAVED_ATTENTION_STATS)
    assert SA.SAVED_ATTENTION_STATS["reused"] == 0


# Automatic mode: unset / =auto, engaged by a caller (Unsloth's attention dispatch) that routes calls.


def _fake_memory(monkeypatch, base_peak, capacity):
    """CPU stand-in for the allocator: a fixed run peak plus whatever is kept on top of it."""
    state = {"peak": base_peak}

    def peak(device):
        live = sum(b.live for b in SA._BUDGETS.values())
        state["peak"] = max(state["peak"], base_peak + live)
        return state["peak"]

    monkeypatch.setattr(SA, "_device_peak", peak)
    monkeypatch.setattr(SA, "_device_capacity", lambda device: capacity)
    monkeypatch.setattr(SA, "_AUTO_MARGIN_BYTES", 0)
    monkeypatch.setattr(SA, "_AUTO_MARGIN_FRACTION", 0.0)
    return state


@pytest.fixture
def auto(monkeypatch):
    monkeypatch.setattr(SA, "_AUTO_ACTIVE", [True])
    monkeypatch.setattr(SA, "_BUDGETS", {})
    yield


def _main(monkeypatch, params, x, steps):
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "0")
    out = _run(params, x, ckpt = True, steps = steps)
    monkeypatch.delenv("UNSLOTH_GC_SAVE_ATTENTION")
    assert sum(SA.SAVED_ATTENTION_STATS.values()) == 0 and not SA._BUDGETS
    return out


# One layer keeps the attention output (B, H, S, D) plus a float32 logsumexp (B, H, S).
_LAYER_BYTES = B * H * S * D * 4 + B * H * S * 4


def test_auto_keeps_nothing_until_a_step_was_measured_then_reuses(monkeypatch, auto):
    params, x = _make()
    full = _main(monkeypatch, params, x, 2)
    _fake_memory(monkeypatch, base_peak = 1 << 20, capacity = 1 << 30)
    saved = _run(params, x, ckpt = True, steps = 2)
    _close(saved, full)
    stats = SA.SAVED_ATTENTION_STATS
    # First step: no step of this size measured yet; second step: every layer kept and reused.
    assert stats["declined_unmeasured"] == 3, dict(stats)
    assert stats["stored"] == 3 and stats["reused"] == 3, dict(stats)
    assert saved[2] == 6 + 3
    assert stats["kept_bytes"] == 3 * _LAYER_BYTES
    assert all(b.live == 0 for b in SA._BUDGETS.values())


def test_auto_keeps_only_what_the_budget_allows(monkeypatch, auto):
    params, x = _make()
    full = _main(monkeypatch, params, x, 3)
    base_peak = 1 << 20
    # Room for two layers' outputs on top of the run's own peak, not three.
    _fake_memory(monkeypatch, base_peak, capacity = base_peak + 2 * _LAYER_BYTES + _LAYER_BYTES // 2)
    saved = _run(params, x, ckpt = True, steps = 3)
    _close(saved, full)
    stats = SA.SAVED_ATTENTION_STATS
    assert stats["declined_unmeasured"] == 3
    assert stats["stored"] == 4 and stats["reused"] == 4, dict(stats)
    assert stats["declined_budget"] == 2, dict(stats)


def test_auto_declines_everything_without_headroom(monkeypatch, auto):
    params, x = _make()
    _fake_memory(monkeypatch, base_peak = 1 << 20, capacity = (1 << 20) + _LAYER_BYTES // 2)
    saved = _run(params, x, ckpt = True, steps = 3)
    stats = SA.SAVED_ATTENTION_STATS
    assert stats["stored"] == 0 and stats["reused"] == 0
    assert stats["declined_budget"] == 6
    assert saved[2] == 2 * 3 * 3


def test_auto_peak_reading_charges_kept_bytes_exactly(monkeypatch, auto):
    """Kept-output peaks must not be learnt as the run's peak: the budget shrinks each step."""
    params, x = _make()
    _fake_memory(monkeypatch, base_peak = 1 << 20, capacity = 1 << 30)
    _run(params, x, ckpt = True, steps = 4)
    stats = SA.SAVED_ATTENTION_STATS
    assert stats["stored"] == 9 and stats["reused"] == 9
    (budget,) = SA._BUDGETS.values()
    assert budget.base_peak == 1 << 20


def test_auto_a_longer_call_than_measured_keeps_nothing(monkeypatch, auto):
    params, x = _make()
    _fake_memory(monkeypatch, base_peak = 1 << 20, capacity = 1 << 30)
    _run(params, x, ckpt = True, steps = 2)
    (budget,) = SA._BUDGETS.values()
    budget.measured = _LAYER_BYTES - 1
    SA.SAVED_ATTENTION_STATS.clear()
    _run(params, x, ckpt = True, steps = 1)
    assert SA.SAVED_ATTENTION_STATS["stored"] == 0
    assert SA.SAVED_ATTENTION_STATS["declined_unmeasured"] == 3


def test_auto_frames_need_a_routing_caller_and_respect_the_kill_switch(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising = False)  # ROCm keeps the mode off
    t = [torch.zeros(1, requires_grad = True)]
    monkeypatch.setattr(SA, "_AUTO_ACTIVE", [False])
    assert SA.new_saved_attention_frame(t) is None
    monkeypatch.setattr(SA, "_AUTO_ACTIVE", [True])
    frame = SA.new_saved_attention_frame(t)
    assert frame is not None and frame.auto
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "auto")
    assert SA.new_saved_attention_frame(t).auto
    for off in ("0", "false", "off"):
        monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", off)
        assert SA.new_saved_attention_frame(t) is None
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    assert SA.new_saved_attention_frame(t).auto is False


def test_auto_kill_switch_runs_like_main(monkeypatch, auto):
    params, x = _make()
    _fake_memory(monkeypatch, base_peak = 1 << 20, capacity = 1 << 30)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "0")
    off = _run(params, x, ckpt = True, steps = 2)
    assert off[2] == 12
    assert sum(SA.SAVED_ATTENTION_STATS.values()) == 0


def _fake_gpu(monkeypatch, capability, hip = None, cuda = True):
    monkeypatch.setattr(SA, "_AUTO_ACTIVE", [False])
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)


@pytest.mark.parametrize("capability, min_seq", [((7, 5), 8192), ((8, 0), 8192), ((8, 9), 16384)])
def test_auto_min_seq_on_measured_gpus(monkeypatch, capability, min_seq):
    _fake_gpu(monkeypatch, capability)
    assert SA.save_attention_auto_min_seq() == min_seq
    assert SA._AUTO_ACTIVE[0] is True
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION_MIN_SEQ", "256")
    assert SA.save_attention_auto_min_seq() == 256


@pytest.mark.parametrize(
    "capability, hip, cuda, setting",
    [
        ((8, 0), None, True, "0"),
        ((8, 0), None, True, "1"),
        ((8, 0), None, True, "offload"),
        ((8, 6), None, True, None),
        ((9, 0), None, True, None),
        ((8, 0), "6.4", True, None),
        ((8, 0), None, False, None),
        ((10, 0), None, True, None),
        ((12, 0), None, True, None),
    ],
)
def test_auto_min_seq_is_off(monkeypatch, capability, hip, cuda, setting):
    _fake_gpu(monkeypatch, capability, hip, cuda)
    if setting is not None:
        monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", setting)
    assert SA.save_attention_auto_min_seq() is None
    assert SA._AUTO_ACTIVE[0] is False
    # An explicit setting keeps its own (unbudgeted) mode.
    assert SA.save_attention_enabled() is (setting in ("1", "offload") and hip is None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA (the real allocator's peak and free memory)")
def test_cuda_auto_mode_with_the_real_allocator(monkeypatch, auto):
    params, x = _make()
    params = [[p.detach().to("cuda", torch.bfloat16).requires_grad_() for p in w] for w in params]
    x = x.to("cuda", torch.bfloat16)
    full = _main(monkeypatch, params, x, 3)
    peak = torch.cuda.max_memory_allocated()
    saved = _run(params, x, ckpt = True, steps = 3)
    torch.testing.assert_close(saved[0], full[0])
    for a, b in zip(saved[1], full[1]):
        torch.testing.assert_close(a, b)
    stats = SA.SAVED_ATTENTION_STATS
    assert stats["declined_unmeasured"] == 3 and stats["stored"] == 6 and stats["reused"] == 6, dict(stats)
    (budget,) = SA._BUDGETS.values()
    assert budget.live == 0 and budget.base_peak >= peak
    # The user's peak statistics are read, never reset.
    assert torch.cuda.max_memory_allocated() >= peak


def test_auto_measures_each_device_separately(monkeypatch, auto):
    # A replay with a long call on one device and a short one on another measures each device at its own size.
    _fake_memory(monkeypatch, base_peak = 0, capacity = 1 << 40)
    long_cpu = torch.empty(1, 2, 64, 4, device = "cpu")
    short_meta, long_meta = torch.empty(1, 2, 8, 4, device = "meta"), torch.empty(1, 2, 64, 4, device = "meta")
    frame = SA.SavedAttentionFrame(auto = True)
    assert not SA._auto_admit(frame, [long_cpu]) and not SA._auto_admit(frame, [short_meta])  # nothing measured yet
    SA._auto_release(frame, True)
    assert SA._auto_admit(SA.SavedAttentionFrame(auto = True), [long_cpu])
    assert SA._auto_admit(SA.SavedAttentionFrame(auto = True), [short_meta])
    assert not SA._auto_admit(SA.SavedAttentionFrame(auto = True), [long_meta])  # never measured at this size there
