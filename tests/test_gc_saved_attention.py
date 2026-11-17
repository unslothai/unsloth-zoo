# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""UNSLOTH_GC_SAVE_ATTENTION=1: the checkpoint replay reuses the first pass's attention kernel outputs.

CPU only: SDPA picks its CPU flash kernel (_scaled_dot_product_flash_attention_for_cpu), which returns
the logsumexp the backward needs, exactly like the CUDA flash / memory-efficient kernels."""
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
            # Replay: a different (but equivalent) call shape.
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
