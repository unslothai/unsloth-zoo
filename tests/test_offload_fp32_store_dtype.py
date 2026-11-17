# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""float32 activations offload exactly when no autocast is active (true float32 training: bf16 storage drifted grads
~0.1%) and as bfloat16 under autocast (FORCE_FLOAT32 residuals), with UNSLOTH_OFFLOAD_FP32=exact|bf16 overrides.
Streams are stubbed so the end-to-end checks run on CPU."""
import contextlib

import pytest

torch = pytest.importorskip("torch")

from unsloth_zoo import gradient_checkpointing as G

_SAVED = (
    "CPU_BUFFERS", "CPU_INDEX", "GPU_BUFFERS", "GPU_BUFFERS_B", "USE_DOUBLE_BUFFER", "EXTRA_STREAMS", "MAIN_STREAMS",
    "NEXT_BUFFER_SLOT", "MINIMUM_SIZE", "FP32_OFFLOAD_EXACT", "FP32_OFFLOAD_MODE", "BACKWARD_PASS", "FIRST_PASS",
    "LAST_GC_INDEX", "CURRENT_GC_INDEX", "USE_UNSLOTH_GC", "BUFFER_EVENTS_A", "BUFFER_EVENTS_B", "GPU_BUFFER_DTYPE",
)


class _NoStream:
    def wait_stream(self, other): pass
    def wait_event(self, event): pass
    def record_event(self): return None


@pytest.fixture
def mode(monkeypatch):
    saved = {n: getattr(G, n, None) for n in ("FP32_OFFLOAD_EXACT", "FP32_OFFLOAD_MODE")}

    def set_mode(value):
        if value is None: monkeypatch.delenv("UNSLOTH_OFFLOAD_FP32", raising = False)
        else: monkeypatch.setenv("UNSLOTH_OFFLOAD_FP32", value)
        G.FP32_OFFLOAD_MODE = G._fp32_offload_mode()
        G.FP32_OFFLOAD_EXACT = G.FP32_OFFLOAD_MODE == "exact"
    yield set_mode
    for n, v in saved.items(): setattr(G, n, v)


def _x(dtype):
    return torch.zeros(4, dtype = dtype)


@pytest.mark.parametrize("env", [None, "auto", "", "0"])
def test_default_stores_fp32_exactly_without_autocast(mode, env):
    mode(env)
    assert G._offload_store_dtype(_x(torch.float32)) == torch.float32


@pytest.mark.parametrize("autocast_dtype", [torch.bfloat16, torch.float16])
def test_default_stores_fp32_as_bf16_under_autocast(mode, autocast_dtype):
    mode(None)
    with torch.autocast("cpu", dtype = autocast_dtype):
        assert G._offload_store_dtype(_x(torch.float32)) == torch.bfloat16
    assert G._offload_store_dtype(_x(torch.float32)) == torch.float32


def test_autocast_on_another_device_does_not_count(mode):
    mode(None)
    with torch.autocast("cpu", dtype = torch.bfloat16, enabled = False):
        assert G._offload_store_dtype(_x(torch.float32)) == torch.float32
    assert not G._autocast_enabled("cuda")


@pytest.mark.parametrize("env", [None, "exact", "bf16"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("autocast", [False, True])
def test_half_activations_keep_their_dtype(mode, env, dtype, autocast):
    mode(env)
    ctx = torch.autocast("cpu", dtype = torch.bfloat16) if autocast else contextlib.nullcontext()
    with ctx:
        assert G._offload_store_dtype(_x(dtype)) == dtype


@pytest.mark.parametrize("env", ["exact", "EXACT", "fp32", "float32"])
@pytest.mark.parametrize("autocast", [False, True])
def test_exact_override_always_keeps_fp32(mode, env, autocast):
    mode(env)
    ctx = torch.autocast("cpu", dtype = torch.bfloat16) if autocast else contextlib.nullcontext()
    with ctx:
        assert G._offload_store_dtype(_x(torch.float32)) == torch.float32


@pytest.mark.parametrize("env", ["bf16", "bfloat16", " BF16 "])
@pytest.mark.parametrize("autocast", [False, True])
def test_bf16_override_always_stores_bf16(mode, env, autocast):
    mode(env)
    ctx = torch.autocast("cpu", dtype = torch.bfloat16) if autocast else contextlib.nullcontext()
    with ctx:
        assert G._offload_store_dtype(_x(torch.float32)) == torch.bfloat16


# End to end through UnslothCheckpointFunction on CPU.
@pytest.fixture
def offload(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ALLOW_CPU", "1")
    monkeypatch.setenv("UNSLOTH_DISABLE_PINNED_MEMORY", "1")
    monkeypatch.setenv("UNSLOTH_DISABLE_DOUBLE_BUFFER", "1")
    G._double_buffer_disabled.cache_clear()
    saved = {n: getattr(G, n, None) for n in _SAVED}

    def setup(env):
        if env is None: monkeypatch.delenv("UNSLOTH_OFFLOAD_FP32", raising = False)
        else: monkeypatch.setenv("UNSLOTH_OFFLOAD_FP32", env)
        G.initialize_unsloth_gradient_checkpointing(torch.float32)
        stream = _NoStream()
        monkeypatch.setattr(G, "_ensure_device_buffers", lambda i: None)
        monkeypatch.setattr(G, "_compute_stream", lambda i: stream)
        monkeypatch.setattr(G, "torch_gpu_stream", lambda s: contextlib.nullcontext(), raising = False)
        G.GPU_BUFFERS = {None: torch.empty(G.INITIAL_GPU_BUFFER_SIZE)}
        G.GPU_BUFFERS_B = None
        G.USE_DOUBLE_BUFFER = False
        G.EXTRA_STREAMS = {None: stream}
        G.NEXT_BUFFER_SLOT = {None: 0}
        G.MINIMUM_SIZE = 0
        return G
    yield setup
    for n, v in saved.items(): setattr(G, n, v)
    G._reset_cpu_slots()
    G._double_buffer_disabled.cache_clear()


H, S, L = 32, 16, 4


def _run(G, ckpt, autocast):
    g = torch.Generator().manual_seed(0)
    ws = [(torch.randn(H, H, generator = g) / H ** 0.5).requires_grad_() for _ in range(L)]
    x = (torch.randn(1, S, H, generator = g) * 3).requires_grad_()
    seen = []

    def layer(w):
        def f(h):
            seen.append(h.detach().clone())
            return h + torch.tanh(h @ w).float()
        return f

    ctx = torch.autocast("cpu", dtype = torch.bfloat16) if autocast else contextlib.nullcontext()
    with ctx:
        h = x * 1.0
        for w in ws:
            h = G.UnslothCheckpointFunction.apply(layer(w), True, h) if ckpt else layer(w)(h)
        loss = h.square().mean()
    loss.backward()
    return [w.grad for w in ws], seen


def _settle(G, autocast):
    _run(G, True, autocast)  # FIRST_PASS / LAST_GC_INDEX


@pytest.mark.parametrize("env", [None, "exact"])
def test_fp32_training_offloads_exactly(offload, env):
    G = offload(env)
    _settle(G, False)
    ref, _ = _run(G, False, False)
    got, seen = _run(G, True, False)
    assert G.CPU_INDEX >= 1, "nothing was offloaded, so this test proved nothing"
    assert all(h.dtype == torch.float32 for h in seen)
    for a, b in zip(ref, got):
        assert torch.equal(a, b), "float32 training without autocast must match no checkpointing bit for bit"
    # Recomputes (second half of seen) get the forward inputs back unchanged.
    fwd, rec = seen[:L], seen[L:][::-1]
    for i in range(L - 1):  # the last layer is not offloaded after the first pass
        assert torch.equal(fwd[i], rec[i]), i


@pytest.mark.parametrize("env", [None, "bf16"])
def test_fp32_under_autocast_still_stores_bf16(offload, env):
    G = offload(env)
    _settle(G, True)
    _, seen = _run(G, True, True)
    assert G.CPU_INDEX >= 1, "nothing was offloaded, so this test proved nothing"
    fwd, rec = seen[:L], seen[L:][::-1]
    assert all(h.dtype == torch.float32 for h in seen), "the recompute must get float32 back"
    rounded = 0
    for i in range(1, L - 1):  # layer 0's input is x * 1.0 (requires grad, offloaded); check the residual stream
        assert torch.equal(rec[i], fwd[i].to(torch.bfloat16).float()), i
        rounded += int(not torch.equal(rec[i], fwd[i]))
    assert rounded, "bf16 storage left every input unchanged, so the test cannot tell the dtypes apart"


def test_bf16_override_without_autocast_stores_bf16(offload):
    G = offload("bf16")
    _settle(G, False)
    _, seen = _run(G, True, False)
    fwd, rec = seen[:L], seen[L:][::-1]
    for i in range(L - 1):
        assert torch.equal(rec[i], fwd[i].to(torch.bfloat16).float()), i
    assert any(not torch.equal(rec[i], fwd[i]) for i in range(L - 1))


@pytest.mark.parametrize("env, expected", [(None, "auto"), ("bf16", "bf16"), ("exact", "exact")])
def test_initialize_rereads_the_env(offload, env, expected):
    G = offload(env)
    assert G.FP32_OFFLOAD_MODE == expected
    assert G.FP32_OFFLOAD_EXACT == (expected == "exact")
