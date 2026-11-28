# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""UNSLOTH_GC_SAVE_ATTENTION=1 reuses the first pass's attention outputs in checkpoint replay; CPU
only, where SDPA's CPU flash kernel returns the logsumexp the backward needs."""
import sys
import types

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
    monkeypatch.setattr(SA, "_WARNED_HIP", set() if isinstance(SA._WARNED_HIP, set) else [False])
    monkeypatch.setattr(SA, "_ROCM_SDPA_SELECT", getattr(SA, "_default_rocm_sdpa_selector", None), raising = False)
    # Automatic mode off unless a test turns it on: a GPU with a measured length (gfx1151, sm75, sm80,
    # sm89) leaves it active once anything asked for the threshold.
    monkeypatch.setattr(SA, "_AUTO_ACTIVE", [False])
    yield
    if hasattr(SA, "release_saved_attention_host_pool"):
        SA.release_saved_attention_host_pool()
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


def _simulated_rocm(monkeypatch, verdicts = None, current = 0):
    """A ROCm build with GPUs 0 and 1: SDPA backend selection runs for real on CPU tensors, ``verdicts``
    (device index -> SDPBackend int, default: the CPU selector's own answer) stands in per GPU."""
    from contextlib import nullcontext

    seen = []

    def redirect(allocate):
        def redirected(*args, **kwargs):
            seen.append(("alloc", kwargs.get("device")))
            return allocate(*args, **{**kwargs, "device": "cpu"})
        return redirected

    def select(q):
        seen.append(("select", q.dtype, q.requires_grad, torch.is_grad_enabled(), torch.is_inference(q)))
        choice = SA._default_rocm_sdpa_selector(q)
        index = seen[-2][1].index if seen[-2][0] == "alloc" else None
        return (verdicts or {}).get(index, choice)

    monkeypatch.setattr(torch.version, "hip", "7.13.0", raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(SA, "_rocm_probe_dtype", lambda device: torch.bfloat16, raising = False)
    # The PR's first probe allocated with torch.zeros, this one with torch.empty.
    monkeypatch.setattr(torch, "empty", redirect(torch.empty))
    monkeypatch.setattr(torch, "zeros", redirect(torch.zeros))
    register = getattr(SA, "register_rocm_saved_attention_selector", None)
    if register is not None:
        register(select)
    if hasattr(SA, "_ROCM_SDPA_KEPT"):  # the PR's first probe: a per-ordinal cache stands in for a GPU's verdict
        monkeypatch.setattr(SA, "_ROCM_SDPA_KEPT", {i: v != _MATH for i, v in (verdicts or {}).items()})
    return seen


_FLASH, _MATH = 1, 0  # SDPBackend.FLASH_ATTENTION, SDPBackend.MATH


def test_rocm_keeps_the_mode_off_where_sdpa_is_not_intercepted(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    monkeypatch.setattr(torch.version, "hip", "6.4", raising = False)
    monkeypatch.setattr(SA, "_rocm_sdpa_is_kept", lambda device = None: False)
    assert SA.save_attention_enabled() is False
    assert SA.new_saved_attention_frame([torch.zeros(1, requires_grad = True)]) is None
    monkeypatch.setattr(SA, "_rocm_sdpa_is_kept", lambda device = None: True)
    assert SA.save_attention_enabled() is True
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(SA, "_rocm_sdpa_is_kept", lambda device = None: False)
    assert SA.save_attention_enabled() is True


def test_rocm_eligibility_reads_the_operand_device(monkeypatch):
    # cuda:1 is asked about cuda:1, not the current GPU (0), which keeps nothing.
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    seen = _simulated_rocm(monkeypatch, verdicts = {0: _MATH, 1: _FLASH}, current = 0)
    assert SA.save_attention_enabled(torch.device("cuda", 1)) is True
    assert ("alloc", torch.device("cuda", 1)) in seen
    assert SA.save_attention_enabled(torch.device("cuda", 0)) is False
    assert SA.save_attention_enabled() is False  # no operand: the current GPU
    # A device without an index resolves to the current GPU.
    assert SA.save_attention_enabled(torch.device("cuda")) is False


def test_rocm_cpu_checkpoint_keeps_attention_when_the_gpu_does_not(monkeypatch):
    # A rejected visible GPU must not turn the mode off for CPU tensors (the CPU kernel is intercepted).
    _simulated_rocm(monkeypatch, verdicts = {0: _MATH})
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    assert SA.save_attention_enabled() is False
    assert SA.new_saved_attention_frame([torch.zeros(1, requires_grad = True)]) is not None
    params, x = _make()
    full = _run(params, x, ckpt = True)
    assert full[2] == 3, "the CPU checkpoint did not reuse its attention outputs on a ROCm build"
    assert SA.SAVED_ATTENTION_STATS["stored"] == 3 and SA.SAVED_ATTENTION_STATS["reused"] == 3
    assert SA.save_attention_enabled(torch.device("cpu")) is True


def test_rocm_frame_asks_the_checkpoint_operand_device(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    monkeypatch.setattr(torch.version, "hip", "7.13.0", raising = False)
    asked = []
    monkeypatch.setattr(SA, "_rocm_sdpa_is_kept", lambda device = None: asked.append(device) or True)
    t = torch.zeros(2, requires_grad = True)
    assert SA.new_saved_attention_frame([torch.zeros(1), t, t * 2]) is not None
    assert asked == [t.device]


def test_rocm_probe_never_launches_sdpa_and_uses_the_registered_selector(monkeypatch):
    # The probe selects a backend and runs no kernel (kernels are validated by Unsloth's isolated
    # rocm_sdpa_check); Unsloth registers a selector applying that check's filter.
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    _simulated_rocm(monkeypatch)
    ops = []

    class _Ops(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args = (), kwargs = None):
            ops.append(str(func))
            return func(*args, **(kwargs or {}))

    with _Ops():
        assert SA.save_attention_enabled() is True
    assert not [op for op in ops if "attention" in op or "softmax" in op or "bmm" in op], ops
    picked = []
    SA.register_rocm_saved_attention_selector(lambda q: picked.append(q.requires_grad) or -1)  # SDPBackend.ERROR
    assert SA.save_attention_enabled() is False
    assert picked == [True]
    SA.register_rocm_saved_attention_selector(None)
    assert SA._ROCM_SDPA_SELECT is SA._default_rocm_sdpa_selector


def test_rocm_probe_follows_the_live_backend_policy(monkeypatch):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    seen = _simulated_rocm(monkeypatch)
    with sdpa_kernel(SDPBackend.MATH):
        assert SA.save_attention_enabled() is False
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        assert SA.save_attention_enabled() is True
    with sdpa_kernel(SDPBackend.MATH):
        assert SA.save_attention_enabled() is False
    # Inference mode outside: the sample is still a grad-requiring training tensor.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.inference_mode():
        assert SA.save_attention_enabled() is True
    assert seen[-1] == ("select", torch.bfloat16, True, True, False)


def test_rocm_failed_probe_is_inconclusive_and_not_remembered(monkeypatch, capsys):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    _simulated_rocm(monkeypatch)
    for name in ("empty", "zeros"):
        allocate = getattr(torch, name)

        def short_once(*args, _allocate = allocate, **kwargs):
            if kwargs.get("device") is not None and not short_once.failed:
                short_once.failed = True
                raise torch.OutOfMemoryError("HIP out of memory. Tried to allocate 4.00 KiB")
            return _allocate(*args, **kwargs)

        short_once.failed = False
        monkeypatch.setattr(torch, name, short_once)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        assert SA.save_attention_enabled() is False  # this frame keeps nothing
        assert "keeps nothing" not in capsys.readouterr().out  # not reported as unsupported hardware
        assert SA.save_attention_enabled() is True  # asked again: no permanent verdict


def test_rocm_traced_lookup_is_deferred_without_a_warning(monkeypatch, capsys):
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    _simulated_rocm(monkeypatch, verdicts = {0: _MATH, 1: _MATH})
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert SA._rocm_sdpa_is_kept() is None
    assert SA.save_attention_enabled() is False
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    # One warning per device, not one per process.
    SA.save_attention_enabled(torch.device("cuda", 0))
    SA.save_attention_enabled(torch.device("cuda", 0))
    SA.save_attention_enabled(torch.device("cuda", 1))
    assert capsys.readouterr().out.count("keeps nothing") == 2


@pytest.mark.parametrize("arch, bf16, dtype", [
    ("gfx1151", True, torch.bfloat16), ("gfx1030:sramecc-", True, torch.float16), ("gfx906", False, torch.float16),
])
def test_rocm_probe_dtype(monkeypatch, arch, bf16, dtype):
    from contextlib import nullcontext
    from types import SimpleNamespace

    monkeypatch.setattr(SA, "_ROCM_PROBE_DTYPE", {})
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(gcnArchName = arch))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: bf16)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    assert SA._rocm_probe_dtype(torch.device("cuda", 0)) is dtype


@pytest.fixture(params = ["7.13.0", None], ids = ["rocm", "not_rocm"])
def hip_build(request, monkeypatch):
    """The transformers wrappers route the same on every build: run their tests as ROCm and as CUDA / CPU."""
    monkeypatch.setattr(torch.version, "hip", request.param, raising = False)
    return request.param


@pytest.mark.parametrize("kind", ["interface", "flex"])
def test_transformers_wrappers_stay_in_a_fullgraph_compile_without_a_frame(monkeypatch, hip_build, kind):
    module = torch.nn.Module()
    module.train()
    if kind == "interface":
        wrapped = SA._wrap_attention_interface(lambda m, q, k, v, **kw: _sdpa(q, k, v))
        call = lambda q: wrapped(module, q, q, q)  # noqa: E731
    else:
        wrapped = SA._wrap_flex_kernel(lambda q, k, v, **kw: _sdpa(q, k, v))
        call = lambda q: wrapped(q, q, q, training = True)  # noqa: E731
    q = torch.randn(1, 2, 8, 16, requires_grad = True)
    torch._dynamo.reset()
    try:
        out = torch.compile(call, backend = "eager", fullgraph = True)(q)
    finally:
        torch._dynamo.reset()
    torch.testing.assert_close(out, _sdpa(q, q, q))


def _frame_wrapper(monkeypatch, kind):
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    module = torch.nn.Module()
    module.train()
    if kind == "interface":
        wrapped = SA._wrap_attention_interface(lambda m, q, k, v, **kw: _sdpa(q, k, v))
        return lambda q: wrapped(module, q, q, q)  # noqa: E731
    wrapped = SA._wrap_flex_kernel(lambda q, k, v, **kw: _sdpa(q, k, v))
    return lambda q: wrapped(q, q, q, training = True)  # noqa: E731


@pytest.mark.parametrize("kind", ["interface", "flex"])
def test_transformers_wrappers_stay_in_a_fullgraph_compile_inside_a_frame(monkeypatch, hip_build, kind):
    """A live checkpoint frame on an eligible device (CPU tensors are kept) must not send a user's
    fullgraph=True training compile into the compiler-disabled helpers: forward + backward compile, and
    both checkpoint passes take the in-graph route, so the frame keeps nothing."""
    call = _frame_wrapper(monkeypatch, kind)
    q = torch.randn(1, 2, 8, 16, requires_grad = True)
    frame = SA.new_saved_attention_frame([q])
    assert frame is not None
    ref_q = q.detach().requires_grad_()
    _sdpa(ref_q, ref_q, ref_q).sum().backward()
    torch._dynamo.reset()
    try:
        compiled = torch.compile(call, backend = "eager", fullgraph = True)
        with SA.saved_attention_scope(frame, False), torch.no_grad():
            first = compiled(q)
        with SA.saved_attention_scope(frame, True):
            out = compiled(q)
            out.sum().backward()
    finally:
        torch._dynamo.reset()
    torch.testing.assert_close(first, _sdpa(q, q, q).detach())
    torch.testing.assert_close(out, _sdpa(q, q, q))
    torch.testing.assert_close(q.grad, ref_q.grad)
    assert frame.entries == [] and SA.SAVED_ATTENTION_STATS.get("stored", 0) == 0
    assert SA.SAVED_ATTENTION_STATS.get("unmatched", 0) == 0


def test_frame_still_keeps_inside_an_unsloth_compiled_block(monkeypatch, hip_build):
    """Unsloth's own block compile (fullgraph=False) may break the graph: there the frame keeps and the
    replay reuses as in eager code."""
    import types

    call = _frame_wrapper(monkeypatch, "interface")
    block = types.ModuleType("unsloth.models._block_compile")
    block._IN_COMPILED_REGION = [True]
    monkeypatch.setitem(sys.modules, "unsloth.models._block_compile", block)
    q = torch.randn(1, 2, 8, 16, requires_grad = True)
    frame = SA.new_saved_attention_frame([q])
    torch._dynamo.reset()
    try:
        compiled = torch.compile(call, backend = "eager", fullgraph = False)
        with SA.saved_attention_scope(frame, False), torch.no_grad():
            compiled(q)
        with SA.saved_attention_scope(frame, True):
            out = compiled(q)
    finally:
        torch._dynamo.reset()
    torch.testing.assert_close(out, _sdpa(q, q, q))
    assert SA.SAVED_ATTENTION_STATS["stored"] >= 1


def _frame_qkv_wrapper(monkeypatch, kind):
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    module = torch.nn.Module()
    module.train()
    if kind == "interface":
        wrapped = SA._wrap_attention_interface(lambda m, q, k, v, **kw: _sdpa(q, k, v))
        return lambda q, k, v: wrapped(module, q, k, v)  # noqa: E731
    wrapped = SA._wrap_flex_kernel(lambda q, k, v, **kw: _sdpa(q, k, v))
    return lambda q, k, v: wrapped(q, k, v, training = True)  # noqa: E731


def _two_attention_frame(call, first_call, inputs):
    """One checkpoint frame holding two attention calls with equal shapes; the first goes through
    `first_call(replaying)`. Returns the loss and the Q / K / V gradients."""
    q, k, v = (x.clone().requires_grad_() for x in inputs)
    frame = SA.new_saved_attention_frame([q, k, v])
    assert frame is not None

    def layer(replaying):
        first = first_call(replaying)(q, k, v)
        second = call(q + 0.7, k * 0.4, v + 3.0)
        return (first + 2 * second).square().mean()

    with SA.saved_attention_scope(frame, False), torch.no_grad():
        layer(False)
    with SA.saved_attention_scope(frame, True):
        loss = layer(True)
        loss.backward()
    return loss.detach(), [x.grad for x in (q, k, v)]


def _two_attention_reference(inputs):
    q, k, v = (x.clone().requires_grad_() for x in inputs)
    loss = (_sdpa(q, k, v) + 2 * _sdpa(q + 0.7, k * 0.4, v + 3.0)).square().mean()
    loss.backward()
    return loss.detach(), [x.grad for x in (q, k, v)]


@pytest.mark.parametrize("kind", ["interface", "flex"])
def test_frame_replays_the_right_call_when_a_compile_falls_back_to_eager(monkeypatch, hip_build, kind):
    """A user's fullgraph=False compile at its recompile limit runs the replay eagerly while the first
    pass was compiled. Both passes must take the same route, or the replay of one call takes the entry
    another call kept (entries are matched by call order) and its gradients are wrong."""
    call = _frame_qkv_wrapper(monkeypatch, kind)
    torch.manual_seed(0)
    inputs = tuple(torch.randn(1, 2, 32, 16) for _ in range(3))
    ref_loss, ref_grads = _two_attention_reference(inputs)
    torch._dynamo.reset()
    try:
        compiled = torch.compile(call, backend = "eager", fullgraph = False)
        # The first pass (no_grad) compiles; the replay needs a grad-mode graph and runs eagerly.
        with torch._dynamo.config.patch(recompile_limit = 1):
            loss, grads = _two_attention_frame(call, lambda replaying: compiled, inputs)
    finally:
        torch._dynamo.reset()
    torch.testing.assert_close(loss, ref_loss)
    for got, want in zip(grads, ref_grads):
        torch.testing.assert_close(got, want, rtol = 1e-4, atol = 1e-6)


@pytest.mark.parametrize("kind", ["interface", "flex"])
def test_frame_keeps_inside_a_fullgraph_false_user_compile(monkeypatch, hip_build, kind):
    """A user's fullgraph=False compile may break the graph: inside a frame it keeps and the replay
    reuses, as in eager code and in Unsloth's own blocks."""
    call = _frame_qkv_wrapper(monkeypatch, kind)
    torch.manual_seed(0)
    inputs = tuple(torch.randn(1, 2, 32, 16) for _ in range(3))
    ref_loss, ref_grads = _two_attention_reference(inputs)
    torch._dynamo.reset()
    try:
        compiled = torch.compile(call, backend = "eager", fullgraph = False)
        loss, grads = _two_attention_frame(call, lambda replaying: compiled, inputs)
    finally:
        torch._dynamo.reset()
    torch.testing.assert_close(loss, ref_loss)
    for got, want in zip(grads, ref_grads):
        torch.testing.assert_close(got, want, rtol = 1e-4, atol = 1e-6)
    assert SA.SAVED_ATTENTION_STATS.get("stored", 0) == 2, dict(SA.SAVED_ATTENTION_STATS)
    assert SA.SAVED_ATTENTION_STATS.get("reused", 0) == 2, dict(SA.SAVED_ATTENTION_STATS)


def test_frame_stays_in_the_graph_under_error_on_graph_break(monkeypatch, hip_build):
    """torch._dynamo.error_on_graph_break(True) inside a fullgraph=False compile is as strict as fullgraph."""
    if not hasattr(torch._dynamo, "error_on_graph_break"):
        pytest.skip("torch._dynamo.error_on_graph_break is torch 2.9+")
    call = _frame_wrapper(monkeypatch, "interface")
    q = torch.randn(1, 2, 8, 16, requires_grad = True)
    frame = SA.new_saved_attention_frame([q])

    def strict(q):
        with torch._dynamo.error_on_graph_break(True):
            return call(q)

    torch._dynamo.reset()
    try:
        compiled = torch.compile(strict, backend = "eager", fullgraph = False)
        with SA.saved_attention_scope(frame, False), torch.no_grad():
            compiled(q)
        with SA.saved_attention_scope(frame, True):
            out = compiled(q)
    finally:
        torch._dynamo.reset()
    torch.testing.assert_close(out, _sdpa(q, q, q))
    assert frame.entries == [] and SA.SAVED_ATTENTION_STATS.get("stored", 0) == 0


@pytest.mark.parametrize("kind", ["interface", "flex"])
@pytest.mark.parametrize("backend", ["eager", "aot_eager", "inductor"])
def test_strict_frame_replays_the_right_call_when_a_compile_falls_back_to_eager(monkeypatch, hip_build, kind, backend):
    """error_on_graph_break(True) in a fullgraph=False compile: the first pass (no_grad) compiles and
    keeps nothing for that call, the replay hits the recompile limit and runs eagerly. The eager replay
    of the strict call must not take the entry the next (eager) call kept."""
    if not hasattr(torch._dynamo, "error_on_graph_break"):
        pytest.skip("torch._dynamo.error_on_graph_break is torch 2.9+")
    call = _frame_qkv_wrapper(monkeypatch, kind)
    torch.manual_seed(0)
    inputs = tuple(torch.randn(1, 2, 32, 16) for _ in range(3))
    ref_loss, ref_grads = _two_attention_reference(inputs)

    def strict(q, k, v):
        with torch._dynamo.error_on_graph_break(True):
            return call(q, k, v)

    torch._dynamo.reset()
    try:
        compiled = torch.compile(strict, backend = backend, fullgraph = False)
        with torch._dynamo.config.patch(recompile_limit = 1):
            loss, grads = _two_attention_frame(call, lambda replaying: compiled, inputs)
    finally:
        torch._dynamo.reset()
    torch.testing.assert_close(loss, ref_loss)
    for got, want in zip(grads, ref_grads):
        torch.testing.assert_close(got, want, rtol = 1e-4, atol = 1e-6)
    assert SA.SAVED_ATTENTION_STATS.get("unmatched", 0) == 0, dict(SA.SAVED_ATTENTION_STATS)


@pytest.mark.parametrize("kind", ["interface", "flex"])
def test_strict_replay_after_an_eager_first_pass_keeps_call_order(monkeypatch, hip_build, kind):
    """The reverse: the first pass runs the strict call eagerly (it keeps an entry), the replay runs it
    compiled in the graph. The compiled replay must consume that entry so the next call takes its own."""
    if not hasattr(torch._dynamo, "error_on_graph_break"):
        pytest.skip("torch._dynamo.error_on_graph_break is torch 2.9+")
    call = _frame_qkv_wrapper(monkeypatch, kind)
    torch.manual_seed(0)
    inputs = tuple(torch.randn(1, 2, 32, 16) for _ in range(3))
    ref_loss, ref_grads = _two_attention_reference(inputs)

    def strict(q, k, v):
        with torch._dynamo.error_on_graph_break(True):
            return call(q, k, v)

    torch._dynamo.reset()
    try:
        compiled = torch.compile(strict, backend = "eager", fullgraph = False)
        loss, grads = _two_attention_frame(call, lambda replaying: compiled if replaying else strict, inputs)
    finally:
        torch._dynamo.reset()
    torch.testing.assert_close(loss, ref_loss)
    for got, want in zip(grads, ref_grads):
        torch.testing.assert_close(got, want, rtol = 1e-4, atol = 1e-6)


@pytest.mark.parametrize("kind", ["interface", "flex"])
def test_wrappers_in_eager_code_are_bitwise_the_plain_call(monkeypatch, hip_build, kind):
    """Eager calls, with or without a frame, return exactly what the unwrapped attention returns."""
    call = _frame_wrapper(monkeypatch, kind)
    q = torch.randn(1, 2, 8, 16, requires_grad = True)
    ref_q = q.detach().requires_grad_()
    ref = _sdpa(ref_q, ref_q, ref_q)
    ref.sum().backward()
    assert torch.equal(call(q), ref)
    frame = SA.new_saved_attention_frame([q])
    with SA.saved_attention_scope(frame, False), torch.no_grad():
        first = call(q)
    with SA.saved_attention_scope(frame, True):
        out = call(q)
        out.sum().backward()
    assert torch.equal(first, ref.detach()) and torch.equal(out, ref)
    assert torch.equal(q.grad, ref_q.grad)
    assert SA.SAVED_ATTENTION_STATS["stored"] >= 1 and SA.SAVED_ATTENTION_STATS["reused"] >= 1


def test_rocm_hooks_install_from_the_setting_not_the_current_gpu(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    _simulated_rocm(monkeypatch, verdicts = {0: _MATH})
    installed = []
    monkeypatch.setattr(G, "patch_transformers_saved_attention", lambda: installed.append(1))
    G.patch_unsloth_smart_gradient_checkpointing(torch.float32)
    assert installed == [1]


@pytest.mark.skipif(
    not torch.cuda.is_available() or not getattr(torch.version, "hip", None), reason = "needs a ROCm GPU",
)
def test_rocm_probe_matches_a_real_training_call(monkeypatch):
    class _Kept(TorchDispatchMode):
        found = False

        def __torch_dispatch__(self, func, types, args = (), kwargs = None):
            self.found |= SA._is_attention_forward(func)
            return func(*args, **(kwargs or {}))

    device = torch.device("cuda", torch.cuda.current_device())
    dtype = SA._rocm_probe_dtype(device)
    probed = SA._rocm_sdpa_is_kept(device)
    q = torch.randn(1, 8, 128, 64, device = device, dtype = dtype, requires_grad = True)
    mode = _Kept()
    with torch.enable_grad(), mode:
        F.scaled_dot_product_attention(q, q, q, is_causal = True)
    assert probed is mode.found


@pytest.mark.skipif(
    not torch.cuda.is_available() or not getattr(torch.version, "hip", None), reason = "needs a ROCm GPU",
)
def test_rocm_gfx1151_checkpoint_reuses_attention(monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
    if arch != "gfx1151":
        pytest.skip("positive replay case measured on gfx1151")
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    dtype = SA._rocm_probe_dtype(device)
    params, x = _make(n_layers = 1)
    params = [[w.detach().to(device = device, dtype = dtype).requires_grad_() for w in layer] for layer in params]
    _run(params, x.to(device = device, dtype = dtype), ckpt = True)
    assert SA.SAVED_ATTENTION_STATS["stored"] == 1
    assert SA.SAVED_ATTENTION_STATS["reused"] == 1


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
    # Automatic routing, which runs on measured non-ROCm GPUs only.
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
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


def _fake_gpu(monkeypatch, capability, hip = None, cuda = True, arch = "gfx1100"):
    monkeypatch.setattr(SA, "_AUTO_ACTIVE", [False])
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda *a: types.SimpleNamespace(gcnArchName = arch)
    )


@pytest.mark.parametrize(
    "capability, min_seq", [((7, 5), 8192), ((8, 0), 8192), ((8, 9), 16384), ((12, 0), 8192)],
)
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
        ((8, 0), "6.4", True, "1"),
        ((8, 0), None, False, None),
        ((10, 0), None, True, None),
        ((12, 0), None, True, "0"),
    ],
)
def test_auto_min_seq_is_off(monkeypatch, capability, hip, cuda, setting):
    _fake_gpu(monkeypatch, capability, hip, cuda)
    if setting is not None:
        monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", setting)
    assert SA.save_attention_auto_min_seq() is None
    assert SA._AUTO_ACTIVE[0] is False
    # An explicit setting keeps its own (unbudgeted) mode; ROCm eligibility is stubbed (tested above).
    monkeypatch.setattr(SA, "_rocm_sdpa_is_kept", lambda device = None: True)
    assert SA.save_attention_enabled() is (setting in ("1", "offload"))


@pytest.mark.parametrize("arch", ["gfx1151", "gfx1151:sramecc-:xnack-"])
def test_auto_min_seq_on_measured_rocm_arch(monkeypatch, arch):
    _fake_gpu(monkeypatch, (11, 5), hip = "7.13.0", arch = arch)
    assert SA.save_attention_auto_min_seq() == 8192
    assert SA._AUTO_ACTIVE[0] is True
    monkeypatch.setattr(SA, "_rocm_sdpa_is_kept", lambda device = None: True)
    assert SA._auto_frame_min_seq(torch.device("cuda", 0)) == 8192
    # SDPA not kept (or no answer): no frame.
    for kept in (False, None):
        monkeypatch.setattr(SA, "_rocm_sdpa_is_kept", lambda device = None, kept = kept: kept)
        assert SA._auto_frame_min_seq(torch.device("cuda", 0)) is None


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201", "gfx942", "gfx90a", ""])
def test_auto_min_seq_off_on_unmeasured_rocm_arch(monkeypatch, arch):
    _fake_gpu(monkeypatch, (11, 0), hip = "7.13.0", arch = arch)
    assert SA.save_attention_auto_min_seq() is None
    assert SA._AUTO_ACTIVE[0] is False
    assert SA._auto_frame_min_seq(torch.device("cuda", 0)) is None


def test_auto_min_seq_is_decided_per_device(monkeypatch):
    # Mixed host: A100 (8K), L4 (16K), unmeasured; each checkpoint uses its own device's length.
    caps = {0: (8, 0), 1: (8, 9), 2: (9, 0)}
    _fake_gpu(monkeypatch, None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: caps[index])
    assert SA.save_attention_auto_min_seq() == 8192
    assert SA.save_attention_auto_min_seq(torch.device("cuda", 1)) == 16384
    assert SA._auto_frame_min_seq(torch.device("cuda", 0)) == 8192
    assert SA._auto_frame_min_seq(torch.device("cuda", 1)) == 16384
    assert SA._auto_frame_min_seq(torch.device("cuda", 2)) is None
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert SA.save_attention_auto_min_seq("cuda") == 16384
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION_MIN_SEQ", "256")
    assert SA._auto_frame_min_seq(torch.device("cuda", 1)) == 256


def test_auto_declines_calls_shorter_than_the_frame_device_length(monkeypatch, auto):
    _fake_memory(monkeypatch, base_peak = 0, capacity = 1 << 40)
    meta = torch.device("meta")
    short, long = torch.empty(1, 2, 8, 4, device = meta), torch.empty(1, 2, 64, 4, device = meta)
    measure = SA.SavedAttentionFrame(auto = True)
    SA._auto_admit(measure, [long])
    SA._auto_release(measure, True)
    frame = SA.SavedAttentionFrame(True, {meta: 16})
    assert not SA._auto_admit(frame, [short])
    assert SA.SAVED_ATTENTION_STATS["declined_short"] == 1
    assert SA._auto_admit(frame, [long])
    assert SA._auto_admit(SA.SavedAttentionFrame(auto = True), [short])


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA (the real allocator's peak and free memory)")
def test_cuda_auto_mode_with_the_real_allocator(monkeypatch, auto):
    # The budget against the real allocator, not the per-device length: every call long enough on any GPU.
    monkeypatch.setattr(SA, "save_attention_auto_min_seq", lambda device = None: 1)
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


def test_rejected_function_graph_is_freed_before_backward(monkeypatch):
    """A refused first-pass graph (too big here, random likewise) is never backpropagated: its saved tensors
    must not hold it. sigmoid saves its own output, which an identity pack hook makes hold its own node."""
    import gc

    from torch.multiprocessing.reductions import StorageWeakRef

    params, x = _make(n_layers = 1)
    first_pass = []

    def attention(q, k, v):
        gate = torch.sigmoid(q * 1.0)
        first_pass.append(StorageWeakRef(gate.untyped_storage()))
        return _sdpa(gate, (k * 1.0).repeat(1, 1, 1, 1), (v * 1.0).repeat(1, 1, 1, 1))

    call = lambda fn, q, k, v: SA.saved_function_call(fn, 3, q, k, v)  # noqa: E731
    monkeypatch.setattr(SA, "_FUNCTION_SLACK_BYTES", 0)
    full = _run(params, x, ckpt = True, attention = attention)
    first_pass.clear()
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "1")
    for w in params:
        for p in w:
            p.grad = None
    h = x.clone().requires_grad_()
    h = G.UnslothCheckpointFunction.apply(_layer(params[0], attention, call), True, h)
    gc.collect()
    assert SA.SAVED_ATTENTION_STATS["too_big"] == 1 and len(first_pass) == 1
    assert first_pass[0].expired(), "the refused first-pass graph outlived the forward"
    h.float().pow(2).mean().backward()
    for a, b in zip([p.grad for p in params[0]], full[1]):
        torch.testing.assert_close(a, b, rtol = 1e-4, atol = 1e-6)


# Automatic mode, host fallback: a call the device budget declines is kept in pinned host memory
# (the GC's pooled blocks, under its host RAM cap) instead of being recomputed.


def _fake_host(monkeypatch, refuse = False):
    """CPU stand-in for the pinned host copy: every tensor counts as offloadable; a refused copy is None."""
    moved = []

    def plan(t, require_pinned):
        return None if refuse and require_pinned else "plan"

    def copy(t, plan):
        moved.append(t.numel() * t.element_size())
        return SA._OnHost(t.clone(), t.device, None, None, SA._HOST_POOL_EPOCH[0], True)

    monkeypatch.setattr(SA, "_offloadable", lambda t: isinstance(t, torch.Tensor))
    monkeypatch.setattr(G, "_pinned_memory_disabled", lambda: False)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION_AUTO_OFFLOAD", "1")
    monkeypatch.setattr(SA, "_plan_host", plan)
    monkeypatch.setattr(SA, "_copy_to_host", copy)
    monkeypatch.setattr(SA, "_to_device", lambda item: item.host if isinstance(item, SA._OnHost) else item)
    return moved


def test_auto_offloads_what_the_device_budget_declines(monkeypatch, auto):
    params, x = _make()
    full = _main(monkeypatch, params, x, 3)
    moved = _fake_host(monkeypatch)
    _fake_memory(monkeypatch, base_peak = 1 << 20, capacity = (1 << 20) + 4 * _LAYER_BYTES + _LAYER_BYTES // 2)
    # The margin (4 layers) holds the reload prefetched a layer ahead, nothing else fits.
    monkeypatch.setattr(SA, "_AUTO_MARGIN_BYTES", 4 * _LAYER_BYTES)
    saved = _run(params, x, ckpt = True, steps = 3)
    _close(saved, full)
    stats = SA.SAVED_ATTENTION_STATS
    # First step unmeasured (recomputed as before); then every layer goes to the host and is reused.
    assert stats["declined_unmeasured"] == 3, dict(stats)
    assert stats["stored_host"] == 6 and stats["reused"] == 6 and stats["declined_budget"] == 0, dict(stats)
    assert saved[2] == 6 + 3 + 3
    assert len(moved) >= 6
    # Nothing was charged to the device budget for outputs that live on the host.
    assert all(b.live == 0 and b.reserve > 0 for b in SA._BUDGETS.values())


def test_auto_host_fallback_declines_when_host_ram_does_not_fit(monkeypatch, auto):
    params, x = _make()
    full = _main(monkeypatch, params, x, 2)
    _fake_host(monkeypatch, refuse = True)
    _fake_memory(monkeypatch, base_peak = 1 << 20, capacity = (1 << 20) + 4 * _LAYER_BYTES + _LAYER_BYTES // 2)
    monkeypatch.setattr(SA, "_AUTO_MARGIN_BYTES", 4 * _LAYER_BYTES)
    saved = _run(params, x, ckpt = True, steps = 2)
    _close(saved, full)
    stats = SA.SAVED_ATTENTION_STATS
    # The first refusal turns the device back to GPU-only keeping (no reserve, no more host attempts).
    assert stats["declined_host"] == 1 and stats["declined_budget"] == 2, dict(stats)
    assert stats["stored"] == 0 and stats["reused"] == 0, dict(stats)
    assert all(b.host_refused and b.reserve == 0 for b in SA._BUDGETS.values())
    assert saved[2] == 12


def test_auto_host_fallback_kill_switch(monkeypatch, auto):
    params, x = _make()
    _fake_host(monkeypatch)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION_AUTO_OFFLOAD", "0")
    _fake_memory(monkeypatch, base_peak = 1 << 20, capacity = (1 << 20) + 4 * _LAYER_BYTES + _LAYER_BYTES // 2)
    monkeypatch.setattr(SA, "_AUTO_MARGIN_BYTES", 4 * _LAYER_BYTES)
    saved = _run(params, x, ckpt = True, steps = 2)
    stats = SA.SAVED_ATTENTION_STATS
    assert stats["stored"] == 0 and stats["declined_budget"] == 3 and "stored_host" not in stats, dict(stats)
    assert saved[2] == 12


def test_auto_offloading_device_leaves_a_layer_for_the_prefetch(monkeypatch, auto):
    params, x = _make()
    full = _main(monkeypatch, params, x, 3)
    _fake_host(monkeypatch)
    base_peak = 1 << 20
    # Room for two layers: step 2 keeps two on the device and offloads one; from then on one more layer
    # stays free for the reload prefetched ahead, so step 3 keeps one and offloads two.
    _fake_memory(monkeypatch, base_peak, capacity = base_peak + 4 * _LAYER_BYTES + _LAYER_BYTES // 2)
    monkeypatch.setattr(SA, "_AUTO_MARGIN_BYTES", 2 * _LAYER_BYTES)
    saved = _run(params, x, ckpt = True, steps = 3)
    _close(saved, full)
    stats = SA.SAVED_ATTENTION_STATS
    assert stats["stored"] == 6 and stats["reused"] == 6, dict(stats)
    assert stats["stored_host"] == 3, dict(stats)


def test_auto_host_fallback_needs_room_for_the_prefetched_reload(monkeypatch, auto):
    params, x = _make()
    _fake_host(monkeypatch)
    # Margin smaller than one layer and no room at all: the reload prefetched a layer ahead would not fit.
    _fake_memory(monkeypatch, base_peak = 1 << 20, capacity = 1 << 20)
    monkeypatch.setattr(SA, "_AUTO_MARGIN_BYTES", _LAYER_BYTES)
    saved = _run(params, x, ckpt = True, steps = 2)
    stats = SA.SAVED_ATTENTION_STATS
    assert stats["stored"] == 0 and stats["declined_prefetch"] == 3, dict(stats)
    assert saved[2] == 12


def test_auto_host_fallback_is_opt_in_until_a_gpu_is_measured(monkeypatch, auto):
    query = torch.empty(1, 2, 8, 4)
    monkeypatch.setattr(SA, "_offloadable", lambda t: True)
    monkeypatch.setattr(G, "_pinned_memory_disabled", lambda: False)
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset())
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 0))
    monkeypatch.delenv("UNSLOTH_GC_SAVE_ATTENTION_AUTO_OFFLOAD", raising = False)
    assert SA._AUTO_HOST_MIN_SEQ == {} and not SA._auto_host_allowed(query)
    monkeypatch.setattr(SA, "_AUTO_HOST_MIN_SEQ", {(8, 0): 8})
    assert SA._auto_host_allowed(query)
    monkeypatch.setattr(SA, "_AUTO_HOST_MIN_SEQ", {(8, 0): 16})
    assert not SA._auto_host_allowed(query)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION_AUTO_OFFLOAD", "1")
    assert SA._auto_host_allowed(query)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION_AUTO_OFFLOAD", "0")
    monkeypatch.setattr(SA, "_AUTO_HOST_MIN_SEQ", {(8, 0): 8})
    assert not SA._auto_host_allowed(query)


def test_auto_host_fallback_off_on_unified_memory(monkeypatch, auto):
    query = torch.empty(1, 2, 8, 4)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION_AUTO_OFFLOAD", "1")
    monkeypatch.setattr(SA, "_offloadable", lambda t: True)
    monkeypatch.setattr(G, "_pinned_memory_disabled", lambda: False)
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset({None}))
    assert not SA._auto_host_allowed(query)
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset())
    assert SA._auto_host_allowed(query)


def test_host_pool_reuses_blocks_and_returns_their_charge(monkeypatch):
    released = []
    monkeypatch.setattr(SA, "_HOST_POOL", {})
    monkeypatch.setattr(SA, "_block_pinned", lambda block: True)
    monkeypatch.setattr(G, "_release_pinned", lambda nbytes: released.append(nbytes))
    size = 5_000_000
    block, event, epoch = SA._take_block(0, size)
    assert block.numel() == SA._bucket_bytes(size) >= size and event is None
    SA._give_block(0, block, "e", epoch)
    # A slightly longer call of the same bucket reuses the block, with the event of its last copy.
    again, event, _ = SA._take_block(0, size + 4096)
    assert again is block and event == "e"
    SA._give_block(0, again, None, epoch)
    assert SA.host_pool_bytes() == block.numel()
    # Another device never shares a block (its copies run on another stream).
    other, _, _ = SA._take_block(1, size)
    assert other is not block
    SA.release_saved_attention_host_pool()
    assert SA.host_pool_bytes() == 0 and released == [block.numel()]
    # A block taken before the reset is freed when returned, not pooled.
    SA._give_block(1, other, None, epoch)
    assert SA.host_pool_bytes() == 0 and released == [block.numel(), other.numel()]


def test_host_pool_stays_bounded_across_lengths(monkeypatch):
    monkeypatch.setattr(SA, "_HOST_POOL", {})
    monkeypatch.setattr(SA, "_HOST_USE", {})
    monkeypatch.setattr(SA, "_block_pinned", lambda block: True)
    monkeypatch.setattr(G, "_release_pinned", lambda nbytes: None)
    big, small = 64 << 20, 5 << 20
    held = [SA._take_block(0, big) for _ in range(3)]
    for block, _, epoch in held:
        SA._give_block(0, block, None, epoch)
    # A small request never takes a block over twice its size (the logsumexp of a layer, a short batch).
    block, _, epoch = SA._take_block(0, small)
    assert block.numel() < 2 * small
    SA._give_block(0, block, None, epoch)
    # Idle bytes stay within 1.25x the most ever held at once, whatever the mix of lengths.
    for n in (small, 2 * small, 3 * small, big, small):
        taken = [SA._take_block(0, n) for _ in range(3)]
        for block, _, epoch in taken:
            SA._give_block(0, block, None, epoch)
        in_use, most, idle = SA._HOST_USE[0]
        assert in_use == 0 and idle == SA.host_pool_bytes() <= 1.25 * most


def test_host_pool_frees_blocks_outgrown_by_a_longer_call(monkeypatch):
    monkeypatch.setattr(SA, "_HOST_POOL", {})
    monkeypatch.setattr(SA, "_HOST_USE", {})
    monkeypatch.setattr(SA, "_block_pinned", lambda block: True)
    released = []
    monkeypatch.setattr(G, "_release_pinned", lambda nbytes: released.append(nbytes))
    short, long = 8 << 20, 32 << 20
    taken = [SA._take_block(0, short) for _ in range(3)]
    for block, _, epoch in taken:
        SA._give_block(0, block, None, epoch)
    assert SA.host_pool_bytes() == 3 * short
    # A longer step: the short blocks can serve none of its calls, so the first miss frees them, and
    # holds no reference to them while the replacement is allocated.
    import gc
    from torch.multiprocessing.reductions import StorageWeakRef
    refs = [StorageWeakRef(b.untyped_storage()) for b, _, _ in taken]
    del taken, block
    seen = []
    monkeypatch.setattr(SA, "_new_block", lambda n: (gc.collect(), seen.append([r.expired() for r in refs]))
                        and torch.empty(n, dtype = torch.uint8))
    block, _, epoch = SA._take_block(0, long)
    assert seen == [[True] * 3], seen
    assert SA.host_pool_bytes() == 0 and released == [short] * 3
    SA._give_block(0, block, None, epoch)
    assert SA.host_pool_bytes() == block.numel() and SA._HOST_USE[0][2] == block.numel()


def test_host_pool_refusal_is_latched(monkeypatch):
    monkeypatch.setattr(SA, "_HOST_POOL", {})
    monkeypatch.setattr(SA, "_HOST_REFUSED", set())
    made = []
    monkeypatch.setattr(SA, "_new_block", lambda n: made.append(n) or torch.empty(n, dtype = torch.uint8))
    monkeypatch.setattr(SA, "_block_pinned", lambda block: False)
    assert SA._take_block(0, 8 << 20, require_pinned = True)[0] is None
    assert SA._take_block(0, 8 << 20, require_pinned = True)[0] is None
    # The pinned cap is asked once per size (it may flush the host cache, a device sync), not per layer.
    assert len(made) == 1
    SA.release_saved_attention_host_pool()
    assert SA._take_block(0, 8 << 20, require_pinned = True)[0] is None and len(made) == 2


def test_offload_entry_keeps_non_tensor_outputs(monkeypatch):
    _fake_host(monkeypatch)
    t = torch.ones(4)
    entry = [("f", "sig", (t, None, 3, t))]
    assert SA._offload_entry(entry, require_pinned = True)
    func, sig, out = entry[0]
    assert isinstance(out[0], SA._OnHost) and out[1] is None and out[2] == 3 and isinstance(out[3], SA._OnHost)


def test_finalizer_never_takes_the_pool_lock(monkeypatch):
    monkeypatch.setattr(SA, "_HOST_POOL", {})
    monkeypatch.setattr(SA, "_HOST_USE", {})
    monkeypatch.setattr(SA, "_block_pinned", lambda block: True)
    block, _, epoch = SA._take_block(0, 8 << 20)
    item = SA._OnHost(block, torch.device("cpu"), block, None, epoch, True)
    item.device = types.SimpleNamespace(index = 0)
    # The garbage collector can finalize an entry while this thread holds the lock: it only queues.
    with SA._HOST_POOL_LOCK:
        item.__del__()
    assert len(SA._HOST_RETURNS) == 1
    again, _, _ = SA._take_block(0, 8 << 20)
    assert again is block and not SA._HOST_RETURNS


def test_pageable_blocks_are_never_pooled(monkeypatch):
    monkeypatch.setattr(SA, "_HOST_POOL", {})
    monkeypatch.setattr(SA, "_HOST_USE", {})
    monkeypatch.setattr(SA, "_block_pinned", lambda block: False)
    block, _, epoch = SA._take_block(0, 8 << 20)  # explicit offload with the pinned cap full: pageable
    SA._give_block(0, block, None, epoch)
    assert SA.host_pool_bytes() == 0 and SA._HOST_USE[0][0] == 0


def test_refused_output_copies_nothing(monkeypatch):
    taken, copies = [], []
    monkeypatch.setattr(SA, "_offloadable", lambda t: isinstance(t, torch.Tensor))
    monkeypatch.setattr(SA, "_plan_host", lambda t, req: None if t.numel() == 2 else taken.append(t) or ("b", None, 0))
    monkeypatch.setattr(SA, "_copy_to_host", lambda t, plan: copies.append(t))
    given = []
    monkeypatch.setattr(SA, "_give_block", lambda *a: given.append(a))
    entry = [("f", "sig", (torch.ones(4), torch.ones(2)))]
    # Every block is taken before any copy: the second output's refusal leaves no copy behind.
    assert SA._offload_entry(entry, require_pinned = True) == 0
    assert copies == [] and len(given) == 1 and isinstance(entry[0][2][0], torch.Tensor)


def _host_budget(monkeypatch, layer, margin):
    _fake_memory(monkeypatch, base_peak = 1 << 20, capacity = (1 << 20) + margin)
    monkeypatch.setattr(SA, "_AUTO_MARGIN_BYTES", margin)
    monkeypatch.setattr(SA, "_auto_host_allowed", lambda query: True)
    frame = SA.SavedAttentionFrame(auto = True)
    query = torch.empty(layer // 4 // 2, 1, 1, 2)  # fp32: estimate = bytes + one fp32 per row
    estimate = SA._bytes(query) + query.numel() // 2 * 4
    SA._auto_admit(frame, [query])
    SA._auto_release(frame, True)  # one measured step
    SA.SAVED_ATTENTION_STATS.clear()
    return query, estimate


def test_prefetch_check_counts_the_whole_frame(monkeypatch, auto):
    query, estimate = _host_budget(monkeypatch, 1 << 16, margin = 4 * (1 << 16))
    frame = SA.SavedAttentionFrame(auto = True)
    room_for = (4 * (1 << 16)) // 2 // estimate  # calls of this frame whose reload fits in half the margin
    results = []
    for _ in range(room_for + 1):
        results.append(SA._auto_admit(frame, [query]))
        if results[-1] is SA._HOST:
            frame.host_bytes += estimate
    assert results[:room_for] == [SA._HOST] * room_for and results[-1] is False
    assert SA.SAVED_ATTENTION_STATS["declined_prefetch"] == 1 and SA.SAVED_ATTENTION_STATS["declined_budget"] == 0
    (budget,) = SA._BUDGETS.values()
    assert budget.reserve == (room_for + 1) * estimate


def test_reserve_only_applies_while_the_host_fallback_can_run(monkeypatch, auto):
    query, estimate = _host_budget(monkeypatch, 1 << 16, margin = 1 << 20)
    (budget,) = SA._BUDGETS.values()
    monkeypatch.setattr(SA, "_device_capacity", lambda device: (1 << 20) + (1 << 20) + 2 * estimate)
    budget.capacity = None
    budget.reserve = 2 * estimate
    # The reserve would leave no room for a GPU keep, but with the fallback off it is not applied.
    monkeypatch.setattr(SA, "_auto_host_allowed", lambda query: False)
    assert SA._auto_admit(SA.SavedAttentionFrame(auto = True), [query]) is True
    budget.live = 0
    monkeypatch.setattr(SA, "_auto_host_allowed", lambda query: True)
    assert SA._auto_admit(SA.SavedAttentionFrame(auto = True), [query]) is SA._HOST
    budget.live = 0
    budget.host_refused = True
    assert SA._auto_admit(SA.SavedAttentionFrame(auto = True), [query]) is True
    SA.release_saved_attention_host_pool()
    assert budget.reserve == 0 and not budget.host_refused


def test_dense_layouts():
    assert SA._dense(torch.empty(2, 3, 4))
    # Flash attention's output: (B, H, S, D) shape over (B, S, H, D) memory.
    assert SA._dense(torch.empty(2, 8, 4, 16).transpose(1, 2))
    assert SA._dense(torch.empty(1, 4, 1, 8)[:, :, :, :].expand(1, 4, 1, 8))
    assert not SA._dense(torch.empty(4, 8)[:, :4])
    assert not SA._dense(torch.empty(4, 1).expand(4, 8))


def test_host_pool_buckets():
    assert SA._bucket_bytes(1000) == 1000
    assert SA._bucket_bytes(4 << 20) == 4 << 20
    for n in (5 << 20, (5 << 20) + 1, 37 * 1000 * 1000, 3 << 30):
        b = SA._bucket_bytes(n)
        assert n <= b < n * 1.125 + 1


def test_gc_buffer_reset_releases_the_host_pool(monkeypatch):
    calls = []
    monkeypatch.setattr(G, "release_saved_attention_host_pool", lambda: calls.append(1))
    G.initialize_unsloth_gradient_checkpointing(torch.float32)
    G.reset_unsloth_gradient_checkpointing_buffers()
    assert len(calls) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA (pinned host memory, side stream)")
def test_cuda_offload_reuses_pooled_pinned_blocks(monkeypatch):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("no bf16 on this GPU")
    monkeypatch.setenv("UNSLOTH_DISABLE_PINNED_MEMORY", "0")
    monkeypatch.setattr(SA, "_OFFLOAD_MIN_BYTES", 0)
    monkeypatch.setattr(SA, "_HOST_POOL", {})
    made = []
    new_block = SA._new_block
    monkeypatch.setattr(SA, "_new_block", lambda n: made.append(n) or new_block(n))
    params, x = _make()
    params = [[p.detach().to("cuda", torch.bfloat16).requires_grad_() for p in w] for w in params]
    x = x.to("cuda", torch.bfloat16)
    full = _run(params, x, ckpt = True, steps = 3)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", "offload")
    pinned_before = G.GC_PINNED_BYTES
    saved = _run(params, x, ckpt = True, steps = 1)
    first = len(made)
    saved = _run(params, x, ckpt = True, steps = 2)
    torch.testing.assert_close(saved[0], full[0])
    for a, b in zip(saved[1], full[1]):
        torch.testing.assert_close(a, b)
    stats = SA.SAVED_ATTENTION_STATS
    assert stats["reused"] == 9 and stats["mismatch"] == 0 and stats["prefetched"] >= 2, dict(stats)
    # Steady state allocates no host memory: later steps reuse the first step's blocks.
    assert first > 0 and len(made) == first, (first, made)
    pool = SA.host_pool_bytes()
    assert pool == sum(made)
    assert all(SA._block_pinned(b) for free in SA._HOST_POOL.values() for b, _ in free)
    # The blocks are charged against the GC's pinned cap.
    assert G.GC_PINNED_BYTES >= pinned_before + pool
    SA.release_saved_attention_host_pool()
    assert SA.host_pool_bytes() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA (pinned host memory, side stream)")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cuda_auto_host_fallback_matches(monkeypatch, auto, dtype):
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("no bf16 on this GPU")
    if torch.cuda.current_device() in G._keep_on_device_indices():
        pytest.skip("unified memory: the host fallback is off by design")
    monkeypatch.setenv("UNSLOTH_DISABLE_PINNED_MEMORY", "0")
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION_AUTO_OFFLOAD", "1")
    monkeypatch.setattr(SA, "_OFFLOAD_MIN_BYTES", 0)
    monkeypatch.setattr(SA, "_HOST_POOL", {})
    monkeypatch.setattr(SA, "save_attention_auto_min_seq", lambda device = None: 1)
    params, x = _make()
    params = [[p.detach().to("cuda", dtype).requires_grad_() for p in w] for w in params]
    x = x.to("cuda", dtype)
    full = _main(monkeypatch, params, x, 3)
    # No device room beyond the margin, which still holds the prefetched reload: admitted calls go to the host.
    monkeypatch.setattr(SA, "_device_capacity", lambda device: 1 << 30)
    monkeypatch.setattr(SA, "_AUTO_MARGIN_BYTES", 1 << 30)
    monkeypatch.setattr(SA, "_AUTO_MARGIN_FRACTION", 0.0)
    monkeypatch.setattr(SA, "_expected_peak", lambda budget: 0)
    saved = _run(params, x, ckpt = True, steps = 3)
    stats = SA.SAVED_ATTENTION_STATS
    torch.testing.assert_close(saved[0], full[0])
    for a, b in zip(saved[1], full[1]):
        torch.testing.assert_close(a, b)
    assert stats["declined_unmeasured"] == 3, dict(stats)
    assert stats["stored_host"] == 6 and stats["reused"] == 6 and stats["mismatch"] == 0, dict(stats)
    assert stats["offloaded_bytes"] > 0 and stats["prefetched"] >= 2, dict(stats)
    assert all(b.live == 0 for b in SA._BUDGETS.values())
