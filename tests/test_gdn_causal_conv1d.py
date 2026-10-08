"""Triton GatedDeltaNet causal conv vs the transformers torch fallback."""

import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth_zoo.temporary_patches import gdn_causal_conv1d as gcc  # noqa: E402

_HAS_CUDA = torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason = "needs a CUDA GPU")


def _inputs(B, D, T, W, dtype, bias, channel_last, seed = 0):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    if channel_last:
        x = torch.randn(B, T, D, device = "cuda", generator = g).to(dtype).transpose(1, 2)
    else:
        x = torch.randn(B, D, T, device = "cuda", generator = g).to(dtype)
    w = (torch.randn(D, W, device = "cuda", generator = g) * 0.5).to(dtype)
    b = (torch.randn(D, device = "cuda", generator = g) * 0.1).to(dtype) if bias else None
    dy = torch.randn(B, D, T, device = "cuda", generator = g).to(dtype)
    return x, w, b, dy


def _run(fn, x, w, b, dy, act):
    x = x.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)
    b = b.detach().clone().requires_grad_(True) if b is not None else None
    y = fn(x, w, b, act)
    y.backward(dy)
    return y.detach(), x.grad, w.grad, (b.grad if b is not None else None)


def _to64(t):
    return t.double() if t is not None else None


def _oracle(x, w, b, dy, act):
    to64 = _to64
    return _run(gcc.causal_conv1d_reference, to64(x), to64(w), to64(b), to64(dy), act)


def _rel(a, ref):
    a, ref = a.double(), ref.double()
    return ((a - ref).norm() / ref.norm().clamp_min(1e-30)).item()


SHAPES = [
    (1, 64, 1, 4), (2, 64, 3, 4), (1, 100, 63, 4), (2, 130, 65, 4),
    (2, 96, 1000, 3), (1, 70, 129, 2), (2, 6144, 4096, 4), (1, 6144, 1031, 4),
]


@requires_cuda
@pytest.mark.parametrize("B,D,T,W", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("act", ["silu", None])
@pytest.mark.parametrize("channel_last", [True, False])
def test_matches_reference(B, D, T, W, dtype, bias, act, channel_last):
    if D * T > 1 << 22 and (bias or act is None or not channel_last or dtype == torch.float16):
        pytest.skip("large shape: one config is enough")
    x, w, b, dy = _inputs(B, D, T, W, dtype, bias, channel_last)
    ours = _run(gcc.triton_causal_conv1d, x, w, b, dy, act)
    torch_ = _run(gcc.causal_conv1d_reference, x, w, b, dy, act)
    oracle = _oracle(x, w, b, dy, act)
    assert ours[0].shape == torch_[0].shape and ours[0].dtype == torch_[0].dtype
    # Output stored channel-last: the GDN forward transposes it back to contiguous.
    assert ours[0].transpose(1, 2).is_contiguous()
    names = ("y", "dx", "dw", "db")
    for name, o, t, r in zip(names, ours, torch_, oracle):
        if r is None:
            assert o is None and t is None
            continue
        e_ours, e_torch = _rel(o, r), _rel(t, r)
        if dtype == torch.float32:
            assert e_ours < 1e-5, (name, e_ours, e_torch)
        else:
            # Within bf16/fp16 rounding: no worse than torch's own error by more than 2x,
            # plus a tiny floor for tensors where torch happens to be near exact.
            floor = 4e-3 if dtype == torch.bfloat16 else 5e-4
            assert e_ours <= 2 * e_torch + floor, (name, e_ours, e_torch)
            # And elementwise close to the torch bf16/fp16 result.
            assert _rel(o, t.to(o.dtype)) < 2 * floor + 2 * e_torch, (name, _rel(o, t), e_torch)


@requires_cuda
def test_bitwise_forward_bf16_no_bias():
    # Kernel 4, fp32 accumulate, round, SiLU: matches torch's forward closely in bf16.
    x, w, _, _ = _inputs(2, 6144, 512, 4, torch.bfloat16, False, True)
    y = gcc.triton_causal_conv1d(x, w, None, "silu")
    ref = gcc.causal_conv1d_reference(x, w, None, "silu")
    frac_equal = (y == ref).float().mean().item()
    assert frac_equal > 0.95, frac_equal


@requires_cuda
def test_frozen_weight_only_dx():
    x, w, _, dy = _inputs(2, 256, 300, 4, torch.bfloat16, False, True)
    xr = x.detach().clone().requires_grad_(True)
    y = gcc.triton_causal_conv1d(xr, w, None, "silu")
    y.backward(dy)
    assert xr.grad is not None and w.grad is None


@requires_cuda
def test_dispatch_fallbacks_and_kill_switch(monkeypatch):
    x, w, _, _ = _inputs(1, 64, 33, 4, torch.bfloat16, False, True)
    calls = []

    def torch_fn(hidden_states, weight, bias = None, activation = None, **kwargs):
        calls.append(1)
        return gcc.causal_conv1d_reference(hidden_states, weight, bias, activation)

    torch_fn.__module__ = "transformers.models.qwen3_5.modeling_qwen3_5"
    fn = gcc._make_hub_dispatch(torch_fn)
    before = dict(gcc.GDN_CAUSAL_CONV1D_STATS)
    fn(x, w, None, "silu")
    assert gcc.GDN_CAUSAL_CONV1D_STATS["triton_fwd"] == before["triton_fwd"] + 1 and not calls
    # CPU, dtype mismatch, odd activation -> torch fallback.
    fn(x.float().cpu(), w.float().cpu(), None, "silu")
    fn(x, w.float(), None, "silu")
    fn(x, w, None, "gelu")
    assert len(calls) == 3
    monkeypatch.setenv(gcc._KILL_SWITCH, "1")
    fn(x, w, None, "silu")
    assert len(calls) == 4
    assert gcc.GDN_CAUSAL_CONV1D_STATS["triton_fwd"] == before["triton_fwd"] + 1


@requires_cuda
def test_legacy_entry_point():
    x, w, b, _ = _inputs(2, 64, 40, 4, torch.bfloat16, True, True)
    y = gcc._legacy_causal_conv1d_fn(x = x, weight = w, bias = b, activation = "silu", seq_idx = None)
    ref = gcc.causal_conv1d_reference(x, w, b, "silu")
    assert _rel(y, ref) < 1e-2
    with pytest.raises(NotImplementedError):
        gcc._legacy_causal_conv1d_fn(x = x.cpu(), weight = w.cpu(), seq_idx = torch.zeros(1))


def _fake_module(name, fn_module):
    import types
    module = types.ModuleType(name)

    def causal_conv1d_fn(hidden_states, weight, bias = None, activation = None, **kwargs):
        return gcc.causal_conv1d_reference(hidden_states, weight, bias, activation)

    causal_conv1d_fn.__module__ = fn_module
    return module, causal_conv1d_fn


def test_hub_decorator_wraps_only_gdn_fallbacks(monkeypatch):
    hub_kernels = pytest.importorskip("transformers.integrations.hub_kernels")
    if not hasattr(hub_kernels, "use_kernel_func_from_hub_with_fallback"):
        pytest.skip("transformers without use_kernel_func_from_hub_with_fallback")
    monkeypatch.setattr(gcc, "_real_causal_conv1d_available", lambda: False)
    original = hub_kernels.use_kernel_func_from_hub_with_fallback
    try:
        patched = gcc._patch_hub_decorator()
        assert getattr(patched, gcc._HUB_MARK, False)
        # Idempotent.
        assert gcc._patch_hub_decorator() is patched
        _, gdn_fn = _fake_module("m", "transformers.models.qwen3_5.modeling_qwen3_5")
        _, other_fn = _fake_module("m", "transformers.models.mamba2.modeling_mamba2")
        assert gcc._wraps_marked(patched("causal_conv1d_fn", "causal_conv1d")(gdn_fn))
        assert not gcc._wraps_marked(patched("causal_conv1d_fn", "causal_conv1d")(other_fn))
        assert not gcc._wraps_marked(patched("causal_conv1d_update", "causal_conv1d")(gdn_fn))

        # Already-decorated module global: rebound to the dispatch.
        module, gdn_fn = _fake_module("transformers.models.qwen3_5.modeling_qwen3_5", "transformers.models.qwen3_5.modeling_qwen3_5")
        module.causal_conv1d_fn = original("causal_conv1d_fn", "causal_conv1d")(gdn_fn)
        module.use_kernel_func_from_hub_with_fallback = original
        assert gcc._rebind_module(module, patched)
        assert gcc._wraps_marked(module.causal_conv1d_fn)
        assert module.use_kernel_func_from_hub_with_fallback is patched

        # transformers < 5.16 layout: a None global becomes the package-style entry point.
        legacy, _ = _fake_module("legacy", "x")
        legacy.causal_conv1d_fn = None
        assert gcc._rebind_module(legacy, patched)
        assert legacy.causal_conv1d_fn is gcc._legacy_causal_conv1d_fn
    finally:
        hub_kernels.use_kernel_func_from_hub_with_fallback = original
        import transformers.integrations as integrations
        if getattr(integrations, "use_kernel_func_from_hub_with_fallback", None) is not original:
            integrations.use_kernel_func_from_hub_with_fallback = original


def test_rebind_leaves_a_real_kernel_alone():
    module, gdn_fn = _fake_module("transformers.models.qwen3_5.modeling_qwen3_5", "transformers.models.qwen3_5.modeling_qwen3_5")

    def real_kernel(x, weight, bias = None, activation = None):
        return x

    def make_wrapper(implementation, torch_function):
        import functools

        @functools.wraps(torch_function)
        def wrapped(*args, **kwargs):
            return implementation(*args, **kwargs)
        return wrapped

    module.causal_conv1d_fn = make_wrapper(real_kernel, gdn_fn)
    assert not gcc._rebind_module(module, lambda *a, **k: (lambda f: f))
    assert module.causal_conv1d_fn.__wrapped__ is gdn_fn
