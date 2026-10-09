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

"""Triton GatedDeltaNet causal conv vs the transformers torch fallback."""

import inspect
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


def _oracle(x, w, b, dy, act):
    to64 = lambda t: t.double() if t is not None else None
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
            # At most 2x torch's own rounding error, plus a small floor.
            floor = 4e-3 if dtype == torch.bfloat16 else 5e-4
            assert e_ours <= 2 * e_torch + floor, (name, e_ours, e_torch)
            assert _rel(o, t.to(o.dtype)) < 2 * floor + 2 * e_torch, (name, _rel(o, t), e_torch)


@requires_cuda
def test_bitwise_forward_bf16_no_bias():
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
    monkeypatch.delenv(gcc._KILL_SWITCH, raising = False)
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
    fn(x.float().cpu(), w.float().cpu(), None, "silu")
    fn(x, w.float(), None, "silu")
    fn(x, w, None, "gelu")
    assert len(calls) == 3
    monkeypatch.setenv(gcc._KILL_SWITCH, "1")
    fn(x, w, None, "silu")
    assert len(calls) == 4
    assert gcc.GDN_CAUSAL_CONV1D_STATS["triton_fwd"] == before["triton_fwd"] + 1


@requires_cuda
def test_legacy_entry_point(monkeypatch):
    monkeypatch.delenv(gcc._KILL_SWITCH, raising = False)
    x, w, b, _ = _inputs(2, 64, 40, 4, torch.bfloat16, True, True)
    before = gcc.GDN_CAUSAL_CONV1D_STATS["triton_fwd"]
    y = gcc._legacy_causal_conv1d_fn(x = x, weight = w, bias = b, activation = "silu", seq_idx = None)
    assert gcc.GDN_CAUSAL_CONV1D_STATS["triton_fwd"] == before + 1
    assert _rel(y, gcc.causal_conv1d_reference(x, w, b, "silu")) < 1e-2


def test_oom_propagates_without_disabling(monkeypatch):
    monkeypatch.delenv(gcc._KILL_SWITCH, raising = False)
    monkeypatch.setattr(gcc, "_broken", False)
    monkeypatch.setattr(gcc, "_eligible", lambda *a: True)
    x = torch.zeros(1, 4, 3)

    def oom(*a):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(gcc, "triton_causal_conv1d", oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        gcc._try_fast(x, x[0], None, "silu")
    assert gcc._broken is False

    def fail(*a):
        raise RuntimeError("triton launch failed")

    monkeypatch.setattr(gcc, "triton_causal_conv1d", fail)
    assert gcc._try_fast(x, x[0], None, "silu") is None
    assert gcc._broken is True


def test_legacy_seq_idx_cpu():
    import inspect
    # Not advertised: unsloth's hybrid packing gate keys on a named `seq_idx`.
    assert "seq_idx" not in inspect.signature(gcc._legacy_causal_conv1d_fn).parameters
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 8, 10, generator = g, dtype = torch.float64)
    w = torch.randn(8, 4, generator = g, dtype = torch.float64)
    b = torch.randn(8, generator = g, dtype = torch.float64)
    seq_idx = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 2, 2, 2]], dtype = torch.int32)
    y = gcc._legacy_causal_conv1d_fn(x = x, weight = w, bias = b, activation = "silu", seq_idx = seq_idx)
    ref = torch.cat([
        gcc.causal_conv1d_reference(x[:, :, s:e], w, b, "silu") for s, e in ((0, 3), (3, 7), (7, 10))
    ], dim = -1)
    torch.testing.assert_close(y, ref)
    y = gcc._legacy_causal_conv1d_fn(x = x, weight = w, bias = b, activation = "silu", seq_idx = torch.zeros(1, 10, dtype = torch.int32))
    torch.testing.assert_close(y, gcc.causal_conv1d_reference(x, w, b, "silu"))
    with pytest.raises(NotImplementedError):
        gcc._legacy_causal_conv1d_fn(x = x, weight = w, initial_states = torch.zeros(1))


def _gdn_fn(module_name):
    def causal_conv1d_fn(hidden_states, weight, bias = None, activation = None, **kwargs):
        return gcc.causal_conv1d_reference(hidden_states, weight, bias, activation)

    causal_conv1d_fn.__module__ = module_name
    return causal_conv1d_fn


@pytest.fixture
def hub_kernels(monkeypatch):
    hub_kernels = pytest.importorskip("transformers.integrations.hub_kernels")
    if not hasattr(hub_kernels, "use_kernel_func_from_hub_with_fallback"):
        pytest.skip("transformers without use_kernel_func_from_hub_with_fallback")
    import transformers.integrations as integrations
    # On a CUDA host importing unsloth_zoo already patched it: start unpatched.
    original = _unpatched_decorator(hub_kernels.use_kernel_func_from_hub_with_fallback)
    monkeypatch.setattr(hub_kernels, "use_kernel_func_from_hub_with_fallback", original)
    monkeypatch.setattr(integrations, "use_kernel_func_from_hub_with_fallback", original)
    monkeypatch.setattr(gcc, "_real_causal_conv1d_available", lambda: False)
    monkeypatch.delenv(gcc._KILL_SWITCH, raising = False)
    return hub_kernels


def _unpatched_decorator(decorator):
    return decorator.__wrapped__ if getattr(decorator, gcc._HUB_MARK, False) else decorator


def _real_gdn_modeling(monkeypatch):
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    if getattr(modeling, "causal_conv1d_fn", None) is None:
        pytest.skip("transformers < 5.15 GatedDeltaNet")
    decorator = _unpatched_decorator(modeling.use_kernel_func_from_hub_with_fallback)
    fn = modeling.causal_conv1d_fn
    if gcc._is_marked(fn):
        # Patched at import on a CUDA host: rebuild the unpatched hub function.
        dispatch = inspect.unwrap(fn, stop = lambda f: getattr(f, gcc._MARK, False))
        fn = decorator("causal_conv1d_fn", "causal_conv1d")(dispatch.__wrapped__)
    monkeypatch.setattr(modeling, "causal_conv1d_fn", fn)
    monkeypatch.setattr(modeling, "use_kernel_func_from_hub_with_fallback", decorator)
    return modeling


def test_hub_decorator_wraps_only_gdn_fallbacks(hub_kernels):
    patched = gcc._patch_hub_decorator()
    assert getattr(patched, gcc._HUB_MARK, False)
    assert gcc._patch_hub_decorator() is patched
    gdn_fn = _gdn_fn("transformers.models.qwen3_5.modeling_qwen3_5")
    assert gcc._is_marked(patched("causal_conv1d_fn", "causal_conv1d")(gdn_fn))
    assert not gcc._is_marked(patched("causal_conv1d_fn", "causal_conv1d")(_gdn_fn("transformers.models.mamba2.modeling_mamba2")))
    assert not gcc._is_marked(patched("causal_conv1d_update", "causal_conv1d")(gdn_fn))

    import types
    legacy = types.ModuleType("legacy")
    legacy.causal_conv1d_fn = None
    assert gcc._rebind_module(legacy, patched)
    assert legacy.causal_conv1d_fn is gcc._legacy_causal_conv1d_fn


def test_patch_rebinds_the_real_modeling_module(hub_kernels, monkeypatch):
    pytest.importorskip("triton")
    modeling = _real_gdn_modeling(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    assert not gcc._is_marked(modeling.causal_conv1d_fn)

    monkeypatch.setenv(gcc._KILL_SWITCH, "1")
    gcc.patch_gdn_causal_conv1d()
    monkeypatch.delenv(gcc._KILL_SWITCH)
    monkeypatch.setattr(gcc, "_real_causal_conv1d_available", lambda: True)
    gcc.patch_gdn_causal_conv1d()
    assert not gcc._is_marked(modeling.causal_conv1d_fn)
    assert not getattr(hub_kernels.use_kernel_func_from_hub_with_fallback, gcc._HUB_MARK, False)

    monkeypatch.setattr(gcc, "_real_causal_conv1d_available", lambda: False)
    gcc.patch_gdn_causal_conv1d()
    assert gcc._is_marked(modeling.causal_conv1d_fn)
    assert getattr(modeling.use_kernel_func_from_hub_with_fallback, gcc._HUB_MARK, False)
    x, w = torch.randn(1, 8, 5), torch.randn(8, 4)
    before = gcc.GDN_CAUSAL_CONV1D_STATS["fallback"]
    torch.testing.assert_close(modeling.causal_conv1d_fn(x, w, None, activation = "silu"), gcc.causal_conv1d_reference(x, w, None, "silu"))
    assert gcc.GDN_CAUSAL_CONV1D_STATS["fallback"] == before + 1


def test_real_package_must_be_usable(monkeypatch):
    import sys
    import types
    import importlib.util
    fake = types.ModuleType("causal_conv1d")
    fake.causal_conv1d_fn = None
    monkeypatch.setitem(sys.modules, "causal_conv1d", fake)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())
    assert not gcc._real_causal_conv1d_available()
    fake.causal_conv1d_fn = lambda *a, **k: None
    assert gcc._real_causal_conv1d_available()


@requires_cuda
def test_gated_delta_net_layer_engages_triton(hub_kernels, monkeypatch):
    modeling = _real_gdn_modeling(monkeypatch)
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    config = Qwen3_5TextConfig(
        hidden_size = 64, linear_num_value_heads = 2, linear_num_key_heads = 2,
        linear_key_head_dim = 16, linear_value_head_dim = 16, num_hidden_layers = 1,
        layer_types = ["linear_attention"],
    )
    torch.manual_seed(0)
    layer = modeling.Qwen3_5GatedDeltaNet(config, 0).cuda().to(torch.bfloat16)
    x = torch.randn(2, 33, 64, device = "cuda", dtype = torch.bfloat16)

    def run():
        layer.zero_grad(set_to_none = True)
        xr = x.clone().requires_grad_(True)
        y = layer(xr)
        y.float().square().sum().backward()
        return y.detach(), xr.grad, layer.conv1d.weight.grad

    ref = run()
    stats = dict(gcc.GDN_CAUSAL_CONV1D_STATS)
    gcc.patch_gdn_causal_conv1d()
    ours = run()
    assert gcc.GDN_CAUSAL_CONV1D_STATS["triton_fwd"] == stats["triton_fwd"] + 1
    assert gcc.GDN_CAUSAL_CONV1D_STATS["triton_bwd"] == stats["triton_bwd"] + 1
    for o, r in zip(ours, ref):
        assert _rel(o, r) < 2e-2
    monkeypatch.setenv(gcc._KILL_SWITCH, "1")
    run()
    assert gcc.GDN_CAUSAL_CONV1D_STATS["triton_fwd"] == stats["triton_fwd"] + 1


def test_legacy_modeling_imported_at_patch_time(monkeypatch):
    # transformers < 5.15: the patch imports modeling modules itself.
    import importlib
    import sys
    import types
    monkeypatch.setattr(gcc, "_patch_hub_decorator", lambda: None)
    monkeypatch.setattr(gcc, "_real_causal_conv1d_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.delenv(gcc._KILL_SWITCH, raising = False)
    pytest.importorskip("triton")
    imported = []
    for name in [f"transformers.models.{p}.modeling_{p}" for p in gcc._GDN_MODELING]:
        monkeypatch.delitem(sys.modules, name, raising = False)

    def fake_import(name):
        module = types.ModuleType(name)
        module.causal_conv1d_fn = None
        sys.modules[name] = module
        imported.append(name)
        return module

    monkeypatch.setattr(importlib, "import_module", fake_import)
    try:
        gcc.patch_gdn_causal_conv1d()
        assert len(imported) == len(gcc._GDN_MODELING)
        assert all(sys.modules[name].causal_conv1d_fn is gcc._legacy_causal_conv1d_fn for name in imported)
    finally:
        for name in imported:
            sys.modules.pop(name, None)


def test_fallback_counter_is_not_read_while_tracing(monkeypatch):
    # A global read inside a compiled region would become a guard and recompile every call.
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    x, w = torch.randn(1, 8, 5), torch.randn(8, 4)
    before = dict(gcc.GDN_CAUSAL_CONV1D_STATS)
    gcc._legacy_causal_conv1d_fn(x, w, None, activation = "silu")
    gcc._make_hub_dispatch(gcc.causal_conv1d_reference)(x, w, None, "silu")
    assert gcc.GDN_CAUSAL_CONV1D_STATS == before
