# SPDX-License-Identifier: AGPL-3.0-only
import importlib
import inspect
import textwrap
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx_simulation import mlx_is_simulated

if mlx_is_simulated():
    pytest.skip("Requires native MLX", allow_module_level = True)
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason = "Requires Metal")

import mlx.nn as nn
from mlx_vlm.models.qwen3_5 import language as native
from unsloth_zoo.mlx import inference as decode


def _equal(a, b):
    assert np.array_equal(np.array(a.view(mx.uint8)), np.array(b.view(mx.uint8)))


def _model(kernel = 4):
    model = native.Qwen3_5GatedDeltaNet(SimpleNamespace(
        hidden_size = 32, linear_num_value_heads = 2, linear_num_key_heads = 1,
        linear_key_head_dim = 32, linear_value_head_dim = 32,
        linear_conv_kernel_dim = kernel, rms_norm_eps = 1e-6,
    ))
    model.set_dtype(mx.bfloat16)
    model.eval()
    model.freeze()
    return model


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_kernel_preserves_native_window_reduction_rounding_and_split(dtype):
    for width, channels, key_dim, batch in [(2, 3, 1, 1), (4, 96, 32, 3), (4, 8192, 2048, 1), (8, 131, 40, 2)]:
        spread = lambda shape: mx.random.normal(shape) * mx.exp(3 * mx.random.normal(shape))  # reaches exp rounding
        state = spread((batch, width - 1, channels)).astype(dtype)
        x = spread((batch, 1, 2 * channels)).astype(dtype)[..., channels:]  # strided, like a projection slice
        weight = mx.random.normal((channels, width)).T
        window = mx.concatenate([state, x], axis = 1)
        out = nn.silu(native._qwen3_5_decode_depthwise_conv(window, weight))
        for actual, expected in zip(decode._decode_conv(state, x, weight, key_dim),
                                    [window, *mx.split(out, [key_dim, 2 * key_dim], -1)]):
            _equal(actual, expected)


def test_scope_preserves_native_method_weights_and_restores(monkeypatch):
    models = [_model(), _model()]
    root = nn.Sequential(*models)
    root.eval()
    x = mx.random.normal((2, 4, 128)).astype(mx.bfloat16)
    with pytest.raises(RuntimeError, match = "cancel"):
        with decode.fused_decode_conv_silu(root):
            classes = [type(m) for m in models]
            assert all(c is not native.Qwen3_5GatedDeltaNet for c in classes)
            with decode.fused_decode_conv_silu(root):
                assert [type(m) for m in models] == classes
            step = mx.random.normal((2, 1, 32)).astype(mx.bfloat16)
            for model in models:
                for factor in (2, .25):
                    model.conv1d.weight = model.conv1d.weight * factor
                    assert type(model)._causal_conv1d_decode is native.Qwen3_5GatedDeltaNet._causal_conv1d_decode
                    _equal(model(step), native.Qwen3_5GatedDeltaNet.__call__(model, step))
            raise RuntimeError("cancel")
    assert all(type(m) is native.Qwen3_5GatedDeltaNet for m in models)


def test_decode_and_prefill_preserve_outputs_and_cache(monkeypatch):
    model = _model()
    samples = [mx.random.normal((2, n, 32)).astype(mx.bfloat16) for n in (3, 1, 1)]
    cache = native.ArraysCache(size = 2)
    expected = []
    for x in samples:
        out = model(x, cache = cache)
        mx.eval(out, cache.state)
        expected.append((out, list(cache.state)))
    cache = native.ArraysCache(size = 2)
    fused = decode._decode_conv
    calls = []
    def observed(*args):
        calls.append(None)
        return fused(*args)
    monkeypatch.setattr(decode, "_decode_conv", observed)
    with decode.fused_decode_conv_silu(model):
        for x, (out, states) in zip(samples, expected):
            calls.clear()
            _equal(model(x, cache = cache), out)
            for actual, state in zip(cache.state, states):
                _equal(actual, state)
            assert len(calls) == (1 if x.shape[1] == 1 else 0)


def test_unsupported_geometry_uses_native(monkeypatch):
    monkeypatch.setattr(decode, "_decode_conv", lambda *args: pytest.fail("unsupported window reached Metal fusion"))
    wide, model = _model(9), _model()
    with decode.fused_decode_conv_silu(nn.Sequential(wide, model)):
        state, step = mx.zeros((2, 3, 128), mx.bfloat16), mx.random.normal((2, 1, 128)).astype(mx.bfloat16)
        _equal(wide(step[..., :32]), native.Qwen3_5GatedDeltaNet.__call__(wide, step[..., :32]))
        for args in [(state, step, 1, False), (state, step.astype(mx.float16), 1, True), (state[:, 1:], step, 1, True)]:
            window, qkv = model._unsloth_decode_conv(*args)
            assert qkv is None
            _equal(window, mx.concatenate(args[:2], axis = 1))


def test_training_and_missing_kernel_keep_native(monkeypatch):
    model = _model()
    model.train()
    with decode.fused_decode_conv_silu(model):
        assert type(model) is native.Qwen3_5GatedDeltaNet
    model.eval()
    monkeypatch.setattr(decode, "_decode_conv_kernel", lambda: None)
    with decode.fused_decode_conv_silu(model):
        assert type(model) is native.Qwen3_5GatedDeltaNet


def test_live_globals_and_changed_arithmetic_keep_native(monkeypatch):
    model = _model()
    x = mx.random.normal((2, 1, 32)).astype(mx.bfloat16)
    with decode.fused_decode_conv_silu(model):
        update = native.gated_delta_update
        calls = []
        def observed(*args, **kwargs):
            calls.append(None)
            return update(*args, **kwargs)
        monkeypatch.setattr(native, "gated_delta_update", observed)
        mx.eval(model(x))
        assert calls
        silu = nn.silu
        monkeypatch.setattr(nn, "silu", lambda value: silu(value) * 2)
        _equal(model(x), native.Qwen3_5GatedDeltaNet.__call__(model, x))
        monkeypatch.setattr(nn, "silu", silu)
        conv = native._qwen3_5_decode_depthwise_conv
        monkeypatch.setattr(native, "_qwen3_5_decode_depthwise_conv", lambda *args: conv(*args) * 2)
        _equal(model(x), native.Qwen3_5GatedDeltaNet.__call__(model, x))
    with decode.fused_decode_conv_silu(model):
        assert type(model) is native.Qwen3_5GatedDeltaNet


@pytest.mark.parametrize("drift", ["replaced", "globals"])
def test_silu_drifting_inside_an_open_scope_keeps_native(drift, monkeypatch):
    model = _model(9 if drift == "replaced" else 4)
    x = mx.random.normal((2, 1, 32)).astype(mx.bfloat16)
    plain = getattr(nn.silu, "__wrapped__", nn.silu)
    twin = lambda: FunctionType(plain.__code__, dict(plain.__globals__), plain.__name__)
    captured = twin()
    monkeypatch.setattr(nn, "silu", captured)
    with decode.fused_decode_conv_silu(model):
        if drift == "replaced":
            monkeypatch.setattr(nn, "silu", twin())  # same source, so a later scope entry still finds it held
            with decode.fused_decode_conv_silu(_model(9)):
                pass
        captured.__globals__["mx"] = SimpleNamespace(sigmoid = lambda value: mx.sigmoid(value) * 2)
        _equal(model(x), native.Qwen3_5GatedDeltaNet.__call__(model, x))


@pytest.mark.parametrize("drift", ["hash", "body"])
def test_a_changed_silu_body_keeps_native(drift, monkeypatch):
    if drift == "hash":
        monkeypatch.setitem(decode._CONV_SILU_CONTRACT["mlx.nn"], "silu", "stale")
    else:
        monkeypatch.setattr(nn, "silu", nn.gelu)  # a real function whose body is not the pinned one
    model = _model()
    with decode.fused_decode_conv_silu(model):
        assert type(model) is native.Qwen3_5GatedDeltaNet


def test_inherited_contract_has_no_model_name_restriction():
    class OtherRecurrentBlock(native.Qwen3_5GatedDeltaNet):
        pass
    model = _model()
    model.__class__ = OtherRecurrentBlock
    with decode.fused_decode_conv_silu(model):
        assert type(model) is not OtherRecurrentBlock
    assert type(model) is OtherRecurrentBlock

def test_overlapping_scopes_keep_the_module_fused_until_the_last_exit():
    # Generation enters this beside fused_moe_gate_up and Studio enters it again per
    # request, so two scopes own the same module. Without a count the first exit
    # restores the native class and the still-open scope silently loses the kernel.
    model = _model()
    root = nn.Sequential(model)
    root.eval()
    outer = decode.fused_decode_conv_silu(root)
    outer.__enter__()
    fused = type(model)
    assert fused is not native.Qwen3_5GatedDeltaNet
    inner = decode.fused_decode_conv_silu(root)
    inner.__enter__()
    try:
        outer.__exit__(None, None, None)
        assert type(model) is fused
    finally:
        inner.__exit__(None, None, None)
    assert type(model) is native.Qwen3_5GatedDeltaNet
    # Both stores: nn.Module.__setattr__ routes only mx.array/dict/list/tuple into the
    # mapping and pops the key there for anything else, so the count and the two class
    # references live in __dict__. Assert the mapping too, so a value that later starts
    # landing there cannot leak past the exit unnoticed.
    leftover = [key for key in (*model.__dict__, *dict.keys(model))
                if key.startswith("_unsloth_decode")]
    assert leftover == []


def test_tests_ahead_of_the_decode_branch_still_decide(monkeypatch, tmp_path):
    # mlx-vlm 0.6.0-0.6.15 reach the decode branch through an elif behind a verify sink.
    source = textwrap.dedent(inspect.getsource(native.Qwen3_5GatedDeltaNet.__call__)).replace(
        "cache: Optional[Any] = None,", "cache: Optional[Any] = None, sink = None,", 1).replace(
        "    if (\n        S == 1", "    if sink is not None:\n        conv_out = nn.silu(self.conv1d(conv_input))"
        "\n    elif (\n        S == 1", 1)
    (tmp_path / "guarded_decode.py").write_text(
        f"import {native.__name__} as base\nglobals().update((k, v) for k, v in vars(base).items() if k[:2] != '__')"
        f"\nclass Guarded(Qwen3_5GatedDeltaNet):\n{textwrap.indent(source, '    ')}")
    monkeypatch.syspath_prepend(str(tmp_path))
    model, calls, fused = _model(), [], decode._decode_conv
    model.__class__ = guarded = importlib.import_module("guarded_decode").Guarded
    monkeypatch.setattr(decode, "_decode_conv", lambda *args: calls.append(None) or fused(*args))
    x = mx.random.normal((2, 1, 32)).astype(mx.bfloat16)
    with decode.fused_decode_conv_silu(model):
        for sink, launches_so_far in ((None, 1), ([], 1)):
            _equal(model(x, sink = sink), guarded.__call__(model, x, sink = sink))
            assert len(calls) == launches_so_far
