# SPDX-License-Identifier: AGPL-3.0-only
from types import SimpleNamespace

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
def test_kernel_preserves_native_reduction_and_rounding(dtype):
    for width in range(2, 9):
        for channels in (2, 31, 128):
            mx.random.seed(width + channels)
            x = (mx.random.normal((2, width, channels)) * 10).astype(dtype)
            weight = mx.random.normal((channels, width)).T
            _equal(decode._decode_conv_silu(x, weight), nn.silu(native._qwen3_5_decode_depthwise_conv(x, weight)))


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
            for model in models:
                for factor in (2, .25):
                    model.conv1d.weight = model.conv1d.weight * factor
                    expected = model._causal_conv1d_decode(x)
                    assert type(model)._causal_conv1d_decode is native.Qwen3_5GatedDeltaNet._causal_conv1d_decode
                    _equal(model._unsloth_decode_conv_silu(x), nn.silu(expected))
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
    fused = decode._decode_conv_silu
    calls = []
    def observed(*args):
        calls.append(None)
        return fused(*args)
    monkeypatch.setattr(decode, "_decode_conv_silu", observed)
    with decode.fused_decode_conv_silu(model):
        for x, (out, states) in zip(samples, expected):
            calls.clear()
            _equal(model(x, cache = cache), out)
            for actual, state in zip(cache.state, states):
                _equal(actual, state)
            assert len(calls) == (1 if x.shape[1] == 1 else 0)


def test_unsupported_geometry_uses_native(monkeypatch):
    model = _model(9)
    def unexpected(*args):
        pytest.fail("unsupported reduction reached Metal fusion")
    monkeypatch.setattr(decode, "_decode_conv_silu", unexpected)
    with decode.fused_decode_conv_silu(model):
        for shape, dtype in [((2, 9, 128), mx.bfloat16), ((2, 4, 1), mx.float16), ((2, 4, 128), mx.float32)]:
            x = mx.random.normal(shape).astype(dtype)
            weight = mx.random.normal(shape[1:])
            _equal(model._unsloth_apply_conv_silu(x, weight), nn.silu(native._qwen3_5_decode_depthwise_conv(x, weight)))


def test_training_and_missing_kernel_keep_native(monkeypatch):
    model = _model()
    model.train()
    with decode.fused_decode_conv_silu(model):
        assert type(model) is native.Qwen3_5GatedDeltaNet
    model.eval()
    monkeypatch.setattr(decode, "_decode_conv_silu_kernel", lambda: None)
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
        conv = native._qwen3_5_decode_depthwise_conv
        monkeypatch.setattr(native, "_qwen3_5_decode_depthwise_conv", lambda *args: conv(*args) * 2)
        _equal(model(x), native.Qwen3_5GatedDeltaNet.__call__(model, x))
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
