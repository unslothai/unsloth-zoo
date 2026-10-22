# SPDX-License-Identifier: AGPL-3.0-only
import ast
import contextlib
import functools
import importlib
import inspect
import json
import math
import os
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


def test_reentering_the_scope_does_not_reread_the_bodies(monkeypatch):
    model = _model()
    with decode.fused_decode_conv_silu(model):
        pass
    monkeypatch.setattr(decode, "_source_expression", lambda node: pytest.fail("read again"))
    with decode.fused_decode_conv_silu(model):
        assert type(model) is not native.Qwen3_5GatedDeltaNet


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
        "cache: Optional[Any] = None,", "cache: Optional[Any] = None, sink = None,", 1)
    keyword = "if" if "    if (\n        S == 1" in source else "elif"  # 0.6.0-0.6.15 already sit behind gdn_sink
    source = source.replace(f"    {keyword} (\n        S == 1", f"    {keyword} sink is not None:\n        conv_out = "
                            "nn.silu(self.conv1d(conv_input))\n    elif (\n        S == 1", 1)
    assert "if sink is not None:" in source
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


@pytest.mark.parametrize("anchor, inserted", [
    ("    conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)\n", "    conv_input[:, :, 0] = 0\n"),
    ("    q, k, v = [", "    self.key_dim = 1\n"),
])
def test_statements_the_fused_call_would_skip_keep_native(anchor, inserted):
    source = textwrap.dedent(inspect.getsource(native.Qwen3_5GatedDeltaNet.__call__))
    assert decode._decode_conv_sites(ast.parse(source).body[0]) is not None
    edited = source.replace(anchor, anchor + inserted if anchor.endswith("\n") else inserted + anchor, 1)
    assert edited != source and decode._decode_conv_sites(ast.parse(edited).body[0]) is None


def _text_model(vlm, hidden = 256):
    from mlx_lm.models import qwen3_5 as lm35
    from mlx_vlm.models.qwen3_5.config import TextConfig
    config = dict(
        model_type = "qwen3_5", hidden_size = hidden, intermediate_size = 128, num_hidden_layers = 4,
        num_attention_heads = 2, rms_norm_eps = 1e-6, vocab_size = 64, num_key_value_heads = 1,
        max_position_embeddings = 512, linear_num_value_heads = 2, linear_num_key_heads = 1,
        linear_key_head_dim = 32, linear_value_head_dim = 32, linear_conv_kernel_dim = 4, head_dim = 32,
        rope_parameters = {"type": "default", "mrope_section": [11, 11, 10], "rope_theta": 100000,
                           "partial_rotary_factor": 0.25},
    )
    outer = native.LanguageModel(TextConfig(**config)) if vlm else lm35.TextModel(lm35.TextModelArgs(**config))
    outer.set_dtype(mx.bfloat16)
    outer.eval()
    for _, module in outer.named_modules():
        if type(module) is nn.RMSNorm:
            module.weight = (1 + mx.random.normal(module.weight.shape) * 0.3).astype(mx.bfloat16)
    return outer


@pytest.mark.parametrize("width", [3, 1030, 8193])
def test_add_rms_norm_partial_rows_bitwise(width):
    x, r = (mx.random.normal((3, width)).astype(mx.bfloat16) for _ in range(2))
    w = mx.random.normal((width,)).astype(mx.bfloat16)
    h, out = decode._fused_add_rms_norm(x[1:], r[1:], w, 1e-6)
    _equal(h, x[1:] + r[1:])
    _equal(out, mx.fast.rms_norm(x[1:] + r[1:], w, 1e-6))


@pytest.mark.parametrize("hidden", [256, 4104])
@pytest.mark.parametrize("vlm", [False, True], ids = ["lm", "vlm"])
def test_prenorm_layers_hand_the_normalized_residual_on_bitwise(vlm, hidden, monkeypatch):
    outer = _text_model(vlm, hidden)
    model = outer.model
    layers, base = model.layers, type(model.layers[0])
    assert decode._add_norm_verified(mx.bfloat16, hidden)
    launches, fused = [], decode._fused_add_rms_norm
    monkeypatch.setattr(decode, "_fused_add_rms_norm", lambda *args: launches.append(None) or fused(*args))

    # mlx-vlm 0.7.4's Qwen3.5 decode reads memory it never wrote from two rows on, so its batched
    # output depends on the allocator's state and cannot be a bitwise reference.
    rows = 1 if vlm else 3

    def generate():
        cache = outer.make_cache()
        steps = [mx.random.randint(0, 64, (rows, 5), key = mx.random.key(1)),
                 *(mx.random.randint(0, 64, (rows, 1), key = mx.random.key(i)) for i in range(2, 5))]
        return [model(tokens, cache = cache) for tokens in steps]

    expected = generate()
    # as generation_mode enters them: the existing scope leaves these layers to the handoff
    with decode.fused_residual_norm(model), decode.fused_residual_norm_handoff(model):
        assert all(type(layer).__bases__ == (base,) for layer in layers)
        for actual, native_out in zip(generate(), expected):
            _equal(actual, native_out)
        # Per forward: every layer's residual add and all but the last layer's output handoff.
        assert len(launches) == 4 * (2 * len(layers) - 1)
        assert all(layer._unsloth_handoff_in.value is None for layer in layers[1:])
    assert all(type(layer) is base for layer in layers)
    assert not any(key.startswith("_unsloth_handoff") for layer in layers for key in vars(layer))


@pytest.mark.parametrize("change", ["none", "other_input", "weight", "eps", "rebound"])
def test_prenorm_handoff_is_taken_only_for_the_residual_it_normalized(change, monkeypatch):
    model = _text_model(False).model
    first, second = model.layers[:2]
    base = type(second)
    x = mx.random.normal((1, 2, 256)).astype(mx.bfloat16)
    with decode.fused_residual_norm_handoff(model):
        out = first(x)
        residual, normed, _, _ = second._unsloth_handoff_in.value
        assert residual is out
        if change == "none":
            assert second._unsloth_take_norm(second.input_layernorm, out) is normed
            return
        if change == "other_input":
            out = out + 1
        elif change == "weight":
            second.input_layernorm.weight = second.input_layernorm.weight * 3
        elif change == "eps":
            second.input_layernorm.eps = 0.5
        else:
            original = nn.RMSNorm.__call__
            monkeypatch.setattr(nn.RMSNorm, "__call__", lambda self, value: original(self, value) * 0.5)
        _equal(second(out), base.__call__(second, out))


def test_overlapping_handoff_scopes_keep_the_layers_fused_until_the_last_exit():
    # model.generate enters the scope outside generation_mode's lock, so two calls can exit out of order.
    outer = _text_model(False)
    model = outer.model
    layers, base = model.layers, type(model.layers[0])
    tokens = mx.random.randint(0, 64, (1, 3))
    expected = outer(tokens)
    first, second = decode.fused_residual_norm_handoff(model), decode.fused_residual_norm_handoff(model)
    first.__enter__()
    fused = type(layers[0])
    second.__enter__()
    first.__exit__(None, None, None)
    assert all(type(layer) is fused for layer in layers)
    assert all("_unsloth_handoff_in" in vars(layer) for layer in layers[1:])
    _equal(outer(tokens), expected)
    second.__exit__(None, None, None)
    assert all(type(layer) is base for layer in layers)
    assert not any(key.startswith("_unsloth_handoff") for layer in layers for key in vars(layer))


class _NormLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = nn.RMSNorm(256), nn.RMSNorm(256)

    def loosen(self, value = None):
        self.post_attention_layernorm.eps = 0.5
        return value


class _RescaledLayer(_NormLayer):
    def __call__(self, x):
        r = self.input_layernorm(x)
        h = x + r
        if x.ndim:
            h = h * 2
        return h + self.post_attention_layernorm(h)


class _InterveningLayer(_NormLayer):
    def __call__(self, x):
        r = self.input_layernorm(x)
        h = x + r
        self.post_attention_layernorm.eps = 0.5
        return h + self.post_attention_layernorm(h)


class _NestedLayer(_NormLayer):
    def __call__(self, x):
        r = self.input_layernorm(x)
        h = x + r
        if x.ndim:
            h = h + 1
            r = self.post_attention_layernorm(h)
        return h + r


class _WalrusLayer(_NormLayer):
    def __call__(self, x):
        r = self.input_layernorm(x)
        h = x + r
        r = mx.add((h := h + 1), self.post_attention_layernorm(h))
        return h + r


class _ShadowingLayer(_NormLayer):
    def __call__(self, x):
        r = self.input_layernorm(x)
        h = x + r
        r = [self.post_attention_layernorm(h) for h in (x, r)][0]
        return h + r


class _EarlierCallLayer(_NormLayer):
    def __call__(self, x):
        r = self.input_layernorm(x)
        h = x + r
        r = mx.add(self.loosen(r), self.post_attention_layernorm(h))
        return h + r


@pytest.mark.parametrize("layer", [_RescaledLayer, _InterveningLayer, _NestedLayer, _WalrusLayer, _ShadowingLayer, _EarlierCallLayer])
def test_a_norm_that_cannot_move_up_to_its_addition_keeps_native(layer):
    root = nn.Sequential(layer(), layer())
    root.eval()
    with decode.fused_residual_norm_handoff(root):
        assert all(type(module) is layer for module in root.layers)


def test_prenorm_layers_keep_native_when_the_kernel_disagrees(monkeypatch):
    model = _text_model(False).model
    fused = decode._fused_add_rms_norm
    monkeypatch.setattr(decode, "_fused_add_rms_norm", lambda *args: (fused(*args)[0], fused(*args)[1] * 2))
    monkeypatch.setattr(decode, "_ADD_NORM_VERDICTS", {})
    tokens = mx.random.randint(0, 64, (1, 3))
    expected = model(tokens)
    with decode.fused_residual_norm_handoff(model):
        _equal(model(tokens), expected)


from mlx_lm.models import cache as lm_cache, gated_delta as lm_gated_delta, qwen3_5 as lm_qwen3_5
try:
    from mlx_vlm.models.qwen3_5 import gated_delta as vlm_gated_delta
except ImportError:  # mlx-vlm < 0.6 layers call mlx-lm's module
    vlm_gated_delta = lm_gated_delta

# The scope leaves a module that packs on its own (mlx-vlm 0.7.8, mlx-lm 0.32) alone, so the route is exercised on one
# that does not, when one is installed.
_PACKED_SOURCES = [case for case in [(native, vlm_gated_delta, native.Qwen3_5GatedDeltaNet, native.ArraysCache),
                                     (lm_qwen3_5, lm_gated_delta, lm_qwen3_5.GatedDeltaNet, lm_cache.ArraysCache)]
                   if not hasattr(case[1], "_gated_delta_kernel_packed")]
_, routed_gated_delta, _ROUTED_LAYER, _ = _PACKED_SOURCES[0] if _PACKED_SOURCES else (None,) * 4
_needs_unpacked_source = pytest.mark.skipif(not _PACKED_SOURCES, reason = "every installed gated-delta module packs")


def _packable_model(layer = _ROUTED_LAYER):
    model = layer(SimpleNamespace(
        hidden_size = 64, linear_num_value_heads = 4, linear_num_key_heads = 2,
        linear_key_head_dim = 128, linear_value_head_dim = 24,
        linear_conv_kernel_dim = 4, rms_norm_eps = 1e-6,
    ))
    model.set_dtype(mx.bfloat16)
    model.eval()
    model.freeze()
    return model


def _recurrence(dtype, B = 2, T = 7, Hk = 2, Hv = 4, Dk = 128, Dv = 24, gate = None):
    q, k = (mx.random.normal((B, T, Hk, Dk)).astype(dtype) * Dk ** -0.5 for _ in range(2))
    v = mx.random.normal((B, T, Hv, Dv)).astype(dtype)
    g = mx.random.uniform(0.5, 1.0, gate or (B, T, Hv))
    beta = mx.random.uniform(0.0, 1.0, (B, T, Hv)).astype(dtype)
    return q, k, v, g, beta, mx.random.normal((B, Hv, Dv, Dk))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_packed_gated_delta_kernel_matches_the_generic_kernel_bitwise(dtype):
    for shape in [dict(B = 1, T = 2, Hk = 1, Hv = 1, Dv = 8), dict(), dict(B = 1, T = 300, Hk = 16, Hv = 32, Dv = 128)]:
        args = _recurrence(dtype, **shape)
        for actual, expected in zip(decode._packed_gated_delta(*args), vlm_gated_delta.gated_delta_kernel(*args)):
            _equal(actual, expected)


@pytest.mark.parametrize("consumer, source, layer, cache_type", _PACKED_SOURCES)
def test_packed_scope_preserves_prefill_and_decode_and_restores(consumer, source, layer, cache_type, monkeypatch):
    model = _packable_model(layer)
    samples = [mx.random.normal((2, n, 64)).astype(mx.bfloat16) for n in (5, 1, 3)]
    cache, expected = cache_type(size = 2), []
    for x in samples:
        out = model(x, cache = cache)
        mx.eval(out, cache.state)
        expected.append((out, list(cache.state)))
    stock, update, packed, calls = source.gated_delta_kernel, source.gated_delta_update, decode._packed_gated_delta, []
    patched = lambda *args, **kwargs: update(*args, **kwargs)  # the training patches rebind source and consumers alike
    monkeypatch.setattr(source, "gated_delta_update", patched)
    monkeypatch.setattr(consumer, "gated_delta_update", patched)
    monkeypatch.setattr(decode, "_packed_gated_delta", lambda *args: calls.append(None) or packed(*args))
    monkeypatch.setattr(decode, "_GATED_DELTA_VERDICTS", {})
    cache = cache_type(size = 2)
    with pytest.raises(RuntimeError, match = "cancel"):
        with decode.fused_decode_conv_silu(model), decode.fused_packed_gated_delta(model):
            route = source.gated_delta_kernel
            with decode.fused_packed_gated_delta(model):
                assert source.gated_delta_kernel is route is not stock
            assert source.gated_delta_kernel is route
            for x, (out, states) in zip(samples, expected):
                calls.clear()
                _equal(model(x, cache = cache), out)
                for actual, state in zip(cache.state, states):
                    _equal(actual, state)
                assert len(calls) == (x.shape[1] > 1)
            assert all(decode._GATED_DELTA_VERDICTS.values())
            raise RuntimeError("cancel")
    assert source.gated_delta_kernel is stock


@_needs_unpacked_source
def test_packed_route_keeps_other_shapes_and_a_differing_kernel_on_the_generic_kernel(monkeypatch, caplog):
    stock, packed = routed_gated_delta.gated_delta_kernel, decode._packed_gated_delta
    monkeypatch.setattr(decode, "_GATED_DELTA_VERDICTS", {})
    with decode.fused_packed_gated_delta(_packable_model()):
        route = routed_gated_delta.gated_delta_kernel
        args = _recurrence(mx.bfloat16)
        route(*args)
        monkeypatch.setattr(decode, "_packed_gated_delta", lambda *a: ("packed",))
        assert route(*args) == ("packed",)
        monkeypatch.setattr(decode, "_packed_gated_delta", lambda *a: pytest.fail("ineligible call was packed"))
        others = [_recurrence(mx.bfloat16, T = 1), _recurrence(mx.bfloat16, Dk = 64), _recurrence(mx.bfloat16, Dv = 12),
                  _recurrence(mx.bfloat16, gate = (2, 7, 4, 128)), (*args, mx.ones((2, 7), mx.bool_)),
                  (*args[:3], args[3].astype(mx.float16), *args[4:]), (*args[:5], args[5].astype(mx.float16))]
        for other in others:
            for actual, expected in zip(route(*other), stock(*other)):
                _equal(actual, expected)
        for dtype, y, state in [(mx.float16, 0, 1), (mx.float32, 1, 0)]:
            monkeypatch.setattr(decode, "_packed_gated_delta", lambda *a: (packed(*a)[0] + y, packed(*a)[1] + state))
            args = _recurrence(dtype)
            for _ in range(2):
                for actual, expected in zip(route(*args), stock(*args)):
                    _equal(actual, expected)
                monkeypatch.setattr(decode, "_packed_gated_delta", lambda *a: pytest.fail("rejected layout was packed"))
            assert caplog.text.count("packed gated-delta kernel differs") == y + 1
        monkeypatch.setattr(decode, "_packed_gated_delta", lambda *a: pytest.fail("drifted binding was packed"))
        monkeypatch.setattr(routed_gated_delta, "_gated_delta_kernel", routed_gated_delta._make_gated_delta_kernel())
        route(*_recurrence(mx.bfloat16))


@_needs_unpacked_source
@pytest.mark.parametrize("reason", ["switch", "upstream", "body", "kernel", "distributed", "model"])
def test_packed_scope_steps_aside(reason, monkeypatch, caplog):
    stock, model = routed_gated_delta.gated_delta_kernel, _packable_model()
    if reason == "switch":
        monkeypatch.setenv("UNSLOTH_MLX_PACKED_GATED_DELTA", "0")
    elif reason == "upstream":
        monkeypatch.setattr(routed_gated_delta, "_gated_delta_kernel_packed", object(), raising = False)
    elif reason == "body":
        monkeypatch.setattr(routed_gated_delta, "_make_gated_delta_kernel", lambda *args, **kwargs: None)
    elif reason == "kernel":
        monkeypatch.setattr(decode, "_packed_gated_delta_kernel", lambda: None)
    elif reason == "distributed":
        model._unsloth_mlx_distributed_parallel_mode = "tensor"
    else:
        model = _model().in_proj_qkv
    with caplog.at_level("WARNING"), decode.fused_packed_gated_delta(model):
        assert routed_gated_delta.gated_delta_kernel is stock
    assert ("differs from the version" in caplog.text) == (reason == "body")


@_needs_unpacked_source
def test_packed_scope_leaves_a_binding_it_no_longer_owns(monkeypatch):
    stock, model, foreign = routed_gated_delta.gated_delta_kernel, _packable_model(), lambda *args: None
    with decode.fused_packed_gated_delta(model):
        monkeypatch.setattr(routed_gated_delta, "gated_delta_kernel", foreign)
        with decode.fused_packed_gated_delta(model):
            pass
    assert routed_gated_delta.gated_delta_kernel is foreign
    monkeypatch.setattr(routed_gated_delta, "gated_delta_kernel", stock)
    with decode.fused_packed_gated_delta(model):
        assert routed_gated_delta.gated_delta_kernel is not stock
    assert routed_gated_delta.gated_delta_kernel is stock


def _step_model(key_heads = 16, value_heads = 32, dtype = mx.bfloat16, gates = (mx.float32, mx.bfloat16), spread = 1.0):
    model = native.Qwen3_5GatedDeltaNet(SimpleNamespace(
        hidden_size = 64, linear_num_value_heads = value_heads, linear_num_key_heads = key_heads,
        linear_key_head_dim = 128, linear_value_head_dim = 128, linear_conv_kernel_dim = 4, rms_norm_eps = 1e-6,
    ))
    model.set_dtype(dtype)
    model.conv1d.weight = model.conv1d.weight * spread  # wide convolutions reach the SiLU's exp rounding
    model.A_log = (mx.random.normal((value_heads,)) * 1.5).astype(gates[0])
    model.dt_bias = (mx.random.normal((value_heads,)) * 3).astype(gates[1])
    model.norm.weight = (1 + 0.3 * mx.random.normal((128,))).astype(dtype)
    model.eval()
    model.freeze()
    return model


def _steps(model, xs, cache = None, lengths = None):
    cache, out = cache or native.ArraysCache(size = 2), []
    for i, x in enumerate(xs):
        if i == 1 and lengths is not None:  # decode rows of a ragged batch inside a speculative block
            cache.lengths = mx.array(lengths)
            cache.start_speculation(8)
        out.append((model(x, cache = cache), *cache.state))
    if lengths is not None:
        out.append(tuple(a for record in cache._speculation["records"].values() for a in record if isinstance(a, mx.array)))
    return out


@pytest.mark.parametrize("heads, dtype, gates, rows, spread, lengths", [
    ((16, 32), mx.bfloat16, (mx.float32, mx.bfloat16), 1, 1.0, None),  # Qwen3.5-9B
    ((16, 32), mx.bfloat16, (mx.bfloat16, mx.bfloat16), 2, 4.0, [1, 0]),  # Qwen3.6-35B-A3B
    ((2, 6), mx.float16, (mx.float32, mx.float32), 3, 0.3, None),
])
def test_decode_step_matches_the_native_layer_bitwise(heads, dtype, gates, rows, spread, lengths, monkeypatch):
    model = _step_model(*heads, dtype, gates, spread)
    xs = [(mx.random.normal((rows, n, 64)) * 4).astype(dtype) for n in (5, 1, 1, 1, 2, 1)]
    expected = _steps(model, xs, lengths = lengths)
    fused, calls = decode._decode_step, []
    monkeypatch.setattr(decode, "_decode_step", lambda *args: calls.append(None) or fused(*args))
    monkeypatch.setattr(decode, "_DECODE_STEP_VERDICTS", {})
    with decode.fused_decode_conv_silu(model):
        for actual, stock in zip(_steps(model, xs, lengths = lengths), expected, strict = True):
            for x, y in zip(actual, stock):
                _equal(x, y)
    assert list(decode._DECODE_STEP_VERDICTS.values()) == [True] and len(calls) == 4  # one check, three steps


def test_decode_step_keeps_the_native_layer_when_it_differs_or_cannot_serve(monkeypatch, caplog):
    model, cache, stock = _step_model(2, 4), native.ArraysCache(size = 2), native.ArraysCache(size = 2)
    xs = [(mx.random.normal((2, n, 64)) * 4).astype(mx.bfloat16) for n in (3, 1, 1)]
    fused = decode._decode_step
    monkeypatch.setattr(decode, "_decode_step", lambda *args: [t + (i == 1) for i, t in enumerate(fused(*args))])
    monkeypatch.setattr(decode, "_DECODE_STEP_VERDICTS", {})
    with caplog.at_level("WARNING"), decode.fused_decode_conv_silu(model):
        native_steps = [(native.Qwen3_5GatedDeltaNet.__call__(model, x, cache = stock), *stock.state) for x in xs]
        for actual, expected in zip(_steps(model, xs, cache), native_steps):
            for x, y in zip(actual, expected):
                _equal(x, y)
    assert list(decode._DECODE_STEP_VERDICTS.values()) == [False] and "differs from the native layer" in caplog.text
    monkeypatch.setattr(decode, "_decode_step", lambda *args: pytest.fail("the decode step served this call"))
    monkeypatch.setattr(decode, "_DECODE_STEP_VERDICTS", {})
    project, projected = model.in_proj_z, []
    zeroed = lambda c: lambda *args: (lambda out: (out[0] * 0, out[1]))(type(c).update_recurrent(c, *args))
    # Instance attributes the kernel would bypass; set through __dict__ so undoing one leaves nothing behind.
    overrides = [[(model, "in_proj_z", lambda x: projected.append(None) or project(x).astype(mx.float32))],
                 [(c, "update_recurrent", zeroed(c)) for c in (cache, stock)],
                 [(model, "_normalize_qk", lambda q, k: (q * 2, k))]]
    for call, env, override in ((dict(mask = mx.array([[True], [False]])), "1", []), ({}, "0", []),
                                *[({}, "1", o) for o in overrides]):
        with monkeypatch.context() as scoped:
            scoped.setenv("UNSLOTH_MLX_FUSED_GDN_DECODE", env)
            for target, name, value in override:
                scoped.setitem(vars(target), name, value)
            with decode.fused_decode_conv_silu(model):
                _equal(model(xs[1], cache = cache, **call),
                       native.Qwen3_5GatedDeltaNet.__call__(model, xs[1], cache = stock, **call))
                for x, y in zip(cache.state, stock.state):
                    _equal(x, y)
    assert len(projected) == 2  # a projection the kernel cannot take is not run again on the way back


def test_decode_step_reaches_the_update_behind_the_training_wrapper(monkeypatch):
    original = vlm_gated_delta.gated_delta_update
    wrapper = lambda *args, **kwargs: original(*args, **kwargs)
    for module in (vlm_gated_delta, native):
        monkeypatch.setattr(module, "gated_delta_update", wrapper)
    monkeypatch.setattr(vlm_gated_delta, "_unsloth_gated_delta_original", original, raising = False)
    assert decode._decode_step_contract(native.Qwen3_5GatedDeltaNet) is None
    monkeypatch.setattr(vlm_gated_delta, "_unsloth_gated_delta_patched", True, raising = False)
    assert decode._decode_step_contract(native.Qwen3_5GatedDeltaNet)[1:3] == (wrapper, original)


def _attention_block(gqa, rows, keys, dtype = mx.bfloat16, heads = 8):
    # Keys and values as a KVCache returns them: a strided slice of a longer buffer.
    mx.random.seed(gqa * 100 + rows)
    buffer = mx.random.normal((2, 1, heads // gqa, keys + 256, 512)).astype(dtype)
    queries = (mx.random.normal((1, rows, heads, 512)) * 0.05).astype(dtype).transpose(0, 2, 1, 3)
    return queries, buffer[0, ..., :keys, :] * 0.5, buffer[1, ..., :keys, :]


def _mlx_vector_launch_fits(simds):
    # Before M3, MLX 0.32.4's unrolled head-dim-512 vector kernel holds 448 threads per threadgroup.
    return decode._gpu_generation() >= 15 or 32 * simds <= 448


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("gqa", [4, 8])
@pytest.mark.parametrize("rows", [2, 3, 5, 8])
def test_verify_attention_matches_mlx_vector_kernel_and_decode_rows_bitwise(gqa, rows, dtype, monkeypatch):
    queries, keys, values = _attention_block(gqa, rows, 4500, dtype)
    if gqa == 4:   # decode stays unfused there, so the kernel only serves it when asked
        assert decode._verify_attention(queries, keys, values, None, 1.0, "causal", None) is None
        monkeypatch.setenv("UNSLOTH_MLX_VERIFY_ATTENTION", "1")
    out = decode._verify_attention(queries, keys, values, None, 1.0, "causal", None)
    _equal(out, decode._decode_attention(queries, keys, values, 1.0))
    if gqa == 8:   # unpaired heads: exactly what each decode step computes
        horizon = lambda i: 4500 - rows + i + 1
        _equal(out, mx.concatenate([mx.fast.scaled_dot_product_attention(
            queries[:, :, i : i + 1], keys[..., : horizon(i), :], values[..., : horizon(i), :], scale = 1.0)
            for i in range(rows)], axis = 2))
    monkeypatch.setenv("MLX_SDPA_D512_MIN_KL", "0")   # MLX's own vector kernel over each launch's rows
    step = decode._launch_simds() // gqa
    if _mlx_vector_launch_fits(gqa * min(step, rows)):
        _equal(out, mx.concatenate([mx.fast.scaled_dot_product_attention(
            queries[:, :, i : i + step], keys[..., : 4500 - rows + min(i + step, rows), :],
            values[..., : 4500 - rows + min(i + step, rows), :], scale = 1.0, mask = "causal")
            for i in range(0, rows, step)], axis = 2))
    assert not mx.array_equal(out[:, :, :1], out[:, :, 1:2]).item()


def test_launches_before_m3_fit_their_threadgroup_limit(monkeypatch):
    monkeypatch.setattr(decode, "_gpu_generation", lambda: 13)
    launched, kernel = [], decode._vector_attention
    monkeypatch.setattr(decode, "_vector_attention", lambda *args: launched.append(args[0].shape[2]) or kernel(*args))
    queries, keys, values = _attention_block(8, 8, 4500)
    out = decode._verify_attention(queries, keys, values, None, 1.0, "causal", None)
    _equal(out, decode._decode_attention(queries, keys, values, 1.0))
    assert launched == [3, 3, 2]   # 32 lanes x 8 query heads x 3 rows = 768 of the 832 threads allowed


def _gemma4_attention(gqa):
    from mlx_vlm.models.gemma4 import language as gemma4
    from mlx_vlm.models.gemma4.config import TextConfig
    config = TextConfig(hidden_size = 64, num_hidden_layers = 2, num_attention_heads = 8, num_key_value_heads = 8 // gqa,
                        num_kv_shared_layers = 0, layer_types = ["sliding_attention", "full_attention"])
    layers = nn.Sequential(gemma4.Attention(config, 0), gemma4.Attention(config, 1))
    layers.set_dtype(mx.bfloat16)
    layers.eval()
    return gemma4, layers


def _filled(gemma4, keys, kv_heads):
    mx.random.seed(keys)
    cache = gemma4.KVCache()
    for _ in range(2):
        cache.update_and_fetch(*(mx.random.normal((1, kv_heads, keys // 2, 512)).astype(mx.bfloat16) for _ in range(2)))
    return cache


def _attend(gemma4, module, x, keys = 4096, mask = "causal", call = None):
    out = (call or module)(*([module] if call else []), x, mask = mask, cache = _filled(gemma4, keys, module.n_kv_heads))[0]
    mx.eval(out)
    return out


def test_verify_scope_serves_only_marked_global_blocks_and_restores(monkeypatch):
    gemma4, layers = _gemma4_attention(gqa = 8)
    sliding, full = layers.layers
    x = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)   # one launch on every GPU, M1 included
    launched, kernel = [], decode._vector_attention
    monkeypatch.setattr(decode, "_vector_attention", lambda *args: launched.append(args[0].shape[2]) or kernel(*args))
    native = lambda *args: _attend(gemma4, full, *args, call = gemma4.Attention.__call__)
    with monkeypatch.context() as patch:
        patch.setenv("MLX_SDPA_D512_MIN_KL", "0")   # the native call on MLX's vector kernel
        vector = native(x) if _mlx_vector_launch_fits(8 * 3) else None
    with decode.fused_verify_attention(layers):
        assert type(sliding) is gemma4.Attention and type(full) is not gemma4.Attention
        _equal(_attend(gemma4, full, x), native(x))
        with decode.verify_block_attention():
            served = _attend(gemma4, full, x)
            if vector is not None:
                _equal(served, vector)
            assert launched == [3]
            # A decode step, an unmasked block, and too few keys for MLX's two-pass kernel stay native.
            for args in [(x[:, :1],), (x, 4096, None), (x, 600)]:
                _equal(_attend(gemma4, full, *args), native(*args))
            full.train()
            _equal(_attend(gemma4, full, x), native(x))
            full.eval()
        assert launched == [3]
    assert type(full) is gemma4.Attention
    with pytest.raises(RuntimeError, match = "cancel"):
        with decode.fused_verify_attention(layers):
            with decode.fused_verify_attention(layers):
                pass
            assert type(full) is not gemma4.Attention   # the outer scope still holds it
            with decode.verify_block_attention():
                raise RuntimeError("cancel")
    assert type(full) is gemma4.Attention and not decode._VERIFY_STATE.attention


def test_verify_scope_keeps_native_when_the_first_launch_differs_from_decode(monkeypatch, caplog):
    gemma4, layers = _gemma4_attention(gqa = 8)
    full = layers.layers[1]
    x = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)
    monkeypatch.setattr(decode, "_VERIFY_ATTENTION_VERDICTS", {})
    monkeypatch.setattr(decode, "_decode_attention", lambda queries, *args: queries)
    expected = _attend(gemma4, full, x, call = gemma4.Attention.__call__)
    with decode.fused_verify_attention(layers), decode.verify_block_attention():
        _equal(_attend(gemma4, full, x), expected)
        monkeypatch.setattr(decode, "_vector_attention", lambda *args: pytest.fail("a refused shape launched again"))
        _equal(_attend(gemma4, full, x), expected)
    assert list(decode._VERIFY_ATTENTION_VERDICTS.values()) == [False] and "differs" in caplog.text


def test_a_launch_straddling_a_block_count_boundary_is_not_judged(monkeypatch):
    keys = next((n for n in (8194, 16386, 32770, 65538) if decode._vector_blocks(8, n - 3) != decode._vector_blocks(32, n)), None)
    if keys is None:
        pytest.skip("this GPU class interleaves keys alike on both sides of every boundary")
    for admitted in ({}, {(mx.bfloat16, 8, 4, decode._vector_blocks(32, keys)): True}):
        monkeypatch.setattr(decode, "_VERIFY_ATTENTION_VERDICTS", dict(admitted))
        monkeypatch.setattr(decode, "_vector_attention", lambda *args: pytest.fail("a straddling launch ran"))
        assert decode._verify_attention(*_attention_block(8, 4, keys), None, 1.0, "causal", None) is None
        assert decode._VERIFY_ATTENTION_VERDICTS == admitted


try:
    from mlx_vlm.models.qwen4_exp import language as qwen4
    from mlx_vlm.models.qwen4_exp.config import TextConfig as Qwen4TextConfig
except ImportError:  # mlx-vlm before qwen4_exp
    qwen4 = None
from unsloth_zoo.mlx import attention as mlx_attention, nax


def _qsa_layer():
    if qwen4 is None:
        pytest.skip("mlx-vlm without qwen4_exp")
    config = Qwen4TextConfig(
        model_type = "qwen4_exp_text", hidden_size = 64, num_hidden_layers = 4, num_attention_heads = 4,
        linear_num_value_heads = 2, linear_num_key_heads = 2, linear_key_head_dim = 16, linear_value_head_dim = 16,
        linear_conv_kernel_dim = 4, num_experts = 4, num_experts_per_tok = 2, shared_expert_intermediate_size = 16,
        moe_intermediate_size = 16, rms_norm_eps = 1e-6, vocab_size = 128, num_key_value_heads = 2,
        max_position_embeddings = 4096, hc_count = 2, hc_lowrank = 8, head_dim = 32, indexer_head_dim = 32,
        indexer_budget = 32, ngram_vocab_size_base = 100,
    )
    mx.random.seed(0)
    attention = qwen4.Qwen4ExpAttention(config)
    attention.set_dtype(mx.bfloat16)
    attention.eval()
    attention.freeze()
    return attention


def _qsa_run(attention):
    # Dense while the prefix fits the eight-block budget, then sparse prefill chunks and decode.
    cache, outs = qwen4.QSAKVCache(), []
    mx.random.seed(1)
    for n in (20, 30, 41, 1, 1, 1, 2):
        out = attention(mx.random.normal((1, n, 64)).astype(mx.bfloat16), cache = cache)
        mx.eval(out, cache.state)
        outs.append(out)
    return outs


def test_qsa_scope_preserves_attention_and_restores(monkeypatch):
    attention, native = _qsa_layer(), qwen4.Qwen4ExpQSAIndexer.select_from_projected
    expected, select, calls = _qsa_run(attention), mlx_attention.select_blocks, []
    monkeypatch.setattr(decode, "_QSA_VERDICTS", {})
    monkeypatch.setattr(mlx_attention, "select_blocks", lambda *args: calls.append(args[0].shape[2]) or select(*args))
    with pytest.raises(RuntimeError, match = "cancel"):
        with decode.fused_qsa_attention(attention):
            route = qwen4.Qwen4ExpQSAIndexer.select_from_projected
            with decode.fused_qsa_attention(attention):
                assert qwen4.Qwen4ExpQSAIndexer.select_from_projected is route is not native
            for actual, out in zip(_qsa_run(attention), expected):
                _equal(actual, out)
            assert calls == [30, 41, 1, 1, 1, 2] and all(decode._QSA_VERDICTS.values())
            raise RuntimeError("cancel")
    assert qwen4.Qwen4ExpQSAIndexer.select_from_projected is native


def test_qsa_scope_keeps_native_picks_when_the_kernel_disagrees(monkeypatch, caplog):
    attention = _qsa_layer()
    expected, select = _qsa_run(attention), mlx_attention.select_blocks
    monkeypatch.setattr(decode, "_QSA_VERDICTS", {})
    monkeypatch.setattr(mlx_attention, "select_blocks", lambda *args: mx.maximum(select(*args) - 1, -1))
    with caplog.at_level("WARNING"), decode.fused_qsa_attention(attention):
        for actual, out in zip(_qsa_run(attention), expected):
            _equal(actual, out)
    assert decode._QSA_VERDICTS and not any(decode._QSA_VERDICTS.values())
    assert caplog.text.count("picked other blocks than argpartition") == len(decode._QSA_VERDICTS)


@pytest.mark.parametrize("where", ["check", "synchronize"])
def test_qsa_selection_check_raises_a_failed_command_buffer(where, monkeypatch):
    attention, failed = _qsa_layer(), RuntimeError("[METAL] Command buffer execution failed: Internal Error")
    monkeypatch.setattr(decode, "_QSA_VERDICTS", {})

    def fail(*_, **__):
        raise failed

    if where == "check":
        monkeypatch.setattr(mx, "array_equal", fail)
    else:
        monkeypatch.setattr(mx, "array_equal", lambda *_, **__: mx.array(False))
        monkeypatch.setattr(mx, "synchronize", fail)
    with decode.fused_qsa_attention(attention), pytest.raises(RuntimeError) as raised:
        _qsa_run(attention)
    assert raised.value is failed and not decode._QSA_VERDICTS


@pytest.mark.parametrize("reason", ["switch", "body", "training", "distributed", "model"])
def test_qsa_scope_steps_aside(reason, monkeypatch, caplog):
    attention, native = _qsa_layer(), qwen4.Qwen4ExpQSAIndexer.select_from_projected
    if reason == "switch":
        monkeypatch.setenv("UNSLOTH_MLX_QSA_SELECT", "0")
    elif reason == "body":
        monkeypatch.setattr(qwen4.Qwen4ExpQSAIndexer, "select_from_projected", lambda *args: native(*args))
        native = qwen4.Qwen4ExpQSAIndexer.select_from_projected
    elif reason == "training":
        attention.train()
    elif reason == "distributed":
        attention._unsloth_mlx_distributed_parallel_mode = "tensor"
    else:
        attention = attention.q_proj
    with caplog.at_level("WARNING"), decode.fused_qsa_attention(attention):
        assert qwen4.Qwen4ExpQSAIndexer.select_from_projected is native
    assert ("differs from the version" in caplog.text) == (reason == "body")


def test_qsa_route_skips_an_indexer_in_training(monkeypatch):
    attention, calls = _qsa_layer(), []
    monkeypatch.setattr(mlx_attention, "select_blocks", lambda *args: calls.append(args) or None)
    with decode.fused_qsa_attention(attention):
        attention.indexer.train()
        _qsa_run(attention)
    assert calls == []


def test_qsa_route_skips_a_zero_block_budget(monkeypatch):
    attention, calls = _qsa_layer(), []
    attention.indexer.block_topk = 0
    expected = _qsa_run(attention)
    monkeypatch.setattr(mlx_attention, "select_blocks", lambda *args: calls.append(args) or None)
    with decode.fused_qsa_attention(attention):
        for actual, out in zip(_qsa_run(attention), expected):
            _equal(actual, out)
    assert calls == []


def test_qsa_scope_leaves_a_binding_it_no_longer_owns(monkeypatch):
    attention, foreign = _qsa_layer(), lambda *args: None
    native = qwen4.Qwen4ExpQSAIndexer.select_from_projected
    # Recorded first so teardown restores the native method, not the route open below.
    monkeypatch.setattr(qwen4.Qwen4ExpQSAIndexer, "select_from_projected", native)
    with decode.fused_qsa_attention(attention):
        monkeypatch.setattr(qwen4.Qwen4ExpQSAIndexer, "select_from_projected", foreign)
    assert qwen4.Qwen4ExpQSAIndexer.select_from_projected is foreign
    monkeypatch.setattr(qwen4.Qwen4ExpQSAIndexer, "select_from_projected", native)
    with decode.fused_qsa_attention(attention):
        assert qwen4.Qwen4ExpQSAIndexer.select_from_projected is not native
    assert qwen4.Qwen4ExpQSAIndexer.select_from_projected is native


def _qsa_prefill_scope(monkeypatch, calls):
    from mlx_vlm.models.qwen4_exp import qsa_kernel
    monkeypatch.delenv("UNSLOTH_MLX_FAST_QSA", raising = False)
    monkeypatch.setattr(nax, "kernel_probe_passed", lambda *args: True)
    monkeypatch.setattr(mlx_attention, "fast_prefill_supported", lambda queries, *args: queries.shape[2] >= 20)
    monkeypatch.setattr(mlx_attention, "fast_prefill_attention", lambda q, k, v, blocks, ends, scale: calls.append(
        q.shape[2]) or qsa_kernel.qsa_sparse_attention(q, k, v, blocks, ends, scale = scale, block_size = 4))


def test_qsa_prefill_route_takes_indexed_prefill_and_restores(monkeypatch):
    attention, native, calls = _qsa_layer(), qwen4.dispatch_qsa_attention, []
    expected = _qsa_run(attention)
    _qsa_prefill_scope(monkeypatch, calls)
    q, kv = mx.zeros((1, 4, 20, 32), mx.bfloat16), mx.zeros((1, 2, 20, 32), mx.bfloat16)
    picks, ends = mx.zeros((1, 20, 8), mx.int32), mx.arange(1, 21, dtype = mx.int32)[None]
    with decode.inference_scopes(attention):
        assert qwen4.dispatch_qsa_attention is not native
        for actual, out in zip(_qsa_run(attention), expected):
            _equal(actual, out)
        qwen4.dispatch_qsa_attention(q, kv, kv, picks, ends, cache = None, scale = 1.0, block_size = 4, causal = False,
                                     mask = None, mask_factory = lambda: mx.ones((20, 20), mx.bool_))
    assert calls == [20, 30, 41] and qwen4.dispatch_qsa_attention is native


def test_qsa_prefill_route_takes_the_simdgroup_kernel_without_tensor_units(monkeypatch):
    from mlx_vlm.models.qwen4_exp import qsa_kernel
    attention, calls, probe = _qsa_layer(), [], nax.kernel_probe_passed
    _qsa_prefill_scope(monkeypatch, [])
    monkeypatch.setattr(nax, "kernel_probe_passed", probe)
    monkeypatch.setattr(nax, "nax_available", lambda: False)
    monkeypatch.setattr(nax, "_PROBES", {})
    monkeypatch.setattr(nax, "_stored_probes", lambda: {})
    monkeypatch.setattr(nax, "_store_probe", lambda *args: None)
    monkeypatch.setattr(nax, "_run_probe", lambda module, function: "")
    monkeypatch.setattr(mlx_attention, "simd_prefill_supported", lambda queries, *args: queries.shape[2] >= 20)
    monkeypatch.setattr(mlx_attention, "simd_prefill_attention", lambda q, k, v, blocks, ends, scale: calls.append(
        q.shape[2]) or qsa_kernel.qsa_sparse_attention(q, k, v, blocks, ends, scale = scale, block_size = 4))
    expected = _qsa_run(attention)
    with decode.inference_scopes(attention):
        for actual, out in zip(_qsa_run(attention), expected):
            _equal(actual, out)
    assert calls == [20, 30, 41]


def test_qsa_prefill_route_keeps_working_after_training(monkeypatch):
    from unsloth_zoo.mlx import utils
    attention, calls = _qsa_layer(), []
    native = getattr(qwen4.dispatch_qsa_attention, "_unsloth_sparse_attention_original", qwen4.dispatch_qsa_attention)
    # What the trainer leaves bound for the rest of the process.
    trained = mlx_attention._training_wrapper(native, mlx_attention._qsa_attention_training_over)
    monkeypatch.setattr(qwen4, "dispatch_qsa_attention", trained)
    _qsa_prefill_scope(monkeypatch, calls)
    expected = _qsa_run(attention)
    with decode.inference_scopes(attention):
        for actual, out in zip(_qsa_run(attention), expected):
            _equal(actual, out)
        assert calls == [20, 30, 41]
        token = utils._MLX_TRAINING_ACTIVE_DEPTH.set(1)
        try:
            _qsa_run(attention)
        finally:
            utils._MLX_TRAINING_ACTIVE_DEPTH.reset(token)
    assert calls == [20, 30, 41] and qwen4.dispatch_qsa_attention is trained


@pytest.mark.parametrize("reason", ["switch", "probe", "body", "wrapped"])
def test_qsa_prefill_route_steps_aside(reason, monkeypatch):
    attention, calls = _qsa_layer(), []
    _qsa_prefill_scope(monkeypatch, calls)
    if reason == "wrapped":  # around the training wrapper, which functools.wraps copies the marker from
        native = getattr(qwen4.dispatch_qsa_attention, "_unsloth_sparse_attention_original", qwen4.dispatch_qsa_attention)
        trained = mlx_attention._training_wrapper(native, mlx_attention._qsa_attention_training_over)
        monkeypatch.setattr(qwen4, "dispatch_qsa_attention", functools.wraps(trained)(lambda *args, **kwargs: None))
    elif reason == "switch":
        monkeypatch.setenv("UNSLOTH_MLX_FAST_QSA", "0")
    elif reason == "probe":
        monkeypatch.setattr(nax, "kernel_probe_passed", lambda *args: False)
    else:
        monkeypatch.setattr(qwen4, "dispatch_qsa_attention", lambda *args, **kwargs: None)
    native = qwen4.dispatch_qsa_attention
    with decode.inference_scopes(attention):
        assert qwen4.dispatch_qsa_attention is native


def _qsa_attention():
    if qwen4 is None:
        pytest.skip("mlx-vlm without qwen4_exp")
    mx.random.seed(5)
    attention = qwen4.Qwen4ExpAttention(Qwen4TextConfig(
        model_type = "qwen4_exp", hidden_size = 64, num_hidden_layers = 4, num_attention_heads = 2, num_key_value_heads = 1,
        linear_num_value_heads = 1, linear_num_key_heads = 1, linear_key_head_dim = 16, linear_value_head_dim = 16,
        linear_conv_kernel_dim = 4, num_experts = 2, num_experts_per_tok = 1, shared_expert_intermediate_size = 16,
        moe_intermediate_size = 16, rms_norm_eps = 1e-6, vocab_size = 64, max_position_embeddings = 4096, head_dim = 64,
        indexer_n_heads = 2, indexer_head_dim = 32, indexer_budget = 16, rope_parameters = {
            "type": "default", "mrope_section": [3, 3, 2], "rope_theta": 10000, "partial_rotary_factor": 0.25}))
    attention.set_dtype(mx.bfloat16)
    attention.eval()
    return attention


def _speculative_rounds(attention, cache):
    # Rejections that do and do not cross a pooled-block boundary, and a full acceptance. Each round
    # brings new rows, as a corrected token replaces a rejected draft.
    x = mx.random.normal((1, 81, 64), key = mx.random.key(9)).astype(mx.bfloat16)
    steps = [attention(x[:, :61], cache = cache), cache.index_block_keys]
    for start, reject in zip(range(61, 81, 4), (3, 1, 0, 2, 3)):
        steps += [attention(x[:, start:start + 4], cache = cache), cache.index_block_keys]
        cache.trim(reject)
        steps.append(cache.index_block_keys)
    return steps


@pytest.mark.parametrize("cache, args", [("QSAKVCache", {}), ("QSAQuantizedKVCache", {"group_size": 32, "bits": 8}),
                                         ("BatchQSAKVCache", {"left_padding": [0]})])
@pytest.mark.parametrize("scopes", [("exact_qsa_trim",), ("fused_qsa_attention", "exact_qsa_trim"),
                                    ("exact_qsa_trim", "fused_qsa_attention")])
def test_qsa_trim_keeps_the_blocks_a_repool_rebuilds_bitwise(cache, args, scopes):
    attention, select = _qsa_attention(), qwen4.Qwen4ExpQSAIndexer.select_from_projected
    stock = _speculative_rounds(attention, getattr(qwen4, cache)(**args))
    with contextlib.ExitStack() as stack:
        for scope in scopes:
            stack.enter_context(getattr(decode, scope)(attention))
        assert (qwen4.Qwen4ExpQSAIndexer.select_from_projected is select) == (len(scopes) == 1)
        kept = _speculative_rounds(attention, getattr(qwen4, cache)(**args))
    for index, (a, b) in enumerate(zip(stock, kept)):
        if index % 3 == 1 and index > 1:  # just after a trim: stock drops every block, the scope keeps a prefix
            assert a is None and b.shape[2] > 0
        else:
            _equal(a, b)


def test_qsa_trim_scope_leaves_other_models_drifted_bodies_and_foreign_bindings(monkeypatch):
    attention, stock = _qsa_attention(), qwen4.QSAKVCache.trim
    monkeypatch.setattr(qwen4.QSAKVCache, "trim", stock)
    with decode.exact_qsa_trim(_model()):
        assert qwen4.QSAKVCache.trim is stock
    with decode.exact_qsa_trim(attention):
        assert qwen4.QSAKVCache.trim is not stock
        with decode.exact_qsa_trim(attention):
            pass
        assert qwen4.QSAKVCache.trim is not stock
    assert qwen4.QSAKVCache.trim is stock
    with decode.exact_qsa_trim(attention):
        qwen4.QSAKVCache.trim = foreign = lambda self, n: n
    assert qwen4.QSAKVCache.trim is foreign
    qwen4.QSAKVCache.trim = stock
    indexer = qwen4.Qwen4ExpQSAIndexer
    monkeypatch.setattr(indexer, "select_from_projected", indexer.select_from_projected)  # teardown restores native
    for wrap_first in (True, False):  # functools.wraps carries the fused route's mark onto a foreign wrapper
        with decode.fused_qsa_attention(attention):
            route = indexer.select_from_projected
            wrapper = functools.wraps(route)(lambda *args: route(*args))
            indexer.select_from_projected = wrapper if wrap_first else route
            with decode.exact_qsa_trim(attention):
                indexer.select_from_projected = wrapper
                assert _speculative_rounds(attention, qwen4.QSAKVCache())[4] is None
            indexer.select_from_projected = route
    contract ={name: dict(names) for name, names in decode._QSA_TRIM_CONTRACT.items()}
    contract[decode._QSA_LANGUAGE]["Qwen4ExpQSAIndexer.select_from_projected"] = "0" * 16
    monkeypatch.setattr(decode, "_QSA_TRIM_CONTRACT", contract)
    with decode.exact_qsa_trim(attention):
        assert qwen4.QSAKVCache.trim is stock


def _dilated_conv(dtype = mx.bfloat16, channels = 320, dilation = 3, bias = False, groups = None, out = None):
    mx.random.seed(channels + dilation)
    conv = nn.Conv1d(channels, out or channels, 4, dilation = dilation, groups = groups or channels, bias = bias)
    if bias:
        conv.bias = mx.random.normal(conv.bias.shape)
    conv.set_dtype(dtype)
    conv.eval()
    return conv


@pytest.mark.parametrize("dtype, dilation, bias", [(mx.bfloat16, 3, False), (mx.float16, 2, True),
                                                   (mx.float32, 5, False)])
def test_dilated_depthwise_conv_matches_mx_conv1d_bitwise_and_restores(dtype, dilation, bias):
    conv = _dilated_conv(dtype, dilation = dilation, bias = bias)
    inputs = [(mx.random.normal((batch, rows + 3 * dilation, 320), key = mx.random.key(rows)) * 8).astype(dtype)
              for batch, rows in ((1, 1), (2, 5), (1, 300))]
    expected = [conv(x) for x in inputs]
    with decode.exact_dilated_depthwise_conv(conv):
        assert type(conv) is not nn.Conv1d
        for x, reference in zip(inputs, expected):
            _equal(conv(x), reference)
    assert type(conv) is nn.Conv1d


def test_dilated_depthwise_conv_scope_leaves_other_convs_and_a_differing_kernel(monkeypatch, caplog):
    # Two output channels per group still has one input channel per weight row.
    for conv in (_dilated_conv(dilation = 1), _dilated_conv(channels = 32, out = 64), _dilated_conv().train()):
        with decode.exact_dilated_depthwise_conv(conv):
            assert type(conv) is nn.Conv1d
    conv = _dilated_conv(channels = 96)
    with decode.exact_dilated_depthwise_conv(conv):
        with decode.exact_dilated_depthwise_conv(conv):
            pass
        assert type(conv) is not nn.Conv1d
    assert type(conv) is nn.Conv1d
    monkeypatch.setattr(decode, "_depthwise_conv", lambda x, weight, dilation: mx.zeros(
        (x.shape[0], x.shape[1] - 3 * dilation, x.shape[2]), x.dtype))
    monkeypatch.setattr(decode, "_depthwise_conv_verified", functools.cache(decode._depthwise_conv_verified.__wrapped__))
    with monkeypatch.context() as failing:  # a failed command buffer reads as a mismatch until the next synchronize
        failing.setattr(mx, "synchronize", lambda: (_ for _ in ()).throw(RuntimeError(decode._FAILED_BUFFER)))
        with pytest.raises(RuntimeError, match = "Command buffer"):
            with decode.exact_dilated_depthwise_conv(_dilated_conv(channels = 64)):
                pytest.fail("the generation ran past a failed command buffer")
    assert decode._depthwise_conv_verified.cache_info().currsize == 0 and "differs" not in caplog.text
    with caplog.at_level("WARNING"), decode.exact_dilated_depthwise_conv(_dilated_conv(channels = 64)) as conv:
        assert type(conv) is nn.Conv1d
    assert "differs from mx.conv1d" in caplog.text


def _ple_table(path, cache_rows = 0, layout = "ranges"):
    storage = pytest.importorskip("mlx_vlm.models.qwen4_exp.ple_storage")
    prefix, weights = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards", {}
    rng = np.random.default_rng(0)
    with mx.stream(mx.cpu):
        for shard, rows in enumerate((300, 500)):
            weights[f"{prefix}.{shard}.weight"] = mx.array(rng.integers(0, 1 << 32, (rows, 8), dtype = np.uint32))
            for name in ("scales", "biases"):
                weights[f"{prefix}.{shard}.{name}"] = mx.array(rng.standard_normal((rows, 2), np.float32)).astype(mx.bfloat16)
        mx.save_safetensors(str(path / "model.safetensors"), weights)
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(weights, "model.safetensors")}))
    (path / "config.json").write_text(json.dumps({"quantization": {"bits": 4, "group_size": 32, "mode": "affine"}}))
    if layout == "ranges":
        storage.build_quantized_ple_manifest(path, path / "ple.json")
    else:
        storage.materialize_interleaved_ple_store(path, path / "ple.json")
    table = storage.QuantizedMMapNGramEmbedding(path / "ple.json", cache_rows = cache_rows)
    return table, SimpleNamespace(named_modules = lambda: [("ple", SimpleNamespace(ngram_embedding = table))])


def _row_bytes(table, rows):
    if table._interleaved is not None:
        return sorted(table._interleaved[row].tobytes() for row in rows)
    return sorted(arrays[name][row - start].tobytes() for row in rows for start, end, arrays in table._shards
                  if start <= row < end for name in ("weight", "scales", "biases"))


def _recorded_reads(monkeypatch):
    read, native = [], os.preadv

    def preadv(descriptor, buffers, offset):
        read.append(bytes(buffers[0][:(count := native(descriptor, buffers, offset))]))
        return count

    monkeypatch.setattr(os, "preadv", preadv)
    return read


def _same_rows(got, expected):
    for a, b in zip(got, expected):
        assert a.dtype == b.dtype and np.array_equal(a, b)


@pytest.mark.parametrize("layout", ["ranges", "interleaved"])
def test_ple_reads_warm_exactly_the_cold_uncached_rows_and_return_stock_bytes(layout, tmp_path, monkeypatch):
    table, model = _ple_table(tmp_path, cache_rows = 4, layout = layout)
    rows = np.array([2, 299, 300, 301, 799])
    table._read_rows(np.array([299, 301]))  # now in the row cache, so not read again
    expected = type(table)(tmp_path / "ple.json")._read_rows(rows)
    read, gathered, native = _recorded_reads(monkeypatch), [], np.memmap.__getitem__
    monkeypatch.setattr(decode, "_ple_resident", lambda parts: 0.0)
    with decode.exact_ple_row_reads(model):
        assert type(table).__name__ != "QuantizedMMapNGramEmbedding"
        # The cold rows come from the reads, not a fault-taking gather over the memory map.
        monkeypatch.setattr(np.memmap, "__getitem__", lambda self, index: gathered.append(index) or native(self, index))
        got = table._read_rows(rows)
        monkeypatch.setattr(np.memmap, "__getitem__", native)
    assert gathered == [] and sorted(read) == _row_bytes(table, (2, 300, 799))
    _same_rows(got, expected)


def test_ple_reads_probe_and_read_only_rows_the_table_has_not_read(tmp_path, monkeypatch):
    table, model = _ple_table(tmp_path)
    expected = type(table)(tmp_path / "ple.json")._read_rows(np.array([2, 3, 300, 799]))
    read, probed = _recorded_reads(monkeypatch), []
    monkeypatch.setattr(decode, "_ple_resident", lambda parts: probed.append([rows.tolist() for _, rows in parts]) or 0.0)
    with decode.exact_ple_row_reads(model):
        table._read_rows(np.array([2, 300]))
        del probed[:], read[:]
        table._read_rows(np.array([300, 2]))
        assert probed == [] and read == []
        got = table._read_rows(np.array([2, 3, 300, 799]))
    assert probed == [[[3], [499]]] and sorted(read) == _row_bytes(table, (3, 799))
    _same_rows(got, expected)


@pytest.mark.parametrize("layout", ["ranges", "interleaved"])
def test_ple_reads_keep_concurrent_gathers_of_one_table_apart(layout, tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    table, model = _ple_table(tmp_path, layout = layout)
    reference = type(table)(tmp_path / "ple.json")
    monkeypatch.setattr(decode, "_ple_resident", lambda parts: 0.0)
    barrier, prepare = threading.Barrier(2), decode._ple_pread_rows
    # Both gathers run while the other's reads are pending.
    monkeypatch.setattr(decode, "_ple_pread_rows", lambda *args: (prepare(*args), barrier.wait(5))[0])
    picks = [np.array([1, 302, 650]), np.array([5, 299, 777])]
    with decode.exact_ple_row_reads(model), ThreadPoolExecutor(2) as pool:
        results = list(pool.map(table._read_rows, picks))
    for rows, got in zip(picks, results):
        _same_rows(got, reference._read_rows(rows))


def test_ple_reads_skip_resident_tables_drifted_bodies_and_restore(tmp_path, monkeypatch):
    table, model = _ple_table(tmp_path)
    read, stock = _recorded_reads(monkeypatch), type(table)
    with decode.exact_ple_row_reads(model):
        with decode.exact_ple_row_reads(model):
            pass
        assert type(table) is not stock
        table._read_rows(np.arange(0, 800, 7))  # just written, so every page is in core
    assert type(table) is stock and read == [] and all(
        type(memmap) is np.memmap for _, _, arrays in table._shards for memmap in arrays.values())
    contract = {name: dict.fromkeys(names, "0" * 16) for name, names in decode._PLE_READ_CONTRACT.items()}
    monkeypatch.setattr(decode, "_PLE_READ_CONTRACT", contract)
    with decode.exact_ple_row_reads(model):
        assert type(table) is stock


_needs_qwen4 = pytest.mark.skipif(qwen4 is None, reason = "mlx-vlm without qwen4_exp")


def _qwen4_hc(combine = True, streams = 4, lowrank = 32, dtype = mx.bfloat16):
    config = SimpleNamespace(hidden_size = 1280, hc_count = streams, hc_lowrank = lowrank, rms_norm_eps = 1e-6)
    module = qwen4.Qwen4ExpGatedResidual(config, use_combine = combine)
    module.update(nn.utils.tree_map(lambda p: mx.random.normal(p.shape) * mx.exp(mx.random.normal(p.shape)) * 0.2,
                                    module.parameters()))
    module.set_dtype(dtype)
    return module.eval()


def _assert_same(actual, expected):
    actual, expected = nn.utils.tree_flatten(actual), nn.utils.tree_flatten(expected)
    assert [key for key, _ in actual] == [key for key, _ in expected]
    for a, b in zip(actual, expected):
        _equal(a[1], b[1])


@_needs_qwen4
@pytest.mark.parametrize("dtype, combine", [(mx.bfloat16, True), (mx.float16, False)])
def test_qwen4_prefill_kernels_match_mlx_vlm_bitwise(dtype, combine, monkeypatch):
    hc = _qwen4_hc(combine, dtype = dtype)
    # Prefill chunks around a decode step, the last at the row bound.
    chunks = [(mx.random.normal((2, n, 4 * 1280)) * mx.exp(2 * mx.random.normal((2, n, 1)))).astype(dtype) for n in (37, 300, 1, 16)]
    expected = [hc(x) for x in chunks]
    calls = {name: [] for name in ("_qwen4_hc_norm", "_qwen4_hc_mix")}
    for name, log in calls.items():
        monkeypatch.setattr(decode, name, lambda *a, f = getattr(decode, name), log = log: log.append(1) or f(*a))
    monkeypatch.setattr(decode, "_QWEN4_PREFILL_VERDICTS", {})
    with pytest.raises(RuntimeError, match = "cancel"), decode.fused_qwen4_prefill(hc):
        with decode.fused_qwen4_prefill(hc):
            pass
        for x, reference in zip(chunks, expected):
            _assert_same(hc(x), reference)
        raise RuntimeError("cancel")
    assert [len(log) for log in calls.values()] == [3, 3]   # a first-use comparison and two served calls
    assert list(decode._QWEN4_PREFILL_VERDICTS.values()) == [True]
    assert type(hc) is qwen4.Qwen4ExpGatedResidual


@_needs_qwen4
@pytest.mark.parametrize("reason", [
    "switch", "body", "wrapped", "short", "differs", "wide", "streams", "child", "subclass", "rebound", "sliced"])
def test_qwen4_prefill_scope_steps_aside(reason, monkeypatch, caplog):
    hc = _qwen4_hc(streams = 9 if reason == "streams" else 4)
    x = mx.random.normal((1, 31 if reason == "short" else 40, hc.hc_count * 1280)).astype(mx.bfloat16)
    norm, builder, residual, rms = decode._qwen4_hc_norm, qwen4._qwen4_rms_norm, qwen4.Qwen4ExpGatedResidual, qwen4.Qwen4ExpRMSNorm
    stock, shifted = residual.__call__, lambda base, call: type("Shifted", (base,), {"__call__": call})
    fail = lambda *a: pytest.fail("an ineligible call was fused")
    for name, value in (("_QWEN4_PREFILL_VERDICTS", {}), ("_QWEN4_PREFILL_BINDINGS", [None]), ("_CONTRACT_MISSES", set())):
        monkeypatch.setattr(decode, name, value)
    if reason == "body":
        monkeypatch.setattr(rms, "__call__", lambda self, h: h)
    elif reason == "wrapped":   # a wrapper around the cached norm builder, which unwraps to the stock body
        monkeypatch.setattr(qwen4, "_qwen4_rms_norm", functools.wraps(builder)(lambda *a: lambda *h: builder(*a)(*h) + 1))
    elif reason == "sliced":   # a projection result that the native mix broadcasts
        hc.input_mix_weight_up.__class__ = shifted(nn.Linear, lambda self, h: nn.Linear.__call__(self, h)[:, :1])
    elif reason == "child":   # an overridden norm the fused read would bypass
        hc.hc_norm.__class__ = shifted(rms, lambda self, h: rms.__call__(self, h) + 1)
    elif reason == "subclass":   # a residual whose own call the class swap would replace
        hc.__class__ = shifted(residual, lambda self, h: stock(self, h + 1))
    elif reason == "rebound":   # the exported norm class replaced after an earlier scope resolved the contract
        with decode.fused_qwen4_prefill(_qwen4_hc()):
            pass
        monkeypatch.setattr(qwen4, "Qwen4ExpRMSNorm", shifted(rms, lambda self, h: rms.__call__(self, h) + 1))
        hc.hc_norm.__class__ = qwen4.Qwen4ExpRMSNorm
    cls, expected = type(hc), hc(x)
    monkeypatch.setenv("UNSLOTH_MLX_QWEN4_PREFILL", "0" if reason == "switch" else "1")
    monkeypatch.setattr(decode, "_qwen4_hc_norm", fail if reason in ("short", "streams", "child", "rebound") else
                        (lambda *a: norm(*a) + 1) if reason == "differs" else norm)
    with caplog.at_level("WARNING"), decode.fused_qwen4_prefill(hc):
        patched = type(hc) is not cls
        if reason == "wide":   # widened while the scope is open, after a 16-bit call verified the key
            mx.eval(hc(x))
            hc.input_mix_weight_up.set_dtype(mx.float32)
            expected = cls.__call__(hc, x)
            monkeypatch.setattr(decode, "_qwen4_hc_mix", fail)
        for _ in range(2):
            _assert_same(hc(x), expected)
    assert patched == (reason in ("short", "differs", "wide", "sliced"))
    assert ("differs from the version" in caplog.text) == (reason in ("body", "wrapped", "rebound"))
    assert caplog.text.count("differs from mlx-vlm's") == (reason == "differs")
    assert type(hc) is cls


def _qwen4_read(combine = True, lowrank = 160, bits = 4, group_size = 32, dtype = mx.bfloat16):
    module = _qwen4_hc(combine, lowrank = lowrank, dtype = dtype)
    nn.quantize(module, group_size = group_size, bits = bits)
    module.eval()
    return module


def _qwen4_rows(rows, dtype = mx.bfloat16):
    return (mx.random.normal((1, rows, 4 * 1280)) * mx.exp(2 * mx.random.normal((1, rows, 1)))).astype(dtype)


def _logged_decode_kernels(monkeypatch, log, shift = 0):
    kernels = decode._qwen4_decode_kernels()

    def logged(kernel):
        return lambda **kw: log.append(kernel) or [out + shift if shift else out for out in kernel(**kw)]
    monkeypatch.setattr(decode, "_qwen4_decode_kernels", lambda: tuple(logged(k) for k in kernels))


@_needs_qwen4
@pytest.mark.parametrize("dtype, bits, group_size, lowrank, combine", [
    (mx.bfloat16, 4, 32, 160, True), (mx.float16, 8, 64, 192, False)])
def test_qwen4_decode_read_matches_mlx_vlm_bitwise(dtype, bits, group_size, lowrank, combine, monkeypatch):
    mx.random.seed(0)
    module, forward = _qwen4_read(combine, lowrank, bits, group_size, dtype), qwen4._QWEN4_BATCH_INVARIANT_FORWARD
    # Enough one-row reads that a norm rounding off by one float32 ulp shows in some 16-bit result.
    inputs = [_qwen4_rows(1, dtype) for _ in range(128)] + [_qwen4_rows(2, dtype)]
    expected = [forward._hyper_connection(module, x) for x in inputs]
    log = []
    _logged_decode_kernels(monkeypatch, log)
    monkeypatch.setattr(decode, "_QWEN4_PREFILL_VERDICTS", {})
    # Generation also holds the dense prefill scope, which swaps every quantized linear's class; it opens before
    # this scope or after it, closing first.
    outer, inner = ((decode.dense_prefill_linear, decode.fused_qwen4_decode) if combine
                    else (decode.fused_qwen4_decode, decode.dense_prefill_linear))
    with outer(module), decode.fused_qwen4_decode(module):
        with inner(module), decode.fused_qwen4_decode(module):
            assert type(module.input_mix_weight_down) is not nn.QuantizedLinear
            for x, reference in zip(inputs[:64], expected):
                _assert_same(forward._hyper_connection(module, x), reference)
        for x, reference in zip(inputs[64:], expected[64:]):   # the outer scope still routes
            _assert_same(forward._hyper_connection(module, x), reference)
    # A first-use comparison and 127 served one-row reads, two launches each; the two-row read stays native.
    assert len(log) == 256 and list(decode._QWEN4_PREFILL_VERDICTS.values()) == [True]
    assert "_hyper_connection" not in vars(forward)
    assert not {"_unsloth_qwen4_decode_scopes", "_unsloth_qwen4_decode_plan"} & set(vars(module))


@_needs_qwen4
@pytest.mark.parametrize("reason", [
    "switch", "body", "patched", "outside", "differs", "aligned", "quad", "child", "override", "linear",
    "foreign", "late", "rebound", "lora", "dtype", "verified", "relinear"])
def test_qwen4_decode_read_steps_aside(reason, monkeypatch, caplog):
    module = _qwen4_read(lowrank = {"aligned": 512, "quad": 128}.get(reason, 160))
    other, forward = _qwen4_read(), qwen4._QWEN4_BATCH_INVARIANT_FORWARD
    x = _qwen4_rows(1, mx.float16 if reason == "dtype" else mx.bfloat16)   # "dtype": not the plan's
    if reason == "body":
        monkeypatch.setattr(qwen4.Qwen4ExpRMSNorm, "__call__", lambda self, x: x)
    elif reason == "child":
        module.hc_norm.__class__ = type("Shifted", (qwen4.Qwen4ExpRMSNorm,), {
            "__call__": lambda self, x: qwen4.Qwen4ExpRMSNorm.__call__(self, x) + 1})
    elif reason == "patched":
        stock_linear = nn.QuantizedLinear.__call__
        monkeypatch.setattr(nn.QuantizedLinear, "__call__", lambda self, x: stock_linear(self, x) * 2)
    elif reason == "lora":   # an unfused adapter wraps the projection, which has no weight of its own
        from mlx_lm.tuner.lora import LoRALinear
        module.input_mix_weight_down = LoRALinear.from_base(module.input_mix_weight_down)
        module.input_mix_weight_down.lora_b = mx.random.normal(module.input_mix_weight_down.lora_b.shape) * 0.01
    elif reason == "override":
        module.input_mix_weight_down.__class__ = type("Scaled", (nn.QuantizedLinear,), {
            "__call__": lambda self, x: nn.QuantizedLinear.__call__(self, x) * 2})
    elif reason == "linear":
        monkeypatch.setitem(vars(forward), "_linear", lambda linear, h: linear(h) * 2)
    elif reason in ("verified", "relinear"):   # exact_batch_invariant_decode's class already over the forward
        native = type(forward)
        methods = (getattr(native, name, None) for name in decode._VERIFY_METHODS)
        verified = decode._exact_verify_class(native, "__call__", False, *methods)
        if reason == "relinear":   # a class under that one decorating its `_linear`
            sliced = verified._linear
            verified = type("Scaled", (verified,), {
                "_linear": functools.wraps(sliced)(lambda self, linear, h: sliced(self, linear, h) * 2)})
        monkeypatch.setattr(forward, "__class__", verified)
    foreign = lambda m, h: type(forward)._hyper_connection(forward, m, h)
    if reason == "foreign":   # another library's hook on the shared forward object
        monkeypatch.setitem(vars(forward), "_hyper_connection", foreign)
    for name, value in (("_QWEN4_PREFILL_VERDICTS", {}), ("_QWEN4_DECODE_BINDINGS", [None]), ("_CONTRACT_MISSES", set())):
        monkeypatch.setattr(decode, name, value)
    if reason == "rebound":   # the exported norm class replaced after an earlier scope resolved the contract
        with decode.fused_qwen4_decode(other):
            pass
        stock_norm = qwen4.Qwen4ExpRMSNorm
        monkeypatch.setattr(qwen4, "Qwen4ExpRMSNorm", type("Shifted", (stock_norm,), {
            "__call__": lambda self, h: stock_norm.__call__(self, h) + 1}))
        module.hc_norm.__class__ = qwen4.Qwen4ExpRMSNorm
    target = other if reason == "outside" else module
    expected = forward._hyper_connection(target, x)
    log = []
    _logged_decode_kernels(monkeypatch, log, shift = 1 if reason == "differs" else 0)
    monkeypatch.setenv("UNSLOTH_MLX_QWEN4_DECODE", "0" if reason == "switch" else "1")
    with caplog.at_level("WARNING"), decode.fused_qwen4_decode(module):
        hooked = getattr(vars(forward).get("_hyper_connection"), "func", None) is decode._qwen4_decode_read
        if reason == "late":   # replaced while the scope is open
            vars(forward)["_hyper_connection"] = foreign
        for _ in range(2):
            _assert_same(forward._hyper_connection(target, x), expected)
    left = vars(forward).pop("_hyper_connection", None) if reason == "late" else vars(forward).get("_hyper_connection")
    assert left is (foreign if reason in ("foreign", "late") else None)
    assert hooked == (reason in ("outside", "differs", "late", "dtype", "verified"))
    assert len(log) == {"differs": 2, "verified": 4}.get(reason, 0)
    assert ("differs from the version" in caplog.text) == (reason in ("body", "patched", "rebound"))
    assert caplog.text.count("differs from mlx-vlm's") == (reason == "differs")


@_needs_qwen4
@pytest.mark.parametrize("removed", ["residual", "linear"])
def test_qwen4_decode_scope_stays_native_when_a_pinned_name_is_gone(removed, monkeypatch):
    # A drifted mlx-vlm without one of these names fails the contract; opening the scope must not raise.
    module, forward = _qwen4_read(), qwen4._QWEN4_BATCH_INVARIANT_FORWARD
    for name, value in (("_QWEN4_PREFILL_VERDICTS", {}), ("_QWEN4_DECODE_BINDINGS", [None]), ("_CONTRACT_MISSES", set())):
        monkeypatch.setattr(decode, name, value)
    if removed == "residual":
        monkeypatch.delattr(qwen4, "Qwen4ExpGatedResidual")
    else:
        monkeypatch.delattr(next(c for c in type(forward).__mro__ if "_linear" in vars(c)), "_linear")
    with decode.fused_qwen4_decode(module):
        assert "_hyper_connection" not in vars(forward)
        assert "_unsloth_qwen4_decode_plan" not in vars(module)
