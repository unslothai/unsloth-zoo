# SPDX-License-Identifier: AGPL-3.0-only
import ast
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

# The scope leaves a module that packs on its own (mlx-vlm 0.7.8) alone, so the route is exercised on one that does not.
_PACKED_SOURCES = [case for case in [(native, vlm_gated_delta, native.Qwen3_5GatedDeltaNet, native.ArraysCache),
                                     (lm_qwen3_5, lm_gated_delta, lm_qwen3_5.GatedDeltaNet, lm_cache.ArraysCache)]
                   if not hasattr(case[1], "_gated_delta_kernel_packed")]
_, routed_gated_delta, _ROUTED_LAYER, _ = _PACKED_SOURCES[0]


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
