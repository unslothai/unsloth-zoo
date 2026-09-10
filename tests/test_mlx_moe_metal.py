# SPDX-License-Identifier: AGPL-3.0-only
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx_simulation import mlx_is_simulated

if mlx_is_simulated():
    pytest.skip("Requires native MLX", allow_module_level = True)
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason = "Requires Metal")

import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models import switch_layers as lm
from mlx_vlm.models import switch_layers as vlm
from unsloth_zoo.mlx.inference import fused_moe_gate_up


def _model(native, dtype = mx.bfloat16, dims = (2048, 512), bits = 8, activation = None, bias = True):
    mx.random.seed(19)
    kwargs = {} if activation is None else {"activation": activation}
    model = native.SwitchGLU(*dims, 8, bias = bias, **kwargs)
    model.set_dtype(dtype)
    nn.quantize(model, bits = bits, group_size = 64)
    model.eval()
    model.freeze()
    mx.eval(model.parameters())
    return model


def _equal(a, b):
    assert np.array_equal(np.array(a.view(mx.uint8)), np.array(b.view(mx.uint8)))


def _sample(width = 2048):
    return (
        mx.random.normal((1, 2, width)).astype(mx.bfloat16),
        mx.array([[[7, 2, 0, 5], [6, 4, 1, 3]]]),
    )


@pytest.mark.parametrize("native", [lm, vlm])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("dims", [(2048, 512), (4096, 1408)])
def test_fusion_preserves_outputs_parameters_and_native_class(native, dtype, dims, monkeypatch):
    model = _model(native, dtype, dims)
    original_call = native.SwitchGLU.__call__
    names = {name for name, _ in tree_flatten(model.parameters())}
    samples = []
    for batch, length in [(1, 1), (2, 1), (1, 8), (1, 64), (1, 256)]:
        x = mx.random.normal((batch, length, dims[0])).astype(dtype)
        indices = mx.random.randint(0, 8, (batch, length, 8))
        expected = model(x, indices)
        mx.eval(x, indices, expected)
        samples.append((x, indices, expected))
    gather_qmm = mx.gather_qmm
    calls = []

    def counted(*args, **kwargs):
        calls.append(None)
        return gather_qmm(*args, **kwargs)

    monkeypatch.setattr(mx, "gather_qmm", counted)
    with fused_moe_gate_up(model):
        assert type(model) is not native.SwitchGLU
        assert native.SwitchGLU.__call__ is original_call
        assert {name for name, _ in tree_flatten(model.parameters())} == names
        for x, indices, expected in samples:
            calls.clear()
            _equal(model(x, indices), expected)
            assert len(calls) == 2
    assert type(model) is native.SwitchGLU
    assert {name for name, _ in tree_flatten(model.parameters())} == names


@pytest.mark.parametrize("native", [lm, vlm])
def test_nested_scopes_restore_all_modules_after_exception(native):
    first, second = _model(native), _model(native)
    root = nn.Sequential(first, second)
    root.eval()
    before = [set(module.__dict__) for module in (first, second)]
    x, indices = _sample()
    expected = [module(x, indices) for module in (first, second)]
    mx.eval(expected)
    with pytest.raises(RuntimeError, match = "injected"):
        with fused_moe_gate_up(root):
            outer_classes = [type(module) for module in (first, second)]
            assert all(cls is not native.SwitchGLU for cls in outer_classes)
            with fused_moe_gate_up(root):
                assert [type(module) for module in (first, second)] == outer_classes
            assert [type(module) for module in (first, second)] == outer_classes
            for module, answer in zip((first, second), expected):
                _equal(module(x, indices), answer)
            raise RuntimeError("injected")
    for module, keys, answer in zip((first, second), before, expected):
        assert type(module) is native.SwitchGLU
        assert set(module.__dict__) == keys
        _equal(module(x, indices), answer)


@pytest.mark.parametrize("native", [lm, vlm])
@pytest.mark.parametrize("first_exit", [0, 1])
def test_overlapping_scopes_keep_shared_modules_patched(native, first_exit):
    modules = [_model(native), _model(native)]
    root = nn.Sequential(*modules)
    root.eval()
    x, indices = _sample()
    expected = [module(x, indices) for module in modules]
    mx.eval(expected)
    scopes = [fused_moe_gate_up(root), fused_moe_gate_up(modules[1])]
    active = []
    try:
        for scope in scopes:
            scope.__enter__()
            active.append(scope)
        bound_call = modules[1].__call__
        active.remove(scopes[first_exit])
        scopes[first_exit].__exit__(None, None, None)
        for index, module in enumerate(modules):
            still_active = first_exit == 1 or index == 1
            assert (type(module) is not native.SwitchGLU) == still_active
            assert hasattr(module, "_unsloth_moe_gate_up") == still_active
            _equal(module(x, indices), expected[index])
        _equal(bound_call(x, indices), expected[1])
    finally:
        for scope in reversed(active):
            scope.__exit__(None, None, None)
    for module, answer in zip(modules, expected):
        assert type(module) is native.SwitchGLU
        assert not hasattr(module, "_unsloth_moe_gate_up")
        _equal(module(x, indices), answer)


def test_fusion_does_not_clear_the_global_allocator(monkeypatch):
    model = _model(lm)
    clears = []
    monkeypatch.setattr(mx, "clear_cache", lambda: clears.append(None))
    with fused_moe_gate_up(model):
        assert type(model) is not lm.SwitchGLU
    assert clears == []


@pytest.mark.parametrize("native", [lm, vlm])
def test_inplace_edits_between_scopes_are_used(native):
    model = _model(native)
    x, indices = _sample()
    with fused_moe_gate_up(model):
        before = model(x, indices)
        mx.eval(before)
    for projection in (model.gate_proj, model.up_proj):
        weight = projection.weight
        weight[-1, -64:, :] = 0
        projection.scales[-2, :, :] *= 1.5
        projection.bias[-1, :] += 0.25
        assert projection.weight is weight
        expected = model(x, indices)
        mx.eval(expected)
        assert not np.array_equal(
            np.array(before.view(mx.uint8)), np.array(expected.view(mx.uint8)),
        )
        with fused_moe_gate_up(model):
            _equal(model(x, indices), expected)
        _equal(model(x, indices), expected)
        before = expected


@pytest.mark.parametrize("native", [lm, vlm])
def test_training_replacement_and_adapters_use_native_path(native, monkeypatch):
    model = _model(native)
    x, indices = _sample()
    original_call = native.SwitchGLU.__call__
    gather_qmm = mx.gather_qmm
    calls = []

    def counted(*args, **kwargs):
        calls.append(None)
        return gather_qmm(*args, **kwargs)

    class AdaptedProjection(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def __call__(self, inputs, indices, sorted_indices = False):
            return self.base(inputs, indices, sorted_indices = sorted_indices) + 0.5

    monkeypatch.setattr(mx, "gather_qmm", counted)
    with fused_moe_gate_up(model):
        model.train()
        calls.clear()
        actual = model(x, indices)
        assert len(calls) == 3
        _equal(actual, original_call(model, x, indices))
        model.eval()
        model.gate_proj.scales = model.gate_proj.scales * 1.25
        calls.clear()
        actual = model(x, indices)
        assert len(calls) == 3
        _equal(actual, original_call(model, x, indices))
    with fused_moe_gate_up(model):
        model.up_proj = AdaptedProjection(model.up_proj)
        calls.clear()
        actual = model(x, indices)
        assert len(calls) == 3
        _equal(actual, original_call(model, x, indices))


@pytest.mark.parametrize("reason", ["4bit", "custom", "training", "trainable", "distributed"])
def test_ineligible_models_stay_native(reason):
    model = _model(vlm, bits = 4 if reason == "4bit" else 8)
    if reason == "custom":
        class CustomSwitch(vlm.SwitchGLU):
            pass
        model.__class__ = CustomSwitch
    elif reason == "training":
        model.train()
    elif reason == "trainable":
        model.up_proj.unfreeze()
    elif reason == "distributed":
        model._unsloth_mlx_distributed_parallel_mode = "tensor"
    original_class = type(model)
    with fused_moe_gate_up(model):
        assert type(model) is original_class


def test_generation_mode_applies_fusion_and_restores_training():
    from unsloth_zoo.mlx.generate import generation_mode

    model = _model(lm)
    x, indices = _sample()
    expected = model(x, indices)
    mx.eval(expected)
    model.train()
    with generation_mode(model):
        assert type(model) is not lm.SwitchGLU
        assert not model.training
        _equal(model(x, indices), expected)
    assert type(model) is lm.SwitchGLU
    assert model.training


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("family", ["gemma", "gpt_oss"])
@pytest.mark.parametrize("bias", [False, True])
def test_family_activation_is_preserved(dtype, family, bias):
    if family == "gemma":
        from mlx_vlm.models.gemma4.language import GeGLU
        activation, dims = GeGLU(), (2816, 704)
    else:
        from mlx_lm.models.gpt_oss import SwiGLU
        activation, dims = SwiGLU(), (2880, 2880)
    model = _model(vlm, dtype, dims, activation = activation, bias = bias)
    samples = []
    for batch, length in [(1, 1), (2, 1), (1, 32)]:
        x = mx.random.normal((batch, length, dims[0])).astype(dtype)
        indices = mx.random.randint(0, 8, (batch, length, 8))
        expected = model(x, indices)
        mx.eval(x, indices, expected)
        samples.append((x, indices, expected))
    with fused_moe_gate_up(model):
        assert type(model) is not vlm.SwitchGLU
        for x, indices, expected in samples:
            _equal(model(x, indices), expected)
