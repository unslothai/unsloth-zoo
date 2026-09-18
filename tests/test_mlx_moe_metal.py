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
from mlx_lm.models.gpt_oss import SwiGLU
from mlx_vlm.models import switch_layers as vlm
from mlx_vlm.models.gemma4.language import GeGLU
from unsloth_zoo.mlx import inference as fusion
from unsloth_zoo.mlx.generate import generation_mode
from unsloth_zoo.mlx.inference import fused_moe_gate_up


QUANTIZATIONS = [(8, "affine", 64), (4, "affine", 64), (4, "affine", 32), (6, "affine", 32),
                 (4, "mxfp4", 32), (8, "mxfp8", 32), (4, "nvfp4", 16)]


def _model(native, dtype = mx.bfloat16, dims = (2048, 512), quantization = (8, "affine", 64),
           activation = None, bias = True):
    mx.random.seed(19)
    bits, mode, group_size = quantization
    model = native.SwitchGLU(*dims, 8, bias = bias, **({} if activation is None else {"activation": activation}))
    model.set_dtype(dtype)
    nn.quantize(model, bits = bits, group_size = group_size, mode = mode,
                # a hidden width the group size does not divide leaves down_proj native, which the pack ignores
                class_predicate = lambda _, m: hasattr(m, "to_quantized") and not m.weight.shape[-1] % group_size)
    model.eval()
    model.freeze()
    mx.eval(model.parameters())
    return model


def _equal(a, b):
    assert np.array_equal(np.array(a.view(mx.uint8)), np.array(b.view(mx.uint8)))


def _samples(model, dtype, width, shapes):
    built = []
    for batch, length in shapes:
        x, indices = mx.random.normal((batch, length, width)).astype(dtype), mx.random.randint(0, 8, (batch, length, 8))
        built.append((x, indices, model(x, indices)))
    mx.eval(built)
    return built


def _counting_gather_qmm(monkeypatch):
    native_gather_qmm, calls = mx.gather_qmm, []
    monkeypatch.setattr(mx, "gather_qmm", lambda *a, **k: calls.append(None) or native_gather_qmm(*a, **k))
    return calls


def _sample(width = 2048):
    # past the upstream decode window, so eligibility is what decides the path taken, not length
    length = getattr(vlm, "DECODE_BLOCK_SIZE", 0) + 2
    rows = mx.array([[7, 2, 0, 5], [6, 4, 1, 3]])
    return (mx.random.normal((1, length, width)).astype(mx.bfloat16),
            mx.tile(rows, ((length + 1) // 2, 1))[None, :length])


@pytest.mark.parametrize("native", [lm, vlm])
@pytest.mark.parametrize("dtype, dims", [(mx.bfloat16, (2048, 512)), (mx.float16, (4096, 1408))])
@pytest.mark.parametrize("quantization", QUANTIZATIONS)
def test_fusion_preserves_outputs_parameters_and_native_class(native, dtype, dims, quantization, monkeypatch):
    model = _model(native, dtype, dims, quantization)
    original_call = native.SwitchGLU.__call__
    decode_block = getattr(native, "DECODE_BLOCK_SIZE", 0)
    names = {name for name, _ in tree_flatten(model.parameters())}
    samples = _samples(model, dtype, dims[0], [(1, 1), (2, 1), (1, 8), (1, 64), (1, 256)])
    calls, clears = _counting_gather_qmm(monkeypatch), []
    monkeypatch.setattr(mx, "clear_cache", lambda: clears.append(None))
    with fused_moe_gate_up(model):
        assert type(model) is not native.SwitchGLU
        assert native.SwitchGLU.__call__ is original_call
        assert {name for name, _ in tree_flatten(model.parameters())} == names
        for x, indices, expected in samples:
            calls.clear()
            original_call(model, x, indices)
            native_calls = len(calls)
            calls.clear()
            _equal(model(x, indices), expected)
            # packing folds the separate gate and up projections into one, except where a short
            # sequence belongs to an upstream decode path this body does not reimplement
            assert len(calls) == (native_calls if 1 < x.shape[1] <= decode_block else 2)
        monkeypatch.setattr(native, "_gather_sort", None)  # a rebound switch layer function falls back per call
        calls.clear()
        _equal(model(*samples[0][:2]), samples[0][2])
        assert len(calls) == 3
    assert type(model) is native.SwitchGLU
    assert {name for name, _ in tree_flatten(model.parameters())} == names
    assert clears == []  # the scope never clears the allocator other models share


def test_each_contract_resolution_gets_its_own_bindings():
    # one list shared between resolutions would vouch for helpers a cached fused class never captured
    first, second = fusion._moe_switch_specs(), fusion._moe_switch_specs()
    assert first[lm.SwitchGLU][-1] is not second[lm.SwitchGLU][-1]
    fused = [fusion._fused_moe_gate_up_class(lm.SwitchGLU, kind, None, None, 0, []) for kind in
             (lm.QuantizedSwitchLinear, type("Other", (lm.QuantizedSwitchLinear,), {}))]
    assert fused[0] is not fused[1]  # one class cached across projection types would guard the wrong one


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
    entered = []
    try:
        for scope in scopes:
            scope.__enter__()
            entered.append(scope)
        bound_call = modules[1].__call__
        entered.remove(scopes[first_exit])
        scopes[first_exit].__exit__(None, None, None)
        for index, module in enumerate(modules):
            still_active = first_exit == 1 or index == 1
            assert (type(module) is not native.SwitchGLU) == still_active
            assert hasattr(module, "_unsloth_moe_gate_up") == still_active
            _equal(module(x, indices), expected[index])
        _equal(bound_call(x, indices), expected[1])
    finally:
        for scope in reversed(entered):
            scope.__exit__(None, None, None)
    for module, answer in zip(modules, expected):
        assert type(module) is native.SwitchGLU
        assert not hasattr(module, "_unsloth_moe_gate_up")
        _equal(module(x, indices), answer)


@pytest.mark.parametrize("native", [lm, vlm])
def test_inplace_edits_between_scopes_are_used(native):
    model = _model(native)
    x, indices = _sample()
    with fused_moe_gate_up(model):
        mx.eval(before := model(x, indices))
    for projection in (model.gate_proj, model.up_proj):
        weight = projection.weight
        weight[-1, -64:, :] = 0
        projection.scales[-2, :, :] *= 1.5
        projection.bias[-1, :] += 0.25
        assert projection.weight is weight
        mx.eval(expected := model(x, indices))
        assert not np.array_equal(np.array(before.view(mx.uint8)), np.array(expected.view(mx.uint8)))
        with fused_moe_gate_up(model):
            _equal(model(x, indices), expected)
        _equal(model(x, indices), expected)
        before = expected


@pytest.mark.parametrize("native", [lm, vlm])
def test_training_replacement_and_adapters_use_native_path(native, monkeypatch):
    model = _model(native)
    x, indices = _sample()
    original_call = native.SwitchGLU.__call__
    calls = _counting_gather_qmm(monkeypatch)

    class AdaptedProjection(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def __call__(self, inputs, indices, sorted_indices = False):
            return self.base(inputs, indices, sorted_indices = sorted_indices) + 0.5

    def assert_native_path():
        calls.clear()
        actual = model(x, indices)
        observed = len(calls)
        calls.clear()
        expected = original_call(model, x, indices)
        assert observed == len(calls)  # packing would fold two projections into one call
        _equal(actual, expected)

    with fused_moe_gate_up(model):
        model.train()
        assert_native_path()
        model.eval()
        model.gate_proj.scales = model.gate_proj.scales * 1.25
        assert_native_path()
    with fused_moe_gate_up(model):
        model.up_proj = AdaptedProjection(model.up_proj)
        assert_native_path()


@pytest.mark.parametrize("reason", ["rows", "mixed", "partial", "stale", "helper", "custom", "training", "trainable", "distributed"])
def test_ineligible_models_stay_native(reason, monkeypatch):
    # 516 rows pack to 1032, which MLX sends to the aligned kernel the 516-row pair cannot use.
    model = _model(vlm, dims = (2048, 516) if reason == "rows" else (2048, 512),
                   quantization = (4, "affine", 64) if reason == "partial" else (8, "affine", 64))
    if reason == "mixed":
        model.up_proj = _model(vlm, quantization = (4, "affine", 64)).up_proj
    elif reason == "partial":
        model.gate_proj.biases = None  # one projection carrying a field the other lacks cannot pack
    elif reason == "stale":
        monkeypatch.setitem(fusion._MOE_SWITCH_GLU_CALLS, "mlx_vlm.models.switch_layers", ("stale",))
    elif reason == "helper":  # a sort helper whose body is not the one the fused call was written against
        monkeypatch.setattr(vlm, "_gather_sort", vlm._scatter_unsort)
    elif reason == "custom":
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
    model = _model(lm)
    (x, indices, expected), = _samples(model, mx.bfloat16, 2048, [(1, 2)])
    model.train()
    with generation_mode(model):
        assert type(model) is not lm.SwitchGLU
        assert not model.training
        _equal(model(x, indices), expected)
    assert type(model) is lm.SwitchGLU and model.training


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("family", ["gemma", "gpt_oss"])
@pytest.mark.parametrize("bias", [False, True])
def test_family_activation_is_preserved(dtype, family, bias):
    activation, dims = (GeGLU(), (2816, 704)) if family == "gemma" else (SwiGLU(), (2880, 2880))
    model = _model(vlm, dtype, dims, activation = activation, bias = bias)
    samples = _samples(model, dtype, dims[0], [(1, 1), (2, 1), (1, 32)])
    with fused_moe_gate_up(model):
        assert type(model) is not vlm.SwitchGLU
        for x, indices, expected in samples:
            _equal(model(x, indices), expected)


def test_accepted_switch_glu_bodies_cover_the_installed_packages():
    # the fused body reimplements these; an unlisted body must be caught here, not by wrong numbers
    for path, accepted in fusion._MOE_SWITCH_GLU_CALLS.items():
        native = {"mlx_lm.models.switch_layers": lm, "mlx_vlm.models.switch_layers": vlm}[path]
        assert fusion._ast_fingerprint(native.SwitchGLU.__call__) in accepted


@pytest.mark.skipif("weights" not in vlm.SwitchGLU.__call__.__code__.co_varnames,
                    reason = "Requires an mlx-vlm that combines routing weights")
@pytest.mark.parametrize("with_shared", [False, True])
def test_combined_arguments_fall_back_to_the_native_path(with_shared, monkeypatch):
    model = _model(vlm)
    # longer than the upstream decode path covers, so only the combined arguments can defer
    length = getattr(vlm, "DECODE_BLOCK_SIZE", 0) + 8
    x = mx.random.normal((1, length, 2048)).astype(mx.bfloat16)
    indices = mx.random.randint(0, 8, (1, length, 4))
    original_call = vlm.SwitchGLU.__call__
    kwargs = {"weights": mx.random.normal(indices.shape).astype(mx.bfloat16)}
    if with_shared:  # upstream weights the routed experts before adding the shared one
        kwargs["shared"] = mx.random.normal((1, length, 2048)).astype(mx.bfloat16)
    mx.eval((x, indices, kwargs))
    expected = original_call(model, x, indices, **kwargs)
    calls = _counting_gather_qmm(monkeypatch)
    with fused_moe_gate_up(model):
        assert type(model) is not vlm.SwitchGLU
        calls.clear()
        actual = model(x, indices, **kwargs)
        observed = len(calls)
        calls.clear()
        original_call(model, x, indices, **kwargs)
        assert observed == len(calls)  # packing here would drop the combine the native call performs
    _equal(actual, expected)
    _equal(model(x, indices, **kwargs), expected)
