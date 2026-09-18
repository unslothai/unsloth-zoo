# SPDX-License-Identifier: AGPL-3.0-only
import functools
import sys
import threading
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx_simulation import mlx_is_simulated

if mlx_is_simulated():
    pytest.skip("Requires native MLX", allow_module_level = True)
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason = "Requires Metal")

import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models import gemma4_text as lm_gemma
from mlx_lm.models import qwen3_next as lm_qwen
from mlx_lm.models import switch_layers as lm
from mlx_lm.models.gpt_oss import SwiGLU
from mlx_vlm.models import switch_layers as vlm
from mlx_vlm.models.gemma4 import language as vlm_gemma
from mlx_vlm.models.gemma4.language import GeGLU
from mlx_vlm.models.qwen3_5_moe import language as vlm_qwen
from unsloth_zoo.mlx import inference as fusion
from unsloth_zoo.mlx.generate import generation_mode
from unsloth_zoo.mlx.inference import fused_moe_gate_up


QWEN, GEMMA = fusion._QWEN_ROUTING, fusion._GEMMA_ROUTING

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


def _identical(a, b, nan_payload_may_differ = False):
    assert a.dtype == b.dtype
    if nan_payload_may_differ and mx.issubdtype(a.dtype, mx.floating):
        # A poisoned row is NaN on both paths, but WHICH NaN is not stable: the same native chain
        # yields bfloat16 0x7FC0 on one Apple GPU family and 0x7FFF on another, while the kernel
        # writes Metal's canonical NAN. IEEE-754 does not specify payloads and neither does MLX, so
        # requiring those bits to agree asserts something no fixed kernel output can satisfy on all
        # hardware. Poisoning is what has to match, and it is still checked exactly: the NaN
        # positions must be identical, and every other byte -- every index, every finite weight --
        # still compares bit for bit.
        assert mx.array_equal(mx.isnan(a), mx.isnan(b)), "the poisoned positions differ"
        quiet = mx.array(float("nan"), dtype = a.dtype)
        a, b = mx.where(mx.isnan(a), quiet, a), mx.where(mx.isnan(b), quiet, b)
    _equal(a, b)


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


def _inputs(dtype, experts, rows, mode):
    scale = mx.random.uniform(0.5, 1.5, (experts,)).astype(dtype) if mode == GEMMA else fusion._MOE_ROUTER_NO_SCALE
    logits = (mx.random.randint(0, 4, (rows, experts)) if rows == 8 else mx.random.normal((rows, experts)) * 3).astype(dtype)
    if rows == 4096:  # 8 integer rows tie at the top-k boundary; the large arm carries a poisoned expert and a poisoned row
        logits[0, 5], logits[1] = float("nan"), float("nan")
    return logits, scale


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("experts, top_k", [(32, 1), (64, 5), (128, 8), (160, 3), (256, 8), (1024, 8)])
@pytest.mark.parametrize("mode, normalize", [(QWEN, True), (QWEN, False), (GEMMA, False)])
def test_kernel_reproduces_native_routing(dtype, experts, top_k, mode, normalize):
    for rows in (0, 1, 8, 4096):
        mx.random.seed(rows + experts + top_k)
        logits, scale = _inputs(dtype, experts, rows, mode)
        fused = fusion._fused_moe_router(logits, scale, top_k, mode, normalize)
        if rows == 0 or (mode == QWEN and experts > 256):
            assert fused is None
            continue
        assert fused is not None
        assert fusion._fused_moe_router(logits[0], scale, top_k, mode, normalize) is None  # rank-1 takes the native chain
        for a, b in zip(fusion._native_moe_router(logits, scale, top_k, mode, normalize), fused):
            _identical(a, b, nan_payload_may_differ = True)


def _prepared(module):
    module.set_dtype(mx.bfloat16)
    nn.quantize(module, bits = 8, group_size = 32)
    module.eval()
    mx.eval(module.parameters())
    return module


def _block(native, experts = 256, top_k = 8, norm_topk_prob = True, cls = None):
    mx.random.seed(7)
    args = SimpleNamespace(hidden_size = 64, moe_intermediate_size = 32, num_experts = experts,
                           num_experts_per_tok = top_k, shared_expert_intermediate_size = 32,
                           norm_topk_prob = norm_topk_prob)
    cls = cls or (vlm_qwen.Qwen3_5MoeSparseMoeBlock if native is vlm_qwen else lm_qwen.Qwen3NextSparseMoeBlock)
    return _prepared(cls(args))


def _router(native, experts = 128, top_k = 8, scale_dtype = mx.bfloat16):
    mx.random.seed(11)
    config = SimpleNamespace(hidden_size = 64, num_experts = experts, top_k_experts = top_k, rms_norm_eps = 1e-6)
    router = _prepared(native.Router(config))
    router.scale = mx.random.uniform(0.5, 1.5, (64,)).astype(mx.bfloat16)
    router.per_expert_scale = mx.random.uniform(0.5, 1.5, (experts,)).astype(scale_dtype)
    return router


def _routing_samples(module):
    xs = [mx.random.normal(shape).astype(mx.bfloat16) for shape in ((1, 1, 64), (2, 3, 64), (1, 70, 64))]
    return [(x, module(x)) for x in xs]


def _check(module, samples):
    as_tuple = lambda out: out if isinstance(out, tuple) else (out,)
    for x, expected in samples:
        for a, b in zip(as_tuple(module(x)), as_tuple(expected)):
            _identical(a, b)


def _counting_argpartition(monkeypatch):
    partitions, argpartition = [], mx.argpartition
    monkeypatch.setattr(mx, "argpartition", lambda *a, **k: partitions.append(None) or argpartition(*a, **k))
    return partitions


def _skip_unless_gate_up_resolves(native):
    if native.SwitchGLU not in fusion._moe_switch_specs():
        pytest.skip("the gate/up contract does not resolve for this package")


@pytest.mark.parametrize("native", [vlm_qwen, lm_qwen])
@pytest.mark.parametrize("gate_up_first", [False, True])
def test_scope_fuses_routing_composes_with_gate_up_in_either_order_and_restores(native, gate_up_first, monkeypatch):
    _skip_unless_gate_up_resolves(native)
    block = _block(native)
    samples = _routing_samples(block)
    base = type(block)
    partitions = _counting_argpartition(monkeypatch)
    outer, inner = (fusion.fused_moe_gate_up, fusion.fused_moe_router) if gate_up_first else (fusion.fused_moe_router, fusion.fused_moe_gate_up)
    with outer(block):
        with inner(block):
            assert type(block) is not base and isinstance(block, base)
            assert type(block.switch_mlp) is not native.SwitchGLU
            mx.eval(block(samples[0][0]))  # the first fused call verifies the kernel against the native chain
            partitions.clear()
            _check(block, samples)
            assert not partitions
        _check(block, samples)
        assert bool(partitions) is gate_up_first  # the routing stays fused until its own scope exits
    assert type(block) is base and type(block.switch_mlp) is native.SwitchGLU
    _check(block, samples)
    assert partitions


def test_unnormalized_top_k_and_subclass_bodies_are_fused(monkeypatch):
    class Scaled(vlm_qwen.Qwen3_5MoeSparseMoeBlock):  # qwen4_exp inherits the body and overrides the shared gate
        def _shared_expert_scale(self, x):
            return 2 * super()._shared_expert_scale(x)

    partitions = _counting_argpartition(monkeypatch)
    for block in (_block(lm_qwen, norm_topk_prob = False), _block(vlm_qwen, cls = Scaled)):
        samples = _routing_samples(block)
        base = type(block)
        with fusion.fused_moe_router(block):
            assert type(block) is not base and isinstance(block, base)
            mx.eval(block(samples[0][0]))
            partitions.clear()
            _check(block, samples)
            assert not partitions


@pytest.mark.parametrize("native, scale_dtype", [(vlm_gemma, mx.bfloat16), (lm_gemma, mx.float32)])
def test_gemma_router_is_fused_and_falls_back_on_the_raw_input(native, scale_dtype, monkeypatch):
    router = _router(native, scale_dtype = scale_dtype)
    samples = _routing_samples(router)
    assert samples[0][1][1].dtype == scale_dtype  # a float32 per_expert_scale promotes the weights
    partitions = _counting_argpartition(monkeypatch)
    with fusion.fused_moe_router(router):
        assert type(router) is not native.Router
        mx.eval(router(samples[0][0]))
        partitions.clear()
        _check(router, samples)
        assert not partitions
        monkeypatch.setattr(fusion, "_fused_moe_router", lambda *a: None)
        _check(router, samples)
        assert partitions
    assert type(router) is native.Router


def test_gemma_holds_the_norm_weight_for_the_scope_and_refuses_a_stale_one(monkeypatch):
    router = _router(vlm_gemma)
    samples = _routing_samples(router)
    partitions = _counting_argpartition(monkeypatch)
    weights, rms_norm = [], mx.fast.rms_norm
    monkeypatch.setattr(mx.fast, "rms_norm", lambda x, w, eps: weights.append(w) or rms_norm(x, w, eps))
    with fusion.fused_moe_router(router):
        held = router._unsloth_router_norm
        mx.eval(router(samples[0][0]))
        partitions.clear()
        weights.clear()
        _check(router, samples)
        assert not partitions and router._unsloth_router_norm is held
        assert weights and all(w is held.weight for w in weights)  # the held product, not a fresh one
        router.scale = router.scale * 1.5
        edited = _routing_samples(router)  # a replaced weight is not the held one, so these take the native chain
        assert partitions
    assert "_unsloth_router_norm" not in router.__dict__
    with fusion.fused_moe_router(router):
        assert router._unsloth_router_norm.weight is not held.weight
        mx.eval(router(edited[0][0]))
        partitions.clear()
        _check(router, edited)
        assert not partitions


def _unpinned_call(self, x):
    return x


@pytest.mark.parametrize("reason", ["training", "top_k", "experts", "few_experts", "unpinned", "sharded", "no_kernel", "distributed"])
def test_ineligible_modules_keep_native(reason, monkeypatch):
    module = (_router(vlm_gemma, experts = 4, top_k = 2) if reason == "few_experts"
              else _block(lm_qwen, experts = 288 if reason == "experts" else 256, top_k = 10 if reason == "top_k" else 8))
    base = type(module)
    if reason == "training":
        module.train()
    elif reason == "unpinned":
        monkeypatch.setattr(base, "__call__", _unpinned_call)
        monkeypatch.setattr(fusion, "_moe_router_class", functools.cache(fusion._moe_router_class.__wrapped__))
    elif reason == "sharded":
        module.sharding_group = type("Group", (), {"size": lambda self: 1})()  # hashable for the cached native helper
    elif reason == "no_kernel":
        monkeypatch.setattr(fusion, "_moe_router_kernel", lambda: None)
    elif reason == "distributed":
        module._unsloth_mlx_distributed_parallel_mode = "pipeline"
    with fusion.fused_moe_router(module):
        if reason == "sharded":
            assert type(module) is not base
            assert fusion._moe_router_verified(mx.bfloat16, mx.float32, 256, 8, QWEN, True)
            monkeypatch.setattr(mx, "argpartition", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
            with pytest.raises(AssertionError):
                module(mx.random.normal((1, 1, 64)).astype(mx.bfloat16))
        else:
            assert type(module) is base
    assert type(module) is base


def test_train_mode_or_target_verify_inside_an_open_scope_uses_the_native_chain(monkeypatch):
    partitions = _counting_argpartition(monkeypatch)
    for module in (_block(vlm_qwen), _router(lm_gemma)):
        x = mx.random.normal((1, 2, 64)).astype(mx.bfloat16)
        with fusion.fused_moe_router(module):
            mx.eval(module(x))
            partitions.clear()
            if type(module)._unsloth_router_native.__call__.__defaults__:  # only mlx-vlm_qwen 0.6.0-0.6.15 takes target_verify
                mx.eval(module(x, target_verify = True))
                assert partitions
                partitions.clear()
            module.train()
            mx.eval(module(x))
            assert partitions
            partitions.clear()
            module.eval()
            mx.eval(module(x))
            assert not partitions


@pytest.mark.parametrize("native, body, build", [(vlm_qwen, "Qwen3_5MoeSparseMoeBlock", _block),
                                                 (vlm_gemma, "Router", _router)])
def test_body_rebound_or_globals_drifting_inside_an_open_scope_is_called_instead(monkeypatch, native, body, build):
    owner = getattr(native, body)
    pinned = owner.__call__
    twin = FunctionType(pinned.__code__, dict(pinned.__globals__), pinned.__name__, pinned.__defaults__)
    monkeypatch.setattr(owner, "__call__", twin)
    monkeypatch.setattr(fusion, "_moe_router_class", functools.cache(fusion._moe_router_class.__wrapped__))  # scoped cache
    block = build(native)
    x = mx.random.normal((1, 2, 64)).astype(mx.bfloat16)
    partitions = []
    with fusion.fused_moe_router(block):
        assert type(block) is not owner
        mx.eval(block(x))
        monkeypatch.setattr(owner, "__call__", _unpinned_call)
        assert block(x) is x
        monkeypatch.setattr(owner, "__call__", twin)
        twin.__globals__["mx"] = SimpleNamespace(**{**vars(mx), "argpartition": lambda *a, **k: partitions.append(None) or mx.argpartition(*a, **k)})
        mx.eval(block(x))
        assert partitions


def test_unverifiable_rounding_keeps_native_per_call(monkeypatch):
    block = _block(vlm_qwen)
    x = mx.random.normal((1, 2, 64)).astype(mx.bfloat16)
    expected = block(x)
    native = fusion._native_moe_router
    monkeypatch.setattr(fusion, "_moe_router_verified", functools.cache(fusion._moe_router_verified.__wrapped__))
    monkeypatch.setattr(fusion, "_native_moe_router", lambda *a: (mx.zeros((1, 8), dtype = mx.uint32), mx.zeros((1, 8))))
    with fusion.fused_moe_router(block):
        _identical(block(x), expected)
        assert not fusion._moe_router_verified(mx.bfloat16, mx.float32, 256, 8, QWEN, True)
    fusion._moe_router_verified.cache_clear()
    monkeypatch.setattr(fusion, "_native_moe_router", lambda *a: tuple(v.astype(mx.float32) for v in native(*a)))
    assert not fusion._moe_router_verified(mx.bfloat16, mx.float32, 256, 8, QWEN, True)  # equal values, wrong dtype
    fusion._moe_router_verified.cache_clear()
    monkeypatch.setattr(fusion, "_native_moe_router", native)
    lower_index_wins = lambda a, kth, axis: mx.argsort(a.astype(mx.float32) - mx.arange(a.shape[-1]) * 1e-8, axis = axis)
    monkeypatch.setattr(mx, "argpartition", lower_index_wins)
    assert not fusion._moe_router_verified(mx.float16, mx.float32, 64, 8, QWEN, True)  # only the integer probe ties


def test_nested_scopes_and_generation_mode_restore():
    modules = [_block(vlm_qwen), _block(lm_qwen), _router(lm_gemma)]
    root = nn.Sequential(*modules)
    root.eval()
    with pytest.raises(RuntimeError, match = "cancel"):
        with fusion.fused_moe_router(root):
            classes = [type(m) for m in modules]
            with fusion.fused_moe_router(root):
                assert [type(m) for m in modules] == classes
            assert [type(m) for m in modules] == classes
            raise RuntimeError("cancel")
    natives = [vlm_qwen.Qwen3_5MoeSparseMoeBlock, lm_qwen.Qwen3NextSparseMoeBlock, lm_gemma.Router]
    assert [type(m) for m in modules] == natives
    assert all("_unsloth_router_scopes" not in m.__dict__ for m in modules)
    with generation_mode(root):
        assert not any(type(m) is native for m, native in zip(modules, natives))
    assert [type(m) for m in modules] == natives


def test_contended_scopes_leave_no_module_patched():
    # The loader's generate() wrappers enter this scope without the lock generation_mode holds, so
    # two requests can be in the entry loop at once. Without _MOE_ROUTER_LOCK a lost increment or
    # decrement strands the patched class for the life of the process, and popping the count
    # between another thread's guard and its `+= 1` raises AttributeError out of generation.
    # Unpatched, this reproduced in 5 of 40 rounds; the switch interval makes it prompt.
    block = _block(lm_qwen)
    base = type(block)
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    failures = []

    def worker():
        try:
            for _ in range(40):
                with fusion.fused_moe_router(block):
                    pass
        except BaseException as error:          # noqa: BLE001 -- reported, not swallowed
            failures.append(error)

    try:
        for _ in range(25):
            threads = [threading.Thread(target = worker) for _ in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout = 60)
            assert not failures, failures[0]
            assert type(block) is base, "a patched class outlived every scope that asked for it"
            assert "_unsloth_router_scopes" not in block.__dict__
    finally:
        sys.setswitchinterval(previous)
