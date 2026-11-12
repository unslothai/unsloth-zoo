# SPDX-License-Identifier: AGPL-3.0-only
import contextlib
import functools
import gc
import inspect
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
try:
    from mlx_lm.models.activations import SwigluOAI as SwiGLU
except ImportError:  # mlx-lm < 0.32
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
        # NaN payload bits vary by hardware (bfloat16 0x7FFF on M1, 0x7FC0 on M3 Max), so only NaN positions are compared; all else stays bit-exact.
        assert mx.array_equal(mx.isnan(a), mx.isnan(b)), "the poisoned positions differ"
        quiet = mx.array(float("nan"), dtype = a.dtype)
        a, b = mx.where(mx.isnan(a), quiet, a), mx.where(mx.isnan(b), quiet, b)
    _equal(a, b)


def _plain_qwen4(width, dtype = mx.bfloat16, quantization = (4, 32), shared_gate = (8, 64), **config):
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpSparseMoeBlock
    if fusion._plain_moe_kernels() is None:
        pytest.skip("plain MoE kernels require the supported M3 Metal runtime")
    mx.random.seed(21)
    args = SimpleNamespace(hidden_size = 512, moe_intermediate_size = width, shared_expert_intermediate_size = width,
                           num_experts = 512, num_experts_per_tok = 10, norm_topk_prob = True)
    vars(args).update(config)
    block = Qwen4ExpSparseMoeBlock(args)
    block.set_dtype(dtype)
    block.gate = block.gate.to_quantized(group_size = 64, bits = 8)
    block.shared_expert_gate = block.shared_expert_gate.to_quantized(group_size = shared_gate[1], bits = shared_gate[0])
    nn.quantize(block, group_size = quantization[1], bits = quantization[0],
                class_predicate = lambda _, m: hasattr(m, "to_quantized") and not isinstance(m, nn.QuantizedLinear))
    block.eval(); block.freeze(); mx.eval(block.parameters())
    return block


@pytest.mark.parametrize("hidden,width,packed", [(512, 256, True), (1536, 640, True), (512, 256, False)])
@pytest.mark.parametrize("rows", [1, 2, 3, 7])
def test_plain_moe_qwen4_matches_stock_without_gather_graph(hidden, width, packed, rows, monkeypatch):
    block = _plain_qwen4(width, hidden_size = hidden)
    inputs = [mx.ones((1, rows, hidden), mx.bfloat16),
              *(mx.random.normal((1, rows, hidden)).astype(mx.bfloat16) * scale for scale in (3, -2)),
              mx.random.normal((1, rows, hidden * 2)).astype(mx.bfloat16)[..., ::2],
              mx.random.normal((1, rows * 2, hidden)).astype(mx.bfloat16)[:, ::2]]
    expected = [mx.concatenate([block(x[:, i:i + 1]) for i in range(rows)], axis = 1) for x in inputs]
    other = mx.random.normal((1, rows + 1, hidden)).astype(mx.bfloat16)
    other_expected = mx.concatenate([block(other[:, i:i + 1]) for i in range(rows + 1)], axis = 1)
    mx.eval(inputs, expected, other, other_expected); assert memoryview(inputs[-2]).strides[-1] == 2 * inputs[-2].itemsize
    with fusion.inference_scopes(block):
        if not packed: block.switch_mlp.__class__ = block.switch_mlp._unsloth_moe_gate_up.original_class
        experts = block._unsloth_moe_routed
        built, factory = [], fusion._plain_moe_plan
        monkeypatch.setattr(fusion, "_plain_moe_plan", lambda *a, **k: built.append(1) or factory(*a, **k))
        call = lambda x, ref: experts.verify(block, x, lambda: ref) if rows > 1 else block(x)
        _equal(call(inputs[0], expected[0]), expected[0])
        plan = experts.verify_plans[rows] if rows > 1 else experts.plain_plan
        assert plan is not None and plan.verdict is True and (plan.router_kernel is None) == (rows == 1)
        if rows > 1:
            checked = []
            _equal(experts.verify(block, other, lambda: checked.append(1) or other_expected), other_expected)
            assert checked == [1] and built == [1]
        calls = _counting_gather_qmm(monkeypatch)
        projected, gate_up = [], plan.gate_up
        plan.gate_up = lambda *, inputs, **kwargs: projected.append(inputs[1]) or gate_up(inputs = inputs, **kwargs)
        for x, y in zip(inputs[1:], expected[1:]):
            _equal(call(x, y), y)
            _equal(projected.pop(), mx.concatenate([block.gate(x[:, i:i + 1]) for i in range(rows)], axis = 1))
        assert calls == []


def test_verify_moe_qwen4_refuses_strided_native_router():
    block = _plain_qwen4(256)
    block.gate.weight = mx.stack([block.gate.weight] * 2, axis = -1)[..., 0]
    x = mx.ones((1, 3, 512), mx.bfloat16)
    reference = block(x)
    with fusion.inference_scopes(block):
        experts = block._unsloth_moe_routed
        assert fusion._plain_moe_plan(block, experts, 3) is None
        assert experts.verify(block, x, lambda: reference) is None  # the caller's routed path takes it


@pytest.mark.parametrize("method,where", [(m, w) for m in ("_linear", "_linears", "_switch_glu") for w in ("instance", "class")]
                         + [("swiglu", "global"), ("_linear", "mro"), ("_linear", "before_instance"), ("_linear", "before_class"), ("_linear", "cached_instance"), ("_linear", "code"), ("_linear", "closure"), ("_linear", "namespace"), ("_linear", "body")])
def test_verify_moe_qwen4_guards_native_verifier_dispatch(method, where, monkeypatch):
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpBatchInvariantForward as Native
    block, x, shift = _plain_qwen4(256), mx.ones((1, 3, 512), mx.bfloat16), [0]
    before = where.startswith("before") or where in ("namespace", "body")
    if where == "closure": monkeypatch.setenv("UNSLOTH_MLX_VERIFY_QMV_ROWS", "0")
    with fusion.inference_scopes(block), contextlib.ExitStack() as restore:
        if where == "namespace":
            fn = Native._linear.__globals__["_target_verify_linear"]
            monkeypatch.setitem(Native._linear.__globals__, "_target_verify_linear", FunctionType(fn.__code__, dict(fn.__globals__), closure = fn.__closure__))
        if where == "body":
            fn = Native._linear.__globals__["_target_verify_linear"]
            def unverified(linear, x): return True
            monkeypatch.setitem(fn.__globals__, "_use_target_verify_dense", FunctionType(unverified.__code__, fn.__globals__))
        if where == "before_class":
            linear = Native._linear
            monkeypatch.setattr(Native, "_linear", lambda self, *a: linear(self, *a) + shift[0])
        verifier = fusion._exact_verify_class(Native, "qwen4_exp", Native._linear, where)()
        if where == "mro": restore.callback(setattr, type(verifier), "__bases__", type(verifier).__bases__)
        monkeypatch.setattr(fusion._VERIFY_STATE, "serving", True, raising = False)
        monkeypatch.setattr(fusion._VERIFY_STATE, "plain_decode", False, raising = False)
        original = Native._feed_forward.__globals__[method] if where == "global" else getattr(verifier, method)
        def changed(*args):
            result = original(*args)
            return tuple(v + shift[0] for v in result) if isinstance(result, tuple) else result + shift[0]
        def replace():
            if where == "global": monkeypatch.setitem(Native._feed_forward.__globals__, method, changed)
            elif where == "closure":
                fn = verifier._linear.__func__
                cell = dict(zip(fn.__code__.co_freevars, fn.__closure__))["projected"]
                projected = cell.cell_contents
                monkeypatch.setattr(cell, "cell_contents", lambda *a: projected(*a) + 16)
            elif where == "code":
                def code():
                    base = eligible = projected = None
                    def changed(self, p, x):
                        (base, eligible, projected)
                        return projected(self, p, x) + 16
                    return changed.__code__
                replacement = code()
                assert replacement.co_freevars == verifier._linear.__func__.__code__.co_freevars
                monkeypatch.setattr(verifier._linear.__func__, "__code__", replacement)
            elif where == "mro": type(verifier).__bases__ = (type("Changed", (Native,), {method: lambda self, *a: changed(*a)}),)
            elif where.endswith("instance"): monkeypatch.setattr(verifier, method, changed)
            else: monkeypatch.setattr(type(verifier), method, lambda self, *a: changed(*a))
        if where == "before_instance": replace()
        _equal(verifier._feed_forward(block, x), Native._feed_forward(verifier, block, x))
        plan = block._unsloth_moe_routed.verify_plans.get(3)
        assert plan is None if before else plan.verdict is True
        if where == "cached_instance":
            context = fusion._moe_verifier_context(verifier)
            monkeypatch.setattr(fusion._VERIFY_STATE, "moe_verifier", (verifier, context, True), raising = False)
        if not before: replace()
        shift[0] = 16
        calls, run = [], fusion._PlainMoE.run
        monkeypatch.setattr(fusion._PlainMoE, "run", lambda self, x: (calls.append(1), run(self, x))[1])
        _equal(verifier._feed_forward(block, x), Native._feed_forward(verifier, block, x))
        assert calls == []


def test_verify_moe_dispatch_needs_no_unrelated_family(monkeypatch):
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
    monkeypatch.delitem(sys.modules, "mlx_vlm.models.qwen4_exp.language", raising = False)
    cls = type("OtherFamily", (Qwen3_5BatchInvariantForward,), {})
    assert fusion._moe_verifier_class(cls) is cls and cls not in fusion._MOE_VERIFY_DISPATCH


@pytest.mark.parametrize("change", ["training", "weight", "activation", "top_k", "projection", "quantization", "router_class", "shared_class", "mlp_class", "bias", "combine", "shared_gate", "router_bias", "router_training", "scale_global", "scale_namespace", "scale_shadow", "router_global", "switch_call", "mro_call", "instance_call", "pack_call", "pack_global", "bases",
    *((path, "__call__") for path in ("", "_unsloth_router_native", "switch_mlp", "shared_expert", "gate", "shared_expert_gate", "switch_mlp.activation",
                                      *(f"{parent}.{projection}" for parent in ("switch_mlp", "shared_expert") for projection in ("gate_proj", "up_proj", "down_proj"))))])
@pytest.mark.parametrize("rows", [1, 3])
def test_plain_moe_qwen4_refuses_changed_plan(change, rows, monkeypatch):
    block, x = _plain_qwen4(256), mx.ones((1, rows, 512), mx.bfloat16)
    reference = mx.concatenate([block(x[:, i:i + 1]) for i in range(rows)], axis = 1)
    if change == "scale_namespace":
        method = type(block)._shared_expert_scale
        monkeypatch.setattr(type(block), "_shared_expert_scale", FunctionType(method.__code__, dict(method.__globals__)))
    with fusion.inference_scopes(block), contextlib.ExitStack() as restore:
        if change == "bases": restore.callback(setattr, type(block.switch_mlp), "__bases__", type(block.switch_mlp).__bases__)
        routed = block._unsloth_moe_routed
        mx.eval(routed.verify(block, x, lambda: reference) if rows > 1 else block(x))
        plan = routed.verify_plans[rows] if rows > 1 else routed.plain_plan
        assert plan.verdict is True
        if change == "training": block.train()
        elif change == "weight": block.switch_mlp.down_proj.weight = mx.zeros_like(block.switch_mlp.down_proj.weight)
        elif change == "activation": block.switch_mlp.activation = lambda a, b: a + b
        elif change == "top_k": block.top_k = 9
        elif change == "quantization": block.shared_expert.down_proj.group_size = 64
        elif change == "router_class": block.gate.__class__ = type("ChangedRouter", (type(block.gate),), {})
        elif change == "shared_class": block.shared_expert.__class__ = type("ChangedShared", (type(block.shared_expert),), {})
        elif change == "mlp_class": block.switch_mlp.__class__ = type("EqualType", (type,), {"__eq__": lambda *a: True, "__hash__": type.__hash__})("ChangedMLP", (type(block.switch_mlp),), {})
        elif change == "bias": block.shared_expert.down_proj.bias = mx.ones((512,), mx.bfloat16)
        elif change == "combine": block.switch_mlp._combine = lambda *args: x
        elif change == "shared_gate": block.shared_expert_gate = block.gate
        elif change == "router_bias": block.gate.bias = mx.ones((512,), mx.float32)
        elif change == "router_training": block.gate.train()
        elif isinstance(change, tuple):
            target = fusion._resolve_dotted(block, change[0]) if change[0] else block
            monkeypatch.setattr(target if isinstance(target, type) else type(target), change[1], lambda *a, **kw: x, raising = False)
        elif change == "switch_call":
            call = type(block.switch_mlp).__call__
            monkeypatch.setattr(type(block.switch_mlp), "__call__", lambda self, *a, **kw: call(self, *a, **kw) + 16)
        elif change == "mro_call": monkeypatch.setattr(nn.Module, "__call__", lambda *a: x, raising = False)
        elif change == "instance_call": monkeypatch.setitem(vars(block.switch_mlp), "__call__", lambda *a: x)
        elif change == "pack_global": monkeypatch.setattr(fusion, "_nax_sorted_gather", lambda *a, **kw: None)
        elif change == "bases": type(block.switch_mlp).__bases__ = (type("ChangedMRO", type(block.switch_mlp).__bases__, {}),)
        elif change == "pack_call":
            project = type(block.switch_mlp._unsloth_moe_gate_up).project
            monkeypatch.setattr(type(block.switch_mlp._unsloth_moe_gate_up), "project", lambda self, *a, **kw: tuple(y + 16 for y in project(self, *a, **kw)))
        elif change == "scale_shadow": monkeypatch.setattr(type(block), "_shared_expert_scale", lambda self, a: mx.ones((1, 1, 1)), raising = False)
        elif change == "router_global": monkeypatch.setitem(nn.QuantizedLinear.__call__.__globals__, "mx", SimpleNamespace(**vars(mx)))
        elif change in ("scale_global", "scale_namespace"):
            method = type(block)._shared_expert_scale
            monkeypatch.setitem(method.__globals__, "_row_invariant_quantized_projection", lambda linear, a: linear(a) + 1)
        else: block.shared_expert.down_proj = block.shared_expert.gate_proj
        assert plan(block, x, lambda: None) is None
        assert change == "training" or (routed.verify if rows > 1 else routed.plain)(block, x, lambda: x) is x
        if change in ("scale_global", "scale_namespace", "switch_call", "pack_call", "pack_global"): _equal(block(x), type(block)._unsloth_router_native.__call__(block, x))


@pytest.mark.parametrize("config", [{"num_experts": 256}, {"num_experts_per_tok": 8},
                                    {"norm_topk_prob": False}, {"shared_expert_intermediate_size": 128},
                                    {"hidden_size": 384}, {"hidden_size": 768}, {"training": True}, {"dtype": mx.float16}, {"quantization": (8, 32)},
                                    {"quantization": (4, 64)}, {"shared_gate": (4, 64)}, {"shared_gate": (8, 32)}, {"router_dtype": mx.float32},
                                    *({"narrow": (m, p)} for m, p in (("switch_mlp", "up_proj"), ("shared_expert", "up_proj"), ("shared_expert", "down_proj")))])
@pytest.mark.parametrize("rows", [1, 3])
def test_plain_moe_qwen4_leaves_other_configs_native(config, rows):
    config = dict(config)
    narrow, training, router_dtype = config.pop("narrow", None), config.pop("training", False), config.pop("router_dtype", None)
    block = _plain_qwen4(256, **config)
    if router_dtype: block.gate.scales, block.gate.biases = block.gate.scales.astype(router_dtype), block.gate.biases.astype(router_dtype)
    if narrow:
        projection = getattr(getattr(block, narrow[0]), narrow[1])
        for name in ("weight", "scales", "biases"): projection[name] = projection[name][..., :1, :]
    if training: block.shared_expert.gate_proj.train()
    mx.eval(block(mx.ones((1, 1, block.gate.weight.shape[1] * 32 // block.gate.bits), block.switch_mlp.gate_proj.scales.dtype)))
    with fusion.inference_scopes(block):
        assert fusion._plain_moe_plan(block, block._unsloth_moe_routed, rows) is None


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("change", ["class", "global", "instance"])
def test_plain_moe_qwen4_rechecks_shared_bindings_between_steps(change, rows, monkeypatch):
    from mlx_vlm.models.qwen4_exp.language import LanguageModel, Qwen4ExpBatchInvariantForward
    owner = LanguageModel.__new__(LanguageModel)
    nn.Module.__init__(owner)
    owner.blocks = [_plain_qwen4(256) for _ in range(2)]
    owner.eval()
    x, ids = mx.ones((1, rows, 512), mx.bfloat16), mx.ones((1, rows), mx.int32)
    if rows > 1:
        monkeypatch.setenv("UNSLOTH_MLX_VERIFY_MOE", "1")
        owner._unsloth_speculative_call = True
    entered = fusion._exact_verify_class(type("Steps", (Qwen4ExpBatchInvariantForward,), {"__call__": lambda self, lm, inputs: tuple(self._feed_forward(block, x) for block in lm.blocks)}), "qwen4_exp")()
    with fusion.inference_scopes(owner):
        mx.eval(entered(owner, ids))
        plans = [block._unsloth_moe_routed.plain_plan if rows == 1 else block._unsloth_moe_routed.verify_plans[rows] for block in owner.blocks]
        shared = plans[0].shared
        assert all(plan.verdict is True for plan in plans) and shared is plans[1].shared is owner._unsloth_plain_moe_shared and all(a[0] is b[0] for a, b in zip(plans[0].shared_bindings, plans[1].shared_bindings))
        checks, check = [], fusion._PlainMoEShared.check
        guards = [entered._unsloth_moe_verify_context[1].shared, shared] if rows > 1 else [shared]
        monkeypatch.setattr(fusion._PlainMoEShared, "check", lambda self: (checks.append(self), check(self))[1])
        mx.eval(entered(owner, ids))
        assert checks == guards
        if change == "class":
            call = type(owner.blocks[1].switch_mlp).__call__
            monkeypatch.setattr(type(owner.blocks[1].switch_mlp), "__call__", lambda self, *a, **kw: call(self, *a, **kw) + 16)
        elif change == "global": monkeypatch.setattr(nn, "silu", lambda a: a + 1)
        else: owner.blocks[1].switch_mlp.down_proj.weight = mx.zeros_like(owner.blocks[1].switch_mlp.down_proj.weight)
        monkeypatch.setattr(fusion._PlainMoE, "run", lambda self, rows, run = fusion._PlainMoE.run: (checks.append(self), run(self, rows))[1])
        _equal(mx.stack(entered(owner, ids)), mx.stack([Qwen4ExpBatchInvariantForward()._feed_forward(block, x) for block in owner.blocks]))
        assert checks == guards + guards + ([plans[0]] if change == "instance" else [])
        assert getattr(fusion._VERIFY_STATE, "plain_moe", None) is None
    assert "_unsloth_plain_moe_shared" not in vars(owner)


@pytest.mark.parametrize("shared_gate", [False, True, "shared", "router", "scale", "scale_helper", "scale_namespace", "switch_call", "router_call", "switch_bases", "pack_call"])
def test_plain_moe_qwen4_keeps_custom_shared_projections_native(shared_gate, monkeypatch):
    block, x = _plain_qwen4(256), mx.ones((1, 1, 512), mx.bfloat16)
    projection = block.switch_mlp if shared_gate in ("switch_call", "switch_bases", "pack_call") else block if shared_gate in ("scale", "scale_helper", "scale_namespace", "router_call") else block.gate if shared_gate == "router" else block.shared_expert if shared_gate == "shared" else block.shared_expert_gate if shared_gate else block.shared_expert.gate_proj
    name = "_shared_expert_scale" if shared_gate in ("scale", "scale_helper", "scale_namespace") else "__call__"
    native = getattr(type(projection), name)
    if shared_gate == "scale":
        shifted = lambda self, a: native(self, a) + self.shift
        monkeypatch.setattr(type(projection), name, FunctionType(shifted.__code__, native.__globals__, closure = shifted.__closure__))
    elif shared_gate == "scale_helper":
        monkeypatch.setitem(native.__globals__, "_row_invariant_quantized_projection", lambda linear, a: linear(a) + block.shift)
    elif shared_gate == "scale_namespace":
        monkeypatch.setattr(type(block), name, FunctionType(native.__code__, {**native.__globals__, "_row_invariant_quantized_projection": lambda linear, a: linear(a) + block.shift}))
    elif shared_gate == "pack_call":
        project = fusion._PackedMoEGateUp.project
        monkeypatch.setattr(fusion._PackedMoEGateUp, "project", lambda self, *a, **kw: tuple(y + block.switch_mlp.shift for y in project(self, *a, **kw)))
    elif shared_gate not in ("switch_call", "router_call", "switch_bases"):
        projection.__class__ = type("ShiftedProjection", (type(projection),), {name: lambda self, a: native(self, a) + self.shift})
    projection.shift = 0
    with fusion.inference_scopes(block), contextlib.ExitStack() as restore:
        if shared_gate == "switch_bases": restore.callback(setattr, type(projection), "__bases__", type(projection).__bases__)
        if shared_gate == "switch_bases": type(projection).__bases__ = (type("ChangedMRO", type(projection).__bases__, {}),)
        if shared_gate in ("switch_call", "router_call"):
            native = type(projection).__call__
            monkeypatch.setattr(type(projection), name, lambda self, *a, **kw: native(self, *a, **kw) + self.shift)
        mx.eval(block(x))
        assert getattr(getattr(block, "_unsloth_moe_routed", None), "plain_plan", None) is None
        projection.shift = 1
        expected = type(block)._unsloth_router_native.__call__(block, x)
        _equal(block(x), expected + projection.shift if shared_gate == "router_call" else expected)


@pytest.mark.parametrize("target,name", [("shared_expert.__class__", "__call__"), ("_unsloth_router_native", "_shared_expert_scale"), ("QuantizedLinear", "__call__"), ("_unsloth_router_native", "__call__"),
    ("switch_mlp._unsloth_moe_gate_up.original_class", "__call__"), ("switch_mlp.activation.__class__", "__call__"), ("switch_mlp._unsloth_moe_gate_up.original_class", "_combine"), ("__class__", "__call__"), ("switch_mlp.__class__", "__call__"), ("switch_mlp._unsloth_moe_gate_up.__class__", "project")])
def test_plain_moe_qwen4_refuses_cached_contract_drift(target, name, monkeypatch):
    block = _plain_qwen4(256)
    with fusion.inference_scopes(block):
        experts = block._unsloth_moe_routed
        assert (initial := fusion._plain_moe_plan(block, experts)) is not None
        resolve, contract = fusion._resolved_bindings, {**fusion._PLAIN_MOE_CONTRACT, **fusion._DENSE_QMM_CONTRACT}
        monkeypatch.setattr(fusion, "_resolved_bindings", lambda query, snapshot = resolve(contract): snapshot if query == contract else resolve(query))
        holder = fusion._resolve_dotted(nn if target == "QuantizedLinear" else block, target)
        original, raw, shift = getattr(holder, name), inspect.getattr_static(holder, name), [0]
        def changed(*args, **kwargs):
            output = original(*args, **kwargs)
            return tuple(value + shift[0] for value in output) if isinstance(output, tuple) else output + shift[0]
        replacement = FunctionType(changed.__code__, original.__globals__, closure = changed.__closure__)
        monkeypatch.setattr(holder, name, staticmethod(replacement) if isinstance(raw, staticmethod) else replacement)
        assert fusion._plain_moe_plan(block, experts) is None
        assert not fusion._PlainMoE(block, experts, initial.shared_bindings).shared.check()


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("failure", ["bits", "build", "compile", "plan"])
def test_plain_moe_qwen4_failed_actual_verdict_stays_native(monkeypatch, failure, rows):
    block, x = _plain_qwen4(256), mx.ones((1, rows, 512), mx.bfloat16)
    reference = mx.concatenate([block(x[:, i:i + 1]) for i in range(rows)], axis = 1)
    run, ready, staged = fusion._PlainMoE.run, [], fusion._staged_ready
    monkeypatch.setattr(fusion, "_staged_ready", lambda: ready.append(True) or staged())
    def wrong(self, value, *args):
        assert failure == "plan" or ready
        if failure in ("build", "plan"): raise ValueError("candidate construction failed")
        if failure == "compile":
            kernel = mx.fast.metal_kernel(name = "unsloth_plain_moe_test_failure", input_names = ["x"], output_names = ["y"], source = "invalid_plain_moe_source;")
            return kernel(inputs = [value], grid = (32, 1, 1), threadgroup = (32, 1, 1), output_shapes = [value.shape], output_dtypes = [value.dtype])[0]
        return run(self, value) + 1
    monkeypatch.setattr(fusion._PlainMoE, "__init__" if failure == "plan" else "run", wrong)
    with fusion.inference_scopes(block):
        ready.clear()
        experts = block._unsloth_moe_routed
        call = lambda: experts.verify(block, x, lambda: reference) if rows > 1 else block(x)
        def check():  # an unplanned verify block returns None: the caller's routed path takes it
            out = call()
            assert out is None if failure == "plan" and rows > 1 else _equal(out, reference) is None
        check()
        plan = experts.verify_plans[rows] if rows > 1 else experts.plain_plan
        assert (plan is None if failure == "plan" else plan.verdict is False)
        check()


def test_plain_moe_keeps_qwen35_on_its_existing_route():
    block = _expert_block(vlm_qwen, (4, "affine", 32), mx.bfloat16, experts = 512)
    with fusion.inference_scopes(block):
        assert not block._unsloth_moe_routed.plain_enabled


def test_plain_moe_qwen4_leaves_speculative_calls_native():
    block = _plain_qwen4(256)
    with fusion.inference_scopes(block):
        block.__dict__["_unsloth_speculative_call"] = 1
        routed = block._unsloth_moe_routed
        assert routed.plain(block, mx.ones((1, 1, 512), mx.bfloat16), lambda: None) is None
        assert not routed.plain_built


@pytest.mark.parametrize("low", [0, 4])
def test_verify_moe_qwen4_takes_fast_verify_blocks_unless_lanes_hold_them(low, monkeypatch):
    block = _plain_qwen4(256)
    inputs = {rows: mx.ones((1, rows, 512), mx.bfloat16) for rows in (3, 5)}
    references = {rows: mx.concatenate([block(x[:, i:i + 1]) for i in range(rows)], axis = 1) for rows, x in inputs.items()}
    mx.eval(references)
    monkeypatch.setattr(fusion, "_LANE_LOW_ROWS", low)
    with fusion.inference_scopes(block):
        for p in (block.shared_expert.gate_proj, block.shared_expert.up_proj, block.shared_expert.down_proj):
            p._unsloth_lane_verify_scopes = 1
            p.__class__ = fusion._lane_module_class(type(p))
        experts = block._unsloth_moe_routed
        assert (fusion._plain_moe_plan(block, experts, 3) is not None) == bool(low)
        for rows in (3, 5):   # a plan admitted within the threshold is not reused past it
            _equal(experts.verify(block, inputs[rows], lambda: references[rows]), references[rows])
        assert [experts.verify_plans[rows] is not None for rows in (3, 5)] == [bool(low), False]


@pytest.mark.parametrize("width", [64, 128])
def test_plain_moe_qwen4_keeps_native_quad_shared_projection(width):
    block, x = _plain_qwen4(width), mx.ones((1, 1, 512), mx.bfloat16)
    expected = block(x)
    with fusion.inference_scopes(block):
        _equal(block(x), expected)
        assert block._unsloth_moe_routed.plain_plan is None


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
            # Packing folds gate and up into one projection, except short sequences belonging to an upstream decode path this body does not reimplement.
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
    fused = [fusion._fused_moe_gate_up_class(lm.SwitchGLU, kind, None, 0, []) for kind in
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
def test_packing_that_fails_partway_leaves_every_module_native(native, monkeypatch):
    """A refused scope must put back the modules it had already patched."""
    first, second = _model(native), _model(native)
    root = nn.Sequential(first, second)
    root.eval()
    before = [set(module.__dict__) for module in (first, second)]
    x, indices = _sample()
    expected = [module(x, indices) for module in (first, second)]
    mx.eval(expected)

    native_init, built = fusion._PackedMoEGateUp.__init__, []

    def refuse_the_second(self, module):
        built.append(module)
        if len(built) > 1:
            raise RuntimeError("injected: no headroom")
        native_init(self, module)

    monkeypatch.setattr(fusion._PackedMoEGateUp, "__init__", refuse_the_second)
    with pytest.raises(RuntimeError, match = "injected"):
        with fused_moe_gate_up(root):
            pass  # pragma: no cover
    assert len(built) == 2
    for module, keys, answer in zip((first, second), before, expected):
        assert type(module) is native.SwitchGLU
        assert set(module.__dict__) == keys
        _equal(module(x, indices), answer)


def _cache_limit():
    current = mx.set_cache_limit(0)
    mx.set_cache_limit(current)
    return current


@pytest.mark.parametrize("native", [lm, vlm])
def test_packing_retains_nothing_it_replaces(native, monkeypatch):
    """Packing must retain nothing it replaces: replaced arrays cached rather than returned would be a second copy of the packed half of the model."""
    modules = [_model(native) for _ in range(4)]
    root = nn.Sequential(*modules)
    root.eval()
    one_layer = max(
        sum(module.gate_proj[name].nbytes + module.up_proj[name].nbytes
            for name in fusion._MOE_PROJECTION_FIELDS if module.gate_proj.get(name) is not None)
        for module in modules
    )

    footprints = []
    attach = fusion._PackedMoEGateUp.attach

    def record(self):
        attach(self)
        footprints.append(mx.get_active_memory() + mx.get_cache_memory())

    monkeypatch.setattr(fusion._PackedMoEGateUp, "attach", record)
    gc.collect()
    mx.clear_cache()
    # room for every layer, so no inherited limit explains a flat footprint
    previous_limit = mx.set_cache_limit(8 * one_layer)
    try:
        before = mx.get_active_memory() + mx.get_cache_memory()
        with fused_moe_gate_up(root):
            end = mx.get_active_memory() + mx.get_cache_memory()
            fused = [type(module) for module in modules]
    finally:
        mx.set_cache_limit(previous_limit)

    assert all(cls is not native.SwitchGLU for cls in fused)
    assert len(footprints) == len(modules)
    assert [footprint - before for footprint in footprints] == [0] * len(modules)
    assert end == before


@pytest.mark.parametrize("native", [lm, vlm])
def test_packing_restores_the_cache_limit_it_borrowed(native):
    """Process-global: left at zero it would stop MLX reusing any buffer at all."""
    root = nn.Sequential(_model(native))
    root.eval()
    borrowed = 8 << 20
    previous_limit = mx.set_cache_limit(borrowed)
    try:
        with fused_moe_gate_up(root):
            assert _cache_limit() == borrowed
        assert _cache_limit() == borrowed
    finally:
        mx.set_cache_limit(previous_limit)


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


@pytest.mark.skipif("global_scale" not in inspect.signature(lm.SwitchLinear.to_quantized).parameters,
                    reason = "Requires an mlx-lm with nvfp4 global scales")
def test_global_scale_projections_keep_the_native_path():
    # gate and up each carry their own per-expert scale, which one packed matmul cannot apply
    mx.random.seed(19)
    model = lm.SwitchGLU(2048, 512, 8, bias = False)
    model.set_dtype(mx.bfloat16)
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(model, name, getattr(model, name).to_quantized(16, 4, mode = "nvfp4", global_scale = True))
    model.eval()
    model.freeze()
    samples = _samples(model, mx.bfloat16, 2048, [(1, 1), (1, 32)])
    with fused_moe_gate_up(model):
        assert type(model) is lm.SwitchGLU
        for x, indices, expected in samples:
            _equal(model(x, indices), expected)


def test_routing_bodies_cover_the_installed_packages():
    # an unlisted routing body keeps its native call without a word, so it has to fail here
    from mlx_vlm.models.gemma4_text import language as vlm_gemma_text
    from mlx_vlm.models.qwen3_next import language as vlm_next
    for router in (vlm_qwen.Qwen3_5MoeSparseMoeBlock, vlm_next.Qwen3NextSparseMoeBlock,
                   lm_qwen.Qwen3NextSparseMoeBlock, vlm_gemma.Router, vlm_gemma_text.Router, lm_gemma.Router):
        assert fusion._moe_router_class(router) is not None, router.__module__


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
@pytest.mark.parametrize("experts, top_k", [(32, 1), (64, 5), (128, 8), (160, 3), (256, 8), (288, 8), (512, 10), (1024, 8), (1024, 16)])
@pytest.mark.parametrize("mode, normalize", [(QWEN, True), (QWEN, False), (GEMMA, False)])
def test_kernel_reproduces_native_routing(dtype, experts, top_k, mode, normalize):
    for rows in (0, 1, 8, 4096):
        mx.random.seed(rows + experts + top_k)
        logits, scale = _inputs(dtype, experts, rows, mode)
        fused = fusion._fused_moe_router(logits, scale, top_k, mode, normalize)
        if rows == 0 or (mode == GEMMA and top_k > 8):
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


def test_unnormalized_wide_and_subclass_bodies_are_fused(monkeypatch):
    class Scaled(vlm_qwen.Qwen3_5MoeSparseMoeBlock):  # qwen4_exp inherits the body and overrides the shared gate
        def _shared_expert_scale(self, x):
            return 2 * super()._shared_expert_scale(x)

    partitions = _counting_argpartition(monkeypatch)
    # qwen3_next and qwen4_exp route 512 experts to 10
    for block in (_block(lm_qwen, norm_topk_prob = False), _block(vlm_qwen, norm_topk_prob = False),
                  _block(vlm_qwen, cls = Scaled), _block(lm_qwen, experts = 512, top_k = 10),
                  _block(vlm_qwen, experts = 512, top_k = 10, cls = Scaled)):
        samples = _routing_samples(block)
        base = type(block)
        with fusion.fused_moe_router(block):
            assert type(block) is not base and isinstance(block, base)
            mx.eval(block(samples[0][0]))
            partitions.clear()
            _check(block, samples)
            assert not partitions


def _native_combine(y, inv_order, scores):
    return (y[inv_order].reshape(*scores.shape, y.shape[-1]) * scores[..., None]).sum(axis = -2)


@pytest.mark.parametrize("dtype", fusion._MOE_COMBINE_DTYPES)
@pytest.mark.parametrize("top_k, width", [(8, 2048), (3, 96), (10, 2560)])  # past eight routes MLX sums in lanes
def test_combine_kernel_reproduces_the_native_weighted_sum(dtype, top_k, width):
    for tokens in (1, 300):
        mx.random.seed(tokens + top_k)
        y = (mx.random.normal((tokens * top_k, 1, width)) * 3).astype(dtype)
        scores = mx.softmax(mx.random.normal((1, tokens, top_k)) * 2, axis = -1).astype(dtype)
        y[..., 0] = -0.0
        inv_order = mx.random.permutation(tokens * top_k)
        _identical(fusion._run_moe_combine(y, inv_order, scores), _native_combine(y, inv_order, scores))
    assert not fusion._moe_combine_verified(dtype, 1, width)  # one route is not summed, so its zero keeps its sign
    fusion._moe_combine_verified.cache_clear()


@pytest.mark.parametrize("native", [vlm_qwen, lm_qwen])
@pytest.mark.parametrize("verified", [True, False])
def test_prefill_combines_sorted_expert_rows_under_the_gate_up_pack(native, verified, monkeypatch):
    _skip_unless_gate_up_resolves(native)
    block = _block(native)
    prefill, decode = (mx.random.normal(shape).astype(mx.bfloat16) for shape in ((2, 70, 64), (9, 1, 64)))
    expected = block(prefill), block(decode)
    combines, run = [], fusion._run_moe_combine
    monkeypatch.setattr(fusion, "_run_moe_combine",
                        lambda *a: combines.append(None) or run(*a) + (0 if verified else 1))
    fusion._moe_combine_verified.cache_clear()
    with fusion.fused_moe_router(block):
        _identical(block(prefill), expected[0])
        assert not combines  # without the pack the experts are not reachable in sorted order
        with fusion.fused_moe_gate_up(block):
            mx.eval(block(prefill))  # the first call verifies the kernel against the native chain
            combines.clear()
            _identical(block(prefill), expected[0])
            assert bool(combines) is verified
            combines.clear()
            _identical(block(decode), expected[1])
            with monkeypatch.context() as capped:  # past 32-bit indexing the native chain runs
                capped.setattr(fusion, "_MOE_COMBINE_MAX_ELEMENTS", 2 * 70 * 8 * 64 - 1)
                _identical(block(prefill), expected[0])
                assert not combines
            switch = vlm if native is vlm_qwen else lm
            sort = switch._gather_sort
            with monkeypatch.context() as rebound:  # a pinned helper replaced after the scope resolved it
                rebound.setattr(switch, "_gather_sort", lambda *a: sort(*a))
                _identical(block(prefill), expected[0])
            block.switch_mlp.gate_proj.scales = block.switch_mlp.gate_proj.scales * 1.25  # the pack is now stale
            stale = block(prefill)
            assert not combines
    _identical(stale, block(prefill))
    fusion._moe_combine_verified.cache_clear()


@contextlib.contextmanager
def _routed(block):
    with fusion.fused_moe_router(block), fusion.fused_moe_routed_experts(block):
        yield


@contextlib.contextmanager
def _routed_speculative(block):
    with _routed(block), fusion.speculative_call(block):
        yield


def _expert_block(native, quantization, dtype, widths = (512, 512), experts = 32):
    mx.random.seed(23)
    args = SimpleNamespace(hidden_size = widths[0], moe_intermediate_size = widths[1], num_experts = experts, num_experts_per_tok = 8,
                           shared_expert_intermediate_size = 512, norm_topk_prob = True)
    block = (vlm_qwen.Qwen3_5MoeSparseMoeBlock if native is vlm_qwen else lm_qwen.Qwen3NextSparseMoeBlock)(args)
    block.set_dtype(dtype)
    bits, mode, group_size = quantization
    nn.quantize(block.switch_mlp, bits = bits, group_size = group_size, mode = mode)
    nn.quantize(block, bits = 8, group_size = 32)
    block.eval()
    mx.eval(block.parameters())
    return block


@pytest.mark.parametrize("native", [vlm_qwen, lm_qwen])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
# Widths where MLX runs qmv for a K its bit width's qmv_fast step does not divide (512 at 4 bits, 256 at 8); here down, gate/up, both or neither at 8 bits.
@pytest.mark.parametrize("quantization, widths", [((8, "affine", 64), (512, 512)), ((4, "affine", 32), (512, 512)), ((4, "mxfp4", 32), (512, 512)),
                                                  ((4, "nvfp4", 16), (512, 512)), ((4, "affine", 32), (512, 640)), ((4, "affine", 32), (640, 512)),
                                                  ((8, "affine", 32), (640, 640)), ((6, "affine", 32), (640, 640)), ((2, "affine", 32), (640, 640)),
                                                  ((8, "affine", 64), (768, 768))])
def test_routed_experts_run_on_the_decode_kernels_bitwise(native, dtype, quantization, widths, monkeypatch):
    block = _expert_block(native, quantization, dtype, widths)
    # decode rows, the last row alone, and the first sorted width, whose 64 routes stay native
    xs = [mx.random.normal((rows, 1, widths[0])).astype(dtype) for rows in (1, 7, 8)]
    xs.insert(2, xs[1][6:])
    expected = [block(x) for x in xs]
    gathers = _counting_gather_qmm(monkeypatch)
    router, routed = fusion.fused_moe_router, fusion.fused_moe_routed_experts
    pack = fusion.fused_moe_gate_up  # the pack leaves strided expert views
    for scopes in ((router,), (router, routed), (routed, router), (pack, router, routed), (router, routed, routed, pack)):
        with contextlib.ExitStack() as stack:
            for scope in scopes:
                stack.enter_context(scope(block))
            mx.eval(block(xs[0]))  # the first call verifies the kernels against the native experts
            for x, out in zip(xs, expected):
                gathers.clear()
                _identical(block(x), out)
                assert bool(gathers) is (routed not in scopes or x.shape[0] == 8)
        assert "_unsloth_moe_routed" not in block.__dict__


# 64 experts take the tile map across two simdgroups.
@pytest.mark.parametrize("quantization, widths, experts", [((8, "affine", 64), (512, 512), 32), ((8, "affine", 64), (512, 512), 64),
                                                           ((4, "affine", 32), (640, 512), 32), ((4, "affine", 32), (512, 640), 32),
                                                           ((6, "affine", 32), (640, 640), 32)])
def test_grouped_routed_experts_serve_bitwise(quantization, widths, experts, monkeypatch):
    monkeypatch.setattr(fusion.nax, "_nax_gpu", lambda: True)
    block = _expert_block(vlm_qwen, quantization, mx.bfloat16, widths, experts)
    xs = [mx.random.normal((rows, 1, widths[0])).astype(mx.bfloat16) for rows in (1, 2, 5, 7)]
    expected = [block(x) for x in xs]
    grouped, served = fusion._grouped_routed, []
    monkeypatch.setattr(fusion, "_grouped_routed", lambda *a: served.append(out := grouped(*a)) or out)
    monkeypatch.setattr(fusion, "_MOE_ROUTED_VERDICTS", {})
    monkeypatch.setattr(fusion, "_MOE_GROUPED_CHOICES", {})
    monkeypatch.setattr(fusion, "_grouped_routed_faster", lambda routed, grouped: True)
    with _routed(block):   # a plain call keeps its route
        _identical(block(xs[1]), expected[1])
        assert not served
    with _routed_speculative(block):
        mx.eval(block(xs[0]))  # the first call verifies the kernels against the native experts
        for x, out in zip(xs, expected):
            served.clear()
            _identical(block(x), out)
            assert [out is not None for out in served] == [True] * (x.shape[0] > 1)
    # It serves only what the routed kernels do: routing weights in another dtype are refused.
    mlp = block.switch_mlp
    tensors = tuple(map(fusion._projection_tensors, (mlp.gate_proj, mlp.up_proj, mlp.down_proj)))
    x, inds = xs[2], mx.zeros((5, 1, 8), mx.uint32)
    scores, shared, s = mx.full((5, 1, 8), 0.125, mx.bfloat16), mx.ones((5, 1, 1), mx.bfloat16), mx.zeros((5, 1, widths[0]), mx.bfloat16)
    args = (mlp.gate_proj.group_size, mlp.gate_proj.bits)
    routed = fusion._routed_gate_up(x, inds, *tensors[:2], *args, "affine")
    _identical(grouped(x, inds, scores, shared, s, tensors, *args),
               fusion._routed_down(routed, inds, scores, shared, s, tensors[2], *args, "affine"))
    assert grouped(x, inds, scores.astype(mx.float32), shared, s, tensors, *args) is None



@pytest.mark.parametrize("tensor_units, rows, raced", [(False, 2, False), (False, 8, False), (False, 9, True), (True, 2, True), (True, 8, True)])
def test_grouped_tiles_are_raced_only_where_they_can_win(monkeypatch, tensor_units, rows, raced):
    monkeypatch.setattr(fusion.nax, "_nax_gpu", lambda: tensor_units)
    block = _expert_block(vlm_qwen, (8, "affine", 64), mx.bfloat16)
    mlp = block.switch_mlp
    tensors = tuple(map(fusion._projection_tensors, (mlp.gate_proj, mlp.up_proj, mlp.down_proj)))
    races = []
    monkeypatch.setattr(fusion, "_MOE_GROUPED_CHOICES", {})
    monkeypatch.setattr(fusion, "_grouped_routed_faster", lambda routed, grouped: races.append(True) or False)
    x = mx.random.normal((rows, 1, 512)).astype(mx.bfloat16)
    inds = mx.random.randint(0, 32, (rows, 1, 8)).astype(mx.uint32)
    scores = mx.softmax(mx.random.normal((rows, 1, 8)), axis = -1).astype(mx.bfloat16)
    mx.eval(fusion._moe_routed_experts(block)._run(tensors, x, inds, scores, mx.ones((rows, 1, 1), mx.bfloat16), mx.zeros_like(x), select = True))
    assert bool(races) is raced


@pytest.mark.parametrize("routed_times, grouped_times, faster", [
    ((2,) * 9, (1,) * 9, True), ((2,) * 9, (3,) * 9, False), ((2,) * 9, (2,) * 9, False),
    ((2,) * 9, (1,) * 8 + (100,), True), ((2,) * 9, (1,) * 7 + (100,) * 2, False),
    ((1,) * 5 + (4,) * 4, (2,) * 9, True), ((2,) * 9, (3,) * 3 + (1,) * 6, True), ((2,) * 9, (0,) + (2,) * 8, True),
])
def test_grouped_selection_checks_equality_then_keeps_the_measured_winner(monkeypatch, routed_times, grouped_times, faster):
    values = mx.array([1, float("nan"), 3], mx.bfloat16)
    calls, times = [], []
    assert fusion._GROUPED_RACE_CALLS == len(routed_times) == len(grouped_times)
    for repeat in range(fusion._GROUPED_RACE_CALLS):
        for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            duration = (routed_times, grouped_times)[index][repeat]
            times.extend([10 * len(times), 10 * len(times) + duration])
    ticks = iter(times)
    def run(index, different = False):
        calls.append(index)
        return values + int(different)
    assert fusion._grouped_routed_faster(lambda: run(0), lambda: run(1), lambda: next(ticks)) is faster
    assert calls == [0, 1] + [0, 1, 1, 0] * 4 + [0, 1]
    calls.clear()
    assert not fusion._grouped_routed_faster(lambda: run(0), lambda: run(1, True), lambda: pytest.fail("timed before bitwise gate"))
    assert calls == [0, 1]
    assert not fusion._grouped_routed_faster(lambda: values, lambda: None)
    def unavailable():
        raise RuntimeError("unsupported grouped kernel")
    assert not fusion._grouped_routed_faster(lambda: values, unavailable)

def test_verify_blocks_run_their_routed_experts_on_the_decode_kernels_bitwise(monkeypatch):
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward as native
    block = _expert_block(vlm_qwen, (8, "affine", 64), mx.bfloat16)
    xs = [mx.random.normal(shape).astype(mx.bfloat16) for shape in ((2, 5, 512), (1, 9, 512))]
    expected = [native()._feed_forward(block, x) for x in xs]
    verifier = fusion._exact_verify_class(native, "qwen3_5", *(vars(native).get(name) for name in fusion._VERIFY_METHODS))()
    monkeypatch.setattr(fusion, "_VERIFY_VERDICTS", {})
    gathers, partitions = _counting_gather_qmm(monkeypatch), _counting_argpartition(monkeypatch)
    with fusion.fused_moe_routed_experts(block):
        for serving in (False, True):
            monkeypatch.setattr(fusion._VERIFY_STATE, "serving", serving, raising = False)
            for x, out in zip(xs, expected):
                mx.eval(verifier._feed_forward(block, x))  # a served shape's first block is compared and returned native
                gathers.clear()
                partitions.clear()
                _identical(verifier._feed_forward(block, x), out)
                assert len(gathers) == (0 if serving else 4)  # mlx-vlm gathers per expert and for the shared gate
                assert bool(partitions) is not serving  # served blocks route in the fused router's launch
        assert len(fusion._VERIFY_VERDICTS) >= 2 and all(fusion._VERIFY_VERDICTS.values())
        block.switch_mlp.down_proj.unfreeze()
        gathers.clear()
        _identical(verifier._feed_forward(block, xs[0]), expected[0])
        assert len(gathers) == 3
        # A projection swapped for one the kernels were not built for takes the native call as well.
        block.switch_mlp.down_proj = vlm.SwitchLinear(512, 512, 32, bias = False)
        block.switch_mlp.down_proj.set_dtype(mx.bfloat16)
        block.eval().freeze()
        _identical(verifier._feed_forward(block, xs[0]), native()._feed_forward(block, xs[0]))


def _fold_block(native, bits, group_size, dtype = mx.bfloat16):
    block = _expert_block(native, (bits, "affine", group_size), dtype)
    # the shared expert quantized as the routes are
    for name in ("gate_proj", "up_proj", "down_proj"):
        projection = nn.Linear(512, 512, bias = False)
        projection.set_dtype(dtype)
        setattr(block.shared_expert, name, nn.QuantizedLinear.from_linear(projection, group_size, bits))
    block.eval()
    mx.eval(block.parameters())
    return block


def _counting_folds(monkeypatch):
    monkeypatch.setattr(fusion.nax, "_nax_gpu", lambda: False)
    folds, fold = [], fusion._RoutedExperts._fold
    monkeypatch.setattr(fusion._RoutedExperts, "_fold", lambda self, *a: folds.append(a[2].shape) or fold(self, *a))
    return folds


@pytest.mark.parametrize("native", [vlm_qwen, lm_qwen])
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("wrapped", [False, True])
def test_shared_expert_folds_into_the_routed_launches_bitwise(native, group_size, wrapped, monkeypatch):
    block = _fold_block(native, 4, group_size)
    if wrapped:  # generation wraps projections in call-preserving subclasses
        block.shared_expert.gate_proj.__class__ = type("Wrapped", (nn.QuantizedLinear,), {})
    xs = [mx.random.normal((rows, 1, 512)).astype(mx.bfloat16) for rows in (1, 2, 5)]
    xs.append(xs[2][3:5])
    expected = [mx.concatenate([block(x[i : i + 1]) for i in range(x.shape[0])]) for x in xs]
    monkeypatch.setattr(fusion, "_MOE_FOLD_VERDICTS", {})
    folds = _counting_folds(monkeypatch)
    with _routed(block):   # a plain call keeps the shared expert's own launches
        _identical(block(xs[1]), expected[1])
        assert not folds
    with _routed_speculative(block):
        mx.eval(block(xs[0]))
        for x, out in zip(xs, expected):
            folds.clear()
            _identical(block(x), out)
            assert folds == [x.shape]


def test_verify_blocks_fold_their_shared_expert_bitwise(monkeypatch):
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward as native
    block = _fold_block(vlm_qwen, 4, 32)
    xs = [mx.random.normal(shape).astype(mx.bfloat16) for shape in ((1, 3, 512), (1, 5, 512), (1, 6, 512))]
    # one-token decoding's bits, which mlx-vlm's own batch forward does not keep at 4 bits
    expected = [mx.concatenate([block(x[:, i : i + 1]) for i in range(x.shape[1])], axis = 1) for x in xs]
    verifier = fusion._exact_verify_class(native, "qwen3_5", *(vars(native).get(name) for name in fusion._VERIFY_METHODS))()
    monkeypatch.setattr(fusion, "_VERIFY_VERDICTS", {})
    monkeypatch.setattr(fusion._VERIFY_STATE, "serving", True, raising = False)
    folds = _counting_folds(monkeypatch)
    with fusion.fused_moe_routed_experts(block), fusion.speculative_call(block):
        for x, out, folded in zip(xs, expected, (True, True, False)):
            mx.eval(verifier._feed_forward(block, x))
            folds.clear()
            _identical(verifier._feed_forward(block, x), out)
            assert bool(folds) is folded


@pytest.mark.parametrize("change", ["override", "projection", "unverified", "bits", "group", "strided", "swapped", "tensor_units", "eight_bits"])
def test_shared_expert_stays_on_its_own_launches_when_changed(change, monkeypatch):
    block = _fold_block(vlm_qwen, 8 if change == "eight_bits" else 4, 64 if change == "eight_bits" else 32)
    x = mx.random.normal((2, 1, 512)).astype(mx.bfloat16)
    monkeypatch.setattr(fusion, "_MOE_FOLD_VERDICTS", {})
    monkeypatch.setattr(fusion.nax, "_nax_gpu", lambda: change == "tensor_units")
    calls = []
    if change in ("override", "projection", "strided"):  # after verification: only their own checks see these
        with _routed_speculative(block):
            mx.eval(block(x))
    if change == "override":
        stock = type(block.shared_expert)
        block.shared_expert.__class__ = type("Doubled", (stock,), {"__call__": lambda self, x: stock.__call__(self, x) * 2})
    elif change == "projection":
        stock = nn.QuantizedLinear
        block.shared_expert.down_proj.__class__ = type("Scaled", (stock,), {"__call__": lambda self, x: stock.__call__(self, x) * 2})
    elif change == "group":  # compared on the routes' width, which the shared expert's scales cannot give
        block.shared_expert.down_proj = nn.QuantizedLinear.from_linear(nn.Linear(512, 512, bias = False), 64, 4)
    elif change == "unverified":
        launch = fusion._fold_plan
        monkeypatch.setattr(fusion, "_fold_plan", lambda *a: (lambda plan: (plan[0], lambda *b: plan[1](*b) + 1))(launch(*a)))
    elif change == "bits":
        block.shared_expert.down_proj = nn.QuantizedLinear.from_linear(nn.Linear(512, 512, bias = False), 32, 8)
    elif change == "strided":
        w = block.shared_expert.down_proj.weight
        block.shared_expert.down_proj.weight = mx.broadcast_to(w[:1], w.shape)
    elif change == "swapped":
        block.shared_expert.down_proj.__class__ = fusion._lane_module_class(nn.QuantizedLinear)
        block.shared_expert.down_proj._unsloth_lane_qmm = False
    mx.eval(block.parameters())
    expected = block(x)
    run = fusion._RoutedExperts._run
    monkeypatch.setattr(fusion._RoutedExperts, "_run", lambda self, *a, **k: calls.append(a) or run(self, *a, **k))
    with _routed_speculative(block):
        mx.eval(block(x))
        calls.clear()
        _identical(block(x), expected)
        assert calls  # the routes still run, with the shared expert's result as an input


def test_routed_experts_kill_switch_keeps_native(monkeypatch):
    block = _expert_block(vlm_qwen, (8, "affine", 64), mx.bfloat16)
    x = mx.random.normal((2, 1, 512)).astype(mx.bfloat16)
    expected = block(x)
    gathers = _counting_gather_qmm(monkeypatch)
    monkeypatch.setenv("UNSLOTH_MLX_ROUTED_EXPERTS", "0")
    with _routed(block):
        assert "_unsloth_moe_routed" not in block.__dict__
        gathers.clear()
        _identical(block(x), expected)
        assert gathers


@pytest.mark.parametrize("int8_prefill_first", [False, True])
def test_routed_experts_run_under_the_int8_prefill_scope(int8_prefill_first, monkeypatch):
    from unsloth_zoo.mlx import nax
    block = _expert_block(vlm_qwen, (8, "affine", 64), mx.bfloat16)
    x = mx.random.normal((2, 1, 512)).astype(mx.bfloat16)
    expected = block(x)
    monkeypatch.setattr(nax, "nax_available", lambda: True)
    monkeypatch.setattr(nax, "kernel_probe_passed", lambda *args: True)
    monkeypatch.setattr(nax, "_QMM_ROWS_BY_GPU", {})
    monkeypatch.setattr(nax, "_INT8_PREFILL_EXPERT_ROWS", {8: (8, 8)})
    gathers = _counting_gather_qmm(monkeypatch)
    scopes = [functools.partial(fusion.nax_quantized_linear, int8_prefill = True), _routed]
    with contextlib.ExitStack() as stack:
        for scope in scopes if int8_prefill_first else scopes[::-1]:
            stack.enter_context(scope(block))
        assert type(block.switch_mlp.down_proj).__name__.startswith("_NaxGather")
        mx.eval(block(x))
        gathers.clear()
        _identical(block(x), expected)
        assert not gathers


class _Doubled(vlm.SwitchGLU):
    def __call__(self, x, indices):
        return super().__call__(x, indices) * 2


@pytest.mark.parametrize("change", ["unverified", "rebound", "rebound_between_scopes", "override", "strided", "activation",
                                    "combine_before", "combine", "combine_instance"])
def test_routed_experts_keep_native_when_unverified_or_changed(change, monkeypatch):
    block = _expert_block(vlm_qwen, (8, "affine", 64), mx.bfloat16)
    x = mx.random.normal((2, 1, 512)).astype(mx.bfloat16)
    gathers = _counting_gather_qmm(monkeypatch)
    monkeypatch.setattr(fusion, "_MOE_ROUTED_VERDICTS", {})
    # compiled like the pinned one and equal on every probe input, so only its pinned body refuses it
    clamped = functools.partial(mx.compile, shapeless = True)(lambda gate, x: nn.silu(mx.minimum(gate, 30)) * x)
    rebind = lambda: monkeypatch.setattr(vlm, "swiglu", clamped)
    combine = vlm.SwitchGLU.__dict__["_combine"].__func__
    clamped_combine = staticmethod(lambda *a: mx.minimum(combine(*a), 1e4))  # equal on every probe input too
    if change == "combine_before":  # a wrapper, whose source inspect reads as the original's
        monkeypatch.setattr(vlm.SwitchGLU, "_combine", staticmethod(functools.wraps(combine)(clamped_combine.__func__)))
    elif change == "unverified":
        down = fusion._routed_down
        monkeypatch.setattr(fusion, "_routed_down", lambda *a: down(*a) + 1)
    elif change == "override":
        block.switch_mlp.__class__ = _Doubled
    elif change == "rebound_between_scopes":
        with _routed(block):
            mx.eval(block(x))  # verified with the pinned activation
        rebind()
    with _routed(block):
        mx.eval(block(x))
        if change == "rebound":
            rebind()
        elif change == "combine":
            monkeypatch.setattr(vlm.SwitchGLU, "_combine", clamped_combine)
        elif change == "combine_instance":
            block.switch_mlp._combine = clamped_combine.__func__
        elif change == "activation":
            block.switch_mlp.activation = GeGLU()
        elif change == "strided":  # after verification: only the layout check sees it
            w = block.switch_mlp.down_proj.weight
            block.switch_mlp.down_proj.weight = mx.broadcast_to(w[:, :1], w.shape)
        gathers.clear()
        fused = block(x)
        assert gathers
    _identical(fused, block(x))


class _ScaledShared(vlm_qwen.Qwen3_5MoeSparseMoeBlock):
    def _shared_expert_scale(self, x):
        return 2 * super()._shared_expert_scale(x)


@pytest.mark.parametrize("case", ["lm", "vlm", "vlm_override", "sigmoid_disagrees"])
def test_shared_expert_gate_sigmoid_rides_on_the_routing_launch(case, monkeypatch):
    block = _expert_block(lm_qwen if case == "lm" else vlm_qwen, (8, "affine", 64), mx.bfloat16)
    if case == "vlm_override":
        block.__class__ = _ScaledShared
    # decode rows, and the first sorted width, which the routed-experts kernel leaves native
    xs = [mx.random.normal((rows, 1, 512)).astype(mx.bfloat16) for rows in (1, 7, 8)]
    expected = [block(x) for x in xs]
    launches, run, skewed = [], fusion._run_moe_router, [case == "sigmoid_disagrees"]

    def recording(*args):
        out = run(*args)
        launches.append(len(args) > 6)
        return out[:2] + (out[2] + 1,) if skewed[0] and len(args) > 6 else out

    monkeypatch.setattr(fusion, "_run_moe_router", recording)
    fusion._moe_router_shared_verified.cache_clear()
    try:
        with _routed(block):
            mx.eval(block(xs[0]))
            for x, out in zip(xs, expected):
                launches.clear()
                _identical(block(x), out)
                assert launches == [case in ("lm", "vlm") and x.shape[0] < 8]
            skewed[0] = True  # past the probe, the folded scale is what scales the shared expert
            assert mx.array_equal(block(xs[0]), expected[0]).item() is (case not in ("lm", "vlm"))
    finally:
        fusion._moe_router_shared_verified.cache_clear()


@pytest.mark.parametrize("native", [vlm_gemma, lm_gemma])
def test_router_packing_that_fails_partway_leaves_every_module_native(native, monkeypatch):
    """Router shares the gate/up pack's cleanup contract; it builds a per-Gemma-module norm scale after the class swap, so it can fail with modules already patched."""
    first, second = _router(native), _router(native)
    root = nn.Sequential(first, second)
    root.eval()
    before = [set(module.__dict__) for module in (first, second)]
    samples = [_routing_samples(module) for module in (first, second)]

    native_init, built = fusion._RouterNormScale.__init__, []

    def refuse_the_second(self, module):
        built.append(module)
        if len(built) > 1:
            raise RuntimeError("injected: no headroom")
        native_init(self, module)

    monkeypatch.setattr(fusion._RouterNormScale, "__init__", refuse_the_second)
    with pytest.raises(RuntimeError, match = "injected"):
        with fusion.fused_moe_router(root):
            pass  # pragma: no cover
    assert len(built) == 2
    for module, keys, pairs in zip((first, second), before, samples):
        assert type(module) is native.Router
        assert set(module.__dict__) == keys
        _check(module, pairs)


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
              else _block(lm_qwen, experts = 1056 if reason == "experts" else 256, top_k = 17 if reason == "top_k" else 8))
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
    # generate() wrappers enter this scope without the generation_mode lock, so two requests can race; without
    # _MOE_ROUTER_LOCK the patched class is stranded; a tiny switch interval exposes it promptly.
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
