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

"""Scoped MLX inference fusions: quantized MoE gate and up projections,
and recurrent decode convolution and SiLU."""

import ast
import functools
import copy
import hashlib
import inspect
import logging
import re
import sys
import textwrap
from contextlib import contextmanager
from threading import RLock
from types import FunctionType

import mlx.core as mx
import mlx.nn as nn


logger = logging.getLogger(__name__)


# Pin the upstream function bodies the fusions read, rewrite or reimplement, since a fusion
# mirroring an upstream body would compute the old arithmetic once that body changes.
_ALIASES = (("mx", mx), ("nn", nn))
_CONTRACT_MISSES = set()


@functools.cache
def _function_ast(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]


def _ast_fingerprint(function):
    # ast.unparse is version-stable once the 3.9/3.10 tuple parentheses are dropped; ast.dump is not.
    source = ast.unparse(_function_ast(function))
    source = re.sub(r"^(\s*)\(([^()\n]+)\) = ", r"\1\2 = ", source, flags = re.M)
    source = re.sub(r"\bfor \(([^()\n]+)\) in ", r"for \1 in ", source)
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def _resolve_dotted(module, path):
    target = module
    for part in path.split("."):
        target = getattr(target, part, None)
        if target is None:
            return None
    return target


def _resolved_bindings(contract):
    """Resolve a `{module: {dotted name: fingerprint}}` contract, None pinning the name not the body.

    Returns what each name resolved to, for `_bindings_intact` to recheck, or None when an entry
    differs. A fusion has to recheck the bindings of the resolution that produced the callables it
    captured, so these are returned rather than written into a list a later resolution would reuse.
    """
    holds, found = True, {}
    for module_name, names in contract.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue  # never imported, so no model instance can use it
        for path, expected in names.items():
            owner, _, name = path.rpartition(".")
            holder = _resolve_dotted(module, owner) if owner else module
            target = getattr(holder, name, None)
            try:
                # inspect follows __wrapped__ to the original source, so a wrapped target is refused.
                matches = target is not None and (expected is None or (
                    not hasattr(target, "__wrapped__") and _ast_fingerprint(target) == expected
                    and all(target.__globals__.get(alias, mod) is mod for alias, mod in _ALIASES)))
            except (OSError, TypeError, SyntaxError, AttributeError):
                matches = False
            if matches:
                found[id(holder), name] = (vars(holder), name, target)
                for alias, mod in (_ALIASES if expected is not None else ()):
                    if alias in target.__globals__:
                        found[id(target.__globals__), alias] = (target.__globals__, alias, mod)
            else:
                holds = False
                if (module_name, path) not in _CONTRACT_MISSES:
                    _CONTRACT_MISSES.add((module_name, path))
                    logger.warning("%s.%s differs from the version the MLX fusions were written "
                                   "against; the native method stays in use", module_name, path)
    return list(found.values()) if holds else None


def _bindings_intact(bindings):
    for namespace, name, target in bindings:
        if namespace.get(name) is not target:
            return False
    return True


_MOE_PROJECTION_FIELDS = ("weight", "scales", "biases", "bias")


_MOE_PACK_ROW_MULTIPLE = 8


_MOE_GATE_UP_CLASSES = {}
_MOE_GATE_UP_LOCK = RLock()


# Bodies the fused expert call reimplements. Both switch layer packages carry the same sources,
# so one set of hashes covers either namespace.
_MOE_GATE_UP_FUNCTIONS = {
    "SwitchGLU.__call__": "ed00798a68bad37d",
    "QuantizedSwitchLinear.__call__": "59bbb193612cbe06",
    "_gather_sort": "75657d7fb03060c6",
    "_scatter_unsort": "78d5aa7dbef7183e",
}


def _moe_switch_specs():
    specs = {}
    for path in ("mlx_lm.models.switch_layers", "mlx_vlm.models.switch_layers"):
        native = sys.modules.get(path)
        if native is None or not hasattr(native, "QuantizedSwitchLinear"):
            continue
        bindings = _resolved_bindings({path: _MOE_GATE_UP_FUNCTIONS})
        if bindings is not None:  # one drifted package still leaves the other
            specs[native.SwitchGLU] = (
                native.QuantizedSwitchLinear, native._gather_sort, native._scatter_unsort, bindings,
            )
    return specs


def _moe_gate_up_eligible(module, projection_type):
    gate, up = module.gate_proj, module.up_proj
    if type(gate) is not projection_type or type(up) is not projection_type:
        return False
    if (module.training or gate.training or up.training
            or gate.trainable_parameters() or up.trainable_parameters()):
        return False
    if (gate.group_size, gate.bits, gate.mode) != (up.group_size, up.bits, up.mode):
        return False
    for name in _MOE_PROJECTION_FIELDS:
        a, b = gate.get(name), up.get(name)
        if (a is None) != (b is None):
            return False
        if a is not None and (
            not isinstance(a, mx.array)
            or not isinstance(b, mx.array)
            or a.shape != b.shape
            or a.dtype != b.dtype
        ):
            return False
    if gate.weight.ndim != 3 or gate.scales.ndim != 3:
        return False
    # MLX reads a row-count remainder to pick the single-row gather matmul, so a pack whose
    # doubled rows cross that boundary would take a different kernel than the pair it replaces.
    if gate.weight.shape[1] % _MOE_PACK_ROW_MULTIPLE:
        return False
    return all(
        gate[name].shape[:-1] == gate.weight.shape[:-1]
        for name in ("scales", "biases") if gate.get(name) is not None
    ) and (gate.get("bias") is None or gate.bias.shape == gate.weight.shape[:-1])


class _PackedMoEGateUp:
    def __init__(self, module):
        gate, up = module.gate_proj, module.up_proj
        self.original_class = type(module)
        self.active_scopes = 0
        self.gate, self.up = gate, up
        self.metadata = (gate.group_size, gate.bits, gate.mode)
        self.arrays = {
            name: mx.concatenate([gate[name], up[name]], axis = -1 if name == "bias" else -2)
            for name in _MOE_PROJECTION_FIELDS
            if gate.get(name) is not None
        }
        mx.eval(list(self.arrays.values()))
        self.views = {
            name: mx.split(array, 2, axis = -1 if name == "bias" else -2)
            for name, array in self.arrays.items()
        }
        mx.eval([view for pair in self.views.values() for view in pair])

    def attach(self):
        for name, (gate_view, up_view) in self.views.items():
            setattr(self.gate, name, gate_view)
            setattr(self.up, name, up_view)

    def matches(self, module):
        if module.gate_proj is not self.gate or module.up_proj is not self.up:
            return False
        for projection in (self.gate, self.up):
            if (
                projection.training
                or projection.trainable_parameters()
                or (projection.group_size, projection.bits, projection.mode) != self.metadata
            ):
                return False
        for name in _MOE_PROJECTION_FIELDS:
            expected = self.views.get(name, (None, None))
            if self.gate.get(name) is not expected[0] or self.up.get(name) is not expected[1]:
                return False
        return True

    def project(self, x, indices, sorted_indices):
        group_size, bits, mode = self.metadata
        result = mx.gather_qmm(
            x,
            self.arrays["weight"],
            self.arrays["scales"],
            self.arrays.get("biases"),
            rhs_indices = indices,
            transpose = True,
            group_size = group_size,
            bits = bits,
            mode = mode,
            sorted_indices = sorted_indices,
        )
        if "bias" in self.arrays:
            result = result + mx.expand_dims(self.arrays["bias"][indices], -2)
        return mx.split(result, 2, axis = -1)


def _fused_moe_gate_up_class(original_class, projection_type, gather_sort, scatter_unsort, bindings):
    key = (original_class, projection_type, gather_sort, scatter_unsort)  # each class guards its own resolution
    if key not in _MOE_GATE_UP_CLASSES:

        def fused_call(self, x, indices):
            packed = self._unsloth_moe_gate_up
            if self.training or not packed.matches(self) or not _bindings_intact(bindings):
                return original_class.__call__(self, x, indices)
            x = mx.expand_dims(x, (-2, -3))
            do_sort = indices.size >= 64
            idx = indices
            inv_order = None
            if do_sort:
                x, idx, inv_order = gather_sort(x, indices)
            x_gate, x_up = packed.project(x, idx, do_sort)
            x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices = do_sort)
            if do_sort:
                x = scatter_unsort(x, inv_order, indices.shape)
            return x.squeeze(-2)

        _MOE_GATE_UP_CLASSES[key] = type(
            f"_FusedMoEGateUp{original_class.__name__}", (original_class,), {"__call__": fused_call}
        )
    return _MOE_GATE_UP_CLASSES[key]


@contextmanager
def fused_moe_gate_up(model):
    """Fuse quantized MoE gate and up projections while their weights stay fixed.

    Repacking at entry makes weight edits between scopes visible. Overlapping scopes
    share packing until the last exit; training or custom projections retain native calls.
    """
    patched = []
    try:
        with _MOE_GATE_UP_LOCK:
            if not getattr(model, "_unsloth_mlx_distributed_parallel_mode", None):
                specs = _moe_switch_specs()
                modules = model.named_modules() if hasattr(model, "named_modules") else ()
                for _, module in modules:
                    packed = getattr(module, "_unsloth_moe_gate_up", None)
                    if (isinstance(packed, _PackedMoEGateUp)
                            and type(module) in _MOE_GATE_UP_CLASSES.values()):
                        original, fused = packed.original_class, type(module)
                    else:
                        original = type(module)
                        spec = specs.get(original)
                        if spec is None or not _moe_gate_up_eligible(module, spec[0]):
                            continue
                        packed = _PackedMoEGateUp(module)
                        fused = _fused_moe_gate_up_class(original, *spec)
                    packed.active_scopes += 1
                    patched.append((module, original, fused, packed))
                    if type(module) is not fused:
                        module._unsloth_moe_gate_up = packed
                        packed.attach()
                        module.__class__ = fused
        yield model
    finally:
        with _MOE_GATE_UP_LOCK:
            for module, original, fused, packed in reversed(patched):
                packed.active_scopes -= 1
                if packed.active_scopes:
                    continue
                if type(module) is fused:
                    module.__class__ = original
                if getattr(module, "_unsloth_moe_gate_up", None) is packed:
                    del module._unsloth_moe_gate_up


@functools.cache
def _decode_conv_silu_kernel():
    try:
        if not mx.metal.is_available():
            return None
        return mx.fast.metal_kernel(
            name = "unsloth_decode_conv_silu",
            input_names = ["x", "w"], output_names = ["out"],
            ensure_row_contiguous = False,
            compile_options = {"math_mode": "safe"},
            source = """
                #pragma clang fp contract(off) reassociate(off)
                uint c = thread_position_in_grid.x;
                uint b = thread_position_in_grid.y;
                uint C = x_shape[2];
                if (c >= C) return;
                float acc = 0.0f;
                #pragma unroll
                for (uint k = 0; k < K; ++k) {
                    float v = float(x[b*x_strides[0] + k*x_strides[1] + c*x_strides[2]]);
                    float weight = w[k*w_strides[0] + c*w_strides[1]];
                    float product = v * weight;
                    acc = product + acc;
                }
                T v = T(acc);
                auto y = 1 / (1 + metal::exp(metal::abs(v)));
                T sigmoid = (v < 0) ? y : 1 - y;
                out[b*C + c] = v * sigmoid;
            """,
        )
    except (AttributeError, TypeError):
        return None


@mx.compile
def _decode_conv_silu(x, weight):
    # Preserve the native small-column reduction order and both half-precision casts.
    return _decode_conv_silu_kernel()(
        inputs = [x, weight], template = [("T", x.dtype), ("K", x.shape[1])],
        grid = (x.shape[2], x.shape[0], 1), threadgroup = (256, 1, 1),
        output_shapes = [(x.shape[0], 1, x.shape[2])], output_dtypes = [x.dtype],
    )[0]


_CONV_SILU_CONTRACT = {"mlx.nn": {"silu": "78867fdb7e42731c"}}


def _source_expression(node):
    return ast.unparse(node)


def _function_from_ast(original, tree):
    tree.decorator_list = []
    tree.returns = None
    for arg in (*tree.args.posonlyargs, *tree.args.args, *tree.args.kwonlyargs):
        arg.annotation = None
    tree.args.defaults = []
    tree.args.kw_defaults = [None] * len(tree.args.kwonlyargs)
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body = [tree], type_ignores = [])),
                 original.__code__.co_filename, "exec"), original.__globals__, namespace)
    function = namespace[tree.name]
    result = FunctionType(function.__code__, original.__globals__, tree.name,
                          original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    result.__annotations__ = original.__annotations__
    return result


def _decode_conv_silu_contract(base):
    call = getattr(base, "__call__", None)
    prepare = getattr(base, "_causal_conv1d_decode", None)
    if not isinstance(call, FunctionType) or not isinstance(prepare, FunctionType):
        return None
    try:
        outer, inner = _function_ast(call), _function_ast(prepare)
        target = inner.body[-1].value
        if not (isinstance(inner.body[-1], ast.Return) and isinstance(target, ast.Call)
                and isinstance(target.func, ast.Name)
                and [_source_expression(a) for a in target.args] == ["conv_input", "weight"]
                and not target.keywords):
            return None
        conv = prepare.__globals__.get(target.func.id)
        if any(hasattr(f, "__wrapped__") for f in (call, prepare, conv)):
            return None
        arithmetic = _function_ast(conv)
        if [_source_expression(n) for n in arithmetic.body] != [
            "out = mx.sum(conv_input.astype(mx.float32) * weight[None, :, :], axis=1)",
            "return out.astype(conv_input.dtype)[:, None, :]",
        ] or inspect.unwrap(conv).__globals__.get("mx") is not mx:
            return None
        if _resolved_bindings(_CONV_SILU_CONTRACT) is None:
            return None
        if call.__closure__ or prepare.__closure__:
            return None
        if call.__globals__.get("nn") is not nn or call.__globals__.get("mx") is not mx:
            return None
        assignments = [n for n in ast.walk(outer) if isinstance(n, ast.Name)
                       and n.id == "conv_input" and isinstance(n.ctx, ast.Store)]
        concatenations = [n for n in ast.walk(outer) if isinstance(n, ast.Assign)
                          and _source_expression(n) == "conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)"]
        if len(assignments) != 1 or len(concatenations) != 1:
            return None
        matches = [n for n in ast.walk(outer) if isinstance(n, ast.Call)
                   and _source_expression(n) == "nn.silu(self._causal_conv1d_decode(conv_input))"]
        if len(matches) != 1:
            return None
        return call, prepare, conv, nn.silu
    except (OSError, TypeError, SyntaxError, AttributeError, IndexError):
        return None


@functools.cache
def _fused_decode_conv_silu_class(base, call, prepare, conv, silu):
    bindings = _resolved_bindings(_CONV_SILU_CONTRACT) or []  # this resolution's, not a later one's
    outer, inner = copy.deepcopy(_function_ast(call)), copy.deepcopy(_function_ast(prepare))

    class Rewrite(ast.NodeTransformer):
        def visit_Call(self, node):
            if _source_expression(node) == "nn.silu(self._causal_conv1d_decode(conv_input))":
                node = ast.Call(func = ast.Attribute(value = ast.Name(id = "self", ctx = ast.Load()),
                    attr = "_unsloth_decode_conv_silu", ctx = ast.Load()), args = node.args[0].args, keywords = [])
            return self.generic_visit(node)

    outer = Rewrite().visit(outer)
    target = inner.body[-1].value
    conv_name = target.func.id
    target.func = ast.Attribute(value = ast.Name(id = "self", ctx = ast.Load()),
                                attr = "_unsloth_apply_conv_silu", ctx = ast.Load())
    inner.name = "_unsloth_decode_conv_silu"
    adapted_call = _function_from_ast(call, outer)

    def conv_silu(self, x, weight):
        if (x.ndim == 3 and x.shape[0] > 0 and 2 <= x.shape[1] <= 8 and x.shape[2] > 1
                and x.dtype in (mx.bfloat16, mx.float16)
                and weight.shape == x.shape[1:] and weight.dtype == mx.float32):
            return _decode_conv_silu(x, weight)
        return silu(conv(x, weight))

    def fused_call(self, *args, **kwargs):
        if (self.training or prepare.__globals__.get(conv_name) is not conv
                or not _bindings_intact(bindings)):
            return call(self, *args, **kwargs)
        return adapted_call(self, *args, **kwargs)

    return type(f"_FusedDecodeConvSiLU{base.__name__}", (base,), {
        "__call__": fused_call,
        "_unsloth_decode_conv_silu": _function_from_ast(prepare, inner),
        "_unsloth_apply_conv_silu": conv_silu,
    })


@contextmanager
def fused_decode_conv_silu(model):
    """Fuse recurrent decode convolution and SiLU during serialized inference.

    Prefill, unsupported convolution shapes, and training keep their native paths.
    Instance classes are restored when the context exits, including on cancellation.
    """
    changed = []
    specs = {}
    try:
        modules = model.named_modules() if hasattr(model, "named_modules") else ()
        if not getattr(model, "_unsloth_mlx_distributed_parallel_mode", None):
            for _, module in modules:
                base = type(module)
                if module.training or hasattr(base, "_unsloth_decode_conv_silu"):
                    continue
                if "_causal_conv1d_decode" in module or "__call__" in module:
                    continue
                if base not in specs:
                    specs[base] = _decode_conv_silu_contract(base)
                contract = specs[base]
                if contract is None or _decode_conv_silu_kernel() is None:
                    continue
                patched = _fused_decode_conv_silu_class(base, *contract)
                module.__class__ = patched
                changed.append((module, base, patched))
        yield model
    finally:
        for module, base, patched in reversed(changed):
            if type(module) is patched:
                module.__class__ = base
