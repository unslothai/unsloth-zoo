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
recurrent decode convolution and SiLU, and the MoE routing chain."""

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

    A fingerprint may be a tuple of alternatives, for a body that differs across the supported
    range of an upstream package.

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
                accepted = (expected,) if isinstance(expected, str) else expected
                matches = target is not None and (expected is None or (
                    not hasattr(target, "__wrapped__") and _ast_fingerprint(target) in accepted
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


# Bodies the fused expert call reimplements. The two switch layer packages no longer share a
# `SwitchGLU.__call__`: mlx-vlm 0.7.1 gave it a short-sequence decode path that mlx-lm has not
# taken, so that one name is hashed per package and mlx-vlm accepts either body.
_MOE_GATE_UP_FUNCTIONS = {
    "QuantizedSwitchLinear.__call__": "59bbb193612cbe06",
    "_gather_sort": "75657d7fb03060c6",
    "_scatter_unsort": "78d5aa7dbef7183e",
}
_MOE_SWITCH_GLU_CALLS = {
    "mlx_lm.models.switch_layers": ("ed00798a68bad37d",),
    "mlx_vlm.models.switch_layers": ("ed00798a68bad37d", "4e3d80f1396fb265"),
}


def _moe_switch_specs():
    specs = {}
    for path, switch_glu_call in _MOE_SWITCH_GLU_CALLS.items():
        native = sys.modules.get(path)
        if native is None or not hasattr(native, "QuantizedSwitchLinear"):
            continue
        contract = dict(_MOE_GATE_UP_FUNCTIONS, **{"SwitchGLU.__call__": switch_glu_call})
        bindings = _resolved_bindings({path: contract})
        if bindings is not None:  # one drifted package still leaves the other
            specs[native.SwitchGLU] = (
                native.QuantizedSwitchLinear, native._gather_sort, native._scatter_unsort,
                # 0 where upstream has no short-sequence decode path, making the guard inert.
                getattr(native, "DECODE_BLOCK_SIZE", 0), bindings,
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


def _fused_moe_gate_up_class(original_class, projection_type, gather_sort, scatter_unsort,
                             decode_block, bindings):
    key = (original_class, projection_type, gather_sort, scatter_unsort, decode_block)  # each class guards its own resolution
    if key not in _MOE_GATE_UP_CLASSES:

        def fused_call(self, x, indices, *args, **kwargs):
            packed = self._unsloth_moe_gate_up
            # Short sequences reach a separate upstream decode path, and routing weights, shared
            # expert output and residuals are combined by one this body does not reimplement.
            if (self.training or not packed.matches(self) or not _bindings_intact(bindings)
                    or (x.ndim == 3 and 1 < x.shape[1] <= decode_block)
                    or any(extra is not None for extra in (*args, *kwargs.values()))):
                return original_class.__call__(self, x, indices, *args, **kwargs)
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
                # mlx's nn.Module subclasses dict, and the membership tests below read
                # the instance's own entries. Generation enters this scope for whatever
                # named_modules() yields, including the plain stand-ins the generate
                # tests pass, so anything that is not a Module is simply not a candidate.
                # Type first: a stand-in need not carry `training` either, and reading
                # it before the check raises out of generation instead of skipping.
                if not isinstance(module, dict) or module.training:
                    continue
                if "_causal_conv1d_decode" in module or "__call__" in module:
                    continue
                if hasattr(base, "_unsloth_decode_conv_silu"):
                    # Already fused by another scope. Count this one as an owner too,
                    # so that scope's exit cannot unfuse a module this scope still
                    # holds; only the last owner restores. Generation enters this
                    # alongside fused_moe_gate_up, and Studio enters it again per
                    # request, so overlapping ownership is the normal case.
                    if getattr(module, "_unsloth_decode_scopes", 0):
                        module._unsloth_decode_scopes += 1
                        changed.append(module)
                    continue
                if base not in specs:
                    specs[base] = _decode_conv_silu_contract(base)
                contract = specs[base]
                if contract is None or _decode_conv_silu_kernel() is None:
                    continue
                patched = _fused_decode_conv_silu_class(base, *contract)
                module.__class__ = patched
                module._unsloth_decode_native = base
                module._unsloth_decode_patched = patched
                module._unsloth_decode_scopes = 1
                changed.append(module)
        yield model
    finally:
        for module in reversed(changed):
            scopes = getattr(module, "_unsloth_decode_scopes", 0)
            if scopes > 1:
                module._unsloth_decode_scopes = scopes - 1
                continue
            if type(module) is getattr(module, "_unsloth_decode_patched", None):
                module.__class__ = module._unsloth_decode_native
            for name in ("_unsloth_decode_scopes", "_unsloth_decode_native",
                         "_unsloth_decode_patched"):
                module.__dict__.pop(name, None)

def _single_row(x):
    # The fused kernel reduces one row per launch, so anything wider keeps the native path.
    return x.size == x.shape[-1]

@functools.cache
def _residual_norm_kernel():
    if not mx.metal.is_available():
        return None
    try:
        return mx.fast.metal_kernel(
            name='unsloth_norm_add', input_names=['x', 'w', 'residual', 'scale', 'epsilon'], output_names=['out'],
            compile_options={'math_mode': 'safe'},
            source='''
            // The scope only replaces the native path because it reproduces mx.fast.rms_norm
            // bit for bit, which pins the arithmetic: each thread folds a contiguous SPAN of
            // the row, both reduction levels are simd_sum, and the squares accumulate through
            // fma. Regrouping the spans already diverges on ordinary activations; dropping the
            // fma needs a row spanning a wide dynamic range to show up.
            #pragma clang fp contract(off) reassociate(off)
            constexpr uint LANES = 32;
            constexpr uint SPAN = 4;

            const uint slot = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint group = simdgroup_index_in_threadgroup;
            const uint row = threadgroup_position_in_grid.x * D;

            threadgroup float staged[LANES];

            float kept[SPAN];
            float square = 0.0f;
            for (uint i = 0; i < SPAN; ++i) {
                const uint c = slot * SPAN + i;
                kept[i] = c < D ? float(x[row + c]) : 0.0f;
                square = metal::fma(kept[i], kept[i], square);
            }
            square = simd_sum(square);

            // Live partials and padding have disjoint writers.
            if (lane == 0) staged[group] = square;
            if (slot >= (D + LANES * SPAN - 1) / (LANES * SPAN) && slot < LANES)
                staged[slot] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const float total = simd_sum(staged[lane]);
            const float inv = metal::precise::rsqrt(total / D + epsilon);

            for (uint i = 0; i < SPAN; ++i) {
                const uint c = slot * SPAN + i;
                if (c < D) {
                    const T weighted = T(w[c] * T(kept[i] * inv));
                    const T merged = T(residual[row + c] + weighted);
                    out[row + c] = SCALED ? T(merged * scale[0]) : merged;
                }
            }
            ''')
    except (TypeError, RuntimeError):
        return None

@functools.cache
def _norm_epsilon(eps):
    return mx.array(eps, mx.float32)

@mx.compile
def _norm_add_apply(x, weight, residual, scale, eps, scaled):
    width = x.shape[-1]
    group = ((width + 127) // 128) * 32
    return _residual_norm_kernel()(
        inputs = [x, weight, residual, scale.reshape(-1), eps],
        template = [("T", x.dtype), ("D", width), ("SCALED", scaled)],
        grid = (group, 1, 1), threadgroup = (group, 1, 1),
        output_shapes = [x.shape], output_dtypes = [x.dtype],
    )[0]

def _residual_norm_add(norm, x, residual, scale = None):
    weight = norm.weight if type(norm) is nn.RMSNorm else None
    if (weight is not None and not norm.training and mx.default_device() == mx.gpu
            and x.ndim > 0 and x.shape == residual.shape and _single_row(x)
            and 0 < x.shape[-1] <= 4096 and weight.shape == (x.shape[-1],)
            and x.dtype in (mx.float16, mx.bfloat16, mx.float32)
            and x.dtype == weight.dtype == residual.dtype
            and (scale is None or (isinstance(scale, mx.array) and scale.size == 1 and scale.ndim <= x.ndim
                                   and scale.dtype == x.dtype))):
        return _norm_add_apply(x, weight, residual, weight if scale is None else scale,
                               _norm_epsilon(norm.eps), scale is not None)
    out = residual + norm(x)
    return out if scale is None else out * scale

_RESIDUAL_NORM_CONTRACT = {
    "mlx.nn": {"RMSNorm.__call__": "db2a29e3c13e7ef9"},
    "mlx.core": {"fast.rms_norm": None},
}

@functools.cache
def _residual_norm_class(base):
    rms = mx.fast.rms_norm
    if not (type(rms) is type(mx.add) and rms.__module__ == "mlx.core.fast" and rms.__name__ == "rms_norm"):
        return None
    call = getattr(base, "__call__", None)
    if not isinstance(call, FunctionType) or call.__closure__ or hasattr(call, "__wrapped__"):
        return None
    try:
        tree = copy.deepcopy(_function_ast(call))
    except (OSError, TypeError, SyntaxError, AttributeError):
        return None
    names = set()
    scale = None
    tail_return = None
    scale_branch = None
    if len(tree.body) >= 3 and isinstance(tree.body[-1], ast.Return):
        tail = tree.body[-2]
        if (isinstance(tail, ast.If) and not tail.orelse and len(tail.body) == 1
                and isinstance(tail.test, ast.Compare) and len(tail.test.ops) == 1
                and isinstance(tail.test.ops[0], ast.IsNot)
                and isinstance(tail.test.comparators[0], ast.Constant)
                and tail.test.comparators[0].value is None):
            product = tail.body[0]
            if (isinstance(product, ast.Assign) and len(product.targets) == 1
                    and isinstance(product.targets[0], ast.Name)
                    and isinstance(product.value, ast.BinOp) and isinstance(product.value.op, ast.Mult)
                    and _source_expression(product.value.left) == product.targets[0].id
                    and _source_expression(product.value.right) == _source_expression(tail.test.left)
                    and isinstance(tail.test.left, ast.Attribute)
                    and isinstance(tail.test.left.value, ast.Name) and tail.test.left.value.id == "self"
                    and isinstance(tree.body[-3], ast.If) and not tree.body[-3].orelse
                    and isinstance(tree.body[-1].value, ast.Tuple)
                    and all(isinstance(item, ast.Name) for item in tree.body[-1].value.elts)):
                scale = (product.targets[0].id, tail.test.left)
                tail_return, scale_branch = tree.body[-1], tree.body[-3]

    def rewrite(body, owner):
        index = 0
        while index + 1 < len(body):
            first, second = body[index:index + 2]
            if (isinstance(first, ast.Assign) and len(first.targets) == 1
                    and isinstance(first.targets[0], ast.Name) and isinstance(first.value, ast.Call)
                    and len(first.value.args) == 1 and not first.value.keywords
                    and isinstance(first.value.func, ast.Attribute)
                    and isinstance(first.value.func.value, ast.Name) and first.value.func.value.id == "self"
                    and isinstance(first.value.args[0], ast.Name)
                    and first.value.args[0].id == first.targets[0].id
                    and isinstance(second, ast.Assign) and len(second.targets) == 1
                    and isinstance(second.targets[0], ast.Name) and isinstance(second.value, ast.BinOp)
                    and isinstance(second.value.op, ast.Add) and isinstance(second.value.left, ast.Name)
                    and isinstance(second.value.right, ast.Name)
                    and second.value.right.id == first.targets[0].id
                    and second.value.left.id != first.targets[0].id
                    and (second.targets[0].id == first.targets[0].id
                         or (owner is scale_branch and index + 2 == len(body)
                             and all(item.id != first.targets[0].id for item in tail_return.value.elts)))):
                names.add(first.value.func.attr)
                args = [first.value.func, first.value.args[0], second.value.left]
                fused_tail = (owner is scale_branch and index + 2 == len(body)
                              and second.targets[0].id == scale[0])
                if fused_tail:
                    args.append(scale[1])
                second.value = ast.Call(func = ast.Attribute(value = ast.Name(id = "self", ctx = ast.Load()),
                    attr = "_unsloth_norm_add", ctx = ast.Load()), args = args, keywords = [])
                body[index:index + 2] = [second]
                if fused_tail:
                    body.append(copy.deepcopy(tail_return))
            index += 1
        for statement in body:
            for field in ("body", "orelse", "finalbody"):
                nested = getattr(statement, field, None)
                if isinstance(nested, list):
                    rewrite(nested, statement)

    rewrite(tree.body, tree)
    if not names:
        return None
    fused = _function_from_ast(call, tree)
    bindings = _resolved_bindings(_RESIDUAL_NORM_CONTRACT)
    if bindings is None:
        return None

    def invoke(self, *args, **kwargs):
        if self.training or base.__call__ is not call:
            return base.__call__(self, *args, **kwargs)
        return fused(self, *args, **kwargs)

    def norm_add(norm, x, residual, scale = None):
        if not _bindings_intact(bindings):
            out = residual + norm(x)
            return out if scale is None else out * scale
        return _residual_norm_add(norm, x, residual, scale)

    return type(f"_ResidualNorm{base.__name__}", (base,), {
        "__call__": invoke, "_unsloth_norm_add": staticmethod(norm_add),
        "_unsloth_residual_norm_base": base, "_unsloth_residual_norm_names": tuple(names),
    })

_RESIDUAL_NORM_LOCK = RLock()

@contextmanager
def fused_residual_norm(model):
    """Fuse eligible single-row RMS normalization and residual additions during inference."""
    patched = []
    try:
        with _RESIDUAL_NORM_LOCK:
            if not getattr(model, "_unsloth_mlx_distributed_parallel_mode", None) and _residual_norm_kernel() is not None:
                for _, module in model.named_modules() if hasattr(model, "named_modules") else ():
                    base = type(module)
                    # Type first, for the reason fused_decode_conv_silu gives: generation enters
                    # this scope for whatever named_modules() yields, including plain stand-ins
                    # that need not carry `training`, and reading it before the check raises out
                    # of generation instead of skipping an ineligible entry.
                    if not isinstance(module, dict) or module.training:
                        continue
                    if hasattr(base, "_unsloth_residual_norm_base"):
                        continue
                    fused = _residual_norm_class(base)
                    if (fused is not None and all(type(getattr(module, name, None)) is nn.RMSNorm
                                                 for name in fused._unsloth_residual_norm_names)):
                        patched.append((module, base, fused))
                        module.__class__ = fused
        yield model
    finally:
        with _RESIDUAL_NORM_LOCK:
            for module, base, fused in reversed(patched):
                if type(module) is fused:
                    module.__class__ = base

_QWEN_ROUTING, _GEMMA_ROUTING = 0, 1  # kernel MODE

# The kernel emulates the chain's rounding: softmax partials and reciprocal, stable tie order,
# sequential bf16 sum over <= 8 values. `_moe_router_verified` checks that at first use.
@functools.cache
def _moe_router_kernel():
    try:
        if not mx.metal.is_available():
            return None
        return mx.fast.metal_kernel(
            name = "unsloth_moe_router",
            input_names = ["logits", "scale"], output_names = ["inds", "weights"],
            compile_options = {"math_mode": "safe"},
            source = """
                #pragma clang fp contract(off) reassociate(off)
                // One simdgroup per row. Register r of lane l holds element
                // 128*(r/4) + 4*l + r%4, the element the native softmax's thread 32*(r/4)+l
                // reads at offset r%4, so the fp32 normalizer sums in the native order.
                constexpr int R = (E + 127) / 128 * 4;
                const uint row = thread_position_in_grid.x / 32;
                const uint lane = thread_index_in_simdgroup;
                if (row >= uint(logits_shape[0])) return;
                const device T* x = logits + row * E;

                float v[R];
                uint taken = 0u;
                for (int r = 0; r < R; ++r) {
                    const int e = 128 * (r / 4) + 4 * lane + (r % 4);
                    if (e < E) {
                        v[r] = float(x[e]);
                    } else {
                        v[r] = -INFINITY;
                        taken |= 1u << r;
                    }
                }

                if (MODE == 0) {
                    float m = v[0];
                    for (int r = 1; r < R; ++r) m = max(m, v[r]);
                    m = simd_max(m);
                    float total = 0.0f;
                    for (int g = 0; g < R / 4; ++g) {
                        float partial = 0.0f;
                        for (int i = 0; i < 4; ++i) {
                            v[4 * g + i] = fast::exp(v[4 * g + i] - m);
                            partial += v[4 * g + i];
                        }
                        total += simd_sum(partial);
                    }
                    const float normalizer = 1 / total;
                    for (int r = 0; r < R; ++r) v[r] = float(T(v[r] * normalizer));
                }

                bool nan_row = false;  // NaN sorts last in mx.argpartition and poisons the weights
                for (int r = 0; r < R; ++r) { nan_row |= isnan(v[r]); if (isnan(v[r])) v[r] = INFINITY; }
                nan_row = simd_any(nan_row);
                float sel_val[K];
                uint sel_idx[K];
                for (int k = 0; k < K; ++k) {
                    float best = -INFINITY;
                    int bi = -1;
                    for (int r = 0; r < R; ++r) {
                        if (!(taken & (1u << r)) && v[r] >= best) { best = v[r]; bi = r; }
                    }
                    const float m = simd_max(best);
                    const uint cand = (bi >= 0 && best == m)
                        ? uint(128 * (bi / 4) + 4 * lane + (bi % 4)) : 0u;
                    const uint win = simd_max(cand);
                    if ((win / 4) % 32 == lane) taken |= 1u << (4 * (win / 128) + win % 4);
                    sel_idx[K - 1 - k] = win;
                    sel_val[K - 1 - k] = m;
                }

                if (lane != 0) return;
                device uint* oi = inds + row * K;
                device OUT_T* ow = weights + row * K;
                for (int k = 0; k < K; ++k) oi[k] = sel_idx[k];
                if (nan_row) { for (int k = 0; k < K; ++k) ow[k] = OUT_T(NAN); return; }
                if (MODE == 0) {
                    float denom = sel_val[0];
                    for (int k = 1; k < K; ++k) denom = float(T(denom + sel_val[k]));
                    for (int k = 0; k < K; ++k) {
                        ow[k] = OUT_T(NORM ? sel_val[k] / denom : sel_val[k]);
                    }
                } else {
                    const float m = sel_val[K - 1];
                    float e[K];
                    float part[2] = {0.0f, 0.0f};
                    for (int k = 0; k < K; ++k) {
                        e[k] = float(T(fast::exp(float(T(sel_val[k] - m)))));
                        part[k / 4] = float(T(part[k / 4] + e[k]));
                    }
                    const float recip = float(T(1.0f / float(T(part[0] + part[1]))));
                    for (int k = 0; k < K; ++k) {
                        ow[k] = OUT_T(float(T(e[k] * recip)) * float(scale[sel_idx[k]]));
                    }
                }
            """,
        )
    except (AttributeError, TypeError):
        return None


_MOE_ROUTER_DTYPES = (mx.bfloat16, mx.float16, mx.float32)


def _moe_router_shape_ok(experts, top_k, mode):
    # Qwen sums softmax partials of at most two simdgroups in the native order;
    # the 32-bit `taken` mask caps both modes at 1024.
    return experts % 32 == 0 and experts <= (256 if mode == _QWEN_ROUTING else 1024) and 1 <= top_k <= 8


def _run_moe_router(logits, scale, top_k, mode, normalize, out_dtype):
    experts = logits.shape[-1]
    rows = logits.size // experts
    inds, weights = _moe_router_kernel()(
        inputs = [logits.reshape(rows, experts), scale],
        template = [("T", logits.dtype), ("OUT_T", out_dtype), ("E", experts), ("K", top_k),
                    ("MODE", mode), ("NORM", int(normalize))],
        grid = ((rows + 7) // 8 * 256, 1, 1), threadgroup = (256, 1, 1),
        output_shapes = [(rows, top_k), (rows, top_k)], output_dtypes = [mx.uint32, out_dtype],
    )
    return inds.reshape(*logits.shape[:-1], top_k), weights.reshape(*logits.shape[:-1], top_k)


def _native_moe_router(logits, scale, top_k, mode, normalize):
    if mode == _QWEN_ROUTING:
        gates = mx.softmax(logits, axis = -1, precise = True)
        inds = mx.argpartition(gates, kth = -top_k, axis = -1)[..., -top_k:]
        weights = mx.take_along_axis(gates, inds, axis = -1)
        if normalize:
            weights = weights / weights.sum(axis = -1, keepdims = True)
        return inds, weights
    inds = mx.argpartition(logits, kth = -top_k, axis = -1)[..., -top_k:]
    weights = mx.softmax(mx.take_along_axis(logits, inds, axis = -1), axis = -1)
    return inds, weights * scale[inds]


def _moe_router_out_dtype(logits, scale, mode):
    return mx.result_type(logits, scale) if mode == _GEMMA_ROUTING else logits.dtype


# Qwen routing never reads `scale`; the kernel takes one because its input list is fixed.
_MOE_ROUTER_NO_SCALE = mx.zeros((1,), dtype = mx.float32)


@functools.cache
def _moe_router_verified(dtype, scale_dtype, experts, top_k, mode, normalize):
    if mode == _QWEN_ROUTING:
        scale = _MOE_ROUTER_NO_SCALE
    else:
        scale = mx.random.uniform(0.5, 1.5, (experts,), key = mx.random.key(experts)).astype(scale_dtype)
    probes = [mx.random.normal((rows, experts), key = mx.random.key(rows)) * 3 for rows in (1, 9)]
    probes.append(mx.random.randint(0, 4, (9, experts), key = mx.random.key(0)))  # top-k boundary ties
    for logits in probes:
        logits = logits.astype(dtype)
        native = _native_moe_router(logits, scale, top_k, mode, normalize)
        fused = _run_moe_router(logits, scale, top_k, mode, normalize,
                                _moe_router_out_dtype(logits, scale, mode))
        if not all(a.dtype == b.dtype and mx.array_equal(a, b) for a, b in zip(native, fused)):
            logger.warning("the fused MoE router does not reproduce this MLX build's native rounding "
                           "for %s x%d top-%d; the native chain stays in use", dtype, experts, top_k)
            return False
    return True


def _fused_moe_router(logits, scale, top_k, mode, normalize):
    """The fused routing, or None when this call has to take the native chain."""
    # A 1-D input normalizes with a sum that rounds differently.
    if (logits.dtype not in _MOE_ROUTER_DTYPES or logits.size == 0 or logits.ndim < 2
            or not _moe_router_shape_ok(logits.shape[-1], top_k, mode)
            or not _moe_router_verified(logits.dtype, scale.dtype, logits.shape[-1], top_k,
                                        mode, normalize)):
        return None
    return _run_moe_router(logits, scale, top_k, mode, normalize,
                           _moe_router_out_dtype(logits, scale, mode))


def _drifted_call(self, native):
    """The base class's routing body once it or its `mx` alias drifted from the pinned one, else None."""
    current = type(self)._unsloth_router_native.__call__
    if current is native and native.__globals__.get("mx") is mx:
        return None
    return current


def _qwen3_5_moe_call(native, scaled_shared, top_k_norm):
    # 0.7.1 scales the shared expert by `_shared_expert_scale`, earlier bodies gate it
    # by a sigmoid. mlx_lm honours `norm_topk_prob`; mlx_vlm always normalizes.
    def fused_call(self, x, target_verify = False):
        drifted = _drifted_call(self, native)
        if drifted is not None:
            return drifted(self, x, target_verify) if target_verify else drifted(self, x)
        if target_verify or self.training or getattr(self, "sharding_group", None) is not None:
            return native(self, x, target_verify) if target_verify else native(self, x)
        normalize = bool(self.norm_topk_prob) if top_k_norm else True
        routed = _fused_moe_router(self.gate(x), _MOE_ROUTER_NO_SCALE, self.top_k, _QWEN_ROUTING, normalize)
        if routed is None:
            return native(self, x)
        inds, scores = routed
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis = -2)
        shared_y = self.shared_expert(x)
        if scaled_shared:
            shared_y = self._shared_expert_scale(x) * shared_y
        else:
            shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y

    return fused_call


class _RouterNormScale:
    """The Gemma router's `scale * root_size`, built at scope entry, used while `scale` is that array."""

    def __init__(self, module):
        self.scale = module.scale
        self.weight = self.scale * module._root_size

    def matches(self, module):
        return module.scale is self.scale


def _gemma4_router_call(native):
    def fused_call(self, x):
        drifted = _drifted_call(self, native)
        if drifted is not None:
            return drifted(self, x)
        norm = getattr(self, "_unsloth_router_norm", None)
        if self.training or norm is None or not norm.matches(self):
            return native(self, x)
        normed = mx.fast.rms_norm(x, norm.weight, self.eps)
        routed = _fused_moe_router(self.proj(normed), self.per_expert_scale,
                                   self.config.top_k_experts, _GEMMA_ROUTING, False)
        return native(self, x) if routed is None else routed

    return fused_call


# Routing bodies the fused calls reimplement: builder, kernel mode, expert-count and top-k attributes.
_MOE_ROUTER_BODIES = {
    # mlx_vlm qwen3_5_moe: 0.4.4-0.5.0 and 0.6.16-0.7.0; 0.6.0-0.6.15 takes a target-verify
    # argument; 0.7.1 scales the shared expert. qwen4_exp reuses the block, subclassing it in 0.7.1.
    "903a5a99afcad95a": (functools.partial(_qwen3_5_moe_call, scaled_shared = False, top_k_norm = False), _QWEN_ROUTING, "num_experts", "top_k"),
    "0ca80e0451802015": (functools.partial(_qwen3_5_moe_call, scaled_shared = False, top_k_norm = False), _QWEN_ROUTING, "num_experts", "top_k"),
    "b948b94a9d18333f": (functools.partial(_qwen3_5_moe_call, scaled_shared = True, top_k_norm = False), _QWEN_ROUTING, "num_experts", "top_k"),
    # qwen3_next in mlx_lm and mlx_vlm, reused by mlx_lm qwen3_5
    "6d29bc869fdde5aa": (functools.partial(_qwen3_5_moe_call, scaled_shared = False, top_k_norm = True), _QWEN_ROUTING, "num_experts", "top_k"),
    # gemma4 Router in mlx_vlm gemma4_text, gemma4 and mlx_lm gemma4_text
    "c64fed4e7e7514ea": (_gemma4_router_call, _GEMMA_ROUTING, "config.num_experts", "config.top_k_experts"),
}


@functools.cache
def _moe_router_class(base):
    call = getattr(base, "__call__", None)
    if not isinstance(call, FunctionType) or hasattr(call, "__wrapped__") or call.__closure__:
        return None
    try:
        spec = _MOE_ROUTER_BODIES.get(_ast_fingerprint(call))
    except (OSError, TypeError, SyntaxError):
        return None
    if spec is None or call.__globals__.get("mx") is not mx:
        return None
    build, mode, experts_path, top_k_path = spec
    patched = type(f"_FusedMoERouter{base.__name__}", (base,),
                   {"__call__": build(call), "_unsloth_router_native": base})
    return patched, mode, experts_path, top_k_path


@contextmanager
def fused_moe_router(model):
    """Fuse the MoE routing chain into one Metal dispatch during serialized inference.

    Modules whose routing body, expert count, or top-k the kernel does not cover keep
    their native call, as do training and distributed models; the native call is also
    taken per invocation when the kernel cannot reproduce this MLX build's rounding.
    Instance classes are restored when the context exits, including on cancellation.
    Gemma routers fold `scale` into a normalization weight held until the outermost
    scope exits, so `scale` must stay fixed while the scope is open, as the gate and up
    packing requires of its own weights: replacing it is detected and takes the native
    call, editing it in place is not detected. Edits between scopes are always picked up.
    """
    changed = []
    try:
        modules = model.named_modules() if hasattr(model, "named_modules") else ()
        if (not getattr(model, "_unsloth_mlx_distributed_parallel_mode", None)
                and _moe_router_kernel() is not None):
            for _, module in modules:
                # Type before `training`: named_modules() may yield plain stand-ins.
                if not isinstance(module, dict) or module.training or "__call__" in module:
                    continue
                base = type(module)
                if getattr(base, "_unsloth_router_native", None) is not None:
                    if getattr(module, "_unsloth_router_scopes", 0):
                        module._unsloth_router_scopes += 1
                        changed.append(module)
                    continue
                spec = _moe_router_class(base)
                if spec is None:
                    continue
                patched, mode, experts_path, top_k_path = spec
                experts = _resolve_dotted(module, experts_path)
                top_k = _resolve_dotted(module, top_k_path)
                if not (isinstance(experts, int) and isinstance(top_k, int)
                        and _moe_router_shape_ok(experts, top_k, mode)):
                    continue
                module.__class__ = patched
                module._unsloth_router_scopes = 1
                changed.append(module)
                if mode == _GEMMA_ROUTING:
                    module._unsloth_router_norm = _RouterNormScale(module)
        yield model
    finally:
        for module in reversed(changed):
            scopes = getattr(module, "_unsloth_router_scopes", 0)
            if scopes > 1:
                module._unsloth_router_scopes = scopes - 1
                continue
            native = getattr(type(module), "_unsloth_router_native", None)
            if native is not None:
                module.__class__ = native
            module.__dict__.pop("_unsloth_router_scopes", None)
            module.__dict__.pop("_unsloth_router_norm", None)
