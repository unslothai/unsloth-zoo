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
the recurrent decode convolution, the MoE routing chain, Qwen MoE routed experts,
small-batch quantized projections on the neural accelerators, the pre-norm residual handoff,
Qwen exact speculative verify blocks, and Gemma 4 verify-block attention."""

import ast
import functools
import copy
import hashlib
import inspect
import json
import logging
import math
import os
import re
import struct
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from threading import RLock, local
from types import FunctionType
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from . import nax


logger = logging.getLogger(__name__)


def _fusion_modules(model, modules):
    return modules if modules is not None else (
        model.named_modules() if hasattr(model, "named_modules") else ()
    )


# Pin the upstream function bodies the fusions read, rewrite or reimplement, since a fusion
# mirroring an upstream body would compute the old arithmetic once that body changes.
_ALIASES = (("mx", mx), ("nn", nn))
_CONTRACT_MISSES = set()


@functools.cache
def _function_ast(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]


@functools.cache
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
                # A static method's class entry is its descriptor, which is what a rebinding replaces.
                raw = vars(holder).get(name)
                found[id(holder), name] = (vars(holder), name, raw if isinstance(raw, staticmethod) else target)
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
                native.QuantizedSwitchLinear, native._scatter_unsort,
                # 0 where upstream has no short-sequence decode path, making the guard inert.
                getattr(native, "DECODE_BLOCK_SIZE", 0), bindings,
            )
    return specs


def _native_type(projection):
    return getattr(type(projection), "_unsloth_nax_qmm_native", type(projection))


def _moe_gate_up_eligible(module, projection_type):
    return (_moe_gate_up_packable(module, projection_type)
            and not (module.training or module.gate_proj.training or module.up_proj.training))


def _moe_gate_up_packable(module, projection_type):
    """`_moe_gate_up_eligible` in any training mode."""
    gate, up = module.gate_proj, module.up_proj
    if _native_type(gate) is not projection_type or _native_type(up) is not projection_type:
        return False
    if gate.trainable_parameters() or up.trainable_parameters():
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

    def project(self, x, indices, sorted_indices, token_rows = None):
        """`x` holds one row per token when `token_rows` maps each sorted row to its token."""
        group_size, bits, mode = self.metadata
        call = (self.gate, x, self.arrays["weight"], self.arrays["scales"], self.arrays.get("biases"), indices,
                sorted_indices, group_size, bits, mode, None, token_rows)
        result = _nax_int8_prefill_gather(*call)
        if result is None:
            result = _nax_sorted_gather(*call)
        if result is None:
            result = mx.gather_qmm(
                x if token_rows is None else x[token_rows],
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


def _fused_moe_gate_up_class(original_class, projection_type, scatter_unsort, decode_block, bindings):
    key = (original_class, projection_type, scatter_unsort, decode_block)  # each class guards its own resolution
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
            inv_order = token_rows = None
            if do_sort:   # `_gather_sort` inlined: projections read the token rows through `token_rows`
                order = mx.argsort(indices.flatten())
                inv_order = mx.argsort(order)
                x, idx, token_rows = x.flatten(0, -3), indices.flatten()[order], order // indices.shape[-1]
            x_gate, x_up = packed.project(x, idx, do_sort, token_rows)
            x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices = do_sort)
            if do_sort:
                x = scatter_unsort(x, inv_order, indices.shape)
            return x.squeeze(-2)

        def sorted_experts(self, x, indices):
            """Expert outputs in expert-sorted route order and the inverse order, or None for the native call."""
            packed = self._unsloth_moe_gate_up
            if (self.training or not packed.matches(self) or not _bindings_intact(bindings)
                    or x.ndim != 3 or x.shape[1] <= max(decode_block, 1) or indices.size < 64):
                return None
            order = mx.argsort(indices.flatten())
            x, idx, token_rows = mx.expand_dims(x, (-2, -3)).flatten(0, -3), indices.flatten()[order], order // indices.shape[-1]
            x_gate, x_up = packed.project(x, idx, True, token_rows)
            inv_order = mx.argsort(order)
            return self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices = True), inv_order

        _MOE_GATE_UP_CLASSES[key] = type(
            f"_FusedMoEGateUp{original_class.__name__}", (original_class,),
            {"__call__": fused_call, "sorted_experts": sorted_experts},
        )
    return _MOE_GATE_UP_CLASSES[key]


@contextmanager
def _uncached_allocations():
    """Free buffers to the driver: a pack replaces two arrays with one of their combined size,
    which neither freed buffer can serve, so cached they only accumulate."""
    previous_limit = mx.set_cache_limit(0)
    try:
        yield
    finally:
        mx.set_cache_limit(previous_limit)


@contextmanager
def fused_moe_gate_up(model, *, _modules = None):
    """Fuse quantized MoE gate and up projections while their weights stay fixed.

    Repacking at entry makes weight edits between scopes visible. Overlapping scopes
    share packing until the last exit; training or custom projections retain native calls.
    """
    patched = []
    try:
        with _MOE_GATE_UP_LOCK, _uncached_allocations():
            if not getattr(model, "_unsloth_mlx_distributed_parallel_mode", None):
                specs = _moe_switch_specs()
                modules = _fusion_modules(model, _modules)
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
def _decode_conv_kernel():
    try:
        if not mx.metal.is_available():
            return None
        return mx.fast.metal_kernel(
            name = "unsloth_decode_conv",
            input_names = ["state", "x", "w"], output_names = ["window", "q", "k", "v"],
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
                for (uint tap = 0; tap < K; ++tap) {
                    T value = tap + 1 < K
                        ? state[b*state_strides[0] + tap*state_strides[1] + c*state_strides[2]]
                        : x[b*x_strides[0] + c*x_strides[2]];
                    window[(b*K + tap)*C + c] = value;
                    float product = float(value) * w[tap*w_strides[0] + c*w_strides[1]];
                    acc = product + acc;
                }
                T conv = T(acc);
                auto y = 1 / (1 + metal::precise::exp(metal::abs(conv)));
                T sigmoid = (conv < 0) ? y : 1 - y;
                T out = conv * sigmoid;
                if (c < KEY) q[b*KEY + c] = out;
                else if (c < 2*KEY) k[b*KEY + c - KEY] = out;
                else v[b*(C - 2*KEY) + c - 2*KEY] = out;
            """,
        )
    except (AttributeError, TypeError):
        return None


@mx.compile
def _decode_conv(state, x, weight, key_dim):
    # Native small-column reduction order and both half-precision casts, plus the window and q/k/v split.
    batch, taps, channels = x.shape[0], weight.shape[0], x.shape[2]
    return _decode_conv_kernel()(
        inputs = [state, x, weight], template = [("T", x.dtype), ("K", taps), ("KEY", key_dim)],
        grid = (channels, batch, 1), threadgroup = (256, 1, 1),
        output_shapes = [(batch, taps, channels), (batch, 1, key_dim), (batch, 1, key_dim),
                         (batch, 1, channels - 2 * key_dim)],
        output_dtypes = [x.dtype] * 4,
    )


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


_CONCATENATE = "conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)"
_DECODE_TEST = ("S == 1 and conv_input.shape[1] == self.conv_kernel_size"
                " and self.conv1d.weight.dtype in (mx.bfloat16, mx.float16)")
_DECODE_BRANCH = "conv_out = nn.silu(self._causal_conv1d_decode(conv_input))"
_SPLIT = "mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1)"
_PURE_TEST_NODES = (ast.expr_context, ast.boolop, ast.unaryop, ast.cmpop,
                    ast.Name, ast.Attribute, ast.Constant, ast.Compare, ast.BoolOp, ast.UnaryOp)


def _decode_conv_sites(outer):
    """(concatenate index, branch chain index, tests ahead of the decode branch), or None.

    The fused call decides at the concatenate, so those tests must be call-free and read nothing assigned after it."""
    body = outer.body
    starts = [i for i, statement in enumerate(body) if _source_expression(statement) == _CONCATENATE]
    if len(starts) != 1:
        return None
    start = starts[0]
    for index in range(start + 1, len(body)):
        node, before = body[index], []
        while isinstance(node, ast.If):
            if (ast.dump(node.test) == ast.dump(ast.parse(_DECODE_TEST, mode = "eval").body) and len(node.body) == 1
                    and _source_expression(node.body[0]) == _DECODE_BRANCH):
                between = [n for statement in body[start:index] for n in ast.walk(statement)]
                assigned = {n.id for n in between if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                # A window write would land after the fused call already read it; cache slots are native.
                if "S" in assigned or any(isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, ast.Store)
                                          and not (isinstance(n, ast.Subscript) and _source_expression(n.value) == "cache")
                                          for n in between):
                    return None
                # The split reads conv_out and key_dim, which the fused call also fixes at the concatenate.
                if index + 1 == len(body) or _SPLIT not in [
                        _source_expression(n) for n in ast.walk(body[index + 1]) if isinstance(n, ast.Call)]:
                    return None
                if all(isinstance(n, _PURE_TEST_NODES) and not (isinstance(n, ast.Name) and n.id in assigned)
                       for test in before for n in ast.walk(test)):
                    return start, index, before
                return None
            before.append(node.test)
            node = node.orelse[0] if len(node.orelse) == 1 else None
    return None


@functools.cache
def _decode_conv_bodies_hold(call, prepare, conv):
    """Whether these bodies are the ones the fused decode convolution rewrites; live bindings are the caller's."""
    outer, inner = _function_ast(call), _function_ast(prepare)
    target = inner.body[-1].value
    if not (isinstance(inner.body[-1], ast.Return) and isinstance(target, ast.Call)
            and isinstance(target.func, ast.Name)
            and [_source_expression(a) for a in target.args] == ["conv_input", "weight"]
            and not target.keywords):
        return False
    # The fused call reads the prepared taps before any window exists.
    if any(isinstance(n, ast.Name) and n.id == "conv_input"
           for statement in inner.body[:-1] for n in ast.walk(statement)):
        return False
    if [_source_expression(n) for n in _function_ast(conv).body] != [
        "out = mx.sum(conv_input.astype(mx.float32) * weight[None, :, :], axis=1)",
        "return out.astype(conv_input.dtype)[:, None, :]",
    ]:
        return False
    names = [n for n in ast.walk(outer) if isinstance(n, ast.Name)]
    if (sum(n.id == "conv_input" and isinstance(n.ctx, ast.Store) for n in names) != 1
            or sum(n.id == "conv_out" and isinstance(n.ctx, ast.Load) for n in names) != 1
            or any(n.id == "_unsloth_qkv" for n in names)):
        return False
    calls = [_source_expression(n) for n in ast.walk(outer) if isinstance(n, ast.Call)]
    return (calls.count("nn.silu(self._causal_conv1d_decode(conv_input))") == 1
            and calls.count(_SPLIT) == 1 and _decode_conv_sites(outer) is not None)


def _decode_conv_contract(base):
    call = getattr(base, "__call__", None)
    prepare = getattr(base, "_causal_conv1d_decode", None)
    if not isinstance(call, FunctionType) or not isinstance(prepare, FunctionType):
        return None
    try:
        conv = prepare.__globals__.get(_function_ast(prepare).body[-1].value.func.id)
        if any(hasattr(f, "__wrapped__") for f in (call, prepare, conv)):
            return None
        if not _decode_conv_bodies_hold(call, prepare, conv) or inspect.unwrap(conv).__globals__.get("mx") is not mx:
            return None
        if _resolved_bindings(_CONV_SILU_CONTRACT) is None:
            return None
        if call.__closure__ or prepare.__closure__:
            return None
        if call.__globals__.get("nn") is not nn or call.__globals__.get("mx") is not mx:
            return None
        return call, prepare, conv, nn.silu
    except (OSError, TypeError, SyntaxError, AttributeError, IndexError):
        return None


# The single-token step below repeats each stock kernel's per-lane order and calls the elementwise functors
# MLX's compiled kernels call, on the same typed temporaries, so every intermediate rounds where stock rounds.
_DECODE_STEP_SOURCE = """
    constexpr uint NT = 32 * TY;
    constexpr uint KEY = Hk * Dk;
    constexpr uint C = 2 * KEY + Hv * Dv;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint tid = sg * 32 + lane;
    uint n = threadgroup_position_in_grid.z;
    uint row = n / Hv;
    uint hv = n % Hv;
    uint hk = hv / (Hv / Hk);
    threadgroup T qs[Dk];
    threadgroup T ks[Dk];
    threadgroup T vs[Dv];
    threadgroup T ys[Dv];
    threadgroup float inv_norm[1];

    {
        // The window, depthwise taps and SiLU, as the decode convolution kernel computes them.
        #pragma clang fp contract(off) reassociate(off)
        for (uint i = tid; i < 2 * Dk + Dv; i += NT) {
            uint c = i < Dk ? hk * Dk + i : i < 2 * Dk ? KEY + hk * Dk + i - Dk : 2 * KEY + hv * Dv + i - 2 * Dk;
            bool owner = i >= 2 * Dk || hv % (Hv / Hk) == 0;
            float acc = 0.0f;
            for (uint tap = 0; tap < K; ++tap) {
                T value = tap + 1 < K ? conv_state[(row * (K - 1) + tap) * C + c] : qkv[row * C + c];
                if (owner) window[(row * K + tap) * C + c] = value;
                float product = float(value) * conv_weight[tap * C + c];
                acc = product + acc;
            }
            T conv = static_cast<T>(acc);
            T act = Multiply()(conv, static_cast<T>(Sigmoid()(conv)));
            if (i < Dk) qs[i] = act;
            else if (i < 2 * Dk) ks[i - Dk] = act;
            else vs[i - 2 * Dk] = act;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // mx.fast.rms_norm without a weight (rms_single_row), then the scalar head scale.
    if (sg < 2) {
        threadgroup T* x = sg == 0 ? qs : ks;
        T scale = sg == 0 ? q_scale[0] : k_scale[0];
        float thread_x[4];
        float acc = 0;
        for (int i = 0; i < 4; i++) {
            thread_x[i] = x[lane * 4 + i];
            acc += thread_x[i] * thread_x[i];
        }
        acc = simd_sum(acc);
        acc = simd_sum(lane == 0 ? acc : 0.0f);
        float inv = metal::precise::rsqrt(acc / Dk + 1e-6f);
        for (int i = 0; i < 4; i++) {
            T normed = static_cast<T>(1) * static_cast<T>(thread_x[i] * inv);
            x[lane * 4 + i] = Multiply()(scale, normed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    PT gate = Add()(static_cast<PT>(a[n]), static_cast<PT>(dt_bias[hv]));
    float softplus = static_cast<float>(static_cast<PT>(LogAddExp()(gate, static_cast<PT>(0))));
    float decay = Exp()(Multiply()(Negative()(Exp()(static_cast<float>(A_log[hv]))), softplus));
    T beta = Sigmoid()(b[n]);

    // The generic gated-delta kernel's recurrence, one simdgroup per value row.
    for (uint j = 0; j < Dv / TY; ++j) {
        uint dv = sg * (Dv / TY) + j;
        auto i_state = state_in + (n * Dv + dv) * Dk + lane * 4;
        auto o_state = state_out + (n * Dv + dv) * Dk + lane * 4;
        float state[4];
        for (int i = 0; i < 4; ++i) {
            state[i] = static_cast<float>(i_state[i]);
        }
        float kv_mem = 0.0f;
        for (int i = 0; i < 4; ++i) {
            state[i] = state[i] * decay;
            kv_mem += state[i] * ks[lane * 4 + i];
        }
        kv_mem = simd_sum(kv_mem);
        auto delta = (vs[dv] - kv_mem) * beta;
        float out = 0.0f;
        for (int i = 0; i < 4; ++i) {
            state[i] = state[i] + ks[lane * 4 + i] * delta;
            out += state[i] * qs[lane * 4 + i];
        }
        out = simd_sum(out);
        if (lane == 0) {
            ys[dv] = static_cast<T>(out);
        }
        for (int i = 0; i < 4; ++i) {
            o_state[i] = state[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The gated RMSNorm: rms_single_row with the norm weight, then the float32 SiLU gate.
    if (sg == 0) {
        float acc = 0;
        for (int i = 0; i < 4; i++) {
            float xi = ys[lane * 4 + i];
            acc += xi * xi;
        }
        acc = simd_sum(acc);
        acc = simd_sum(lane == 0 ? acc : 0.0f);
        if (lane == 0) {
            inv_norm[0] = metal::precise::rsqrt(acc / Dv + norm_eps[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < Dv; d += NT) {
        T normed = norm_weight[d] * static_cast<T>(static_cast<float>(ys[d]) * inv_norm[0]);
        float zf = static_cast<float>(z[n * Dv + d]);
        float silu = Multiply()(zf, Sigmoid()(zf));
        y[n * Dv + d] = static_cast<T>(Multiply()(silu, static_cast<float>(normed)));
    }
"""
# Simdgroups per value head; each carries Dv / 16 state rows. Layers run one after another in a decode step, so
# a few heads per row must fill the GPU: 16 measured up to 18% faster per layer than 8 at <= 64 heads.
_DECODE_STEP_ROWS = 16


@functools.cache
def _decode_step_kernel():
    try:
        header = mx.metal.is_available() and nax._mlx_kernel_header("unary_ops.h")
        binary = header and nax._mlx_kernel_header("binary_ops.h")
        return binary and mx.fast.metal_kernel(
            name = "unsloth_gated_delta_decode_step",
            input_names = ["conv_state", "qkv", "conv_weight", "a", "b", "A_log", "dt_bias", "z", "norm_weight",
                           "q_scale", "k_scale", "norm_eps", "state_in"],
            output_names = ["window", "y", "state_out"], header = header + binary,
            compile_options = {"math_mode": "safe"}, source = _DECODE_STEP_SOURCE,
        ) or None
    except (AttributeError, TypeError):
        return None


@functools.lru_cache(maxsize = 64)
def _decode_step_scalars(head_dim, dtype, eps):
    # `inv_scale * x` converts the Python scale to x's dtype; going through the same multiply keeps its rounding.
    inv_scale, one = head_dim ** -0.5, mx.ones((1,), dtype)
    scalars = ((inv_scale ** 2) * one, inv_scale * one, mx.array([eps], mx.float32))
    mx.eval(*scalars)
    return scalars


def _decode_step(conv_state, qkv, weight, a, b, A_log, dt_bias, z, norm_weight, eps, state, key_heads):
    B, value_heads, Dv, Dk = state.shape
    dtype = qkv.dtype
    q_scale, k_scale, norm_eps = _decode_step_scalars(Dk, dtype, eps)
    return _decode_step_kernel()(
        inputs = [conv_state, qkv, weight, a, b, A_log, dt_bias, z, norm_weight, q_scale, k_scale, norm_eps, state],
        template = [("T", dtype), ("PT", a.dtype if a.dtype == dt_bias.dtype else mx.float32), ("K", weight.shape[0]),
                    ("Hk", key_heads), ("Hv", value_heads), ("Dk", Dk), ("Dv", Dv), ("TY", _DECODE_STEP_ROWS)],
        grid = (32, _DECODE_STEP_ROWS, B * value_heads), threadgroup = (32, _DECODE_STEP_ROWS, 1),
        output_shapes = [(B, weight.shape[0], qkv.shape[-1]), (B, 1, value_heads, Dv), state.shape],
        output_dtypes = [dtype, dtype, state.dtype],
    )


# Bodies the decode step mirrors. `gated_delta_kernel` is left out: the packed scope rebinds it, and its
# single-token route is the generic kernel whose arithmetic `_make_gated_delta_kernel` pins.
_DECODE_STEP_CONTRACT = {
    "mlx_vlm.models.qwen3_5.language": {
        "Qwen3_5GatedDeltaNet.__call__": "f8deff65c803d4da",
        "Qwen3_5GatedDeltaNet._normalize_qk": "8f16e229aec08b7a",
        "Qwen3_5GatedDeltaNet._project_gates": "817d9767396ec3f8",
        "Qwen3_5RMSNormGated.__call__": "7c0c74be3b59d596",
        "_precise_swiglu": "d5cc9afa38657b66",
        "_qwen3_5_advance_lengths_info": "67db74645bc98ea9",
    },
    "mlx_vlm.models.qwen3_5.gated_delta": {
        "_compute_g_beta": "1185629ef7a77fba", "compute_g": "837fdb794c06f7f5",
        "_make_gated_delta_kernel": "d3141d83bfd26d47",
    },
    "mlx_vlm.models.cache": {"ArraysCache.update_window": "5580299b90d3a0b1",
                             "ArraysCache.update_recurrent": "c181bba469444c22"},
    "mlx.nn": {"silu": "78867fdb7e42731c", "softplus": "4d48a9bee57aeadd"},
}
_DECODE_STEP_UPDATE = "dc29a058a40f2dce"
_DECODE_STEP_METHODS = ("_causal_conv1d_decode", "_normalize_qk")  # the kernel replaces their calls
_DECODE_STEP_VERDICTS = {}


def _decode_step_contract(base):
    """The functions the decode step of `base` calls, when it decodes through the pinned bodies, else None."""
    language, source = (sys.modules.get(f"mlx_vlm.models.qwen3_5.{name}") for name in ("language", "gated_delta"))
    cache = sys.modules.get("mlx_vlm.models.cache")
    if language is None or source is None or cache is None or os.environ.get("UNSLOTH_MLX_FUSED_GDN_DECODE") == "0":
        return None
    layer = language.Qwen3_5GatedDeltaNet
    if any(getattr(base, name, None) is not getattr(layer, name)
           for name in ("__call__", "_normalize_qk", "_project_gates", "_causal_conv1d_decode")):
        return None
    if language.ArraysCache is not cache.ArraysCache or "gated_delta_update" not in vars(language):
        return None
    if _resolved_bindings(_DECODE_STEP_CONTRACT) is None:
        return None
    # Unsloth's training wrapper hands inference calls, which carry a state, to the update it replaced.
    bound = update = vars(language)["gated_delta_update"]
    if getattr(source, "_unsloth_gated_delta_patched", False) and update is source.gated_delta_update:
        update = source._unsloth_gated_delta_original
    try:
        if _ast_fingerprint(update) != _DECODE_STEP_UPDATE or update.__globals__ is not vars(source):
            return None
    except (OSError, TypeError, SyntaxError, AttributeError):
        return None
    return (language, bound, update, cache.ArraysCache.update_window, cache.ArraysCache.update_recurrent,
            language._qwen3_5_advance_lengths_info, language.Qwen3_5RMSNormGated.__call__)


def _decode_step_route(language, bound, update, update_window, update_recurrent, advance_lengths, norm_call):
    bindings = _resolved_bindings(_DECODE_STEP_CONTRACT)  # this resolution's, not a later one's
    if bindings is None:
        return None
    bindings.append((vars(language), "gated_delta_update", bound))

    def decode_step(self, inputs, mask = None, cache = None):
        """The layer's output for one decode token on the fused kernel, or None where the native call decides."""
        if (mask is not None or cache is None or inputs.ndim != 3 or inputs.shape[1] != 1
                or getattr(type(cache), "update_window", None) is not update_window
                or getattr(type(cache), "update_recurrent", None) is not update_recurrent
                or any(name in getattr(cache, "__dict__", ()) for name in ("update_window", "update_recurrent"))
                or type(self.norm).__call__ is not norm_call or not _bindings_intact(bindings)):
            return None
        B, K, Dk, Dv = inputs.shape[0], self.conv_kernel_size, self.head_k_dim, self.head_v_dim
        conv_state, state = cache[0], cache[1]
        if (conv_state is None or state is None or B == 0 or Dk != 128 or Dv != 128 or not 2 <= K <= 8
                or self.num_v_heads % self.num_k_heads or any(name in vars(self) for name in _DECODE_STEP_METHODS)
                or conv_state.shape != (B, K - 1, self.conv_dim) or conv_state.dtype not in (mx.bfloat16, mx.float16)
                or state.shape != (B, self.num_v_heads, Dv, Dk) or state.dtype != mx.float32
                or self.conv1d.weight.dtype not in (mx.bfloat16, mx.float16)
                or self.norm.weight.dtype != conv_state.dtype or self.norm.weight.shape != (Dv,)
                or self.A_log.shape != (self.num_v_heads,) or self.dt_bias.shape != self.A_log.shape
                or any(t.dtype not in (mx.float32, mx.bfloat16, mx.float16) for t in (self.A_log, self.dt_bias))):
            return None
        dtype = conv_state.dtype
        verdict_key = (self.num_k_heads, self.num_v_heads, K, dtype, self.A_log.dtype, self.dt_bias.dtype)
        verdict = _DECODE_STEP_VERDICTS.get(verdict_key)
        if verdict is False:
            return None
        mixed_qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs)
        b, a = self._project_gates(inputs)
        # Past the projections the stock branch finishes whatever the kernel cannot take, so none runs twice.
        fits = (all(t.dtype == dtype for t in (mixed_qkv, z, a, b)) and mixed_qkv.shape == (B, 1, self.conv_dim)
                and z.shape == (B, 1, self.value_dim) and a.shape == (B, 1, self.num_v_heads) and b.shape == a.shape)
        z = z.reshape(B, 1, -1, Dv)
        if fits:
            weight = self._unsloth_decode_conv_weight(None)
            cached = self.__dict__.get("_unsloth_decode_step_weight")
            if cached is None or cached[0] is not weight:
                cached = (weight, mx.contiguous(weight))
                mx.eval(cached[1])
                self.__dict__["_unsloth_decode_step_weight"] = cached
            fused = lambda: _decode_step(conv_state, mixed_qkv, cached[1], a, b, self.A_log, self.dt_bias, z,
                                         self.norm.weight, self.norm.eps, state, self.num_k_heads)
        if fits and verdict:
            window, out, new_state = fused()
        else:
            window = mx.concatenate([conv_state, mixed_qkv], axis = 1)
            conv_out = nn.silu(self._causal_conv1d_decode(window))
            q, k, v = [t.reshape(B, 1, h, d) for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads], [Dk, Dk, Dv])]
            q, k = self._normalize_qk(q, k)
            out, new_state = update(q, k, v, a, b, self.A_log, self.dt_bias, state = state, use_kernel = True)
            out = self.norm(out, z)
            if fits and verdict is None:
                try:
                    verdict = all(bool(mx.array_equal(x.view(mx.uint8), y.view(mx.uint8)).item())
                                  for x, y in zip(fused(), (window, out, new_state)))
                except (RuntimeError, ValueError):
                    verdict = None  # inside a function transformation, which cannot evaluate; checked later
                else:
                    _DECODE_STEP_VERDICTS[verdict_key] = verdict
                    if not verdict:
                        logger.warning("the fused gated-delta decode step differs from the native layer for %d key "
                                       "heads, %d value heads, a %d-tap convolution and %s inputs; the native "
                                       "layer stays in use", *verdict_key[:4])
        cache.update_window(0, window, K - 1, lengths = cache.lengths)
        cache.update_recurrent(1, 1, lambda initial, steps: (out, new_state))
        if hasattr(cache, "advance"):
            cache.advance(1)
            advance_lengths(cache, 1)
        return self.out_proj(out.reshape(B, 1, -1))

    return decode_step


@functools.cache
def _fused_decode_conv_class(base, call, prepare, conv, silu, step = None):
    bindings = _resolved_bindings(_CONV_SILU_CONTRACT) or []  # this resolution's, not a later one's
    outer, inner = copy.deepcopy(_function_ast(call)), copy.deepcopy(_function_ast(prepare))
    start, index, before = _decode_conv_sites(outer)
    guard = " and ".join(f"not ({_source_expression(test)})" for test in before) or "True"
    outer.body[start] = ast.parse(
        f"conv_input, _unsloth_qkv = self._unsloth_decode_conv(conv_state, mixed_qkv, S, {guard})").body[0]
    outer.body[index] = ast.If(test = ast.parse("_unsloth_qkv is None", mode = "eval").body,
                               body = [outer.body[index]], orelse = [])

    class Rewrite(ast.NodeTransformer):
        def visit_Call(self, node):
            if _source_expression(node) == _SPLIT:
                return ast.IfExp(test = ast.parse("_unsloth_qkv is not None", mode = "eval").body,
                                 body = ast.Name(id = "_unsloth_qkv", ctx = ast.Load()), orelse = node)
            return self.generic_visit(node)

    outer = Rewrite().visit(outer)
    conv_name = inner.body[-1].value.func.id
    inner.body[-1] = ast.Return(value = ast.Name(id = "weight", ctx = ast.Load()))
    inner.name = "_unsloth_decode_conv_weight"
    adapted_call = _function_from_ast(call, outer)

    def decode_conv(self, state, x, rows, eligible):
        if (eligible and rows == 1 and x.ndim == 3 and state.ndim == 3 and x.shape[1] == 1
                and state.shape[0] == x.shape[0] > 0 and state.shape[2] == x.shape[2]
                and state.shape[1] + 1 == self.conv_kernel_size
                and self.conv1d.weight.dtype in (mx.bfloat16, mx.float16)
                and x.dtype in (mx.bfloat16, mx.float16) and state.dtype == x.dtype):
            weight = self._unsloth_decode_conv_weight(None)
            if (weight.dtype == mx.float32 and weight.shape == (state.shape[1] + 1, x.shape[2])
                    and 2 <= weight.shape[0] <= 8 and 0 < 2 * self.key_dim < x.shape[2]):
                window, q, k, v = _decode_conv(state, x, weight, self.key_dim)
                return window, (q, k, v)
        return mx.concatenate([state, x], axis = 1), None

    decode_step = step and _decode_step_route(*step)

    def fused_call(self, *args, **kwargs):
        if (self.training or prepare.__globals__.get(conv_name) is not conv
                or not _bindings_intact(bindings)):
            return call(self, *args, **kwargs)
        if decode_step is not None and (out := decode_step(self, *args, **kwargs)) is not None:
            return out
        return adapted_call(self, *args, **kwargs)

    return type(f"_FusedDecodeConv{base.__name__}", (base,), {
        "__call__": fused_call,
        "_unsloth_decode_conv_weight": _function_from_ast(prepare, inner),
        "_unsloth_decode_conv": decode_conv,
    })


@contextmanager
def fused_decode_conv_silu(model, *, _modules = None):
    """Fuse the recurrent decode conv window, convolution, SiLU and q/k/v split into one launch during serialized inference.

    mlx-vlm's Qwen3.5 layer runs its whole single-token step, through the gated RMSNorm, in that launch instead,
    once its first step of each head layout and dtype matched the native layer bit for bit;
    `UNSLOTH_MLX_FUSED_GDN_DECODE=0` keeps it on the convolution-only launch.
    Prefill, unsupported convolution shapes, and training keep their native paths.
    Instance classes are restored when the context exits, including on cancellation.
    """
    changed = []
    specs = {}
    try:
        modules = _fusion_modules(model, _modules)
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
                if hasattr(base, "_unsloth_decode_conv"):
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
                    contract = _decode_conv_contract(base)
                    step = contract and _decode_step_kernel() and _decode_step_contract(base)
                    specs[base] = contract and (*contract, step or None)
                contract = specs[base]
                if contract is None or _decode_conv_kernel() is None:
                    continue
                patched = _fused_decode_conv_class(base, *contract)
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
                         "_unsloth_decode_patched", "_unsloth_decode_step_weight"):
                module.__dict__.pop(name, None)


# The generic kernel builder is pinned because the packed kernel repeats its arithmetic.
_GATED_DELTA_BODIES = {"gated_delta_kernel": "84cd89879cc05264", "_make_gated_delta_kernel": "d3141d83bfd26d47",
                       "_gated_delta_kernel": None}
_GATED_DELTA_CONTRACT = {"mlx_vlm.models.qwen3_5.gated_delta": _GATED_DELTA_BODIES,
                         "mlx_lm.models.gated_delta": _GATED_DELTA_BODIES}
_GATED_DELTA_LOCK = RLock()
_GATED_DELTA_SCOPES = {}
_GATED_DELTA_VERDICTS = {}


@functools.cache
def _packed_gated_delta_kernel():
    # The layout and reduction order of mlx-lm's packed kernel (ml-explore/mlx-lm#1559): four lanes own
    # a value row and hold 32 state elements each, and the sums follow the tree simd_sum lowers to.
    try:
        if not mx.metal.is_available():
            return None
        return mx.fast.metal_kernel(
            name = "unsloth_gated_delta_packed",
            input_names = ["q", "k", "v", "g", "beta", "state_in", "T"], output_names = ["y", "state_out"],
            source = """
                constexpr int lanes_per_row = 4;
                constexpr int rows_per_simdgroup = 32 / lanes_per_row;
                constexpr int values_per_lane = Dk / lanes_per_row;
                constexpr int partials_per_lane = values_per_lane / 4;

                auto n = thread_position_in_grid.z;
                auto b_idx = n / Hv;
                auto hv_idx = n % Hv;
                auto hk_idx = hv_idx / (Hv / Hk);

                auto lane = thread_index_in_simdgroup;
                auto row_in_simdgroup = lane / lanes_per_row;
                auto lane_in_row = lane & (lanes_per_row - 1);
                auto dv_idx = thread_position_in_grid.y * rows_per_simdgroup + row_in_simdgroup;

                auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
                auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
                auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
                y += (b_idx * T * Hv + hv_idx) * Dv;
                auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
                auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

                float state[values_per_lane];
                for (int i = 0; i < values_per_lane; ++i) {
                    state[i] = static_cast<float>(i_state[i]);
                }

                auto g_ = g + b_idx * T * Hv;
                auto beta_ = beta + b_idx * T * Hv;

                for (int t = 0; t < T; ++t) {
                    float gt = static_cast<float>(g_[hv_idx]);

                    // Each 4-element chain is one lane's sequential accumulation in the generic kernel.
                    float part[partials_per_lane];
                    for (int pb = 0; pb < partials_per_lane; ++pb) {
                        float acc = 0.0f;
                        for (int i = 0; i < 4; ++i) {
                            int e = pb * 4 + i;
                            state[e] = state[e] * gt;
                            acc += state[e] * static_cast<float>(k_[e]);
                        }
                        part[pb] = acc;
                    }
                    // Butterfly levels xor 1, 2, 4 fall inside this lane; 8 and 16 are the row shuffles.
                    float kv_mem =
                        ((part[0] + part[1]) + (part[2] + part[3])) +
                        ((part[4] + part[5]) + (part[6] + part[7]));
                    kv_mem += simd_shuffle_xor(kv_mem, 1);
                    kv_mem += simd_shuffle_xor(kv_mem, 2);

                    auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * static_cast<float>(beta_[hv_idx]);

                    for (int pb = 0; pb < partials_per_lane; ++pb) {
                        float acc = 0.0f;
                        for (int i = 0; i < 4; ++i) {
                            int e = pb * 4 + i;
                            state[e] = state[e] + static_cast<float>(k_[e]) * delta;
                            acc += state[e] * static_cast<float>(q_[e]);
                        }
                        part[pb] = acc;
                    }
                    float out =
                        ((part[0] + part[1]) + (part[2] + part[3])) +
                        ((part[4] + part[5]) + (part[6] + part[7]));
                    out += simd_shuffle_xor(out, 1);
                    out += simd_shuffle_xor(out, 2);
                    if (lane_in_row == 0) {
                        y[dv_idx] = static_cast<InT>(out);
                    }

                    q_ += Hk * Dk;
                    k_ += Hk * Dk;
                    v_ += Hv * Dv;
                    y += Hv * Dv;
                    g_ += Hv;
                    beta_ += Hv;
                }

                for (int i = 0; i < values_per_lane; ++i) {
                    o_state[i] = static_cast<StT>(state[i]);
                }
            """,
        )
    except (AttributeError, TypeError):
        return None


def _packed_gated_delta(q, k, v, g, beta, state):
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    return _packed_gated_delta_kernel()(
        inputs = [q, k, v, g, beta, state, T],
        template = [("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid = (32, Dv // 8, B * Hv), threadgroup = (32, 2, 1),
        output_shapes = [(B, T, Hv, Dv), state.shape], output_dtypes = [q.dtype, state.dtype],
    )


def _packed_gated_delta_route(native, bindings):
    def gated_delta_kernel(q, k, v, g, beta, state, mask = None):
        if (mask is not None or g.ndim != 3 or k.shape[1] < 2 or k.shape[3] != 128 or v.shape[3] % 8
                or g.dtype != mx.float32 or state.dtype != mx.float32 or not _bindings_intact(bindings)):
            return native(q, k, v, g, beta, state, mask)
        key = (k.shape[2], *v.shape[2:], q.dtype, k.dtype, v.dtype, beta.dtype)
        verdict = _GATED_DELTA_VERDICTS.get(key)
        if verdict:
            return _packed_gated_delta(q, k, v, g, beta, state)
        stock = native(q, k, v, g, beta, state, mask)
        if verdict is None:
            try:
                verdict = all(bool(mx.array_equal(a.view(mx.uint8), b.view(mx.uint8)).item())
                              for a, b in zip(_packed_gated_delta(q, k, v, g, beta, state), stock))
            except (RuntimeError, ValueError):
                return stock  # inside a function transformation, which cannot evaluate; checked later
            _GATED_DELTA_VERDICTS[key] = verdict
            if not verdict:
                logger.warning("the packed gated-delta kernel differs from the generic one for %d key heads, "
                               "%d value heads of width %d and %s inputs; the generic kernel stays in use",
                               *key[:4])
        return stock

    return gated_delta_kernel


def _gated_delta_sources(modules):
    """The pinned gated-delta modules whose `gated_delta_update` a layer of these modules calls."""
    updates = set()
    for cls in {type(module) for _, module in modules}:
        for base in cls.__mro__:
            update = getattr(sys.modules.get(base.__module__), "gated_delta_update", None)
            if update is not None:
                updates.add(id(update))
    return [name for name in _GATED_DELTA_CONTRACT
            if id(getattr(sys.modules.get(name), "gated_delta_update", None)) in updates]


@contextmanager
def fused_packed_gated_delta(model, *, _modules = None):
    """Run multi-token gated-delta calls on a packed kernel, bit-identical to the generic one.

    Each head layout and dtype is compared bit for bit on first use. `gated_delta_kernel` is rebound
    process-wide while any scope is open. `UNSLOTH_MLX_PACKED_GATED_DELTA=0` turns the route off.
    """
    entered = []
    try:
        with _GATED_DELTA_LOCK:
            if (os.environ.get("UNSLOTH_MLX_PACKED_GATED_DELTA", "1") != "0"
                    and not getattr(model, "_unsloth_mlx_distributed_parallel_mode", None)
                    and _packed_gated_delta_kernel() is not None):
                for name in _gated_delta_sources(_fusion_modules(model, _modules)):
                    namespace = vars(sys.modules[name])
                    held = _GATED_DELTA_SCOPES.get(name)
                    if held is None:
                        if "_gated_delta_kernel_packed" in namespace:
                            continue
                        bindings = _resolved_bindings({name: _GATED_DELTA_BODIES})
                        if bindings is None:
                            continue
                        native = namespace["gated_delta_kernel"]
                        route = _packed_gated_delta_route(native, [b for b in bindings if b[2] is not native])
                        namespace["gated_delta_kernel"] = route
                        held = _GATED_DELTA_SCOPES[name] = [0, name, namespace, native, route]
                    held[0] += 1
                    entered.append(held)
        yield model
    finally:
        with _GATED_DELTA_LOCK:
            for held in entered:
                held[0] -= 1
                if not held[0]:
                    _, name, namespace, native, route = held
                    del _GATED_DELTA_SCOPES[name]
                    if namespace.get("gated_delta_kernel") is route:
                        namespace["gated_delta_kernel"] = native

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

_HALF = (mx.bfloat16, mx.float16)


@functools.lru_cache(maxsize = 64)
def _scalar(value):
    return mx.array(value, mx.int32)


# MLX's rms_norm reads N_READS values per thread, and rows wider than RMS_LOOPED_LIMIT take its
# looped kernel with a full 1024-thread group; the add-then-norm kernels partition rows the same way.
_NORM_READS = 4
_NORM_LOOPED_LIMIT = 4096
_NORM_LOOPED_GROUP = 1024

_NORM = r"""
constant constexpr int N_READS = 4;
constant constexpr int SIMD_SIZE = 32;

template <typename T>
using Vec = vec<T, N_READS>;

template <typename T>
inline Vec<T> load(const device T* p) {
  return *reinterpret_cast<const device Vec<T>*>(p);
}

template <typename T>
inline Vec<T> load(const constant T* p) {
  return *reinterpret_cast<const constant Vec<T>*>(p);
}

template <typename T>
inline void store(device T* p, Vec<T> v) {
  *reinterpret_cast<device Vec<T>*>(p) = v;
}

// Rows and buffer offsets are multiples of N_READS elements, so N_READS-wide vector loads are legal.
template <typename T, typename P>
inline bool vector_aligned(P p) {
  return reinterpret_cast<ulong>(p) % (N_READS * sizeof(T)) == 0;
}

// MLX zeroes local_sums behind a barrier; reading unused lanes as 0 gives simd_sum the same inputs.
inline float inv_rms(
    float acc,
    uint axis_size,
    float eps,
    uint simd_lane_id,
    uint simd_group_id,
    uint simd_groups,
    threadgroup float* local_sums,
    threadgroup float* local_inv_mean) {
  acc = simd_sum(acc);
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    acc = simd_sum(simd_lane_id < simd_groups ? local_sums[simd_lane_id] : 0.0f);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return local_inv_mean[0];
}

template <typename T, bool divisible, typename X, typename R, typename W>
METAL_FUNC void add_rms_single_row_body(
    X x,
    R r,
    W w,
    device T* h,
    device T* out,
    float eps,
    uint axis,
    uint gid,
    uint lid,
    uint simd_lane_id,
    uint simd_group_id,
    uint simd_groups,
    threadgroup float* local_sums,
    threadgroup float* local_inv_mean) {
  const uint axis_size = axis;
  const bool aligned = divisible && vector_aligned<T>(x) && vector_aligned<T>(r) && vector_aligned<T>(w);

  size_t offset = gid * size_t(axis_size) + lid * N_READS;
  x += offset;
  r += offset;
  h += offset;
  out += offset;
  w += lid * N_READS;

  float acc = 0;
  float thread_x[N_READS];
  bool full = lid * N_READS + N_READS <= axis_size;
  if (full && aligned) {
    Vec<T> v = load(x) + load(r);
    store(h, v);
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = v[i];
      acc += thread_x[i] * thread_x[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (full || lid * N_READS + i < axis_size) {
        T v = x[i] + r[i];
        h[i] = v;
        thread_x[i] = v;
      } else {
        thread_x[i] = 0;
      }
      acc += thread_x[i] * thread_x[i];
    }
  }
  float inv = inv_rms(
      acc, axis_size, eps, simd_lane_id, simd_group_id, simd_groups, local_sums, local_inv_mean);

  if (full && aligned) {
    Vec<T> wv = load(w);
    Vec<T> o;
    for (int i = 0; i < N_READS; i++) {
      o[i] = wv[i] * static_cast<T>(thread_x[i] * inv);
    }
    store(out, o);
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (full || lid * N_READS + i < axis_size) {
        out[i] = w[i] * static_cast<T>(thread_x[i] * inv);
      }
    }
  }
}

template <typename T, bool divisible, typename X, typename R, typename W>
METAL_FUNC void add_rms_looped_body(
    X x,
    R r,
    W w,
    device T* h,
    device T* out,
    float eps,
    uint axis,
    uint gid,
    uint lid,
    uint lsize,
    uint simd_lane_id,
    uint simd_group_id,
    uint simd_groups,
    threadgroup float* local_sums,
    threadgroup float* local_inv_mean) {
  const uint axis_size = axis;
  const bool aligned = divisible && vector_aligned<T>(x) && vector_aligned<T>(r) && vector_aligned<T>(w);

  size_t offset = gid * size_t(axis_size) + lid * N_READS;
  x += offset;
  r += offset;
  h += offset;
  out += offset;
  w += lid * N_READS;

  float acc = 0;
  for (uint rr = 0; rr < axis_size; rr += lsize * N_READS) {
    bool full = rr + lid * N_READS + N_READS <= axis_size;
    if (full && aligned) {
      Vec<T> v = load(x + rr) + load(r + rr);
      store(h + rr, v);
      for (int i = 0; i < N_READS; i++) {
        float xi = v[i];
        acc += xi * xi;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if (full || rr + lid * N_READS + i < axis_size) {
          T v = x[rr + i] + r[rr + i];
          h[rr + i] = v;
          float xi = v;
          acc += xi * xi;
        }
      }
    }
  }
  float inv = inv_rms(
      acc, axis_size, eps, simd_lane_id, simd_group_id, simd_groups, local_sums, local_inv_mean);

  for (uint rr = 0; rr < axis_size; rr += lsize * N_READS) {
    bool full = rr + lid * N_READS + N_READS <= axis_size;
    if (full && aligned) {
      Vec<T> hv = load(h + rr);
      Vec<T> wv = load(w + rr);
      Vec<T> o;
      for (int i = 0; i < N_READS; i++) {
        o[i] = wv[i] * static_cast<T>(hv[i] * inv);
      }
      store(out + rr, o);
    } else {
      for (int i = 0; i < N_READS; i++) {
        if (full || rr + lid * N_READS + i < axis_size) {
          out[rr + i] = w[rr + i] * static_cast<T>(h[rr + i] * inv);
        }
      }
    }
  }
}
"""

_NORM_BODIES = {
    False: r"""
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];
  add_rms_single_row_body<T, divisible>(
      x, r, w, h, out, eps, axis, threadgroup_position_in_grid.x, thread_position_in_threadgroup.x,
      thread_index_in_simdgroup, simdgroup_index_in_threadgroup, simdgroups_per_threadgroup, local_sums,
      local_inv_mean);
""",
    True: r"""
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];
  add_rms_looped_body<T, divisible>(
      x, r, w, h, out, eps, axis, threadgroup_position_in_grid.x, thread_position_in_threadgroup.x,
      threads_per_threadgroup.x, thread_index_in_simdgroup, simdgroup_index_in_threadgroup, simdgroups_per_threadgroup,
      local_sums, local_inv_mean);
""",
}

@functools.cache
def _norm_kernel(looped):
    try:
        if not mx.metal.is_available():
            return None
        return mx.fast.metal_kernel(
            name = f"unsloth_add_rms_{'looped' if looped else 'single_row'}",
            input_names = ["x", "r", "w", "eps", "axis"],
            output_names = ["h", "out"],
            source = _NORM_BODIES[looped],
            header = _NORM,
        )
    except (AttributeError, RuntimeError, TypeError):
        return None


@functools.lru_cache(maxsize = 64)
def _epsilon(eps):
    return mx.array(eps, mx.float32)


@functools.lru_cache(maxsize = 64)
def _add_rms_plan(shape, dtype):
    axis = shape[-1]
    looped = axis > _NORM_LOOPED_LIMIT
    call = _norm_kernel(looped)
    if call is None:
        return None
    group = _NORM_LOOPED_GROUP if looped else 32 * -(-axis // (32 * _NORM_READS))
    args = dict(
        template = [("T", dtype), ("divisible", axis % _NORM_READS == 0)],
        grid = (math.prod(shape) // axis * group, 1, 1),
        threadgroup = (group, 1, 1),
        output_shapes = [shape, shape],
        output_dtypes = [dtype, dtype],
    )
    width = _scalar(axis)
    return mx.compile(lambda x, r, w, eps: tuple(call(inputs = [x, r, w, eps, width], **args)))


def _fused_add_rms_norm(x, r, w, eps):
    """`(h, mx.fast.rms_norm(h, w, eps))` with `h = x + r`, bitwise, or None for inputs the kernels do not take."""
    if (x.dtype != w.dtype or r.dtype != w.dtype or x.shape != r.shape or x.ndim == 0 or w.ndim != 1
            or x.shape[-1] != w.shape[0] or x.dtype not in (*_HALF, mx.float32) or x.size == 0):
        return None
    launch = _add_rms_plan(x.shape, x.dtype)
    return None if launch is None else launch(x, r, w, _epsilon(eps))


class _Handoff:
    """The residual a decoder layer returns, already normalized by the next layer's input norm."""

    # One (residual, normed, weight, eps) tuple, swapped whole: overlapping generate calls share the
    # slot, and a field-by-field read could pair one call's residual with another's normed output.
    __slots__ = ("consumer", "value")

    def __init__(self, consumer):
        self.consumer = consumer
        self.value = None


_ADD_NORM_VERDICTS = {}


def _add_norm_verified(dtype, axis):
    verdict = _ADD_NORM_VERDICTS.get((dtype, axis))
    if verdict is None:
        keys = mx.random.split(mx.random.key(axis), 3)
        x, r = ((mx.random.normal((3, axis), key = key) * 4).astype(dtype) for key in keys[:2])
        w = mx.random.normal((axis,), key = keys[2]).astype(dtype)
        fused = _fused_add_rms_norm(x, r, w, 1e-6)
        h = x + r
        verdict = fused is not None and bool(mx.array_equal(fused[0], h).item()) and bool(
            mx.array_equal(fused[1], mx.fast.rms_norm(h, w, 1e-6), equal_nan = True).item())
        _ADD_NORM_VERDICTS[dtype, axis] = verdict
        if not verdict:
            logger.warning("The fused residual RMS norm differs from mx.fast.rms_norm for %s rows of %d; "
                           "the native ops stay in use", dtype, axis)
    return verdict


def _add_rms_norm(norm, a, b):
    if (type(norm) is not nn.RMSNorm or mx.default_device() != mx.gpu
            or not isinstance(a, mx.array) or not isinstance(b, mx.array)):
        return None
    out = _fused_add_rms_norm(a, b, norm.weight, norm.eps)
    return out if out is not None and _add_norm_verified(a.dtype, a.shape[-1]) else None


def _is_self_call(node, arg):
    return (isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords
            and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self" and isinstance(node.args[0], ast.Name) and node.args[0].id == arg)


def _self_method(name, *args):
    return ast.Call(func = ast.Attribute(value = ast.Name(id = "self", ctx = ast.Load()), attr = name,
                                         ctx = ast.Load()), args = list(args), keywords = [])


_PLAIN_NODES = (ast.Assign, ast.Return, ast.Name, ast.Attribute, ast.Call, ast.keyword, ast.BinOp,
                ast.Constant, ast.Tuple, ast.expr_context, ast.operator)


@functools.cache
def _prenorm_class(base):
    """A pre-norm decoder layer whose `h = a + b` feeds one `self.<norm>(h)` and whose output is `h + E`.

    Both additions run with the RMS norm that follows them, and the output's normalization under
    the next layer's input norm is handed to that layer, which takes it only for the exact array
    it is called with.
    """
    call = getattr(base, "__call__", None)
    if not isinstance(call, FunctionType) or call.__closure__ or hasattr(call, "__wrapped__"):
        return None
    try:
        tree = copy.deepcopy(_function_ast(call))
    except (OSError, TypeError, SyntaxError, AttributeError):
        return None
    params = [*tree.args.posonlyargs, *tree.args.args]
    names = [node for node in ast.walk(tree) if isinstance(node, ast.Name)]
    if len(params) < 2 or any(node.id == "_unsloth_normed" for node in names):
        return None
    stored = {node.id for node in names if isinstance(node.ctx, ast.Store)}
    body = tree.body
    tail = body[-1] if isinstance(body[-1], ast.Return) else None
    if len(body) >= 2 and isinstance(body[-2], ast.Assign) and tail is not None and isinstance(tail.value, ast.Name):
        tail = body[-2] if [_source_expression(t) for t in body[-2].targets] == [tail.value.id] else None
    total = tail.value if tail is not None else None
    if not (isinstance(total, ast.BinOp) and isinstance(total.op, ast.Add) and isinstance(total.left, ast.Name)):
        return None
    h = total.left.id
    adds = [i for i, s in enumerate(body) if isinstance(s, ast.Assign) and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name) and s.targets[0].id == h]
    if len(adds) != 1:
        return None
    first = body[adds[0]]
    if not (isinstance(first.value, ast.BinOp) and isinstance(first.value.op, ast.Add)
            and isinstance(first.value.left, ast.Name) and isinstance(first.value.right, ast.Name)):
        return None
    # The norm moves up to the addition, so it must be the first call of the plain statement right
    # after it (nothing there binds a name): every other call there encloses it.
    norms = [node for s in body[adds[0] + 1:] for node in ast.walk(s) if _is_self_call(node, h)]
    nodes = list(ast.walk(body[adds[0] + 1]))
    calls = [node for node in nodes if isinstance(node, ast.Call)]
    if (len(norms) != 1 or not any(call is norms[0] for call in calls)
            or not all(isinstance(node, _PLAIN_NODES) for node in nodes)
            or not all(any(inner is norms[0] for inner in ast.walk(call)) for call in calls)):
        return None
    post = norms[0].func.attr

    incoming = set()

    class Rewrite(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if node is norms[0]:
                return ast.Name(id = "_unsloth_normed", ctx = ast.Load())
            if params[1].arg not in stored and _is_self_call(node, params[1].arg):
                incoming.add(node.func.attr)
                return _self_method("_unsloth_take_norm", node.func, node.args[0])
            return node

    tree = Rewrite().visit(tree)
    if len(incoming) > 1:
        return None
    first.targets = [ast.Tuple(elts = [ast.Name(id = h, ctx = ast.Store()),
                                       ast.Name(id = "_unsloth_normed", ctx = ast.Store())], ctx = ast.Store())]
    first.value = _self_method("_unsloth_add_norm", ast.Attribute(
        value = ast.Name(id = "self", ctx = ast.Load()), attr = post, ctx = ast.Load()),
        first.value.left, first.value.right)
    tail.value = _self_method("_unsloth_add_handoff", total.left, total.right)
    fused = _function_from_ast(call, tree)
    bindings = _resolved_bindings(_RESIDUAL_NORM_CONTRACT)
    if bindings is None:
        return None

    def invoke(self, *args, **kwargs):
        if self.training or base.__call__ is not call:
            return base.__call__(self, *args, **kwargs)
        return fused(self, *args, **kwargs)

    def add_norm(self, norm, a, b):
        out = _add_rms_norm(norm, a, b) if _bindings_intact(bindings) else None
        if out is not None:
            return out
        out = a + b
        return out, norm(out)

    def add_handoff(self, h, m):
        slot = self.__dict__.get("_unsloth_handoff_out")
        name = getattr(type(slot.consumer), "_unsloth_handoff_norm", None) if slot is not None else None
        if name is not None and _bindings_intact(bindings):
            norm = getattr(slot.consumer, name, None)
            out = _add_rms_norm(norm, h, m)
            if out is not None:
                slot.value = (*out, norm.weight, norm.eps)
                return out[0]
        return h + m

    def take_norm(self, norm, x):
        slot = self.__dict__.get("_unsloth_handoff_in")
        value = None
        if slot is not None:
            value, slot.value = slot.value, None
        if value is not None:
            residual, normed, weight, eps = value
            if (residual is x and type(norm) is nn.RMSNorm and norm.weight is weight and norm.eps == eps
                    and _bindings_intact(bindings)):
                return normed
        return norm(x)

    incoming = next(iter(incoming), None)
    return type(f"_ResidualNorm{base.__name__}", (base,), {
        "__call__": invoke, "_unsloth_add_norm": add_norm, "_unsloth_add_handoff": add_handoff,
        "_unsloth_take_norm": take_norm, "_unsloth_handoff_norm": incoming,
        "_unsloth_residual_norm_base": base,
        "_unsloth_residual_norm_names": (post,) if incoming is None else (post, incoming),
    })


_RESIDUAL_NORM_LOCK = RLock()

@contextmanager
def fused_residual_norm(model, *, _modules = None):
    """Fuse eligible single-row RMS normalization and residual additions during inference."""
    patched = []
    try:
        with _RESIDUAL_NORM_LOCK:
            if not getattr(model, "_unsloth_mlx_distributed_parallel_mode", None) and _residual_norm_kernel() is not None:
                for _, module in _fusion_modules(model, _modules):
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


@contextmanager
def fused_residual_norm_handoff(model, *, _modules = None):
    """Hand each pre-norm decoder layer's output, normalized, to the next layer during inference.

    A layer whose `h = a + b` feeds one RMS norm and whose output is `h + E` adds both residuals
    with the RMS norm that follows them in one launch each, and normalizes its output under the
    next layer's input norm, which that layer takes only for the exact array it is called with.
    Layers `fused_residual_norm` covers keep that scope's path; training and distributed models
    stay native. Instance classes are restored when the context exits.
    """
    patched, fresh = [], set()
    try:
        with _RESIDUAL_NORM_LOCK:
            if not getattr(model, "_unsloth_mlx_distributed_parallel_mode", None) and mx.metal.is_available():
                for _, module in _fusion_modules(model, _modules):
                    base = type(module)
                    # Type first, as in fused_residual_norm: named_modules() may yield plain stand-ins.
                    if not isinstance(module, dict) or module.training:
                        continue
                    # model.generate enters this without generation_mode's lock, so scopes can
                    # overlap: count owners, as fused_decode_conv_silu does, and let the last restore.
                    if module.__dict__.get("_unsloth_handoff_scopes", 0):
                        module._unsloth_handoff_scopes += 1
                        patched.append(module)
                        continue
                    if hasattr(base, "_unsloth_residual_norm_base") or _residual_norm_class(base) is not None:
                        continue
                    fused = _prenorm_class(base)
                    if (fused is not None and all(type(getattr(module, name, None)) is nn.RMSNorm
                                                 for name in fused._unsloth_residual_norm_names)):
                        patched.append(module)
                        module.__class__ = fused
                        # The last owner to exit restores, whichever scope patched the module.
                        module._unsloth_handoff_native = base
                        module._unsloth_handoff_scopes = 1
                        fresh.add(id(module))
                for module in (module for _, module in _fusion_modules(model, _modules)) if fresh else ():
                    layers = module.get("layers") if isinstance(module, dict) else None
                    if not isinstance(layers, list):
                        continue
                    for producer, consumer in zip(layers, layers[1:]):
                        if (id(producer) in fresh and id(consumer) in fresh
                                and getattr(type(consumer), "_unsloth_handoff_norm", None) is not None):
                            producer.__dict__["_unsloth_handoff_out"] = consumer.__dict__["_unsloth_handoff_in"] = _Handoff(consumer)
        yield model
    finally:
        with _RESIDUAL_NORM_LOCK:
            for module in reversed(patched):
                scopes = module.__dict__.get("_unsloth_handoff_scopes", 0)
                if scopes > 1:
                    module._unsloth_handoff_scopes = scopes - 1
                    continue
                base = module.__dict__.get("_unsloth_handoff_native")
                for name in ("_unsloth_handoff_out", "_unsloth_handoff_in", "_unsloth_handoff_scopes",
                             "_unsloth_handoff_native"):
                    module.__dict__.pop(name, None)
                if base is not None and getattr(type(module), "_unsloth_residual_norm_base", None) is base:
                    module.__class__ = base

_QWEN_ROUTING, _GEMMA_ROUTING = 0, 1  # kernel MODE

# The kernel emulates the chain's rounding: softmax partials and reciprocal, stable tie order,
# sequential bf16 sum over <= 8 values. `_moe_router_verified` checks that at first use.
# unary_ops.h's Sigmoid, as mx.sigmoid evaluates the shared-expert gate.
_MOE_ROUTER_SIGMOID = """
    template <typename T>
    METAL_FUNC T unsloth_sigmoid(T x) {
      auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
      return (x < 0) ? y : 1 - y;
    }
"""


@functools.cache
def _moe_router_kernel(shared = False):
    """The routing kernel; `shared` also writes the sigmoid of each row's shared-expert gate logit."""
    try:
        if not mx.metal.is_available():
            return None
        return mx.fast.metal_kernel(
            name = "unsloth_moe_router_shared" if shared else "unsloth_moe_router",
            input_names = ["logits", "scale"] + (["shared_logit"] if shared else []),
            output_names = ["inds", "weights"] + (["shared"] if shared else []),
            compile_options = {"math_mode": "safe"},
            header = _MOE_ROUTER_SIGMOID if shared else "",
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
                SHARED_SCALE
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
            """.replace("SHARED_SCALE", "shared[row] = unsloth_sigmoid(shared_logit[row]);" if shared else ""),
        )
    except (AttributeError, TypeError):
        return None


_MOE_ROUTER_DTYPES = (mx.bfloat16, mx.float16, mx.float32)


def _moe_router_shape_ok(experts, top_k, mode):
    # Qwen sums softmax partials of at most two simdgroups in the native order;
    # the 32-bit `taken` mask caps both modes at 1024.
    return experts % 32 == 0 and experts <= (256 if mode == _QWEN_ROUTING else 1024) and 1 <= top_k <= 8


def _run_moe_router(logits, scale, top_k, mode, normalize, out_dtype, shared_logit = None):
    experts = logits.shape[-1]
    rows = logits.size // experts
    shared = shared_logit is not None
    outputs = _moe_router_kernel(shared)(
        inputs = [logits.reshape(rows, experts), scale] + ([shared_logit.reshape(rows)] if shared else []),
        template = [("T", logits.dtype), ("OUT_T", out_dtype), ("E", experts), ("K", top_k),
                    ("MODE", mode), ("NORM", int(normalize))],
        grid = ((rows + 7) // 8 * 256, 1, 1), threadgroup = (256, 1, 1),
        output_shapes = [(rows, top_k), (rows, top_k)] + ([(rows,)] if shared else []),
        output_dtypes = [mx.uint32, out_dtype] + ([logits.dtype] if shared else []),
    )
    lead = logits.shape[:-1]
    routed = (outputs[0].reshape(*lead, top_k), outputs[1].reshape(*lead, top_k))
    return routed + (outputs[2].reshape(shared_logit.shape),) if shared else routed


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


def _fused_moe_router(logits, scale, top_k, mode, normalize, run = True):
    """The fused routing, or None when this call has to take the native chain; `run=False` only checks."""
    # A 1-D input normalizes with a sum that rounds differently.
    if (logits.dtype not in _MOE_ROUTER_DTYPES or logits.size == 0 or logits.ndim < 2
            or not _moe_router_shape_ok(logits.shape[-1], top_k, mode)
            or not _moe_router_verified(logits.dtype, scale.dtype, logits.shape[-1], top_k,
                                        mode, normalize)):
        return None
    if not run:
        return ()
    return _run_moe_router(logits, scale, top_k, mode, normalize,
                           _moe_router_out_dtype(logits, scale, mode))


@functools.cache
def _moe_router_shared_verified(dtype, experts, top_k, normalize):
    for rows in (1, 9):
        logits = (mx.random.normal((rows, experts), key = mx.random.key(rows)) * 3).astype(dtype)
        gate = (mx.random.normal((rows, 1), key = mx.random.key(rows + 1)) * 8).astype(dtype)
        native = _native_moe_router(logits, _MOE_ROUTER_NO_SCALE, top_k, _QWEN_ROUTING, normalize)
        fused = _run_moe_router(logits, _MOE_ROUTER_NO_SCALE, top_k, _QWEN_ROUTING, normalize, dtype, gate)
        if not all(a.dtype == b.dtype and mx.array_equal(a, b) for a, b in zip(native + (mx.sigmoid(gate),), fused)):
            logger.warning("the fused MoE router does not reproduce this MLX build's sigmoid for %s; "
                           "the shared-expert gate stays a separate sigmoid", dtype)
            return False
    return True


def _fused_moe_router_shared(logits, gate, top_k, normalize):
    """The Qwen routing and `mx.sigmoid(gate)` from one launch, or None for the separate sigmoid."""
    if (gate.dtype != logits.dtype or gate.shape != (*logits.shape[:-1], 1)
            or _fused_moe_router(logits, _MOE_ROUTER_NO_SCALE, top_k, _QWEN_ROUTING, normalize, run = False) is None
            or not _moe_router_shared_verified(logits.dtype, logits.shape[-1], top_k, normalize)):
        return None
    return _run_moe_router(logits, _MOE_ROUTER_NO_SCALE, top_k, _QWEN_ROUTING, normalize, logits.dtype, gate)


# mlx_vlm 0.7.1+ `_shared_expert_scale`: `mx.sigmoid(self.shared_expert_gate(x))`. qwen4_exp overrides it.
_STOCK_SHARED_EXPERT_SCALE = "6399d93293b7fedd"


@functools.cache
def _stock_shared_expert_scale(function):
    if not isinstance(function, FunctionType) or hasattr(function, "__wrapped__") or function.__closure__:
        return False
    try:
        return (_ast_fingerprint(function) == _STOCK_SHARED_EXPERT_SCALE
                and function.__globals__.get("mx") is mx)
    except (OSError, TypeError, SyntaxError):
        return False


def _drifted_call(self, native):
    """The base class's routing body once it or its `mx` alias drifted from the pinned one, else None."""
    current = type(self)._unsloth_router_native.__call__
    if current is native and native.__globals__.get("mx") is mx:
        return None
    return current


# Bitwise copies of MLX's unsorted one-row gather (qmv_fast / fp_qmv_fast), compiled SwiGLU and the block's
# combine. Headers are trimmed from the installed MLX since `metal_kernel` hashes its whole source per launch.
_INCLUDE = re.compile(r'#include "([^"]+)"')
# (header, first declaration kept, first declaration dropped); None is the header's start or end.
_SIGMOID = ("unary_ops.h", "Sigmoid", "Sign")
_FAMILIES = {
    True: (("quantized.h", None, "qdot"), _SIGMOID),
    False: (("fp_quantized.h", None, "qdot"), _SIGMOID),
}

_HALF = (mx.bfloat16, mx.float16)
_INDS = (mx.uint32, mx.int32)
# Stock sums fewer than 32 slots with col_reduce_small, whose order the kernel reproduces.
_MAX_SLOTS = 31
_MAX_SIMDS = 16
_GATE_UP_ROWS = 2
_DOWN_ROWS = 4
_INPUTS = {
    "gate_up": ("wg", "sg", "bg", "wu", "su", "bu", "x", "inds", "k", "n", "top_k"),
    "down": ("w", "ws", "wb", "x", "inds", "scores", "shared", "s", "k", "n", "top_k"),
}

_QDOT = r"""
#define UNROLL _Pragma("clang loop unroll(full)")

#ifdef UNSLOTH_AFFINE
template <typename T, int bits>
struct Lane {
  typedef T scale_t;
  static constexpr constant int packs = bits == 2 ? 1 : 2;
  static constexpr constant int pack_factor = get_pack_factor<bits, 32>();
  static constexpr constant int bytes_per_pack = get_bytes_per_pack<bits, 32>();
};

template <typename U, int values_per_thread, int bits>
struct Terms;

template <typename U, int values_per_thread>
struct Terms<U, values_per_thread, 2> {
  U t[values_per_thread];
  void load(const device uint8_t* w) {
    UNROLL for (int i = 0; i < values_per_thread / 4; i++) {
      t[4 * i] = w[i] & 0x03;
      t[4 * i + 1] = w[i] & 0x0c;
      t[4 * i + 2] = w[i] & 0x30;
      t[4 * i + 3] = w[i] & 0xc0;
    }
  }
  U dot(const thread U* x) {
    U accum = 0;
    UNROLL for (int i = 0; i < values_per_thread / 4; i++) {
      accum += (x[4 * i] * t[4 * i] + x[4 * i + 1] * t[4 * i + 1] + x[4 * i + 2] * t[4 * i + 2] + x[4 * i + 3] * t[4 * i + 3]);
    }
    return accum;
  }
};

template <typename U, int values_per_thread>
struct Terms<U, values_per_thread, 3> {
  U t[values_per_thread / 8 * 10];
  void load(const device uint8_t* w) {
    UNROLL for (int i = 0; i < values_per_thread / 8; i++) {
      t[10 * i] = w[3 * i] & 0x07;
      t[10 * i + 1] = w[3 * i] & 0x38;
      t[10 * i + 2] = w[3 * i] & 0xc0;
      t[10 * i + 3] = w[3 * i + 1] & 0x01;
      t[10 * i + 4] = w[3 * i + 1] & 0x0e;
      t[10 * i + 5] = w[3 * i + 1] & 0x70;
      t[10 * i + 6] = w[3 * i + 1] & 0x80;
      t[10 * i + 7] = w[3 * i + 2] & 0x03;
      t[10 * i + 8] = w[3 * i + 2] & 0x1c;
      t[10 * i + 9] = w[3 * i + 2] & 0xe0;
    }
  }
  U dot(const thread U* x) {
    U accum = 0;
    UNROLL for (int i = 0; i < values_per_thread / 8; i++) {
      accum += t[10 * i] * x[8 * i];
      accum += t[10 * i + 1] * x[8 * i + 1];
      accum += t[10 * i + 2] * x[8 * i + 2];
      accum += t[10 * i + 3] * (x[8 * i + 2] * 256.0f);
      accum += t[10 * i + 4] * x[8 * i + 3];
      accum += t[10 * i + 5] * x[8 * i + 4];
      accum += t[10 * i + 6] * x[8 * i + 5];
      accum += t[10 * i + 7] * (x[8 * i + 5] * 256.0f);
      accum += t[10 * i + 8] * x[8 * i + 6];
      accum += t[10 * i + 9] * x[8 * i + 7];
    }
    return accum;
  }
};

template <typename U, int values_per_thread>
struct Terms<U, values_per_thread, 4> {
  U t[values_per_thread];
  void load(const device uint8_t* w) {
    const device uint16_t* ws = (const device uint16_t*)w;
    UNROLL for (int i = 0; i < values_per_thread / 4; i++) {
      t[4 * i] = ws[i] & 0x000f;
      t[4 * i + 1] = ws[i] & 0x00f0;
      t[4 * i + 2] = ws[i] & 0x0f00;
      t[4 * i + 3] = ws[i] & 0xf000;
    }
  }
  U dot(const thread U* x) {
    U accum = 0;
    UNROLL for (int i = 0; i < values_per_thread / 4; i++) {
      accum += (x[4 * i] * t[4 * i] + x[4 * i + 1] * t[4 * i + 1] + x[4 * i + 2] * t[4 * i + 2] + x[4 * i + 3] * t[4 * i + 3]);
    }
    return accum;
  }
};

template <typename U, int values_per_thread>
struct Terms<U, values_per_thread, 5> {
  U t[values_per_thread / 8 * 12];
  void load(const device uint8_t* w) {
    UNROLL for (int i = 0; i < values_per_thread / 8; i++) {
      t[12 * i] = w[5 * i] & 0x1f;
      t[12 * i + 1] = w[5 * i] & 0xe0;
      t[12 * i + 2] = w[5 * i + 1] & 0x3;
      t[12 * i + 3] = w[5 * i + 1] & 0x7c;
      t[12 * i + 4] = w[5 * i + 1] & 0x80;
      t[12 * i + 5] = w[5 * i + 2] & 0xf;
      t[12 * i + 6] = w[5 * i + 2] & 0xf0;
      t[12 * i + 7] = w[5 * i + 3] & 0x1;
      t[12 * i + 8] = w[5 * i + 3] & 0x3e;
      t[12 * i + 9] = w[5 * i + 3] & 0xc0;
      t[12 * i + 10] = w[5 * i + 4] & 0x7;
      t[12 * i + 11] = w[5 * i + 4] & 0xf8;
    }
  }
  U dot(const thread U* x) {
    U accum = 0;
    UNROLL for (int i = 0; i < values_per_thread / 8; i++) {
      accum += t[12 * i] * x[8 * i];
      accum += t[12 * i + 1] * x[8 * i + 1];
      accum += t[12 * i + 2] * (x[8 * i + 1] * 256.0f);
      accum += t[12 * i + 3] * x[8 * i + 2];
      accum += t[12 * i + 4] * x[8 * i + 3];
      accum += t[12 * i + 5] * (x[8 * i + 3] * 256.0f);
      accum += t[12 * i + 6] * x[8 * i + 4];
      accum += t[12 * i + 7] * (x[8 * i + 4] * 256.0f);
      accum += t[12 * i + 8] * x[8 * i + 5];
      accum += t[12 * i + 9] * x[8 * i + 6];
      accum += t[12 * i + 10] * (x[8 * i + 6] * 256.0f);
      accum += t[12 * i + 11] * x[8 * i + 7];
    }
    return accum;
  }
};

template <typename U, int values_per_thread>
struct Terms<U, values_per_thread, 6> {
  U t[values_per_thread / 4 * 6];
  void load(const device uint8_t* w) {
    UNROLL for (int i = 0; i < values_per_thread / 4; i++) {
      t[6 * i] = w[3 * i] & 0x3f;
      t[6 * i + 1] = w[3 * i] & 0xc0;
      t[6 * i + 2] = w[3 * i + 1] & 0x0f;
      t[6 * i + 3] = w[3 * i + 1] & 0xf0;
      t[6 * i + 4] = w[3 * i + 2] & 0x03;
      t[6 * i + 5] = w[3 * i + 2] & 0xfc;
    }
  }
  U dot(const thread U* x) {
    U accum = 0;
    UNROLL for (int i = 0; i < values_per_thread / 4; i++) {
      accum += t[6 * i] * x[4 * i];
      accum += t[6 * i + 1] * x[4 * i + 1];
      accum += t[6 * i + 2] * (x[4 * i + 1] * 256.0f);
      accum += t[6 * i + 3] * x[4 * i + 2];
      accum += t[6 * i + 4] * (x[4 * i + 2] * 256.0f);
      accum += t[6 * i + 5] * x[4 * i + 3];
    }
    return accum;
  }
};

template <typename U, int values_per_thread>
struct Terms<U, values_per_thread, 8> {
  U t[values_per_thread];
  void load(const device uint8_t* w) {
    UNROLL for (int i = 0; i < values_per_thread; i++) {
      t[i] = w[i];
    }
  }
  U dot(const thread U* x) {
    U accum = 0;
    UNROLL for (int i = 0; i < values_per_thread; i++) {
      accum += x[i] * t[i];
    }
    return accum;
  }
};
#else
template <typename T, int bits>
struct Lane {
  typedef uint8_t scale_t;
  static constexpr constant int packs = 2;
  static constexpr constant int pack_factor = get_pack_factor<32, bits>();
  static constexpr constant int bytes_per_pack = get_bytes_per_pack<32>();
};

template <typename U, int values_per_thread, int bits>
struct Terms {
  U t[values_per_thread];
  void load(const device uint8_t* w) {
    if constexpr (bits == 4) {
      const device uint16_t* ws = (const device uint16_t*)w;
      UNROLL for (int i = 0; i < values_per_thread / 4; i++) {
        t[4 * i] = Dequantize<4>{}(ws[i]);
        t[4 * i + 1] = Dequantize<4>{}(ws[i] >> 4);
        t[4 * i + 2] = Dequantize<4>{}(ws[i] >> 8);
        t[4 * i + 3] = Dequantize<4>{}(ws[i] >> 12);
      }
    } else {
      UNROLL for (int i = 0; i < values_per_thread; i++) {
        t[i] = Dequantize<8>{}(w[i]);
      }
    }
  }
  U dot(const thread U* x) {
    U accum = 0;
    if constexpr (bits == 4) {
      UNROLL for (int i = 0; i < values_per_thread / 4; i++) {
        accum += (x[4 * i] * t[4 * i] + x[4 * i + 1] * t[4 * i + 1] + x[4 * i + 2] * t[4 * i + 2] + x[4 * i + 3] * t[4 * i + 3]);
      }
    } else {
      UNROLL for (int i = 0; i < values_per_thread; i++) {
        accum += x[i] * t[i];
      }
    }
    return accum;
  }
};
#endif
"""

_SLOTS = r"""
// Stock's `(y * scores[..., None]).sum(axis=-2)` over k < 32 products `stride` apart: MLX's col_reduce_small, whose
// min(8, k) row groups each sum from T(0), then fold into the first; a single product is copied as it is.
template <typename T, typename Products>
T slot_sum(Products p, int stride, int k) {
  if (k == 1) {
    return p[0];
  }
  const int groups = min(k, 8);
  T acc;
  for (int j = 0; j < groups; j++) {
    T total = T(0);
    for (int i = j; i < k; i += groups) {
      total = p[i * stride] + total;
    }
    acc = j == 0 ? total : total + acc;
  }
  return acc;
}

template <typename T>
T shared_add(T acc, T shared, T s) {
  T scaled = shared * s;
  return acc + scaled;
}
"""

_ROUTED = r"""
struct ExpertParams {
  int K;
  int N;
  int top_k;
  int64_t w[2][3]; // per-expert byte strides of (weight, scales, biases) per projection
};

template <typename T, int group_size, int bits, int out>
struct Dots {
  typedef float U;
  typedef Lane<T, bits> lane;
  static constexpr constant int values_per_thread = lane::pack_factor * lane::packs;
  static constexpr constant int block_size = values_per_thread * SIMD_SIZE;
  static constexpr constant int scale_step = group_size / values_per_thread;
  struct Proj {
    const device uint8_t* w;
    const device typename lane::scale_t* s;
    const device T* b;
  };

  static Proj proj(const device uint8_t* w, const device uint8_t* s, const device uint8_t* b, int K, int n0, int lane_id) {
    const int K_w = K * lane::bytes_per_pack / lane::pack_factor;
    const int K_g = K / group_size;
    return Proj{
        w + n0 * K_w + lane_id * lane::packs * lane::bytes_per_pack,
        (const device typename lane::scale_t*)s + n0 * K_g + lane_id / scale_step,
        (const device T*)b + n0 * K_g + lane_id / scale_step};
  }

  static void rows(Proj p, int K, thread Terms<U, values_per_thread, bits>& terms, const thread U* xt, U sum, thread U* result) {
    const int K_w = K * lane::bytes_per_pack / lane::pack_factor;
    const int K_g = K / group_size;
    UNROLL for (int o = 0; o < out; o++) {
      terms.load(p.w + o * K_w);
#ifdef UNSLOTH_AFFINE
      U s = p.s[o * K_g];
      U b = p.b[o * K_g];
      result[o] += s * terms.dot(xt) + sum * b;
#else
      U s = dequantize_scale<U, group_size>(p.s[o * K_g]);
      result[o] += s * terms.dot(xt);
#endif
    }
  }

  template <int P>
  static void run(thread Proj* p, const device T* x, int K, int lane_id, thread U* result) {
    thread U xt[values_per_thread];
    U sum = 0;
    Terms<U, values_per_thread, bits> terms;
    x += lane_id * values_per_thread;
    for (int k = 0; k < K; k += block_size) {
#ifdef UNSLOTH_AFFINE
      sum = load_vector<T, U, values_per_thread, bits>(x, xt);
#else
      load_vector<T, U, values_per_thread>(x, xt);
#endif
      UNROLL for (int j = 0; j < P; j++) {
        rows(p[j], K, terms, xt, sum, result + j * out);
      }
      UNROLL for (int j = 0; j < P; j++) {
        p[j].w += block_size * lane::bytes_per_pack / lane::pack_factor;
        p[j].s += block_size / group_size;
        p[j].b += block_size / group_size;
      }
      x += block_size;
    }
    UNROLL for (int i = 0; i < P * out; i++) {
      result[i] = simd_sum(result[i]);
    }
  }
};

template <typename T, int group_size, int bits, int out, typename Inds>
METAL_FUNC void gate_up_rows(
    const device uint8_t* wg,
    const device uint8_t* sg,
    const device uint8_t* bg,
    const device uint8_t* wu,
    const device uint8_t* su,
    const device uint8_t* bu,
    const device T* x,
    Inds inds,
    device T* y,
    ExpertParams p,
    uint3 tid,
    uint3 lid) {
  typedef Dots<T, group_size, bits, out> dots;
  const int lane = lid.x;
  const int n0 = tid.x * (2 * out) + lid.y * out;
  const int64_t e = inds[tid.y];
  x += size_t(tid.y / p.top_k) * p.K;
  y += size_t(tid.y) * p.N + n0;
  thread typename dots::U result[2 * out] = {};
  thread typename dots::Proj proj[2] = {
      dots::proj(wg + e * p.w[0][0], sg + e * p.w[0][1], bg + e * p.w[0][2], p.K, n0, lane),
      dots::proj(wu + e * p.w[1][0], su + e * p.w[1][1], bu + e * p.w[1][2], p.K, n0, lane)};
  dots::template run<2>(proj, x, p.K, lane, result);
  // Lane o finishes activation o.
  UNROLL for (int o = 0; o < out; o++) {
    if (lane == o) {
      T gate = static_cast<T>(result[o]);
      T up = static_cast<T>(result[out + o]);
      T s = Sigmoid{}(gate);
      T a = gate * s;
      y[o] = a * up;
    }
  }
}

template <typename T, int group_size, int bits, int out, typename Inds, typename Scores, typename Shared, typename S>
METAL_FUNC void down_rows(
    const device uint8_t* w,
    const device uint8_t* ws,
    const device uint8_t* wb,
    const device T* x,
    Inds inds,
    Scores scores,
    Shared shared,
    S s,
    device T* y,
    ExpertParams p,
    threadgroup T* products,
    uint3 tid,
    uint3 lid,
    uint3 lsize) {
  typedef Dots<T, group_size, bits, out> dots;
  const int lane = lid.x;
  const int n0 = tid.x * out;
  const size_t route = size_t(tid.y) * p.top_k;
  for (int slot = lid.y; slot < p.top_k; slot += lsize.y) {
    const int64_t e = inds[route + slot];
    thread typename dots::U result[out] = {};
    thread typename dots::Proj proj = dots::proj(w + e * p.w[0][0], ws + e * p.w[0][1], wb + e * p.w[0][2], p.K, n0, lane);
    dots::template run<1>(&proj, x + (route + slot) * p.K, p.K, lane, result);
    const T score = scores[route + slot];
    UNROLL for (int o = 0; o < out; o++) {
      if (lane == o) {
        products[slot * out + o] = static_cast<T>(result[o]) * score;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid.y == 0 && lane < out) {
    const size_t i = size_t(tid.y) * p.N + n0 + lane;
    y[i] = shared_add(slot_sum<T>(products + lane, out, p.top_k), shared[tid.y], s[i]);
  }
}
"""

_BODIES = {
    "gate_up": r"""
  const ExpertParams p{k, n, top_k, {
      {wg_strides[0] * int64_t(sizeof(*wg)), sg_strides[0] * int64_t(sizeof(*sg)), bg_strides[0] * int64_t(sizeof(*bg))},
      {wu_strides[0] * int64_t(sizeof(*wu)), su_strides[0] * int64_t(sizeof(*su)), bu_strides[0] * int64_t(sizeof(*bu))}}};
  gate_up_rows<T, group_size, bits, out>(
      (const device uint8_t*)wg, (const device uint8_t*)sg, (const device uint8_t*)bg,
      (const device uint8_t*)wu, (const device uint8_t*)su, (const device uint8_t*)bu,
      x, inds, y, p, threadgroup_position_in_grid, thread_position_in_threadgroup);
""",
    "down": r"""
  threadgroup T products[31 * out];
  const ExpertParams p{k, n, top_k, {{w_strides[0] * int64_t(sizeof(*w)), ws_strides[0] * int64_t(sizeof(*ws)), wb_strides[0] * int64_t(sizeof(*wb))}, {}}};
  down_rows<T, group_size, bits, out>(
      (const device uint8_t*)w, (const device uint8_t*)ws, (const device uint8_t*)wb, x, inds, scores, shared, s, y, p,
      products, threadgroup_position_in_grid, thread_position_in_threadgroup, threads_per_threadgroup);
""",
}


def _boundary(text, name):
    # Start of the top-level declaration of `name`: the line after the previous closing brace.
    match = re.search(rf"^\w.*\b{name}\s*[({{]", text, re.M)
    if match is None:
        raise RuntimeError(f"the installed MLX headers no longer declare {name}")
    at = match.start()
    return text.index("\n", text.rfind("\n}", 0, at) + 1) + 1


def _inline(path, seen, first = None, stop = None):
    if path in seen:
        return ""
    seen.add(path)
    text = (Path(mx.__file__).parent / "include" / path).read_text()
    text = text[_boundary(text, first) if first else 0 : _boundary(text, stop) if stop else len(text)]
    return _INCLUDE.sub(lambda m: _inline(m.group(1), seen), text).replace("#pragma once", "")


def _mlx_headers(affine):
    # utils.h and what it includes are already in metal_kernel's preamble.
    seen = set()
    _inline("mlx/backend/metal/kernels/utils.h", seen)
    return "".join(
        _inline(f"mlx/backend/metal/kernels/{name}", seen, first, stop)
        for name, first, stop in _FAMILIES[affine]
    )


@functools.cache
def _kernel(name, affine):
    try:
        if not mx.metal.is_available():
            return None
        header = _mlx_headers(affine) + ("#define UNSLOTH_AFFINE 1\n" if affine else "")
        return mx.fast.metal_kernel(
            name = f"unsloth_moe_{name}_{'affine' if affine else 'fp'}",
            input_names = list(_INPUTS[name]),
            output_names = ["y"],
            source = _BODIES[name],
            header = header + _QDOT + _SLOTS + _ROUTED,
            ensure_row_contiguous = False,
        )
    except (AttributeError, OSError, RuntimeError, TypeError):
        return None


@functools.lru_cache(maxsize = 64)
def _scalar(value):
    return mx.array(value, mx.int32)


def _values_per_thread(mode, bits):
    if mode != "affine":
        return 2 * 32 // bits
    pack_factor = 8 if bits in (3, 5) else 4 if bits == 6 else 32 // bits
    return pack_factor * (1 if bits == 2 else 2)


def _trailing(a, dtype, rows, cols):
    return a.dtype == dtype and a.ndim >= 2 and a.shape[-2:] == (rows, cols)


def _one_row_fast(mode, group_size, bits, dtype, k, n, w, scales, biases):
    """Whether MLX's one-row gather kernel for (K, N) is qmv_fast / fp_qmv_fast for these weights."""
    affine = mode == "affine"
    known = (
        (affine and 2 <= bits <= 8 and bits != 7 and group_size in (32, 64, 128))
        or (mode == "mxfp4" and bits == 4 and group_size == 32)
        or (mode == "mxfp8" and bits == 8 and group_size == 32)
        or (mode == "nvfp4" and bits == 4 and group_size == 16)
    )
    if not known or scales is None or affine != (biases is not None):
        return False
    per_thread = _values_per_thread(mode, bits)
    if n % 8 != 0 or n < 16 or k % (per_thread * 32) != 0 or group_size < per_thread:
        return False
    return (
        _trailing(w, mx.uint32, n, k * bits // 32)
        and _trailing(scales, dtype if affine else mx.uint8, n, k // group_size)
        and (not affine or _trailing(biases, dtype, n, k // group_size))
    )


def _key(a):
    return None if a is None else (a.shape, a.dtype)


class _Spec:
    __slots__ = ("shape", "dtype", "ndim")

    def __init__(self, key):
        self.shape, self.dtype = key
        self.ndim = len(self.shape)


def _specs(keys):
    return [None if k is None else _Spec(k) for k in keys]


def _projection(t, mode, group_size, bits, dtype, k):
    """N of one projection's (weight, scales[, biases]) with one fast-kernel weight per expert, else -1."""
    if len(t) != (3 if mode == "affine" else 2) or any(a.ndim != 3 or a.shape[0] != t[0].shape[0] for a in t):
        return -1
    n = t[0].shape[-2]
    scales, biases = (list(t[1:]) + [None])[:2]
    return n if _one_row_fast(mode, group_size, bits, dtype, k, n, t[0], scales, biases) else -1


def _launch(name, affine, dtype, quant, out_rows, per, projections, experts, shape, grid, threadgroup, k, n, top_k, routes):
    call = _kernel(name, affine)
    if call is None:
        return None
    scalars = [_scalar(k), _scalar(n), _scalar(top_k)]
    args = dict(
        template = [("T", dtype), ("group_size", quant[0]), ("bits", quant[1]), ("out", out_rows)],
        grid = grid,
        threadgroup = threadgroup,
        output_shapes = [shape],
        output_dtypes = [dtype],
    )

    def run(*arrays):
        weights, activations = arrays[: projections * per], arrays[projections * per :]
        bound = []
        for j in range(projections):
            t = weights[j * per : (j + 1) * per]
            # The fp kernel never reads the biases stand-in; slicing to reachable experts bounds the command buffer size.
            bound += [t[min(i, per - 1)][: min(routes, experts)] for i in range(3)]
        return call(inputs = [*bound, *map(mx.contiguous, activations), *scalars], **args)[0]

    return mx.compile(run)


@functools.lru_cache(maxsize = 1024)
def _gate_up_plan(x, inds, gate, up, group_size, bits, mode):
    x, inds = _specs((x, inds))
    gate, up = _specs(gate), _specs(up)
    if (
        x.ndim == 0
        or x.dtype not in _HALF
        or inds.ndim != x.ndim
        or inds.dtype not in _INDS
        or inds.shape[-1] < 1
        or x.shape[:-1] != inds.shape[:-1]
    ):
        return None
    k = x.shape[-1]
    n = _projection(gate, mode, group_size, bits, x.dtype, k)
    if (n < 0 or n % (2 * _GATE_UP_ROWS) != 0
            or _projection(up, mode, group_size, bits, x.dtype, k) != n
            or gate[0].shape[0] != up[0].shape[0]):
        return None
    shape, routes = (*inds.shape, n), math.prod(inds.shape)
    if routes == 0:
        return shape, x.dtype
    return _launch(
        "gate_up", mode == "affine", x.dtype, (group_size, bits), _GATE_UP_ROWS, len(gate), 2,
        gate[0].shape[0], shape, grid = (32 * n // (2 * _GATE_UP_ROWS), 2 * routes, 1),
        threadgroup = (32, 2, 1), k = k, n = n, top_k = inds.shape[-1], routes = routes,
    )


def _combinable(x, scores, shared, s, n):
    t = x.dtype
    if t not in _HALF or any(a.dtype != t for a in (scores, shared, s)) or x.ndim < 2 or scores.ndim != x.ndim - 1:
        return False
    if shared.ndim != scores.ndim or s.ndim != scores.ndim:
        return False
    lead = scores.shape[:-1]
    return (
        1 <= scores.shape[-1] <= _MAX_SLOTS
        and x.shape[: scores.ndim] == scores.shape
        and shared.shape == (*lead, 1)
        and s.shape == (*lead, n)
    )


@functools.lru_cache(maxsize = 1024)
def _down_plan(x, inds, scores, shared, s, w, group_size, bits, mode):
    x, inds, scores, shared, s = _specs((x, inds, scores, shared, s))
    w = _specs(w)
    if x.ndim < 2 or inds.dtype not in _INDS or inds.shape != scores.shape:
        return None
    n = _projection(w, mode, group_size, bits, x.dtype, x.shape[-1])
    if n < 0 or n % _DOWN_ROWS != 0 or not _combinable(x, scores, shared, s, n):
        return None
    routes, top_k = math.prod(inds.shape), scores.shape[-1]
    if math.prod(s.shape) == 0:
        return s.shape, x.dtype
    simds = min(top_k, _MAX_SIMDS)
    return _launch(
        "down", mode == "affine", x.dtype, (group_size, bits), _DOWN_ROWS, len(w), 1, w[0].shape[0],
        s.shape, grid = (32 * n // _DOWN_ROWS, simds * (routes // top_k), 1),
        threadgroup = (32, simds, 1), k = x.shape[-1], n = n, top_k = top_k, routes = routes,
    )


def _run(launch, weights, activations):
    if launch is None:
        return None
    if not callable(launch):
        return mx.zeros(*launch)
    return launch(*weights, *activations)


def _routed_gate_up(x, inds, gate, up, group_size, bits, mode):
    """SwiGLU(gate, up) of experts `inds` (..., top_k) on x (..., K): (..., top_k, N), or None.

    `gate` and `up` are a projection's (weight, scales[, biases]) shaped (E, N, ...), read with their
    own expert stride, so row-slice views of a packed weight serve as they are.
    """
    launch = _gate_up_plan(_key(x), _key(inds), tuple(map(_key, gate)), tuple(map(_key, up)), group_size, bits, mode)
    return _run(launch, [*gate, *up], [x, inds])


def _routed_down(x, inds, scores, shared, s, w, group_size, bits, mode):
    """`(y * scores[..., None]).sum(-2) + shared * s` for y the down projections of x (..., top_k, K)
    by experts `inds`, or None."""
    launch = _down_plan(*map(_key, (x, inds, scores, shared, s)), tuple(map(_key, w)), group_size, bits, mode)
    return _run(launch, list(w), [x, inds, scores, shared, s])


_MOE_UNSORTED_ROUTES = 64
# The switch layer bodies the routed-expert kernels reimplement besides the gate/up contract's.
_MOE_ROUTED_FUNCTIONS = {"SwiGLU.__call__": "b6e81906a2a7d4c9", "swiglu": "bc9e3d24f39fe8b1"}
_MOE_ROUTED_VERDICTS = {}


def _projection_tensors(linear):
    return tuple(linear[name] for name in ("weight", "scales", "biases") if name in linear)


def _expert_rows_packed(tensor):
    # The kernels take only the expert stride from the array; gate/up pack row-slice views keep the rest packed.
    try:
        strides = memoryview(tensor).strides
    except (TypeError, ValueError, BufferError):
        return False
    return tensor.ndim == 3 and strides[1:] == (tensor.shape[2] * tensor.itemsize, tensor.itemsize)


def _switch_base(cls, specs):
    """The native switch class `cls` dispatches to unchanged: itself, or the one the gate/up pack wraps."""
    if cls in specs:
        return cls
    if cls in _MOE_GATE_UP_CLASSES.values() and cls.__bases__[0] in specs:
        return cls.__bases__[0]
    return None


class _RoutedExperts:
    """A sparse block's routed experts on the decode kernels, for unsorted routes outside the short-block band."""

    def __init__(self, mlp, base, projection_type, activation_type, decode_block, bindings):
        self.mlp, self.base, self.classes = mlp, base, {base}
        self.projection_type, self.activation_type = projection_type, activation_type
        self.decode_block, self.bindings = decode_block, bindings
        self.verdicts = {}
        self.layout = (None, False)

    def __call__(self, block, x, inds, scores, shared, s):
        mlp = block.switch_mlp
        if (mlp is not self.mlp or mlp.training or (x.ndim == 3 and 1 < x.shape[1] <= self.decode_block)
                or not _bindings_intact(self.bindings)):
            return None
        if type(mlp) not in self.classes:
            if _switch_base(type(mlp), {self.base: None}) is not self.base:
                return None
            self.classes.add(type(mlp))
        projections = (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
        if (type(mlp.activation) is not self.activation_type or "_combine" in mlp.__dict__
                or any(_native_type(p) is not self.projection_type or "bias" in p for p in projections)):
            return None
        tensors = tuple(map(_projection_tensors, projections))
        key = tuple(id(t) for group in tensors for t in group)
        if self.layout[0] != key:
            self.layout = (key, all(_expert_rows_packed(t) for group in tensors for t in group))
        if not self.layout[1]:
            return None
        verdict = self.verdicts.get(x.dtype)
        if verdict is None:
            verdict = self.verdicts[x.dtype] = self._verified(x.dtype, inds.shape[-1])
        if not verdict:
            return None
        return self._run(tensors, x, inds, scores, shared, s)

    def _run(self, tensors, x, inds, scores, shared, s):
        gate, up, down = self.mlp.gate_proj, self.mlp.up_proj, self.mlp.down_proj
        activations = _routed_gate_up(x, inds, tensors[0], tensors[1], gate.group_size, gate.bits, gate.mode)
        if activations is None:
            return None
        return _routed_down(activations, inds, scores, shared, s, tensors[2],
                                   down.group_size, down.bits, down.mode)

    def _verified(self, dtype, top_k):
        mlp = self.mlp
        signature = (dtype, top_k, self.base) + tuple(
            (p.group_size, p.bits, p.mode, p.weight.shape, p.weight.dtype)
            for p in (mlp.gate_proj, mlp.up_proj, mlp.down_proj))
        if signature not in _MOE_ROUTED_VERDICTS:
            _MOE_ROUTED_VERDICTS[signature] = self._probe(dtype, top_k)
            if not _MOE_ROUTED_VERDICTS[signature]:
                logger.warning("the routed-expert decode kernels do not reproduce this MLX build's "
                               "native experts for %s; the native experts stay in use", signature[3:])
        return _MOE_ROUTED_VERDICTS[signature]

    def _probe(self, dtype, top_k):
        mlp = self.mlp
        tensors = tuple(map(_projection_tensors, (mlp.gate_proj, mlp.up_proj, mlp.down_proj)))
        experts, hidden = mlp.gate_proj.weight.shape[0], mlp.down_proj.weight.shape[1]
        width = mlp.gate_proj.scales.shape[-1] * mlp.gate_proj.group_size
        for rows in (1, max(2, (_MOE_UNSORTED_ROUTES - 1) // top_k)):
            kx, ki, kw, kg, ks = mx.random.split(mx.random.key(rows), 5)
            x = mx.random.normal((rows, 1, width), key = kx).astype(dtype)
            inds = mx.random.randint(0, experts, (rows, 1, top_k), key = ki).astype(mx.uint32)
            scores = mx.softmax(mx.random.normal((rows, 1, top_k), key = kw), axis = -1).astype(dtype)
            shared = mx.sigmoid(mx.random.normal((rows, 1, 1), key = kg)).astype(dtype)
            s = mx.random.normal((rows, 1, hidden), key = ks).astype(dtype)
            fused = self._run(tensors, x, inds, scores, shared, s)
            y = self.base.__call__(mlp, x, inds)
            native = (y * scores[..., None]).sum(axis = -2) + shared * s
            if fused is None or fused.dtype != native.dtype or not mx.array_equal(fused, native, equal_nan = True):
                return False
        return True


def _moe_routed_experts(block):
    mlp = getattr(block, "switch_mlp", None)
    if not isinstance(mlp, dict) or mlp.training:
        return None
    specs = _moe_switch_specs()
    base = _switch_base(type(mlp), specs)
    if base is None:
        return None
    projection_type, _, decode_block, bindings = specs[base]
    if any(_native_type(getattr(mlp, name, None)) is not projection_type
           for name in ("gate_proj", "up_proj", "down_proj")):
        return None
    functions = dict(_MOE_ROUTED_FUNCTIONS)
    if "_combine" in vars(base):  # mlx-vlm's SwitchGLU returns through this static helper
        functions[f"{base.__name__}._combine"] = "fe6e29351fdc9b1e"
    routed_bindings = _resolved_bindings({base.__module__: functions, **_CONV_SILU_CONTRACT})
    native = sys.modules[base.__module__]
    if routed_bindings is None or type(getattr(mlp, "activation", None)) is not native.SwiGLU:
        return None
    return _RoutedExperts(mlp, base, projection_type, native.SwiGLU, decode_block, bindings + routed_bindings)


# MLX multiplies and sums in the array dtype, in route order; the kernel rounds at the same two places.
# Single precision is left native: its sum does not round like a sequential one.
_MOE_COMBINE_DTYPES = (mx.bfloat16, mx.float16)
_MOE_COMBINE_MAX_ELEMENTS = 1 << 32
_MOE_COMBINE_SOURCE = """
    uint token = thread_position_in_grid.y;
    uint d = thread_position_in_grid.x;
    T acc = T(0);
    for (uint k = 0; k < K; ++k) {
        uint route = token * K + k;
        acc = T(float(acc) + float(T(float(y[rows[route] * D + d]) * float(scores[route]))));
    }
    out[token * D + d] = acc;
"""


@functools.cache
def _moe_combine_kernel():
    return mx.fast.metal_kernel(name = "unsloth_moe_combine", input_names = ["y", "rows", "scores"],
                                output_names = ["out"], source = _MOE_COMBINE_SOURCE)


def _run_moe_combine(y, inv_order, scores):
    top_k, width = scores.shape[-1], y.shape[-1]
    tokens = scores.size // top_k
    return _moe_combine_kernel()(
        inputs = [y, inv_order.astype(mx.uint32), scores],
        template = [("T", y.dtype), ("K", top_k), ("D", width)],
        grid = (width, tokens, 1), threadgroup = (min(width, 64), 1, 1),
        output_shapes = [(*scores.shape[:-1], width)], output_dtypes = [y.dtype],
    )[0]


@functools.cache
def _moe_combine_verified(dtype, top_k, width):
    for tokens in (1, 67):
        routes = tokens * top_k
        y = (mx.random.normal((routes, 1, width), key = mx.random.key(tokens)) * 3).astype(dtype)
        y[..., 0] = -0.0  # a single route keeps the sign a sum from zero drops
        scores = mx.softmax(mx.random.normal((1, tokens, top_k), key = mx.random.key(tokens + 1)) * 2,
                            axis = -1).astype(dtype)
        inv_order = mx.random.permutation(routes, key = mx.random.key(tokens + 2)).astype(mx.uint32)
        native = (y[inv_order].reshape(1, tokens, top_k, width) * scores[..., None]).sum(axis = -2)
        fused = _run_moe_combine(y, inv_order, scores)
        if native.dtype != fused.dtype or not mx.array_equal(native.view(mx.uint16), fused.view(mx.uint16)):
            logger.warning("the fused MoE combine does not reproduce this MLX build's native rounding "
                           "for %s top-%d; the native chain stays in use", dtype, top_k)
            return False
    return True


def _fused_moe_combine(switch_mlp, x, inds, scores):
    """The routed experts' weighted sum read from their sorted rows, or None for the native chain."""
    if (not callable(getattr(type(switch_mlp), "sorted_experts", None))
            or scores.dtype not in _MOE_COMBINE_DTYPES or scores.shape != inds.shape):
        return None
    routed = switch_mlp.sorted_experts(x, inds)
    if routed is None:
        return None
    y, inv_order = routed
    # The kernel indexes in 32 bits; MLX's own ops switch to 64-bit indexing past that.
    if (y.size > _MOE_COMBINE_MAX_ELEMENTS or y.dtype != scores.dtype
            or not _moe_combine_verified(y.dtype, scores.shape[-1], y.shape[-1])):
        return None
    return _run_moe_combine(y, inv_order, scores)


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
        logits = self.gate(x)
        experts = getattr(self, "_unsloth_moe_routed", None)
        # Fold the shared-gate sigmoid into the routing launch only where the routed kernel consumes it.
        gate = None
        if (experts is not None and x.size // x.shape[-1] * self.top_k < _MOE_UNSORTED_ROUTES
                and (not scaled_shared or ("_shared_expert_scale" not in self.__dict__
                                           and _stock_shared_expert_scale(type(self)._shared_expert_scale)))):
            gate = self.shared_expert_gate(x)
        routed = None if gate is None else _fused_moe_router_shared(logits, gate, self.top_k, normalize)
        if routed is None:
            routed = _fused_moe_router(logits, _MOE_ROUTER_NO_SCALE, self.top_k, _QWEN_ROUTING, normalize)
            if routed is None:
                return native(self, x)
            routed += (None,)
        inds, scores, shared_scale = routed
        shared_y = self.shared_expert(x)
        if gate is not None and shared_scale is None:
            shared_scale = mx.sigmoid(gate)
        elif shared_scale is None:
            shared_scale = self._shared_expert_scale(x) if scaled_shared else mx.sigmoid(self.shared_expert_gate(x))
        if experts is not None and inds.size < _MOE_UNSORTED_ROUTES:
            out = experts(self, x, inds, scores, shared_scale, shared_y)
            if out is not None:
                return out
        y = _fused_moe_combine(self.switch_mlp, x, inds, scores)
        if y is None:
            y = (self.switch_mlp(x, inds) * scores[..., None]).sum(axis = -2)
        return y + shared_scale * shared_y

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
    # mlx_vlm 0.7.5 adds stop_gradient on the indices below; 0.7.8 qwen3_5_moe honours norm_topk_prob.
    "903a5a99afcad95a": (functools.partial(_qwen3_5_moe_call, scaled_shared = False, top_k_norm = False), _QWEN_ROUTING, "num_experts", "top_k"),
    "0ca80e0451802015": (functools.partial(_qwen3_5_moe_call, scaled_shared = False, top_k_norm = False), _QWEN_ROUTING, "num_experts", "top_k"),
    "b948b94a9d18333f": (functools.partial(_qwen3_5_moe_call, scaled_shared = True, top_k_norm = False), _QWEN_ROUTING, "num_experts", "top_k"),
    "10bab29b882b810a": (functools.partial(_qwen3_5_moe_call, scaled_shared = True, top_k_norm = False), _QWEN_ROUTING, "num_experts", "top_k"),
    "bc11d59e3b1a7d5f": (functools.partial(_qwen3_5_moe_call, scaled_shared = True, top_k_norm = True), _QWEN_ROUTING, "num_experts", "top_k"),
    # qwen3_next in mlx_lm and mlx_vlm, reused by mlx_lm qwen3_5
    "6d29bc869fdde5aa": (functools.partial(_qwen3_5_moe_call, scaled_shared = False, top_k_norm = True), _QWEN_ROUTING, "num_experts", "top_k"),
    "6b15c950ad0c22a1": (functools.partial(_qwen3_5_moe_call, scaled_shared = False, top_k_norm = True), _QWEN_ROUTING, "num_experts", "top_k"),
    # gemma4 Router in mlx_vlm gemma4_text, gemma4 and mlx_lm gemma4_text
    "c64fed4e7e7514ea": (_gemma4_router_call, _GEMMA_ROUTING, "config.num_experts", "config.top_k_experts"),
    "09b6eefdbdfb7f24": (_gemma4_router_call, _GEMMA_ROUTING, "config.num_experts", "config.top_k_experts"),
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


# `generation_mode` serializes, but the loader's two generate() wrappers enter this scope directly
# and do not, and this is the only one of the four carrying a per-module count: unlocked,
# `count += 1` against a `finally` that pops the same key strands the patched class or raises
# AttributeError out of generation (5 of 40 contended rounds).
_MOE_ROUTER_LOCK = RLock()

@contextmanager
def fused_moe_router(model, *, _modules = None):
    """Fuse the MoE routing chain into one Metal dispatch during serialized inference.

    Modules whose routing body, expert count, or top-k the kernel does not cover keep
    their native call, as do training and distributed models; the native call is also
    taken per invocation when the kernel cannot reproduce this MLX build's rounding.
    Instance classes are restored when the context exits, including on cancellation.
    Gemma routers fold `scale` into a normalization weight held until the outermost
    scope exits, so `scale` must stay fixed while the scope is open, as the gate and up
    packing requires of its own weights: replacing it is detected and takes the native
    call, editing it in place is not detected. Edits between scopes are always picked up.

    Bit-identity covers every index and every finite weight, not the NaN payload of a poisoned
    row: MLX's own payload is not stable across Apple GPU families, so nothing can match it
    everywhere. Both paths return NaN in the same positions.
    """
    changed = []
    try:
        with _MOE_ROUTER_LOCK:
            modules = _fusion_modules(model, _modules)
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
        with _MOE_ROUTER_LOCK:
            for module in reversed(changed):
                scopes = getattr(module, "_unsloth_router_scopes", 0)
                if scopes > 1:
                    module._unsloth_router_scopes = scopes - 1
                    continue
                native = getattr(type(module), "_unsloth_router_native", None)
                if native is not None:
                    module.__class__ = native
                for name in ("_unsloth_router_scopes", "_unsloth_router_norm"):
                    module.__dict__.pop(name, None)


@contextmanager
def fused_moe_routed_experts(model, *, _modules = None):
    """Run Qwen sparse MoE blocks' routed experts on two decode kernels during serialized inference.

    Gate/up with SwiGLU, and down with the routing-weight combine and shared-expert add, serve
    calls whose routes MLX gathers unsorted, each checked bitwise against the native experts at
    first use; where they serve a call, the routing launch also writes the shared-expert gate's
    sigmoid. The kernels run from `fused_moe_router`'s fused call and from `exact_speculative_verify`'s
    verify blocks, so without either scope the blocks stay native. Training and distributed models keep the native experts, and so does
    every model while UNSLOTH_MLX_ROUTED_EXPERTS=0.
    """
    changed = []
    try:
        with _MOE_ROUTER_LOCK:
            modules = _fusion_modules(model, _modules)
            if (os.environ.get("UNSLOTH_MLX_ROUTED_EXPERTS", "1") != "0"
                    and not getattr(model, "_unsloth_mlx_distributed_parallel_mode", None)
                    and _moe_router_kernel() is not None):
                for _, module in modules:
                    # Type before `training`: named_modules() may yield plain stand-ins.
                    if not isinstance(module, dict) or module.training or "__call__" in module:
                        continue
                    if getattr(module, "_unsloth_moe_routed_scopes", 0):
                        module._unsloth_moe_routed_scopes += 1
                        changed.append(module)
                        continue
                    base = getattr(type(module), "_unsloth_router_native", None) or type(module)
                    spec = _moe_router_class(base)
                    if spec is None or spec[1] != _QWEN_ROUTING:
                        continue
                    experts = _moe_routed_experts(module)
                    if experts is None:
                        continue
                    module._unsloth_moe_routed = experts
                    module._unsloth_moe_routed_scopes = 1
                    changed.append(module)
        yield model
    finally:
        with _MOE_ROUTER_LOCK:
            for module in reversed(changed):
                scopes = module._unsloth_moe_routed_scopes
                if scopes > 1:
                    module._unsloth_moe_routed_scopes = scopes - 1
                    continue
                for name in ("_unsloth_moe_routed_scopes", "_unsloth_moe_routed"):
                    module.__dict__.pop(name, None)


# Bodies the NAX small-M routes stand in for: each falls back to exactly this call.
_NAX_QMM_CONTRACT = {"mlx.nn.layers.quantized": {
    "QuantizedLinear.__call__": "ca5cafb6d038d955",
    "QuantizedEmbedding.as_linear": "4617094b8a99a71a",
}}
_NAX_QMM_LOCK = RLock()
_NAX_QMM_VERIFIED = {}
_NAX_INT8_QMM_VERIFIED = {}
_NAX_SORTED_GATHER_VERIFIED = {}


def _nax_qmm_matches_native(x, w, scales, biases, group_size, bits, routed):
    """Compare the kernel with stock's fp32 qmv of each row, within fp32 reordering of K products.

    A batched native call is no reference: its qmm rounds dequantized weights to the input dtype.
    """
    x32 = x.astype(mx.float32)
    scales32, biases32 = scales.astype(mx.float32), biases.astype(mx.float32)
    row = lambda i, a, s, b: mx.quantized_matmul(a[i:i + 1], w, s, b, transpose = True,
                                                 group_size = group_size, bits = bits)
    rows = range(x.shape[0])
    reference = mx.concatenate([row(i, x32, scales32, biases32) for i in rows])
    # |s q + b| <= |s| q + |b|, so this is a bound on sum_k |x_k w_k|.
    magnitude = mx.concatenate([row(i, mx.abs(x32), mx.abs(scales32), mx.abs(biases32)) for i in rows])
    got = routed.astype(mx.float32)
    bound = mx.finfo(x.dtype).eps * mx.abs(reference) + x.shape[-1] * 2.0 ** -23 * magnitude
    agree = (mx.abs(got - reference) <= bound) | (got == reference) | (mx.isnan(got) & mx.isnan(reference))
    return bool(mx.all(agree).item())


def _nax_small_m_qmm(module, x, bindings):
    """The small-row kernel's result for this call, or None when it takes the native call."""
    low, high = module._unsloth_nax_qmm_rows
    if not isinstance(x, mx.array) or x.ndim < 2 or not x.shape[-1] or not low <= x.size // x.shape[-1] <= high:
        return None
    if module.training or not _bindings_intact(bindings):
        return None
    w, scales, biases = module.get("weight"), module.get("scales"), module.get("biases")
    if not all(isinstance(a, mx.array) for a in (w, scales, biases)):
        return None
    N, K = w.shape[0], x.shape[-1]
    M = x.size // K
    group_size, bits = module.group_size, module.bits
    if (not nax.small_m_qmm_supported(N, K, group_size, bits, module.mode)
            or x.dtype not in (mx.bfloat16, mx.float16) or not x.dtype == scales.dtype == biases.dtype
            or w.dtype != mx.uint32 or w.shape != (N, K * bits // 32)
            or scales.shape != (N, K // group_size) or biases.shape != scales.shape):
        return None
    # Every kernel variant a call can select is checked on its own first use; a ragged row count
    # selects the bounds-checked one.
    geometry = nax.small_m_qmm_geometry(M, N, K, group_size)
    key = (N, K, bits, group_size, x.dtype, geometry, M % geometry[0] == 0)
    verified = _NAX_QMM_VERIFIED.get(key)
    if verified is False:
        return None
    flat = x.reshape(M, K)
    routed = nax.small_m_qmm(flat, w, scales, biases, group_size, bits)
    if verified is None:
        try:
            verified = _nax_qmm_matches_native(flat, w, scales, biases, group_size, bits, routed)
        except (RuntimeError, ValueError):
            return None  # inside a function transformation, which cannot evaluate; checked later
        if verified and not any(_NAX_QMM_VERIFIED.values()):
            logger.info("small-batch quantized projections now run on the NAX kernel; "
                        "UNSLOTH_MLX_NAX_QMM=0 keeps the native call")
        _NAX_QMM_VERIFIED[key] = verified
        if not verified:
            logger.warning("the NAX small-row quantized matmul disagrees with the native one for "
                           "N=%d K=%d %d-bit group %d %s; the native call stays in use", N, K, bits,
                           group_size, x.dtype)
    return routed.reshape(*x.shape[:-1], N) if verified else None


def _nax_qmm_eligible(supported, K, dtype, w, scales, biases, group_size, bits, mode):
    return (all(isinstance(a, mx.array) for a in (w, scales, biases))
            and supported(w.shape[-2], K, group_size, bits, mode)
            and dtype in (mx.bfloat16, mx.float16) and dtype == scales.dtype == biases.dtype
            and w.dtype == mx.uint32 and w.shape[-1] == K * bits // 32
            and scales.shape == (*w.shape[:-1], K // group_size) and biases.shape == scales.shape)


def _nax_int8_qmm_matches_native(x, w, scales, biases, group_size, bits, routed, indices):
    """Whether rows spread over the call quantize to x within half a code step and the kernel matches
    stock's fp32 product of those rounded rows."""
    M, K = x.shape
    rows = mx.array(sorted({round(i * (M - 1) / 31) for i in range(32)}))
    codes, step, *_ = nax._int8_quantize_activations(x[rows], group_size, bits)
    span = nax._int8_qmm_activation_group(group_size)
    if bits == 4:   # stored in the kernel's nibble order
        codes = codes.reshape(-1, K // 16, 2, 2, 4).swapaxes(-1, -2).reshape(-1, K)
    x32 = x[rows].astype(mx.float32)
    rounded = (codes.reshape(-1, K // span, span) * step.T[..., None]).reshape(-1, K)
    finite = mx.isfinite(step.T)[..., None]
    error = mx.abs(rounded - x32).reshape(-1, K // span, span)
    if not mx.all(~finite | (error <= step.T[..., None] * (0.5 + 2.0 ** -12))).item():
        return False
    if indices is None:
        product = lambda a, s, b: mx.quantized_matmul(a, w, s, b, transpose = True, group_size = group_size,
                                                      bits = bits)
    else:
        picked = indices[rows]
        product = lambda a, s, b: mx.gather_qmm(a[:, None], w, s, b, rhs_indices = picked, transpose = True,
                                                group_size = group_size, bits = bits)[:, 0]
    scales32, biases32 = scales.astype(mx.float32), biases.astype(mx.float32)
    reference = product(rounded, scales32, biases32)
    magnitude = product(mx.abs(rounded), mx.abs(scales32), mx.abs(biases32))
    got = routed[rows].astype(mx.float32)
    bound = mx.finfo(x.dtype).eps * mx.abs(reference) + K * 2.0 ** -22 * magnitude
    if bits == 8:   # centred codes against b + 128 * s cancel only to rounding, even where stock gives exactly 0
        bound += 2.0 ** -15 * product(mx.abs(rounded), mx.abs(scales32), mx.abs(biases32) + 256 * mx.abs(scales32))
    agree = (mx.abs(got - reference) <= bound) | (mx.isnan(got) & mx.isnan(reference))
    return bool(mx.all(agree).item())


def _nax_verified_int8_qmm(key, flat, w, scales, biases, group_size, bits, indices = None, token_rows = None):
    verified = _NAX_INT8_QMM_VERIFIED.get(key)
    if verified is False:
        return None
    if indices is None:
        routed = nax.int8_qmm(flat, w, scales, biases, bits)
    else:
        routed = nax.int8_gather_qmm(flat, w, scales, biases, indices, bits, token_rows)
    if verified is None:
        rows = flat if token_rows is None else flat[token_rows]
        try:
            verified = _nax_int8_qmm_matches_native(rows, w, scales, biases, group_size, bits, routed, indices)
        except (RuntimeError, ValueError):
            return None  # inside a function transformation, which cannot evaluate; checked later
        if verified and not any(_NAX_INT8_QMM_VERIFIED.values()):
            logger.info("prefill projections now run with int8 activations on the NAX kernel")
        _NAX_INT8_QMM_VERIFIED[key] = verified
        if not verified:
            logger.warning("the int8-activation NAX kernel disagrees with the native quantized matmul "
                           "for %s; the native call stays in use", key)
    return routed if verified else None


def _nax_int8_prefill_dense(module, x, bindings):
    """The int8-activation kernel's result for this dense call, or None when it takes the native call."""
    low = module._unsloth_nax_int8_prefill_rows
    if not low or not isinstance(x, mx.array) or x.ndim < 2 or not x.shape[-1] or x.size // x.shape[-1] < low:
        return None
    if module.training or not _bindings_intact(bindings):
        return None
    w, scales, biases = module.get("weight"), module.get("scales"), module.get("biases")
    if (not _nax_qmm_eligible(nax.int8_qmm_supported, x.shape[-1], x.dtype, w, scales, biases, module.group_size,
                              module.bits, module.mode)
            or w.ndim != 2):
        return None
    N, K = w.shape[0], x.shape[-1]
    routed = _nax_verified_int8_qmm(("dense", N, K, module.group_size, module.bits, x.dtype), x.reshape(-1, K),
                                    w, scales, biases, module.group_size, module.bits)
    return None if routed is None else routed.reshape(*x.shape[:-1], N)


def _nax_gather_rows(owner, flag, x, w, indices, sorted_indices, bindings, token_rows):
    """The sorted rows of a gathered call that the NAX route `flag` enables on `owner` may take, or 0."""
    if not sorted_indices or not isinstance(x, mx.array) or x.ndim != 3 or x.shape[1] != 1:
        return 0
    if not getattr(owner, flag, False) or owner.training:
        return 0
    if bindings is not None and not _bindings_intact(bindings):
        return 0
    T = x.shape[0] if token_rows is None else token_rows.shape[0]
    if (not isinstance(indices, mx.array) or indices.shape != (T,)
            or indices.dtype not in (mx.uint32, mx.int32) or not isinstance(w, mx.array) or w.ndim != 3):
        return 0
    return T


def _nax_int8_prefill_gather(owner, x, w, scales, biases, indices, sorted_indices, group_size, bits, mode, bindings,
                     token_rows = None):
    """The gathered int8 kernel's result for x of shape [T, 1, K] sorted by expert, or None for the
    native call. `owner` enabled the route; `w` may pack several projections. With `token_rows`, x
    holds one row per token and sorted row i is `x[token_rows[i]]`."""
    T = _nax_gather_rows(owner, "_unsloth_nax_int8_prefill", x, w, indices, sorted_indices, bindings, token_rows)
    if not T or not _nax_qmm_eligible(nax.int8_qmm_supported, x.shape[-1], x.dtype, w, scales, biases, group_size,
                                      bits, mode):
        return None
    (E, N, _), K = w.shape, x.shape[-1]
    low = nax.int8_prefill_expert_min_rows(E, bits, token_rows is not None)
    if not low or T < low:
        return None
    routed = _nax_verified_int8_qmm(("gather", E, N, K, group_size, bits, x.dtype, token_rows is not None),
                                    x.reshape(-1, K), w, scales, biases, group_size, bits, indices, token_rows)
    return None if routed is None else routed.reshape(T, 1, N)


def _nax_sorted_gather(owner, x, w, scales, biases, indices, sorted_indices, group_size, bits, mode, bindings,
                       token_rows = None):
    """The sorted gather kernel's result, or None for the native call."""
    T = _nax_gather_rows(owner, "_unsloth_nax_sorted_gather", x, w, indices, sorted_indices, bindings, token_rows)
    if not T or not _nax_qmm_eligible(nax.sorted_gather_qmm_supported, x.shape[-1], x.dtype, w, scales, biases,
                                      group_size, bits, mode):
        return None
    (E, N, _), K = w.shape, x.shape[-1]
    low, high = nax.sorted_gather_row_range(E, bits)
    if not low <= T <= high or (token_rows is not None and token_rows.dtype not in (mx.uint32, mx.int32)):
        return None
    key = (E, N, K, group_size, bits, x.dtype, token_rows is not None)
    verified = _NAX_SORTED_GATHER_VERIFIED.get(key)
    if verified is False:
        return None
    routed = nax.sorted_gather_qmm(x.reshape(-1, K), w, scales, biases, indices, bits, token_rows).reshape(T, 1, N)
    if verified is None:
        try:
            native = mx.gather_qmm(x if token_rows is None else x[token_rows], w, scales, biases, rhs_indices = indices,
                                   transpose = True, group_size = group_size, bits = bits, sorted_indices = True)
            verified = mx.array_equal(routed.view(mx.uint16), native.view(mx.uint16)).item()
        except (RuntimeError, ValueError):
            return None  # inside a function transformation, which cannot evaluate; checked later
        if verified and not any(_NAX_SORTED_GATHER_VERIFIED.values()):
            logger.info("sorted routed-expert projections now run on the NAX gather kernel; "
                        "UNSLOTH_MLX_NAX_SORTED_GATHER=0 keeps the native call")
        _NAX_SORTED_GATHER_VERIFIED[key] = verified
        if not verified:
            logger.warning("the NAX sorted gather kernel does not reproduce the native one for %s; "
                           "the native call stays in use", key)
    return routed if verified else None


def _nax_sorted_gather_route(module):
    w, scales, biases = module.get("weight"), module.get("scales"), module.get("biases")
    if not isinstance(w, mx.array) or w.ndim != 3:
        return False
    low, high = nax.sorted_gather_row_range(w.shape[0], module.bits)
    return low <= high and _nax_qmm_eligible(
        nax.sorted_gather_qmm_supported, w.shape[-1] * 32 // module.bits, getattr(scales, "dtype", None), w, scales,
        biases, module.group_size, module.bits, module.mode)


# Keyed by the current methods: a transient wrapper, such as the training patches', misses only
# while it is installed. Fallbacks read the base method at call time so a later one is honored.
@functools.cache
def _nax_qmm_classes(*cache_key):
    bindings = _resolved_bindings(_NAX_QMM_CONTRACT)
    if bindings is None:
        return {}

    def linear(self, x):
        routed = _nax_small_m_qmm(self, x, bindings)
        if routed is None:
            routed = _nax_int8_prefill_dense(self, x, bindings)
        if routed is None:
            routed = _dense_prefill_in_scope(self, x)
        if routed is None:
            return nn.QuantizedLinear.__call__(self, x)
        return routed + self["bias"] if "bias" in self else routed

    def as_linear(self, x):
        routed = _nax_small_m_qmm(self, x, bindings)
        return nn.QuantizedEmbedding.as_linear(self, x) if routed is None else routed

    return {
        base: type(f"_NaxSmallM{base.__name__}", (base,),
                   {name: method, "_unsloth_nax_qmm_native": base, "_unsloth_nax_int8_prefill_rows": 0})
        for base, name, method in ((nn.QuantizedLinear, "__call__", linear),
                                   (nn.QuantizedEmbedding, "as_linear", as_linear))
    }


_NAX_SWITCH_CONTRACT = {path: {"QuantizedSwitchLinear.__call__": "59bbb193612cbe06"}
                        for path in ("mlx_lm.models.switch_layers", "mlx_vlm.models.switch_layers")}


@functools.cache
def _nax_switch_class(base, call, path):
    bindings = _resolved_bindings({path: _NAX_SWITCH_CONTRACT[path]})
    if bindings is None:
        return None

    def switch(self, x, indices, sorted_indices = False):
        call = (self, x, self.get("weight"), self.get("scales"), self.get("biases"), indices, sorted_indices,
                self.group_size, self.bits, self.mode, bindings)
        routed = _nax_int8_prefill_gather(*call)
        if routed is None:
            routed = _nax_sorted_gather(*call)
        if routed is None:
            return base.__call__(self, x, indices, sorted_indices = sorted_indices)
        return routed + mx.expand_dims(self["bias"][indices], -2) if "bias" in self else routed

    return type(f"_NaxGather{base.__name__}", (base,), {"__call__": switch, "_unsloth_nax_qmm_native": base})


def _nax_switch_classes():
    classes = {}
    for path in _NAX_SWITCH_CONTRACT:
        base = getattr(sys.modules.get(path), "QuantizedSwitchLinear", None)
        if isinstance(base, type) and base not in classes:
            swapped = _nax_switch_class(base, base.__call__, path)
            if swapped is not None:
                classes[base] = swapped
    return classes


def _nax_qmm_row_range(module):
    """The rows at which this module's calls take the kernel, or None when none ever would."""
    weight = module.get("weight")
    if module.training or not isinstance(weight, mx.array) or weight.ndim != 2 or module.get("biases") is None:
        return None
    N, K = weight.shape[0], weight.shape[1] * 32 // module.bits
    if not nax.small_m_qmm_supported(N, K, module.group_size, module.bits, module.mode):
        return None
    low, high = nax.small_m_qmm_row_range(N, K, module.group_size, module.bits)
    return (low, high) if low <= high else None


def _nax_int8_prefill_packed(model):
    """Ids of the gate and up projections packed now or by a `fused_moe_gate_up` entered later."""
    specs = _moe_switch_specs()
    packed = set()
    for _, module in model.named_modules() if hasattr(model, "named_modules") else ():
        pack = getattr(module, "_unsloth_moe_gate_up", None)
        if isinstance(pack, _PackedMoEGateUp) and type(module) in _MOE_GATE_UP_CLASSES.values():
            packed.update((id(pack.gate), id(pack.up)))
        elif type(module) in specs and _moe_gate_up_packable(module, specs[type(module)][0]):
            packed.update((id(module.gate_proj), id(module.up_proj)))
    return packed


def _nax_int8_prefill_route(module, classes, switches, packed):
    """The fewest rows at which a quantized linear takes the int8 route, True for a routed-expert
    projection whose calls it can take (their rows are checked per call), or 0."""
    base = _native_type(module)
    dense = base is nn.QuantizedLinear and base in classes
    if not (dense or base in switches):
        return 0
    w, scales, biases = module.get("weight"), module.get("scales"), module.get("biases")
    if not isinstance(w, mx.array) or w.ndim != (2 if dense else 3):
        return 0
    N, K = w.shape[-2], w.shape[-1] * 32 // module.bits
    dtype = getattr(scales, "dtype", None)
    if not _nax_qmm_eligible(nax.int8_qmm_supported, K, dtype, w, scales, biases, module.group_size, module.bits,
                             module.mode):
        return 0
    if dense:
        return nax.int8_prefill_min_rows(N, K)
    E, bits = w.shape[0], module.bits
    return bool(nax.int8_prefill_expert_min_rows(E, bits)
                or id(module) in packed and nax.int8_prefill_expert_min_rows(E, bits, True))


class Int8PrefillStatus(NamedTuple):
    available: bool
    # "", nax_unavailable, distributed, not_downloaded, no_eligible_projections or probe_failed
    reason: str
    projections: int   # linears and routed-expert projections the int8 route can take; approximate before load

    def __bool__(self):
        return self.available


def int8_prefill_available(model):
    """Whether `int8_prefill=True` can route any of `model`'s projections to int8 activations
    when it generates here through `generation_mode`.

    Checks the device, the model's quantization and shapes, then, once per macOS build and MLX
    version, that the kernels build. It ignores the flag, `UNSLOTH_MLX_INT8_PREFILL` and training
    mode, and leaves the model unchanged.
    """
    if not nax.nax_available():
        return Int8PrefillStatus(False, "nax_unavailable", 0)
    if getattr(model, "_unsloth_mlx_distributed_parallel_mode", None):
        return Int8PrefillStatus(False, "distributed", 0)
    classes = _nax_qmm_classes(nn.QuantizedLinear.__call__, nn.QuantizedEmbedding.as_linear)
    switches, packed = _nax_switch_classes(), _nax_int8_prefill_packed(model)
    modules = {id(module): module for _, module in model.named_modules()} if hasattr(model, "named_modules") else {}
    return _int8_prefill_status(sum(bool(_nax_int8_prefill_route(module, classes, switches, packed))
                                    for module in modules.values()))


def int8_prefill_checkpoint_available(model_dir):
    """`int8_prefill_available` for the directory of a downloaded MLX checkpoint, judged from its
    config.json and safetensors headers without loading it: reason "not_downloaded" when either is
    missing. Headers cannot tell an embedding from a linear, and an expert projection counts when
    either its packed or its other calls can route, so `projections` is approximate."""
    if not nax.nax_available():
        return Int8PrefillStatus(False, "nax_unavailable", 0)
    model_dir = Path(model_dir)
    shards = list(model_dir.glob("*.safetensors"))
    if not (model_dir / "config.json").is_file() or not shards:
        return Int8PrefillStatus(False, "not_downloaded", 0)
    quantization = json.loads((model_dir / "config.json").read_text()).get("quantization") or {}
    tensors = {}
    for shard in shards:
        with open(shard, "rb") as file:
            tensors.update(json.loads(file.read(struct.unpack("<Q", file.read(8))[0])))
    projections = 0
    for name, scales in tensors.items():
        if not name.endswith(".scales"):
            continue
        prefix = name.removesuffix(".scales")
        weight = tensors.get(f"{prefix}.weight")
        # Per-layer settings are keyed by module path, which mlx-vlm also matches without "language_model.",
        # and what they leave out takes MLX's defaults rather than the checkpoint-wide setting.
        layer = next((quantization[key] for key in (prefix, prefix.removeprefix("language_model."))
                      if key in quantization), quantization)
        if weight is None or not isinstance(layer, dict) or scales["dtype"] not in ("BF16", "F16"):
            continue
        bits, shape = layer.get("bits", 4), weight["shape"]
        N, K = shape[-2], shape[-1] * 32 // bits
        if not nax.int8_qmm_supported(N, K, layer.get("group_size", 64), bits, layer.get("mode", "affine")):
            continue
        if len(shape) == 2:
            projections += bool(nax.int8_prefill_min_rows(N, K))
        else:
            projections += bool(nax.int8_prefill_expert_min_rows(shape[0], bits)
                                or nax.int8_prefill_expert_min_rows(shape[0], bits, True))
    return _int8_prefill_status(projections)


def _int8_prefill_status(projections):
    if not projections:
        return Int8PrefillStatus(False, "no_eligible_projections", 0)
    if not nax.kernel_probe_passed(nax.INT8_QMM_PROBE_KEY, nax.__name__, "probe_int8_qmm"):
        return Int8PrefillStatus(False, "probe_failed", projections)
    return Int8PrefillStatus(True, "", projections)


@contextmanager
def nax_quantized_linear(model, int8_prefill = None, *, _modules = None):
    """Run affine quantized projections on the M5 neural accelerators where measured faster.

    Quantized linears and tied quantized embedding heads called with the row counts measured
    faster than stock for their shape on this GPU use a matmul2d kernel that reads the stock
    packed codes and keeps stock's per-group factorization in fp32; each (shape, quantization,
    dtype, kernel variant) is checked against stock's fp32 arithmetic at first use. Single rows,
    training, other quantizations and devices without NAX keep the native call.
    `UNSLOTH_MLX_NAX_QMM=0` turns the route off.

    `int8_prefill=True`, or `UNSLOTH_MLX_INT8_PREFILL=1` when it is None, also runs affine 3- to
    8-bit linears and routed experts with int8 activations: faster, but lossy enough to raise the
    loss of precision-sensitive checkpoints. Linears route from 17 rows, so batched decoding of 17
    or more sequences and wider speculative verification also run int8. The outermost scope
    decides; `int8_prefill_available(model)` says whether it takes effect here.

    Sorted routed-expert projections run on a NAX gather kernel that reproduces `mx.gather_qmm`
    bit for bit where it is measured faster; `UNSLOTH_MLX_NAX_SORTED_GATHER=0` turns it off.
    """
    changed, tracked = [], hasattr(model, "__dict__")
    try:
        with _NAX_QMM_LOCK:
            depth, outer = getattr(model, "__dict__", {}).get("_unsloth_nax_int8_prefill_scope", (0, None))
            if outer is not None:
                int8_prefill = outer
            elif int8_prefill is None:
                int8_prefill = os.environ.get("UNSLOTH_MLX_INT8_PREFILL", "0") == "1"
            if tracked:
                model.__dict__["_unsloth_nax_int8_prefill_scope"] = (depth + 1, bool(int8_prefill))
            small_m = os.environ.get("UNSLOTH_MLX_NAX_QMM", "1") != "0" and nax.gap_open("small_m_qmm")
            sorted_gather = (os.environ.get("UNSLOTH_MLX_NAX_SORTED_GATHER", "1") != "0"
                             and nax.gap_open("sorted_gather_qmm"))
            if ((small_m or int8_prefill or sorted_gather) and nax.nax_available()
                    and not getattr(model, "_unsloth_mlx_distributed_parallel_mode", None)):
                classes = _nax_qmm_classes(nn.QuantizedLinear.__call__, nn.QuantizedEmbedding.as_linear)
                switches = _nax_switch_classes() if int8_prefill or sorted_gather else {}
                packed = _nax_int8_prefill_packed(model) if int8_prefill else set()
                nested, fresh, seen = [], [], set()
                for _, module in _fusion_modules(model, _modules):
                    if id(module) in seen:   # named_modules() yields a shared module once per path
                        continue
                    seen.add(id(module))
                    base = type(module)
                    if base in classes.values() or base in switches.values():
                        if getattr(module, "_unsloth_nax_qmm_scopes", 0):
                            nested.append(module)
                    elif base in classes:
                        rows = _nax_qmm_row_range(module) if small_m else None
                        int8_prefill_rows = (int8_prefill and not module.training
                                     and _nax_int8_prefill_route(module, classes, switches, packed))
                        if rows is not None or int8_prefill_rows:
                            fresh.append((module, rows, int8_prefill_rows, False))
                    elif base in switches and not module.training:
                        int8 = bool(int8_prefill and _nax_int8_prefill_route(module, classes, switches, packed))
                        gather = sorted_gather and _nax_sorted_gather_route(module)
                        if int8 or gather:
                            fresh.append((module, None, int8, gather))
                if any(rows is not None for _, rows, _, _ in fresh) and not nax.kernel_probe_passed(
                        nax.QMM_PROBE_KEY, nax.__name__, "probe_small_m_qmm"):
                    fresh = [(module, None, int8, gather) for module, _, int8, gather in fresh]
                if any(int8 for _, _, int8, _ in fresh) and not nax.kernel_probe_passed(
                        nax.INT8_QMM_PROBE_KEY, nax.__name__, "probe_int8_qmm"):
                    fresh = [(module, rows, 0, gather) for module, rows, _, gather in fresh]
                if any(gather for *_, gather in fresh) and not nax.kernel_probe_passed(
                        nax.SORTED_GATHER_PROBE_KEY, nax.__name__, "probe_sorted_gather_qmm"):
                    fresh = [(module, rows, int8, False) for module, rows, int8, _ in fresh]
                fresh = [entry for entry in fresh if entry[1] is not None or entry[2] or entry[3]]
                for module, rows, int8_prefill_rows, gather in fresh:
                    if type(module) in switches:
                        if int8_prefill_rows:
                            module._unsloth_nax_int8_prefill = True
                        if gather:
                            module._unsloth_nax_sorted_gather = True
                        module.__class__ = switches[type(module)]
                        continue
                    module._unsloth_nax_qmm_rows = (0, -1) if rows is None else rows
                    if int8_prefill_rows:
                        module._unsloth_nax_int8_prefill_rows = int8_prefill_rows
                    module.__class__ = classes[type(module)]
                for module in nested + [entry[0] for entry in fresh]:
                    module._unsloth_nax_qmm_scopes = getattr(module, "_unsloth_nax_qmm_scopes", 0) + 1
                    changed.append(module)
        yield model
    finally:
        with _NAX_QMM_LOCK:
            if tracked:
                depth, decision = model.__dict__.pop("_unsloth_nax_int8_prefill_scope")
                if depth > 1:
                    model.__dict__["_unsloth_nax_int8_prefill_scope"] = (depth - 1, decision)
            for module in reversed(changed):
                scopes = getattr(module, "_unsloth_nax_qmm_scopes", 0)
                if scopes > 1:
                    module._unsloth_nax_qmm_scopes = scopes - 1
                    continue
                native = getattr(type(module), "_unsloth_nax_qmm_native", None)
                if native is not None:
                    module.__class__ = native
                for name in ("_unsloth_nax_qmm_scopes", "_unsloth_nax_int8_prefill_rows", "_unsloth_nax_int8_prefill",
                             "_unsloth_nax_sorted_gather"):
                    module.__dict__.pop(name, None)
                module.pop("_unsloth_nax_qmm_rows", None)   # a tuple is stored in the module mapping


_DENSE_QMM_CONTRACT = {"mlx.nn.layers.quantized": {"QuantizedLinear.__call__": "ca5cafb6d038d955"}}
_DENSE_QMM_LOCK = RLock()
_DENSE_QMM_VERIFIED = {}
_DENSE_QMM_MIN_ROWS = 1024
# MLX's quantized and dense matmuls each split the sum over K for small outputs, by different rules,
# and then round differently; at or below this many 16x16 output tiles the comparison would mostly fail.
_DENSE_QMM_MIN_TILES = 4096
# The dequantized weight is a transient the size of the float layer; a vocabulary head would be gigabytes.
_DENSE_QMM_MAX_WEIGHTS = 1 << 28


def _dense_prefill_qmm(module, x, bindings):
    """`x` times the dequantized weight for a prefill-sized call, or None when it takes the native call.

    MLX's quantized matmul dequantizes inside every tile; dequantizing once is the same arithmetic.
    Each (shape, quantization, dtypes) is compared bit for bit on its first use; the row count is part of
    the shape because MLX picks its kernels by it.
    """
    if not isinstance(x, mx.array) or x.ndim < 2 or x.dtype not in (mx.bfloat16, mx.float16) or not x.shape[-1]:
        return None
    w, scales, biases = module.get("weight"), module.get("scales"), module.get("biases")
    if not isinstance(w, mx.array) or w.ndim != 2 or not isinstance(scales, mx.array):
        return None
    N, K, M = w.shape[0], x.shape[-1], x.shape[-2]
    if x.size != M * K:  # MLX reads the row count of a batch from its memory layout
        return None
    if (M < _DENSE_QMM_MIN_ROWS or -(-M // 16) * -(-N // 16) <= _DENSE_QMM_MIN_TILES
            or N * K > _DENSE_QMM_MAX_WEIGHTS or module.training or not _bindings_intact(bindings)):
        return None
    group_size, bits, mode = module.group_size, module.bits, module.mode
    family = (N, K, bits, group_size, mode, x.dtype, w.dtype, scales.dtype, getattr(biases, "dtype", None))
    key = (x.shape, *family)
    verified = _DENSE_QMM_VERIFIED.get(key)
    if verified is False or family in _DENSE_QMM_VERIFIED:   # one differing row count retires the projection shape
        return None
    weight = mx.dequantize(w, scales, biases, group_size = group_size, bits = bits, mode = mode).astype(x.dtype)
    dense = x @ weight.T
    if verified is None:
        native = mx.quantized_matmul(x, w, scales = scales, biases = biases, transpose = True,
                                     group_size = group_size, bits = bits, mode = mode)
        try:
            verified = native.dtype == dense.dtype and bool(
                mx.array_equal(native.view(mx.uint16), dense.view(mx.uint16)).item())
        except (RuntimeError, ValueError):
            return None  # inside a function transformation, which cannot evaluate; checked later
        _DENSE_QMM_VERIFIED[key] = verified
        if not verified:
            _DENSE_QMM_VERIFIED[family] = False
            logger.warning("the dense prefill matmul differs from the quantized one for N=%d K=%d %d-bit "
                           "%s group %d %s at %d rows; the native call stays in use",
                           N, K, bits, mode, group_size, x.dtype, M)
    return dense if verified else None


@functools.cache
def _dense_qmm_bindings(*cache_key):
    return _resolved_bindings(_DENSE_QMM_CONTRACT)


def _dense_prefill_in_scope(module, x):
    """The dense result for a NAX-routed linear inside a `dense_prefill_linear` scope, else None."""
    if not module.__dict__.get("_unsloth_dense_qmm_scopes"):
        return None
    bindings = _dense_qmm_bindings(nn.QuantizedLinear.__call__)
    return None if bindings is None else _dense_prefill_qmm(module, x, bindings)


@functools.cache
def _dense_qmm_class(*cache_key):
    bindings = _dense_qmm_bindings(*cache_key)
    if bindings is None:
        return None

    def linear(self, x):
        routed = _dense_prefill_qmm(self, x, bindings)
        if routed is None:
            return nn.QuantizedLinear.__call__(self, x)
        return routed + self["bias"] if "bias" in self else routed

    return type("_DensePrefillQuantizedLinear", (nn.QuantizedLinear,), {"__call__": linear})


@contextmanager
def dense_prefill_linear(model, *, _modules = None):
    """Run prefill-sized quantized projections as one dequantize and a dense matmul.

    Calls of 1024 or more rows on a large enough output dequantize the weight once and multiply
    densely, bit-identical to the quantized matmul; smaller calls,
    training, and any shape that fails its first-use comparison keep the native call. A linear the
    NAX scope already routes keeps its small-row and int8 routes and takes this one for the rest.
    `UNSLOTH_MLX_DENSE_PREFILL=0` turns the route off.
    """
    changed = []
    try:
        with _DENSE_QMM_LOCK:
            fused = None
            if (os.environ.get("UNSLOTH_MLX_DENSE_PREFILL", "1") != "0"
                    and not getattr(model, "_unsloth_mlx_distributed_parallel_mode", None)):
                fused = _dense_qmm_class(nn.QuantizedLinear.__call__)
            if fused is not None:
                seen = set()
                for _, module in _fusion_modules(model, _modules):
                    if id(module) in seen:   # named_modules() yields a shared module once per path
                        continue
                    seen.add(id(module))
                    if (type(module) not in (fused, nn.QuantizedLinear)
                            and getattr(type(module), "_unsloth_nax_qmm_native", None) is not nn.QuantizedLinear):
                        continue
                    if module.__dict__.get("_unsloth_dense_qmm_scopes"):
                        module._unsloth_dense_qmm_scopes += 1
                    elif module.training or type(module) is fused:
                        continue
                    else:
                        if type(module) is nn.QuantizedLinear:
                            module.__class__ = fused
                        module._unsloth_dense_qmm_scopes = 1
                    changed.append((module, fused))
        yield model
    finally:
        with _DENSE_QMM_LOCK:
            for module, fused in reversed(changed):
                module._unsloth_dense_qmm_scopes -= 1
                if not module._unsloth_dense_qmm_scopes:
                    if type(module) is fused:   # a NAX-routed linear keeps the NAX scope's class
                        module.__class__ = nn.QuantizedLinear
                    module.__dict__.pop("_unsloth_dense_qmm_scopes", None)


# The mlx-vlm bodies the verify scope wraps, and the projection kernels whose row bound its slices follow.
# 0.7.1 added the verifier's `verify` entry; earlier releases miss it and stay native.
_EXACT_VERIFY_CONTRACT = {
    "mlx_vlm.models.qwen3_5.speculative_verifier": {
        "Qwen3_5BatchInvariantForward.verify": "14bdd46f35e8adfa",
        "Qwen3_5BatchInvariantForward._linear": "6e38296bd3f89d5a",
        "Qwen3_5BatchInvariantForward._linears": "1f44efb5714938af",
        "Qwen3_5BatchInvariantForward._layer": "f3f59ff9d0b68a50",
        "Qwen3_5BatchInvariantForward._model": "142df8a42c27c4f7",
        # 0.7.5 stops the gradient on the routed indices, which inference never reads.
        "Qwen3_5BatchInvariantForward._feed_forward": ("d1c424074fbf0c21", "b04b66c926777a23"),
        "Qwen3_5BatchInvariantForward._switch_glu": "6aa1f4defb45f54a",
        "_target_verify_linear": "190517e318ba4d7b",
        "_target_verify_linears": "2d4c9adbc50be66e",
    },
    "mlx_vlm.speculative.ops.linear": {
        "_target_verify_quantized_linear": "539dce1a6026f87e",
        "_target_verify_optimized_affine_linear": "44955d8d8dd32a15",
        "_target_verify_quantized_linears": "447c68b85e4e7dda",
    },
    "mlx_vlm.models.quantized_verifier": {
        "_can_optimized_affine_linear": "22c2675d5a30e1f4",
        "supports_optimized_affine_head": "b0611c0de3000a20",
    },
}
# mlx-vlm's `optimized_affine_linears` serves 1 < rows <= 8; a wider block leaves its fused verify kernels.
_VERIFY_FUSED_ROWS = 8
_VERIFY_METHODS = ("verify", "_linear", "_linears", "_layer", "_feed_forward")
_VERIFY_LOCK = RLock()
_VERIFY_VERDICTS = {}
# The verifier is one object per process: whether a call is served is kept per thread, not on it.
_VERIFY_STATE = local()


def _verify_admitted(key, native, fused):
    """`fused()` once `key`'s first block matched `native()` bit for bit; that block, a failed key and a `fused()` of None take `native()`."""
    verdict = _VERIFY_VERDICTS.get(key)
    served = fused() if verdict else None
    if served is not None:
        return served
    reference = native()
    if verdict is None and (candidate := fused()) is not None:
        pairs = zip(reference, candidate) if isinstance(reference, tuple) else ((reference, candidate),)
        try:
            verdict = all(a.dtype == b.dtype and a.shape == b.shape
                          and mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item() for a, b in pairs)
        except (RuntimeError, ValueError):
            return reference  # inside a function transformation, which cannot evaluate; checked later
        _VERIFY_VERDICTS[key] = verdict
        if not verdict:
            logger.warning("a verify block differs from mlx-vlm's verifier for %s; the native verifier "
                           "stays in use there", key)
    return reference


def _verify_sliced(project, linears, x, eligible):
    """`project(x)`, over balanced row slices when the block is wider than mlx-vlm's fused verify bound."""
    if (not getattr(_VERIFY_STATE, "serving", False) or x.ndim != 3 or x.shape[1] <= _VERIFY_FUSED_ROWS
            or not all(eligible(linear, x) and not linear.trainable_parameters() for linear in linears)):
        return project(x)
    # As few slices as the bound allows and as equal as possible, so none is a one- or two-row tail.
    count = -(-x.shape[1] // _VERIFY_FUSED_ROWS)
    size, extra = divmod(x.shape[1], count)
    stops = [(i + 1) * size + min(i + 1, extra) for i in range(count)]

    def sliced():
        outs = [project(x[:, start:stop]) for start, stop in zip([0, *stops], stops)]
        if isinstance(outs[0], tuple):
            return tuple(mx.concatenate(column, axis = 1) for column in zip(*outs))
        return mx.concatenate(outs, axis = 1)

    key = (x.shape[:2], x.dtype, *((p.weight.shape, p.bits, p.group_size, p.mode, "bias" in p) for p in linears))
    return _verify_admitted(key, lambda: project(x), sliced)


@functools.cache
def _exact_verify_class(base, *cache_key):
    bindings = _resolved_bindings(_EXACT_VERIFY_CONTRACT)
    eligible = getattr(sys.modules.get("mlx_vlm.models.quantized_verifier"), "_can_optimized_affine_linear", None)
    if bindings is None or eligible is None:
        return None

    def verify(self, language_model, inputs, *, cache = None, **kwargs):
        outer = getattr(_VERIFY_STATE, "serving", False)
        _VERIFY_STATE.serving = bool(getattr(language_model, "_unsloth_verify_scopes", 0)
                                     and not language_model.training and _bindings_intact(bindings))
        try:
            return base.verify(self, language_model, inputs, cache = cache, **kwargs)
        finally:
            _VERIFY_STATE.serving = outer

    def sliced_linear(self, linear, x):
        return _verify_sliced(lambda rows: base._linear(self, linear, rows), (linear,), x, eligible)

    def sliced_linears(self, linears, x):
        return _verify_sliced(lambda rows: base._linears(self, linears, rows), linears, x, eligible)

    def submitted_layer(self, *args, **kwargs):
        hidden = base._layer(self, *args, **kwargs)
        if getattr(_VERIFY_STATE, "serving", False):
            try:
                mx.async_eval(hidden)
            except ValueError:  # not allowed inside a function transformation
                pass
        return hidden

    def routed_feed_forward(self, feed_forward, x):
        experts, mlp = getattr(feed_forward, "_unsloth_moe_routed", None), getattr(feed_forward, "switch_mlp", None)
        projections = [getattr(mlp, name, None) for name in ("gate_proj", "up_proj", "down_proj")]
        # The routed object refuses projections it was not built for; only quantized, frozen ones reach it.
        if (experts is None or x.ndim != 3 or not getattr(_VERIFY_STATE, "serving", False)
                or any(getattr(p, "bits", None) is None or p.trainable_parameters() for p in projections)):
            return base._feed_forward(self, feed_forward, x)

        def routed():
            # The verifier's own routing, shared expert and shared gate; the routed kernels take every row at once.
            gates = mx.softmax(self._linear(feed_forward.gate, x), axis = -1, precise = True)
            indices = mx.argpartition(gates, kth = -feed_forward.top_k, axis = -1)[..., -feed_forward.top_k:]
            scores = mx.take_along_axis(gates, indices, axis = -1)
            scores = scores / scores.sum(axis = -1, keepdims = True)
            shared = self._feed_forward(feed_forward.shared_expert, x)
            scale = mx.sigmoid(self._linear(feed_forward.shared_expert_gate, x))
            rows = (x.shape[0] * x.shape[1], 1, -1)
            out = experts(feed_forward, *(a.reshape(rows) for a in (x, indices, scores, scale, shared)))
            return None if out is None else out.reshape(*x.shape[:2], -1)

        key = (x.shape[:2], x.dtype, feed_forward.top_k,
               *((p.weight.shape, p.bits, p.group_size, p.mode) for p in projections))
        return _verify_admitted(key, lambda: base._feed_forward(self, feed_forward, x), routed)

    return type(f"_ScopedVerify{base.__name__}", (base,), {
        "verify": verify, "_linear": sliced_linear, "_linears": sliced_linears, "_layer": submitted_layer,
        "_feed_forward": routed_feed_forward, "_unsloth_verify_native": base})


@contextmanager
def exact_speculative_verify(model, *, _modules = None):
    """Speed up mlx-vlm's exact speculative verifier for Qwen3.5 language models during serialized inference.

    Each layer of a verify block is submitted for evaluation as it is built, so the GPU runs while the host builds on.
    A verify block wider than the eight rows mlx-vlm's fused verify kernels serve runs each frozen
    quantized projection over balanced row slices, once its first block of that shape matched the
    native call bit for bit. A sparse block's routed experts run on the decode kernels on the same condition. Only verify calls of the language models found here are served; other,
    training and distributed models, and an mlx-vlm whose verifier drifted, keep the native methods.
    """
    owners, verifier = [], None
    try:
        with _VERIFY_LOCK:
            language = sys.modules.get("mlx_vlm.models.qwen3_5.language")
            native = getattr(sys.modules.get("mlx_vlm.models.qwen3_5.speculative_verifier"),
                             "Qwen3_5BatchInvariantForward", None)
            call = getattr(getattr(language, "LanguageModel", None), "__call__", None)
            target = getattr(language, "_EXACT_SPECULATIVE_VERIFIER", None)
            if (call is not None and native is not None
                    and getattr(type(target), "_unsloth_verify_native", type(target)) is native
                    and not getattr(model, "_unsloth_mlx_distributed_parallel_mode", None)
                    and mx.metal.is_available()):
                # qwen3_5_moe inherits the call; a shared module is yielded once per path.
                found = {id(module): module for _, module in _fusion_modules(model, _modules)
                         if getattr(type(module), "__call__", None) is call}
                methods = [vars(native).get(name) for name in _VERIFY_METHODS]
                patched = _exact_verify_class(native, *methods) if found else None
                if patched is not None:
                    target.__class__ = patched
                    target._unsloth_verify_scopes = getattr(target, "_unsloth_verify_scopes", 0) + 1
                    verifier = target
                    for module in found.values():
                        module._unsloth_verify_scopes = getattr(module, "_unsloth_verify_scopes", 0) + 1
                        owners.append(module)
        yield model
    finally:
        with _VERIFY_LOCK:
            for owner in owners + ([verifier] if verifier is not None else []):
                owner._unsloth_verify_scopes -= 1
                if not owner._unsloth_verify_scopes:
                    owner.__dict__.pop("_unsloth_verify_scopes", None)
                    if owner is verifier:
                        verifier.__class__ = type(verifier)._unsloth_verify_native


# MLX's sdpa_vector_2pass_1/2 (mlx/backend/metal/kernels/sdpa_vector.h, Copyright © 2024 Apple Inc., MIT),
# reading strided keys and values in place. Rows attend end-aligned causally, as MLX's kernel does for a block.
_VECTOR_ATTENTION_PASS1 = """
    constexpr int BD = 32;
    constexpr int per_thread = D / BD;
    typedef float U;
    thread U q[per_thread];
    thread U o[per_thread] = {0};

    const int kv_head = threadgroup_position_in_grid.x;
    const int batch = threadgroup_position_in_grid.y;
    const int block = threadgroup_position_in_grid.z;
    const int gqa = threads_per_threadgroup.y;
    const int rows = threads_per_threadgroup.z;
    const int row = thread_position_in_threadgroup.z;
    const int head = gqa * kv_head + thread_position_in_threadgroup.y;
    const int out_row = (batch * threadgroups_per_grid.x * gqa + head) * rows + row;
    const uint lane = thread_index_in_simdgroup;
    const int N = keys_shape[2];

    const device T* qp = queries + batch * queries_strides[0] + head * queries_strides[1]
        + row * queries_strides[2] + lane * per_thread;
    const device T* kp = keys + batch * keys_strides[0] + kv_head * keys_strides[1]
        + block * keys_strides[2] + lane * per_thread;
    const device T* vp = values + batch * values_strides[0] + kv_head * values_strides[1]
        + block * values_strides[2] + lane * per_thread;
    for (int i = 0; i < per_thread; i++) {
        q[i] = static_cast<U>(scale) * qp[i];
    }

    U max_score = -metal::numeric_limits<U>::max();
    U sum_exp_score = 0;
    const int horizon = N - rows + row;
    for (int i = block; i < N; i += BLOCKS) {
        if (i <= horizon) {
            U score = 0;
            for (int j = 0; j < per_thread; j++) {
                score += q[j] * kp[j];
            }
            score = simd_sum(score);
            U new_max = max(max_score, score);
            U factor = metal::fast::exp(max_score - new_max);
            U exp_score = metal::fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (int j = 0; j < per_thread; j++) {
                o[j] = o[j] * factor + exp_score * vp[j];
            }
        }
        kp += BLOCKS * keys_strides[2];
        vp += BLOCKS * values_strides[2];
    }

    if (lane == 0) {
        sums[out_row * BLOCKS + block] = sum_exp_score;
        maxs[out_row * BLOCKS + block] = max_score;
    }
    device T* op = partials + (out_row * BLOCKS + block) * D + lane * per_thread;
    for (int i = 0; i < per_thread; i++) {
        op[i] = static_cast<T>(o[i]);
    }
"""

_VECTOR_ATTENTION_PASS2 = """
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int per_thread = D / BD;
    typedef float U;
    thread U o[per_thread] = {0};
    threadgroup U outputs[BN * BD];

    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int out_row = threadgroup_position_in_grid.x * threadgroups_per_grid.y + threadgroup_position_in_grid.y;
    const device T* pp = partials + out_row * BLOCKS * D + simd_gid * D + lane * per_thread;
    const device float* sp = sums + out_row * BLOCKS;
    const device float* mp = maxs + out_row * BLOCKS;

    U sum_exp_score = 0.0;
    U max_score = -metal::numeric_limits<U>::max();
    for (int b = 0; b < BLOCKS / BN; ++b) {
        max_score = max(max_score, mp[lane + BN * b]);
    }
    max_score = simd_max(max_score);
    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = metal::fast::exp(mp[lane + BN * b] - max_score);
        sum_exp_score += factor * sp[lane + BN * b];
    }
    sum_exp_score = simd_sum(sum_exp_score);
    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = metal::fast::exp(mp[simd_gid] - max_score);
        for (int i = 0; i < per_thread; i++) {
            o[i] += factor * static_cast<U>(pp[i]);
        }
        mp += BN;
        sp += BN;
        pp += BN * D;
    }
    for (int i = 0; i < per_thread; i++) {
        outputs[lane * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + lane]);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        device T* op = out + out_row * D + simd_gid * per_thread;
        for (int i = 0; i < per_thread; i++) {
            op[i] = static_cast<T>(o[i]);
        }
    }
"""


@functools.cache
def _vector_attention_kernels():
    try:
        if not mx.metal.is_available():
            return None
        return (
            mx.fast.metal_kernel(name = "unsloth_vector_attention_pass1",
                                 input_names = ["queries", "keys", "values", "scale"],
                                 output_names = ["partials", "sums", "maxs"],
                                 source = _VECTOR_ATTENTION_PASS1, ensure_row_contiguous = False),
            mx.fast.metal_kernel(name = "unsloth_vector_attention_pass2",
                                 input_names = ["partials", "sums", "maxs"], output_names = ["out"],
                                 source = _VECTOR_ATTENTION_PASS2),
        )
    except (AttributeError, TypeError):
        return None


@functools.cache
def _gpu_class():
    info = (getattr(mx, "device_info", None) or mx.metal.device_info)()
    return str(info.get("architecture", ""))[-1:]


@functools.cache
def _gpu_generation():
    info = (getattr(mx, "device_info", None) or mx.metal.device_info)()
    match = re.match(r"applegpu_g(\d+)", str(info.get("architecture", "")))
    return int(match.group(1)) if match else 0


def _launch_simds():
    # Before M3 (no dynamic caching), this kernel's pipeline allows 832 threads per threadgroup, not 1024.
    return 32 if _gpu_generation() >= 15 else 24


def _vector_blocks(simds, keys):
    """The key-interleave count MLX 0.32.3's sdpa_vector_2pass picks for this launch."""
    forced = int(os.environ.get("MLX_SDPA_BLOCKS", "0") or 0)
    if forced > 0:
        return (forced + 31) // 32 * 32
    gpu = _gpu_class()
    if gpu == "s":
        if keys > 1024 and simds > 4:
            return 128 if keys <= 8192 else 256 if keys <= 32768 else 512 if keys <= 65536 else 1024
        return 64
    if gpu == "d":
        if simds <= 2 and keys > 8192:
            return 256
        if simds >= 6 and keys >= 16384:
            return 512 if keys < 65536 else 1024
        return 128
    return 64 if simds >= 4 else 32


def _vector_attention(queries, keys, values, scale, blocks):
    pass1, pass2 = _vector_attention_kernels()
    batch, heads, rows, dim = queries.shape
    kv_heads = keys.shape[1]
    gqa, partial_rows = heads // kv_heads, batch * heads * rows * blocks
    partials, sums, maxs = pass1(
        inputs = [queries, keys, values, mx.array(scale, mx.float32)],
        template = [("T", queries.dtype), ("D", dim), ("BLOCKS", blocks)],
        grid = (32 * kv_heads, gqa * batch, rows * blocks), threadgroup = (32, gqa, rows),
        output_shapes = [(partial_rows, dim), (partial_rows,), (partial_rows,)],
        output_dtypes = [queries.dtype, mx.float32, mx.float32])
    return pass2(inputs = [partials, sums, maxs], template = [("T", queries.dtype), ("D", dim), ("BLOCKS", blocks)],
                 grid = (1024 * batch * heads, rows, 1), threadgroup = (1024, 1, 1),
                 output_shapes = [queries.shape], output_dtypes = [queries.dtype])[0]


def _decode_attention(queries, keys, values, scale):
    """Each row as a decode step attends it; at head dim 512 MLX puts a lone query on its vector kernel for
    GQA 8 only, so GQA-4 heads are paired up first and that row is MLX's vector kernel rather than decode."""
    _, heads, rows, dim = queries.shape
    kv_heads = keys.shape[1]
    repeat = 8 // (heads // kv_heads)
    outs = []
    for row in range(rows):
        query = queries[:, :, row : row + 1].reshape(1, kv_heads, -1, 1, dim)
        query = mx.concatenate([query] * repeat, axis = 2).reshape(1, -1, 1, dim)
        end = keys.shape[2] - rows + row + 1
        out = mx.fast.scaled_dot_product_attention(query, keys[..., :end, :], values[..., :end, :], scale = scale)
        outs.append(out.reshape(1, kv_heads, repeat, -1, 1, dim)[:, :, 0].reshape(1, heads, 1, dim))
    return mx.concatenate(outs, axis = 2)


# Query heads per KV head served by default: at 8 (E2B, 26B-A4B, 31B) MLX decodes a lone query on the same
# vector kernel, so each verify row is the attention a decode step computes. At 4 (E4B) decode stays unfused
# and the kernel only changes rounding, so it is opt-in with UNSLOTH_MLX_VERIFY_ATTENTION=1. At 16 (12B) the
# unfused attention is already as fast.
_VERIFY_ATTENTION_GQA = (8,)
_VERIFY_ATTENTION_OPT_IN_GQA = (4,)
_VERIFY_ATTENTION_DIM = 512
_VERIFY_ATTENTION_ROWS = 8
_VERIFY_ATTENTION_LOCK = RLock()
_VERIFY_ATTENTION_VERDICTS = {}


def _verify_attention_serves(gqa):
    return gqa in _VERIFY_ATTENTION_GQA or (
        gqa in _VERIFY_ATTENTION_OPT_IN_GQA and os.environ.get("UNSLOTH_MLX_VERIFY_ATTENTION", "0") == "1")


def _two_pass_keys(keys):
    # MLX's single-pass vector kernel serves fewer keys and computes them in another order.
    return keys >= (1024 if _gpu_class() in ("s", "d") else 4096)


def _verify_attention(queries, keys, values, cache, scale, mask, kv_cache):
    """A verify block's head-dim-512 causal attention on MLX's vector kernel, or None for the native call."""
    if not (isinstance(mask, str) and mask == "causal" and (cache is None or type(cache) is kv_cache)
            and queries.ndim == 4 and queries.shape[0] == 1 and keys.shape == values.shape
            and queries.shape[-1] == keys.shape[-1] == _VERIFY_ATTENTION_DIM
            and queries.dtype in (mx.bfloat16, mx.float16) and queries.dtype == keys.dtype == values.dtype
            and 2 <= queries.shape[2] <= min(_VERIFY_ATTENTION_ROWS, keys.shape[2])
            and keys.shape[1] and queries.shape[1] % keys.shape[1] == 0
            and _verify_attention_serves(queries.shape[1] // keys.shape[1])
            and _two_pass_keys(keys.shape[2] - queries.shape[2] + 1)
            and _vector_attention_kernels() is not None):
        return None
    rows, gqa = queries.shape[2], queries.shape[1] // keys.shape[1]
    step = _launch_simds() // gqa   # a launch holds gqa * rows simdgroups
    outs = []
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        horizon = keys.shape[2] - rows + stop
        chunk = (queries[:, :, start:stop], keys[..., :horizon, :], values[..., :horizon, :])
        blocks = _vector_blocks(gqa * (stop - start), horizon)
        if _vector_blocks(8, horizon - (stop - start) + 1) != blocks:
            return None   # decode steps on either side of this context interleave their keys differently
        key = (queries.dtype, gqa, stop - start, blocks)
        verdict = _VERIFY_ATTENTION_VERDICTS.get(key)
        if verdict is False:
            return None
        out = _vector_attention(*chunk, scale, blocks)
        if verdict is None:
            with _VERIFY_ATTENTION_LOCK:
                try:
                    verdict = mx.array_equal(out.view(mx.uint16), _decode_attention(*chunk, scale).view(mx.uint16)).item()
                except (RuntimeError, ValueError):
                    return None   # inside a function transformation, which cannot evaluate; checked later
                _VERIFY_ATTENTION_VERDICTS[key] = verdict
            if not verdict:
                logger.warning("verify attention differs from MLX's decode attention for %s; the native attention "
                               "stays in use there", key)
                return None
        outs.append(out)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis = 2)


# The gemma4 attention body whose call into mlx-vlm's attention helper the scope reroutes for verify blocks.
_VERIFY_ATTENTION_CONTRACT = {"mlx_vlm.models.gemma4.language": {"Attention.__call__": "7a040b99424f57fd"}}


@functools.cache
def _verify_attention_class(base):
    bindings = _resolved_bindings(_VERIFY_ATTENTION_CONTRACT)
    call = vars(base).get("__call__")
    if bindings is None or not any(target is call for _, _, target in bindings):
        return None
    namespace = call.__globals__
    native, kv_cache = namespace.get("scaled_dot_product_attention"), namespace.get("KVCache")
    if native is None or kv_cache is None:
        return None

    def attention(queries, keys, values, cache, scale, mask, sinks = None):
        out = None if sinks is not None else _verify_attention(queries, keys, values, cache, scale, mask, kv_cache)
        return native(queries, keys, values, cache = cache, scale = scale, mask = mask) if out is None else out

    routed = FunctionType(call.__code__, {**namespace, "scaled_dot_product_attention": attention},
                          call.__name__, call.__defaults__, call.__closure__)
    routed.__kwdefaults__ = call.__kwdefaults__

    def verify_call(self, *args, **kwargs):
        if (getattr(_VERIFY_STATE, "attention", False) and not self.training
                and namespace.get("scaled_dot_product_attention") is native and _bindings_intact(bindings)):
            return routed(self, *args, **kwargs)
        return call(self, *args, **kwargs)

    return type(f"_VerifyAttention{base.__name__}", (base,),
                {"__call__": verify_call, "_unsloth_verify_attention_native": base})


@contextmanager
def verify_block_attention():
    """Mark the forwards built inside as a speculative verify block for `fused_verify_attention`."""
    outer = getattr(_VERIFY_STATE, "attention", False)
    _VERIFY_STATE.attention = True
    try:
        yield
    finally:
        _VERIFY_STATE.attention = outer


@contextmanager
def fused_verify_attention(model, *, _modules = None):
    """Run Gemma 4's head-dim-512 attention for a speculative verify block on MLX's vector kernel.

    MLX 0.32.3 sends 2 to 8 query rows at head dim 512 to an unfused attention whose cost dwarfs a decode
    step's at long context. Inside `verify_block_attention`, an eligible single-sequence causal block with 8
    query heads per KV head runs on a copy of MLX's own two-pass vector kernel instead, once its first launch
    of that shape matched, row by row and bit for bit, the attention a decode step computes; 4 query heads per
    KV head join with `UNSLOTH_MLX_VERIFY_ATTENTION=1`. Other calls, training and distributed models keep the
    native attention. Instance classes are restored when the context exits.
    """
    changed = []
    try:
        with _VERIFY_ATTENTION_LOCK:
            if (not getattr(model, "_unsloth_mlx_distributed_parallel_mode", None)
                    and _vector_attention_kernels() is not None):
                for _, module in _fusion_modules(model, _modules):
                    # Type before `training`: named_modules() may yield plain stand-ins.
                    if not isinstance(module, dict) or module.training:
                        continue
                    base = type(module)
                    if getattr(base, "_unsloth_verify_attention_native", None) is not None:
                        if module.__dict__.get("_unsloth_verify_attention_scopes", 0):
                            module._unsloth_verify_attention_scopes += 1
                            changed.append(module)
                        continue
                    if getattr(module, "head_dim", None) != _VERIFY_ATTENTION_DIM:
                        continue
                    patched = _verify_attention_class(base)
                    if patched is not None:
                        module.__class__ = patched
                        module._unsloth_verify_attention_scopes = 1
                        changed.append(module)
        yield model
    finally:
        with _VERIFY_ATTENTION_LOCK:
            for module in reversed(changed):
                scopes = module.__dict__.get("_unsloth_verify_attention_scopes", 0)
                if scopes > 1:
                    module._unsloth_verify_attention_scopes = scopes - 1
                    continue
                module.__dict__.pop("_unsloth_verify_attention_scopes", None)
                native = getattr(type(module), "_unsloth_verify_attention_native", None)
                if native is not None:
                    module.__class__ = native
