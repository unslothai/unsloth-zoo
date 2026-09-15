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

"""Scoped MLX recurrent decode convolution and SiLU fusion."""

import ast
import copy
import functools
import hashlib
import inspect
import logging
import re
import sys
import textwrap
from contextlib import contextmanager
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
