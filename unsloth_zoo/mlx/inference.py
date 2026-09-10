# SPDX-License-Identifier: AGPL-3.0-only
"""Scoped MLX recurrent decode convolution and SiLU fusion."""

import ast
import copy
import functools
import inspect
import textwrap
from contextlib import contextmanager
from types import FunctionType

import mlx.core as mx
import mlx.nn as nn


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


@functools.cache
def _function_ast(call):
    return ast.parse(textwrap.dedent(inspect.getsource(call))).body[0]


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
        if any(hasattr(f, "__wrapped__") for f in (call, prepare, conv, nn.silu)):
            return None
        arithmetic = _function_ast(conv)
        if [_source_expression(n) for n in arithmetic.body] != [
            "out = mx.sum(conv_input.astype(mx.float32) * weight[None, :, :], axis=1)",
            "return out.astype(conv_input.dtype)[:, None, :]",
        ] or inspect.unwrap(conv).__globals__.get("mx") is not mx:
            return None
        if [_source_expression(n) for n in _function_ast(nn.silu).body if not isinstance(n, ast.Expr)] != ["return x * mx.sigmoid(x)"]:
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
                or nn.silu is not silu):
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
