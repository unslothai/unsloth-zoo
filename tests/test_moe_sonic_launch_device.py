# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The sonic-moe dgated Triton kernel launches under its operand's device (CPU only, AST): Triton
launches on the current device, which is not the operand's when experts are spread with device_map."""

import ast
import pathlib

SONIC = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo" / "temporary_patches" / "moe_utils_sonic.py"
KERNEL = "_swiglu_bwd_interleaved_kernel"


def _launches(tree):
    """(launch call, enclosing `with` context expressions) for every `KERNEL[grid](...)`."""
    out = []

    def visit(node, ctx):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            ctx = ctx + [item.context_expr for item in node.items]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == KERNEL
        ):
            out.append((node, ctx))
        for child in ast.iter_child_nodes(node):
            visit(child, ctx)

    visit(tree, [])
    return out


def _under_operand_device(ctx):
    for expr in ctx:
        if (
            isinstance(expr, ast.Call)
            and ast.unparse(expr.func) == "torch.cuda.device"
            and expr.args
            and isinstance(expr.args[0], ast.Attribute)
            and expr.args[0].attr == "device"
        ):
            return True
    return False


def test_dgated_kernel_launches_under_the_operand_device():
    launches = _launches(ast.parse(SONIC.read_text(encoding = "utf-8")))
    assert launches, f"no {KERNEL}[grid](...) launch found"
    for call, ctx in launches:
        assert _under_operand_device(ctx), f"{KERNEL} launch at line {call.lineno} is not under torch.cuda.device(x.device)"


def test_the_check_can_fail():
    bare = "def f(x):\n    _swiglu_bwd_interleaved_kernel[(1,)](x)\n"
    wrapped = "def f(x):\n    with torch.cuda.device(x.device):\n        _swiglu_bwd_interleaved_kernel[(1,)](x)\n"
    assert not _under_operand_device(_launches(ast.parse(bare))[0][1])
    assert _under_operand_device(_launches(ast.parse(wrapped))[0][1])
