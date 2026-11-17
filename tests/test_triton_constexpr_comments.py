# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Every tl.constexpr parameter of an unsloth_zoo Triton kernel (vendored code excluded) says what it is on its own line, and no
kernel takes a per-call size (rows, tokens, sequence length, batch) as a constexpr: each distinct
constexpr value compiles a new kernel, so a size that changes per call recompiles per call."""

import ast
import io
import pathlib
import tokenize

ROOT = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo"
# Vendored upstream kernels (fla, vllm) keep their upstream source.
SKIP_DIRS = {"_vendored", "__pycache__"}
KERNEL_DECORATORS = {"jit", "autotune", "heuristics"}
# Names that only ever mean a per-call size in this codebase.
PER_CALL_NAMES = {
    "M",
    "T",
    "n_rows",
    "num_rows",
    "seq_len",
    "seqlen",
    "q_len",
    "kv_len",
    "batch",
    "bsz",
    "num_tokens",
    "n_tokens",
}


def _dotted(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):  # quoted annotation
        return node.value
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_kernel_decorator(node):
    name = _dotted(node.func if isinstance(node, ast.Call) else node)
    last = name.split(".")[-1]
    # jit_any_alignment(...) / _jit_any_alignment: triton.jit wrappers with specialization options.
    return (last in KERNEL_DECORATORS and (name.startswith("triton") or name == last)) or last.lstrip(
        "_"
    ).startswith("jit_")


def _kernels(tree):
    """FunctionDefs decorated as Triton kernels, or wrapped as `k = triton.jit(f)`."""
    # Keyed by node: two factories may each define a kernel with the same name.
    defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    found = {id(n): n for n in defs if any(_is_kernel_decorator(d) for d in n.decorator_list)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Name):
            outer = node.func.func if isinstance(node.func, ast.Call) else node.func
            if _is_kernel_decorator(outer):
                found.update({id(n): n for n in defs if n.name == node.args[0].id})
    return found.values()


def _constexpr_params(path):
    source = path.read_text(encoding = "utf-8")
    tree = ast.parse(source)
    comments = {
        tok.start[0]
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type == tokenize.COMMENT
    }
    for fn in _kernels(tree):
        for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
            if arg.annotation is not None and _dotted(arg.annotation).endswith("constexpr"):
                yield fn.name, arg.arg, arg.lineno, arg.lineno in comments


def _files():
    for path in sorted(ROOT.rglob("*.py")):
        if SKIP_DIRS.isdisjoint(path.relative_to(ROOT).parts):
            text = path.read_text(encoding = "utf-8")
            if "constexpr" in text:
                yield path


def test_finds_kernels():
    # Guards the walk itself: a refactor that hides every kernel would pass the checks below vacuously.
    params = [p for path in _files() for p in _constexpr_params(path)]
    assert len(params) > 150
    assert {path.name for path in _files()} >= {"moe_routed.py", "fused_linear.py", "mxfp4_gemm.py"}


def test_every_constexpr_line_has_a_comment():
    missing = [
        f"{path.relative_to(ROOT.parent)}:{line} {kernel}.{name}"
        for path in _files()
        for kernel, name, line, commented in _constexpr_params(path)
        if not commented
    ]
    assert not missing, "tl.constexpr parameters without a comment:\n" + "\n".join(missing)


def test_no_per_call_size_is_constexpr():
    bad = [
        f"{path.relative_to(ROOT.parent)}:{line} {kernel}.{name}"
        for path in _files()
        for kernel, name, line, _ in _constexpr_params(path)
        if name in PER_CALL_NAMES
    ]
    assert not bad, "per-call sizes must be runtime arguments:\n" + "\n".join(bad)


def test_scanner_sees_same_name_positional_only_and_quoted(tmp_path):
    src = tmp_path / "k.py"
    src.write_text(
        "import triton\nimport triton.language as tl\n\n"
        "def make_a():\n"
        "    @triton.jit\n"
        "    def kernel(x, BLOCK: tl.constexpr):\n"
        "        pass\n\n"
        "def make_b():\n"
        "    @triton.jit\n"
        "    def kernel(x, BLOCK: tl.constexpr):  # elements per program\n"
        "        pass\n\n"
        "@triton.jit\n"
        "def pos(n_rows: tl.constexpr, /, x, Q: \"tl.constexpr\"):  # rows\n"
        "    pass\n",
        encoding = "utf-8",
    )
    found = sorted((k, n, line, c) for k, n, line, c in _constexpr_params(src))
    assert found == [
        ("kernel", "BLOCK", 6, False),
        ("kernel", "BLOCK", 11, True),
        ("pos", "Q", 15, True),
        ("pos", "n_rows", 15, True),
    ]
