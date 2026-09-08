# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The package must be importable on the Python floor ``pyproject.toml`` declares.

It was not. ``requires-python`` said ``>=3.9`` while ``__init__.py`` annotated
``log_dir: str | Path`` at module level, so ``import unsloth_zoo`` died with
``TypeError: unsupported operand type(s) for |`` before reaching anything else, and
``temporary_patches/gpt_oss.py`` used a ``match`` statement that 3.9 cannot even parse.
Nothing caught it: no CI job runs the floor, and the whole matrix starts at 3.10.

The floor is read from ``pyproject.toml`` rather than hardcoded, so raising
``requires-python`` relaxes these checks instead of turning them into a false alarm.

Two exclusions, both asserted below rather than trusted:
``_vendored/fla`` is skipped entirely under 3.10 by ``fla_vendor.py``, and the lazy
``mlx`` modules cannot run on the floor because ``mlx`` itself requires 3.10+.
"""

import ast
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "unsloth_zoo"

# Imported eagerly by unsloth_zoo/__init__.py, so they must hold on the floor even
# though the rest of mlx/ is lazy. Package-relative, since mlx/cce/__init__.py shares
# a file name with mlx/__init__.py but is lazy like the rest of the subpackage.
MLX_IMPORT_TIME = {"mlx/__init__.py", "mlx/runtime.py"}


def declared_floor():
    """The ``>=X.Y`` in requires-python, as a tuple for ast.parse(feature_version=)."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"^requires-python\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
    assert match, "no requires-python in pyproject.toml"
    floor = re.search(r">=\s*(\d+)\.(\d+)", match.group(1))
    assert floor, f"no >= lower bound in requires-python = {match.group(1)!r}"
    return int(floor.group(1)), int(floor.group(2))


def in_scope(path):
    """Files that must work on the floor, with the two exclusions applied."""
    rel = path.relative_to(PACKAGE_ROOT)
    if rel.parts[0] == "_vendored":
        return False
    if rel.parts[0] == "mlx" and rel.as_posix() not in MLX_IMPORT_TIME:
        return False
    return True


def scoped_files():
    files = sorted(p for p in PACKAGE_ROOT.rglob("*.py")
                   if "__pycache__" not in p.parts and in_scope(p))
    assert len(files) > 50, f"only found {len(files)} files, the glob is wrong"
    return files


def signature_annotations(node):
    """Parameter and return annotations, evaluated when the ``def`` executes."""
    args = node.args
    out = [a.annotation for a in
           [*args.args, *args.kwonlyargs, *args.posonlyargs, args.vararg, args.kwarg]
           if a is not None and a.annotation is not None]
    if node.returns:
        out.append(node.returns)
    return out


def evaluated_annotations(tree):
    """Annotations Python evaluates, so the future import is what defers them.

    Variable annotations are evaluated at module and class scope only; inside a
    function body they are never evaluated, so flagging those is a false positive.
    A class body nested in a function is still class scope, so it goes back to
    being evaluated. Signature annotations are evaluated wherever their ``def`` is.
    """
    out = []

    def walk(node, in_function):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.AnnAssign) and child.annotation and not in_function:
                out.append(child.annotation)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.extend(signature_annotations(child))
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                walk(child, True)
            elif isinstance(child, ast.ClassDef):
                walk(child, False)
            else:
                walk(child, in_function)

    walk(tree, False)
    return out


# A `|` between these is a union, not arithmetic: builtin types, `None`, and whatever
# the module pulled in from typing.
TYPE_ANCHORS = frozenset({
    "str", "int", "float", "bool", "bytes", "bytearray", "complex", "object", "type",
    "list", "dict", "tuple", "set", "frozenset",
})
TYPING_MODULES = frozenset({"typing", "typing_extensions", "collections.abc"})


def typing_names(tree):
    """Names this module bound with ``from typing import X`` and friends."""
    return {alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module in TYPING_MODULES
            for alias in node.names}


def union_operands(node):
    """Operands of an ``A | B | C`` chain, or None if any part is not name-shaped.

    ``str``, ``os.PathLike``, ``List[int]``, ``None`` and a ``"ForwardRef"`` string
    are name-shaped; a call, a set/dict/list display or a number is not.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, right = union_operands(node.left), union_operands(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.Subscript):
        return union_operands(node.value)
    if isinstance(node, (ast.Name, ast.Attribute)):
        return [node]
    if isinstance(node, ast.Constant) and (node.value is None
                                           or isinstance(node.value, str)):
        return [node]
    return None


def looks_like_a_type_alias(node, known_typing_names):
    """``PathLike = str | Path`` yes; ``defaults | extra`` and ``re.A | re.M`` no.

    Every operand must be name-shaped and at least one must be a recognisable type.
    Without that anchor a ``|`` between plain names is far more likely to be a dict
    merge (PEP 584, valid on 3.9), a set union or flag arithmetic, and failing the
    gate on those would block code that runs perfectly well on the floor.
    """
    operands = union_operands(node)
    if not operands:
        return False
    for operand in operands:
        if isinstance(operand, ast.Constant):
            if operand.value is None:
                return True
            continue  # a bare string is a forward reference, never an anchor alone
        name = operand.id if isinstance(operand, ast.Name) else operand.attr
        if name in TYPE_ANCHORS or name in known_typing_names:
            return True
    return False


def evaluated_values(tree):
    """Expressions that run at import, at module or class scope.

    Type aliases are the case that matters: ``PathLike = str | Path`` raises below
    3.10 and, unlike an annotation, the future import does not defer it. Decorator
    expressions and parameter defaults are evaluated the same way, so they belong
    here too. Control flow is descended into (a branch that runs, runs) but function
    bodies are not, since those only execute when called.
    """
    out = []
    scoped = (ast.If, ast.Try, ast.With, ast.AsyncWith, ast.For, ast.AsyncFor,
              ast.While, ast.ExceptHandler, ast.ClassDef)

    def walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.extend(child.decorator_list)
                out.extend(d for d in child.args.defaults if d is not None)
                out.extend(d for d in child.args.kw_defaults if d is not None)
                continue  # the body only runs when called
            if isinstance(child, ast.ClassDef):
                out.extend(child.decorator_list)
            if isinstance(child, (ast.Assign, ast.AnnAssign)) and child.value is not None:
                out.append(child.value)
            if isinstance(child, ast.Expr):
                out.append(child.value)   # a bare call runs on import too
            if isinstance(child, scoped):
                walk(child)

    walk(tree)
    return out


def union_nodes(expression):
    """``BinOp(BitOr)`` nodes in an annotation, skipping ``Literal[...]`` contents.

    ``Literal[re.I | re.M]`` is flag arithmetic over values, evaluates fine on 3.9,
    and must not be read as a union.
    """
    if isinstance(expression, ast.Subscript):
        base = expression.value
        name = getattr(base, "id", None) or getattr(base, "attr", None)
        if name == "Literal":
            return union_nodes(base)
        return union_nodes(base) + union_nodes(expression.slice)
    found = []
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.BitOr):
        found.append(expression)
    for child in ast.iter_child_nodes(expression):
        found.extend(union_nodes(child))
    return found


def has_future_annotations(tree):
    return any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
               and any(alias.name == "annotations" for alias in n.names)
               for n in tree.body)


def test_every_module_parses_on_the_declared_floor():
    floor = declared_floor()
    broken = []
    for path in scoped_files():
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path),
                      feature_version=floor)
        except SyntaxError as error:
            broken.append(f"{path.relative_to(REPO_ROOT)}:{error.lineno}: {error.msg}")
    assert not broken, (
        "syntax newer than the declared floor "
        f"{floor[0]}.{floor[1]}; either rewrite it or raise requires-python:\n  "
        + "\n  ".join(broken)
    )


def test_every_module_compiles():
    """``ast.parse`` accepts things ``compile`` rejects, and only compile runs on import.

    A misplaced ``from __future__ import annotations`` is the case that bites here: it
    parses, it makes ``has_future_annotations`` suppress the union check, and it still
    raises SyntaxError on import.
    """
    broken = []
    for path in scoped_files():
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec",
                    dont_inherit=True)
        except SyntaxError as error:
            broken.append(f"{path.relative_to(REPO_ROOT)}:{error.lineno}: {error.msg}")
    assert not broken, "these modules do not compile:\n  " + "\n  ".join(broken)


def test_no_pep604_unions_are_evaluated_on_the_declared_floor():
    """``X | Y`` is a TypeError below 3.10 unless the module defers it."""
    if declared_floor() >= (3, 10):
        pytest.skip("floor is 3.10+, PEP 604 evaluates fine")
    offenders = []
    for path in scoped_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        where = path.relative_to(REPO_ROOT)
        known_typing_names = typing_names(tree)
        # The future import defers annotations only; an assigned value still runs.
        deferred = has_future_annotations(tree)
        for expression in ([] if deferred else evaluated_annotations(tree)):
            for inner in union_nodes(expression):
                offenders.append(
                    f"{where}:{inner.lineno}: {ast.unparse(inner)} (annotation)")
        for expression in evaluated_values(tree):
            for inner in union_nodes(expression):
                if looks_like_a_type_alias(inner, known_typing_names):
                    offenders.append(
                        f"{where}:{inner.lineno}: {ast.unparse(inner)} (type alias)")
    assert not offenders, (
        "PEP 604 unions evaluated below 3.10. Annotations are deferred by `from "
        "__future__ import annotations`; type aliases need typing.Union, which that "
        "import does NOT defer:\n  " + "\n  ".join(sorted(set(offenders)))
    )


def _is_floor_version_check(node):
    """``sys.version_info < (3, 10)``, as an expression."""
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Lt)):
        return False
    left = node.left
    if not (isinstance(left, ast.Attribute) and left.attr == "version_info"):
        return False
    right = node.comparators[0]
    return (isinstance(right, ast.Tuple) and len(right.elts) == 2
            and [getattr(e, "value", None) for e in right.elts] == [3, 10])


def test_vendored_fla_stays_gated_below_310():
    """Excluding _vendored/fla is sound only while this guard bails out on the floor.

    Asserted structurally, not by grepping: a commented-out or dead-code check would
    satisfy a text search while the injection went ahead anyway.
    """
    path = PACKAGE_ROOT / "temporary_patches" / "fla_vendor.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    guard = next((n for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == "_torch_triton_cuda_supported"), None)
    assert guard is not None, (
        "_torch_triton_cuda_supported is gone from fla_vendor.py; this test pins the "
        "guard that lets _vendored/fla stay out of scope, so re-point it or drop that "
        "exclusion in in_scope()"
    )
    bails_out = any(
        _is_floor_version_check(node.test)
        and any(isinstance(inner, ast.Return)
                and getattr(inner.value, "value", None) is False
                for inner in ast.walk(node))
        for node in ast.walk(guard) if isinstance(node, ast.If)
    )
    assert bails_out, (
        "no `if sys.version_info < (3, 10): return False` controls "
        "_torch_triton_cuda_supported, so the vendored FLA kernels can now be injected "
        "on the floor - restore the guard or bring _vendored/fla into in_scope()"
    )

    # The guard only matters if the injection path actually consults it.
    injectors = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and any(isinstance(c, ast.Call)
                         and getattr(c.func, "id", None) == "_inject_vendored_fla"
                         for c in ast.walk(n))]
    assert injectors, "nothing calls _inject_vendored_fla; re-point this test"
    for caller in injectors:
        consults = any(isinstance(c, ast.Call)
                       and getattr(c.func, "id", None) == "_torch_triton_cuda_supported"
                       for c in ast.walk(caller))
        assert consults, (
            f"{caller.name} injects the vendored FLA kernels without consulting "
            "_torch_triton_cuda_supported, so the <3.10 guard no longer gates injection "
            "and excluding _vendored/fla from this gate is unsound"
        )


def strict_patch_calls(tree):
    """``patch_function(...)`` calls that keep the default ``match_level="strict"``."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != "patch_function":
            continue
        if not any(kw.arg == "match_level" for kw in node.keywords):
            out.append(node)
    return out


def test_modules_with_strict_patches_do_not_defer_annotations():
    """Deferring annotations is not a safe way to reach the floor in a patching module.

    ``can_safely_patch`` compares ``inspect.signature(...).annotation`` objects directly,
    with no ``get_type_hints``, so under the default strict match level a stringified
    annotation never equals the live upstream one. ``patch_function`` then declines the
    patch and only logs it, which is silent in normal use. Reach the floor with
    ``typing.Optional`` / ``typing.Union`` in these modules instead.
    """
    offenders = []
    for path in scoped_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not has_future_annotations(tree):
            continue
        for call in strict_patch_calls(tree):
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{call.lineno}")
    assert not offenders, (
        "`from __future__ import annotations` in a module that calls patch_function at "
        "the default strict match level; the patch will be silently skipped. Use "
        "typing.Optional / typing.Union for the annotations instead:\n  "
        + "\n  ".join(sorted(set(offenders)))
    )


def test_mlx_modules_reachable_at_import_are_in_scope():
    """mlx/ is excluded as lazy, but these two are imported by unsloth_zoo/__init__.py."""
    for relative in sorted(MLX_IMPORT_TIME):
        path = PACKAGE_ROOT / relative
        assert path.exists(), f"{path} vanished; update MLX_IMPORT_TIME"
        assert in_scope(path), f"{relative} must stay in scope, it is imported eagerly"
