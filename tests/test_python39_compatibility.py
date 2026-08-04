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
# though the rest of mlx/ is lazy.
MLX_IMPORT_TIME = {"__init__.py", "runtime.py"}


def declared_floor():
    """The ``>=X.Y`` in requires-python, as a tuple for ast.parse(feature_version=)."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"^requires-python\s*=\s*[\"']([^\"']+)[\"']", text, re.M)
    assert match, "no requires-python in pyproject.toml"
    floor = re.search(r">=\s*(\d+)\.(\d+)", match.group(1))
    assert floor, f"no >= lower bound in requires-python = {match.group(1)!r}"
    return int(floor.group(1)), int(floor.group(2))


def in_scope(path):
    """Files that must work on the floor, with the two exclusions applied."""
    rel = path.relative_to(PACKAGE_ROOT).parts
    if rel[0] == "_vendored":
        return False
    if rel[0] == "mlx" and path.name not in MLX_IMPORT_TIME:
        return False
    return True


def scoped_files():
    files = sorted(p for p in PACKAGE_ROOT.rglob("*.py")
                   if "__pycache__" not in p.parts and in_scope(p))
    assert len(files) > 50, f"only found {len(files)} files, the glob is wrong"
    return files


def annotations_of(node):
    """Annotation expressions that this node evaluates at def/exec time."""
    if isinstance(node, ast.AnnAssign):
        return [node.annotation] if node.annotation else []
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    args = node.args
    out = [a.annotation for a in
           [*args.args, *args.kwonlyargs, *args.posonlyargs, args.vararg, args.kwarg]
           if a is not None and a.annotation is not None]
    if node.returns:
        out.append(node.returns)
    return out


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


def test_no_pep604_unions_are_evaluated_on_the_declared_floor():
    """`X | Y` is a TypeError below 3.10 unless the module defers annotations."""
    if declared_floor() >= (3, 10):
        pytest.skip("floor is 3.10+, PEP 604 evaluates fine")
    offenders = []
    for path in scoped_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if has_future_annotations(tree):
            continue
        for node in ast.walk(tree):
            for annotation in annotations_of(node):
                for inner in ast.walk(annotation):
                    if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.BitOr):
                        offenders.append(
                            f"{path.relative_to(REPO_ROOT)}:{inner.lineno}: "
                            f"{ast.unparse(inner)}"
                        )
    assert not offenders, (
        "PEP 604 unions evaluated at def time; add `from __future__ import "
        "annotations` to these modules:\n  " + "\n  ".join(sorted(set(offenders)))
    )


def test_vendored_fla_stays_gated_below_310():
    """The exclusion above is only sound while this guard skips the injection."""
    source = (PACKAGE_ROOT / "temporary_patches" / "fla_vendor.py").read_text(encoding="utf-8")
    assert re.search(r"sys\.version_info\s*<\s*\(\s*3\s*,\s*10\s*\)", source), (
        "the <3.10 guard in fla_vendor.py is gone, so the vendored kernels can now be "
        "injected on the floor - either restore it or bring _vendored/fla into scope here"
    )


def test_mlx_modules_reachable_at_import_are_in_scope():
    """mlx/ is excluded as lazy, but these two are imported by unsloth_zoo/__init__.py."""
    for name in sorted(MLX_IMPORT_TIME):
        path = PACKAGE_ROOT / "mlx" / name
        assert path.exists(), f"{path} vanished; update MLX_IMPORT_TIME"
        assert in_scope(path), f"{name} must stay in scope, it is imported eagerly"
