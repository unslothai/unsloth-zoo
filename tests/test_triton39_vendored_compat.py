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

"""Triton 3.9 (torch 2.15) forward-compat guards for zoo and vendored fla Triton code.

Triton 3.9's Autotuner raises ValueError at decoration time (module import) when ``key`` /
``reset_to_zero`` / ``restore_value`` name something that is not a kernel argument; older Triton
silently ignored it. These checks are static, so they also hold on CI hosts with older Triton.
"""

import ast
import importlib
import pathlib

import pytest

ZOO = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo"
_NAME_LISTS = ("key", "reset_to_zero", "restore_value")


def _trees():
    for path in sorted(ZOO.rglob("*.py")):
        yield path.relative_to(ZOO).as_posix(), ast.parse(path.read_text(encoding = "utf-8"))


def _autotune_name_lists():
    for rel, tree in _trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            a = node.args
            params = {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs}
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and ast.unparse(dec.func).endswith("autotune"):
                    for kw in dec.keywords:
                        if kw.arg in _NAME_LISTS:
                            yield rel, node.name, dec.lineno, kw.arg, kw.value, params


def test_autotune_names_are_kernel_arguments():
    checked, bad = 0, []
    for rel, fn, line, arg, value, params in _autotune_name_lists():
        if isinstance(value, ast.Constant) and value.value is None:
            continue
        assert isinstance(value, (ast.List, ast.Tuple)), f"{rel}:{line} {arg}= is not a literal list"
        checked += 1
        for elt in value.elts:
            if isinstance(elt, ast.Constant) and elt.value not in params:
                bad.append(f"{rel}:{line} {fn} {arg}={elt.value!r}")
    assert checked > 50, checked  # the vendored fla tree alone has ~60
    assert not bad, "Triton 3.9 raises at import for: " + "; ".join(bad)


def _triton_aliases(tree):
    alias = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "triton":
                    alias[a.asname or "triton"] = a.name if a.asname else "triton"
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] == "triton":
            for a in node.names:
                alias[a.asname or a.name] = f"{node.module}.{a.name}"
    return alias


def _resolve(dotted):
    parts = dotted.split(".")
    obj = importlib.import_module(parts[0])
    for i, name in enumerate(parts[1:], start = 1):
        if isinstance(getattr(type(obj), name, None), property):
            return True  # lazy property needing a GPU driver (DriverConfig.active): cannot go deeper here
        if hasattr(obj, name):
            obj = getattr(obj, name)
            continue
        try:
            obj = importlib.import_module(".".join(parts[: i + 1]))
        except Exception:
            return False
    return True


# Probed with hasattr() before use (fla ops/utils/op.py, utils/_device.py), or Unsloth's own marker.
_GUARDED = {
    "triton.language._experimental_make_tensor_descriptor",
    "triton._unsloth_allocator_set",
}


def test_triton_attributes_resolve_on_installed_triton():
    triton = pytest.importorskip("triton")
    if tuple(int(x) for x in triton.__version__.split(".")[:2]) < (3, 6):
        # Older Triton reaches newer names (triton.knobs, libdevice.fast_tanhf) only behind version checks.
        pytest.skip(f"forward-compat check targets Triton >= 3.6, found {triton.__version__}")
    missing = set()
    for rel, tree in _trees():
        alias = _triton_aliases(tree)
        if not alias:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            chain, base = [], node
            while isinstance(base, ast.Attribute):
                chain.append(base.attr)
                base = base.value
            if isinstance(base, ast.Name) and base.id in alias:
                dotted = ".".join([alias[base.id], *reversed(chain)])
                if dotted not in _GUARDED and not any(dotted.startswith(g + ".") for g in _GUARDED):
                    if not _resolve(dotted):
                        missing.add(f"{rel}: {dotted}")
        for name in alias.values():
            if name not in _GUARDED and not _resolve(name):
                missing.add(f"{rel}: {name}")
    assert not missing, sorted(missing)
