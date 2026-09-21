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

"""Where the GGUF read-back tree may come from.

Post-conversion verification runs on every export by default, and to read a file
back with the `gguf` that wrote it the parent puts a directory at its own
sys.path[0] and imports it. That directory is derived from a path string the
preflight probe CHILD reported, and `python -c` puts the WORKING DIRECTORY at the
child's sys.path[0], so a `gguf` package anyone can drop in the CWD both wins the
probe and lands in the training process, which holds the user's token.

Two things are pinned here: the child no longer takes the working directory as a
search path, and the parent accepts a reported tree only from a root Unsloth
itself chose. Neither may cost an ordinary export the writer's own gguf-py.
"""

from __future__ import annotations

import importlib.util

import os

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope = "module")
def llama_cpp():
    """Loaded by path: importing the package would run its device detection."""
    module_path = REPO_ROOT / "unsloth_zoo" / "llama_cpp.py"
    spec = importlib.util.spec_from_file_location("llama_cpp_readback_provenance", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _gguf_package(directory, body = "# gguf\n"):
    """A well-formed, importable `gguf` package under `directory`. Returns the tree."""
    package = Path(directory) / "gguf"
    package.mkdir(parents = True, exist_ok = True)
    (package / "__init__.py").write_text(body, encoding = "utf-8")
    return str(directory)


# --- The parent's sys.path takes trees Unsloth chose, not trees a child names ---

def test_a_tree_outside_every_chosen_root_is_refused(llama_cpp, tmp_path):
    """The planted package answers every structural check; only provenance separates
    it from the real one."""
    planted = _gguf_package(tmp_path / "writable-cwd")
    assert llama_cpp._gguf_tree_of_location(
        os.path.join(planted, "gguf", "__init__.py"),
    ) == planted, "the shape check is supposed to accept it; provenance is the gate"
    assert llama_cpp._trusted_gguf_tree(planted) is None
    assert llama_cpp._trusted_gguf_tree(planted, converter_location = None) is None


def test_a_relative_sys_path_entry_does_not_launder_the_working_directory(
        llama_cpp, tmp_path, monkeypatch):
    """'' and '.' are on sys.path in an interactive parent and resolve to the CWD,
    which is the entry the rule exists to exclude."""
    planted = _gguf_package(tmp_path / "cwd")
    monkeypatch.chdir(tmp_path / "cwd")
    monkeypatch.syspath_prepend("")
    sys.path.insert(0, ".")
    try:
        assert llama_cpp._trusted_gguf_tree(planted) is None
    finally:
        sys.path.remove(".")


@pytest.mark.parametrize("root", ["default_dir", "scripts_pin", "converter", "sys_path"])
def test_a_tree_inside_a_chosen_root_is_kept(llama_cpp, tmp_path, monkeypatch, root):
    """The ordinary export must still read back with the gguf that wrote the file."""
    install = tmp_path / "install"
    tree = _gguf_package(install / "gguf-py")
    converter = install / "convert_hf_to_gguf.py"
    converter.parent.mkdir(parents = True, exist_ok = True)
    converter.write_text("# converter\n", encoding = "utf-8")

    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", raising = False)
    converter_location = None

    if root == "default_dir":
        monkeypatch.setattr(llama_cpp, "LLAMA_CPP_DEFAULT_DIR", str(install))
    elif root == "scripts_pin":
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(install))
    elif root == "converter":
        converter_location = str(converter)
    else:
        monkeypatch.syspath_prepend(str(tree))

    assert llama_cpp._trusted_gguf_tree(tree, converter_location) == tree


def test_the_conversion_screens_the_tree_before_the_verifier_gets_it(llama_cpp):
    """The rule is worth nothing unless convert_to_gguf applies it. Checked through
    the AST so a refactor of the call site does not quietly drop it."""
    import ast

    tree = ast.parse(Path(llama_cpp.__file__).read_text(encoding = "utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "convert_to_gguf")
    screened = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        names = [c.id for c in ast.walk(node.value) if isinstance(c, ast.Name)]
        if "_gguf_readback_tree" in names:
            assert "_trusted_gguf_tree" in names, (
                "the child-reported tree reaches sys.path without a provenance check"
            )
            screened.update(t.id for t in node.targets if isinstance(t, ast.Name))
    assert screened, "convert_to_gguf no longer derives a read-back tree"


# --- The probe child does not search the working directory ---

PROBE_PAYLOAD = """
import os
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "EXECUTED"), "a").close()
"""


@pytest.mark.skipif(sys.version_info < (3, 11),
                    reason = "PYTHONSAFEPATH is honored from 3.11")
def test_the_probe_does_not_resolve_gguf_from_the_working_directory(
        llama_cpp, tmp_path, monkeypatch):
    """`python -c` would otherwise put the CWD at the child's sys.path[0], ahead of
    the converter's own tree and of every installed one."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _gguf_package(cwd, body = PROBE_PAYLOAD)
    monkeypatch.chdir(cwd)

    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    report = llama_cpp._probe_child_gguf(
        sys.executable, env, (), None, timeout = 60,
    )
    assert report is not None
    location = report.get("location") or ""
    assert not str(location).startswith(str(cwd)), location
    assert not (tmp_path / "EXECUTED").exists(), "the planted package ran in the child"


def test_the_probe_still_finds_the_converter_own_gguf_py(llama_cpp, tmp_path):
    """The whole point of the probe: the tree the converter self-locates must still
    be the one it reports."""
    install = tmp_path / "install"
    _gguf_package(install / "gguf-py", body = "__version__ = '9.9.9'\n")
    converter = install / "convert_hf_to_gguf.py"
    converter.write_text("# converter\n", encoding = "utf-8")

    report = llama_cpp._probe_child_gguf(
        sys.executable, dict(os.environ), (), str(converter), timeout = 60,
    )
    assert report is not None, "the probe must still run"
    assert report.get("location") == str(install / "gguf-py" / "gguf" / "__init__.py")


def test_the_probe_environment_is_not_mutated_for_the_caller(llama_cpp, tmp_path):
    """The converter child builds its env from the live environment, so the probe
    may not leave PYTHONSAFEPATH behind in it."""
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    llama_cpp._probe_child_gguf(sys.executable, env, (), None, timeout = 60)
    assert "PYTHONSAFEPATH" not in env
