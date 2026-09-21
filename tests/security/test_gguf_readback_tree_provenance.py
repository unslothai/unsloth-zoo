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

import json

import os

import subprocess

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


def test_a_shadowed_sys_path_entry_is_not_the_one_we_would_import(
        llama_cpp, tmp_path, monkeypatch):
    """The sys.path branch is allowed only because the swap changes nothing. That
    holds for the tree that WINS resolution, not for one sitting behind it: putting
    a shadowed package at index 0 is what makes this process start running it."""
    winner = _gguf_package(tmp_path / "winner")
    shadowed = _gguf_package(tmp_path / "shadowed")
    monkeypatch.syspath_prepend(shadowed)
    monkeypatch.syspath_prepend(winner)      # ahead of it, so this is what imports

    assert llama_cpp._trusted_gguf_tree(winner) == os.path.realpath(winner)
    assert llama_cpp._trusted_gguf_tree(shadowed) is None


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

    # Resolved, not the string handed in: see the symlink case below.
    assert llama_cpp._trusted_gguf_tree(tree, converter_location) == os.path.realpath(tree)


def test_the_checked_path_and_the_returned_path_cannot_diverge(
        llama_cpp, tmp_path, monkeypatch):
    """Resolving separately for the check and for the return is not the same rule
    twice: the link belongs to the principal being screened, so the two resolutions
    need not agree, and the value that matters is the one handed back to be imported.
    Driven with a realpath that answers differently the first time, which is the race
    made deterministic."""
    install = tmp_path / "install"
    inside = _gguf_package(install / "gguf-py")
    outside = _gguf_package(tmp_path / "attacker")
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_DEFAULT_DIR", str(install))

    link = tmp_path / "link"
    os.symlink(outside, str(link))

    real_realpath = os.path.realpath
    seen = {"n": 0}

    def flipping(path, *args, **kwargs):
        if str(path) == str(link):
            seen["n"] += 1
            # First answer: outside. By the second, the link has "moved" inside.
            return outside if seen["n"] == 1 else inside
        return real_realpath(path, *args, **kwargs)

    monkeypatch.setattr(os.path, "realpath", flipping)
    verdict = llama_cpp._trusted_gguf_tree(str(link))

    assert verdict != outside, "returned a tree the containment check never approved"
    assert seen["n"] == 1, f"resolved the attacker's link {seen['n']} times, not once"


def test_a_symlink_into_a_chosen_root_is_returned_resolved(llama_cpp, tmp_path, monkeypatch):
    """Accepting a link and handing back the link is a check that can be undone: the
    caller puts the returned string on sys.path and imports it, so a link the lower-
    trust principal still owns can be re-pointed in between."""
    install = tmp_path / "install"
    tree = _gguf_package(install / "gguf-py")
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_DEFAULT_DIR", str(install))

    attacker = tmp_path / "attacker"
    attacker.mkdir()
    link = attacker / "link"
    os.symlink(str(tree), str(link))

    assert llama_cpp._trusted_gguf_tree(str(link)) == os.path.realpath(str(tree))

    # And the returned path does not follow the link once it is re-pointed.
    elsewhere = _gguf_package(tmp_path / "elsewhere")
    os.remove(str(link))
    os.symlink(str(elsewhere), str(link))
    assert llama_cpp._trusted_gguf_tree(str(link)) is None


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


def test_the_probe_models_the_converter_script_directory(llama_cpp, tmp_path):
    """A script run has its own directory at sys.path[0], so a `gguf` sitting beside
    the converter outranks the sibling gguf-py it inserts at 1. The probe has to
    report that one, because it is the one the conversion will use."""
    install = tmp_path / "install"
    _gguf_package(install / "gguf-py", body = "__version__ = 'sibling'\n")
    beside = _gguf_package(install, body = "__version__ = 'beside'\n")
    converter = install / "convert_hf_to_gguf.py"
    converter.write_text(
        "import sys, os\n"
        "sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gguf-py'))\n"
        "import gguf\n"
        "print(gguf.__file__)\n",
        encoding = "utf-8",
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    env.pop("PYTHONSAFEPATH", None)
    run = subprocess.run(
        [sys.executable, "-S", str(converter)], env = env, stdout = subprocess.PIPE,
        stderr = subprocess.PIPE, encoding = "utf-8", timeout = 60,
    )
    actual = (run.stdout or "").strip().splitlines()[-1] if run.stdout else ""
    assert actual.startswith(str(beside)), f"the converter itself resolved {actual}"

    probe_env = dict(env)
    probe_env["PYTHONSAFEPATH"] = "1"
    completed = subprocess.run(
        [sys.executable, "-S", "-c", llama_cpp._GGUF_PROBE_SOURCE, str(converter), "0"],
        input = "[]", env = probe_env, stdout = subprocess.PIPE, stderr = subprocess.PIPE,
        encoding = "utf-8", timeout = 60,
    )
    reports = [
        json.loads(line) for line in (completed.stdout or "").splitlines()
        if line.strip().startswith("{")
    ]
    assert reports, completed.stderr
    assert reports[-1].get("location") == actual


def test_the_probe_models_a_symlinked_converter_from_its_target(llama_cpp, tmp_path):
    """Reached through a symlink the two slots part company: `__file__` stays the
    link, so the entrypoint's own gguf-py insertion is link-relative, while CPython
    prepends the resolved directory ("if it's a symbolic link, resolve symbolic
    links", sys.path docs). Each slot has to be modelled from its own directory."""
    target = tmp_path / "target"
    target.mkdir()
    resolved_gguf = _gguf_package(target, body = "__version__ = 'target'\n")
    linkdir = tmp_path / "linkdir"
    _gguf_package(linkdir / "gguf-py", body = "__version__ = 'link-sibling'\n")

    real = target / "convert_hf_to_gguf.py"
    real.write_text(
        "import sys, os\n"
        "sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gguf-py'))\n"
        "import gguf\n"
        "print(gguf.__file__)\n",
        encoding = "utf-8",
    )
    link = linkdir / "convert_hf_to_gguf.py"
    os.symlink(str(real), str(link))

    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    env.pop("PYTHONSAFEPATH", None)
    run = subprocess.run(
        [sys.executable, "-S", str(link)], env = env, stdout = subprocess.PIPE,
        stderr = subprocess.PIPE, encoding = "utf-8", timeout = 60,
    )
    actual = (run.stdout or "").strip().splitlines()[-1] if run.stdout else ""
    assert actual.startswith(str(resolved_gguf)), f"the converter itself resolved {actual}"

    probe_env = dict(env)
    probe_env["PYTHONSAFEPATH"] = "1"
    completed = subprocess.run(
        [sys.executable, "-S", "-c", llama_cpp._GGUF_PROBE_SOURCE, str(link), "0"],
        input = "[]", env = probe_env, stdout = subprocess.PIPE, stderr = subprocess.PIPE,
        encoding = "utf-8", timeout = 60,
    )
    reports = [
        json.loads(line) for line in (completed.stdout or "").splitlines()
        if line.strip().startswith("{")
    ]
    assert reports, completed.stderr
    assert reports[-1].get("location") == actual


def test_the_probe_drops_the_working_directory_without_interpreter_support(
        llama_cpp, tmp_path, monkeypatch):
    """The 3.9/3.10 case, simulated by withholding the flag the way those versions
    do. Refusing the tree afterwards is not enough there: the child has already run
    the planted package, with this process's environment and token. So the probe
    drops the entry itself, which needs no interpreter support."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _gguf_package(cwd, body = PROBE_PAYLOAD)

    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    env.pop("PYTHONSAFEPATH", None)          # exactly what a pre-3.11 child sees
    completed = subprocess.run(
        [sys.executable, "-c", llama_cpp._GGUF_PROBE_SOURCE, "", "0"],
        input = "[]", cwd = str(cwd), env = env, stdout = subprocess.PIPE,
        stderr = subprocess.PIPE, encoding = "utf-8", timeout = 60,
    )
    reports = [
        json.loads(line) for line in (completed.stdout or "").splitlines()
        if line.strip().startswith("{")
    ]
    assert reports, completed.stderr
    location = reports[-1].get("location") or ""
    assert not str(location).startswith(str(cwd)), location
    assert not (tmp_path / "EXECUTED").exists(), "the planted package ran in the child"


@pytest.mark.skipif(sys.version_info < (3, 11),
                    reason = "PYTHONSAFEPATH is honored from 3.11")
@pytest.mark.parametrize("caller_safe_path", [False, True])
def test_the_probe_reports_the_tree_the_converter_will_actually_use(
        llama_cpp, tmp_path, caller_safe_path):
    """Safe-path mode drops the leading entry from a SCRIPT run too, so a caller who
    already exports PYTHONSAFEPATH gets a converter whose own `insert(1, ...)` lands
    the sibling tree BEHIND PYTHONPATH. The probe has to match whichever side it is
    measuring, not assume the converter runs the way it does."""
    ambient = tmp_path / "ambient"
    _gguf_package(ambient, body = "__version__ = 'ambient'\n")
    install = tmp_path / "install"
    _gguf_package(install / "gguf-py", body = "__version__ = 'sibling'\n")
    converter = install / "convert_hf_to_gguf.py"
    converter.write_text(
        "import sys, os\n"
        "sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gguf-py'))\n"
        "import gguf\n"
        "print(gguf.__file__)\n",
        encoding = "utf-8",
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ambient)
    env.pop("PYTHONSAFEPATH", None)
    if caller_safe_path:
        env["PYTHONSAFEPATH"] = "1"

    # What the real converter resolves, run exactly as convert_to_gguf runs it.
    run = subprocess.run(
        [sys.executable, "-S", str(converter)], env = env,
        stdout = subprocess.PIPE, stderr = subprocess.PIPE, encoding = "utf-8", timeout = 60,
    )
    actual = (run.stdout or "").strip().splitlines()[-1] if run.stdout else ""
    assert actual, run.stderr

    # What the probe reports, through the shipped probe source with the same env.
    probe_env = dict(env)
    converter_safe = "1" if probe_env.get("PYTHONSAFEPATH") else "0"
    probe_env["PYTHONSAFEPATH"] = "1"
    completed = subprocess.run(
        [sys.executable, "-S", "-c", llama_cpp._GGUF_PROBE_SOURCE, str(converter), converter_safe],
        input = "[]", env = probe_env, stdout = subprocess.PIPE, stderr = subprocess.PIPE,
        encoding = "utf-8", timeout = 60,
    )
    reports = [
        json.loads(line) for line in (completed.stdout or "").splitlines()
        if line.strip().startswith("{")
    ]
    assert reports, completed.stderr
    assert reports[-1].get("location") == actual, (
        f"probe reported {reports[-1].get('location')}, converter used {actual}"
    )


@pytest.mark.skipif(sys.version_info < (3, 11),
                    reason = "PYTHONSAFEPATH is honored from 3.11")
def test_the_probe_keeps_the_converter_gguf_ahead_of_pythonpath(llama_cpp, tmp_path):
    """The real run has its script directory at sys.path[0] and the sibling tree at
    1, so the sibling outranks all of PYTHONPATH. Safe-path mode vacates index 0, so
    inserting at 1 there would rank the sibling BEHIND the first PYTHONPATH entry and
    the probe would report a gguf the conversion never uses.

    Run with `-S` rather than through `_probe_child_gguf`: a site-packages .pth that
    prepends its own entry occupies index 0 anyway and would hide the inversion.
    """
    ambient = tmp_path / "ambient"
    _gguf_package(ambient, body = "__version__ = 'ambient'\n")
    install = tmp_path / "install"
    _gguf_package(install / "gguf-py", body = "__version__ = 'sibling'\n")
    converter = install / "convert_hf_to_gguf.py"
    converter.write_text("# converter\n", encoding = "utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ambient)
    env["PYTHONSAFEPATH"] = "1"
    completed = subprocess.run(
        [sys.executable, "-S", "-c", llama_cpp._GGUF_PROBE_SOURCE, str(converter)],
        input = "[]", env = env, stdout = subprocess.PIPE, stderr = subprocess.PIPE,
        encoding = "utf-8", timeout = 60,
    )
    reports = [
        json.loads(line) for line in (completed.stdout or "").splitlines()
        if line.strip().startswith("{")
    ]
    assert reports, completed.stderr
    location = reports[-1].get("location") or ""
    assert location == str(install / "gguf-py" / "gguf" / "__init__.py"), location
