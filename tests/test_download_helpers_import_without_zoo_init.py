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

"""The Hub download helpers import without ``unsloth_zoo``'s package init, and their
``unsloth_zoo.<module>`` names stay the same module objects."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULES = ("hf_cache", "hf_cache_state", "hf_xet_tuning", "hf_xet_health", "hf_xet_fallback")
HEAVY = ("unsloth_zoo", "torch", "transformers", "triton", "torchao", "huggingface_hub")


def _modules_loaded_by(statement: str) -> list[str]:
    watched = HEAVY + ("unsloth_zoo_downloads",)
    code = (
        "import json, sys\n"
        f"{statement}\n"
        "print(json.dumps(sorted("
        f"name for name in sys.modules if name.split('.')[0] in {watched!r}"
        ")))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd = REPO_ROOT,
        capture_output = True,
        text = True,
        timeout = 300,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name", MODULES)
def test_each_download_helper_imports_without_the_zoo_init(name):
    loaded = _modules_loaded_by(f"import unsloth_zoo_downloads.{name}")
    assert f"unsloth_zoo_downloads.{name}" in loaded
    heavy = [module for module in loaded if module.split(".")[0] in HEAVY]
    assert heavy == [], heavy


def test_the_old_names_are_the_same_module_objects():
    import unsloth_zoo_downloads.hf_xet_health as light
    from unsloth_zoo import hf_xet_health as aliased

    import unsloth_zoo

    assert aliased is light
    assert sys.modules["unsloth_zoo.hf_xet_health"] is light
    assert unsloth_zoo.hf_xet_health is light


def test_the_fallback_resolves_its_siblings_inside_the_light_package():
    from unsloth_zoo_downloads import hf_cache_state, hf_xet_fallback

    assert hf_xet_fallback.INCOMPLETE_SUFFIX is hf_cache_state.INCOMPLETE_SUFFIX
    assert hf_xet_fallback.__name__ == "unsloth_zoo_downloads.hf_xet_fallback"


def test_the_old_file_still_serves_a_loader_that_executes_it_by_path():
    """unsloth's _gpu_init loads unsloth_zoo/hf_cache.py from its file, under another name, before
    importing the package, to redirect a read-only cache ahead of huggingface_hub."""
    path = REPO_ROOT / "unsloth_zoo" / "hf_cache.py"
    spec = importlib.util.spec_from_file_location("unsloth_zoo._early_hf_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from unsloth_zoo_downloads import hf_cache as light

    assert module.redirect_hf_cache_if_readonly is light.redirect_hf_cache_if_readonly
    assert module._active_caches is light._active_caches
    assert "unsloth_zoo._early_hf_cache" not in sys.modules


def test_the_package_init_registers_the_old_names_before_anything_heavy():
    """A late failure of unsloth_zoo/__init__ must still leave unsloth_zoo.hf_cache and
    unsloth_zoo.hf_xet_tuning in sys.modules: Unsloth's backend and unsloth.dataset_num_proc read
    them from there when the package itself cannot import."""
    tree = ast.parse((REPO_ROOT / "unsloth_zoo" / "__init__.py").read_text(encoding = "utf-8"))
    relative = sorted(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module in {"hf_cache", "hf_xet_tuning"}
    )
    assert relative == ["hf_cache", "hf_xet_tuning"]


def test_the_fallback_file_runs_as_the_spawn_childs_main_module():
    """``_run_download_attempt`` points ``__main__.__file__`` at hf_xet_fallback.py, so every spawn
    child first runs that file as ``__mp_main__`` with no parent package; a package-relative import
    at its top level would crash the child before the download starts."""
    from unsloth_zoo_downloads import hf_xet_fallback

    loaded = _modules_loaded_by(
        f"import runpy\nrunpy.run_path({hf_xet_fallback.__file__!r}, run_name = '__mp_main__')"
    )
    assert "unsloth_zoo_downloads.hf_cache_state" in loaded
    heavy = [module for module in loaded if module.split(".")[0] in HEAVY]
    assert heavy == [], heavy
