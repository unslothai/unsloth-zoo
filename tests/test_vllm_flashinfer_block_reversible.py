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

"""Blocking `import flashinfer` for vLLM must not be a one-way door.

`load_vllm` blocks the import when FlashInfer is installed but nvcc/ninja are
not, so vLLM cannot pick a backend it would have to JIT-compile. The block is
process wide: `sys.modules["flashinfer"] = None` makes every later
`import flashinfer` raise and makes `importlib.util.find_spec("flashinfer")`
report the package as absent, for unrelated code as much as for vLLM. Left
permanent, a long-lived session that installs the missing tool afterwards can
never get FlashInfer back.

These tests use a synthetic `flashinfer` package, so nothing here imports the
real one or needs a GPU.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types

import pytest


vllm_utils = importlib.import_module("unsloth_zoo.vllm_utils")


@pytest.fixture
def fake_flashinfer(monkeypatch):
    """Install a synthetic `flashinfer` package plus one submodule."""
    root = types.ModuleType("flashinfer")
    root.__spec__ = importlib.machinery.ModuleSpec("flashinfer", loader = None)
    root.__path__ = []
    sub = types.ModuleType("flashinfer.decode")
    sub.__spec__ = importlib.machinery.ModuleSpec("flashinfer.decode", loader = None)

    monkeypatch.setitem(sys.modules, "flashinfer", root)
    monkeypatch.setitem(sys.modules, "flashinfer.decode", sub)
    monkeypatch.setattr(vllm_utils, "_UNSLOTH_BLOCKED_FLASHINFER_MODULES", {})
    return types.SimpleNamespace(root = root, sub = sub)


def _import_flashinfer_fails() -> bool:
    try:
        importlib.import_module("flashinfer")
    except ImportError:
        return True
    return False


def test_block_then_unblock_restores_the_modules(fake_flashinfer):
    vllm_utils._block_flashinfer_import()
    assert sys.modules["flashinfer"] is None
    assert "flashinfer.decode" not in sys.modules
    assert importlib.util.find_spec("flashinfer") is None
    assert _import_flashinfer_fails()

    vllm_utils._unblock_flashinfer_import()
    assert sys.modules["flashinfer"] is fake_flashinfer.root
    assert sys.modules["flashinfer.decode"] is fake_flashinfer.sub
    assert importlib.util.find_spec("flashinfer") is not None
    assert not _import_flashinfer_fails()


def test_unblock_leaves_a_package_that_was_never_imported_absent(monkeypatch):
    monkeypatch.delitem(sys.modules, "flashinfer", raising = False)
    monkeypatch.setattr(vllm_utils, "_UNSLOTH_BLOCKED_FLASHINFER_MODULES", {})

    vllm_utils._block_flashinfer_import()
    assert sys.modules["flashinfer"] is None

    vllm_utils._unblock_flashinfer_import()
    assert "flashinfer" not in sys.modules


def test_unblock_without_a_block_is_a_no_op(fake_flashinfer):
    vllm_utils._unblock_flashinfer_import()
    assert sys.modules["flashinfer"] is fake_flashinfer.root


def test_repeated_blocks_keep_the_original_modules(fake_flashinfer):
    vllm_utils._block_flashinfer_import()
    # A second call must not record the None sentinel as the thing to restore.
    vllm_utils._block_flashinfer_import()
    vllm_utils._unblock_flashinfer_import()
    assert sys.modules["flashinfer"] is fake_flashinfer.root
    assert sys.modules["flashinfer.decode"] is fake_flashinfer.sub


def test_load_vllm_lifts_the_block_when_the_opt_out_is_cleared(fake_flashinfer):
    """The reachable path: a second `load_vllm()` after the caller re-enabled
    FlashInfer must re-probe, which it cannot do while find_spec answers None."""
    import inspect

    source = inspect.getsource(vllm_utils.load_vllm)
    unblock = source.find("_unblock_flashinfer_import()")
    probe = source.find('elif importlib.util.find_spec("flashinfer")')
    assert unblock != -1, "load_vllm never lifts a previously installed block"
    assert probe != -1
    assert unblock < probe, "the block must be lifted before the FlashInfer probe"
