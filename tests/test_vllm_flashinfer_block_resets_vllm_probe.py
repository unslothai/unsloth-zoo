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

"""The FlashInfer import block must also reset vLLM's cached has_flashinfer().

vllm/utils/flashinfer.py caches has_flashinfer() and evaluates it at import, before the
block. With nvcc on PATH but ninja / libcuda.so missing, the stale True made
kernel_warmup() on sm90+ run `from flashinfer.autotuner import AutoTuner` against the
None entry: `No module named 'flashinfer.autotuner'` (every vLLM run on an sm120 G4).
CPU only: the real vllm.utils.flashinfer module, a fake flashinfer package."""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types

import pytest


fi_utils = pytest.importorskip("vllm.utils.flashinfer")
vllm_utils = importlib.import_module("unsloth_zoo.vllm_utils")


def _kernel_warmup_gate():
    """The has_flashinfer() kernel_warmup() consults (same cached object when importable)."""
    try:
        from vllm.model_executor.warmup import kernel_warmup
        return kernel_warmup.has_flashinfer
    except Exception:
        return fi_utils.has_flashinfer


def _flashinfer_autotune_import():
    """kernel_warmup.flashinfer_autotune's first line, reached when the gate is True on sm90+."""
    from flashinfer.autotuner import AutoTuner  # noqa: F401


@pytest.fixture
def fake_flashinfer_with_nvcc(monkeypatch):
    root = types.ModuleType("flashinfer")
    root.__spec__ = importlib.machinery.ModuleSpec("flashinfer", loader = None, is_package = True)
    root.__path__ = []
    autotuner = types.ModuleType("flashinfer.autotuner")
    autotuner.__spec__ = importlib.machinery.ModuleSpec("flashinfer.autotuner", loader = None)
    autotuner.AutoTuner = object
    root.autotuner = autotuner
    for name in list(sys.modules):
        if name == "flashinfer" or name.startswith("flashinfer."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "flashinfer", root)
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", autotuner)
    monkeypatch.setattr(vllm_utils, "_UNSLOTH_BLOCKED_FLASHINFER_MODULES", {})
    # vLLM's probe only asks for nvcc on PATH; the zoo probe additionally wants ninja + libcuda.so.
    real_which = fi_utils.shutil.which
    monkeypatch.setattr(fi_utils.shutil, "which", lambda cmd, *a, **k: "/usr/bin/nvcc" if cmd == "nvcc" else real_which(cmd, *a, **k))
    fi_utils.has_flashinfer.cache_clear()
    yield
    vllm_utils._unblock_flashinfer_import()
    fi_utils.has_flashinfer.cache_clear()


def test_block_resets_vllm_cached_has_flashinfer(fake_flashinfer_with_nvcc):
    gate = _kernel_warmup_gate()
    assert gate() is True  # cached at vLLM import time, before Unsloth's probe runs

    vllm_utils._block_flashinfer_import()

    assert gate() is False, "stale has_flashinfer() cache: kernel_warmup would import flashinfer.autotuner"
    if gate():
        _flashinfer_autotune_import()


def test_unblock_lets_vllm_reprobe(fake_flashinfer_with_nvcc):
    gate = _kernel_warmup_gate()
    vllm_utils._block_flashinfer_import()
    assert gate() is False
    vllm_utils._unblock_flashinfer_import()
    assert gate() is True
    _flashinfer_autotune_import()


def test_unusable_flashinfer_disables_vllm_autotune_engine_arg():
    src = importlib.import_module("inspect").getsource(vllm_utils.load_vllm)
    assert 'engine_args["enable_flashinfer_autotune"] = False' in src
