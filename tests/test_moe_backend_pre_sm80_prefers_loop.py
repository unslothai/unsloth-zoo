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

"""Without grouped_mm, auto MoE selection picks the loop on CUDA below sm80; sm80+ / ROCm keep unsloth_triton."""

import importlib.util
import itertools

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils as M

CAPABILITIES = [(7, 0), (7, 5), (8, 0), (8, 6), (9, 0), (12, 0)]
PLATFORMS = ["cuda", "hip", "xpu", "cpu"]
ENVS = [None, "unsloth_triton", "native_torch", "grouped_mm"]


def _patch(module, monkeypatch, platform, capability, grouped_mm, env):
    if env is None:
        monkeypatch.delenv("UNSLOTH_MOE_BACKEND", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_MOE_BACKEND", env)
    on_gpu = platform in ("cuda", "hip")
    monkeypatch.setattr(module, "_check_torch_grouped_mm_supported", lambda *a, **k: grouped_mm)
    monkeypatch.setattr(module, "_check_grouped_gemm_available", lambda: on_gpu)
    monkeypatch.setattr(torch.version, "hip", "6.3" if platform == "hip" else None)

    def _capability(*args, **kwargs):
        if not on_gpu:
            raise RuntimeError("no CUDA device")
        return capability
    monkeypatch.setattr(torch.cuda, "get_device_capability", _capability)


def _expected(platform, capability, grouped_mm, env):
    on_gpu = platform in ("cuda", "hip")
    if env == "grouped_mm" and grouped_mm: return "grouped_mm"
    if env == "unsloth_triton" and on_gpu: return "unsloth_triton"
    if env == "native_torch": return "native_torch"
    if grouped_mm: return "grouped_mm"
    if on_gpu and (platform == "hip" or capability[0] >= 8): return "unsloth_triton"
    return "native_torch"


CELLS = [
    cell for cell in itertools.product(PLATFORMS, CAPABILITIES, [True, False], ENVS)
    if cell[0] in ("cuda", "hip") or cell[1] == CAPABILITIES[0]
]


@pytest.mark.parametrize("platform,capability,grouped_mm,env", CELLS)
def test_select_moe_backend_matrix(monkeypatch, platform, capability, grouped_mm, env):
    _patch(M, monkeypatch, platform, capability, grouped_mm, env)
    assert M._select_moe_backend() == _expected(platform, capability, grouped_mm, env)


def test_t4_without_grouped_mm_takes_loop(monkeypatch):
    _patch(M, monkeypatch, "cuda", (7, 5), False, None)
    M.select_moe_backend.cache_clear()
    try:
        assert M.select_moe_backend() == "native_torch"
    finally:
        M.select_moe_backend.cache_clear()


def test_compiled_cache_copy_selects_the_same(monkeypatch):
    # unsloth_compiled_cache/moe_utils.py is a byte copy with its own globals.
    spec = importlib.util.spec_from_file_location("_moe_utils_cache_copy_under_test", M.__file__)
    copy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copy)
    _patch(copy, monkeypatch, "cuda", (7, 5), False, None)
    copy.select_moe_backend.cache_clear()
    assert copy.select_moe_backend() == "native_torch"
    _patch(copy, monkeypatch, "cuda", (8, 0), False, None)
    copy.select_moe_backend.cache_clear()
    assert copy.select_moe_backend() == "unsloth_triton"
