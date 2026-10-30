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

"""Select the MoE backend at forward resolution, else the first traced call recompiles."""
import sys

import torch

import unsloth_zoo.temporary_patches.moe_utils as M


def _compiles(fn, calls = 3):
    compiled = []

    def backend(gm, example_inputs):
        compiled.append(gm)
        return gm.forward

    torch._dynamo.reset()
    f = torch.compile(fn, backend = backend)
    x = torch.randn(4)
    for _ in range(calls):
        f(x)
    torch._dynamo.reset()
    return len(compiled)


def _forward(x):
    if M.select_moe_backend() == "native_torch":
        return x * 3
    return x * 2


def test_resolving_the_forward_chooses_the_backend_before_any_trace(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_BACKEND", "native_torch")
    monkeypatch.delitem(sys.modules, "unsloth", raising = False)
    monkeypatch.setattr(M, "_load_cached_moe_utils_module", lambda: None)
    M.select_moe_backend.cache_clear()
    try:
        M.get_forward_moe_backend()
        assert M._SELECTED_MOE_BACKEND == "native_torch"
        assert _compiles(_forward) == 1
    finally:
        M.select_moe_backend.cache_clear()


def test_the_compiled_cache_copy_chooses_its_own_backend_too(monkeypatch):
    class Copy:
        selected = []

        @staticmethod
        def select_moe_backend():
            Copy.selected.append(True)
            return "native_torch"

        @staticmethod
        def forward_moe_backend(*args):
            return None

    monkeypatch.setattr(M, "_load_cached_moe_utils_module", lambda: Copy)
    assert M.get_forward_moe_backend() is Copy.forward_moe_backend
    assert Copy.selected == [True]


def test_a_traced_resolution_does_not_select(monkeypatch):
    monkeypatch.setattr(M, "_load_cached_moe_utils_module", lambda: None)
    monkeypatch.setattr(M.torch.compiler, "is_compiling", lambda: True)
    M.select_moe_backend.cache_clear()
    try:
        M.get_forward_moe_backend()
        assert M._SELECTED_MOE_BACKEND is None
    finally:
        M.select_moe_backend.cache_clear()
