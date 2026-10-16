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

"""A backend chosen while unsloth itself imports must not stick: its Triton kernels cannot load yet."""
import sys
import types

import unsloth_zoo.temporary_patches.moe_utils as M


def test_backend_chosen_during_unsloth_import_is_chosen_again(monkeypatch):
    spec = types.SimpleNamespace(_initializing = True)
    fake = types.ModuleType("unsloth")
    fake.__spec__ = spec
    monkeypatch.setitem(sys.modules, "unsloth", fake)
    # An earlier test on this worker may have imported the kernels; mid-import they are not there yet.
    for name in [name for name in sys.modules if name.startswith("unsloth.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setenv("UNSLOTH_MOE_BACKEND", "unsloth_triton")
    monkeypatch.setattr(M, "_GROUPED_GEMM_AVAILABLE", None)
    monkeypatch.setattr(M, "_check_torch_grouped_mm_supported", lambda: False)
    monkeypatch.setattr(M, "is_mlx_available", lambda: False)
    monkeypatch.setattr(M.torch.cuda, "is_available", lambda: True)
    M.select_moe_backend.cache_clear()
    try:
        # Mid-import the kernels module is unreachable (the fake package has no submodules).
        assert M.select_moe_backend() == "native_torch"
        assert M._GROUPED_GEMM_AVAILABLE is None
        spec._initializing = False
        monkeypatch.setattr(M, "_check_grouped_gemm_available", lambda: True)
        assert M.select_moe_backend() == "unsloth_triton"
        monkeypatch.setattr(M, "_check_grouped_gemm_available", lambda: False)
        assert M.select_moe_backend() == "unsloth_triton"  # cached once decided after the import
    finally:
        M.select_moe_backend.cache_clear()
