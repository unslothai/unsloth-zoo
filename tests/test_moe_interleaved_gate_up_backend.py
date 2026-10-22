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

"""GPT-OSS (interleaved gate_up) cannot run the Triton kernels: asking for them takes grouped_mm, not the loop."""
import pytest

import unsloth_zoo.temporary_patches.moe_utils as M


@pytest.mark.parametrize("selected, grouped_mm_ok, expected", [
    ("unsloth_triton", True, "grouped_mm"),
    ("unsloth_triton", False, "unsloth_triton"),  # callers then take the loop, as before
    ("grouped_mm", True, "grouped_mm"),
    ("native_torch", True, "native_torch"),
])
def test_interleaved_gate_up_backend(monkeypatch, selected, grouped_mm_ok, expected):
    monkeypatch.setattr(M, "select_moe_backend", lambda: selected)
    monkeypatch.setattr(M, "_check_torch_grouped_mm_supported", lambda: grouped_mm_ok)
    assert M.select_moe_backend_for_interleaved_gate_up() == expected


def test_gpt_oss_and_mxfp4_ask_the_interleaved_selector():
    import inspect
    import unsloth_zoo.temporary_patches.gpt_oss as G
    import unsloth_zoo.temporary_patches.mxfp4 as X
    for fn in (G.patch_gpt_oss_moe_for_lora, G.patch_gpt_oss_linearized, X.keep_mxfp4_experts_packed):
        source = inspect.getsource(fn)
        assert "select_moe_backend_for_interleaved_gate_up()" in source
        assert "select_moe_backend()" not in source
