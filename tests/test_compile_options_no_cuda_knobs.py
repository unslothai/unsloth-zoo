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

"""Compile options skip the CUTLASS-only cuda.* knobs: Unsloth never selects CUTLASS, and on torch 2.16 cuda.* is a
deprecated alias of cutlass.* that warns when set. CPU-only."""
from __future__ import annotations

import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo"


def test_compile_options_and_inductor_config_skip_cuda_knobs():
    src = (_ROOT / "temporary_patches" / "common.py").read_text()
    assert '"cuda.compile_opt_level"' not in src and '"cuda.enable_cuda_lto"' not in src
    src = (_ROOT / "patching_utils.py").read_text()
    assert "config.cuda.enable_cuda_lto" not in src and "config.cuda.compile_opt_level" not in src
