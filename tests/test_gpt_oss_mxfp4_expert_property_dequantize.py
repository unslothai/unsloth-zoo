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

"""GPT-OSS Mxfp4GptOssExperts property dequant must not use removed mxfp4.dequantize.

transformers 5.16+ deleted ``transformers.integrations.mxfp4.dequantize`` (DTensor TP
rewrite). Zoo's loader hook already routes through ``dequantize_convertops``; the expert
property getters were still importing the deleted symbol (unsloth-zoo #1251).

Testing notes (for reviewers): the ``dequantize_convertops`` parity test runs when that
symbol exists (transformers 5.0+). The ``convert_moe_packed_tensors`` fallback inside
``dequantize_mxfp4_moe_blocks_scales`` is the same path exercised by
``tests/test_mxfp4_load_path_layout.py`` on CPU CI (transformers 4.x / 5.x layouts).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from unsloth_zoo.temporary_patches.mxfp4 import (
    dequantize_mxfp4_moe_blocks_scales,
    patch_convert_moe_packed_tensors,
)


_GPT_OSS_PATCH = (
    Path(__file__).resolve().parents[1]
    / "unsloth_zoo"
    / "temporary_patches"
    / "gpt_oss.py"
)


def test_gpt_oss_mxfp4_expert_properties_avoid_removed_dequantize_import():
    source = _GPT_OSS_PATCH.read_text(encoding="utf-8")
    assert "from transformers.integrations.mxfp4 import dequantize" not in source
    assert "dequantize_mxfp4_moe_blocks_scales" in source


def test_dequantize_mxfp4_moe_blocks_scales_matches_patched_convertops():
    pytest.importorskip("transformers.integrations.mxfp4")
    import transformers.integrations.mxfp4 as mxfp4_mod

    if not hasattr(mxfp4_mod, "dequantize_convertops"):
        pytest.skip("dequantize_convertops absent (pre-5.0 MXFP4 ConversionOps path)")

    patch_convert_moe_packed_tensors()

    convertops = mxfp4_mod.dequantize_convertops
    n_pos = len(
        [
            p
            for p in inspect.signature(convertops).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    )

    def _reference(blocks, scales):
        if n_pos >= 3:
            out = convertops(blocks, scales, blocks.device)
        else:
            out = convertops(blocks, scales)
        return out.data if isinstance(out, torch.nn.Parameter) else out

    E, D, G, B = 2, 4, 3, 16
    torch.manual_seed(0)
    blocks = torch.randint(0, 255, (E, D, G, B), dtype=torch.uint8)
    scales = torch.full((E, D, G), 127, dtype=torch.uint8)

    want = _reference(blocks.clone(), scales.clone())
    got = dequantize_mxfp4_moe_blocks_scales(blocks.clone(), scales.clone())

    assert tuple(got.shape) == tuple(want.shape)
    assert torch.equal(got.cpu(), want.cpu())
