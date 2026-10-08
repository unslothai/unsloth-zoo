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

"""Per-call host trimming of the transformers<5 ModuleList grouped MoE forward.

The per-expert projection lists are built once per readiness verdict (_block_projs) and must follow
a projection swap on any expert.
"""
import pytest
import torch

from unsloth_zoo.temporary_patches import moe_grouped_modulelist as ML

import test_moe_grouped_modulelist_lora as L  # noqa: E402  (pytest puts tests/ on sys.path)
import test_moe_grouped_modulelist_layout as LY  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (L.DEV == "cuda" and ML._grouped_mm_supported()),
    reason = "needs CUDA torch._grouped_mm",
)


def test_projection_lists_cached_and_follow_a_swap():
    model, blk = L.build("qwen3")
    L.enable(model, blk)
    x = torch.randn(1, 64, L.H, device = "cuda", dtype = L.DT)
    assert LY._engages(blk, x)
    first = blk.__dict__["_moe_projs"][3]
    assert LY._engages(blk, x)
    assert blk.__dict__["_moe_projs"][3] is first   # reused, not rebuilt
    assert first.gate[5] is blk.experts[5].gate_proj and first.gate_up[11] is blk.experts[5].up_proj
    # Replace one middle expert's down projection: the readiness entry moves, so the lists follow.
    old = blk.experts[4].down_proj
    new = type(old).__new__(type(old))
    new.__dict__.update(old.__dict__)
    blk.experts[4].down_proj = new
    assert LY._engages(blk, x)
    assert blk.__dict__["_moe_projs"][3].down[4] is new


def test_projection_lists_spot_check_without_readiness_change():
    """A readiness cache that kept its entry across a swap of an end expert still rebuilds."""
    model, blk = L.build("qwen3")
    L.enable(model, blk)
    x = torch.randn(1, 64, L.H, device = "cuda", dtype = L.DT)
    assert LY._engages(blk, x)
    spec = blk._unsloth_moe_spec
    projs = ML._block_projs(blk, blk.experts, spec)
    blk.experts[-1]._modules["up_proj"] = blk.experts[0].up_proj   # entry untouched
    again = ML._block_projs(blk, blk.experts, spec)
    assert again is not projs and again.up[-1] is blk.experts[0].up_proj


def test_disable_drops_the_projection_lists():
    """disable_grouped_moe leaves no reference to the experts or their projections on the block."""
    model, blk = L.build("qwen3")
    L.enable(model, blk)
    x = torch.randn(1, 64, L.H, device = "cuda", dtype = L.DT)
    assert LY._engages(blk, x)
    assert "_moe_projs" in blk.__dict__
    ML.disable_grouped_moe(model)
    assert "_moe_projs" not in blk.__dict__ and "_moe_ready" not in blk.__dict__
