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
"""unsloth_zoo::packed_experts_mm_v2 reads the routed counts on the host when the experts need several chunks, so it is
tagged cudagraph_unsafe (as manual_grouped_mm and nf4_grouped_mm_auto): Inductor keeps it out of CUDA graphs. CPU."""
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth_zoo import packed_experts  # noqa: E402


def test_packed_experts_mm_is_cudagraph_unsafe():
    tag = getattr(getattr(torch, "Tag", None), "cudagraph_unsafe", None)
    if tag is None:
        pytest.skip("this torch has no cudagraph_unsafe tag")
    op = packed_experts._register_op()
    assert op is not None, "packed_experts_mm did not register"
    assert tag in op.tags
    # versioned with the tag: Inductor's FX graph cache does not key on tags, so the old name would replay untagged graphs
    assert op is torch.ops.unsloth_zoo.packed_experts_mm_v2.default
