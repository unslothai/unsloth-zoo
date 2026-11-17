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

"""The grouped ModuleList MoE forward compiles once: its engagement counters are eager only.

A traced `CALLS["grouped"] += 1` makes Dynamo guard on the counter's old value, so every call
recompiled (Unsloth's per-layer block compile on transformers 4.x Qwen3-MoE / Mixtral / OLMoE:
4 recompiles per training step until the region limit, then the block ran eager)."""
import pytest
import torch

from unsloth_zoo.temporary_patches import moe_grouped_modulelist as ML

import test_moe_grouped_modulelist_lora as lora_helpers  # noqa: E402

pytestmark = lora_helpers.pytestmark


def test_compiled_grouped_forward_does_not_recompile_per_call():
    from torch._dynamo.testing import CompileCounter

    model, blk = lora_helpers.build("qwen3", prefer_hf = False, targets = "all")
    lora_helpers.enable(model, blk)
    x = torch.randn(2, 16, lora_helpers.H, device = lora_helpers.DEV, dtype = lora_helpers.DT)

    before = dict(ML.CALLS)
    with torch.no_grad():
        want = blk(x)
    assert ML.CALLS["grouped"] == before["grouped"] + 1   # eager calls still count

    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(blk.forward, backend = counter, dynamic = False)
    with torch.no_grad():
        outs = [compiled(x)]
        first = counter.frame_count
        outs += [compiled(x) for _ in range(3)]
    torch._dynamo.reset()
    assert first >= 1
    assert counter.frame_count == first, f"{counter.frame_count - first} recompiles in 3 repeat calls"
    for out in outs:
        got = out[0] if isinstance(out, tuple) else out
        ref = want[0] if isinstance(want, tuple) else want
        torch.testing.assert_close(got, ref)
