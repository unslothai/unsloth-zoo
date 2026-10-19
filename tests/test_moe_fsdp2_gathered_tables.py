# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""FSDP2 all-gathered 4-bit experts (unsloth fsdp2_qlora marks them `_unsloth_fsdp2_gathered`):
FSDP2 frees and re-allocates their storage around every forward and backward, so the NF4 pointer
tables keyed by weight address must not be trusted across calls.

1. moe_ready_epoch.scan treats such a subtree like an accelerate-hooked one (no cached record), so
   the grouped readiness and the pointer tables re-key over every expert on each call.
2. The gpt-oss grouped QLoRA weight provider re-reads its table when it runs (the backward
   recompute), instead of the table captured in forward, which pointed at the freed forward
   all-gather (wrong dX on 2 GPUs: max |dX - ref| 0.4, bit-exact with the refresh).
"""

import torch
import torch.nn as nn

from unsloth_zoo.temporary_patches import moe_ready_epoch
from unsloth_zoo.temporary_patches import gpt_oss_grouped_qlora as gq


def _experts(marked):
    experts = nn.Module()
    experts.gate_up_projs = nn.ModuleList([nn.Linear(4, 8, bias = False) for _ in range(2)])
    if marked:
        for lin in experts.gate_up_projs:
            lin.weight._unsloth_fsdp2_gathered = True
    return experts


def test_fsdp2_gathered_subtree_gets_no_cached_record():
    assert moe_ready_epoch.FSDP2_GATHERED == "_unsloth_fsdp2_gathered"
    hooked, _ = moe_ready_epoch.scan(_experts(marked = True))
    assert hooked
    hooked, _ = moe_ready_epoch.scan(_experts(marked = False))
    assert not hooked


def test_stack_provider_rereads_its_table_when_refreshing(monkeypatch):
    seen = []
    monkeypatch.setattr(gq, "nf4_dequant_expert_stack", lambda tb, dtype: seen.append(tb["id"]) or torch.zeros(1, 2, 3))
    stale, fresh = {"id": "forward", "dtype": torch.bfloat16}, {"id": "backward", "dtype": torch.bfloat16}
    provider = gq._StackProvider(stale, torch.bfloat16, lambda: None, lambda: fresh)
    assert provider().shape == (1, 3, 2)
    assert seen == ["backward"]
    # Without a refresh (one GPU, no FSDP2) the table captured in forward is used, as before.
    seen.clear()
    gq._StackProvider(stale, torch.bfloat16, lambda: None)()
    assert seen == ["forward"]
