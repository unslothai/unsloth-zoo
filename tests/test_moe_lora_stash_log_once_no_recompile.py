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

"""The once-only "stash unread" message must not guard the traced MoE forward on its logged-set contents: the first
traced call added its key and the next call recompiled the whole block (DeepSeek-V4.1 SparseMoeBlock)."""

import torch

import unsloth_zoo.temporary_patches.moe_utils as mu


class _Experts(torch.nn.Module):
    pass


def test_logging_the_unread_stash_does_not_recompile_the_caller(monkeypatch):
    from torch._dynamo.testing import CompileCounter

    monkeypatch.setattr(mu, "_MOE_LORA_STASH_UNREAD_LOGGED", set())
    experts = _Experts()

    def forward(x):
        mu._log_moe_lora_stash_unread_once(experts, "down_proj")
        return x * 2

    counter = CompileCounter()
    compiled = torch.compile(forward, backend = counter, fullgraph = True)
    torch._dynamo.reset()
    try:
        for _ in range(3):
            compiled(torch.ones(4))
    finally:
        torch._dynamo.reset()
    assert counter.frame_count == 1, counter.frame_count
