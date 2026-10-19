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

"""swap_moe_weights_for_call (the dequantize-then-dispatch path of the bnb 4-bit and FP8 MoE
backends) must leave the experts module's parameters registered exactly as it found them.

It installs the dense weights with object.__setattr__, which writes the instance __dict__. Writing
the original Parameter back the same way left it in __dict__, shadowing `_parameters`: FSDP2 swaps
`_parameters[name]` between the sharded and the all-gathered parameter around every forward, so
after one call the module kept returning a stale all-gather buffer (`'Params4bit' object has no
attribute '_local_tensor'` in FSDP2's lazy init, or freed storage in the next forward).
"""

import torch
import torch.nn as nn

from unsloth_zoo.temporary_patches.moe_utils import swap_moe_weights_for_call


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 8, 4), requires_grad = False)
        self.down_proj = nn.Parameter(torch.zeros(2, 4, 4), requires_grad = False)


def test_swap_does_not_shadow_registered_parameters():
    experts = _Experts()
    dense_gu, dense_dn = torch.ones(2, 8, 4), torch.ones(2, 4, 4)
    seen = {}

    def forward_fn(module, x):
        seen["gu"] = module.gate_up_proj
        seen["dn"] = module.down_proj
        return x + 1

    out = swap_moe_weights_for_call(experts, dense_gu, dense_dn, forward_fn, torch.zeros(1))
    assert out.item() == 1
    assert seen["gu"] is dense_gu and seen["dn"] is dense_dn
    assert "gate_up_proj" not in experts.__dict__
    assert "down_proj" not in experts.__dict__
    assert set(dict(experts.named_parameters())) == {"gate_up_proj", "down_proj"}
    # What FSDP2's unsafe_setattr_param does around each forward: replace the registered Parameter.
    replacement = nn.Parameter(torch.full((2, 8, 4), 3.0), requires_grad = False)
    experts._parameters["gate_up_proj"] = replacement
    assert experts.gate_up_proj is replacement


def test_swap_restores_plain_instance_attributes_and_raises_through():
    experts = nn.Module()
    plain_gu, plain_dn = torch.zeros(1), torch.zeros(2)
    experts.gate_up_proj = plain_gu  # a plain tensor attribute lives in __dict__
    experts.down_proj = plain_dn

    def boom(module, x):
        assert module.gate_up_proj is not plain_gu
        raise RuntimeError("inner")

    try:
        swap_moe_weights_for_call(experts, torch.ones(1), torch.ones(2), boom, None)
    except RuntimeError as error:
        assert str(error) == "inner"
    else:
        raise AssertionError("expected the forward's error")
    assert experts.__dict__["gate_up_proj"] is plain_gu
    assert experts.__dict__["down_proj"] is plain_dn
