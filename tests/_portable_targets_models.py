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

"""Toy models for test_portable_adapter_target_modules.py; imports nothing from Unsloth."""

import sys

import torch
import torch.nn as nn


class ClippableLinear(nn.Module):
    """Stand-in for Gemma4ClippableLinear: an nn.Linear PEFT cannot wrap from outside."""

    def __init__(self, n):
        super().__init__()
        self.linear = nn.Linear(n, n, bias = False)

    def forward(self, x):
        return self.linear(x).clamp(-10, 10)


class Attn(nn.Module):
    def __init__(self, n, clippable):
        super().__init__()
        self.q_proj = ClippableLinear(n) if clippable else nn.Linear(n, n, bias = False)

    def forward(self, x):
        return self.q_proj(x)


class TwoTowers(nn.Module):
    def __init__(self, n = 6):
        super().__init__()
        self.vision = nn.ModuleList([Attn(n, True) for _ in range(2)])
        self.text = nn.ModuleList([Attn(n, False) for _ in range(2)])

    def forward(self, x):
        for layer in list(self.vision) + list(self.text):
            x = layer(x)
        return x



def plain_peft_reload(adapter_dir, state_path, x_path):
    """Run in a fresh interpreter: load the adapter with plain PEFT, print the output."""
    from peft import PeftModel
    base = TwoTowers()
    base.load_state_dict(torch.load(state_path))
    model = PeftModel.from_pretrained(base, adapter_dir)
    with torch.no_grad():
        torch.save(model(torch.load(x_path)), x_path + ".out")


if __name__ == "__main__":
    plain_peft_reload(*sys.argv[1:4])
