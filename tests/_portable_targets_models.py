"""Toy models for test_portable_adapter_target_modules.py. Imports nothing from Unsloth,
so a subprocess can load an adapter with plain PEFT exactly as a user would."""

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
