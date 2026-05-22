"""Importable base model for the cross-process pickle test. Lives in its own
module so a fresh subprocess can resolve the base class by name (mirrors a real
transformers model class)."""
import torch.nn as nn


class PickleBaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.lin = nn.Linear(8, 8)

    def forward(self, x):
        return self.lin(self.norm(x))
