"""Importable base model for cross-process pickle tests; living in its own module lets a fresh subprocess resolve the class by name."""
import torch.nn as nn


class PickleBaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.lin = nn.Linear(8, 8)

    def _impl(self, x):
        return self.lin(self.norm(x))

    def forward(self, x):
        return self._impl(x)
