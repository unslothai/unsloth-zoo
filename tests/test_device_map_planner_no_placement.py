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

"""The device-map planner leaves `_no_placement_params` owners out of the plan (on CPU)."""
import pytest
import torch
from torch import nn


class Table(nn.Module):
    def __init__(self):
        super().__init__()
        self.ngram_embedding = nn.Embedding(4000, 64)


class Layer(nn.Module):
    def __init__(self, ple):
        super().__init__()
        self.mlp = nn.Linear(64, 64)
        if ple:
            self.ple = nn.Module()
            self.ple.ple_embedding = Table()
            self.ple.key_proj = nn.Linear(64, 64)


class Model(nn.Module):
    _no_placement_params = ["ple.ple_embedding.ngram_embedding.weight"]

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer(i == 1) for i in range(3)])


def test_drop_no_placement_modules(monkeypatch):
    from unsloth_zoo.device_map_planner import drop_no_placement_modules
    with torch.device("meta"):
        model = Model()
    before = sum(p.numel() for p in model.parameters())
    dropped = drop_no_placement_modules(model)
    assert dropped == ["layers.1.ple.ple_embedding.ngram_embedding"]
    names = [n for n, _ in model.named_parameters()]
    assert not any("ngram_embedding" in n for n in names)
    assert "layers.1.ple.key_proj.weight" in names
    assert before - sum(p.numel() for p in model.parameters()) == 4000 * 64


def test_kill_switch_and_plain_models(monkeypatch):
    from unsloth_zoo.device_map_planner import drop_no_placement_modules
    with torch.device("meta"):
        model = Model()
    monkeypatch.setenv("UNSLOTH_PLACE_NO_PLACEMENT_PARAMS", "1")
    assert drop_no_placement_modules(model) == []
    monkeypatch.delenv("UNSLOTH_PLACE_NO_PLACEMENT_PARAMS")
    plain = nn.Linear(2, 2)
    assert drop_no_placement_modules(plain) == []


def test_planner_calls_it():
    import inspect
    from unsloth_zoo import device_map_planner
    src = inspect.getsource(device_map_planner.plan_device_map_for_pretrained)
    assert "drop_no_placement_modules(model)" in src
