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


def test_planned_map_has_no_ancestor_of_the_table(monkeypatch):
    # transformers expands map keys by prefix: an entry for the table's no-split layer would
    # still place the table on that card.
    from types import SimpleNamespace
    from unsloth_zoo import device_map_planner

    with torch.device("meta"):
        model = Model()
    monkeypatch.setattr(device_map_planner, "_usable_devices", lambda max_memory: [0, 1])
    monkeypatch.setattr(device_map_planner, "build_meta_model", lambda *a, **k: (model, None, None))
    monkeypatch.setattr(
        device_map_planner, "plan_device_map",
        lambda m, **k: SimpleNamespace(device_map={"layers.0": 0, "layers.1": 1, "layers.2": 1}),
    )
    plan = device_map_planner.plan_device_map_for_pretrained("x")
    table = "layers.1.ple.ple_embedding.ngram_embedding.weight"
    assert not any(k == "" or table.startswith(k + ".") for k in plan.device_map)
    assert plan.device_map == {
        "layers.0": 0, "layers.2": 1, "layers.1.mlp": 1, "layers.1.ple.key_proj": 1,
    }
