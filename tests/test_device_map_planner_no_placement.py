"""The device-map planner leaves a model's `_no_placement_params` owners out of the plan.

Qwen4Exp (Qwen3.8-Flash-Next) declares its hashed n-gram table (~102 GB bf16 / ~51 GB FP8)
unplaceable. Planned as part of its no-split decoder layer it needed a whole card, and the
load placed a frozen lookup table on the GPU. Dropped from the META model, it is neither
sized nor mapped, so the load keeps it on CPU."""
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
