# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team.
# Licensed under the GNU Affero General Public License, version 3 or later.

"""MLX LoRA coverage for routed SwitchLinear expert projections."""

import pytest


pytest.importorskip("mlx")


class _Model:
    def __init__(self, modules):
        self._modules = modules
        self.layers = [self]

    def named_modules(self):
        yield from self._modules


def test_mlx_lm_switch_targets_are_discovered():
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
    from unsloth_zoo.mlx.loader import (
        _collect_all_linear_target_names,
        _resolve_lora_keys,
    )

    modules = [
        ("mlp.gate_proj", SwitchLinear(8, 16, 2, bias=False)),
        ("mlp.down_proj", QuantizedSwitchLinear(64, 16, 2, bias=False)),
        ("mlp.quant_proj", nn.QuantizedLinear(64, 16, bias=False)),
    ]
    model = _Model(modules)
    assert _collect_all_linear_target_names(model) == [
        "down_proj", "gate_proj", "quant_proj",
    ]
    assert _resolve_lora_keys(model, ["gate_proj", "down_proj", "quant_proj"]) == {
        name for name, _ in modules
    }
