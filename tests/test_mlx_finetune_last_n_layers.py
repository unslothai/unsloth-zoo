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

"""Tests for `finetune_last_n_layers` in FastMLXModel.get_peft_model.

mlx-lm CLI's CONFIG_DEFAULTS sets num_layers=16 (lora.py:56), meaning
LoRA is applied to the last 16 transformer blocks only. unsloth-zoo's
get_peft_model historically applied LoRA to ALL transformer layers,
matching HF PEFT/CUDA semantics. The two paths could produce different
trained models on the same fixture (different basin) because (a) the
extra layers contribute extra LoRA modules whose lora_a init consumes
mx.random state, shifting init for the later layers, and (b) the
trainable-parameter set differs.

`finetune_last_n_layers` lets users opt into mlx-lm CLI semantics
without changing the existing default (which remains None = all layers).
"""

from __future__ import annotations

import inspect

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def test_get_peft_model_has_finetune_last_n_layers_param():
    """The parameter exists and defaults to None (= all layers)."""
    from unsloth_zoo.mlx.loader import FastMLXModel
    sig = inspect.signature(FastMLXModel.get_peft_model)
    assert "finetune_last_n_layers" in sig.parameters
    assert sig.parameters["finetune_last_n_layers"].default is None


def test_get_peft_model_passes_finetune_last_n_layers_through():
    """When finetune_last_n_layers is set, linear_to_lora_layers is
    called with that num_layers value.

    Patches mlx_lm.tuner.utils.linear_to_lora_layers to record the
    num_layers it sees. The test runs against a tiny synthetic text
    model (no real layers needed -- the value of num_layers passed is
    what we assert, not the side effects on a real architecture).
    """
    import sys
    import unsloth_zoo.mlx.loader as loader_mod

    # Build a minimal text-only fake model with .model.layers of len=8.
    class FakeLayer: pass
    class FakeInner:
        layers = [FakeLayer() for _ in range(8)]
    class FakeModel:
        model = FakeInner()
        _unsloth_full_finetuning = False
        _is_vlm_model = False
        def freeze(self): pass
        def unfreeze(self, **kwargs): pass

    # Capture num_layers values seen by linear_to_lora_layers.
    captured = {"calls": []}
    def fake_linear_to_lora_layers(model, num_layers, config, use_dora=False):
        captured["calls"].append(num_layers)

    # Stub out the helpers get_peft_model uses internally so the test
    # doesn't need to walk a real model tree.
    import unsloth_zoo.mlx.loader as L
    L._fix_missing_no_grad = lambda m: None
    L._resolve_lora_keys = lambda m, t: [
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.mlp.gate_proj",
    ]
    L._apply_mlx_lora_initialization = lambda m, init: None
    L.linear_to_lora_layers = fake_linear_to_lora_layers
    # mlx_lm.tuner.utils is imported inside the function:
    fake_mod = type(sys)("mlx_lm.tuner.utils")
    fake_mod.linear_to_lora_layers = fake_linear_to_lora_layers
    sys.modules["mlx_lm.tuner.utils"] = fake_mod

    # Case 1: default (None) -> all 8 layers
    captured["calls"].clear()
    loader_mod.FastMLXModel.get_peft_model(
        FakeModel(),
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "gate_proj"],
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        use_gradient_checkpointing=False,
    )
    assert captured["calls"] == [8]

    # Case 2: finetune_last_n_layers=5 -> 5 layers
    captured["calls"].clear()
    loader_mod.FastMLXModel.get_peft_model(
        FakeModel(),
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "gate_proj"],
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        finetune_last_n_layers=5,
        use_gradient_checkpointing=False,
    )
    assert captured["calls"] == [5]

    # Case 3: finetune_last_n_layers > actual layer count -> clamp to total
    captured["calls"].clear()
    loader_mod.FastMLXModel.get_peft_model(
        FakeModel(),
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "gate_proj"],
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        finetune_last_n_layers=999,  # exceeds layer count
        use_gradient_checkpointing=False,
    )
    assert captured["calls"] == [8]

    # Case 4: finetune_last_n_layers=0 or negative -> clamp to 1
    captured["calls"].clear()
    loader_mod.FastMLXModel.get_peft_model(
        FakeModel(),
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "gate_proj"],
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        finetune_last_n_layers=0,
        use_gradient_checkpointing=False,
    )
    assert captured["calls"] == [1]
