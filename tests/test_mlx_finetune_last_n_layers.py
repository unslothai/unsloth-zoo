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

mlx-lm CLI defaults to num_layers=16 (lora.py:56): LoRA on the last 16 blocks
only. unsloth-zoo historically applied LoRA to ALL layers (HF PEFT/CUDA
parity). The paths can diverge because extra layers add LoRA modules whose
lora_a init consumes mx.random state (shifting later-layer init) and change the
trainable set. `finetune_last_n_layers` opts into mlx-lm semantics; the default
stays None (all layers).
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
    """linear_to_lora_layers receives the right num_layers value.

    Patches mlx_lm.tuner.utils.linear_to_lora_layers to record num_layers and
    runs against a synthetic text model (we assert the passed value, not real
    side effects).
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
        # The trainable-parameter summary at the end of get_peft_model walks
        # these; empty trees keep the count at 0 without touching the asserts.
        def parameters(self): return {}
        def trainable_parameters(self): return {}

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
