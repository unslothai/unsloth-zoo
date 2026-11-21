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

"""CPU: use_gradient_checkpointing = False wins over SFTConfig's default."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


_TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def _tiny_model():
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(_TINY, dtype = torch.float32, attn_implementation = "sdpa")
    except Exception as error:  # offline without the tiny model cached
        pytest.skip(f"{_TINY} unavailable: {error}")


class _GcArgs:
    def __init__(self, gc = True, length = 128, batch = 2):
        self.gradient_checkpointing = gc
        self.max_length = length
        self.per_device_train_batch_size = batch


def test_use_gradient_checkpointing_false_beats_trl_default(monkeypatch):
    """SFTConfig defaults gradient_checkpointing = True; the model's use_gradient_checkpointing = False wins."""
    from unsloth_zoo.cpu import trainer
    # With the memory defaults, "unsloth" may also drop checkpointing when RAM is plentiful: pin it to none free.
    monkeypatch.setattr(trainer, "available_ram", lambda: 0, raising = False)
    model = _tiny_model()
    model._unsloth_cpu_gradient_checkpointing = False
    args = _GcArgs()
    assert trainer._auto_cpu_gradient_checkpointing(model, args) is True
    assert args.gradient_checkpointing is False
    model._unsloth_cpu_gradient_checkpointing = "unsloth"
    args = _GcArgs()
    assert not trainer._auto_cpu_gradient_checkpointing(model, args) and args.gradient_checkpointing is True


def test_get_peft_model_records_the_choice():
    from unsloth_zoo.cpu import loader
    model = _tiny_model()
    lora = loader.FastCPUModel.get_peft_model(model, r = 4, use_gradient_checkpointing = False)
    assert getattr(lora, "_unsloth_cpu_gradient_checkpointing") is False
    loader.FastCPUModel.for_training(lora, use_gradient_checkpointing = True)
    assert getattr(lora, "_unsloth_cpu_gradient_checkpointing") is True
