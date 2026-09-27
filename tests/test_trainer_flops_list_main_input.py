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

"""Trainer.floating_point_ops with a list-valued main input.

Nemotron-3-Nano-Omni (trust_remote_code) sets main_input_name = "pixel_values", which its
processor returns as a list of (3, H, W) tiles for a batch of images of different sizes.
Trainer counts FLOPs from inputs[main_input_name].numel() (transformers 4 through
PreTrainedModel.estimate_tokens), so the first training step failed with
"'list' object has no attribute 'numel'". CPU only.
"""

from collections import UserDict

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Trainer


@pytest.fixture(scope = "module")
def trainer():
    try:
        from unsloth_zoo.temporary_patches.misc import patch_trainer_flops_list_main_input
    except ImportError:  # unsloth-zoo without the patch: the tests show the original failure
        patch_trainer_flops_list_main_input = lambda: None
    patch_trainer_flops_list_main_input()

    class TinyPixelMain(LlamaForCausalLM):
        main_input_name = "pixel_values"

    config = LlamaConfig(
        vocab_size = 32, hidden_size = 16, intermediate_size = 32, num_hidden_layers = 1,
        num_attention_heads = 2, num_key_value_heads = 1, max_position_embeddings = 64,
    )
    trainer = Trainer.__new__(Trainer)
    trainer.model = TinyPixelMain(config)
    return trainer


def test_list_main_input_counted(trainer):
    n_params = trainer.model.num_parameters(exclude_embeddings = True)
    tiles = [torch.zeros(3, 8, 8), torch.zeros(3, 8, 12)]
    assert trainer.floating_point_ops({"pixel_values": tiles}) == 6 * (192 + 288) * n_params
    assert trainer.floating_point_ops({"pixel_values": tuple(tiles)}) == 6 * (192 + 288) * n_params


def test_batch_feature_list_main_input_counted(trainer):
    # Collators return a BatchFeature, a UserDict and not a dict.
    n_params = trainer.model.num_parameters(exclude_embeddings = True)
    tiles = [torch.zeros(3, 8, 8), torch.zeros(3, 8, 12)]
    assert trainer.floating_point_ops(UserDict({"pixel_values": tiles})) == 6 * (192 + 288) * n_params


def test_tensor_main_input_unchanged(trainer):
    # A tensor main input still goes through transformers' own count.
    n_params = trainer.model.num_parameters(exclude_embeddings = True)
    pixel_values = torch.zeros(2, 3, 8, 8)
    assert trainer.floating_point_ops({"pixel_values": pixel_values}) == 6 * 384 * n_params


def test_list_of_non_tensors_counts_zero(trainer):
    assert trainer.floating_point_ops({"pixel_values": ["a.png", "b.png"]}) == 0
