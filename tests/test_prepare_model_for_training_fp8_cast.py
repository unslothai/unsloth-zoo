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

"""Full finetuning refuses to cast a scaled FP8 weight to the training dtype, which drops its scale."""

import pytest
import torch
import torch.nn as nn


class _Cfg:
    torch_dtype = torch.bfloat16


class _FP8Linear(nn.Module):
    # transformers' FP8Linear layout: an fp8 weight next to its block scale.
    def __init__(self, scale_name):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(256, 256).to(torch.float8_e4m3fn), requires_grad = False)
        if scale_name is not None:
            setattr(self, scale_name, nn.Parameter(torch.ones(2, 2), requires_grad = False))


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 256, 256).to(torch.float8_e4m3fn), requires_grad = False)
        self.gate_up_proj_scale_inv = nn.Parameter(torch.ones(2, 2, 2), requires_grad = False)


class _Model(nn.Module):
    def __init__(self, proj):
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].proj = proj
        self.config = _Cfg()


def _prepare(model, full_finetuning = True):
    from unsloth_zoo.training_utils import prepare_model_for_training

    return prepare_model_for_training(
        model,
        use_gradient_checkpointing = False,
        use_reentrant = False,
        full_finetuning = full_finetuning,
        float32_mixed_precision = False,
    )


@pytest.mark.parametrize("scale_name", ["weight_scale_inv", "weight_scale"])
def test_full_finetuning_refuses_scaled_fp8_weight(scale_name):
    model = _Model(_FP8Linear(scale_name))
    with pytest.raises(RuntimeError, match = "drop its scale"):
        _prepare(model)
    assert model.layers[0].proj.weight.dtype == torch.float8_e4m3fn


def test_full_finetuning_refuses_scaled_fp8_expert_stack():
    model = _Model(_Experts())
    with pytest.raises(RuntimeError, match = "gate_up_proj"):
        _prepare(model)


def test_unscaled_fp8_weight_is_still_cast():
    # No scale to lose: the old cast stays.
    model = _Model(_FP8Linear(None))
    _prepare(model)
    assert model.layers[0].proj.weight.dtype == torch.bfloat16


def test_lora_leaves_frozen_scaled_fp8_weight_alone():
    model = _Model(_FP8Linear("weight_scale_inv"))
    _prepare(model, full_finetuning = False)
    assert model.layers[0].proj.weight.dtype == torch.float8_e4m3fn
