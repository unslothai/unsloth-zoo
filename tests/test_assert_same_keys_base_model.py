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

"""Regression for PR 95: assert_same_keys must resolve inner_model with the guarded find_lora_base_model walk, since
model.base_model.model raised AttributeError when base_model has no .model (Phi-3, Phi-4-mini); the same walk keeps
key comparison like-for-like."""
import pytest
import torch
import torch.nn as nn

from unsloth_zoo.saving_utils import assert_same_keys, find_lora_base_model


class _Cfg:
    tie_word_embeddings = False


class _BaseNoModel(nn.Module):
    """A PEFT base whose object has NO ``.model`` attribute (Phi-3 / Phi-4-mini layout)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)  # -> state_dict key "linear.weight"
        self.config = _Cfg()


class _PeftLikeNoInnerModel(nn.Module):
    """model.base_model exists but has no .model attribute, so the old model.base_model.model access raised AttributeError."""

    def __init__(self, base):
        super().__init__()
        self.base_model = base


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        self.config = _Cfg()


class _LoraModel(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.model = inner


class _PeftLike(nn.Module):
    def __init__(self, lora):
        super().__init__()
        self.base_model = lora


def test_assert_same_keys_survives_peft_base_without_model():
    base = _BaseNoModel()
    model = _PeftLikeNoInnerModel(base)
    assert find_lora_base_model(model) is base
    # new_state_dict is built the way create_lora_statistics does: from find_lora_base_model(model).
    new_state_dict = dict(base.state_dict())  # {"linear.weight": ...}
    # Pre-fix: AttributeError at `model.base_model.model`. Post-fix: validates cleanly.
    assert_same_keys(model, new_state_dict)


def test_assert_same_keys_normal_peft_layout_unchanged():
    inner = _Inner()
    model = _PeftLike(_LoraModel(inner))
    assert find_lora_base_model(model) is inner
    assert_same_keys(model, dict(inner.state_dict()))  # matching keys -> no error


def test_assert_same_keys_still_detects_real_mismatch():
    # The guard must not mask genuine key mismatches.
    inner = _Inner()
    model = _PeftLike(_LoraModel(inner))
    bad = dict(inner.state_dict())
    bad["not_a_real.weight"] = inner.linear.weight
    with pytest.raises(RuntimeError):
        assert_same_keys(model, bad)
