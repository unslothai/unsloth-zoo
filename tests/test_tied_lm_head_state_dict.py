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

"""A tied lm_head is excused from the difference check, but the per-key compare then
KeyError-ed on it anyway (Idefics3). Excused must not mean values can disagree unseen.
"""
import pytest
import torch

from unsloth_zoo.vllm_utils import (
    TIED_EMBED_KEYS,
    TIED_LM_HEAD_KEYS,
    assert_same_state_dict,
)

PAIRS = [
    ("lm_head.weight", "model.embed_tokens.weight"),
    ("model.lm_head.weight", "model.embed_tokens.weight"),
    ("model.language_model.lm_head.weight", "model.language_model.embed_tokens.weight"),
    ("model.text_model.lm_head.weight", "model.text_model.embed_tokens.weight"),
]


def w(value = 0.0):
    return torch.full((4, 4), value)


def test_every_tied_name_is_covered_by_both_sets():
    assert "model.text_model.lm_head.weight" in TIED_LM_HEAD_KEYS
    assert "model.text_model.embed_tokens.weight" in TIED_EMBED_KEYS
    for head, embed in PAIRS:
        assert head in TIED_LM_HEAD_KEYS
        assert embed in TIED_EMBED_KEYS


@pytest.mark.parametrize("head, embed", PAIRS)
def test_tied_lm_head_may_be_missing_from_either_side(head, embed):
    base = {embed: w()}
    assert_same_state_dict(dict(base), {**base, head: w()})
    assert_same_state_dict({**base, head: w()}, dict(base))
    assert_same_state_dict({**base, head: w()}, {**base, head: w()})


@pytest.mark.parametrize("head, embed", PAIRS)
def test_a_tied_lm_head_that_disagrees_is_still_reported(head, embed):
    base = {embed: w(0.0)}
    with pytest.raises(RuntimeError):
        assert_same_state_dict({**base, head: w(1.0)}, dict(base))


def test_genuine_key_differences_still_raise():
    base = {"model.embed_tokens.weight": w()}
    with pytest.raises(RuntimeError):
        assert_same_state_dict(dict(base), {**base, "model.layers.0.q_proj.weight": w()})
    with pytest.raises(RuntimeError):
        assert_same_state_dict({**base, "model.layers.0.q_proj.weight": w()}, dict(base))


def test_genuine_value_differences_still_raise():
    with pytest.raises(RuntimeError):
        assert_same_state_dict({"a.weight": w(0.0)}, {"a.weight": w(1.0)})


def test_identical_state_dicts_pass():
    sd = {"model.embed_tokens.weight": w(), "model.layers.0.q_proj.weight": w()}
    assert_same_state_dict(dict(sd), dict(sd))
