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

"""past_key_values replacements must install over a 5.24 `cache` forward and receive `cache=`."""

import inspect

import pytest

from unsloth_zoo.temporary_patches import utils as U


def _stock_class():
    class Attention:
        def forward(self, hidden_states, position_embeddings, attention_mask, cache = None, **kwargs):
            raise AssertionError("stock forward must have been replaced")

    return Attention


def _replacement(seen):
    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values = None, **kwargs):
        seen.append((past_key_values, dict(kwargs)))
        return hidden_states

    return forward


@pytest.mark.parametrize("match_level", ["strict", "relaxed"])
def test_past_key_values_replacement_installs_over_stock_cache(match_level):
    cls, seen = _stock_class(), []
    assert U.patch_function(cls, "forward", _replacement(seen), match_level = match_level)
    assert list(inspect.signature(cls.forward).parameters)[4] == "cache"

    cache = object()
    assert cls().forward(1, 2, 3, cache = cache, use_cache = True) == 1
    assert cls().forward(1, 2, 3, cache) == 1  # positional
    assert cls().forward(1, 2, 3, past_key_values = cache) == 1  # Unsloth's own callers
    assert [s[0] for s in seen] == [cache, cache, cache]
    assert seen[0][1] == {"use_cache": True}, "cache= must not leak into **kwargs"


def test_patch_function_past_key_values_accepts_stock_cache():
    cls, seen = _stock_class(), []

    def singular(self, hidden_states, position_embeddings, attention_mask, past_key_value = None, **kwargs):
        raise AssertionError("singular variant must not be chosen")

    assert U.patch_function_past_key_values(cls, "forward", [singular, _replacement(seen)])
    cache = object()
    cls().forward(1, 2, 3, cache = cache)
    assert seen and seen[0][0] is cache


def test_older_stock_names_are_left_alone():
    class Attention:
        def forward(self, hidden_states, past_key_values = None, **kwargs):
            pass

    seen = []

    def forward(self, hidden_states, past_key_values = None, **kwargs):
        seen.append(past_key_values)

    assert U.patch_function(Attention, "forward", forward)
    assert Attention.forward is forward, "no adapter when stock still says past_key_values"


def test_cache_params_replacement_adapts_too():
    class Mixer:
        def forward(self, hidden_states, cache = None, attention_mask = None):
            pass

    seen = []

    def forward(self, hidden_states, cache_params = None, attention_mask = None):
        seen.append(cache_params)

    assert U.patch_function(Mixer, "forward", forward)
    cache = object()
    Mixer().forward(1, cache = cache)
    assert seen == [cache]


def test_cache_kwarg_name_and_output_cache():
    def new(self, x, cache = None, **kwargs):
        pass

    def old(self, x, past_key_values = None, **kwargs):
        pass

    def none(self, x):
        pass

    assert U.cache_kwarg_name(new) == "cache"
    assert U.cache_kwarg_name(old) == "past_key_values"
    assert U.cache_kwarg_name(none) == "past_key_values"
    assert U.output_cache({"cache": 1}) == 1
    assert U.output_cache({"past_key_values": 2}) == 2
    assert U.output_cache(None) is None


def test_process_return_uses_the_output_class_field_name():
    transformers = pytest.importorskip("transformers")
    from transformers.modeling_outputs import BaseModelOutputWithPast

    cache = object()
    out = U.process_return(BaseModelOutputWithPast, {"last_hidden_state": 1, "past_key_values": cache})
    assert U.output_cache(out) is cache
    assert out.past_key_values is cache
