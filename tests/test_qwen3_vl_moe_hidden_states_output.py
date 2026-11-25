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

"""Qwen3-VL-MoE hidden-states output must not pass rope_deltas (dropped in 5.24, raised TypeError)."""

import dataclasses
import inspect

import pytest
import torch

modeling = pytest.importorskip("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe")
if not hasattr(modeling, "Qwen3VLMoeCausalLMOutputWithPast"):
    pytest.skip("Qwen3VLMoeCausalLMOutputWithPast absent on this transformers", allow_module_level = True)


class _ModelOutput:
    def __init__(self, hidden, cache, rope_deltas):
        self.hidden_states = None
        self.attentions = None
        self.rope_deltas = rope_deltas
        self.past_key_values = cache  # 4.x field, 5.24 BC property
        self._items = {"last_hidden_state": hidden, "cache": cache}

    def __getitem__(self, index):
        return self._items["last_hidden_state"] if index == 0 else None

    def get(self, key, default = None):
        return self._items.get(key, default)


class _Self:
    def __init__(self, outputs):
        self.outputs = outputs
        self.seen = {}

    def model(self, **kwargs):
        self.seen = kwargs
        return self.outputs


def test_hidden_states_output_is_constructible(monkeypatch):
    cls = modeling.Qwen3VLMoeForConditionalGeneration
    monkeypatch.setattr(cls, "forward", cls.forward)  # restore after the patch
    from unsloth_zoo.temporary_patches import qwen3_vl_moe

    qwen3_vl_moe.patch_qwen3_vl_moe()
    assert getattr(cls.forward, "_unsloth_hidden_states_wrapper", False), "hidden-states wrapper not installed"

    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "1")
    hidden = torch.randn(1, 5, 8)
    cache = object()
    rope_deltas = torch.zeros(1, 1)
    fake = _Self(_ModelOutput(hidden, cache, rope_deltas))
    stock = getattr(cls.forward, "__wrapped__", cls.forward)
    call_name = "cache" if "cache" in inspect.signature(stock).parameters else "past_key_values"
    out = cls.forward(
        fake, input_ids = torch.ones(1, 5, dtype = torch.long), logits_to_keep = 2, **{call_name: cache},
    )

    torch.testing.assert_close(out.logits, hidden[:, -2:, :])
    fields = {f.name for f in dataclasses.fields(modeling.Qwen3VLMoeCausalLMOutputWithPast)}
    cache_field = "cache" if "cache" in fields else "past_key_values"
    assert out.get(cache_field) is cache
    if "rope_deltas" in fields:
        assert out.rope_deltas is rope_deltas
    assert sum(k in fake.seen for k in ("cache", "past_key_values")) == 1, fake.seen
    assert fake.seen.get(cache_field) is cache
