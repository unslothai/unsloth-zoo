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

"""The requires-grad hook search reads each ancestor forward's source once per call, not once per
trainable parameter (an MoE LoRA has tens of thousands), and still picks the same hook targets."""

import functools
import inspect
import types

import torch.nn as nn

import unsloth_zoo.peft_utils as peft_utils
from unsloth_zoo.peft_utils import requires_grad_for_gradient_checkpointing


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias = False)
        self.v_proj = nn.Linear(4, 4, bias = False)

    def forward(self, hidden_states):
        return self.v_proj(self.q_proj(hidden_states))


class _Backbone(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([_Layer() for _ in range(n)])

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states)
        return hidden_states


class _Outer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.model = _Backbone(n)

    def forward(self, input_ids):
        return self.model(input_ids)


class _Twin(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = _Backbone(2)
        self.b = _Backbone(2)

    def forward(self, input_ids):
        return self.a(input_ids), self.b(input_ids)


def _freeze_all_but_layers(model):
    for name, param in model.named_parameters():
        param.requires_grad_(".layers." in name)
    return model


def _hooked(model):
    return [
        (name, [getattr(h, "__qualname__", "") for h in module._forward_hooks.values()])
        for name, module in model.named_modules()
        if module._forward_hooks
    ]


def _run(model, monkeypatch):
    calls = []
    real = inspect.getsource

    def counting(obj):
        calls.append(obj)
        return real(obj)

    monkeypatch.setattr(peft_utils.inspect, "getsource", counting)
    requires_grad_for_gradient_checkpointing(model)
    return len(calls)


def test_source_read_once_per_forward(monkeypatch):
    small = _freeze_all_but_layers(_Outer(2))
    small_calls = _run(small, monkeypatch)
    model = _freeze_all_but_layers(_Outer(16))
    trainable = sum(p.requires_grad for p in model.parameters())
    assert trainable == 32
    calls = _run(model, monkeypatch)
    assert calls == small_calls
    assert _hooked(small) == _hooked(model)
    assert _hooked(model) == [
        ("model.embed_tokens", ["requires_grad_for_gradient_checkpointing.<locals>.requires_grad_post_hook_eager"]),
    ]


def test_instances_sharing_a_forward_get_the_same_targets(monkeypatch):
    model = _freeze_all_but_layers(_Twin())
    calls = _run(model, monkeypatch)
    assert calls < sum(p.requires_grad for p in model.parameters())
    hooked = [name for name, _ in _hooked(model)]
    assert hooked == ["a.embed_tokens", "b.embed_tokens"]


def test_unreadable_and_partial_forwards_behave_as_before(monkeypatch):
    # A forward with no source (exec) and a partial both fall through to the next ancestor.
    model = _freeze_all_but_layers(_Outer(3))
    namespace = {}
    exec("def forward(self, *args, **kwargs):\n    return type(self).forward(self, *args, **kwargs)\n", namespace)
    model.model.forward = types.MethodType(namespace["forward"], model.model)
    for layer in model.model.layers:
        layer.forward = functools.partial(type(layer).forward, layer)
    requires_grad_for_gradient_checkpointing(model)
    expected = _freeze_all_but_layers(_Outer(3))
    expected.model.forward = types.MethodType(namespace["forward"], expected.model)
    requires_grad_for_gradient_checkpointing(expected)
    assert _hooked(model) == _hooked(expected)
    assert _hooked(model) != []
