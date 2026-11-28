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

"""Layer-major windows over decoder stacks with per-call structure beyond the hidden states: a mapping passed to every
layer that layers write but never read back (Gemma 4 without KV sharing), and per-layer inputs whose gradient
reaches nothing but the first decoder input (Gemma 4's per-layer embeddings under LoRA). Bitwise against running the
micro-batches one after another; reading another layer's entry, or a per-layer input feeding a trained weight,
aborts before any decoder gradient exists."""

import collections
import functools
import importlib.util
import os

import pytest
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "block_swap_layer_major_structure_under_test", os.path.join(_HERE, "unsloth_zoo", "block_swap_layer_major.py"))
lm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lm)


class _Layer(torch.nn.Module):
    """x -> tanh(lin(x) + per_layer); writes its output into `shared` under its own key, reads `read_key` if set."""
    def __init__(self, i, w = 16, read_key = None):
        super().__init__()
        self.lin = torch.nn.Linear(w, w)
        self.i, self.read_key = i, read_key

    def forward(self, x, per_layer = None, shared = None):
        if per_layer is not None:
            x = x + per_layer[..., self.i, :]
        if shared is not None and self.read_key is not None and self.read_key in shared:
            x = x + shared[self.read_key]
        h = torch.tanh(self.lin(x))
        if shared is not None:
            shared[f"layer{self.i}"] = h * 0.5
        return h


class _Model(torch.nn.Module):
    def __init__(self, n = 3, w = 16, read_key = None, train_projection = False, user_dict = False):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(50, w)
        self.embed.weight.requires_grad_(False)
        self.projection = torch.nn.Linear(w, n * w)
        self.projection.requires_grad_(train_projection)
        self.layers = torch.nn.ModuleList(
            [_Layer(i, w, read_key if i == n - 1 else None) for i in range(n)])
        self.head = torch.nn.Linear(w, 50)
        self.n, self.w, self.user_dict = n, w, user_dict

    def forward(self, ids):
        x = self.embed(ids).requires_grad_(True)   # what enable_input_require_grads does
        per_layer = self.projection(x).view(*x.shape[:-1], self.n, self.w)
        shared = collections.UserDict() if self.user_dict else {}
        for layer in self.layers:
            x = layer(x, per_layer, shared = shared)
        return torch.nn.functional.cross_entropy(self.head(x).flatten(0, 1), ids.flatten())


def _ids(m):
    return torch.randint(0, 50, (2, 6), generator = torch.Generator().manual_seed(m))


def _step(model, m, losses):
    loss = model(_ids(m)) / 3
    loss.backward()
    losses.append(loss.detach())


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _compare(**kw):
    model = _Model(**kw)
    standard = []
    for m in range(3):
        _step(model, m, standard)
    base = _grads(model)
    model.zero_grad(set_to_none = True)
    window = []
    results, why = lm.run_layer_major(model.layers, [functools.partial(_step, model, m, window) for m in range(3)],
                                      model = model)
    return base, standard, _grads(model), window, why


@pytest.mark.parametrize("user_dict", [False, True])
def test_written_but_unread_mapping_and_frozen_per_layer_inputs_engage_bitwise(user_dict):
    base, standard, grads, window, why = _compare(user_dict = user_dict)
    assert why is None
    assert all(torch.equal(a, b) for a, b in zip(standard, window))
    assert base.keys() == grads.keys() and "layers.0.lin.weight" in base
    for k in base:
        assert torch.equal(base[k], grads[k]), k


def test_reading_another_layers_mapping_entry_aborts():
    base, standard, grads, window, why = _compare(read_key = "layer0")
    assert "a decoder layer read a mapping entry another decoder layer wrote" in why
    assert not grads


def test_a_per_layer_input_feeding_a_trained_weight_aborts():
    base, standard, grads, window, why = _compare(train_projection = True)
    assert "a decoder layer input other than the hidden states needs gradients" in why
    assert not grads


def test_writes_reach_the_real_mapping_in_layer_order():
    model = _Model()
    seen = []

    def fn(m):
        x = model.embed(_ids(m)).requires_grad_(True)
        shared = {}
        for layer in model.layers:
            x = layer(x, shared = shared)
        seen.append(list(shared))
        x.sum().backward()

    results, why = lm.run_layer_major(model.layers, [functools.partial(fn, m) for m in range(2)], model = model)
    assert why is None and seen == [["layer0", "layer1", "layer2"]] * 2
