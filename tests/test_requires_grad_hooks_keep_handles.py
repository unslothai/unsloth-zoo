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

import torch

from unsloth_zoo.peft_utils import requires_grad_for_gradient_checkpointing


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora = torch.nn.Linear(4, 4)

    def forward(self, hidden_states):
        return self.lora(hidden_states)


class _Tower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(), _Block()])

    def get_input_embeddings(self):
        raise NotImplementedError

    def forward(self, hidden_states = None):
        for blk in self.blocks:
            hidden_states = blk(hidden_states)
        return hidden_states


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _Tower()

    def forward(self, hidden_states = None):
        return self.visual(hidden_states = hidden_states)


def _trainable_model():
    model = _Model()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def test_removed_forward_hook_stops_running():
    model = _trainable_model()
    calls = []
    handle = model.visual.blocks[0].register_forward_hook(lambda module, args, output: calls.append(1))
    requires_grad_for_gradient_checkpointing(model)
    model(hidden_states = torch.randn(2, 4))
    assert calls == [1], calls
    handle.remove()
    model(hidden_states = torch.randn(2, 4))
    assert calls == [1], calls


def test_removed_forward_pre_hook_stops_running():
    model = _trainable_model()
    calls = []

    def count(module, args, kwargs):
        calls.append(1)

    handle = model.visual.register_forward_pre_hook(count, with_kwargs = True)
    requires_grad_for_gradient_checkpointing(model)
    names = [getattr(h, "__name__", "") for h in model.visual._forward_pre_hooks.values()]
    assert "requires_grad_pre_hook" in names, names
    model(hidden_states = torch.randn(2, 4))
    assert calls == [1], calls
    handle.remove()
    model(hidden_states = torch.randn(2, 4))
    assert calls == [1], calls


def test_dropped_hook_leaves_no_flag_ids_and_the_kept_hook_keeps_its_own():
    model = _trainable_model()
    block = model.visual.blocks[0]
    calls = []

    def make_inputs_require_grads(module, args, kwargs, output):
        calls.append("dropped")

    def kept(module, args, kwargs, output):
        calls.append("kept")

    block.register_forward_hook(make_inputs_require_grads, with_kwargs = True, always_call = True)
    handle = block.register_forward_hook(kept, with_kwargs = True, always_call = True)
    requires_grad_for_gradient_checkpointing(model)
    assert list(block._forward_hooks.values()) == [kept]
    assert list(block._forward_hooks_with_kwargs) == [handle.id]
    assert list(block._forward_hooks_always_called) == [handle.id]
    model(hidden_states = torch.randn(2, 4))
    assert calls == ["kept"], calls
    handle.remove()
    assert not block._forward_hooks
    assert not block._forward_hooks_with_kwargs
    assert not block._forward_hooks_always_called
