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

"""A GradientCheckpointingLayer called only by keyword must still get gradients.

The Llama 4 and Mllama vision encoders call `encoder_layer(hidden_state = hidden_states, ...)`,
so GradientCheckpointingLayer hands the checkpointer `partial(layer.__call__, **kwargs)` and no
positional input. A reentrant checkpoint then sees nothing requiring grad and every LoRA in the
layer stays at zero. Mllama's cross-attention layers take the vision states by keyword, so once
the vision tower trains, each nested backward walks that shared graph and the second one raises.
CPU only, except the smart-offload case.
"""

import functools

import pytest
import torch
import torch.nn as nn
from transformers.modeling_layers import GradientCheckpointingLayer

from unsloth_zoo.gradient_checkpointing import unsloth_gradient_checkpoint
from unsloth_zoo.temporary_patches.misc import patch_GradientCheckpointingLayer_keyword_inputs


@pytest.fixture(autouse = True)
def _patched():
    original = GradientCheckpointingLayer.__call__
    patch_GradientCheckpointingLayer_keyword_inputs()
    yield
    GradientCheckpointingLayer.__call__ = original


class _Layer(GradientCheckpointingLayer):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias = False)

    def forward(self, hidden_state, attention_mask = None, scale = 1.0):
        return (hidden_state + scale * self.proj(hidden_state),)


_CHECKPOINTERS = {
    "torch_reentrant": functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant = True),
    "unsloth_gradient_checkpoint": unsloth_gradient_checkpoint,
}


def _layers(checkpointer, device = "cpu"):
    torch.manual_seed(0)
    layers = nn.ModuleList([_Layer() for _ in range(2)]).to(device).train()
    for layer in layers:
        layer.gradient_checkpointing = checkpointer is not None
        layer._gradient_checkpointing_func = checkpointer
    return layers


def _grads(layers, by_keyword, device = "cpu"):
    torch.manual_seed(1)
    x = torch.randn(2, 8, device = device, requires_grad = True)
    h = x
    for layer in layers:
        if by_keyword:
            h = layer(hidden_state = h, attention_mask = None, scale = 0.5)[0]
        else:
            h = layer(h, attention_mask = None, scale = 0.5)[0]
    h.sum().backward()
    return [layer.proj.weight.grad for layer in layers], x.grad


@pytest.mark.parametrize("name", sorted(_CHECKPOINTERS))
def test_keyword_only_call_matches_no_checkpointing(name):
    got, got_x = _grads(_layers(_CHECKPOINTERS[name]), by_keyword = True)
    ref, ref_x = _grads(_layers(None), by_keyword = True)
    for a, b in zip(got, ref):
        assert a is not None
        assert torch.allclose(a, b)
    assert torch.allclose(got_x, ref_x)


@pytest.mark.parametrize("name", sorted(_CHECKPOINTERS))
def test_positional_call_is_unchanged(name):
    got, _ = _grads(_layers(_CHECKPOINTERS[name]), by_keyword = False)
    ref, _ = _grads(_layers(None), by_keyword = False)
    for a, b in zip(got, ref):
        assert torch.allclose(a, b)


def test_without_the_patch_a_keyword_only_call_loses_the_gradient():
    # Guards the tests above: stock transformers leaves the reentrant checkpoint no input.
    call = GradientCheckpointingLayer.__call__
    GradientCheckpointingLayer.__call__ = getattr(call, "_unsloth_original", call)
    try:
        with pytest.raises(RuntimeError, match = "does not require grad"):
            _grads(_layers(_CHECKPOINTERS["torch_reentrant"]), by_keyword = True)
    finally:
        GradientCheckpointingLayer.__call__ = call


class _CrossLayer(GradientCheckpointingLayer):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias = False)

    def forward(self, hidden_states, cross_attention_states = None, use_cache = False):
        return (hidden_states + self.proj(cross_attention_states),)


@pytest.mark.parametrize("name", sorted(_CHECKPOINTERS))
def test_a_keyword_tensor_shared_by_several_layers_backpropagates_once(name):
    # Mllama: text layers take the (now trainable) vision states as a keyword argument.
    def run(checkpointer):
        torch.manual_seed(0)
        tower = nn.Linear(8, 8, bias = False)
        layers = nn.ModuleList([_CrossLayer() for _ in range(2)]).train()
        for layer in layers:
            layer.gradient_checkpointing = checkpointer is not None
            layer._gradient_checkpointing_func = checkpointer
        torch.manual_seed(1)
        vision = tower(torch.randn(2, 8))
        h = torch.randn(2, 8, requires_grad = True)
        for layer in layers:
            h = layer(h, cross_attention_states = vision, use_cache = True)[0]
        (h.sum() + vision.sum()).backward()
        return [tower.weight.grad] + [layer.proj.weight.grad for layer in layers]
    for a, b in zip(run(_CHECKPOINTERS[name]), run(None)):
        assert torch.allclose(a, b)


def test_only_tensors_that_require_grad_are_lifted():
    seen = []
    def checkpointer(function, *args):
        seen.append(len(args))
        return function(*args)
    layer = _layers(checkpointer)[0]
    layer(hidden_state = torch.randn(2, 8), attention_mask = None)
    layer(hidden_state = torch.randn(2, 8, requires_grad = True), attention_mask = torch.ones(2))
    assert seen == [0, 1]


def test_eval_mode_is_untouched():
    layer = _layers(unsloth_gradient_checkpoint)[0].eval()
    out = layer(hidden_state = torch.randn(2, 8, requires_grad = True))[0]
    out.sum().backward()
    assert layer.proj.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_smart_offload_checkpointing_gives_keyword_only_layers_their_gradient():
    # use_gradient_checkpointing = "unsloth" swaps torch's CheckpointFunction for
    # UnslothCheckpointFunction; the layers keep torch's reentrant checkpoint function.
    from unsloth_zoo.gradient_checkpointing import (
        patch_unsloth_smart_gradient_checkpointing,
        unpatch_unsloth_smart_gradient_checkpointing,
    )
    patch_unsloth_smart_gradient_checkpointing(dtype = torch.float32)
    try:
        func = functools.partial(torch.utils.checkpoint._old_checkpoint, use_reentrant = True)
        got, _ = _grads(_layers(func, "cuda"), by_keyword = True, device = "cuda")
        ref, _ = _grads(_layers(None, "cuda"), by_keyword = True, device = "cuda")
        for a, b in zip(got, ref):
            assert a is not None
            assert torch.allclose(a, b, atol = 1e-5)
    finally:
        unpatch_unsloth_smart_gradient_checkpointing()


def test_disabling_a_requested_cache_still_warns(monkeypatch):
    from transformers import modeling_layers
    seen = []
    monkeypatch.setattr(modeling_layers.logger, "warning_once", lambda message: seen.append(message))
    torch.manual_seed(0)
    class _CacheLayer(_CrossLayer):
        pass
    cross = _CacheLayer().train()
    cross.gradient_checkpointing = True
    cross._gradient_checkpointing_func = _CHECKPOINTERS["torch_reentrant"]
    vision = nn.Linear(8, 8)(torch.randn(2, 8))
    cross(torch.randn(2, 8, requires_grad = True), cross_attention_states = vision, use_cache = True)
    assert seen and "use_cache=False" in seen[0] and "_CacheLayer" in seen[0]
