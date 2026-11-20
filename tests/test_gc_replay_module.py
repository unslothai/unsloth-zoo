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

"""Unsloth's fused LoRA MLP skips the replay down projection only for this module, so wrappers must resolve to None."""

import functools

import pytest
import torch

from unsloth_zoo import gradient_checkpointing as gc


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.seen = []

    def forward(self, x, scale = 1.0):
        self.seen.append(gc.gradient_checkpoint_replay_module())
        return self.linear(x) * scale

    def helper(self, x):
        return self.forward(x) + 1


def _wrapped(forward):
    @functools.wraps(forward)
    def forward_with_checkpointing(self, *args, **kwargs):
        return forward(self, *args, **kwargs)

    return forward_with_checkpointing


def test_resolves_whole_module_calls_only():
    layer = _Layer()
    resolve = gc._replayed_module
    assert resolve(layer) is layer
    assert resolve(layer.__call__) is layer
    assert resolve(layer.forward) is layer
    assert resolve(functools.partial(torch.nn.Module.__call__.__get__(layer), scale = 2.0)) is layer
    assert resolve(functools.partial(type(layer).forward, layer, scale = 2.0)) is layer
    assert resolve(gc._KeywordArgumentCall(functools.partial(layer.__call__), ("scale",), {})) is layer

    def custom_forward(*inputs):
        return layer(*inputs)

    assert resolve(custom_forward) is None
    custom_forward._unsloth_checkpoint_module = layer
    assert resolve(custom_forward) is layer
    assert resolve(layer.helper) is None
    assert resolve(functools.partial(layer.helper)) is None
    assert resolve(lambda x: layer(x) + 1) is None
    assert resolve(functools.partial(type(layer).forward, _Layer())) is not layer
    assert resolve(None) is None


def test_wrapped_class_forward_resolves(monkeypatch):
    # transformers >= 5.23 / Unsloth block compile wrap type(self).forward.
    inner = _Layer.forward
    monkeypatch.setattr(_Layer, "forward", _wrapped(inner))
    layer = _Layer()
    assert gc._replayed_module(functools.partial(inner, layer)) is layer


_SMART = pytest.param(
    "smart",
    marks = pytest.mark.skipif(not torch.cuda.is_available(), reason = "the offloading checkpointer needs CUDA"),
)


@pytest.mark.parametrize("kind", ["unsloth", "offloaded", _SMART])
def test_replay_module_set_only_during_replay(kind, monkeypatch):
    layer = _Layer()
    x = torch.randn(3, 4, requires_grad = True)
    function = functools.partial(torch.nn.Module.__call__.__get__(layer), scale = 2.0)
    if kind == "unsloth":
        out = gc.unsloth_gradient_checkpoint(function, x, use_reentrant = True)
    elif kind == "offloaded":
        out = gc.Unsloth_Offloaded_Gradient_Checkpointer.apply(function, x)
    else:
        gc.initialize_unsloth_gradient_checkpointing(torch.float32)
        x = x.detach().cuda().requires_grad_(True)
        layer.cuda()
        out = gc.UnslothCheckpointFunction.apply(function, False, x)
    assert layer.seen == [None]
    out.sum().backward()
    assert layer.seen == [None, layer]
    assert gc.gradient_checkpoint_replay_module() is None
    assert not gc.in_gradient_checkpoint_recompute()


def test_nested_marker_restores_function():
    layer = _Layer()
    with gc._gradient_checkpoint_recompute_marker(layer):
        assert gc.gradient_checkpoint_replay_module() is layer
        with gc._gradient_checkpoint_recompute_marker(None):
            assert gc.gradient_checkpoint_replay_module() is None
        assert gc.gradient_checkpoint_replay_module() is layer
    assert gc.gradient_checkpoint_replay_module() is None


def test_module_with_forward_hook_is_never_reported():
    layer = _Layer()
    handle = layer.register_forward_hook(lambda module, args, output: output.square())
    with gc._gradient_checkpoint_recompute_marker(layer):
        assert gc.gradient_checkpoint_replay_module() is None
    handle.remove()
    handle = torch.nn.modules.module.register_module_forward_hook(lambda module, args, output: None)
    try:
        with gc._gradient_checkpoint_recompute_marker(layer):
            assert gc.gradient_checkpoint_replay_module() is None
    finally:
        handle.remove()
    with gc._gradient_checkpoint_recompute_marker(layer):
        assert gc.gradient_checkpoint_replay_module() is layer
