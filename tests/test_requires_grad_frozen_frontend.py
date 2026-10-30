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

"""Audio towers with a frozen front end (Parakeet / Canary conv subsampling, wav2vec2 feature encoder) must skip its
backward; checkpointed layers still get gradients, but front-end activations and input gradients are never kept."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from unsloth_zoo.peft_utils import requires_grad_for_gradient_checkpointing


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias = False)

    def __call__(self, hidden_states, **kwargs):
        # transformers' GradientCheckpointingLayer: module hooks run inside the reentrant checkpoint
        return checkpoint(super().__call__, hidden_states, use_reentrant = True, **kwargs)

    def forward(self, hidden_states):
        return hidden_states + torch.tanh(self.proj(hidden_states))


class _Subsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 8, 3, stride = 2, padding = 1)
        self.seen_input_requires_grad = []
        self.conv.register_forward_hook(
            lambda m, args, out: self.seen_input_requires_grad.append(args[0].requires_grad)
        )

    def forward(self, input_features, attention_mask = None):
        return self.conv(input_features.transpose(1, 2)).transpose(1, 2)


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.subsampling = _Subsampling()
        self.layers = nn.ModuleList([_Layer() for _ in range(2)])
        self.input_scale = 2.0

    def forward(self, input_features, attention_mask = None):
        hidden_states = self.subsampling(input_features, attention_mask)
        hidden_states = hidden_states * self.input_scale
        hidden_states = nn.functional.dropout(hidden_states, p = 0.0, training = self.training)
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states


class _CTC(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()

    def forward(self, input_features):
        return self.encoder(input_features)


def _model():
    torch.manual_seed(0)
    model = _CTC()
    for name, param in model.named_parameters():
        param.requires_grad_("layers" in name)
    return model


def _grads(model, features):
    model.zero_grad(set_to_none = True)
    model(features).square().mean().backward()
    return [layer.proj.weight.grad.clone() for layer in model.encoder.layers]


def test_frozen_frontend_skips_backward_and_keeps_layer_grads():
    features = torch.randn(2, 16, 4)
    reference = _model()
    for layer in reference.encoder.layers:  # no checkpointing: the exact gradients
        layer.__class__ = type("_Plain", (_Layer,), {"__call__": nn.Module.__call__})
    expected = _grads(reference, features)

    model = _model()
    requires_grad_for_gradient_checkpointing(model)
    got = _grads(model, features)
    for g, e in zip(got, expected):
        assert g.abs().sum() > 0
        torch.testing.assert_close(g, e)
    # the frozen conv never saw an input that requires grad, so autograd skipped its backward
    assert model.encoder.subsampling.seen_input_requires_grad == [False]
    assert len(model.encoder._forward_pre_hooks) == 0
    assert len(model.encoder.subsampling._forward_hooks) > 0


def test_hooks_are_not_stacked_on_a_second_call():
    model = _model()
    requires_grad_for_gradient_checkpointing(model)
    requires_grad_for_gradient_checkpointing(model)
    assert len(model.encoder.subsampling._forward_hooks) == 1
    assert all(g.abs().sum() > 0 for g in _grads(model, torch.randn(2, 16, 4)))


class _ReassignedEncoder(_Encoder):
    def forward(self, input_features, attention_mask = None):
        hidden_states = self.subsampling(input_features, attention_mask)
        hidden_states = input_features.new_zeros(hidden_states.shape) + input_features.mean()
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states


def test_layers_not_fed_by_the_frontend_keep_the_input_hook():
    model = _model()
    model.encoder.__class__ = _ReassignedEncoder
    requires_grad_for_gradient_checkpointing(model)
    assert len(model.encoder._forward_pre_hooks) > 0
    assert all(g.abs().sum() > 0 for g in _grads(model, torch.randn(2, 16, 4)))


def test_trainable_frontend_keeps_the_input_hook():
    model = _model()
    model.encoder.subsampling.conv.weight.requires_grad_(True)
    requires_grad_for_gradient_checkpointing(model)
    assert len(model.encoder._forward_pre_hooks) > 0
    _grads(model, torch.randn(2, 16, 4))
    assert model.encoder.subsampling.conv.weight.grad is not None


class _FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 8, 5, stride = 5)
        self._requires_grad = True

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        hidden_states = input_values[:, None]
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True
        return self.conv(hidden_states).transpose(1, 2)


class _W2V(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = _FeatureEncoder()
        self.encoder = nn.ModuleDict({"layers": nn.ModuleList([_Layer() for _ in range(2)])})

    def forward(self, input_values):
        hidden_states = self.feature_extractor(input_values)
        for layer in self.encoder["layers"]:
            hidden_states = layer(hidden_states)
        return hidden_states


def test_frozen_feature_encoder_is_marked_frozen():
    torch.manual_seed(0)
    model = _W2V()
    for name, param in model.named_parameters():
        param.requires_grad_("layers" in name)
    requires_grad_for_gradient_checkpointing(model)
    assert model.feature_extractor._requires_grad is False


def test_trainable_feature_encoder_is_left_alone():
    torch.manual_seed(0)
    model = _W2V()
    for name, param in model.named_parameters():
        param.requires_grad_(True)
    requires_grad_for_gradient_checkpointing(model)
    assert model.feature_extractor._requires_grad is True
