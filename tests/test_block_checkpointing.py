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

"""Tests unsloth_zoo.block_checkpointing on tiny diffusers, transformers and plain models (CPU).

Loaded from its file, like test_block_swap: ``import unsloth_zoo`` needs unsloth installed."""

import functools
import importlib.util
import os

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "block_checkpointing_under_test", os.path.join(_HERE, "unsloth_zoo", "block_checkpointing.py"))
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)


def _counting_checkpoint():
    calls = []

    def run(function, *args, **kwargs):
        calls.append(bc._owner(function))
        return checkpoint(function, *args, use_reentrant = False, **kwargs)

    return run, calls


def _grads(model, step):
    model.zero_grad(set_to_none = True)
    step(model).backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _assert_same(a, b):
    assert a.keys() == b.keys() and a
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol = 1e-5, atol = 1e-6, msg = name)


class _Stack(nn.Module):
    def __init__(self, n = 6):
        super().__init__()
        self.inp = nn.Linear(8, 8)
        self.layers = nn.ModuleList(nn.Sequential(nn.Linear(8, 8), nn.GELU()) for _ in range(n))

    def forward(self, x):
        x = self.inp(x)
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.mark.parametrize("every_n", [1, 2, 3])
def test_plain_module_checkpoints_every_nth_block_with_identical_grads(every_n):
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    ref = _Stack()
    model = _Stack()
    model.load_state_dict(ref.state_dict())
    run, calls = _counting_checkpoint()
    assert bc.enable_block_gradient_checkpointing(model, every_n, checkpoint_function = run) == "modules"
    step = lambda m: m(x).pow(2).mean()  # noqa: E731
    _assert_same(_grads(ref, step), _grads(model, step))
    assert len(calls) == len(range(0, 6, every_n))
    # Idempotent: re-enabling with another N replaces the wrappers instead of stacking them.
    calls.clear()
    bc.enable_block_gradient_checkpointing(model, 1, checkpoint_function = run)
    _grads(model, step)
    assert len(calls) == 6
    model.eval()
    calls.clear()
    with torch.no_grad():
        model(x)
    assert not calls


def test_every_n_counts_callers_in_forward_order():
    run, calls = _counting_checkpoint()
    function = bc.every_n_checkpoint_function(2, checkpoint_function = run)
    blocks = [nn.Linear(2, 2) for _ in range(5)]
    x = torch.randn(1, 2, requires_grad = True)
    for _ in range(2):
        for block in blocks:
            function(block, x)
    # Blocks 0, 2, 4 are checkpointed on every forward; partials of a bound __call__ resolve to the same owner.
    assert calls == [blocks[0], blocks[2], blocks[4]] * 2
    calls.clear()
    function(functools.partial(blocks[2].__call__), x)
    assert calls == [blocks[2]]


def test_diffusers_transformer_checkpoints_with_identical_grads():
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    config = dict(
        patch_size = 1, in_channels = 4, num_layers = 2, num_single_layers = 2, attention_head_dim = 8,
        num_attention_heads = 2, joint_attention_dim = 16, pooled_projection_dim = 8, axes_dims_rope = (4, 2, 2),
    )
    ref = diffusers.FluxTransformer2DModel(**config).train()
    model = diffusers.FluxTransformer2DModel(**config).train()
    model.load_state_dict(ref.state_dict())
    run, calls = _counting_checkpoint()
    assert bc.enable_block_gradient_checkpointing(model, 2, checkpoint_function = run) == "diffusers"
    inputs = dict(
        hidden_states = torch.randn(1, 16, 4), encoder_hidden_states = torch.randn(1, 4, 16),
        pooled_projections = torch.randn(1, 8), timestep = torch.tensor([0.5]),
        img_ids = torch.zeros(16, 3), txt_ids = torch.zeros(4, 3),
    )
    step = lambda m: m(**inputs).sample.pow(2).mean()  # noqa: E731
    _assert_same(_grads(ref, step), _grads(model, step))
    # 2 double + 2 single blocks, every 2nd caller checkpointed.
    assert len(calls) == 2


def test_sdxl_style_unet_skips_blocks_when_every_n_is_above_one():
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    config = dict(
        sample_size = 8, in_channels = 4, out_channels = 4, layers_per_block = 1, block_out_channels = (8, 16),
        down_block_types = ("CrossAttnDownBlock2D", "DownBlock2D"), up_block_types = ("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim = 8, attention_head_dim = 2, norm_num_groups = 4,
    )
    inputs = dict(sample = torch.randn(1, 4, 8, 8), timestep = torch.tensor([10]),
                  encoder_hidden_states = torch.randn(1, 3, 8))
    step = lambda m: m(**inputs).sample.pow(2).mean()  # noqa: E731
    counts = {}
    ref = diffusers.UNet2DConditionModel(**config).train()
    for every_n in (1, 2):
        model = diffusers.UNet2DConditionModel(**config).train()
        model.load_state_dict(ref.state_dict())
        run, calls = _counting_checkpoint()
        bc.enable_block_gradient_checkpointing(model, every_n, checkpoint_function = run)
        _assert_same(_grads(ref, step), _grads(model, step))
        counts[every_n] = len(calls)
    # The UNet also checkpoints resnets, outside its _repeated_blocks list: N=2 must still halve the recomputes.
    assert counts[1] > 2 and counts[2] == (counts[1] + 1) // 2


def test_transformers_model_checkpoints_with_identical_grads():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size = 32, hidden_size = 16, intermediate_size = 32, num_hidden_layers = 4,
        num_attention_heads = 2, num_key_value_heads = 2, max_position_embeddings = 16,
    )
    ref = transformers.LlamaForCausalLM(config).train()
    model = transformers.LlamaForCausalLM(config).train()
    model.load_state_dict(ref.state_dict())
    run, calls = _counting_checkpoint()
    assert bc.enable_block_gradient_checkpointing(model, 2, checkpoint_function = run) == "transformers"
    ids = torch.randint(0, 32, (1, 8))
    step = lambda m: m(input_ids = ids, labels = ids, use_cache = False).loss  # noqa: E731
    _assert_same(_grads(ref, step), _grads(model, step))
    assert len(calls) == 2


def test_default_checkpoint_is_looked_up_per_call(monkeypatch):
    seen = []
    real = torch.utils.checkpoint.checkpoint
    function = bc.every_n_checkpoint_function(1)
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", lambda f, *a, **k: seen.append(k) or real(f, *a, **k))
    function(nn.Linear(2, 2), torch.randn(1, 2, requires_grad = True))
    assert seen == [{"use_reentrant": False}]
    # An Unsloth shim patched over torch's checkpoint is bypassed for the pristine function it stashed.
    pristine = []
    monkeypatch.setattr(
        torch.utils.checkpoint, "_unsloth_pristine_checkpoint",
        lambda f, *a, **k: pristine.append(k) or real(f, *a, **k), raising = False,
    )
    function(nn.Linear(2, 2), torch.randn(1, 2, requires_grad = True))
    assert seen == [{"use_reentrant": False}] and pristine == [{"use_reentrant": False}]


class _Scaled(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x, scale = 1.0):
        return self.lin(x) * scale


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_block_keyword_arguments_reach_the_checkpointed_forward(use_reentrant):
    torch.manual_seed(0)
    ref = nn.ModuleList([_Scaled(), _Scaled()])
    model = nn.ModuleList([_Scaled(), _Scaled()])
    model.load_state_dict(ref.state_dict())
    bc.enable_block_gradient_checkpointing(model, use_reentrant = use_reentrant)
    x = torch.randn(2, 4, requires_grad = True)

    def step(m):
        h = x
        for block in m:
            h = block(h, scale = 2.0)
        return h.pow(2).mean()

    _assert_same(_grads(ref, step), _grads(model, step))


def test_find_repeated_blocks_prefers_declared_classes_then_the_heaviest_list():
    model = _Stack(3)
    assert bc.find_repeated_blocks(model) == list(model.layers)
    model._no_split_modules = ["Linear"]
    assert all(isinstance(m, nn.Linear) for m in bc.find_repeated_blocks(model))
    with pytest.raises(ValueError):
        bc.enable_block_gradient_checkpointing(nn.Linear(2, 2))
