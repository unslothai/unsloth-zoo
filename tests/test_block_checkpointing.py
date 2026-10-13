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
import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "block_checkpointing_under_test", os.path.join(_HERE, "unsloth_zoo", "block_checkpointing.py"))
bc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bc  # pickle imports the wrapper class by module name
_spec.loader.exec_module(bc)


@pytest.fixture(autouse = True)
def _no_stashed_pristine_checkpoint(monkeypatch):
    # The stash is process-global and set once by any real patch_unsloth_gradient_checkpointing, so a file run
    # earlier in the same xdist worker would route _TorchCheckpoint past the checkpoint these tests patch.
    monkeypatch.delattr(torch.utils.checkpoint, "_unsloth_pristine_checkpoint", raising = False)


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

    def forward(self, x, scale = 1.0, debug = False):
        # debug shares its name with a torch.utils.checkpoint keyword: it must still reach the block.
        return self.lin(x) * (scale + float(debug))


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
            h = block(h, scale = 2.0, debug = True)
        return h.pow(2).mean()

    _assert_same(_grads(ref, step), _grads(model, step))


def test_copied_and_saved_plain_models_keep_their_own_checkpointed_blocks(monkeypatch):
    import copy
    import io

    torch.manual_seed(0)
    model = _Stack(4)
    run, calls = _counting_checkpoint()
    bc.enable_block_gradient_checkpointing(model, checkpoint_function = run)
    clone = copy.deepcopy(model)
    x = torch.randn(2, 8)
    clone(x).pow(2).mean().backward()
    # The copy trains its own parameters and checkpoints via torch's function (the custom one stays with the original).
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is not None for p in clone.parameters())
    assert not calls and all(bc._is_wrapped(b) for b in clone.layers)
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only = False)
    torch.testing.assert_close(loaded(x), model(x))
    # The loaded model still checkpoints its own blocks (through torch's function: the custom one is not saved).
    assert all(bc._is_wrapped(b) and b.forward.block is b for b in loaded.layers)
    seen = []
    real = torch.utils.checkpoint.checkpoint
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", lambda f, *a, **k: seen.append(1) or real(f, *a, **k))
    loaded(x).pow(2).mean().backward()
    assert len(seen) == 4


def test_a_forward_patched_before_enabling_is_kept_and_restored():
    model = _Stack(2)
    block = model.layers[0]
    seen = []
    original = block.forward

    def patched(x):
        seen.append(1)
        return original(x)

    block.forward = patched
    bc.enable_block_gradient_checkpointing(model)
    model(torch.randn(1, 8, requires_grad = True)).sum().backward()
    assert seen  # forward ran (and again on recompute) through the earlier patch
    bc.enable_block_gradient_checkpointing(model, 2)
    assert bc._is_wrapped(model.layers[0]) and block.__dict__["forward"].block is block
    bc._unwrap(block)
    assert block.forward is patched


def test_find_repeated_blocks_prefers_declared_classes_then_the_heaviest_list():
    model = _Stack(3)
    assert bc.find_repeated_blocks(model) == list(model.layers)
    model._no_split_modules = ["Linear"]
    assert all(isinstance(m, nn.Linear) for m in bc.find_repeated_blocks(model))
    with pytest.raises(ValueError):
        bc.enable_block_gradient_checkpointing(nn.Linear(2, 2))


class _NeedsState(nn.Module):
    # A forward that only works while its hooks hold transient state, like block swapping's weight fetch / release.
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        # State used by the first op, as fetched weights are: recompute cannot stop before reaching it.
        # A trailing op that saves nothing: non-reentrant recompute would stop before the post-hook.
        return self.lin(x * self.scale) + 1


def _transient_state_stack(events):
    stack = nn.ModuleList([_NeedsState() for _ in range(3)])
    for block in stack:
        def pre(module, args):
            module.scale = 2.0
            events.append("pre")

        def post(module, args, output):
            del module.scale

        block.register_forward_pre_hook(pre)
        block.register_forward_hook(post)
    return stack


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_recompute_runs_the_blocks_hooks_again(use_reentrant):
    torch.manual_seed(0)
    ref_events, events = [], []
    ref, model = _transient_state_stack(ref_events), _transient_state_stack(events)
    model.load_state_dict(ref.state_dict())
    bc.enable_block_gradient_checkpointing(model, use_reentrant = use_reentrant)
    x = torch.randn(2, 4, requires_grad = True)

    def step(m):
        h = x
        for block in m:
            h = block(h)
        return h.pow(2).mean()

    _assert_same(_grads(ref, step), _grads(model, step))
    assert len(ref_events) == 3 and len(events) == 6
    assert not any("scale" in block.__dict__ for block in model)


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_an_argument_rewriting_pre_hook_is_applied_once(use_reentrant):
    torch.manual_seed(0)
    ref = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
    model = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
    model.load_state_dict(ref.state_dict())
    for block in (*ref, *model):
        block.register_forward_pre_hook(lambda module, args: (args[0] * 3.0,))
    bc.enable_block_gradient_checkpointing(model, use_reentrant = use_reentrant)
    x = torch.randn(2, 4, requires_grad = True)

    def step(m):
        h = x
        for block in m:
            h = torch.tanh(block(h))
        return h.pow(2).mean()

    _assert_same(_grads(ref, step), _grads(model, step))


class _FakeDiffusers(nn.Module):
    def __init__(self, n = 4):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(n)])
        self.func = None

    def enable_gradient_checkpointing(self, gradient_checkpointing_func = None):
        self.func = gradient_checkpointing_func

    def forward(self, x):
        for block in self.blocks:
            x = self.func(block, x)
        return x


@pytest.mark.parametrize("copy_after_running", [False, True])
def test_a_copied_model_checkpoints_the_same_blocks_as_its_original(copy_after_running):
    import copy

    run, calls = _counting_checkpoint()
    model = _FakeDiffusers(3)
    bc.enable_block_gradient_checkpointing(model, 2, checkpoint_function = run)
    x = torch.randn(1, 4, requires_grad = True)
    if copy_after_running:
        model(x).sum().backward()
    clone = copy.deepcopy(model)
    if not copy_after_running:
        model(x).sum().backward()
    calls.clear()
    clone(x).sum().backward()
    assert calls == [clone.blocks[0], clone.blocks[2]]


def test_an_aliased_block_is_one_caller():
    shared, other = nn.Linear(4, 4), nn.Linear(4, 4)
    model = nn.Module()
    model.layers = nn.ModuleList([shared, shared, other])
    bc.enable_block_gradient_checkpointing(model, 2)
    assert bc._is_wrapped(shared) and not bc._is_wrapped(other)


def test_a_global_kwargs_forward_hook_reruns_with_its_signature():
    import inspect

    from torch.nn.modules.module import register_module_forward_hook

    if "with_kwargs" not in inspect.signature(register_module_forward_hook).parameters:
        pytest.skip("global kwargs forward hooks need torch >= 2.6")
    seen = []
    handle = register_module_forward_hook(lambda module, args, kwargs, output: seen.append(kwargs), with_kwargs = True)
    try:
        model = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        bc.enable_block_gradient_checkpointing(model)
        h = torch.randn(2, 4, requires_grad = True)
        for block in model:
            h = torch.tanh(block(h))
        h.sum().backward()
    finally:
        handle.remove()
    assert seen


@pytest.mark.parametrize("every_n", [1, 2])
def test_an_enabled_diffusers_model_still_saves_whole(every_n):
    import io

    model = _FakeDiffusers(3)
    bc.enable_block_gradient_checkpointing(model, every_n)
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only = False)
    x = torch.randn(1, 4, requires_grad = True)
    torch.testing.assert_close(loaded(x), model(x))


def test_a_block_with_a_self_referencing_patched_forward_is_still_freed():
    import gc
    import weakref

    model = _Stack(2)
    block = model.layers[0]
    block.forward = functools.partial(lambda module, x: type(module).forward(module, x), block)
    bc.enable_block_gradient_checkpointing(model)
    model(torch.randn(1, 8, requires_grad = True)).sum().backward()
    ref = weakref.ref(block)
    del model, block
    gc.collect()
    assert ref() is None


def test_re_enabling_a_transformers_model_keeps_one_embedding_hook():
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size = 32, hidden_size = 16, intermediate_size = 32, num_hidden_layers = 2,
        num_attention_heads = 2, num_key_value_heads = 2, max_position_embeddings = 16,
    )
    model = transformers.LlamaForCausalLM(config).train()
    # As PEFT marks it: every transformers version wires the input-grads hook then (4.57 only then), so a re-enable
    # that stacked hooks would show on both.
    model._hf_peft_config_loaded = True
    for every_n in (1, 2, 1):
        bc.enable_block_gradient_checkpointing(model, every_n)
    assert len(model.get_input_embeddings()._forward_hooks) == 1


def test_parameterless_repeated_blocks_are_found():
    model = nn.Module()
    model.acts = nn.ModuleList([nn.GELU() for _ in range(3)])
    assert bc.find_repeated_blocks(model) == list(model.acts)
