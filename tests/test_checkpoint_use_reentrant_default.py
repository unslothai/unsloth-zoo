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

"""Pins the torch / transformers checkpoint behaviour UnslothCheckpointFunction (UCF) routing depends on.

UCF replaces `torch.utils.checkpoint.CheckpointFunction`, so it only runs on the reentrant path. Two routes reach it:
- `model._set_gradient_checkpointing()` (training_utils) installs transformers' default `gradient_checkpointing_func`,
  torch's own `checkpoint` bound at import, called without `use_reentrant`: it relies on torch treating
  `use_reentrant=None` as reentrant (with a warning, not an error) and looking `CheckpointFunction` up at call time.
- `gradient_checkpointing_enable()` passes `use_reentrant=False` on transformers 5.x (True on 4.57), so the zoo swaps
  `transformers.modeling_utils.checkpoint` for `unsloth_checkpoint`, which forces the reentrant path.
Checked against torch 2.15 `_checkpoint_impl` (None -> warn -> True) and transformers 5.22."""

import warnings

import pytest
import torch
import torch.utils.checkpoint as torch_checkpoint


@pytest.fixture
def counted(monkeypatch):
    """Stand-in for UCF: torch's reentrant Function, counting how often the reentrant path takes it."""
    calls = []
    original = torch_checkpoint.CheckpointFunction

    class CountingCheckpointFunction(original):
        @staticmethod
        def forward(ctx, run_function, preserve_rng_state, *args):
            calls.append(1)
            return original.forward(ctx, run_function, preserve_rng_state, *args)

    monkeypatch.setattr(torch_checkpoint, "CheckpointFunction", CountingCheckpointFunction)
    return calls


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(vocab_size = 64, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
                         num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 32)
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.train()
    return model


def _step(model):
    ids = torch.randint(0, 64, (2, 8))
    model.get_input_embeddings().weight.requires_grad_(True)
    model(input_ids = ids, labels = ids).loss.backward()


def test_omitted_use_reentrant_is_reentrant_and_only_warns(counted):
    x = torch.randn(4, 4, requires_grad = True)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        torch_checkpoint.checkpoint(lambda t: t.sin() * 2, x).sum().backward()
    assert counted == [1], "use_reentrant=None no longer takes the reentrant CheckpointFunction"
    assert any("use_reentrant" in str(w.message) for w in caught)
    assert x.grad is not None


def test_set_gradient_checkpointing_default_reaches_the_reentrant_function(counted):
    pytest.importorskip("transformers")
    model = _tiny_llama()
    model._set_gradient_checkpointing()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _step(model)
    assert len(counted) == model.config.num_hidden_layers, counted


def test_gradient_checkpointing_enable_default_and_the_unsloth_checkpoint_swap(counted, monkeypatch):
    transformers = pytest.importorskip("transformers")
    import transformers.modeling_utils as modeling_utils
    from unsloth_zoo import gradient_checkpointing as zoo_gc
    model = _tiny_llama()
    model.gradient_checkpointing_enable()
    _step(model)
    major = int(transformers.__version__.split(".")[0])
    # transformers 5.x defaults to use_reentrant=False (bypasses UCF); 4.57 to True.
    assert (len(counted) > 0) == (major < 5), (transformers.__version__, counted)

    counted.clear()
    # What patch_unsloth_smart_gradient_checkpointing does to transformers; unsloth_checkpoint calls UCF itself,
    # so point it at the counting stand-in to observe the route.
    monkeypatch.setattr(modeling_utils, "checkpoint", zoo_gc.unsloth_checkpoint)
    monkeypatch.setattr(zoo_gc, "UnslothCheckpointFunction", torch_checkpoint.CheckpointFunction)
    model = _tiny_llama()
    model.gradient_checkpointing_enable()
    _step(model)
    assert len(counted) == model.config.num_hidden_layers, counted
