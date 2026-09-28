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

import inspect

import pytest
import torch

modeling = pytest.importorskip("transformers.models.gemma3n.modeling_gemma3n")
from transformers import Gemma3nTextConfig

from unsloth_zoo.temporary_patches import gemma3n as zoo_gemma3n

SHARES_THROUGH_CACHE_ONLY = "shared_kv_states[self.kv_shared_layer_index]" not in inspect.getsource(modeling)


def _tiny_model():
    torch.manual_seed(3407)
    config = Gemma3nTextConfig(
        vocab_size = 128,
        vocab_size_per_layer_input = 128,
        hidden_size = 64,
        hidden_size_per_layer_input = 8,
        intermediate_size = 128,
        num_hidden_layers = 4,
        num_kv_shared_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 32,
        layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        sliding_window = 8,
        laurel_rank = 8,
        activation_sparsity_pattern = [0.0] * 4,
        max_position_embeddings = 64,
    )
    config._attn_implementation = "eager"
    return modeling.Gemma3nForCausalLM(config).eval()


def _losses(model):
    ids = torch.randint(0, 128, (2, 12), generator = torch.Generator().manual_seed(0))
    with torch.no_grad():
        cached = model(input_ids = ids, labels = ids, use_cache = True).loss
        uncached = model(input_ids = ids, labels = ids, use_cache = False).loss
    return cached, uncached


@pytest.fixture
def pristine():
    # Importing unsloth_zoo may already have applied the patch; start from transformers' forward.
    current = modeling.Gemma3nTextAttention.forward
    original = zoo_gemma3n._GEMMA3N_ATTENTION_FORWARD.get("original", current)
    modeling.Gemma3nTextAttention.forward = original
    yield original
    modeling.Gemma3nTextAttention.forward = current


@pytest.mark.skipif(not SHARES_THROUGH_CACHE_ONLY, reason = "transformers already shares KV without a cache")
def test_without_the_patch_the_kv_shared_layers_differ_without_a_cache(pristine):
    cached, uncached = _losses(_tiny_model())
    assert not torch.allclose(cached, uncached)


@pytest.mark.skipif(not SHARES_THROUGH_CACHE_ONLY, reason = "transformers already shares KV without a cache")
def test_the_patch_makes_an_uncached_forward_match_the_cached_one(pristine):
    zoo_gemma3n.patch_Gemma3nTextAttention_kv_sharing()
    assert modeling.Gemma3nTextAttention.forward is not pristine
    cached, uncached = _losses(_tiny_model())
    torch.testing.assert_close(uncached, cached)
    # A second pass leaves the patched forward alone.
    patched = modeling.Gemma3nTextAttention.forward
    zoo_gemma3n.patch_Gemma3nTextAttention_kv_sharing()
    assert modeling.Gemma3nTextAttention.forward is patched


def _grads(reentrant):
    torch.manual_seed(3407)
    model = _tiny_model().train()
    if reentrant is not None:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": reentrant})
    ids = torch.randint(0, 128, (2, 12), generator = torch.Generator().manual_seed(0))
    model(input_ids = ids, labels = ids, use_cache = False).loss.backward()
    return {name: param.grad for name, param in model.named_parameters() if param.grad is not None}


def _max_relative_error(reference, other):
    return max(float((other[name] - grad).norm() / (grad.norm() + 1e-12)) for name, grad in reference.items())


@pytest.mark.skipif(SHARES_THROUGH_CACHE_ONLY, reason = "only transformers >= 5.6 shares KV by itself")
def test_without_the_patch_reentrant_checkpointing_drops_shared_kv_gradients(pristine):
    # The first checkpointed forward stores keys and values with no graph, and the shared layers
    # are recomputed in backward before their source layer.
    assert _max_relative_error(_grads(None), _grads(True)) > 0.05


@pytest.mark.parametrize("reentrant", [True, False], ids = ["reentrant", "non_reentrant"])
def test_checkpointed_gradients_match_the_uncheckpointed_ones(pristine, reentrant):
    zoo_gemma3n.patch_Gemma3nTextAttention_kv_sharing()
    reference = _grads(None)
    assert _max_relative_error(reference, _grads(reentrant)) < 1e-5
