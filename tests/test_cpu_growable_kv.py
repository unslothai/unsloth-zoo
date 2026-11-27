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

"""CPU generate: growable full-attention KV layers (no per-token torch.cat of the whole cache), same tokens."""

import pytest
import torch

cache_utils = pytest.importorskip("transformers.cache_utils")
if not hasattr(cache_utils, "DynamicLayer"):
    pytest.skip("transformers before layered caches keeps the stock DynamicCache", allow_module_level = True)

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402


def _tiny(**overrides):
    torch.manual_seed(0)
    kwargs = dict(vocab_size = 128, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
                  num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, max_position_embeddings = 512,
                  tie_word_embeddings = False)
    kwargs.update(overrides)
    model = Qwen3ForCausalLM(Qwen3Config(**kwargs)).eval()
    model.generation_config.pad_token_id = 0
    model.generation_config.eos_token_id = None
    return model


def _attach(model):
    from unsloth_zoo.cpu.loader import FastCPUModel
    FastCPUModel._attach(model)
    return model


def _left_padded_batch():
    torch.manual_seed(1)
    ids = torch.randint(3, 128, (3, 9))
    mask = torch.ones_like(ids)
    ids[0, :4] = 0; mask[0, :4] = 0   # left padding, ragged lengths
    ids[1, :1] = 0; mask[1, :1] = 0
    return ids, mask


def _generate(model, monkeypatch, growable, **kwargs):
    monkeypatch.setenv("UNSLOTH_GROWABLE_KV", "1" if growable else "0")
    ids, mask = _left_padded_batch()
    torch.manual_seed(2)
    with torch.no_grad():
        return model.generate(input_ids = ids, attention_mask = mask, return_dict_in_generate = True,
                              output_scores = True, **kwargs)


def _layer_names(out):
    cache = getattr(out, "past_key_values", None)
    return {type(layer).__name__ for layer in getattr(cache, "layers", ())}


def _kv_storage_pointers(model, monkeypatch, growable, steps):
    """Storage address of layer 0's keys after every forward of a greedy generate."""
    monkeypatch.setenv("UNSLOTH_GROWABLE_KV", "1" if growable else "0")
    seen = []

    def hook(module, args, kwargs, output):
        cache = kwargs.get("past_key_values", None) or kwargs.get("cache", None)
        keys = cache.layers[0].keys
        seen.append(keys.untyped_storage().data_ptr())

    handle = model.register_forward_hook(hook, with_kwargs = True)
    try:
        ids, mask = _left_padded_batch()
        with torch.no_grad():
            model.generate(input_ids = ids, attention_mask = mask, max_new_tokens = steps, min_new_tokens = steps,
                           do_sample = False)
    finally:
        handle.remove()
    return seen


def test_cpu_generate_decode_does_not_reallocate_kv_every_token(monkeypatch):
    """The flip: stock DynamicLayer re-allocates (torch.cat) the whole K/V each decode step; the growable layer
    writes in place, so 20 decode steps inside one 256-slot chunk keep one buffer."""
    monkeypatch.setenv("UNSLOTH_GROWABLE_KV_MIN", "256")
    model = _attach(_tiny())
    stock = _kv_storage_pointers(model, monkeypatch, growable = False, steps = 20)
    grown = _kv_storage_pointers(model, monkeypatch, growable = True, steps = 20)
    # torch.cat allocates the new K before freeing the old one, so consecutive steps never share a buffer.
    assert len(stock) == 20 and all(a != b for a, b in zip(stock, stock[1:])), "stock cache copies every step"
    # Prefill goes through the stock path; the first decode step moves it into the chunked buffer, then stays.
    assert len(set(grown[1:])) == 1, grown


@pytest.mark.parametrize("chunk", ["256", "4"])
def test_greedy_tokens_and_scores_match_stock_with_left_padding(monkeypatch, chunk):
    from unsloth_zoo.cpu import kv_cache
    monkeypatch.setenv("UNSLOTH_GROWABLE_KV_MIN", chunk)  # "4" regrows many times inside the run
    model = _attach(_tiny())
    stock = _generate(model, monkeypatch, False, max_new_tokens = 30, min_new_tokens = 30, do_sample = False)
    engaged = kv_cache.ENGAGED[0]
    grown = _generate(model, monkeypatch, True, max_new_tokens = 30, min_new_tokens = 30, do_sample = False)
    assert kv_cache.ENGAGED[0] == engaged + 1
    assert _layer_names(stock) == {"DynamicLayer"}
    assert _layer_names(grown) == {"GrowableDynamicLayer"}
    assert torch.equal(stock.sequences, grown.sequences)
    for a, b in zip(stock.scores, grown.scores):
        torch.testing.assert_close(a, b, rtol = 1e-5, atol = 1e-5)


def test_sampling_matches_stock(monkeypatch):
    model = _attach(_tiny())
    kwargs = dict(max_new_tokens = 24, min_new_tokens = 24, do_sample = True, top_k = 0, temperature = 1.0,
                  num_return_sequences = 2)
    stock = _generate(model, monkeypatch, False, **kwargs)
    grown = _generate(model, monkeypatch, True, **kwargs)
    assert _layer_names(grown) == {"GrowableDynamicLayer"}
    assert stock.sequences.shape[0] == 6
    assert torch.equal(stock.sequences, grown.sequences)


def test_beam_search_keeps_stock_cache(monkeypatch):
    model = _attach(_tiny())
    kwargs = dict(max_new_tokens = 12, num_beams = 2, do_sample = False)
    stock = _generate(model, monkeypatch, False, **kwargs)
    grown = _generate(model, monkeypatch, True, **kwargs)
    assert "GrowableDynamicLayer" not in _layer_names(grown)
    assert torch.equal(stock.sequences, grown.sequences)


def test_sliding_window_cache_keeps_stock_layers(monkeypatch):
    from unsloth_zoo.cpu import kv_cache
    model = _attach(_tiny(use_sliding_window = True, sliding_window = 8, max_window_layers = 0,
                          layer_types = ["sliding_attention", "full_attention"]))
    engaged = kv_cache.ENGAGED[0]
    stock = _generate(model, monkeypatch, False, max_new_tokens = 16, do_sample = False)
    grown = _generate(model, monkeypatch, True, max_new_tokens = 16, do_sample = False)
    assert kv_cache.ENGAGED[0] == engaged
    assert "DynamicSlidingWindowLayer" in _layer_names(grown)
    assert "GrowableDynamicLayer" not in _layer_names(grown)
    assert torch.equal(stock.sequences, grown.sequences)


def test_hybrid_and_custom_caches_are_left_alone():
    from transformers.cache_utils import DynamicCache, DynamicLayer
    from unsloth_zoo.cpu.kv_cache import make_growable

    class OwnCache(DynamicCache):
        pass

    assert not make_growable(OwnCache())  # model-specific cache classes (hybrid / linear attention)
    mixed = DynamicCache(config = Qwen3Config(num_hidden_layers = 2, sliding_window = 8, use_sliding_window = True,
                                              max_window_layers = 0,
                                              layer_types = ["sliding_attention", "full_attention"]))
    kinds = [type(layer) for layer in mixed.layers]
    assert not make_growable(mixed)
    assert [type(layer) for layer in mixed.layers] == kinds
    lazy = DynamicCache()
    assert make_growable(lazy) and lazy.layer_class_to_replicate is not DynamicLayer


def test_hybrid_linear_attention_model_keeps_stock_cache(monkeypatch):
    """Qwen3-Next mixes linear-attention (GDN) and full-attention layers: its cache is left as transformers builds it
    (Qwen3NextDynamicCache on 4.57, a DynamicCache with LinearAttentionLayer entries on 5.x)."""
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3NextForCausalLM"):
        pytest.skip("no Qwen3-Next in this transformers")
    from unsloth_zoo.cpu import kv_cache
    torch.manual_seed(0)
    config = transformers.Qwen3NextConfig(
        vocab_size = 128, hidden_size = 64, intermediate_size = 128, num_attention_heads = 4, num_key_value_heads = 2,
        head_dim = 16, num_hidden_layers = 2, layer_types = ["linear_attention", "full_attention"], num_experts = 4,
        num_experts_per_tok = 2, moe_intermediate_size = 32, shared_expert_intermediate_size = 32,
        linear_num_value_heads = 4, linear_num_key_heads = 2, linear_key_head_dim = 16, linear_value_head_dim = 16)
    model = _attach(transformers.Qwen3NextForCausalLM(config).eval())
    model.generation_config.pad_token_id = 0
    model.generation_config.eos_token_id = None
    engaged = kv_cache.ENGAGED[0]
    stock = _generate(model, monkeypatch, False, max_new_tokens = 8, do_sample = False)
    grown = _generate(model, monkeypatch, True, max_new_tokens = 8, do_sample = False)
    assert kv_cache.ENGAGED[0] == engaged
    assert "GrowableDynamicLayer" not in _layer_names(grown)
    assert torch.equal(stock.sequences, grown.sequences)


def test_caller_cache_is_not_converted(monkeypatch):
    from transformers.cache_utils import DynamicCache
    model = _attach(_tiny())
    cache = DynamicCache()
    out = _generate(model, monkeypatch, True, max_new_tokens = 6, do_sample = False, past_key_values = cache)
    assert _layer_names(out) == {"DynamicLayer"}


def test_cache_kept_from_inference_mode_is_reusable(monkeypatch):
    """Multi-turn reuse: a cache returned by a generate under inference_mode, passed back to a generate outside it,
    is reallocated rather than written in place (an inference tensor cannot be updated outside inference_mode)."""
    model = _attach(_tiny())
    ids, mask = _left_padded_batch()
    results = []
    for growable in (False, True):
        monkeypatch.setenv("UNSLOTH_GROWABLE_KV", "1" if growable else "0")
        with torch.inference_mode():
            first = model.generate(input_ids = ids, attention_mask = mask, max_new_tokens = 5, min_new_tokens = 5,
                                   do_sample = False, return_dict_in_generate = True)
        follow = torch.cat([first.sequences, torch.full((3, 2), 7)], dim = 1)
        with torch.no_grad():
            second = model.generate(input_ids = follow, attention_mask = torch.cat([mask, torch.ones(3, 7, dtype = mask.dtype)], 1),
                                    past_key_values = first.past_key_values, max_new_tokens = 6, min_new_tokens = 6,
                                    do_sample = False)
        results.append(second)
    assert torch.equal(results[0], results[1])

    from unsloth_zoo.cpu.kv_cache import GrowableDynamicLayer
    layer = GrowableDynamicLayer()
    with torch.inference_mode():
        layer.update(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))
        layer.update(torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))  # now in its own (inference) buffer
    k = torch.randn(1, 2, 1, 4)
    with torch.no_grad():
        keys, _ = layer.update(k, torch.randn(1, 2, 1, 4))
    assert keys.shape[-2] == 5 and torch.equal(keys[:, :, -1:], k) and not keys.is_inference()


def test_layer_matches_stock_through_cache_edits(monkeypatch):
    """crop / batch_select_indices / batch_repeat_interleave / reorder reassign keys; the next update must adopt
    them and never write into a tensor the layer did not allocate."""
    import inspect
    from transformers.cache_utils import DynamicLayer
    from unsloth_zoo.cpu.kv_cache import GrowableDynamicLayer
    monkeypatch.setenv("UNSLOTH_GROWABLE_KV_MIN", "4")
    torch.manual_seed(0)
    stock, grown = DynamicLayer(), GrowableDynamicLayer()
    crop_by_count = "tokens_to_remove" in inspect.signature(DynamicLayer.crop).parameters

    def step(batch, t):
        k = torch.randn(batch, 2, t, 8); v = torch.randn(batch, 2, t, 4)
        a = stock.update(k, v); b = grown.update(k.clone(), v.clone())
        for x, y in zip(a, b):
            assert torch.equal(x, y)

    step(3, 5)
    for _ in range(9):
        step(3, 1)
    stock.crop(-3 if crop_by_count else 7); grown.crop(-3 if crop_by_count else 7)
    step(3, 1)
    index = torch.tensor([2, 0])
    stock.batch_select_indices(index); grown.batch_select_indices(index)
    step(2, 1)
    held = grown.keys.clone()
    stock.batch_repeat_interleave(2); grown.batch_repeat_interleave(2)
    view = grown.keys
    step(4, 1)
    assert torch.equal(view, held.repeat_interleave(2, dim = 0))  # adopted tensor not written in place
    for _ in range(6):
        step(4, 1)
    assert grown.get_seq_length() == stock.get_seq_length()


def test_kill_switch_and_cuda_device_keep_stock_cache(monkeypatch):
    from transformers import GenerationConfig
    from transformers.generation.configuration_utils import GenerationMode
    from unsloth_zoo.cpu.kv_cache import install
    model = _attach(_tiny())
    assert install(model)  # idempotent once bound
    out = _generate(model, monkeypatch, False, max_new_tokens = 4, do_sample = False)
    assert _layer_names(out) == {"DynamicLayer"}

    def prepared(device):
        monkeypatch.setattr(type(model), "device", property(lambda self: torch.device(device)))
        model_kwargs = {}
        model._prepare_cache_for_generation(generation_config = GenerationConfig(max_new_tokens = 4),
                                            model_kwargs = model_kwargs,
                                            generation_mode = GenerationMode.GREEDY_SEARCH,
                                            batch_size = 1, max_cache_length = 16)
        cache = model_kwargs.get("past_key_values", None) or model_kwargs.get("cache", None)
        return getattr(cache, "layer_class_to_replicate", None) or type(cache.layers[0])

    monkeypatch.setenv("UNSLOTH_GROWABLE_KV", "1")
    assert prepared("cpu").__name__ == "GrowableDynamicLayer"
    assert prepared("cuda").__name__ == "DynamicLayer"  # CUDA keeps Unsloth's GPU decode paths
