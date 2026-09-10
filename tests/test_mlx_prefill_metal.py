# SPDX-License-Identifier: AGPL-3.0-only
from types import SimpleNamespace
import importlib

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx_simulation import mlx_is_simulated

if mlx_is_simulated():
    pytest.skip("Requires native MLX", allow_module_level = True)
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason = "Requires Metal")

import mlx.nn as nn
from unsloth_zoo.mlx import inference as prefill


@pytest.mark.parametrize("rows,chunks", [
    (255, [255]), (256, [256]), (257, [256, 1]),
    (767, [256, 256, 255]), (768, [256, 512]),
    (1792, [256, 512, 1024]), (1807, [256, 512, 1024, 15]),
    (1040, [256, 512, 256, 16]),
])
def test_dynamic_grid_preserves_partial_tail(rows, chunks):
    schedule = prefill.DynamicPrefillSchedule()
    cache = [SimpleNamespace(offset = 0)]
    actual = []
    while rows:
        n = schedule.chunk_size(rows, cache)
        actual.append(n)
        rows -= n
        cache[0].offset += n
    assert actual == chunks


def test_resumed_grid_uses_absolute_offset_and_rejects_inconsistent_cache():
    schedule = prefill.DynamicPrefillSchedule()
    assert schedule.chunk_size(2048, [SimpleNamespace(offset = 768)]) == 1024
    assert schedule.chunk_size(2048, [SimpleNamespace(offset = 0)]) == 256
    for rows, expected in [(255, 0), (256, 256), (767, 256), (768, 768), (1807, 1792), (4096, 3840)]:
        assert schedule.boundary_at(rows) == expected
    with pytest.raises(ValueError, match = "consistent absolute"):
        schedule.chunk_size(512, [SimpleNamespace(offset = 0), SimpleNamespace(offset = 256)])


@pytest.mark.parametrize("width,bits,group,dtype", [(256, 4, 32, mx.float16), (512, 8, 64, mx.bfloat16), (1024, 4, 128, mx.float32)])
def test_canonical_projection_matches_256_row_arithmetic_and_restores(width, bits, group, dtype):
    mx.random.seed(51)
    layers = nn.Sequential(nn.Linear(4096, width), nn.Linear(4096, 2048))
    layers.set_dtype(dtype)
    nn.quantize(layers, bits = bits, group_size = group)
    layers.freeze()
    layers.eval()
    model = SimpleNamespace(language_model = layers)
    x = mx.random.normal((1, 768, 4096)).astype(dtype)
    expected = mx.concatenate([layers.layers[0](x[:, start : start + 256]) for start in range(0, 768, 256)], axis = 1)
    with pytest.raises(RuntimeError, match = "close"):
        with prefill.DynamicPrefillSchedule.arithmetic(model):
            assert type(layers.layers[0]) is not nn.QuantizedLinear
            assert type(layers.layers[1]) is nn.QuantizedLinear
            actual = layers.layers[0](x)
            assert np.array_equal(np.array(actual.view(mx.uint8)), np.array(expected.view(mx.uint8)))
            raise RuntimeError("close")
    assert all(type(layer) is nn.QuantizedLinear for layer in layers.layers)


def test_current_vlm_source_adapts_and_changed_contract_does_not(monkeypatch):
    from mlx_vlm.generate.ar import generate_step
    original = getattr(generate_step, "__wrapped__", generate_step)
    adapted = prefill._adapt_prefill_generate_step(original)
    assert adapted is not None
    assert "_unsloth_prefill_schedule" in __import__("inspect").signature(adapted).parameters
    source = __import__("inspect").getsource(original)
    monkeypatch.setattr(prefill.inspect, "getsource", lambda _: source.replace(
        "n_to_process = min(prefill_step_size, inputs_embeds.shape[1] - 1)",
        "n_to_process = 1",
    ))
    assert prefill._adapt_prefill_generate_step(original) is None


def _prefill_loop(input_ids, model, *, prefill_step_size = 256, prompt_cache = None, **kwargs):
    inputs_embeds = input_ids
    should_chunk = inputs_embeds.shape[1] > 1
    if prefill_step_size is not None and should_chunk:
        while inputs_embeds.shape[1] > 1:
            n_to_process = min(prefill_step_size, inputs_embeds.shape[1] - 1)
            model.calls.append((n_to_process, type(model.language_model.layers[0])))
            prompt_cache[0].offset += n_to_process
            inputs_embeds = inputs_embeds[:, n_to_process:]
    yield type(model.language_model.layers[0])


def test_adapted_loop_uses_schedule_and_restores_before_decode():
    layers = nn.Sequential(nn.Linear(256, 256))
    nn.quantize(layers, bits = 8)
    model = SimpleNamespace(language_model = layers, calls = [])
    adapted = prefill._adapt_prefill_generate_step(_prefill_loop)
    result = list(adapted(
        np.empty((1, 1808, 1)), model,
        prompt_cache = [SimpleNamespace(offset = 0)],
        _unsloth_prefill_schedule = prefill.DynamicPrefillSchedule(model),
    ))
    assert [n for n, _ in model.calls] == [256, 512, 1024, 15]
    assert all(cls is not nn.QuantizedLinear for _, cls in model.calls)
    assert result == [nn.QuantizedLinear]


def test_install_preserves_native_call_forms_and_falls_back_for_native_modes(monkeypatch):
    ar = importlib.import_module("mlx_vlm.generate.ar")
    dispatch = importlib.import_module("mlx_vlm.generate.dispatch")
    calls = []

    def original(input_ids, model, pixel_values, mask, **kwargs):
        calls.append((input_ids, model, pixel_values, mask, kwargs))
        yield "native"

    def adapted(*args, **kwargs):
        yield "dynamic"

    monkeypatch.setattr(ar, "generate_step", original)
    monkeypatch.setattr(dispatch, "generate_step", original)
    monkeypatch.setattr(prefill, "_PREFILL_GENERATE_STEP", None)
    monkeypatch.setattr(prefill, "_adapt_prefill_generate_step", lambda _: adapted)
    assert prefill._install_dynamic_prefill()
    assert dispatch.generate_step is ar.generate_step
    assert list(ar.generate_step(1, 2, 3, 4, max_tokens = 5)) == ["native"]
    assert list(ar.generate_step(input_ids = 6, model = 7, pixel_values = 8, mask = 9)) == ["native"]
    assert calls == [(1, 2, 3, 4, {"max_tokens": 5}), (6, 7, 8, 9, {})]
    model = SimpleNamespace(training = False)
    ids = SimpleNamespace(ndim = 2, shape = (1, 10))
    options = {"_unsloth_prefill_schedule": prefill.DynamicPrefillSchedule(model), "prefill_step_size": 256}
    assert list(ar.generate_step(ids, model, None, None, **options)) == ["dynamic"]
    assert list(ar.generate_step(ids, model, None, None, kv_bits = 4, **options)) == ["dynamic"]
    assert list(ar.generate_step(ids, model, object(), None, **options)) == ["dynamic"]
    assert list(ar.generate_step(ids, model, None, None, audio_features = object(), **options)) == ["dynamic"]
    for extra in ({"prompt_cache": [object()]}, {"draft_model": object()}, {"kv_bits": 4, "kv_quant_scheme": "turboquant"}, {"prompt_cache_checkpoint": object()}):
        assert list(ar.generate_step(ids, model, None, None, **options, **extra)) == ["native"]
    options["prefill_step_size"] = None
    assert list(ar.generate_step(ids, model, None, None, **options)) == ["native"]
    options["prefill_step_size"] = 512
    assert list(ar.generate_step(ids, model, None, None, **options)) == ["dynamic"]
    ids.shape = (2, 10)
    assert list(ar.generate_step(ids, model, None, None, **options)) == ["native"]


@pytest.mark.parametrize("bits,group,dtype", [(4, 32, mx.float16), (8, 64, mx.bfloat16), (4, 128, mx.float32)])
def test_factory_accepts_native_formats_and_declines_training_or_adapters(monkeypatch, bits, group, dtype):
    model = nn.Module()
    model.language_model = nn.Sequential(nn.Linear(256, 256))
    model.set_dtype(dtype)
    nn.quantize(model, bits = bits, group_size = group)
    model.freeze()
    model.eval()
    monkeypatch.setattr(prefill, "_install_dynamic_prefill", lambda: True)
    assert prefill.create_dynamic_prefill_schedule(model) is not None
    model.language_model.layers[0].lora_a = mx.zeros((1, 1))
    model.freeze()
    assert prefill.create_dynamic_prefill_schedule(model) is None
    del model.language_model.layers[0].lora_a
    model.language_model.extra_weight = mx.zeros((1, 1))
    assert prefill.create_dynamic_prefill_schedule(model) is not None
    model.language_model.layers[0].unfreeze(keys = ["scales"])
    assert prefill.create_dynamic_prefill_schedule(model) is not None
    model.train()
    assert prefill.create_dynamic_prefill_schedule(model) is None
    model.eval()
    model.language_model.layers[0] = nn.Linear(256, 256)
    model.eval()
    assert prefill.create_dynamic_prefill_schedule(model) is not None


def test_canonical_attention_uses_fixed_query_and_key_lengths():
    from mlx_vlm.models.base import scaled_dot_product_attention
    mx.random.seed(91)
    queries = mx.random.normal((1, 4, 1024, 256)).astype(mx.bfloat16)
    keys = mx.random.normal((1, 1, 4864, 256)).astype(mx.bfloat16)
    values = mx.random.normal(keys.shape).astype(mx.bfloat16)
    options = dict(cache = None, scale = 256 ** -0.5, mask = "causal")
    expected = mx.concatenate([
        scaled_dot_product_attention(
            queries[..., start:start + 256, :], keys[..., :4096 + start, :],
            values[..., :4096 + start, :], **options,
        ) for start in range(0, 1024, 256)
    ], axis = -2)
    calls = []

    def attention(q, k, v, **kwargs):
        calls.append((q.shape[-2], k.shape[-2], kwargs["mask"]))
        return scaled_dot_product_attention(q, k, v, **kwargs)

    actual = prefill._prefill_attention(attention, queries, keys, values, **options)
    assert calls == [(256, n, "causal") for n in (4096, 4352, 4608, 4864)]
    assert np.array_equal(np.array(actual.view(mx.uint8)), np.array(expected.view(mx.uint8)))
    calls.clear()
    prefill._prefill_attention(attention, queries, keys, values, **{**options, "mask": None})
    assert calls == [(1024, 4864, None)]


def test_attention_instance_scope_restores_native_class_and_global():
    from mlx_vlm.models.idefics3.language import Attention
    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    model = SimpleNamespace(language_model = nn.Sequential(layer))
    original = Attention.__call__.__globals__["scaled_dot_product_attention"]
    with pytest.raises(RuntimeError, match = "stop"):
        with prefill.DynamicPrefillSchedule.arithmetic(model):
            assert type(layer) is not Attention
            assert type(layer).__call__.__globals__["scaled_dot_product_attention"] is not original
            assert Attention.__call__.__globals__["scaled_dot_product_attention"] is original
            raise RuntimeError("stop")
    assert type(layer) is Attention


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_full_precision_projection_preserves_each_chunk(dtype):
    mx.random.seed(17)
    layer = nn.Linear(128, 64)
    layer.set_dtype(dtype)
    layer.eval()
    x = mx.random.normal((1, 768, 128)).astype(dtype)
    expected = mx.concatenate([layer(x[:, i:i + 256]) for i in range(0, 768, 256)], axis = 1)
    with prefill.DynamicPrefillSchedule.arithmetic(SimpleNamespace(language_model = nn.Sequential(layer))):
        assert type(layer) is not nn.Linear
        actual = layer(x)
    assert type(layer) is nn.Linear
    assert np.array_equal(np.array(actual.view(mx.uint8)), np.array(expected.view(mx.uint8)))


@pytest.mark.parametrize("prefix,keep,return_array", [(256, 0, True), (1280, 0, True), (256, 4, True), (1280, 4, True), (256, 4, False), (1280, 4, False)])
def test_rotating_attention_matches_native_chunks_and_cache(prefix, keep, return_array):
    from mlx_vlm.models.cache import RotatingKVCache
    from mlx_vlm.models.base import scaled_dot_product_attention
    mx.random.seed(19)
    q = mx.random.normal((1, 4, 1024, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 1, prefix + 1024, 256)).astype(mx.bfloat16)
    v = mx.random.normal(k.shape).astype(mx.bfloat16)
    caches = [RotatingKVCache(1024, keep = keep) for _ in range(2)]
    for cache in caches:
        cache.update_and_fetch(k[..., :prefix, :], v[..., :prefix, :])
    fixed, dynamic = caches
    expected = []
    for start in range(0, 1024, 256):
        mask = fixed.make_mask(256, return_array = return_array)
        keys, values = fixed.update_and_fetch(k[..., prefix + start:prefix + start + 256, :], v[..., prefix + start:prefix + start + 256, :])
        expected.append(scaled_dot_product_attention(q[..., start:start + 256, :], keys, values, fixed, scale = 0.0625, mask = mask))
    mask = dynamic.make_mask(1024, return_array = return_array)
    keys, values = dynamic.update_and_fetch(k[..., prefix:, :], v[..., prefix:, :])
    actual = prefill._prefill_attention(scaled_dot_product_attention, q, keys, values, dynamic, scale = 0.0625, mask = mask)
    for left, right in [(actual, mx.concatenate(expected, axis = -2)), *zip(dynamic.state, fixed.state)]:
        assert np.array_equal(np.array(left.view(mx.uint8)), np.array(right.view(mx.uint8)))
    assert dynamic.meta_state == fixed.meta_state



def test_shared_rotating_attention_uses_producer_geometry_and_releases_references():
    from mlx_vlm.models.cache import RotatingKVCache
    cache = RotatingKVCache(512)
    keys = mx.zeros((1, 1, 511 + 512, 8))
    values = mx.ones(keys.shape)
    queries = mx.zeros((1, 1, 512, 8))
    mask = mx.ones((512, 1023), dtype = mx.bool_)
    calls = []

    def attention(q, k, v, **kwargs):
        calls.append((q.shape[-2], k.shape[-2], kwargs["mask"].shape))
        return q

    cache.keys, cache.values = keys, values
    cache.offset = cache._idx = keys.shape[-2]
    model = SimpleNamespace(language_model = nn.Sequential())
    with prefill.DynamicPrefillSchedule.arithmetic(model):
        prefill._prefill_attention(attention, queries, keys, values, cache, 1.0, mask)
        producer_calls = calls[:]
        calls.clear()
        prefill._prefill_attention(attention, queries, keys, values, None, 1.0, mask)
        assert calls == producer_calls == [(256, 767, (256, 767)), (256, 767, (256, 767))]
        assert prefill._PREFILL_KV_OWNERS.get()
        prefill.DynamicPrefillSchedule().chunk_size(512, [SimpleNamespace(offset = 256)])
        assert not prefill._PREFILL_KV_OWNERS.get()
    assert prefill._PREFILL_KV_OWNERS.get() is None



@pytest.mark.parametrize("module_name,class_name", [("phi4_siglip", "Attention"), ("olmo", "TransformerBlock")])
def test_factory_requires_cache_offset_and_supported_attention_call(monkeypatch, module_name, class_name):
    from mlx_vlm.models.cache import ArraysCache, KVCache
    Attention = getattr(importlib.import_module(f"mlx_vlm.models.{module_name}.language"), class_name)
    model = nn.Module()
    model.language_model = nn.Sequential(nn.Linear(256, 256))
    model.eval()
    monkeypatch.setattr(prefill, "_install_dynamic_prefill", lambda: True)
    model.language_model.make_cache = lambda: [ArraysCache(size = 2)]
    assert prefill.create_dynamic_prefill_schedule(model) is None
    model.language_model.make_cache = lambda: [KVCache()]
    assert prefill.create_dynamic_prefill_schedule(model) is not None
    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    model.language_model.layers.append(layer)
    model.eval()
    assert prefill.create_dynamic_prefill_schedule(model) is None



def test_factory_declines_nested_pooled_cache_contract(monkeypatch):
    from mlx_vlm.models.cache import CacheList, PoolingCache, RotatingKVCache
    model = nn.Module()
    model.language_model = nn.Sequential(nn.Linear(256, 256))
    model.eval()
    monkeypatch.setattr(prefill, "_install_dynamic_prefill", lambda: True)
    model.language_model.make_cache = lambda: [RotatingKVCache(512), CacheList(RotatingKVCache(512), PoolingCache(4))]
    assert prefill.create_dynamic_prefill_schedule(model) is None



def test_rotating_storage_is_independent_of_attention_values():
    from mlx_vlm.models.cache import RotatingKVCache
    from mlx_vlm.models.base import scaled_dot_product_attention
    mx.random.seed(21)
    q = mx.random.normal((1, 1, 512, 64))
    k = mx.random.normal((1, 1, 1023, 64))
    fixed, dynamic = RotatingKVCache(512), RotatingKVCache(512)
    for cache in (fixed, dynamic):
        cache.update_and_fetch(k[..., :511, :], mx.zeros((1, 1, 511, 0)))
    expected = []
    for start in (0, 256):
        mask = fixed.make_mask(256)
        keys, _ = fixed.update_and_fetch(k[..., 511 + start:767 + start, :], mx.zeros((1, 1, 256, 0)))
        expected.append(scaled_dot_product_attention(q[..., start:start + 256, :], keys, keys, fixed, scale = 0.125, mask = mask))
    mask = dynamic.make_mask(512)
    keys, _ = dynamic.update_and_fetch(k[..., 511:, :], mx.zeros((1, 1, 512, 0)))
    actual = prefill._prefill_attention(scaled_dot_product_attention, q, keys, keys, dynamic, scale = 0.125, mask = mask)
    assert np.array_equal(np.array(actual), np.array(mx.concatenate(expected, axis = -2)))
    assert dynamic.values.shape == fixed.values.shape == (1, 1, 767, 0)
    assert dynamic.meta_state == fixed.meta_state
    for cache in (fixed, dynamic):
        cache.update_and_fetch(k[..., :256, :], mx.zeros((1, 1, 256, 0)))
    assert np.array_equal(np.array(dynamic.keys), np.array(fixed.keys))



def test_dynamic_prefill_scopes_native_attention_helper_methods():
    from mlx_vlm.models.iquestloopcoder.language import Attention
    from mlx_vlm.models.cache import RotatingKVCache
    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    layer.scale = 0.125
    model = SimpleNamespace(language_model = nn.Sequential(layer))
    q = mx.zeros((1, 1, 512, 64))
    cache = RotatingKVCache(512)
    cache.update_and_fetch(mx.ones((1, 1, 511, 64)), mx.ones((1, 1, 511, 64)))
    mask = cache.make_mask(512)
    keys, values = cache.update_and_fetch(q, q)
    original = Attention.attention.__globals__["scaled_dot_product_attention"]
    with prefill.DynamicPrefillSchedule.arithmetic(model):
        assert type(layer).attention.__globals__["scaled_dot_product_attention"] is not original
        output = layer.attention(q, keys, values, mask, cache)
        mx.eval(output)
        assert cache.keys.shape[-2] == cache._idx == 767
    assert type(layer) is Attention
    assert Attention.attention.__globals__["scaled_dot_product_attention"] is original



def test_factory_declines_attention_reading_mutable_cache_state(monkeypatch):
    from mlx_vlm.models.gemma3n.language import Gemma3nAttention as Attention
    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    model = nn.Module()
    model.language_model = nn.Sequential(nn.Linear(256, 256), layer)
    model.eval()
    monkeypatch.setattr(prefill, "_install_dynamic_prefill", lambda: True)
    assert prefill.create_dynamic_prefill_schedule(model) is None


@pytest.mark.parametrize("step", [128, 300, 512, 2048])
def test_configured_schedule_grid_and_projection_arithmetic(step):
    schedule = prefill.DynamicPrefillSchedule(step_size = step)
    offset = 0
    for _ in range(7):
        end = schedule.next_boundary(offset)
        assert end > offset and end % step == 0
        assert schedule.boundary_at(end) == end
        assert schedule.boundary_at(end - 1) == offset
        offset = end
    mx.random.seed(42)
    layer = nn.Linear(128, 128)
    nn.quantize(layer, bits = 8)
    x = mx.random.normal((2, step * 3, 128))
    expected = mx.concatenate([layer(x[:, i:i + step]) for i in range(0, x.shape[1], step)], axis = 1)
    with prefill.DynamicPrefillSchedule.arithmetic(SimpleNamespace(language_model = layer), step):
        actual = layer(x)
    for row in (0, 1):
        assert np.array_equal(np.array(actual[row].view(mx.uint8)), np.array(expected[row].view(mx.uint8)))
    assert prefill._PREFILL_ARITHMETIC_STEP.get() == 256


@pytest.mark.parametrize("threshold,expected", [(0, [256, 512, 1024]), (300, [256, 256, 256, 1024]), (800, [256, 512, 256, 768])])
def test_quantized_conversion_boundary_is_part_of_snapshot_grid(threshold, expected):
    schedule = prefill.DynamicPrefillSchedule(quantized_kv_start = threshold)
    cache = [SimpleNamespace(offset = 0)]
    actual = []
    while cache[0].offset < 1792:
        size = schedule.chunk_size(1792 - cache[0].offset, cache)
        actual.append(size)
        previous = cache[0].offset
        cache[0].offset += size
        assert schedule.boundary_at(cache[0].offset) == cache[0].offset
        assert schedule.boundary_at(cache[0].offset - 1) == previous
    assert actual == expected


@pytest.mark.parametrize("bits,container", [(4, tuple), (8, list)])
def test_packed_attention_preserves_all_rows_and_cache_components(bits, container):
    from mlx_vlm.models.cache import QuantizedKVCache
    from mlx_vlm.models.base import scaled_dot_product_attention
    mx.random.seed(14)
    q = mx.random.normal((2, 4, 768, 64)).astype(mx.bfloat16)
    k = mx.random.normal((2, 2, 768, 64)).astype(mx.bfloat16)
    v = mx.random.normal(k.shape).astype(mx.bfloat16)
    fixed, dynamic = QuantizedKVCache(bits = bits), QuantizedKVCache(bits = bits)
    parts = []
    for start in range(0, 768, 256):
        keys, values = fixed.update_and_fetch(k[..., start:start + 256, :], v[..., start:start + 256, :])
        parts.append(scaled_dot_product_attention(q[..., start:start + 256, :], keys, values, fixed, 0.125, "causal"))
    keys, values = dynamic.update_and_fetch(k, v)
    actual = prefill._prefill_attention(scaled_dot_product_attention, q, container(keys), container(values), dynamic, 0.125, "causal")
    expected = mx.concatenate(parts, axis = -2)
    for row in (0, 1):
        assert np.array_equal(np.array(actual[row].view(mx.uint8)), np.array(expected[row].view(mx.uint8)))
    for old, new in zip(fixed.state, dynamic.state):
        for a, b in zip(old, new):
            assert np.array_equal(np.array(a.view(mx.uint8)), np.array(b.view(mx.uint8)))


@pytest.mark.parametrize("kind", ["plain", "quantized", "rotating"])
@pytest.mark.parametrize("prefix", [0, 768])
def test_batched_attention_preserves_padding_and_every_cache_row(kind, prefix):
    from mlx.utils import tree_flatten
    from mlx_vlm.models import cache as caches
    from mlx_vlm.models.base import scaled_dot_product_attention
    mx.random.seed(71)
    q = mx.random.normal((2, 4, 768, 64)).astype(mx.bfloat16)
    k = mx.random.normal((2, 2, prefix + 768, 64)).astype(mx.bfloat16)
    v = mx.random.normal(k.shape).astype(mx.bfloat16)
    constructors = {
        "plain": lambda: caches.BatchKVCache([0, 333]),
        "quantized": lambda: caches.BatchQuantizedKVCache([0, 333], bits = 4),
        "rotating": lambda: caches.BatchRotatingKVCache(512, [0, 333]),
    }
    fixed, dynamic = constructors[kind](), constructors[kind]()
    if prefix:
        for cache in (fixed, dynamic):
            cache.update_and_fetch(k[..., :prefix, :], v[..., :prefix, :])
    parts = []
    for start in range(0, 768, 256):
        mask = fixed.make_mask(256)
        keys, values = fixed.update_and_fetch(k[..., prefix + start:prefix + start + 256, :], v[..., prefix + start:prefix + start + 256, :])
        parts.append(scaled_dot_product_attention(q[..., start:start + 256, :], keys, values, fixed, 0.125, mask))
    mask = dynamic.make_mask(768)
    keys, values = dynamic.update_and_fetch(k[..., prefix:, :], v[..., prefix:, :])
    with prefill.DynamicPrefillSchedule.arithmetic(SimpleNamespace(language_model = nn.Sequential())):
        actual = prefill._prefill_attention(scaled_dot_product_attention, q, keys, values, dynamic, 0.125, mask)
        if kind != "quantized":
            shared = prefill._prefill_attention(scaled_dot_product_attention, q, keys, values, None, 0.125, mask)
            assert np.array_equal(np.array(shared.view(mx.uint8)), np.array(actual.view(mx.uint8)))
    expected = mx.concatenate(parts, axis = -2)
    for row in (0, 1):
        assert np.array_equal(np.array(actual[row].view(mx.uint8)), np.array(expected[row].view(mx.uint8)))
    for (_, left), (_, right) in zip(tree_flatten(fixed.state), tree_flatten(dynamic.state)):
        assert np.array_equal(np.array(left.view(mx.uint8)), np.array(right.view(mx.uint8)))
    assert fixed._idx == dynamic._idx
    assert fixed.meta_state == dynamic.meta_state


@pytest.mark.parametrize("cache_kind", ["plain", "quantized", "rotating"])
def test_native_batch_generator_uses_dynamic_chunks_and_restores_between_steps(monkeypatch, cache_kind):
    from mlx_vlm.generate.ar import BatchGenerator
    from mlx_vlm.models.llama.config import ModelConfig
    from mlx_vlm.models.llama.language import LanguageModel

    class NoStop:
        def add_eos_token_ids(self, tokens):
            pass

        def __call__(self, token):
            return False

    mx.random.seed(23)
    language = LanguageModel(ModelConfig(
        model_type = "llama", hidden_size = 128, num_hidden_layers = 2,
        intermediate_size = 256, num_attention_heads = 2, num_key_value_heads = 1,
        rms_norm_eps = 1e-5, vocab_size = 128, tie_word_embeddings = False,
        layer_types = ["full_attention", "sliding_attention" if cache_kind == "rotating" else "full_attention"],
        sliding_window = 512,
    ))
    nn.quantize(language, bits = 8, group_size = 64)
    language.eval()
    model = nn.Module()
    model.language_model = language
    model.eval()
    schedule = prefill.create_dynamic_prefill_schedule(model)
    assert schedule is not None
    prompts = [[(i * 7 + 3) % 128 for i in range(1808)], [(i * 5 + 17) % 128 for i in range(1395)]]
    prompt_kwargs = [{"inputs_embeds": language.model.embed_tokens(mx.array([ids]))} for ids in prompts]
    calls = []
    original = LanguageModel.__call__

    def forward(self, inputs = None, **kwargs):
        calls.append((inputs.shape, type(self.layers[0].self_attn.q_proj)))
        return original(self, inputs, **kwargs)

    monkeypatch.setattr(LanguageModel, "__call__", forward)
    outputs, traces = [], []
    for dynamic in (False, True):
        calls.clear()
        gen = BatchGenerator(
            language, SimpleNamespace(stopping_criteria = NoStop()),
            prefill_batch_size = 2, completion_batch_size = 2, prefill_step_size = 256,
            kv_bits = 4 if cache_kind == "quantized" else None,
            _unsloth_prefill_schedule = schedule if dynamic else None,
        )
        result = {uid: [] for uid in gen.insert(prompts, max_tokens = [4, 6], prompt_kwargs = prompt_kwargs)}
        try:
            while gen.has_work:
                _, responses = gen.next()
                assert prefill._PREFILL_BATCH_SCHEDULE.get() is None
                assert type(language.layers[0].self_attn.q_proj) is nn.QuantizedLinear
                for response in responses:
                    result[response.uid].append((response.token, response.token_logprob))
        finally:
            gen.close()
        outputs.append(result)
        traces.append(list(calls))
    assert outputs[0] == outputs[1]
    assert len(outputs[0][0]) == 4 and len(outputs[0][1]) == 6
    assert outputs[0][0][0] != outputs[0][1][0]
    assert [shape[1] for shape, _ in traces[0] if shape[1] > 1] == [256] * 7 + [15]
    assert [shape[1] for shape, _ in traces[1] if shape[1] > 1] == [256, 256, 256, 1024, 15]
    assert all(shape[0] == 2 for shape, _ in traces[1] if shape[1] > 1)
    assert all(cls is not nn.QuantizedLinear for shape, cls in traces[1] if shape[1] > 1)
    assert all(cls is nn.QuantizedLinear for shape, cls in traces[1] if shape[1] == 1)

    def fail(self, *args, **kwargs):
        raise RuntimeError("prefill interrupted")

    monkeypatch.setattr(LanguageModel, "__call__", fail)
    gen = BatchGenerator(
        language, SimpleNamespace(stopping_criteria = NoStop()),
        prefill_batch_size = 2, completion_batch_size = 2,
        _unsloth_prefill_schedule = schedule,
    )
    gen.insert(prompts, max_tokens = 4, prompt_kwargs = prompt_kwargs)
    try:
        with pytest.raises(RuntimeError, match = "prefill interrupted"):
            gen.next()
        assert prefill._PREFILL_BATCH_SCHEDULE.get() is None
        assert type(language.layers[0].self_attn.q_proj) is nn.QuantizedLinear
        assert gen._prompt_batch.prefill_step_size == 256
        monkeypatch.setattr(LanguageModel, "__call__", forward)
        gen._prompt_batch._apc_manager = object()
        calls.clear()
        gen.next()
        assert calls[0] == ((2, 256), nn.QuantizedLinear)
    finally:
        gen.close()
