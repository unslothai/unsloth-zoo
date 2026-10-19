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

"""PLE and expert offload: weights read from the checkpoint compute the stock modules' outputs, bit for bit."""

import os

import numpy as np
import pytest

pytest.importorskip("mlx.core")
from mlx_simulation import mlx_is_simulated

if mlx_is_simulated():
    pytest.skip("Requires native MLX", allow_module_level = True)

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from unsloth_zoo.mlx.loader import FastMLXModel, _convert_mlx_dtype, _mlx_save_pretrained_merged
from unsloth_zoo.mlx.offload import DiskEmbedding, OffloadSwitchGLU, offload_experts, offload_fraction, offload_ple


@pytest.fixture(autouse = True)
def _require_real_metal():
    if not (getattr(mx, "metal", None) and mx.metal.is_available() and mx.default_device() == mx.gpu):
        pytest.skip("real Metal required; shim active or no GPU")


def _with_negative_zeros(module):
    weight = module.weight.astype(mx.bfloat16)
    module.weight = mx.where(mx.arange(weight.shape[-1]) % 7 == 3, mx.array(-0.0, mx.bfloat16), weight)
    return module


def _table(kind):
    if kind.startswith("sharded"):
        ShardedEmbedding = pytest.importorskip("mlx_vlm.models.qwen4_exp.language").ShardedEmbedding
        module = ShardedEmbedding(1000, 64, 7)
        for shard in module.shards:
            _with_negative_zeros(shard)
        if kind == "sharded_quantized":
            nn.quantize(module, group_size = 32, bits = 4)
        return "ngram_embedding", module
    module = _with_negative_zeros(nn.Embedding(500, 96))
    if kind in ("quantized", "mxfp4"):
        mode = "mxfp4" if kind == "mxfp4" else "affine"
        module = nn.QuantizedEmbedding.from_embedding(module, group_size = 32, bits = 4, mode = mode)
    return "embed_tokens_per_layer", module


_MLX_PREFIX, _HF_PREFIX = "language_model.model.", "model.language_model."


def _lazy_copy(build, directory, hf_names = False):
    """``build()``'s model with its weights lazily mapped from ``directory``, saved there once,
    under the Hugging Face layout's names if ``hf_names``."""
    model = build()
    path = os.path.join(directory, "model.safetensors")
    rename = (lambda name, a, b: b + name[len(a):]) if hf_names else (lambda name, a, b: name)
    if not os.path.exists(path):
        mx.save_safetensors(path, {
            rename(name, _MLX_PREFIX, _HF_PREFIX): value for name, value in tree_flatten(model.parameters())
        })
    model.load_weights([(rename(name, _HF_PREFIX, _MLX_PREFIX), value) for name, value in mx.load(path).items()])
    return model


def _holder(name, module):
    """``module`` at ``language_model.model.<name>``, where the VLM families keep their tables."""
    inner = nn.Module()
    setattr(inner, name, module)
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.model = inner
    return model


def _site(model, name):
    return getattr(model.language_model.model, name)


def _bits(array):
    return np.array(array.view(mx.uint16) if array.dtype == mx.bfloat16 else array)


@pytest.mark.parametrize("kind", ["dense", "quantized", "mxfp4", "sharded", "sharded_quantized"])
@pytest.mark.parametrize("fraction", [1.0, 0.5])
def test_offloaded_table_matches_stock(tmp_path, kind, fraction):
    name, module = _table(kind)
    model = _lazy_copy(lambda: _holder(name, module), str(tmp_path), hf_names = kind == "quantized")
    # Repeats, unsorted, and rows on both sides of the held prefix of several shards; the second
    # lookup mixes rows the first one read with new ones.
    rng = np.random.default_rng(0)
    first = rng.integers(0, 480, (3, 41))
    lookups = [mx.array(first), mx.array(np.concatenate([first[:, :20], rng.integers(0, 480, (3, 21))], 1))]
    expected = [_site(model, name)(ids) for ids in lookups]
    table_bytes = sum(value.nbytes for _, value in tree_flatten(model.parameters()))
    listing = sorted(os.listdir(tmp_path))

    report = offload_ple(model, str(tmp_path), fraction)

    replacement = _site(model, name)
    assert isinstance(replacement, DiskEmbedding) and not tree_flatten(model.parameters())
    assert report["tables"] == 1 and report["skipped"] == []
    assert report["on_disk_bytes"] + report["resident_bytes"] == table_bytes
    assert abs(report["resident_bytes"] - table_bytes * (1 - fraction)) <= table_bytes * 0.02
    for ids, reference in zip(lookups, expected):
        actual = replacement(ids)
        assert actual.dtype == reference.dtype and actual.shape == reference.shape
        np.testing.assert_array_equal(_bits(actual), _bits(reference))
    assert sorted(os.listdir(tmp_path)) == listing  # nothing is written next to the checkpoint


def test_offloaded_rows_are_never_loaded(tmp_path):
    rows, dims = 8192, 2048  # 64 MiB of float32
    model = _lazy_copy(lambda: _holder("embed_tokens_per_layer", nn.Embedding(rows, dims)), str(tmp_path))
    site = model.language_model.model
    mx.synchronize()
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    offload_ple(model, str(tmp_path), 1.0)
    mx.eval(site.embed_tokens_per_layer(mx.array([[0, rows - 1, 4321]])))
    assert mx.get_peak_memory() - before < rows * dims * 4 // 16
    assert site.embed_tokens_per_layer(mx.zeros((1, 0), mx.int32)).shape == (1, 0, dims)
    for outside in (-1, rows):
        with pytest.raises(IndexError, match = "outside the table"):
            site.embed_tokens_per_layer(mx.array([outside]))


@pytest.mark.parametrize("cast", [True, False])
def test_offloaded_floats_get_the_load_cast(tmp_path, cast):
    def build():
        model = _holder("embed_tokens_per_layer", nn.QuantizedEmbedding.from_embedding(
            _with_negative_zeros(nn.Embedding(300, 64)), group_size = 32, bits = 4,
        ))
        model.cast_predicate = lambda _key: cast
        return model

    stock = _lazy_copy(build, str(tmp_path))
    offloaded = _lazy_copy(build, str(tmp_path))
    _convert_mlx_dtype(stock, mx.float32)
    report = offload_ple(offloaded, str(tmp_path), 0.5, cast_dtype = mx.float32)
    for ids in ([[5, 299, 150, 149, 0]], [[5, 149, 0]]):  # mixed, then held rows only
        expected = _site(stock, "embed_tokens_per_layer")(mx.array(ids))
        actual = _site(offloaded, "embed_tokens_per_layer")(mx.array(ids))
        assert expected.dtype == actual.dtype == (mx.float32 if cast else mx.bfloat16)
        np.testing.assert_array_equal(_bits(actual), _bits(expected))
    held = _site(offloaded, "embed_tokens_per_layer")._tables[0].held
    assert report["resident_bytes"] == sum(column.nbytes for column in held) > 0


def _moe(lib, dims = 512, hidden = 32, scales = mx.bfloat16, eval = True, quantized = True):
    """One MoE layer of 64 quantized experts, top-4, from mlx-vlm's or mlx-lm's ``SwitchGLU``; 512
    wide, as mlx-vlm's decode-block kernels require."""
    switch_layers = pytest.importorskip(f"mlx_{lib}.models.switch_layers")
    mx.random.seed(0)
    block = nn.Module()
    block.top_k = 4
    block.switch_mlp = switch_layers.SwitchGLU(dims, hidden, 64)
    model = nn.Module()
    model.layers = [block]
    if not quantized:
        return model
    nn.quantize(block, group_size = 32, bits = 4, class_predicate = lambda _, module: hasattr(module, "to_quantized"))
    for name in ("gate_proj", "up_proj", "down_proj"):
        linear = block.switch_mlp[name]
        linear.scales, linear.biases = linear.scales.astype(scales), linear.biases.astype(scales)
    model.train(not eval)
    return model


def _routes(shape, seed, among = 64):
    """Top-4 of 64 experts per position, drawn from the first ``among``."""
    logits = mx.random.normal((*shape, 64), key = mx.random.key(seed))
    logits = mx.where(mx.arange(64) < among, logits + 100, logits)
    return mx.argpartition(logits, kth = -4, axis = -1)[..., -4:]


def _moe_calls(dtype):
    # A decode block whose positions route to disjoint experts, so over 8 slots its last chunk holds
    # one position; then (batch, length, experts routed among): decode, batched decode, decode blocks
    # (mlx-vlm's block kernels; one position at a time, here on gather_qmm_rhs, once batch x top-k
    # reaches 64), prefill on the per-route kernel, prefill on gather_qmm_rhs (from exactly 4 routes
    # per expert), prefill among 12 experts, which a 16-slot table alone would run on gather_qmm_rhs
    # and 64 experts do not, and a batch's decode blocks under 64 routes.
    disjoint = mx.arange(12, dtype = mx.uint32).reshape(1, 3, 4)
    calls = [(mx.random.normal((1, 3, 512), key = mx.random.key(99)).astype(dtype), disjoint)]
    shapes = [(1, 1, 64), (8, 1, 64), (1, 4, 64), (64, 2, 64), (1, 20, 64), (1, 64, 64), (2, 300, 64), (1, 30, 12), (8, 2, 64)]
    calls += [
        (mx.random.normal((batch, length, 512), key = mx.random.key(i)).astype(dtype), _routes((batch, length), 100 + i, among))
        for i, (batch, length, among) in enumerate(shapes)
    ]
    # gather_qmm_rhs prefill whose rarely routed experts leave a chunk too small for that kernel.
    skewed = mx.concatenate([_routes((1, 290), 7, 12), _routes((1, 10), 8)], axis = 1)
    return calls + [(mx.random.normal((1, 300, 512), key = mx.random.key(9)).astype(dtype), skewed)]


def _verifier_steps(glu, x, indices):
    """mlx-vlm's batch-invariant verifier runs the projections itself, unsorted; then a down
    projection of another input."""
    x, indices = mx.expand_dims(x.reshape(-1, x.shape[-1]), (-2, -3)), indices.reshape(-1, 4)
    up = glu.up_proj(x, indices, sorted_indices = False)
    gate = glu.gate_proj(x, indices, sorted_indices = False)
    activated = glu.activation(up, gate)
    down = glu.down_proj(activated, indices, sorted_indices = False)
    return mx.concatenate([down, glu.down_proj(activated * 2, indices, sorted_indices = False)])


@pytest.mark.parametrize("lib", ["vlm", "lm"])
@pytest.mark.parametrize("fraction", [1.0, 0.75, 0.1])
def test_offloaded_experts_match_stock(tmp_path, lib, fraction):
    stock = _lazy_copy(lambda: _moe(lib), str(tmp_path))
    calls = _moe_calls(mx.bfloat16)
    expected = [stock.layers[0].switch_mlp(*call) for call in calls]
    expected += [_verifier_steps(stock.layers[0].switch_mlp, *call) for call in calls[5:7]]
    table_bytes = sum(value.nbytes for _, value in tree_flatten(stock.parameters()))

    model = _lazy_copy(lambda: _moe(lib), str(tmp_path))
    report = offload_experts(model, str(tmp_path), fraction)

    glu = model.layers[0].switch_mlp
    assert isinstance(glu, OffloadSwitchGLU) and not tree_flatten(model.parameters())
    assert report["layers"] == 1 and report["on_disk_bytes"] == table_bytes
    slots = max(8, int(64 * (1 - fraction)))
    assert glu._slots.capacity == slots and report["resident_bytes"] == table_bytes * slots // 64
    actual = [glu(*call) for call in calls]
    actual += [_verifier_steps(glu, *call) for call in calls[5:7]]
    for got, want in zip(actual, expected):
        assert got.shape == want.shape
        np.testing.assert_array_equal(_bits(got), _bits(want))


def test_offloaded_experts_are_never_loaded(tmp_path):
    model = _lazy_copy(lambda: _moe("lm", hidden = 512), str(tmp_path))
    table_bytes = sum(value.nbytes for _, value in tree_flatten(model.parameters()))
    mx.synchronize()
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    offload_experts(model, str(tmp_path), 1.0)
    mx.eval(model.layers[0].switch_mlp(mx.ones((1, 1, 512), mx.bfloat16), _routes((1, 1), 0)))
    assert mx.get_peak_memory() - before < table_bytes // 4


def test_offloaded_expert_floats_get_the_load_cast(tmp_path):
    # Loaded in training mode and switched to eval afterwards, as generation does.
    stock = _lazy_copy(lambda: _moe("vlm", scales = mx.float16, eval = False), str(tmp_path))
    offloaded = _lazy_copy(lambda: _moe("vlm", scales = mx.float16, eval = False), str(tmp_path))
    _convert_mlx_dtype(stock, mx.bfloat16)
    offload_experts(offloaded, str(tmp_path), 0.75, cast_dtype = mx.bfloat16)
    stock.eval()
    offloaded.eval()
    for x, indices in _moe_calls(mx.bfloat16)[::2]:
        expected = stock.layers[0].switch_mlp(x, indices)
        actual = offloaded.layers[0].switch_mlp(x, indices)
        assert actual.dtype == expected.dtype == mx.bfloat16
        np.testing.assert_array_equal(_bits(actual), _bits(expected))


def test_offload_controls_and_refusals(tmp_path):
    for off in (None, False, 0, "off"):
        assert offload_fraction("ple_offload", off) is None
    assert offload_fraction("ple_offload", 0.25) == 0.25
    assert offload_fraction("ple_offload", 1) == 1.0
    for bad in (1.5, -0.1, True, "auto", "0.5"):
        with pytest.raises(ValueError, match = "ple_offload"):
            offload_fraction("ple_offload", bad)

    for knob in ("ple_offload", "expert_offload"):
        with pytest.raises(ValueError, match = "full_finetuning"):
            FastMLXModel.from_pretrained("unused", full_finetuning = True, **{knob: 0.5})

    dense = tmp_path / "dense"
    dense.mkdir()
    model = _lazy_copy(lambda: _moe("vlm", quantized = False), str(dense))
    assert offload_experts(model, str(dense), 1.0) == {"layers": 0, "on_disk_bytes": 0, "resident_bytes": 0, "skipped": []}
    top_12 = tmp_path / "top_12"
    top_12.mkdir()
    model = _lazy_copy(lambda: _moe("lm"), str(top_12))
    model.layers[0].top_k = 12  # the slots hold at least one token's routes
    offload_experts(model, str(top_12), 1.0)
    assert model.layers[0].switch_mlp._slots.capacity == 12

    model = _lazy_copy(lambda: _holder("embed_tokens_per_layer", nn.Embedding(16, 8)), str(tmp_path))
    model.update(tree_map(lambda value: value.astype(mx.bfloat16), model.parameters()))
    report = offload_ple(model, str(tmp_path), 1.0)  # the checkpoint holds float32, the model bf16
    assert report["tables"] == 0 and len(report["skipped"]) == 1
    assert isinstance(_site(model, "embed_tokens_per_layer"), nn.Embedding)

    model = _lazy_copy(lambda: _holder("embed_tokens_per_layer", nn.Embedding(16, 8)), str(tmp_path))
    model._unsloth_offload = {"ple": offload_ple(model, str(tmp_path), 1.0)}
    with pytest.raises(ValueError, match = "inference only"):
        FastMLXModel.get_peft_model(model, r = 8)
    with pytest.raises(ValueError, match = "inference only"):
        _mlx_save_pretrained_merged(model, str(tmp_path / "merged"), tokenizer = object())
    assert not (tmp_path / "merged").exists()
