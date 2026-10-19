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

"""PLE offload: tables read by row from the checkpoint are the stock modules, bit for bit."""

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
from unsloth_zoo.mlx.offload import DiskEmbedding, offload_fraction, offload_ple


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


def test_offload_controls_and_refusals(tmp_path):
    for off in (None, False, 0, "off"):
        assert offload_fraction("ple_offload", off) is None
    assert offload_fraction("ple_offload", 0.25) == 0.25
    assert offload_fraction("ple_offload", 1) == 1.0
    for bad in (1.5, -0.1, True, "auto", "0.5"):
        with pytest.raises(ValueError, match = "ple_offload"):
            offload_fraction("ple_offload", bad)

    with pytest.raises(ValueError, match = "full_finetuning"):
        FastMLXModel.from_pretrained("unused", ple_offload = 0.5, full_finetuning = True)

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
