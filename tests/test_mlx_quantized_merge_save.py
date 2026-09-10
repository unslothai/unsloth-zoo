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

"""Quantized-state detection for the merged_4bit save path.

`merged_4bit` used to be a silent no-op on an unquantized model, and the whole
decision hinges on "is anything here quantized", so that predicate is enumerated
across every MLX quantized type and grid, not just the loader's affine 4-bit.
"""

import importlib
import sys

import pytest


def _real_mlx_runtime():
    try:
        importlib.import_module("mlx_lm.tuner.lora")
    except Exception:
        return False
    origin = getattr(sys.modules.get("mlx.core"), "__file__", "") or ""
    return "mlx_simulation" not in origin


if not _real_mlx_runtime():
    pytest.skip("needs the real mlx runtime", allow_module_level=True)

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map as mlx_tree_map

from unsloth_zoo.mlx.utils import (
    _get_model_config,
    _model_has_quantized_module,
    _quantize_merged_model_for_save,
)

DIMS = 256


class _Stack(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.layers = list(modules)


def _linear():
    return nn.Linear(DIMS, DIMS, bias=False)


def test_unquantized_model_is_detected_as_unquantized():
    assert not _model_has_quantized_module(_Stack(_linear(), _linear()))


@pytest.mark.parametrize(
    "mode,group_size,bits",
    [
        ("affine", 64, 4),
        ("affine", 64, 8),
        ("affine", 32, 4),
        ("mxfp4", 32, 4),
    ],
)
def test_every_quantized_grid_is_detected(mode, group_size, bits):
    """Only "is it quantized" may matter, not the grid: testing affine 4-bit
    alone would leave load_in_8bit / mxfp4 models re-quantized on save.
    """
    quantized = nn.QuantizedLinear.from_linear(
        _linear(), group_size=group_size, bits=bits, mode=mode)
    assert _model_has_quantized_module(_Stack(quantized, _linear()))


def test_quantized_embedding_is_detected():
    embedding = nn.QuantizedEmbedding.from_embedding(
        nn.Embedding(512, DIMS), group_size=64, bits=4)
    assert _model_has_quantized_module(_Stack(embedding))


def test_partially_quantized_model_counts_as_quantized():
    """A normally-loaded 4-bit model is partial: the predicate skips
    embed_tokens / lm_head, so "some unquantized modules" must not mean
    "needs quantizing".
    """
    quantized = nn.QuantizedLinear.from_linear(
        _linear(), group_size=64, bits=4, mode="affine")
    model = _Stack(quantized, _linear(), nn.Embedding(512, DIMS))
    assert _model_has_quantized_module(model)


def test_switch_linear_experts_are_detected():
    switch_layers = pytest.importorskip("mlx_lm.models.switch_layers")
    experts = switch_layers.SwitchLinear(DIMS, DIMS, num_experts=2, bias=False)
    experts = experts.to_quantized(group_size=64, bits=4, mode="affine")
    assert _model_has_quantized_module(_Stack(experts))


def _tiny_llama(dtype, hidden_size=128, intermediate_size=256, vocab_size=512):
    """A real mlx-lm llama, small enough to build in-process."""
    llama = pytest.importorskip("mlx_lm.models.llama")
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=hidden_size,
        num_hidden_layers=2,
        intermediate_size=intermediate_size,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=vocab_size,
    )
    model = llama.Model(args)
    model.update(mlx_tree_map(lambda v: v.astype(dtype), model.parameters()))
    return model


def _tiny_llama_config(hidden_size=128, intermediate_size=256, vocab_size=512,
                       **overrides):
    config = {
        "model_type": "llama",
        "hidden_size": hidden_size,
        "num_hidden_layers": 2,
        "intermediate_size": intermediate_size,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "vocab_size": vocab_size,
        "torch_dtype": "float16",
    }
    config.update(overrides)
    return config


def test_quantize_for_save_emits_a_loadable_top_level_grid():
    """mlx-lm's loader indexes ``quantization["group_size"]`` unconditionally,
    and ``quantize_model`` emits per-layer entries instead of that grid when the
    config already has a ``quantization`` key. Stale metadata must not flip that
    switch: the artifact would raise KeyError on reload.
    """
    model = _tiny_llama(mx.float16)
    model._config = _tiny_llama_config(quantization={}, quantization_config={})

    _quantize_merged_model_for_save(model)
    quantization = _get_model_config(model)["quantization"]

    assert quantization["group_size"] == 64, quantization
    assert quantization["bits"] == 4, quantization
    assert quantization["mode"] == "affine", quantization
    # Per-layer entries are what appear *instead of* the grid.
    assert not [k for k in quantization if k not in ("group_size", "bits", "mode")], (
        f"per-layer entries leaked into the saved grid: {quantization}"
    )


def test_quantize_for_save_does_not_inherit_a_stale_grid():
    """A populated stale grid is copied verbatim and mislabels the artifact."""
    model = _tiny_llama(mx.float16)
    model._config = _tiny_llama_config(
        quantization={"group_size": 32, "bits": 8, "mode": "affine"},
    )

    _quantize_merged_model_for_save(model)
    quantization = _get_model_config(model)["quantization"]

    assert quantization["group_size"] == 64, quantization
    assert quantization["bits"] == 4, quantization
    assert quantization["mode"] == "affine", quantization


def test_quantize_for_save_casts_to_the_config_dtype():
    """full_finetuning trains in float32, and mlx-lm casts before quantizing, so
    a float32-trained and a float16-trained model must produce the same artifact.
    """
    model = _tiny_llama(mx.float32)
    model._config = _tiny_llama_config(torch_dtype="float16")

    _quantize_merged_model_for_save(model)

    scales = [
        (path, module.scales.dtype)
        for path, module in model.named_modules()
        if isinstance(module, nn.QuantizedLinear)
    ]
    assert scales, "nothing was quantized"
    assert all(dtype == mx.float16 for _, dtype in scales), scales
    # The modules the predicate deliberately skips must be cast too, or they
    # dominate the artifact.
    assert model.model.embed_tokens.weight.dtype == mx.float16
    assert model.model.norm.weight.dtype == mx.float16


def test_quantize_for_save_leaves_dtype_alone_when_config_says_nothing():
    """No usable dtype is mlx-lm's "do not cast" signal; the fix must not invent
    one mlx-lm would not have picked.
    """
    model = _tiny_llama(mx.float32)
    model._config = _tiny_llama_config()
    model._config.pop("torch_dtype")

    _quantize_merged_model_for_save(model)

    assert model.model.norm.weight.dtype == mx.float32


def test_quantize_for_save_does_not_claim_a_grid_it_could_not_apply():
    """``quantize_model`` writes the grid from its arguments, not from what it
    converted, so dimensions incompatible with the group size come back
    untouched but labelled 4-bit.
    """
    dims = dict(hidden_size=48, intermediate_size=96, vocab_size=100)
    model = _tiny_llama(mx.float16, **dims)
    model._config = _tiny_llama_config(**dims)

    _quantize_merged_model_for_save(model)

    assert not _model_has_quantized_module(model), "fixture is quantizable"
    assert "quantization" not in _get_model_config(model)
    assert "quantization_config" not in _get_model_config(model)


class _StubTokenizer:
    def save_pretrained(self, path):
        pass


def test_merged_4bit_save_hands_the_live_model_back_unchanged(tmp_path):
    """Without a restore the session keeps a 4-bit model: a later merged_16bit
    export writes weights dequantized from 4-bit, and training continues on
    quantized layers.
    """
    from unsloth_zoo.mlx.utils import save_merged_model

    model = _tiny_llama(mx.float32)
    model._config = _tiny_llama_config()
    tokens = mx.array([[1, 2, 3]])
    before = model(tokens)
    mx.eval(before)

    save_merged_model(model, _StubTokenizer(), tmp_path / "merged",
                      quantize_unquantized=True)

    assert not _model_has_quantized_module(model)
    assert model.model.norm.weight.dtype == mx.float32
    assert "quantization" not in _get_model_config(model)
    after = model(tokens)
    mx.eval(after)
    assert mx.array_equal(before, after)


def test_merged_4bit_save_still_writes_a_quantized_checkpoint(tmp_path):
    """Restoring the live model must not walk back the artifact."""
    import json

    from unsloth_zoo.mlx.utils import save_merged_model

    model = _tiny_llama(mx.float32)
    model._config = _tiny_llama_config()

    save_merged_model(model, _StubTokenizer(), tmp_path / "merged",
                      quantize_unquantized=True)

    config = json.loads((tmp_path / "merged" / "config.json").read_text())
    assert config["quantization"]["bits"] == 4
    assert config["quantization"]["group_size"] == 64


def test_merged_4bit_save_removes_a_config_it_had_to_invent(tmp_path):
    """The quantize step assigns ``_config`` unconditionally, so restoring only
    when there was something to restore leaves a 4-bit grid over the
    full-precision weights just handed back, which the next save writes out.
    """
    from unsloth_zoo.mlx.utils import save_merged_model

    model = _tiny_llama(mx.float16)
    assert not hasattr(model, "_config")

    save_merged_model(model, _StubTokenizer(), tmp_path / "merged",
                      quantize_unquantized=True)

    assert not hasattr(model, "_config")


def test_a_failed_merged_4bit_save_still_returns_the_model(tmp_path, monkeypatch):
    """The restore must cover the quantize, not just the write: an OOM there
    fails after the modules are replaced, so a success-path-only restore hands
    back a 4-bit model from a save that did not happen.
    """
    import mlx_lm.utils

    from unsloth_zoo.mlx.utils import save_merged_model

    model = _tiny_llama(mx.float32)
    model._config = _tiny_llama_config()

    def _explode(*args, **kwargs):
        import mlx.nn as _nn
        _nn.quantize(args[0], group_size=64, bits=4)
        raise RuntimeError("simulated failure after the modules were replaced")

    monkeypatch.setattr(mlx_lm.utils, "quantize_model", _explode)

    with pytest.raises(RuntimeError):
        save_merged_model(model, _StubTokenizer(), tmp_path / "merged",
                          quantize_unquantized=True)

    assert not _model_has_quantized_module(model)
    assert model.model.norm.weight.dtype == mx.float32


def test_unquantized_vlm_merge_does_not_claim_to_be_quantized(tmp_path):
    """The VLM bail-out saves full precision, so it must not label it 4-bit."""
    import json

    pytest.importorskip("mlx_vlm")

    from unsloth_zoo.mlx.utils import save_merged_model

    model = _tiny_llama(mx.float16)
    model._config = _tiny_llama_config(
        vision_config={"hidden_size": 32},
        quantization={"group_size": 64, "bits": 4, "mode": "affine"},
        quantization_config={"group_size": 64, "bits": 4, "mode": "affine"},
    )

    save_merged_model(model, _StubTokenizer(), tmp_path / "merged",
                      quantize_unquantized=True)

    config = json.loads((tmp_path / "merged" / "config.json").read_text())
    assert "quantization" not in config, config
    assert "quantization_config" not in config, config


def test_vlm_merge_strips_a_grid_that_lives_on_model_config(tmp_path):
    """The strip has to cover the config that is actually saved.

    ``_get_model_config`` falls back to ``model.config`` / ``model.args``, so
    sanitizing only ``_config`` leaves a raw mlx-vlm model writing its stale
    grid over full-precision weights.
    """
    import json

    pytest.importorskip("mlx_vlm")

    from unsloth_zoo.mlx.utils import save_merged_model

    model = _tiny_llama(mx.float16)
    assert not hasattr(model, "_config")
    model.config = _tiny_llama_config(
        vision_config={"hidden_size": 32},
        quantization={"group_size": 64, "bits": 4, "mode": "affine"},
        quantization_config={"group_size": 64, "bits": 4, "mode": "affine"},
    )

    save_merged_model(model, _StubTokenizer(), tmp_path / "merged",
                      quantize_unquantized=True)

    config = json.loads((tmp_path / "merged" / "config.json").read_text())
    assert "quantization" not in config, config
    assert "quantization_config" not in config, config
