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

`save_method="merged_4bit"` used to be a silent no-op on an unquantized model:
`LoRALinear.fuse(dequantize=False)` only requantizes when the base was already
quantized, so a model loaded with `load_in_16bit=True` or `full_finetuning=True`
was written at full precision with no warning.

Deciding whether to quantize hinges entirely on "is anything here quantized",
so that predicate is enumerated here across every MLX quantized module type and
grid rather than only the affine-4bit one the default loader happens to produce.
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
    """The decision must not depend on the grid, only on 'is it quantized'.

    Enumerated rather than assumed: the default loader produces affine 4-bit,
    so testing only that would leave load_in_8bit / load_in_mxfp4 models silently
    re-quantized on save.
    """
    quantized = nn.QuantizedLinear.from_linear(
        _linear(), group_size=group_size, bits=bits, mode=mode)
    assert _model_has_quantized_module(_Stack(quantized, _linear()))


def test_quantized_embedding_is_detected():
    embedding = nn.QuantizedEmbedding.from_embedding(
        nn.Embedding(512, DIMS), group_size=64, bits=4)
    assert _model_has_quantized_module(_Stack(embedding))


def test_partially_quantized_model_counts_as_quantized():
    """A normally-loaded 4-bit model *is* partial.

    The loader's predicate skips `embed_tokens` / `lm_head`, so a model loaded
    with load_in_4bit=True has unquantized modules in it. Treating "some
    unquantized modules" as "needs quantizing" would re-quantize an
    already-quantized checkpoint.
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


# ---------------------------------------------------------------------------
# What the quantize step hands to mlx-lm
#
# `_quantize_merged_model_for_save` is driven directly on a real (tiny) mlx-lm
# llama so the config it produces can be checked against mlx-lm's load-time
# contract without downloading a published checkpoint.
# ---------------------------------------------------------------------------

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
    """mlx-lm's loader indexes ``quantization["group_size"]`` unconditionally.

    ``quantize_model`` writes that top-level grid only when the config it is
    handed has no ``quantization`` key; with one present it emits per-layer
    entries instead. Nothing is quantized when this branch runs, so inherited
    metadata is stale by construction and must not flip that switch — the
    artifact would raise KeyError on reload.
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
    """A populated stale grid is carried through verbatim and mislabels the
    artifact: the tensors are written at 4-bit/64 whatever the config claims."""
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
    """full_finetuning trains in float32; the checkpoint must not stay there.

    mlx-lm's own conversion casts every floating parameter to the config dtype
    before quantizing, so a float32-trained model and a float16-trained one
    produce the same artifact. Without the cast the "4-bit" checkpoint carries
    float32 scales, biases, embeddings and norms.
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
    """No usable dtype in the config is mlx-lm's "do not cast" signal.

    Guards the fix against over-reach: it must not invent a dtype mlx-lm would
    not itself have picked.
    """
    model = _tiny_llama(mx.float32)
    model._config = _tiny_llama_config()
    model._config.pop("torch_dtype")

    _quantize_merged_model_for_save(model)

    assert model.model.norm.weight.dtype == mx.float32


def test_quantize_for_save_does_not_claim_a_grid_it_could_not_apply():
    """Nothing quantizable must not produce a checkpoint that claims 4-bit.

    ``quantize_model`` writes the grid into the config from its arguments, not
    from what it managed to convert, so a model whose dimensions are not a
    multiple of the group size comes back untouched but labelled 4-bit. That is
    the same silent full-precision-on-a-4-bit-request this path exists to fix,
    one level down.
    """
    dims = dict(hidden_size=48, intermediate_size=96, vocab_size=100)
    model = _tiny_llama(mx.float16, **dims)
    model._config = _tiny_llama_config(**dims)

    _quantize_merged_model_for_save(model)

    assert not _model_has_quantized_module(model), "fixture is quantizable"
    assert "quantization" not in _get_model_config(model)
    assert "quantization_config" not in _get_model_config(model)


# ---------------------------------------------------------------------------
# The save must not keep the model it quantized
#
# Driven through ``save_merged_model`` because the mutation is only observable
# after the checkpoint is written.
# ---------------------------------------------------------------------------

class _StubTokenizer:
    def save_pretrained(self, path):
        pass


def test_merged_4bit_save_hands_the_live_model_back_unchanged(tmp_path):
    """The user's model is still fp32/fp16 after a merged_4bit save.

    The quantize step rewrites the live model, so without a restore the session
    silently keeps a 4-bit model: a later ``merged_16bit`` export writes weights
    dequantized from 4-bit, and continued training trains quantized layers.
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
    """A model that arrived without ``_config`` must not keep the quantized one.

    The quantize step assigns ``_config`` unconditionally, so restoring only
    when there was something to restore leaves the live model advertising a
    4-bit grid over the full-precision weights just handed back -- which the
    next save, or a ``lora`` save stamping the base grid into
    adapter_config.json, then writes out.
    """
    from unsloth_zoo.mlx.utils import save_merged_model

    model = _tiny_llama(mx.float16)
    assert not hasattr(model, "_config")

    save_merged_model(model, _StubTokenizer(), tmp_path / "merged",
                      quantize_unquantized=True)

    assert not hasattr(model, "_config")


def test_a_failed_merged_4bit_save_still_returns_the_model(tmp_path, monkeypatch):
    """The restore has to cover the quantize itself, not just the write.

    Quantizing a large full-finetuned model is exactly where an out-of-memory
    failure happens, and it fails *after* the modules have been replaced -- so
    a snapshot taken but only applied on the success path hands back a 4-bit
    model from a save that did not happen.
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
