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

"""merged_4bit really produces 4-bit, driven through save_pretrained_merged.

Every branch is reached through the public save entry point rather than by
calling the helper directly: a save-path branch that is unreachable from
`save_pretrained_merged` would be dead code that still passed a helper-level
test.

The paths that must NOT change are covered too -- an already-quantized base,
`merged_16bit`, and `push_to_hub_merged` (which shares `save_merged_model` but
has no save_method and must keep writing the model's existing precision).
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx
import mlx.nn as nn

from unsloth_zoo.mlx.loader import FastMLXModel

QUANTIZED_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
FP_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(autouse=True)
def _require_real_metal():
    import mlx.core as _mx
    if not (getattr(_mx, "metal", None) and _mx.metal.is_available()
            and _mx.default_device() == _mx.gpu):
        pytest.skip("real Metal required; shim active or no GPU")


def _saved_artifact(model, tokenizer, method):
    """Save through the public entry point and report what landed on disk."""
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained_merged(directory, tokenizer, save_method=method)
        config = json.load(open(os.path.join(directory, "config.json")))
        weight_bytes = sum(
            os.path.getsize(os.path.join(directory, name))
            for name in os.listdir(directory)
            if name.endswith(".safetensors")
        )
    return config.get("quantization") or config.get("quantization_config"), weight_bytes


def _load_and_adapt(model_name, **load_kwargs):
    model, tokenizer = FastMLXModel.from_pretrained(
        model_name, max_seq_length=64, **load_kwargs)
    if not load_kwargs.get("full_finetuning"):
        model = FastMLXModel.get_peft_model(
            model, r=8, lora_alpha=16, lora_dropout=0,
            use_gradient_checkpointing=False,
        )
    return model, tokenizer


# --------------------------------------------------------------------------
# The bug: merged_4bit on an unquantized model
# --------------------------------------------------------------------------

def test_merged_4bit_quantizes_a_16bit_lora_model():
    model, tokenizer = _load_and_adapt(
        FP_MODEL, load_in_16bit=True, load_in_4bit=False)
    quantization, weight_bytes = _saved_artifact(model, tokenizer, "merged_4bit")

    assert quantization == {"group_size": 64, "bits": 4, "mode": "affine"}, (
        f"merged_4bit wrote no quantization metadata: {quantization}"
    )
    # ~988MB unquantized -> ~474MB at 4-bit; assert well under the fp16 size.
    assert weight_bytes < 700e6, f"weights still {weight_bytes / 1e6:.1f}MB"


def test_merged_4bit_quantizes_a_full_finetuned_model():
    model, tokenizer = _load_and_adapt(FP_MODEL, full_finetuning=True)
    quantization, weight_bytes = _saved_artifact(model, tokenizer, "merged_4bit")

    assert quantization == {"group_size": 64, "bits": 4, "mode": "affine"}
    # full_finetuning trains in float32, so the skipped modules (embed_tokens /
    # lm_head) stay fp32 and the artifact is larger than the fp16-LoRA case --
    # still far below the ~1976MB it used to write.
    assert weight_bytes < 1200e6, f"weights still {weight_bytes / 1e6:.1f}MB"


def test_quantized_save_reloads_and_runs():
    """The artifact has to be loadable, not merely smaller."""
    model, tokenizer = _load_and_adapt(
        FP_MODEL, load_in_16bit=True, load_in_4bit=False)
    ids = mx.array([[3, 11, 19, 27, 35, 43, 51, 59]])

    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained_merged(directory, tokenizer, save_method="merged_4bit")
        reloaded, _ = FastMLXModel.from_pretrained(directory, max_seq_length=64)
        logits = reloaded(ids)
        logits = getattr(logits, "logits", logits)
        mx.eval(logits)

    assert bool(mx.all(mx.isfinite(logits)).item()), "reloaded model produced non-finite logits"


# --------------------------------------------------------------------------
# Paths that must not change
# --------------------------------------------------------------------------

def test_already_quantized_merged_4bit_is_unchanged():
    """Regression guard: an already-4bit base must not be re-quantized.

    The source repo's metadata is passed through verbatim rather than
    rewritten, and published mlx-community configs may omit ``mode`` entirely
    (this one does), so assert the grid rather than an exact dict.
    """
    model, tokenizer = _load_and_adapt(QUANTIZED_MODEL)
    quantization, weight_bytes = _saved_artifact(model, tokenizer, "merged_4bit")
    assert quantization is not None
    assert quantization["bits"] == 4
    assert quantization["group_size"] == 64
    assert quantization.get("mode", "affine") == "affine"
    assert weight_bytes < 700e6


def test_merged_16bit_still_writes_full_precision():
    """merged_16bit must never acquire quantization metadata."""
    model, tokenizer = _load_and_adapt(QUANTIZED_MODEL)
    quantization, weight_bytes = _saved_artifact(model, tokenizer, "merged_16bit")
    assert quantization is None, f"merged_16bit gained quantization: {quantization}"
    assert weight_bytes > 700e6


def test_push_to_hub_merged_does_not_quantize():
    """`push_to_hub_merged` shares save_merged_model but has no save_method.

    It calls it with the default dequantize=False, so inferring "quantize" from
    that flag would silently 4-bit every hub push of an unquantized model. The
    quantize step is opt-in from the merged_4bit branch only.
    """
    from unsloth_zoo.mlx.utils import save_merged_model

    model, tokenizer = _load_and_adapt(
        FP_MODEL, load_in_16bit=True, load_in_4bit=False)
    with tempfile.TemporaryDirectory() as directory:
        # Exactly the call push_to_hub_merged makes.
        save_merged_model(model, tokenizer, directory)
        config = json.load(open(os.path.join(directory, "config.json")))
    quantization = config.get("quantization") or config.get("quantization_config")
    assert quantization is None, (
        f"push_to_hub_merged's save path quantized unexpectedly: {quantization}"
    )
