# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Real-MLX regression: load a bitsandbytes nested/double-quantized (nf4 +
bnb_4bit_use_double_quant=True) checkpoint end to end on Apple Silicon.

The bnb dequant path reconstructs the nested absmax via bitsandbytes' own
`.dequantize()` (bit-exact to a from-scratch reconstruction, verified out of
band) and re-quantizes to MLX affine. A local checkpoint path is used on
purpose: `unsloth/*-bnb-4bit` Hub IDs are remapped to their full-precision base
repo, so only a local (or third-party) path exercises the real dequant route.

Metal-gated so Linux CI collection skips cleanly; also skips when the real
bitsandbytes wheel or the checkpoint download is unavailable.
"""

import pytest

try:
    import mlx.core as mx
    _METAL = mx.metal.is_available()
except Exception:
    _METAL = False

metal_only = pytest.mark.skipif(not _METAL, reason="requires Apple Silicon Metal")

# nf4 + bnb_4bit_use_double_quant=True (verified in the repo's config.json).
DOUBLE_QUANT_REPO = "unsloth/Qwen2.5-0.5B-unsloth-bnb-4bit"


def _real_bitsandbytes_available():
    """True only if a real (non-stub) bitsandbytes wheel is installed on disk.

    Resolved by spec, deliberately without importing bitsandbytes. Importing it
    here would make this probe -- rather than the loader -- the first real bnb
    import in the process; _dequantize_bnb_to_tempdir then purges the package
    from sys.modules and imports it again, and bnb registers torch custom ops at
    import time, so that second import raises "Tried to register an operator"
    (see the note above _REAL_BITSANDBYTES_MODULES in mlx/loader.py). Leaving the
    import to the loader keeps the tests exercising the real dequant path.

    PathFinder is used instead of importlib.util.find_spec so the probe looks
    past unsloth_zoo's bitsandbytes stub, which is injected into sys.meta_path on
    MLX hosts and answers for the real wheel. The stub is permissive (any
    getattr returns a no-op), so an attribute probe cannot tell the two apart and
    would report the real wheel as present when only the stub is installed.
    """
    import os
    import sys
    from importlib.machinery import PathFinder

    spec = PathFinder.find_spec("bitsandbytes", sys.path)
    if spec is None or not spec.origin:
        return False
    if os.path.basename(spec.origin) == "bitsandbytes_stub.py":
        return False
    # A real wheel ships bitsandbytes/functional.py, which backs .dequantize().
    return os.path.isfile(os.path.join(os.path.dirname(spec.origin), "functional.py"))


def _local_checkpoint_path():
    """Resolve DOUBLE_QUANT_REPO to a local snapshot dir (download if needed).

    Passing a filesystem path bypasses the unsloth bnb->base Hub remap, so the
    load actually goes through _dequantize_bnb_to_tempdir.
    """
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(DOUBLE_QUANT_REPO)
    except Exception:
        try:
            return snapshot_download(DOUBLE_QUANT_REPO, local_files_only=True)
        except Exception:
            return None


@pytest.fixture(scope="module")
def dequant_path():
    if not _real_bitsandbytes_available():
        pytest.skip("real bitsandbytes wheel unavailable")
    path = _local_checkpoint_path()
    if path is None:
        pytest.skip(f"{DOUBLE_QUANT_REPO} not available (no network / cache)")
    # Sanity: confirm this really is a nested double-quant checkpoint on disk.
    import json, os
    with open(os.path.join(path, "config.json")) as f:
        qc = json.load(f).get("quantization_config", {})
    assert qc.get("quant_method") == "bitsandbytes"
    assert qc.get("bnb_4bit_use_double_quant") is True, "fixture must be double-quant"
    assert qc.get("bnb_4bit_quant_type") == "nf4"
    return path


@metal_only
def test_double_quant_checkpoint_loads_and_generates(dequant_path):
    """A nested double-quant checkpoint loads via the bnb dequant path and
    produces finite, non-empty generation through MLX."""
    from unsloth_zoo.mlx.loader import FastMLXModel
    from mlx_lm import generate

    model, tokenizer = FastMLXModel.from_pretrained(
        dequant_path, max_seq_length=512, load_in_4bit=True,
    )
    assert model is not None and tokenizer is not None

    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is the capital of France? One short sentence."}],
            add_generation_prompt=True, tokenize=False,
        )
    except Exception:
        prompt = "The capital of France is"
    out = generate(model, tokenizer, prompt=prompt, max_tokens=32, verbose=False)
    assert isinstance(out, str) and len(out.strip()) > 0, "empty generation"


@metal_only
def test_double_quant_checkpoint_lora_steps_finite(dequant_path, tmp_path):
    """A few LoRA SFT steps on the dequantized+re-quantized model produce a
    finite loss (guards the training path, not just inference)."""
    import math
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    model, tokenizer = FastMLXModel.from_pretrained(
        dequant_path, max_seq_length=256, load_in_4bit=True,
    )
    model = FastMLXModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0)

    dataset = [
        {"text": f"### Question: what is {i} plus {i}?\n### Answer: {2 * i}."}
        for i in range(16)
    ]
    config = MLXTrainingConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=6,
        warmup_steps=1,
        learning_rate=5e-4,
        logging_steps=1,
        output_dir=str(tmp_path),
        seed=3407,
        report_to="none",
    )
    trainer = MLXTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset, args=config,
    )
    trainer.train()
    losses = trainer._train_loss_history
    assert losses, "no training losses recorded"
    assert all(
        isinstance(x, float) and math.isfinite(x) for x in losses
    ), f"non-finite loss: {losses}"
