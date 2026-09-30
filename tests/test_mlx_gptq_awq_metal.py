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

"""Real GPTQ / AWQ checkpoints load, generate and train on Apple Silicon."""

import glob
import os
import tempfile

import pytest

try:
    import mlx.core as mx
    _METAL = mx.metal.is_available()
except Exception:
    pytest.skip("requires mlx", allow_module_level=True)
metal_only = pytest.mark.skipif(not _METAL, reason="requires Apple Silicon Metal")

GPTQ_MODEL = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"
AWQ_MODEL = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"


def _answer(model, tokenizer):
    from mlx_lm import generate

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=8)


def _dequant_scratch_dirs():
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "unsloth_mlx_dequant_*")))


@metal_only
@pytest.mark.parametrize("repo", [GPTQ_MODEL, AWQ_MODEL])
@pytest.mark.parametrize("load_in_16bit", [False, True])
def test_prequant_checkpoint_loads_and_answers(repo, load_in_16bit):
    from unsloth_zoo.mlx.loader import FastMLXModel

    before = _dequant_scratch_dirs()
    model, tokenizer = FastMLXModel.from_pretrained(
        repo,
        max_seq_length=256,
        load_in_4bit=not load_in_16bit,
        load_in_16bit=load_in_16bit,
    )
    assert _dequant_scratch_dirs() <= before
    assert model._hf_repo == repo
    # 4-bit keeps the checkpoint's own codes (GPTQ repacked, AWQ via mlx-lm); 16-bit is dense.
    source = getattr(model, "_unsloth_quantized_source", None)
    assert source in ((None, "none") if load_in_16bit else ("mlx_config",)), source
    text = _answer(model, tokenizer)
    print(f"{repo} load_in_16bit={load_in_16bit}: {text!r}")
    assert "paris" in text.lower(), text


@metal_only
@pytest.mark.parametrize("repo", [GPTQ_MODEL, AWQ_MODEL])
def test_prequant_checkpoint_lora_trains(repo, tmp_path):
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    model, tokenizer = FastMLXModel.from_pretrained(repo, max_seq_length=256)
    model = FastMLXModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0)
    dataset = [
        {
            "text": (
                f"<|im_start|>user\nWhat is {i} plus {i}?<|im_end|>\n"
                f"<|im_start|>assistant\nThe answer is {2 * i}.<|im_end|>\n"
            )
        }
        for i in range(12)
    ]
    trainer = MLXTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=MLXTrainingConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=6,
            learning_rate=5e-4,
            logging_steps=1,
            output_dir=str(tmp_path),
            seed=3407,
            report_to="none",
        ),
    )
    trainer.train()
    hist = list(trainer._train_loss_history)
    print(f"{repo} losses: {hist}")
    assert len(hist) == 6
    assert all(loss == loss and abs(loss) < 20 for loss in hist), hist
    assert hist[-1] < hist[0], hist

    import mlx.core as mx
    import mlx.nn as nn

    def text_loss(m):
        ids = mx.array([tokenizer.encode(dataset[3]["text"])])
        logits = m(ids).astype(mx.float32)
        return nn.losses.cross_entropy(logits[0, :-1], ids[0, 1:]).mean().item()

    base, _ = FastMLXModel.from_pretrained(repo, max_seq_length=256)
    base_loss, trained_loss = text_loss(base), text_loss(model)
    model.save_pretrained(str(tmp_path / "adapter"))
    reloaded, _ = FastMLXModel.from_pretrained(str(tmp_path / "adapter"), max_seq_length=256)
    reloaded_loss = text_loss(reloaded)
    print(f"{repo} base/trained/reloaded loss: {base_loss} {trained_loss} {reloaded_loss}")
    # The reloaded adapter must carry the training: near the trained loss, far below the base's.
    assert reloaded_loss < 0.5 * base_loss, (base_loss, trained_loss, reloaded_loss)
    assert abs(reloaded_loss - trained_loss) < 0.25 * base_loss, (base_loss, trained_loss, reloaded_loss)
