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

"""Real-MLX regression for MLXKTOTrainer on Apple Silicon.

Trains a 4-bit Qwen2.5-0.5B + LoRA with KTO on a small unpaired dataset and
checks the loss is finite and decreasing, that the LoRA-disable reference path
restores adapter scales (including when a forward throws mid-step), and that a
non-PEFT model raises the clear LoRA-only error.

Metal-gated so Linux CI collection skips cleanly.
"""

import math

import pytest

try:
    import mlx.core as mx
    _METAL = mx.metal.is_available()
except Exception:
    _METAL = False

if not _METAL:
    print("NOTICE: Metal unavailable; MLX KTO training tests will be skipped.")

metal_only = pytest.mark.skipif(not _METAL, reason="requires Apple Silicon Metal")

MODEL = "unsloth/Qwen2.5-0.5B"


def _dataset(n=24):
    rows = []
    for i in range(n):
        prompt = f"### Question: what is {i} plus {i}?\n### Answer:"
        if i % 2 == 0:
            rows.append({"prompt": prompt, "completion": f" {2 * i}.", "label": True})
        else:
            rows.append({"prompt": prompt, "completion": f" {2 * i + 7}, wrong.", "label": False})
    return rows


def _load_peft():
    from unsloth_zoo.mlx.loader import FastMLXModel
    mx.random.seed(3407)
    model, tok = FastMLXModel.from_pretrained(MODEL, max_seq_length=256, load_in_4bit=True)
    model = FastMLXModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0, random_state=3407)
    return model, tok


def _config(**overrides):
    from unsloth_zoo.mlx.trainer import MLXKTOConfig
    base = dict(per_device_train_batch_size=4, max_steps=6, warmup_steps=1,
                learning_rate=1e-4, beta=0.1, logging_steps=99, seed=3407, report_to="none")
    base.update(overrides)
    return MLXKTOConfig(**base)


@metal_only
def test_kto_trains_finite_and_decreasing(tmp_path):
    from unsloth_zoo.mlx.trainer import MLXKTOTrainer
    model, tok = _load_peft()
    trainer = MLXKTOTrainer(model=model, tokenizer=tok, train_dataset=_dataset(),
                            args=_config(output_dir=str(tmp_path)))
    output = trainer.train()

    # train() returns MLXTrainOutput, matching MLXTrainer; the per-step losses
    # live on the trainer.
    hist = trainer._train_loss_history
    assert len(hist) == 6, f"expected 6 steps, got {len(hist)}"
    assert all(math.isfinite(x) for x in hist), f"non-finite loss: {hist}"
    assert hist[-1] < hist[0], f"loss did not decrease: {hist}"
    assert output.global_step == 6, f"expected 6 steps, got {output.global_step}"
    assert math.isfinite(output.training_loss)
    assert output["total_train_steps"] == 6
    # KL baseline recorded per step, finite and clamped >= 0.
    assert trainer._kl_history and all(math.isfinite(k) and k >= 0.0 for k in trainer._kl_history)


@metal_only
def test_kto_saves_adapters_at_end(tmp_path):
    # save_steps defaults to 0 (save at end); train() must leave adapters on
    # disk, not only in memory, matching MLXTrainer.
    from unsloth_zoo.mlx.trainer import MLXKTOTrainer
    model, tok = _load_peft()
    trainer = MLXKTOTrainer(model=model, tokenizer=tok, train_dataset=_dataset(),
                            args=_config(output_dir=str(tmp_path)))
    trainer.train()
    written = {p.name for p in tmp_path.iterdir()}
    assert "adapters.safetensors" in written, f"no adapters saved: {sorted(written)}"
    assert "adapter_config.json" in written, f"no adapter config saved: {sorted(written)}"


@metal_only
def test_lora_scales_restored_after_training(tmp_path):
    from unsloth_zoo.mlx.trainer import MLXKTOTrainer
    from unsloth_zoo.mlx.utils import iter_mlx_lora_modules
    model, tok = _load_peft()
    before = [m.scale for _, m in iter_mlx_lora_modules(model)]
    assert before and all(s != 0 for s in before)

    trainer = MLXKTOTrainer(model=model, tokenizer=tok, train_dataset=_dataset(),
                            args=_config(output_dir=str(tmp_path)))
    trainer.train()

    after = [m.scale for _, m in iter_mlx_lora_modules(model)]
    assert after == before, "LoRA scales not restored to their pre-training values"


@metal_only
def test_lora_scales_restored_when_forward_throws(tmp_path, monkeypatch):
    """The reference forward runs with scales set to 0; if it throws, the
    finally must restore scales so the model is not left adapter-disabled."""
    import unsloth_zoo.mlx.trainer as T
    from unsloth_zoo.mlx.trainer import MLXKTOTrainer
    from unsloth_zoo.mlx.utils import iter_mlx_lora_modules
    model, tok = _load_peft()
    before = [m.scale for _, m in iter_mlx_lora_modules(model)]

    # Fail on the 2nd sum-logp call: that is the reference forward, which runs
    # while adapter scales are 0 (policy-KL is the 1st call, scales still on).
    real = T._kto_sum_logp
    calls = {"n": 0}
    def flaky(logits, labels):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected forward failure")
        return real(logits, labels)
    monkeypatch.setattr(T, "_kto_sum_logp", flaky)

    trainer = MLXKTOTrainer(model=model, tokenizer=tok, train_dataset=_dataset(),
                            args=_config(output_dir=str(tmp_path)))
    with pytest.raises(RuntimeError, match="injected forward failure"):
        trainer.train()

    after = [m.scale for _, m in iter_mlx_lora_modules(model)]
    assert after == before, "scales left disabled after a throwing reference forward"
    assert all(s != 0 for s in after), "adapters left at scale 0"


@metal_only
def test_kto_sum_logp_matches_numpy():
    """Summed-logp extractor vs a from-scratch numpy log-softmax reference."""
    import numpy as np
    from unsloth_zoo.mlx.trainer import _kto_sum_logp
    rng = np.random.default_rng(1)
    B, T, V = 3, 6, 11
    logits = rng.normal(0, 1, size=(B, T, V)).astype(np.float32)
    labels = rng.integers(0, V, size=(B, T)).astype(np.int64)
    labels[0, :2] = -100; labels[1, :3] = -100; labels[2, :1] = -100  # masked prompt prefixes

    got = np.array(_kto_sum_logp(mx.array(logits), mx.array(labels)))

    inp, tgt = logits[:, :-1, :], labels[:, 1:]
    m = inp.max(axis=-1, keepdims=True)
    logsm = inp - (m + np.log(np.exp(inp - m).sum(axis=-1, keepdims=True)))
    ref = np.zeros(B)
    for b in range(B):
        for t in range(tgt.shape[1]):
            if tgt[b, t] != -100:
                ref[b] += logsm[b, t, tgt[b, t]]
    assert np.abs(got - ref).max() < 1e-4


@metal_only
def test_non_peft_model_raises_lora_only_error(tmp_path):
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXKTOTrainer
    mx.random.seed(3407)
    model, tok = FastMLXModel.from_pretrained(MODEL, max_seq_length=256, load_in_4bit=True)
    # No get_peft_model -> no LoRA adapters -> reference forward is impossible.
    trainer = MLXKTOTrainer(model=model, tokenizer=tok, train_dataset=_dataset(),
                            args=_config(output_dir=str(tmp_path)))
    with pytest.raises(ValueError) as exc:
        trainer.train()
    assert "LoRA" in str(exc.value)
