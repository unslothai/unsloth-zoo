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

"""MLX KTO training on Apple Silicon; skipped where Metal is unavailable."""

import math

import pytest

try:
    import mlx.core as mx
    _METAL = mx.metal.is_available()
except Exception:
    _METAL = False

metal_only = pytest.mark.skipif(not _METAL, reason="requires Apple Silicon Metal")

MODEL = "unsloth/Qwen2.5-0.5B"


def _dataset(n=24):
    rows = []
    for i in range(n):
        prompt = f"### Question: what is {i} plus {i}?\n### Answer:"
        good = i % 2 == 0
        completion = f" {2 * i}." if good else f" {2 * i + 7}, wrong."
        rows.append({"prompt": prompt, "completion": completion, "label": good})
    return rows


def _load_peft():
    from unsloth_zoo.mlx.loader import FastMLXModel
    mx.random.seed(3407)
    model, tok = FastMLXModel.from_pretrained(MODEL, max_seq_length=256, load_in_4bit=True)
    model = FastMLXModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0, random_state=3407)
    return model, tok


def _trainer(tmp_path, data=None, model_tok=None, **overrides):
    from unsloth_zoo.mlx.trainer import MLXKTOConfig, MLXKTOTrainer
    args = dict(per_device_train_batch_size=4, max_steps=6, warmup_steps=1,
                gradient_accumulation_steps=1, learning_rate=1e-4, beta=0.1,
                logging_steps=99, seed=3407, report_to="none", output_dir=str(tmp_path))
    args.update(overrides)
    model, tok = model_tok or _load_peft()
    return MLXKTOTrainer(model=model, tokenizer=tok, train_dataset=data or _dataset(),
                         args=MLXKTOConfig(**args))


@metal_only
def test_kto_trains_and_saves(tmp_path):
    model, tok = _load_peft()
    trainer = _trainer(tmp_path, model_tok=(model, tok))
    output = trainer.train()
    hist = trainer._train_loss_history
    assert len(hist) == 6 and all(math.isfinite(x) for x in hist) and hist[-1] < hist[0], hist
    assert output.global_step == 6 and output["total_train_steps"] == 6
    assert all(math.isfinite(k) and k >= 0.0 for k in trainer._kl_history)
    assert {"adapters.safetensors", "adapter_config.json"} <= {p.name for p in tmp_path.iterdir()}
    trainer.stop_requested = True  # a stop from the finished run must not end the next one
    trainer.train()
    assert len(trainer._train_loss_history) == 6, "a second run must reset its history and stop flag"


@metal_only
def test_kto_step_counts_follow_accumulation_and_epochs(tmp_path):
    out = _trainer(tmp_path / "a", max_steps=0, gradient_accumulation_steps=2).train()
    assert out.global_step == out["total_train_steps"] == 3
    out = _trainer(tmp_path / "b", data=_dataset(8), max_steps=0, num_train_epochs=2).train()
    assert out.global_step == out["total_train_steps"] == 4
    out = _trainer(tmp_path / "d", data=_dataset(8), max_steps=0, num_train_epochs=1.5).train()
    assert out.global_step == out["total_train_steps"] == 3  # fractional epochs stop part-way
    # 3 batches per epoch at accumulation 2: the partial window still steps, per epoch.
    out = _trainer(tmp_path / "c", data=_dataset(12), max_steps=0, gradient_accumulation_steps=2,
                   num_train_epochs=2).train()
    assert out.global_step == out["total_train_steps"] == 4


@metal_only
def test_kto_grad_accum_weights_microbatches_by_rows(tmp_path):
    # Micro-batches [4, 2] at lr=0: the window must be (4*L1+2*L2)/6, not (L1+L2)/2.
    kw = dict(learning_rate=0.0, weight_decay=0.0, warmup_steps=0)
    tr = _trainer(tmp_path / "a", data=_dataset(6), max_steps=2, **kw)
    tr.train()
    L1, L2 = tr._train_loss_history
    tr = _trainer(tmp_path / "b", data=_dataset(6), max_steps=1, gradient_accumulation_steps=2, **kw)
    tr.train()
    weighted, equal = (4 * L1 + 2 * L2) / 6, (L1 + L2) / 2
    assert abs(weighted - equal) > 1e-5, "needs L1 != L2 to discriminate"
    assert tr._train_loss_history[0] == pytest.approx(weighted, abs=1e-4)


@metal_only
@pytest.mark.parametrize("disable_dropout", [True, False])
def test_kto_dropout_disabled_for_scoring(tmp_path, disable_dropout):
    # A trained adapter at lr=0: the policy equals its start snapshot, so without dropout
    # the loss is exactly 0.5 and the KL exactly 0; live dropout draws separate masks.
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.utils import iter_mlx_lora_modules
    model, tok = FastMLXModel.from_pretrained(MODEL, max_seq_length=256, load_in_4bit=True)
    model = FastMLXModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0.5, random_state=3407)
    for _, m in iter_mlx_lora_modules(model):
        m.lora_b = mx.random.normal(m.lora_b.shape, key=mx.random.key(0)).astype(m.lora_b.dtype) * 0.05
    tr = _trainer(tmp_path, model_tok=(model, tok), max_steps=1, learning_rate=0.0,
                  warmup_steps=0, disable_dropout=disable_dropout)
    tr.train()
    exact = tr._train_loss_history[0] == 0.5 and tr._kl_history[0] == 0.0
    assert exact is disable_dropout, (tr._train_loss_history, tr._kl_history)


@metal_only
def test_kto_restores_weights_when_reference_forward_throws(tmp_path, monkeypatch):
    import numpy as np
    import unsloth_zoo.mlx.trainer as T
    from mlx.utils import tree_flatten
    model, tok = _load_peft()
    before = {k: np.array(v) for k, v in tree_flatten(model.trainable_parameters())}

    def failing(model, ids, labels):  # the first scoring call runs under the swapped-in reference
        raise RuntimeError("injected reference failure")

    monkeypatch.setattr(T, "_kto_logps", failing)
    with pytest.raises(RuntimeError, match="injected reference failure"):
        _trainer(tmp_path, model_tok=(model, tok)).train()
    after = dict(tree_flatten(model.trainable_parameters()))
    assert all(np.array_equal(before[k], np.array(after[k])) for k in before)


@metal_only
def test_kto_sum_logp_matches_numpy():
    import numpy as np
    from unsloth_zoo.mlx.trainer import _kto_sum_logp
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 1, size=(3, 6, 11)).astype(np.float32)
    labels = rng.integers(0, 11, size=(3, 6)).astype(np.int64)
    labels[0, :2] = labels[1, :3] = labels[2, :1] = -100
    got = np.array(_kto_sum_logp(mx.array(logits), mx.array(labels)))
    inp, tgt = logits[:, :-1].astype(np.float64), labels[:, 1:]
    logsm = inp - np.log(np.exp(inp).sum(-1, keepdims=True))
    ref = [sum(logsm[b, t, tgt[b, t]] for t in range(tgt.shape[1]) if tgt[b, t] != -100) for b in range(3)]
    assert np.abs(got - np.array(ref)).max() < 1e-4


@metal_only
def test_non_peft_model_is_rejected(tmp_path):
    from unsloth_zoo.mlx.loader import FastMLXModel
    model, tok = FastMLXModel.from_pretrained(MODEL, max_seq_length=256, load_in_4bit=True)
    with pytest.raises(NotImplementedError, match="LoRA"):
        _trainer(tmp_path, model_tok=(model, tok)).train()
