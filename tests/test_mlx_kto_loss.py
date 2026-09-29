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

"""MLX KTO primitives and guards under the torch shim (no Metal needed)."""

from __future__ import annotations

import sys

import numpy as np
import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    # Same isolation as test_mlx_preference.py: fresh unsloth_zoo.mlx on the shim, originals restored.
    from mlx_simulation import restore_modules, simulate_mlx_on_torch, snapshot_modules
    from mlx_simulation.mlx_stub import _MLXFinder

    def _owned(name):
        return any(name == p or name.startswith(p + ".") for p in ("unsloth_zoo.mlx", "mlx", "mlx_lm", "mlx_vlm"))

    real_modules = snapshot_modules(_owned)
    simulate_mlx_on_torch()
    for name in list(sys.modules):
        if name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx."):
            sys.modules.pop(name, None)
    yield
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _MLXFinder)]
    restore_modules(real_modules, _owned)


class _WordTokenizer:
    """One id per word; encode() prepends BOS like an HF tokenizer."""
    BOS, EOS = 7, 99
    bos_token = "<s>"
    pad_token_id = 0
    eos_token_id = EOS

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def encode(self, text, add_special_tokens=True):
        ids = [10 + len(w) for w in text.split()]
        return [self.BOS] + ids if add_special_tokens else ids


def _trl_kto_loss(policy, reference, desirable, kl, beta, wd, wu):
    # trl/trainer/kto_trainer.py _compute_loss, loss_type="kto", in float64.
    lr = policy.astype(np.float64) - reference
    chosen = wd * (1 - 1 / (1 + np.exp(-beta * (lr[desirable] - kl))))
    rejected = wu * (1 - 1 / (1 + np.exp(-beta * (kl - lr[~desirable]))))
    return np.concatenate([chosen, rejected]).mean()


@pytest.mark.parametrize("weights", [(1.0, 1.0), (1.33, 1.0), (1.0, 1.5)])
@pytest.mark.parametrize("labels", ["mixed", "all_desirable", "all_undesirable"])
def test_kto_loss_matches_trl(weights, labels):
    import mlx.core as mx
    from unsloth_zoo.mlx.trainer import _kto_loss, _kto_kl_baseline
    rng = np.random.default_rng(0)
    policy, reference = rng.normal(-8, 2, 5).astype(np.float32), rng.normal(-8, 2, 5).astype(np.float32)
    pol_kl, ref_kl = rng.normal(-8, 2, 5).astype(np.float32), rng.normal(-9, 2, 5).astype(np.float32)
    desirable = {"mixed": np.array([1, 0, 1, 1, 0], bool),
                 "all_desirable": np.ones(5, bool), "all_undesirable": np.zeros(5, bool)}[labels]
    kl = float(_kto_kl_baseline(mx.array(pol_kl), mx.array(ref_kl)))
    assert kl == pytest.approx(max(float((pol_kl.astype(np.float64) - ref_kl).mean()), 0.0), abs=1e-5)
    got = float(_kto_loss(mx.array(policy), mx.array(reference), mx.array(desirable),
                          mx.array(kl), 0.1, *weights))
    assert got == pytest.approx(_trl_kto_loss(policy, reference, desirable, kl, 0.1, *weights), abs=1e-6)


def test_kl_baseline_clamps_negative_to_zero():
    import mlx.core as mx
    from unsloth_zoo.mlx.trainer import _kto_kl_baseline
    low, high = mx.array([-30.0, -28.0]), mx.array([-20.0, -22.0])
    assert float(_kto_kl_baseline(low, high)) == 0.0
    assert float(_kto_kl_baseline(high, low)) == 8.0


def test_kto_tokenize_row_bos_eos_and_caps():
    from unsloth_zoo.mlx.trainer import _kto_tokenize_row, MLXKTOConfig
    tok = _WordTokenizer()
    p, c = _kto_tokenize_row(tok, {"prompt": "Question: two?", "completion": " four"}, MLXKTOConfig())
    assert p[0] == tok.BOS and tok.BOS not in c and c == [14, tok.EOS]
    _, c = _kto_tokenize_row(tok, {"prompt": "q", "completion": " a b"}, MLXKTOConfig(append_eos=False))
    assert tok.EOS not in c
    _, c = _kto_tokenize_row(tok, {"prompt": "q", "completion": " w" * 20}, MLXKTOConfig(max_completion_length=4, max_prompt_length=0))
    assert len(c) == 4 and c[-1] == tok.EOS
    p, c = _kto_tokenize_row(tok, {"prompt": "a b c", "completion": " w" * 20}, MLXKTOConfig(max_length=8, max_prompt_length=0))
    assert p == [] and len(c) == 8 and c[-1] == tok.EOS


def test_kto_tokenize_row_passes_tools_to_the_chat_template(monkeypatch):
    import unsloth_zoo.mlx.trainer as T
    seen = {}

    def fake(tokenizer, item, **kwargs):
        seen.update(item)
        return [1, 2, 3], [-100, 2, 3]

    monkeypatch.setattr(T, "_tokenize_mlx_prompt_completion_row", fake)
    tools, kwargs = [{"type": "function"}], {"enable_thinking": False}
    row = {"prompt": [{"role": "user", "content": "x"}], "completion": [{"role": "assistant", "content": "y"}],
           "label": True, "tools": tools, "chat_template_kwargs": kwargs}
    assert T._kto_tokenize_row(_WordTokenizer(), row, T.MLXKTOConfig()) == ([1], [2, 3])
    assert seen["tools"] is tools and seen["chat_template_kwargs"] is kwargs and "label" not in seen


def test_kto_labels_are_parsed_not_truth_tested():
    from unsloth_zoo.mlx.trainer import _kto_parse_label
    assert [_kto_parse_label(v) for v in ("false", "0", "no", "true", "1", True, 0)] == \
        [False, False, False, True, True, True, False]
    with pytest.raises(ValueError):
        _kto_parse_label("maybe")


def test_kto_rows_roll_kl_completions_and_fit_max_length():
    from unsloth_zoo.mlx.trainer import _kto_batch, _kto_rows, MLXKTOConfig
    rows = [
        {"prompt": "x " * 20, "completion": " y", "label": True},
        {"prompt": "q", "completion": " z" * 20, "label": "false"},
        {"prompt": "lone", "completion": " tail", "label": True},
    ]
    args = MLXKTOConfig(per_device_train_batch_size=2, max_length=24, max_prompt_length=0)
    kto_rows = _kto_rows(rows, _WordTokenizer(), args)
    assert len(kto_rows) == 3
    assert kto_rows[2][3] == kto_rows[1][1]  # the lone tail scores the previous row's completion
    b = _kto_batch(kto_rows[:2], pad_id=0)
    assert b["desirable"].tolist() == [True, False]
    assert b["comp_ids"].shape[1] <= 24 and b["kl_ids"].shape[1] <= 24

    def comp(ids, labels):
        return [i for i, lab in zip(ids, labels) if lab != -100]

    rows_c = [comp(b["comp_ids"][i].tolist(), b["comp_labels"][i].tolist()) for i in range(2)]
    rows_kl = [comp(b["kl_ids"][i].tolist(), b["kl_labels"][i].tolist()) for i in range(2)]
    assert rows_kl == [rows_c[1], rows_c[0]]  # TRL roll: row i scores row i-1's completion


class _ScalarModel:
    """Trainable state is one scalar; the scored value is that scalar."""
    def __init__(self, w):
        import mlx.core as mx
        self.w = mx.array(w)

    def trainable_parameters(self):
        return {"w": self.w}

    def update(self, tree):
        self.w = tree["w"]


def test_kto_reference_scores_start_weights_and_restores(monkeypatch):
    import unsloth_zoo.mlx.trainer as T
    from mlx.utils import tree_flatten
    model = _ScalarModel(1.0)
    start = tree_flatten(model.trainable_parameters())
    model.update({"w": T.mx.array(5.0)})
    monkeypatch.setattr(T, "_kto_logps", lambda m, ids, labels: m.w)
    ref, ref_kl = T._kto_reference_logps(model, start, {k: None for k in ("comp_ids", "comp_labels", "kl_ids", "kl_labels")})
    assert float(ref) == float(ref_kl) == 1.0 and float(model.w) == 5.0

    def boom(m, ids, labels):
        raise RuntimeError("forward failed")
    monkeypatch.setattr(T, "_kto_logps", boom)
    with pytest.raises(RuntimeError):
        T._kto_reference_logps(model, start, {k: None for k in ("comp_ids", "comp_labels", "kl_ids", "kl_labels")})
    assert float(model.w) == 5.0


_UNSUPPORTED = {
    "": dict(),
    "loss_type": dict(args=dict(loss_type="apo_zero_unpaired")),
    "ref_model": dict(ref_model=object()),
    "without LoRA": dict(lora=False),
    "gated-delta": dict(gated=True),
    "vision-language": dict(processor=type("P", (), {"image_processor": None})()),
    "per_device_train_batch_size": dict(args=dict(per_device_train_batch_size=1)),
    "streaming": dict(args=dict(streaming=True)),
    "lora_plus_ratio": dict(args=dict(lora_plus_ratio=16.0)),
    "embedding_learning_rate": dict(args=dict(embedding_learning_rate=5e-5)),
    "neftune_noise_alpha": dict(args=dict(neftune_noise_alpha=5.0)),
    "resume_from_checkpoint": dict(resume="ckpt"),
    "save_steps": dict(args=dict(save_steps=10)),
    "eval_dataset": dict(eval_dataset=[]),
    "callbacks": dict(kwargs=dict(callbacks=[object()])),
}


@pytest.mark.parametrize("name", list(_UNSUPPORTED))
def test_kto_rejects_unsupported_options(name, monkeypatch):
    import unsloth_zoo.mlx.trainer as T
    case = _UNSUPPORTED[name]
    monkeypatch.setattr(T, "iter_mlx_lora_modules",
                        lambda m: iter([("m", object())] if case.get("lora", True) else []))
    monkeypatch.setattr(T, "model_has_gated_delta_layers", lambda m: case.get("gated", False))
    tr = T.MLXKTOTrainer.__new__(T.MLXKTOTrainer)
    tr.model, tr.tokenizer, tr.train_dataset = object(), object(), []
    tr.processor = case.get("processor")
    tr.eval_dataset = case.get("eval_dataset")
    tr.ref_model = case.get("ref_model")
    tr._kto_ignored_kwargs = sorted(case.get("kwargs", {}))
    tr.args = T.MLXKTOConfig(**case.get("args", {}))
    monkeypatch.setattr(T.MLXKTOTrainer, "distributed_world_size", 1, raising=False)
    if not name:
        tr._reject_unsupported(None)  # a supported setup must pass
        return
    with pytest.raises(NotImplementedError, match=name):
        tr._reject_unsupported(case.get("resume"))


def test_kto_config_inherits_parent_init():
    from unsloth_zoo.mlx.trainer import MLXKTOConfig, MLXTrainingConfig
    assert MLXKTOConfig.__init__ is MLXTrainingConfig.__init__
    c = MLXKTOConfig(beta=0.5, desirable_weight=2.0)
    assert hasattr(c, "_unsloth_mlx_warmup_steps_explicit")
    assert (c.beta, c.desirable_weight, c.loss_type) == (0.5, 2.0, "kto")
