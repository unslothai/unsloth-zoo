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

"""MLX KTO loss primitives under the torch shim (no Metal needed)."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _fixed_logps():
    rng = np.random.default_rng(0)
    return dict(
        pol_ch=rng.normal(-8, 2, size=3).astype(np.float32),
        pol_rej=rng.normal(-9, 2, size=2).astype(np.float32),
        pol_kl=rng.normal(-8.5, 2, size=5).astype(np.float32),
        ref_ch=rng.normal(-8, 2, size=3).astype(np.float32),
        ref_rej=rng.normal(-9, 2, size=2).astype(np.float32),
        ref_kl=rng.normal(-8.5, 2, size=5).astype(np.float32),
    )


def _numpy_kto_loss(pol_ch, pol_rej, ref_ch, ref_rej, kl, beta, wd, wu):
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    ch = wd * (1 - sig(beta * ((pol_ch - ref_ch) - kl)))
    rej = wu * (1 - sig(beta * (kl - (pol_rej - ref_rej))))
    return float(np.concatenate([ch, rej]).mean())


# Pinned: MLX == numpy == TRL-torch kto_loss to ~1e-8.
_EXPECTED = {(1.0, 1.0): 0.4772096276283264,
             (1.33, 1.0): 0.566311240196228,
             (1.0, 1.5): 0.5808119177818298}
_EXPECTED_KL = 0.31288090348243713


def test_kto_loss_matches_reference_across_weights():
    import mlx.core as mx
    from unsloth_zoo.mlx.trainer import _kto_loss, _kto_kl_baseline
    f = _fixed_logps()
    kl = float(_kto_kl_baseline(mx.array(f["pol_kl"]), mx.array(f["ref_kl"])))
    assert kl == pytest.approx(_EXPECTED_KL, abs=1e-6)
    for (wd, wu), expected in _EXPECTED.items():
        loss = float(_kto_loss(
            mx.array(f["pol_ch"]), mx.array(f["pol_rej"]),
            mx.array(f["ref_ch"]), mx.array(f["ref_rej"]), mx.array(kl),
            0.1, wd, wu,
        ))
        npy = _numpy_kto_loss(f["pol_ch"], f["pol_rej"], f["ref_ch"], f["ref_rej"], kl, 0.1, wd, wu)
        assert loss == pytest.approx(npy, abs=1e-6), f"w=({wd},{wu}) MLX vs numpy"
        assert loss == pytest.approx(expected, abs=1e-6), f"w=({wd},{wu}) vs pinned"


def test_kl_baseline_clamps_negative_to_zero():
    import mlx.core as mx
    from unsloth_zoo.mlx.trainer import _kto_kl_baseline
    pol_kl = mx.array([-30.0, -28.0, -35.0])
    ref_kl = mx.array([-20.0, -22.0, -19.0])
    assert float(_kto_kl_baseline(pol_kl, ref_kl)) == 0.0
    assert float(_kto_kl_baseline(ref_kl, pol_kl)) > 0.0


@pytest.mark.parametrize("labels_present", ["desirable_only", "undesirable_only"])
def test_kto_loss_single_label_batches_are_finite(labels_present):
    import mlx.core as mx
    from unsloth_zoo.mlx.trainer import _kto_loss
    empty = mx.array(np.array([], dtype=np.float32))
    if labels_present == "desirable_only":
        pol_ch, ref_ch = mx.array([-8.0, -9.0]), mx.array([-8.3, -9.4])
        pol_rej = ref_rej = empty
    else:
        pol_rej, ref_rej = mx.array([-8.0, -9.0]), mx.array([-8.3, -9.4])
        pol_ch = ref_ch = empty
    loss = float(_kto_loss(pol_ch, pol_rej, ref_ch, ref_rej, mx.array(0.5), 0.1, 1.0, 1.0))
    assert np.isfinite(loss) and 0.0 <= loss <= 1.0


# _kto_sum_logp is tested in test_mlx_kto_train_metal.py: the shim mis-resolves its dtype.


def test_build_kto_batches_requires_batch_size_two():
    from unsloth_zoo.mlx.trainer import _build_kto_batches, MLXKTOConfig

    class _DummyTokenizer:
        pad_token_id = 0
        eos_token_id = 0
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2, 3]}

        def encode(self, text, add_special_tokens=True):
            return self(text)["input_ids"]

    dataset = [{"prompt": "a", "completion": " b", "label": True},
               {"prompt": "c", "completion": " d", "label": False}]
    with pytest.raises(ValueError) as exc:
        _build_kto_batches(dataset, _DummyTokenizer(), MLXKTOConfig(per_device_train_batch_size=1))
    msg = str(exc.value)
    assert "per_device_train_batch_size" in msg and ">= 2" in msg

    batches = _build_kto_batches(dataset, _DummyTokenizer(), MLXKTOConfig(per_device_train_batch_size=2))
    assert len(batches) == 1
    for key in ("comp_ids", "comp_labels", "kl_ids", "kl_labels", "label"):
        assert key in batches[0]


def test_kto_string_labels_parsed_not_truthy():
    from unsloth_zoo.mlx.trainer import (
        _build_kto_batches, _kto_parse_label, MLXKTOConfig,
    )

    assert _kto_parse_label("false") is False
    assert _kto_parse_label("0") is False
    assert _kto_parse_label("0.0") is False
    assert _kto_parse_label("true") is True
    assert _kto_parse_label("1") is True
    assert _kto_parse_label(True) is True and _kto_parse_label(0) is False
    with pytest.raises(ValueError):
        _kto_parse_label("maybe")

    class _DummyTokenizer:
        pad_token_id = 0
        eos_token_id = 0
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2, 3]}

        def encode(self, text, add_special_tokens=True):
            return self(text)["input_ids"]

    dataset = [{"prompt": "a", "completion": " b", "label": "false"},
               {"prompt": "c", "completion": " d", "label": "0"}]
    batches = _build_kto_batches(
        dataset, _DummyTokenizer(), MLXKTOConfig(per_device_train_batch_size=2),
    )
    assert batches[0]["label"] == [False, False], batches[0]["label"]


def test_kto_tokenize_row_caps_completion_exceeding_max_length():
    from unsloth_zoo.mlx.trainer import _kto_tokenize_row, MLXKTOConfig

    class _LenTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text.split())))}

        def encode(self, text, add_special_tokens=True):
            return self(text)["input_ids"]

    args = MLXKTOConfig(max_length=8, max_completion_length=None, max_prompt_length=0)
    p, c = _kto_tokenize_row(_LenTokenizer(), "a b c", " ".join(["w"] * 20), args)
    assert len(p) + len(c) <= 8, (len(p), len(c))
    assert len(p) == 0 and len(c) == 8


def test_kto_appends_eos_and_preserves_it_under_truncation():
    from unsloth_zoo.mlx.trainer import _kto_tokenize_row, MLXKTOConfig

    EOS = 99

    class _Tok:
        eos_token_id = EOS
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1 + i for i in range(len(text.split()))]}

        def encode(self, text, add_special_tokens=True):
            return self(text)["input_ids"]

    big = MLXKTOConfig(max_length=1024, max_completion_length=None, max_prompt_length=0)

    _, c = _kto_tokenize_row(_Tok(), "q", "a b c", big)
    assert c[-1] == EOS and c.count(EOS) == 1

    class _TokEndsEos(_Tok):
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": super().__call__(text)["input_ids"] + [EOS]}

        def encode(self, text, add_special_tokens=True):
            return self(text)["input_ids"]
    _, c = _kto_tokenize_row(_TokEndsEos(), "q", "a b", big)
    assert c[-1] == EOS and c.count(EOS) == 1

    no_eos = MLXKTOConfig(max_length=1024, max_completion_length=None,
                          max_prompt_length=0, append_eos=False)
    _, c = _kto_tokenize_row(_Tok(), "q", "a b c", no_eos)
    assert EOS not in c

    cap = MLXKTOConfig(max_length=1024, max_completion_length=4, max_prompt_length=0)
    _, c = _kto_tokenize_row(_Tok(), "q", " ".join(["w"] * 20), cap)
    assert len(c) == 4 and c[-1] == EOS

    ml = MLXKTOConfig(max_length=4, max_completion_length=None, max_prompt_length=0)
    p, c = _kto_tokenize_row(_Tok(), "q", " ".join(["w"] * 20), ml)
    assert len(p) + len(c) <= 4 and c[-1] == EOS


def test_kto_rejects_streaming_dataset():
    from unsloth_zoo.mlx.trainer import _build_kto_batches, MLXKTOConfig

    class _DummyTokenizer:
        pad_token_id = 0
        eos_token_id = 0
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2, 3]}

        def encode(self, text, add_special_tokens=True):
            return self(text)["input_ids"]

    ds = [{"prompt": "a", "completion": " b", "label": True},
          {"prompt": "c", "completion": " d", "label": False}]
    with pytest.raises(NotImplementedError, match="streaming"):
        _build_kto_batches(ds, _DummyTokenizer(),
                           MLXKTOConfig(per_device_train_batch_size=2, streaming=True))

    def _gen():
        yield from ds
    with pytest.raises(NotImplementedError, match="streaming"):
        _build_kto_batches(_gen(), _DummyTokenizer(),
                           MLXKTOConfig(per_device_train_batch_size=2))


def test_kto_rejects_non_kto_loss_type():
    from unsloth_zoo.mlx.trainer import MLXKTOTrainer, MLXKTOConfig
    with pytest.raises(ValueError, match="loss_type='kto'"):
        MLXKTOTrainer(object(), object(), [],
                      args=MLXKTOConfig(loss_type="apo_zero_unpaired"))


def test_kto_rejects_ref_model_kwarg():
    from unsloth_zoo.mlx.trainer import MLXKTOTrainer, MLXKTOConfig
    with pytest.raises(ValueError, match="ref_model"):
        MLXKTOTrainer(object(), object(), [], args=MLXKTOConfig(), ref_model=object())


def test_kto_rejects_gated_delta_and_vlm(monkeypatch):
    import unsloth_zoo.mlx.trainer as T
    from unsloth_zoo.mlx.trainer import MLXKTOTrainer, MLXKTOConfig

    def _mk(tokenizer):
        tr = MLXKTOTrainer.__new__(MLXKTOTrainer)
        tr.args = MLXKTOConfig()
        tr.model = object()
        tr.tokenizer = tokenizer
        return tr

    monkeypatch.setattr(T, "iter_mlx_lora_modules", lambda m: [("m", object())])

    monkeypatch.setattr(T, "model_has_gated_delta_layers", lambda m: True)
    with pytest.raises(NotImplementedError, match="gated-delta"):
        _mk(object()).train()

    monkeypatch.setattr(T, "model_has_gated_delta_layers", lambda m: False)

    class _VLMTok:
        image_processor = object()
    with pytest.raises(NotImplementedError, match="vision-language"):
        _mk(_VLMTok()).train()


def test_kto_rejects_non_lora_trainable_params(monkeypatch):
    import unsloth_zoo.mlx.trainer as T
    from unsloth_zoo.mlx.trainer import (
        MLXKTOTrainer, MLXKTOConfig, _kto_model_has_non_lora_trainable_params,
    )

    class _Empty:
        def trainable_parameters(self):
            return {}
    assert _kto_model_has_non_lora_trainable_params(_Empty()) is False

    tr = MLXKTOTrainer.__new__(MLXKTOTrainer)
    tr.args = MLXKTOConfig()
    tr.model = object()
    tr.tokenizer = object()
    monkeypatch.setattr(T, "iter_mlx_lora_modules", lambda m: [("m", object())])
    monkeypatch.setattr(T, "model_has_gated_delta_layers", lambda m: False)
    monkeypatch.setattr(T, "_kto_model_has_non_lora_trainable_params", lambda m: True)
    with pytest.raises(ValueError, match="structural limit"):
        tr.train()


def test_kto_config_inherits_parent_init_not_a_generated_one():
    from unsloth_zoo.mlx.trainer import MLXKTOConfig, MLXTrainingConfig
    assert MLXKTOConfig.__init__ is MLXTrainingConfig.__init__
    c = MLXKTOConfig()
    assert hasattr(c, "_unsloth_mlx_warmup_steps_explicit")
    assert c.beta == 0.1 and c.loss_type == "kto"
    c2 = MLXKTOConfig(beta=0.5, desirable_weight=2.0)
    assert c2.beta == 0.5 and c2.desirable_weight == 2.0


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


def test_kto_tokenize_row_keeps_bos_on_the_prompt():
    from unsloth_zoo.mlx.trainer import _kto_tokenize_row, MLXKTOConfig

    tok = _WordTokenizer()
    p, c = _kto_tokenize_row(tok, "Question: two?", " four", MLXKTOConfig())
    assert p[0] == tok.BOS and tok.BOS not in c
    assert c == [14, tok.EOS]


def test_kto_kl_rows_refit_prompt_to_max_length():
    from unsloth_zoo.mlx.trainer import _build_kto_batches, MLXKTOConfig

    rows = [
        {"prompt": "x " * 20, "completion": " y", "label": True},
        {"prompt": "q", "completion": " z" * 20, "label": False},
    ]
    args = MLXKTOConfig(per_device_train_batch_size=2, max_length=24, max_prompt_length=0)
    batch = _build_kto_batches(rows, _WordTokenizer(), args)[0]
    assert batch["comp_ids"].shape[1] <= 24
    assert batch["kl_ids"].shape[1] <= 24
