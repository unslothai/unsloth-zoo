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

"""GKD guards under the torch shim. Preflight figures: Qwen2.5-0.5B student +
Qwen2.5-3B teacher (vocab 151936, 2.881 GB resident) on a 16 GB Mac."""

import pytest

GB = 1024 ** 3
VOCAB = 151936
RESIDENT = int(2.881 * GB)
SYSTEM_16GB = 16 * GB


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _distill():
    import unsloth_zoo.mlx.distill as distill
    return distill


class FakeTokenizer:

    def __init__(self, offset=0):
        self.offset = offset

    def encode(self, text):
        return [ord(c) + self.offset for c in text]


def test_identical_tokenizers_pass():
    d = _distill()
    assert d.assert_tokenizers_compatible(FakeTokenizer(), FakeTokenizer(), VOCAB, VOCAB)


def test_width_mismatch_is_rejected():
    d = _distill()
    with pytest.raises(ValueError, match="logit widths differ"):
        d.assert_tokenizers_compatible(FakeTokenizer(), FakeTokenizer(), 128256, 151936)


def test_width_mismatch_is_rejected_without_tokenizers():
    d = _distill()
    with pytest.raises(ValueError, match="logit widths differ"):
        d.assert_tokenizers_compatible(None, FakeTokenizer(), 128256, 151936)
    with pytest.warns(UserWarning, match="only the logit widths were checked"):
        assert d.assert_tokenizers_compatible(None, FakeTokenizer(), VOCAB, VOCAB)


class VocabTokenizer(FakeTokenizer):

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    def get_vocab(self):
        return dict(self.vocab)


def test_unprobed_vocabulary_difference_is_rejected():
    d = _distill()
    student = VocabTokenizer({"a": 0, "b": 1, "c": 2})
    teacher = VocabTokenizer({"a": 0, "b": 2, "c": 1})
    with pytest.raises(ValueError, match="map to different ids"):
        d.assert_tokenizers_compatible(student, teacher, VOCAB, VOCAB)


def test_added_token_difference_is_rejected():
    """Qwen3 student, Qwen2.5 teacher: only <think>-style added tokens differ."""
    d = _distill()
    student = VocabTokenizer({"a": 0, "b": 1, "<think>": 2})
    teacher = VocabTokenizer({"a": 0, "b": 1})
    with pytest.raises(ValueError, match="1 tokens map to different ids"):
        d.assert_tokenizers_compatible(student, teacher, VOCAB, VOCAB)


def test_width_matches_but_ids_differ_is_rejected():
    d = _distill()
    with pytest.raises(ValueError, match="encode text differently"):
        d.assert_tokenizers_compatible(FakeTokenizer(0), FakeTokenizer(1), VOCAB, VOCAB)


def test_probe_set_covers_more_than_plain_ascii():
    d = _distill()
    joined = "".join(d.TOKENIZER_PROBES)
    assert len(d.TOKENIZER_PROBES) >= 4
    assert "\n" in joined and "#" in joined


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_valid_beta_accepted(beta):
    assert _distill().validate_gkd_config(beta, 1.0, 0.0)


@pytest.mark.parametrize("beta", [-0.1, 1.1, 2.0])
def test_out_of_range_beta_rejected(beta):
    with pytest.raises(ValueError, match=r"gkd_beta must be in \[0, 1\]"):
        _distill().validate_gkd_config(beta, 1.0, 0.0)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_temperature_rejected(temperature):
    with pytest.raises(ValueError, match="gkd_temperature must be finite and > 0"):
        _distill().validate_gkd_config(0.5, temperature, 0.0)


def test_on_policy_lmbda_rejected_with_reason():
    with pytest.raises(ValueError, match="generation inside the training loop"):
        _distill().validate_gkd_config(0.5, 1.0, 0.5)


def test_preflight_allows_batch2_seq512():
    """Measured 10.16 GB on 16 GB: must not be rejected."""
    d = _distill()
    estimate = d.preflight_memory(2, 512, VOCAB, RESIDENT, SYSTEM_16GB, chunked=True)
    assert estimate < SYSTEM_16GB * d.MEMORY_SAFETY_FRACTION


def test_preflight_rejects_batch4_seq512():
    """Measured 17.16 GB: wedges Metal if allowed."""
    with pytest.raises(ValueError, match="does not fit in memory"):
        _distill().preflight_memory(4, 512, VOCAB, RESIDENT, SYSTEM_16GB, chunked=True)


def test_preflight_rejects_batch2_seq1024_unchunked():
    """Measured 17.39 GB unchunked: the config that crashed."""
    with pytest.raises(ValueError, match="does not fit in memory"):
        _distill().preflight_memory(2, 1024, VOCAB, RESIDENT, SYSTEM_16GB, chunked=False)


def test_preflight_message_names_everything_needed_to_act():
    d = _distill()
    with pytest.raises(ValueError) as excinfo:
        d.preflight_memory(4, 512, VOCAB, RESIDENT, SYSTEM_16GB, chunked=True)
    message = str(excinfo.value)
    assert "batch_size=4" in message and "seq_len=512" in message
    assert "2048 tokens" in message
    assert "estimated peak" in message
    assert "16.00 GB" in message
    assert f"vocab {VOCAB}" in message
    assert "tokens (e.g. batch_size=1" in message


def test_chunked_is_cheaper_than_unchunked_in_the_estimate():
    d = _distill()
    chunked = d.estimate_distillation_peak_bytes(2, 1024, VOCAB, RESIDENT, chunked=True)
    naive = d.estimate_distillation_peak_bytes(2, 1024, VOCAB, RESIDENT, chunked=False)
    assert chunked < naive
    assert d.CHUNKED_ACTIVATION_MULTIPLIER < d.NAIVE_ACTIVATION_MULTIPLIER


def test_largest_tokens_that_fit_is_consistent_with_preflight():
    d = _distill()
    budget = int(SYSTEM_16GB * d.MEMORY_SAFETY_FRACTION)
    max_tokens = d.largest_tokens_that_fit(VOCAB, RESIDENT, budget, chunked=True)
    assert max_tokens > 0
    d.preflight_memory(1, max_tokens, VOCAB, RESIDENT, SYSTEM_16GB, chunked=True)
    with pytest.raises(ValueError):
        d.preflight_memory(1, int(max_tokens * 1.5), VOCAB, RESIDENT, SYSTEM_16GB, chunked=True)


def test_skip_override_bypasses_the_raise_and_warns():
    d = _distill()
    with pytest.warns(UserWarning) as record:
        estimate = d.preflight_memory(
            4, 512, VOCAB, RESIDENT, SYSTEM_16GB, chunked=True, skip=True,
        )
    assert estimate > SYSTEM_16GB * d.MEMORY_SAFETY_FRACTION, (
        "override should return the same over-budget estimate, not a fudged one"
    )
    message = str(record[0].message)
    assert "gkd_skip_memory_preflight=True" in message
    assert "Command buffer execution failed" in message
    assert "estimated at" in message


def test_skip_override_does_not_warn_when_it_already_fits():
    import warnings
    d = _distill()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d.preflight_memory(2, 512, VOCAB, RESIDENT, SYSTEM_16GB, chunked=True, skip=True)


def test_skip_defaults_to_false_so_the_guard_is_on():
    d = _distill()
    with pytest.raises(ValueError, match="does not fit in memory"):
        d.preflight_memory(4, 512, VOCAB, RESIDENT, SYSTEM_16GB, chunked=True)


def test_larger_system_memory_allows_more():
    d = _distill()
    with pytest.raises(ValueError):
        d.preflight_memory(4, 512, VOCAB, RESIDENT, SYSTEM_16GB, chunked=True)
    assert d.preflight_memory(4, 512, VOCAB, RESIDENT, 64 * GB, chunked=True)


def test_chunk_size_at_or_above_seq_len_is_budgeted_unchunked():
    """chunk_size >= max_seq_length takes the unchunked loss branch at runtime."""
    from types import SimpleNamespace
    d = _distill()
    def build(chunk):
        args = SimpleNamespace(max_seq_length=1300, gkd_chunk_size=chunk)
        return d.build_gkd_loss_fn(None, args, VOCAB, 1, RESIDENT, SYSTEM_16GB)
    assert callable(build(128))
    for chunk in (0, 1299, 1300, 4096):
        with pytest.raises(ValueError, match="unchunked loss"):
            build(chunk)


@pytest.mark.parametrize("max_seq_length", [129, 140])
def test_single_chunk_batches_are_budgeted_unchunked(max_seq_length):
    """A <=128-position batch runs unchunked even when max_seq_length would chunk."""
    from types import SimpleNamespace
    d = _distill()
    args = SimpleNamespace(max_seq_length=max_seq_length, gkd_chunk_size=128)
    with pytest.raises(ValueError, match="unchunked loss"):
        d.build_gkd_loss_fn(None, args, VOCAB, 10, RESIDENT, SYSTEM_16GB)


def test_config_fields_exist_and_are_inert_by_default():
    import dataclasses
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig, _MLX_CONFIG_OPTIONAL_COPY_FIELDS
    names = [f.name for f in dataclasses.fields(MLXTrainingConfig)]
    for field in ("teacher_model_name_or_path", "gkd_beta", "gkd_temperature",
                  "gkd_lmbda", "gkd_chunk_size", "gkd_skip_memory_preflight"):
        assert field in names
        assert field in _MLX_CONFIG_OPTIONAL_COPY_FIELDS
    config = MLXTrainingConfig()
    assert config.teacher_model_name_or_path is None
    assert config.gkd_lmbda == 0.0
    assert config.gkd_chunk_size > 0
    assert config.gkd_skip_memory_preflight is False
    assert names[-6:] == list(_MLX_CONFIG_OPTIONAL_COPY_FIELDS)[-6:]


def test_evaluate_reports_gkd_loss_without_perplexity():
    """exp(JSD) is not a perplexity, and GKD has no preference stats."""
    import mlx.core as mx
    from unsloth_zoo.mlx.trainer import MLXTrainer

    class Model:
        def eval(self):
            pass

        def train(self, mode=True):
            pass

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model = Model()
    trainer.stop_requested = False

    def loss_fn(_model, _batch, _lengths, _labels):
        return mx.array(0.2), mx.array(4)
    loss_fn._unsloth_gkd = True

    loss, ppl = trainer._evaluate([(None, None, None)], loss_fn, is_vlm=False)
    assert loss == pytest.approx(0.2)
    assert ppl is None
    assert "eval_perplexity" not in trainer._last_eval_metrics
