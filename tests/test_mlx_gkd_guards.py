# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""GKD guards: tokenizer compatibility, config validation, memory preflight.

Pure logic and arithmetic, so this runs under the torch shim on Linux CI instead
of skipping. The numeric behaviour of the divergence lives in
test_mlx_gkd_loss.py, which needs real MLX and runs on Apple Silicon.

The preflight numbers below are the measured ones from a Qwen2.5-0.5B LoRA
student with a Qwen2.5-3B teacher (vocab 151936, 2.881 GB resident) on a 16 GB
machine. They matter because an oversized step does not raise a catchable OOM --
it aborts with "[METAL] Command buffer execution failed: Impacting
Interactivity" and wedges the GPU context.
"""

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
    """Encodes by a per-instance rule so probe agreement can be controlled."""

    def __init__(self, offset=0):
        self.offset = offset

    def encode(self, text):
        return [ord(c) + self.offset for c in text]


# --- tokenizer compatibility -------------------------------------------------
def test_identical_tokenizers_pass():
    d = _distill()
    assert d.assert_tokenizers_compatible(FakeTokenizer(), FakeTokenizer(), VOCAB, VOCAB)


def test_width_mismatch_is_rejected():
    d = _distill()
    with pytest.raises(ValueError, match="logit widths differ"):
        d.assert_tokenizers_compatible(FakeTokenizer(), FakeTokenizer(), 128256, 151936)


def test_width_matches_but_ids_differ_is_rejected():
    """The silent-misalignment case: same vocab size, different encodings.

    A width-only check would let this through and train the student against
    targets that do not correspond to its own tokens, with no error.
    """
    d = _distill()
    with pytest.raises(ValueError, match="encode text differently"):
        d.assert_tokenizers_compatible(FakeTokenizer(0), FakeTokenizer(1), VOCAB, VOCAB)


def test_probe_set_covers_more_than_plain_ascii():
    d = _distill()
    joined = "".join(d.TOKENIZER_PROBES)
    assert len(d.TOKENIZER_PROBES) >= 4
    assert "\n" in joined and "#" in joined


# --- config validation -------------------------------------------------------
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_valid_beta_accepted(beta):
    assert _distill().validate_gkd_config(beta, 1.0, 0.0)


@pytest.mark.parametrize("beta", [-0.1, 1.1, 2.0])
def test_out_of_range_beta_rejected(beta):
    with pytest.raises(ValueError, match=r"gkd_beta must be in \[0, 1\]"):
        _distill().validate_gkd_config(beta, 1.0, 0.0)


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_non_positive_temperature_rejected(temperature):
    with pytest.raises(ValueError, match="gkd_temperature must be > 0"):
        _distill().validate_gkd_config(0.5, temperature, 0.0)


def test_on_policy_lmbda_rejected_with_reason():
    """Off-policy only: on-policy needs generation in the training loop, which
    the MLX trainer does not have."""
    with pytest.raises(ValueError, match="generation inside the training loop"):
        _distill().validate_gkd_config(0.5, 1.0, 0.5)


# --- memory preflight --------------------------------------------------------
def test_preflight_allows_batch2_seq512():
    """Measured 10.16 GB on a 16 GB machine -- must not be rejected."""
    d = _distill()
    estimate = d.preflight_memory(2, 512, VOCAB, RESIDENT, SYSTEM_16GB, chunked=True)
    assert estimate < SYSTEM_16GB * d.MEMORY_SAFETY_FRACTION


def test_preflight_rejects_batch4_seq512():
    """Measured 17.16 GB -- over a 16 GB machine, wedges Metal if allowed."""
    with pytest.raises(ValueError, match="does not fit in memory"):
        _distill().preflight_memory(4, 512, VOCAB, RESIDENT, SYSTEM_16GB, chunked=True)


def test_preflight_rejects_batch2_seq1024_unchunked():
    """Measured 17.39 GB unchunked -- the config that actually crashed."""
    with pytest.raises(ValueError, match="does not fit in memory"):
        _distill().preflight_memory(2, 1024, VOCAB, RESIDENT, SYSTEM_16GB, chunked=False)


def test_preflight_message_names_everything_needed_to_act():
    d = _distill()
    with pytest.raises(ValueError) as excinfo:
        d.preflight_memory(4, 512, VOCAB, RESIDENT, SYSTEM_16GB, chunked=True)
    message = str(excinfo.value)
    assert "batch_size=4" in message and "seq_len=512" in message
    assert "2048 tokens" in message              # requested B*L
    assert "estimated peak" in message
    assert "16.00 GB" in message                 # detected system memory
    assert f"vocab {VOCAB}" in message           # largest that fits, at this width
    assert "tokens (e.g. batch_size=1" in message


def test_chunked_is_cheaper_than_unchunked_in_the_estimate():
    d = _distill()
    chunked = d.estimate_distillation_peak_bytes(2, 1024, VOCAB, RESIDENT, chunked=True)
    naive = d.estimate_distillation_peak_bytes(2, 1024, VOCAB, RESIDENT, chunked=False)
    assert chunked < naive
    assert d.CHUNKED_ACTIVATION_MULTIPLIER < d.NAIVE_ACTIVATION_MULTIPLIER


def test_largest_tokens_that_fit_is_consistent_with_preflight():
    """Whatever the message advertises as fitting must actually pass preflight."""
    d = _distill()
    budget = int(SYSTEM_16GB * d.MEMORY_SAFETY_FRACTION)
    max_tokens = d.largest_tokens_that_fit(VOCAB, RESIDENT, budget, chunked=True)
    assert max_tokens > 0
    d.preflight_memory(1, max_tokens, VOCAB, RESIDENT, SYSTEM_16GB, chunked=True)
    with pytest.raises(ValueError):
        d.preflight_memory(1, int(max_tokens * 1.5), VOCAB, RESIDENT, SYSTEM_16GB, chunked=True)


def test_skip_override_bypasses_the_raise_and_warns():
    """Refusing by default is right; refusing with no recourse is not.

    The estimator reads total system memory and is deliberately conservative, so
    a user who knows their configuration fits must be able to proceed -- loudly.
    """
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
    assert "Command buffer execution failed" in message   # names the wedge risk
    assert "estimated at" in message                       # names the estimate


def test_skip_override_does_not_warn_when_it_already_fits():
    """No warning noise for a configuration that was never going to be refused."""
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


def test_config_fields_exist_and_are_inert_by_default():
    import dataclasses
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig, _MLX_CONFIG_OPTIONAL_COPY_FIELDS
    names = [f.name for f in dataclasses.fields(MLXTrainingConfig)]
    for field in ("teacher_model_name_or_path", "gkd_beta", "gkd_temperature",
                  "gkd_lmbda", "gkd_chunk_size", "gkd_skip_memory_preflight"):
        assert field in names
        assert field in _MLX_CONFIG_OPTIONAL_COPY_FIELDS
    config = MLXTrainingConfig()
    assert config.teacher_model_name_or_path is None   # inert unless set
    assert config.gkd_lmbda == 0.0                     # off-policy default
    assert config.gkd_chunk_size > 0                   # chunked by DEFAULT
    assert config.gkd_skip_memory_preflight is False   # preflight ON by default
    # The GKD fields must remain an exact suffix: the initializer binds
    # positional args by field order.
    assert names[-6:] == list(_MLX_CONFIG_OPTIONAL_COPY_FIELDS)[-6:]
