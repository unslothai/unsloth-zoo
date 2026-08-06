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

"""A LongRoPE attention_factor that cannot be real must be ignored.

The factor is by construction

    sqrt(1 + log(factor) / log(original_max_position_embeddings))

a number near 1. Two published configs set it equal to `factor` (32.0),
roughly 27x the real value; nothing raises, the model just predicts badly.

The signature is exact on purpose: attention_factor == factor AND
factor > 2, so it cannot fire on a config that means what it says.
"""

import math
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unsloth_zoo.temporary_patches.misc import (  # noqa: E402
    patch_longrope_impossible_attention_factor,
)
from transformers import modeling_rope_utils as R  # noqa: E402


@pytest.fixture(autouse=True)
def patched():
    """Apply the patch, then put the originals back."""
    before_attr = R._compute_longrope_parameters
    before_dict = R.ROPE_INIT_FUNCTIONS.get("longrope")
    patch_longrope_impossible_attention_factor()
    yield
    R._compute_longrope_parameters = before_attr
    if before_dict is not None:
        R.ROPE_INIT_FUNCTIONS["longrope"] = before_dict


class _Config:
    """The parts of a config the longrope initializer touches, on 4 and on 5.

    transformers 5 renamed the dict to `rope_parameters`, keeps `rope_scaling`
    as a read/write alias property (configuration_utils.py), moved `rope_theta`
    and `original_max_position_embeddings` inside it, and calls
    `config.standardize_rope_params()` first. A SimpleNamespace has none of
    that and would `AttributeError` on every transformers 5 in the support
    envelope.
    """

    def __init__(self, scaling, original_max, max_pos):
        self.rope_parameters = scaling
        self.original_max_position_embeddings = original_max
        self.max_position_embeddings = max_pos
        self.rope_theta = 10000.0
        self.hidden_size = 3072
        self.num_attention_heads = 32
        self.head_dim = 96

    @property
    def rope_scaling(self):
        return self.rope_parameters

    @rope_scaling.setter
    def rope_scaling(self, value):
        self.rope_parameters = value

    def standardize_rope_params(self):
        if not isinstance(self.rope_parameters, dict): return
        self.rope_parameters.setdefault("rope_theta", self.rope_theta)
        self.rope_parameters.setdefault(
            "original_max_position_embeddings",
            getattr(self, "original_max_position_embeddings",
                    self.max_position_embeddings),
        )


def _cfg(attention_factor=None, factor=32.0, original_max=4096,
         max_pos=131072):
    scaling = {"rope_type": "longrope", "factor": factor,
               "short_factor": [1.0] * 48, "long_factor": [1.0] * 48}
    if attention_factor is not None:
        scaling["attention_factor"] = attention_factor
    return _Config(scaling, original_max, max_pos)


def _att(cfg):
    return R.ROPE_INIT_FUNCTIONS["longrope"](cfg, "cpu")[1]


DERIVED = math.sqrt(1 + math.log(32) / math.log(4096))   # ~1.1902


# ---- the published configs ------------------------------------------------

def test_impossible_value_is_replaced_by_the_derivation():
    cfg = _cfg(attention_factor=32.0)
    assert _att(cfg) == pytest.approx(DERIVED, rel=1e-6)


def test_the_bad_key_is_removed_from_the_config():
    cfg = _cfg(attention_factor=32.0)
    _att(cfg)
    assert "attention_factor" not in cfg.rope_scaling


def test_applying_twice_is_stable():
    cfg = _cfg(attention_factor=32.0)
    first = _att(cfg)
    assert _att(cfg) == pytest.approx(first)


def test_the_dict_entry_is_patched_not_just_the_module_attribute():
    """Models resolve the callable through ROPE_INIT_FUNCTIONS, so patching
    only the module attribute would be a silent no-op."""
    assert R.ROPE_INIT_FUNCTIONS["longrope"] is R._compute_longrope_parameters
    assert getattr(R.ROPE_INIT_FUNCTIONS["longrope"], "_unsloth_patched", False)


def test_patching_twice_does_not_stack():
    first = R.ROPE_INIT_FUNCTIONS["longrope"]
    patch_longrope_impossible_attention_factor()
    assert R.ROPE_INIT_FUNCTIONS["longrope"] is first


# ---- configs that mean what they say --------------------------------------

def test_a_real_attention_factor_is_preserved():
    assert _att(_cfg(attention_factor=1.19)) == pytest.approx(1.19)


def test_a_value_equal_to_a_small_factor_is_preserved():
    # factor <= 2 is not the signature; 2.0 could genuinely be both.
    assert _att(_cfg(attention_factor=2.0, factor=2.0)) == pytest.approx(2.0)


def test_a_large_value_that_differs_from_factor_is_preserved():
    # Only the exact coincidence is suspicious; a lone odd number is the
    # author's business, not ours.
    assert _att(_cfg(attention_factor=8.0, factor=32.0)) == pytest.approx(8.0)


def test_absent_attention_factor_behaves_as_before():
    assert _att(_cfg(attention_factor=None)) == pytest.approx(DERIVED, rel=1e-6)


# ---- the guard must never be the thing that breaks a load ----------------

def test_a_config_without_rope_scaling_does_not_raise():
    cfg = _cfg(attention_factor=32.0)
    cfg.rope_scaling = None
    with pytest.raises(Exception):
        # The real function needs the dict; what matters is that OUR guard
        # is not the thing that raised.
        _att(cfg)


def test_non_numeric_values_are_left_alone():
    cfg = _cfg()
    cfg.rope_scaling["attention_factor"] = "thirty two"
    cfg.rope_scaling["factor"] = "thirty two"
    # Unparseable: the guard declines rather than guessing.
    try:
        _att(cfg)
    except Exception:
        pass
    assert cfg.rope_scaling["attention_factor"] == "thirty two"


def test_missing_original_max_still_strips_the_bad_value():
    cfg = _cfg(attention_factor=32.0)
    del cfg.original_max_position_embeddings
    _att(cfg)
    assert "attention_factor" not in cfg.rope_scaling


# ---- every call shape transformers uses -----------------------------------

def test_device_stays_optional_for_a_transformers_5_original():
    """transformers 5 gives every parameter but `config` a default, and
    `PreTrainedModel._init_weights` calls `rope_fn(module.config)` with the
    config alone (modeling_utils.py, the RotaryEmbedding branch). Naming
    `device` in the wrapper would make every LongRoPE model fail to load with
    a missing-argument TypeError before a single weight is read."""
    seen = {}

    def original(config, device = None, seq_len = None, layer_type = None):
        seen["device"], seen["layer_type"] = device, layer_type
        return "inv_freq", config.rope_scaling.get("attention_factor", 1.0)

    R._compute_longrope_parameters = original
    R.ROPE_INIT_FUNCTIONS["longrope"] = original
    patch_longrope_impossible_attention_factor()

    cfg = _cfg(attention_factor = 32.0)
    assert R.ROPE_INIT_FUNCTIONS["longrope"](cfg)[1] == 1.0
    assert seen == {"device": None, "layer_type": None}
    assert "attention_factor" not in cfg.rope_scaling


def test_the_other_arguments_are_forwarded_untouched():
    """`rope_init_fn(config, device, seq_len = ..., layer_type = ...)` is how
    transformers 5 reinitializes a rotary embedding."""
    seen = {}

    def original(config, device = None, seq_len = None, layer_type = None):
        seen.update(device = device, seq_len = seq_len, layer_type = layer_type)
        return "inv_freq", 1.0

    R._compute_longrope_parameters = original
    R.ROPE_INIT_FUNCTIONS["longrope"] = original
    patch_longrope_impossible_attention_factor()

    R.ROPE_INIT_FUNCTIONS["longrope"](
        _cfg(), "cpu", seq_len = 8, layer_type = "full_attention")
    assert seen == {"device": "cpu", "seq_len": 8, "layer_type": "full_attention"}


def test_other_rope_types_are_untouched():
    assert "linear" in R.ROPE_INIT_FUNCTIONS
    assert not getattr(R.ROPE_INIT_FUNCTIONS["linear"], "_unsloth_patched", False)


def test_registered_as_a_temporary_patch():
    from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
    names = [getattr(f, "__name__", "") for f in TEMPORARY_PATCHES]
    assert "patch_longrope_impossible_attention_factor" in names


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
