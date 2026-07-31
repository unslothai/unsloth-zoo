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

"""Regression tests: a pre-preference config dump must keep its warmup_ratio.

MLXTrainingConfig.__init__ treats a wholesale copy carrying a DEFAULT
warmup_steps beside a non-default warmup_ratio as implicit, so the ratio wins.
That check (copied_all_fields) tolerates fields appended since the copy was
written. This PR adds four fields -- loss_type, orpo_beta, dpo_beta,
reference_free -- so a config dumped before it omits exactly those four, fails
the check, and a defaulted warmup_steps=5 silently overrides the ratio.

The four are declared LAST and listed in _MLX_CONFIG_OPTIONAL_COPY_FIELDS, so
the 67 fields that predate them keep their upstream positional slots and both
construction styles round-trip: legacy keyword dump and legacy positional.

UPSTREAM_FIELDS is the init-field order of MLXTrainingConfig on
unslothai/unsloth-zoo main (commit 4d140e03), captured before this PR's fields
were added. append_eos IS present upstream -- it predates this PR -- so it is
deliberately NOT treated as new here.
"""

from __future__ import annotations

import dataclasses
import sys

import pytest


UPSTREAM_FIELDS = [
    'per_device_train_batch_size', 'gradient_accumulation_steps', 'max_steps',
    'num_train_epochs', 'warmup_steps', 'warmup_ratio', 'learning_rate',
    'lr_scheduler_type', 'optim', 'weight_decay', 'adam_beta1', 'adam_beta2',
    'max_grad_norm', 'max_grad_value', 'max_grad_leaf_norm', 'seed',
    'lora_plus_ratio', 'embedding_learning_rate', 'logging_steps',
    'output_dir', 'report_to', 'save_steps', 'save_total_limit', 'eval_steps',
    'load_best_model_at_end', 'metric_for_best_model', 'greater_is_better',
    'early_stopping_patience', 'neftune_noise_alpha', 'dataset_text_field',
    'max_seq_length', 'packing', 'dataset_num_proc', 'chat_template',
    'use_cce', 'compile', 'compile_mode', 'compile_max_variants',
    'compile_arch_overrides', 'compile_backend_overrides', 'patch_mode',
    'compile_auto_tune', 'compile_trace', 'gradient_checkpointing',
    'streaming', 'dataset_order', 'preserve_dataset_order', 'memory_limit_gb',
    'cache_limit_gb', 'wired_limit_gb', 'disable_memory_limits',
    'cast_norm_output_to_input_dtype', 'append_eos', 'train_on_completions',
    'completion_only_loss', 'assistant_only_loss', 'assistant_token_id',
    'vlm_chat_template', 'per_device_eval_batch_size', 'image_size',
    'label_smoothing_factor', 'report_grad_norm', 'max_eval_batches',
    'streaming_text_length_window_batches', 'streaming_prefetch_batches',
    'logging_dir', 'run_name'
]

PREFERENCE_FIELDS = ["loss_type", "orpo_beta", "dpo_beta", "reference_free"]


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    shim_prefixes = ("mlx", "mlx_lm", "mlx_vlm")
    real_mlx_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in shim_prefixes)
    }
    from mlx_simulation import simulate_mlx_on_torch
    from mlx_simulation.mlx_stub import _MLXFinder
    simulate_mlx_on_torch()
    for name in list(sys.modules):
        if name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx."):
            sys.modules.pop(name, None)
    yield
    for name in list(sys.modules):
        if (
            name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx.")
            or any(name == prefix or name.startswith(f"{prefix}.") for prefix in shim_prefixes)
        ):
            sys.modules.pop(name, None)
    sys.meta_path[:] = [
        finder for finder in sys.meta_path
        if not isinstance(finder, _MLXFinder)
    ]
    sys.modules.update(real_mlx_modules)


def _init_fields():
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    return [f.name for f in dataclasses.fields(MLXTrainingConfig) if f.init]


def _resolved_warmup(config, total_steps=100):
    from unsloth_zoo.mlx.trainer import MLXTrainer

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = config
    return trainer._resolve_warmup_steps(total_steps)


# --------------------------------------------------------------------------
# Field layout: the four are a suffix, everything before keeps upstream slots.
# --------------------------------------------------------------------------

def test_exactly_these_four_fields_are_new():
    """append_eos predates this PR; loss_type does not. Pin both directions."""
    current = _init_fields()
    assert [f for f in current if f not in UPSTREAM_FIELDS] == PREFERENCE_FIELDS
    assert "append_eos" in UPSTREAM_FIELDS, "append_eos is not a new field"
    assert "loss_type" not in UPSTREAM_FIELDS, "loss_type IS a new field"


def test_upstream_fields_keep_their_positional_slots():
    """The reviewer's positional requirement: the first 67 slots are unchanged."""
    assert _init_fields()[:len(UPSTREAM_FIELDS)] == UPSTREAM_FIELDS


def test_preference_fields_are_declared_last():
    assert _init_fields()[len(UPSTREAM_FIELDS):] == PREFERENCE_FIELDS


def test_optional_copy_fields_remain_an_exact_suffix():
    """The tuple documents itself as an exact suffix of the field order."""
    from unsloth_zoo.mlx.trainer import _MLX_CONFIG_OPTIONAL_COPY_FIELDS

    tail = _init_fields()[-len(_MLX_CONFIG_OPTIONAL_COPY_FIELDS):]
    assert tail == list(_MLX_CONFIG_OPTIONAL_COPY_FIELDS)
    for name in PREFERENCE_FIELDS:
        assert name in _MLX_CONFIG_OPTIONAL_COPY_FIELDS


# --------------------------------------------------------------------------
# Both constructions the reviewer asked for.
# --------------------------------------------------------------------------

def _legacy_keyword_dump():
    """Exactly upstream's field set, at upstream defaults."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    dump = {name: getattr(MLXTrainingConfig, name) for name in UPSTREAM_FIELDS}
    dump["warmup_steps"] = MLXTrainingConfig.warmup_steps   # the default, 5
    dump["warmup_ratio"] = 0.1                              # non-default
    return dump


def test_legacy_keyword_dump_keeps_the_ratio_schedule():
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    config = MLXTrainingConfig(**_legacy_keyword_dump())
    assert config._unsloth_mlx_warmup_steps_explicit is False
    assert _resolved_warmup(config) == 10                   # ceil(0.1 * 100)


def test_legacy_positional_construction_keeps_the_ratio_schedule():
    """A pre-PR caller passing upstream's 67 values by position."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    values = [getattr(MLXTrainingConfig, name) for name in UPSTREAM_FIELDS]
    values[UPSTREAM_FIELDS.index("warmup_ratio")] = 0.1
    config = MLXTrainingConfig(*values)
    assert config._unsloth_mlx_warmup_steps_explicit is False
    assert _resolved_warmup(config) == 10


def test_legacy_positional_values_land_on_the_right_fields():
    """Positional binding must still map value -> field one-to-one."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    values = [getattr(MLXTrainingConfig, name) for name in UPSTREAM_FIELDS]
    values[UPSTREAM_FIELDS.index("image_size")] = (128, 256)
    values[UPSTREAM_FIELDS.index("output_dir")] = "/tmp/legacy-check"
    config = MLXTrainingConfig(*values)
    assert config.image_size == (128, 256)
    assert config.output_dir == "/tmp/legacy-check"
    # The appended fields were not supplied, so they hold their defaults.
    assert config.loss_type == "sft"
    assert config.reference_free is False


def test_explicit_non_default_warmup_steps_still_wins():
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    dump = _legacy_keyword_dump()
    dump["warmup_steps"] = 7                                # NOT the default
    config = MLXTrainingConfig(**dump)
    assert config._unsloth_mlx_warmup_steps_explicit is True
    assert _resolved_warmup(config) == 7


def test_dump_including_the_new_fields_is_unchanged():
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    dump = _legacy_keyword_dump()
    for name in PREFERENCE_FIELDS:
        dump[name] = getattr(MLXTrainingConfig, name)
    config = MLXTrainingConfig(**dump)
    assert config._unsloth_mlx_warmup_steps_explicit is False
    assert _resolved_warmup(config) == 10
