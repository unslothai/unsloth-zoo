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

"""Regression tests: no preference eval request may be silently dropped.

ORPO/DPO have no eval path. _prepare_eval_batches returns None for them and
_run_eval clears control.should_evaluate and returns False, so every request
degrades to nothing. The warning was gated on ``args.eval_steps > 0``, which
covers only the step cadence -- eval can also be requested by:

  * eval_strategy="epoch"                    (_request_epoch_cadence_actions)
  * a callback setting control.should_evaluate
  * DefaultFlowCallback's final-step eval
  * load_best_model_at_end                   (needs a metric that never arrives)
  * early_stopping_patience > 0              (same)

The first three now warn; the last two raise, because unlike the cadences they
cannot degrade gracefully -- load_best_model_at_end would report success while
restoring nothing, and early stopping would never trigger.
"""

from __future__ import annotations

import sys

import pytest


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


ROWS = [{"prompt": f"q{i} ", "chosen": "good", "rejected": "bad"} for i in range(4)]
EVAL_ROWS = [{"prompt": "q ", "chosen": "good", "rejected": "bad"}]


class _Tok:
    eos_token_id = 7
    bos_token = None

    def encode(self, text, add_special_tokens=True):
        return [(ord(ch) % 90) + 8 for ch in text]


class _Model:
    _config = {"model_type": "llama"}

    def trainable_parameters(self):
        return {}

    def parameters(self):
        return {}

    def named_modules(self):
        return []

    def train(self, *args, **kwargs):
        return self

    def eval(self, *args, **kwargs):
        return self


def _config(loss, **kwargs):
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXORPOConfig

    cls = MLXORPOConfig if loss == "orpo" else MLXDPOConfig
    return cls(per_device_train_batch_size=2, max_seq_length=64,
               max_steps=2, **kwargs)


def _trainer(loss, **kwargs):
    from unsloth_zoo.mlx.trainer import MLXDPOTrainer, MLXORPOTrainer

    cls = MLXORPOTrainer if loss == "orpo" else MLXDPOTrainer
    return cls(model=_Model(), tokenizer=_Tok(), train_dataset=ROWS,
               eval_dataset=EVAL_ROWS, args=_config(loss, **kwargs))


# --------------------------------------------------------------------------
# Paths 1-4: cadence / callback requests must WARN, never pass silently.
# --------------------------------------------------------------------------

WARNING = "eval is not yet supported"


def test_warning_is_not_gated_on_the_step_cadence():
    """The reported defect: the warning only fired for eval_steps > 0.

    The warning lives inside _train_inner, which needs a real model to reach,
    so pin its condition at the source level: it must key off eval_dataset
    alone, not the step interval.
    """
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    idx = src.index(WARNING)
    condition = src[max(0, idx - 400):idx]
    guard = condition[condition.rindex("if ("):]
    assert "self.eval_dataset is not None" in guard
    assert "eval_steps" not in guard, (
        "the warning must not be gated on the step cadence"
    )


@pytest.mark.parametrize("setter,label", [
    ("_request_epoch_cadence_actions", 'eval_strategy="epoch"'),
    ("_request_step_cadence_actions", "DefaultFlowCallback final-step eval"),
])
def test_cadence_paths_really_do_request_eval(setter, label):
    """These are live request paths, so the warning above must cover them."""
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(getattr(MLXTrainer, setter))
    assert "should_evaluate = True" in src, f"{label} no longer requests eval"


def test_callback_requested_eval_reaches_the_same_dead_end():
    """A callback raising should_evaluate is honored by the loop."""
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    assert "or self.control.should_evaluate" in src
    # ... and _run_eval clears it when no batches exist.
    assert "self.control.should_evaluate = False" in src


# --------------------------------------------------------------------------
# Paths 5-6: features that cannot degrade must RAISE.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("loss", ["orpo", "dpo"])
def test_load_best_model_at_end_is_rejected(loss):
    trainer = _trainer(loss, load_best_model_at_end=True)
    with pytest.raises(ValueError, match="load_best_model_at_end is not supported"):
        trainer._prepare_data(False)


@pytest.mark.parametrize("loss", ["orpo", "dpo"])
def test_early_stopping_patience_is_rejected(loss):
    trainer = _trainer(loss, early_stopping_patience=2)
    with pytest.raises(ValueError, match="early_stopping_patience is not supported"):
        trainer._prepare_data(False)


@pytest.mark.parametrize("loss", ["orpo", "dpo"])
def test_defaults_are_not_rejected(loss):
    """Neither feature is on by default, so a plain run must not raise."""
    trainer = _trainer(loss)
    trainer._prepare_data(False)


def test_sft_is_unaffected_by_either_rejection():
    """The guards must be scoped to preference losses only."""
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    trainer = MLXTrainer(
        model=_Model(), tokenizer=_Tok(),
        train_dataset=[{"text": "hello world this is a row"}],
        args=MLXTrainingConfig(
            per_device_train_batch_size=2, max_seq_length=64, max_steps=2,
            load_best_model_at_end=True, early_stopping_patience=2,
        ),
    )
    try:
        trainer._prepare_data(False)
    except ValueError as exc:
        assert "not supported for" not in str(exc), (
            "the preference guards must not fire on SFT"
        )
    except Exception:
        pass  # stub model cannot finish batching; only the guards matter here
