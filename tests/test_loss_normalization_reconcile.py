# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
"""Tests for _reconcile_loss_normalization in loss_utils.

We decide num_items_in_batch from the forward signature; training_step decides
whether to divide by grad-accum from self.model_accepts_loss_kwargs. When a loss
divides by num_items_in_batch (ours, and TRL's chunked_nll) and the model class
sets accepts_loss_kwargs = False, both fire and the gradients are scaled 1/GA.

The reconcile keys off whether the ACTIVE loss consumes the count, not off where
the False came from. Three cases, pinned here:
  * chunked_nll divides by the count itself -> flip the flag, keep the count.
  * plain nll does not -> drop the count so training_step's /GA still runs.
  * trainer asked for the divide (DPO, GRPO, KTO) -> drop the count, keep flag.

The second case is a regression guard: an earlier revision keyed off the origin
of the False and left gradients 4x too large under nll at GA=4.

CPU only, no model downloads.
"""

import importlib
import re

import pytest


def _fn(name):
    mod = pytest.importorskip("unsloth_zoo.loss_utils")
    fn = getattr(mod, name, None)
    if fn is None: pytest.skip(f"{name} not present")
    return fn


class _Args:
    def __init__(self, loss_type="chunked_nll", use_liger_kernel=False):
        self.loss_type = loss_type
        self.use_liger_kernel = use_liger_kernel


class _ModelDerived:
    """Trainer that never assigns the flag itself."""
    compute_loss_func = None
    def __init__(self, accepts, loss_type="chunked_nll"):
        self.model_accepts_loss_kwargs = accepts
        self.args = _Args(loss_type)


class _Deliberate(_ModelDerived):
    def __init__(self, accepts, loss_type="chunked_nll"):
        super().__init__(accepts, loss_type)
        self.model_accepts_loss_kwargs = False


def test_flag_true_is_untouched():
    reconcile = _fn("_reconcile_loss_normalization")
    t = _ModelDerived(True)
    assert reconcile(t, 100) == 100
    assert t.model_accepts_loss_kwargs is True


def test_chunked_nll_flips_flag_and_keeps_count():
    """chunked_nll divides by the count itself, so training_step must not."""
    reconcile = _fn("_reconcile_loss_normalization")
    t = _ModelDerived(False, loss_type="chunked_nll")
    assert reconcile(t, 100) == 100
    assert t.model_accepts_loss_kwargs is True


def test_plain_nll_suppresses_count_and_leaves_flag():
    """Regression for the 4x-too-large gradients this file originally shipped.

    Under loss_type="nll" nothing downstream consumes num_items_in_batch, so
    flipping the flag would suppress the only normalisation there is and leave
    gradients GA times too large. Measured 4.166698 vs 1.041674 at GA=4.
    """
    reconcile = _fn("_reconcile_loss_normalization")
    t = _ModelDerived(False, loss_type="nll")
    assert reconcile(t, 100) is None
    assert t.model_accepts_loss_kwargs is False


def test_liger_kernel_suppresses_count():
    """use_liger_kernel forces loss_type="nll" regardless of the setting."""
    reconcile = _fn("_reconcile_loss_normalization")
    t = _ModelDerived(False, loss_type="chunked_nll")
    t.args.use_liger_kernel = True
    assert reconcile(t, 100) is None
    assert t.model_accepts_loss_kwargs is False


def test_unset_loss_type_is_treated_as_chunked_nll():
    """TRL leaves loss_type None until __post_init__ resolves it."""
    reconcile = _fn("_reconcile_loss_normalization")
    t = _ModelDerived(False, loss_type=None)
    assert reconcile(t, 100) == 100
    assert t.model_accepts_loss_kwargs is True


def test_deliberate_false_drops_count_and_keeps_flag():
    reconcile = _fn("_reconcile_loss_normalization")
    t = _Deliberate(False)
    assert reconcile(t, 100) is None
    assert t.model_accepts_loss_kwargs is False


def test_none_count_and_compute_loss_func_are_untouched():
    reconcile = _fn("_reconcile_loss_normalization")
    t = _ModelDerived(False)
    assert reconcile(t, None) is None
    assert t.model_accepts_loss_kwargs is False

    t = _ModelDerived(False)
    t.compute_loss_func = lambda *a, **k: None
    assert reconcile(t, 100) == 100
    assert t.model_accepts_loss_kwargs is False


# Every TRL trainer that sets the flag False on purpose. Each must be detected,
# otherwise this change would suppress their intended grad-accum scaling.
TRL_DELIBERATE = [
    ("trl.trainer.dpo_trainer", "DPOTrainer"),
    ("trl.trainer.kto_trainer", "KTOTrainer"),
    ("trl.trainer.grpo_trainer", "GRPOTrainer"),
    ("trl.trainer.rloo_trainer", "RLOOTrainer"),
    ("trl.trainer.reward_trainer", "RewardTrainer"),
    ("trl.experimental.cpo.cpo_trainer", "CPOTrainer"),
    ("trl.experimental.orpo.orpo_trainer", "ORPOTrainer"),
    ("trl.experimental.bco.bco_trainer", "BCOTrainer"),
    ("trl.experimental.sdpo.sdpo_trainer", "SDPOTrainer"),
    ("trl.experimental.ssd.ssd_trainer", "SSDTrainer"),
    ("trl.experimental.sdft.sdft_trainer", "SDFTTrainer"),
    ("trl.experimental.distillation.distillation_trainer", "DistillationTrainer"),
    ("trl.experimental.async_grpo.async_grpo_trainer", "AsyncGRPOTrainer"),
]


@pytest.mark.parametrize("module_name,cls_name", TRL_DELIBERATE)
def test_trl_trainers_that_opt_out_are_detected(module_name, cls_name):
    detect = _fn("_trainer_explicitly_disables_loss_kwargs")
    try: mod = importlib.import_module(module_name)
    except Exception: pytest.skip(f"{module_name} not importable")
    cls = getattr(mod, cls_name, None)
    if cls is None: pytest.skip(f"{cls_name} absent in this TRL")
    assert detect(cls.__new__(cls)) is True, f"{cls_name} would lose its /GA scaling"


def test_sft_and_base_trainer_are_not_flagged_as_deliberate():
    detect = _fn("_trainer_explicitly_disables_loss_kwargs")
    from transformers import Trainer
    assert detect(Trainer.__new__(Trainer)) is False
    trl = pytest.importorskip("trl")
    if hasattr(trl, "SFTTrainer"):
        assert detect(trl.SFTTrainer.__new__(trl.SFTTrainer)) is False


def test_detection_is_cached_per_class():
    mod = pytest.importorskip("unsloth_zoo.loss_utils")
    detect = _fn("_trainer_explicitly_disables_loss_kwargs")
    detect(_Deliberate(False))
    assert _Deliberate in mod._TRAINER_DISABLES_CACHE


def test_regex_tolerates_spacing_variants():
    mod = pytest.importorskip("unsloth_zoo.loss_utils")
    rx = getattr(mod, "_ASSIGNS_ACCEPTS_FALSE", None)
    if rx is None: pytest.skip("_ASSIGNS_ACCEPTS_FALSE not present")
    for variant in (
        "self.model_accepts_loss_kwargs = False",
        "self.model_accepts_loss_kwargs=False",
        "self . model_accepts_loss_kwargs  =  False",
    ):
        assert rx.search(variant), variant
    assert not rx.search("self.model_accepts_loss_kwargs = True")


def test_training_step_predicate_still_exists_upstream():
    """If transformers stops keying off this flag, the reconcile is stale."""
    import inspect

    from transformers import Trainer
    try: source = inspect.getsource(Trainer.training_step)
    except (OSError, TypeError): pytest.skip("source unavailable")
    assert "model_accepts_loss_kwargs" in source
    assert "num_items_in_batch" in source
