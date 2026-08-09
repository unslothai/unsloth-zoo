# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
"""Tests for _reconcile_loss_normalization in loss_utils.

We decide num_items_in_batch from the forward signature; training_step decides
whether to divide by grad-accum from self.model_accepts_loss_kwargs. When a loss
divides by num_items_in_batch (ours, and TRL's chunked_nll) and the model class
sets accepts_loss_kwargs = False, both fire and the gradients are scaled 1/GA.

Two cases need opposite treatment, so both are pinned here:
  * flag False because the model class says so -> flip the flag, keep the count.
  * flag False because the trainer asked for it (DPO, GRPO, KTO and friends)
    -> drop the count, leave the flag.

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


class _ModelDerived:
    """Trainer that never assigns the flag itself."""
    compute_loss_func = None
    def __init__(self, accepts): self.model_accepts_loss_kwargs = accepts


class _Deliberate(_ModelDerived):
    def __init__(self, accepts):
        super().__init__(accepts)
        self.model_accepts_loss_kwargs = False


def test_flag_true_is_untouched():
    reconcile = _fn("_reconcile_loss_normalization")
    t = _ModelDerived(True)
    assert reconcile(t, 100) == 100
    assert t.model_accepts_loss_kwargs is True


def test_model_derived_false_flips_flag_and_keeps_count():
    reconcile = _fn("_reconcile_loss_normalization")
    t = _ModelDerived(False)
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
