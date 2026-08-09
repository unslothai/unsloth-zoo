# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
"""The grad-accumulation loss-normalisation contract in _unsloth_get_batch_samples.

We decide num_items_in_batch from the forward signature, but training_step divides
by grad-accum from self.model_accepts_loss_kwargs. Returning a count while that
flag is False normalises twice, since TRL's chunked_nll and our own fused CE both
divide by the count, so loss and gradients end up scaled by 1/GA.

The guard mirrors stock Trainer._get_num_items_in_batch: only count when the flag
is True or a compute_loss_func exists. Every loss that divides by the count falls
back to a mean when it is None, so training_step's /GA is then the single correct
normalisation, and no per-model or per-trainer knowledge is needed.

CPU only, no model downloads.
"""

import inspect
import re

import pytest


def _loss_utils():
    return pytest.importorskip("unsloth_zoo.loss_utils")


def _source():
    mod = _loss_utils()
    fn = getattr(mod, "_unsloth_get_batch_samples", None)
    if fn is None: pytest.skip("_unsloth_get_batch_samples not present")
    return inspect.getsource(fn)


def test_guard_is_present_and_mirrors_the_stock_predicate():
    src = _source()
    assert "model_accepts_loss_kwargs" in src, (
        "the grad-accum guard is gone; a count returned while the flag is False "
        "gets normalised twice and silently scales gradients by 1/GA"
    )
    assert "compute_loss_func" in src, (
        "the guard must exempt compute_loss_func, which suppresses the division "
        "upstream exactly as stock transformers does"
    )


def test_guard_only_suppresses_never_flips_the_flag():
    """Flipping the flag True would break the trainers that disable it on purpose.

    Thirteen TRL trainers set model_accepts_loss_kwargs = False in __init__ to
    enable the grad-accum scaling (DPO, KTO, GRPO, RLOO, Reward, CPO, ORPO, BCO,
    Distillation, SDPO, SSD, SDFT, AsyncGRPO). Suppressing the count is safe for
    all of them; assigning the flag is not.
    """
    src = _source()
    assert not re.search(r"\.model_accepts_loss_kwargs\s*=", src), (
        "the guard must not assign model_accepts_loss_kwargs. Suppressing the "
        "count is enough, and assigning it would override trainers that set it "
        "False deliberately, plus the DeepSpeed sequence-parallel opt-out that "
        "transformers sets in Trainer.__init__"
    )


def test_stock_trainer_gates_its_count_on_the_same_predicate():
    """If upstream changes this, the guard is mirroring the wrong condition."""
    from transformers import Trainer

    fn = getattr(Trainer, "_get_num_items_in_batch", None)
    if fn is None: pytest.skip("no _get_num_items_in_batch on this transformers")
    try: src = inspect.getsource(fn)
    except (OSError, TypeError): pytest.skip("source unavailable")
    assert "model_accepts_loss_kwargs" in src and "compute_loss_func" in src, (
        "stock transformers no longer gates its token count on these, so the "
        "guard in _unsloth_get_batch_samples is mirroring a stale predicate"
    )


def test_training_step_still_divides_on_the_same_flag():
    from transformers import Trainer

    try: src = inspect.getsource(Trainer.training_step)
    except (OSError, TypeError): pytest.skip("source unavailable")
    assert "model_accepts_loss_kwargs" in src
    assert "num_items_in_batch" in src


def test_losses_fall_back_to_a_mean_when_the_count_is_none():
    """The guard is only safe because every such loss handles None as a mean."""
    ce = pytest.importorskip("unsloth_zoo.fused_losses.cross_entropy_loss")
    fn = getattr(ce, "unsloth_fused_ce_loss", None)
    if fn is None: pytest.skip("unsloth_fused_ce_loss not present")
    doc = (fn.__doc__ or "") + inspect.getsource(fn)
    assert "n_items" in doc, (
        "unsloth_fused_ce_loss no longer takes n_items; re-check the guard"
    )

    trl_sft = pytest.importorskip("trl.trainer.sft_trainer")
    chunked = getattr(trl_sft, "_chunked_cross_entropy_loss", None)
    if chunked is None: pytest.skip("this TRL has no chunked CE")
    src = inspect.getsource(chunked)
    assert "num_items_in_batch is None" in src, (
        "TRL's chunked CE no longer branches on a None count. The guard relies on "
        "it falling back to a mean; re-measure GA-invariance before trusting it."
    )


def test_get_batch_samples_still_returns_the_documented_pair():
    src = _source().strip()
    assert src.endswith("return batch_samples, num_items_in_batch"), (
        "unsloth/models/_utils.py asserts on this exact shape at patch time"
    )
