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

"""trl's SFTTrainer touches outputs.logits TWICE, and rebinding entropy_from_logits
only covers the first. On trl 1.x the second one runs FIRST, so the rebind is inert.

    trl 0.22.2  sft_trainer.py:1080  shift_logits = outputs.logits[..., :-1, :]
    trl 0.24.0  sft_trainer.py:1146  shift_logits = outputs.logits[..., :-1, :]
    trl 0.25.1  sft_trainer.py:1151  shift_logits = outputs.logits[..., :-1, :]
    trl 1.9.2   sft_trainer.py:1769  shift_logits = outputs.logits[..., :-1, :]

Two shapes are exercised here. The installed trl's REAL compute_loss (no stubbing
of trl) against both real sentinels, and a stand-in for trl 1.x -- whose liger
branch runs BEFORE the forward and injects forward kwargs, so "pretend liger is
on" is only safe there if those kwargs are kept out of the model call.
"""
import collections
import sys
import warnings
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOGITS_ERROR_STRING = "Unsloth: Logits are empty from 2024.11 onwards."


def _raise(*a, **k): raise NotImplementedError(LOGITS_ERROR_STRING)
def _none(*a, **k): return None


class EmptyLogits:
    """The compiler.py sentinel. The fused-loss path returns torch.empty(0)
    instead, so both are parametrised wherever the sentinel is the input."""

    def raise_getattr_error(self, attr): return _none if attr == "to" else _raise
    __getitem__ = _raise
    __getattr__ = raise_getattr_error
    def __repr__(self): return LOGITS_ERROR_STRING
    def __eq__(self, other): return type(other).__name__ == "EmptyLogits"
    __hash__ = object.__hash__


SENTINELS = [pytest.param(EmptyLogits, id = "EmptyLogits"),
             pytest.param(lambda: torch.empty(0), id = "empty-tensor")]


class _Acc:
    def gather_for_metrics(self, t): return t if torch.is_tensor(t) else torch.tensor(t)


class _Args:
    use_liger_kernel = False
    loss_type = "nll"
    average_tokens_across_devices = False
    prediction_loss_only = False


class _Model:
    training = True
    active_adapter = "default"
    peft_config = {}


@pytest.fixture(autouse = True)
def _pin_the_accelerate_device(monkeypatch):
    """`accelerate.PartialState` is a process-global borg. Anything earlier in
    the session that builds an Accelerator (many tests here do) leaves
    `_shared_state["device"] = cuda`, and the entropy fallback then allocates
    its scalar there, where it meets the CPU tensors below and raises "Expected
    all tensors to be on the same device". Pinned so the result of this module
    does not depend on what ran before it in the same process.
    """
    state = pytest.importorskip("accelerate.state")
    monkeypatch.setitem(state.PartialState._shared_state, "device", torch.device("cpu"))


@pytest.fixture(scope = "module")
def sft():
    pytest.importorskip("trl")
    sftmod = pytest.importorskip("trl.trainer.sft_trainer")
    if not hasattr(sftmod, "SFTTrainer"):
        pytest.skip("this trl has no SFTTrainer")
    from unsloth_zoo.temporary_patches.misc import (
        patch_trl_entropy_from_logits, patch_trl_sft_logits_metrics,
    )
    patch_trl_entropy_from_logits()
    patch_trl_sft_logits_metrics()
    return sftmod


def _trainer(sftmod = None):
    self = object.__new__(sftmod.SFTTrainer) if sftmod is not None else \
        type("FakeTrainer", (), {})()
    self.args = _Args()
    self.model = _Model()
    self.accelerator = _Acc()
    self.num_virtual_tokens = 0
    self.compute_metrics = None
    self.preprocess_logits_for_metrics = None
    self._metrics = {"train": collections.defaultdict(list),
                     "eval": collections.defaultdict(list)}
    self._total_train_tokens = 0
    self.aux_loss_enabled = False
    self.compute_loss_func = None
    return self


def _inputs(B = 2, L = 6, V = 11):
    return {
        "input_ids": torch.randint(0, V, (B, L)),
        "attention_mask": torch.ones(B, L, dtype = torch.long),
        "labels": torch.randint(0, V, (B, L)),
    }


def _run_steps(sftmod, logits, steps = 3, pop_labels = False):
    """Drive trl's real compute_loss with a Trainer.compute_loss that returns
    `logits`, and report the metrics it accumulated."""
    from transformers import Trainer
    Out = collections.namedtuple("Out", "loss logits")
    loss = torch.tensor(1.0)
    calls = [0]

    def fake(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        calls[0] += 1
        # transformers pops labels out of the inputs whenever `compute_loss_func`
        # or label smoothing is set, so a retry that reuses the same mapping
        # would run a forward with no labels and no loss.
        if pop_labels: inputs.pop("labels")
        assert "labels" in inputs or pop_labels
        out = Out(loss, logits)
        return (loss, out) if return_outputs else loss

    original = Trainer.compute_loss
    Trainer.compute_loss = fake
    try:
        self = _trainer(sftmod)
        inputs = _inputs()
        for _ in range(steps):
            sftmod.SFTTrainer.compute_loss(self, self.model, dict(inputs),
                                           num_items_in_batch = None)
        return self, calls[0]
    finally:
        Trainer.compute_loss = original


# ---- against the installed trl -------------------------------------------

@pytest.mark.parametrize("sentinel", SENTINELS)
def test_a_run_with_no_logits_completes(sft, sentinel):
    """The whole point. Before this patch the step aborted -- at the entropy
    line on trl <= 0.25 and at the accuracy line on trl 1.x."""
    self, _ = _run_steps(sft, sentinel())
    assert self._metrics["train"]["num_tokens"] == [36.0]


@pytest.mark.parametrize("sentinel", SENTINELS)
def test_the_token_counter_is_not_double_counted_by_the_retry(sft, sentinel):
    """The failing call already advanced _total_train_tokens; a naive retry
    would count the first step twice."""
    self, _ = _run_steps(sft, sentinel(), steps = 3)
    assert self._total_train_tokens == 36


@pytest.mark.parametrize("sentinel", SENTINELS)
def test_only_the_first_step_pays_for_the_detection(sft, sentinel):
    _, calls = _run_steps(sft, sentinel(), steps = 3)
    assert calls == 4, calls


def test_the_retry_still_has_the_labels_the_first_attempt_popped(sft):
    """transformers `Trainer.compute_loss` pops `labels` out of the inputs when
    a custom loss function or label smoothing is in play. Replaying the same
    mapping into the retry would hand the model a batch with no labels."""
    self, _ = _run_steps(sft, EmptyLogits(), steps = 2, pop_labels = True)
    assert self._metrics["train"]["num_tokens"] == [24.0]


def test_real_logits_keep_their_metrics(sft):
    """A non-Unsloth trl user must not lose entropy or mean_token_accuracy."""
    self, calls = _run_steps(sft, torch.randn(2, 6, 11), steps = 3)
    assert calls == 3
    assert len(self._metrics["train"]["mean_token_accuracy"]) == 3
    assert len(self._metrics["train"]["entropy"]) == 3
    assert all(e != 0.0 for e in self._metrics["train"]["entropy"])


def test_real_logits_do_not_touch_use_liger_kernel(sft):
    self, _ = _run_steps(sft, torch.randn(2, 6, 11))
    assert self.args.use_liger_kernel is False
    assert not getattr(self, "_unsloth_logits_are_empty", False)


def test_use_liger_kernel_is_restored(sft):
    self, _ = _run_steps(sft, EmptyLogits())
    assert self.args.use_liger_kernel is False


def test_a_real_error_is_not_swallowed(sft):
    from transformers import Trainer
    original = Trainer.compute_loss

    def boom(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        raise TypeError("something else entirely")

    Trainer.compute_loss = boom
    try:
        with pytest.raises(TypeError, match = "something else entirely"):
            sft.SFTTrainer.compute_loss(_trainer(sft), _Model(), _inputs(),
                                        num_items_in_batch = None)
    finally:
        Trainer.compute_loss = original


def test_a_failure_with_real_logits_is_not_swallowed(sft):
    """The discriminator is the logits trl was holding, not the exception. A
    genuine bug in the metric block, with real logits, must still raise."""
    from transformers import Trainer
    Out = collections.namedtuple("Out", "loss logits")
    original = Trainer.compute_loss

    def bad_shape(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        out = Out(torch.tensor(1.0), torch.randn(2, 6, 11))
        return (torch.tensor(1.0), out) if return_outputs else torch.tensor(1.0)

    Trainer.compute_loss = bad_shape
    try:
        inputs = _inputs()
        inputs["labels"] = torch.randint(0, 11, (2, 99))  # will not line up
        with pytest.raises(Exception):
            sft.SFTTrainer.compute_loss(_trainer(sft), _Model(), inputs,
                                        num_items_in_batch = None)
    finally:
        Trainer.compute_loss = original


def test_applying_it_twice_does_not_stack(sft):
    from unsloth_zoo.temporary_patches.misc import patch_trl_sft_logits_metrics
    once = sft.SFTTrainer.compute_loss
    patch_trl_sft_logits_metrics()
    assert sft.SFTTrainer.compute_loss is once


def test_it_is_registered(sft):
    from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
    assert any(getattr(f, "__name__", "") == "patch_trl_sft_logits_metrics"
               for f in TEMPORARY_PATCHES)


# ---- against a stand-in for trl 1.x --------------------------------------
#
# trl 1.x is not the version installed here, and its compute_loss differs in the
# one way that decides whether the off switch is usable at all: the liger branch
# runs BEFORE the forward and writes liger-only kwargs into the batch.

FORWARD_KWARGS_SEEN = []


def _trl_1x_compute_loss(self, model, inputs, return_outputs = False,
                         num_items_in_batch = None):
    """trl 1.9.2 sft_trainer.py:1700-1832, trimmed to the parts that matter."""
    from transformers import Trainer
    mode = "train" if self.model.training else "eval"
    prediction_loss_only = inputs.pop("_prediction_loss_only", None)
    labels = inputs["labels"]
    inputs["use_cache"] = False
    if self.args.use_liger_kernel:
        inputs["skip_logits"] = (
            self.model.training or self.args.prediction_loss_only
            or (self.compute_metrics is None
                and self.preprocess_logits_for_metrics is None
                and prediction_loss_only is not False)
        )
        inputs["return_token_accuracy"] = True
        inputs["use_token_scaling"] = self.args.loss_type == "dft"
    (loss, outputs) = Trainer.compute_loss(
        self, model, inputs, return_outputs = True,
        num_items_in_batch = num_items_in_batch,
    )
    if not self.args.use_liger_kernel:
        with torch.no_grad():
            shift_logits = outputs.logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            mask = shift_labels != -100
            predictions = shift_logits.argmax(dim = -1)
            correct = ((predictions == shift_labels) & mask).sum()
            total = mask.sum()
        self._metrics[mode]["entropy"].append(0.0)
        self._metrics[mode]["mean_token_accuracy"].append((correct / total).item())
    if mode == "train":
        self._total_train_tokens += inputs["attention_mask"].sum().item()
    self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
    if self.args.use_liger_kernel:
        if getattr(outputs, "token_accuracy", None) is not None:
            self._metrics[mode]["mean_token_accuracy"].append(outputs.token_accuracy)
        else:
            warnings.warn(
                "liger-kernel did not return token_accuracy when requested. The "
                "mean_token_accuracy metric will not be logged. This is "
                "unexpected; please report it to the liger-kernel repository.",
                stacklevel = 2,
            )
    return (loss, outputs) if return_outputs else loss


def _run_1x(logits, steps = 2):
    from transformers import Trainer
    from unsloth_zoo.temporary_patches.misc import _sft_wrap_compute_loss
    Out = collections.namedtuple("Out", "loss logits")
    FORWARD_KWARGS_SEEN.clear()

    def fake(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        FORWARD_KWARGS_SEEN.append(sorted(inputs.keys()))
        loss = torch.tensor(1.0)
        out = Out(loss, logits)
        return (loss, out) if return_outputs else loss

    wrapped = _sft_wrap_compute_loss(_trl_1x_compute_loss)
    original = Trainer.compute_loss
    Trainer.compute_loss = fake
    try:
        self = _trainer()
        for _ in range(steps):
            wrapped(self, self.model, _inputs(), num_items_in_batch = None)
        return self
    finally:
        Trainer.compute_loss = original


@pytest.mark.parametrize("sentinel", SENTINELS)
def test_the_1x_shape_completes_too(sentinel):
    self = _run_1x(sentinel())
    assert self._total_train_tokens == 24
    assert self._metrics["train"]["mean_token_accuracy"] == []


@pytest.mark.parametrize("sentinel", SENTINELS)
def test_liger_only_kwargs_never_reach_the_forward(sentinel):
    """`skip_logits` / `return_token_accuracy` / `use_token_scaling` are for a
    liger-patched forward. The model here is an Unsloth one, and it would be
    handed them on the retry and on every step after it."""
    _run_1x(sentinel())
    assert FORWARD_KWARGS_SEEN, "the forward was never called"
    for seen in FORWARD_KWARGS_SEEN:
        assert "skip_logits" not in seen, seen
        assert "return_token_accuracy" not in seen, seen
        assert "use_token_scaling" not in seen, seen


def test_the_1x_liger_kernel_bug_report_is_not_shown(recwarn):
    """With the flag forced on, trl 1.x asks the user to report a bug to
    liger-kernel for not returning token_accuracy. There is no liger here."""
    warnings.simplefilter("always")
    _run_1x(EmptyLogits())
    assert not [w for w in recwarn if "liger-kernel" in str(w.message)]


def test_the_1x_shape_keeps_real_metrics():
    self = _run_1x(torch.randn(2, 6, 11))
    assert len(self._metrics["train"]["mean_token_accuracy"]) == 2
    for seen in FORWARD_KWARGS_SEEN:
        assert "skip_logits" not in seen


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
