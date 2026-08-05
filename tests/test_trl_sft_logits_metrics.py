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


# ---- aux_loss is not logits-derived and must survive the skip --------------

def _run_with_aux(sftmod, logits, steps = 3, aux = torch.tensor(0.25),
                  trainer_logs_aux = False):
    """As `_run_steps`, but the outputs carry `aux_loss` and the trainer is a
    MoE one (`output_router_logits = True` -> `aux_loss_enabled`).

    `trainer_logs_aux` stands in for trl 1.x, which moved this metric OUT of
    the `use_liger_kernel` branch (1.9.2 sft_trainer.py:1826 "applies to both
    Liger and non-Liger") and therefore logs it itself.
    """
    from transformers import Trainer
    Out = collections.namedtuple("Out", "loss logits aux_loss")
    loss = torch.tensor(1.0)

    def fake(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        if trainer_logs_aux:
            self._metrics["train"]["aux_loss"].append(float(aux))
        out = Out(loss, logits, aux)
        return (loss, out) if return_outputs else loss

    original = Trainer.compute_loss
    Trainer.compute_loss = fake
    try:
        self = _trainer(sftmod)
        self.aux_loss_enabled = True
        inputs = _inputs()
        for _ in range(steps):
            sftmod.SFTTrainer.compute_loss(self, self.model, dict(inputs),
                                           num_items_in_batch = None)
        return self
    finally:
        Trainer.compute_loss = original


@pytest.mark.parametrize("sentinel", SENTINELS)
def test_aux_loss_survives_the_skipped_metric_block(sft, sentinel):
    """On trl 0.23.0-0.25.x this metric sits INSIDE the block being skipped, so
    without a replay a MoE run silently stops logging it."""
    self = _run_with_aux(sft, sentinel(), steps = 3)
    assert self._metrics["train"]["aux_loss"] == [0.25, 0.25, 0.25]


@pytest.mark.parametrize("sentinel", SENTINELS)
def test_aux_loss_is_not_logged_twice(sft, sentinel):
    """trl 1.x logs it outside the branch, so replaying would double-count.
    Gated on the count rather than on a version number."""
    self = _run_with_aux(sft, sentinel(), steps = 3, trainer_logs_aux = True)
    assert self._metrics["train"]["aux_loss"] == [0.25, 0.25, 0.25]


def test_aux_loss_is_left_alone_when_the_run_is_not_moe(sft):
    """`aux_loss_enabled` is False for every dense model, and nothing should
    appear for them."""
    self, _ = _run_steps(sft, EmptyLogits(), steps = 2)
    assert self._metrics["train"]["aux_loss"] == []


def test_the_ordinary_call_shape_is_unchanged_for_a_dense_run():
    """`return_outputs` is forced only when there is an aux_loss to rescue.

    Asked of the wrapper's own call into trl, not of trl's call into
    transformers: trl passes `return_outputs = True` inward on every version,
    since that is how it reaches `outputs.logits` at all.
    """
    from unsloth_zoo.temporary_patches.misc import _sft_call_without_logits_metrics
    seen = []

    def original(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        seen.append(return_outputs)
        return torch.tensor(1.0)

    self = _trainer()
    for aux_enabled, expected in ((False, False), (True, True)):
        seen.clear()
        self.aux_loss_enabled = aux_enabled
        _sft_call_without_logits_metrics(
            original, self, (self.model, _inputs()), {"num_items_in_batch": None})
        assert seen == [expected], (aux_enabled, seen)


def test_a_trainer_that_asked_for_outputs_still_gets_them():
    """Forcing the flag must not change what the caller receives."""
    from unsloth_zoo.temporary_patches.misc import _sft_call_without_logits_metrics
    Out = collections.namedtuple("Out", "loss logits aux_loss")
    pair = (torch.tensor(1.0), Out(torch.tensor(1.0), EmptyLogits(), torch.tensor(0.5)))

    def original(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        return pair if return_outputs else pair[0]

    self = _trainer()
    self.aux_loss_enabled = True
    result, outputs = _sft_call_without_logits_metrics(
        original, self, (self.model, _inputs(), True), {})
    assert result is pair, "the caller asked, so the tuple is handed back intact"
    # Still harvested for the replay: an eval step arrives already asking, and
    # those are the batches that would otherwise lose the metric.
    assert outputs is pair[1]


# ---- an output object with no logits at all is somebody else's bug ---------

def test_a_missing_logits_attribute_still_raises(sft):
    """`getattr(outputs, "logits", None)` would call an absent attribute
    unusable and retry with the metrics off, turning a broken output contract
    into silent training."""
    from transformers import Trainer
    Out = collections.namedtuple("Out", "loss")

    def fake(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        loss = torch.tensor(1.0)
        out = Out(loss)
        return (loss, out) if return_outputs else loss

    original = Trainer.compute_loss
    Trainer.compute_loss = fake
    try:
        self = _trainer(sft)
        with pytest.raises(AttributeError):
            sft.SFTTrainer.compute_loss(self, self.model, _inputs(),
                                        num_items_in_batch = None)
        assert not getattr(self, "_unsloth_logits_are_empty", False)
    finally:
        Trainer.compute_loss = original


def test_an_explicit_none_logits_is_still_the_sentinel(sft):
    """The distinction is absent-vs-present, not None-vs-not: a model that
    really sets `logits = None` must still be handled."""
    self, _ = _run_steps(sft, None, steps = 2)
    assert self._metrics["train"]["mean_token_accuracy"] == []


# ---- the two warnings must not contradict each other ----------------------

def test_the_entropy_warning_is_not_shown_when_both_are_omitted(sft, caplog):
    """The entropy patch promises "Entropy will be reported as 0.0". When the
    accuracy read then fails, the wrapper reports that BOTH are omitted, so the
    first message is wrong and the user sees two contradictory ones on step 1."""
    import logging
    from unsloth_zoo.temporary_patches.misc import logger as misc_logger
    import trl.trainer.utils as trl_utils
    flag = getattr(trl_utils.entropy_from_logits, "_unsloth_warned", None)
    if flag is None:
        pytest.skip("entropy patch is not installed in this environment")
    was = flag[0]
    flag[0] = False
    try:
        with caplog.at_level(logging.WARNING, logger = misc_logger.name):
            _run_steps(sft, EmptyLogits(), steps = 2)
        text = caplog.text
    finally:
        flag[0] = was
    assert "Both will be omitted" in text
    assert "Entropy will be reported as 0.0" not in text


def test_a_real_failure_leaves_the_entropy_flag_alone(sft):
    """The mute is for the probing call only; an unrelated error must restore
    it rather than swallow the next legitimate entropy warning."""
    from transformers import Trainer
    import trl.trainer.utils as trl_utils
    flag = getattr(trl_utils.entropy_from_logits, "_unsloth_warned", None)
    if flag is None:
        pytest.skip("entropy patch is not installed in this environment")

    def fake(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        raise RuntimeError("something else entirely")

    original = Trainer.compute_loss
    Trainer.compute_loss = fake
    was = flag[0]
    flag[0] = False
    try:
        self = _trainer(sft)
        with pytest.raises(RuntimeError, match = "something else"):
            sft.SFTTrainer.compute_loss(self, self.model, _inputs(),
                                        num_items_in_batch = None)
        assert flag[0] is False
    finally:
        Trainer.compute_loss = original
        flag[0] = was


# ---- nothing may leave an empty metric list behind ------------------------

def _log_would_divide_by_zero(self):
    """Names of metrics trl's own `log` would choke on.

    `SFTTrainer.log` averages every key it finds with `sum(val) / len(val)`
    (0.25.1 sft_trainer.py:1194), so an empty list is a ZeroDivisionError at
    the first logging step, not a missing metric.
    """
    return sorted(n for n, v in self._metrics["train"].items() if len(v) == 0)


def test_a_dense_run_does_not_invent_an_aux_loss_key(sft):
    """`_metrics[mode]` is a defaultdict(list), so reading the key would create
    it, on every step, for models that will never have an aux_loss."""
    self, _ = _run_steps(sft, EmptyLogits(), steps = 3)
    assert "aux_loss" not in self._metrics["train"]
    assert _log_would_divide_by_zero(self) == []


def test_a_healthy_run_does_not_invent_one_either(sft):
    """The count is taken before the probe, so real-logits steps pass through
    it too and a dense trainer would be poisoned without ever failing."""
    self, _ = _run_steps(sft, torch.randn(2, 6, 11), steps = 2)
    assert "aux_loss" not in self._metrics["train"]
    assert _log_would_divide_by_zero(self) == []


@pytest.mark.parametrize("sentinel", SENTINELS)
def test_the_rollback_removes_keys_the_probe_invented(sft, sentinel):
    """The failed probe can get as far as appending entropy before the accuracy
    read fails. Truncating that list to [] leaves the key behind."""
    self, _ = _run_steps(sft, sentinel(), steps = 2)
    assert _log_would_divide_by_zero(self) == []


@pytest.mark.parametrize("sentinel", SENTINELS)
def test_trl_can_actually_log_after_the_retry(sft, sentinel):
    """The end of the story, through trl's real `log` rather than a stand-in."""
    self, _ = _run_steps(sft, sentinel(), steps = 2)
    logs = {}
    import transformers
    original = transformers.Trainer.log
    transformers.Trainer.log = lambda self, d, start_time = None: logs.update(d)
    try:
        sft.SFTTrainer.log(self, {"loss": 1.0})
    finally:
        transformers.Trainer.log = original
    assert logs["loss"] == 1.0


def test_a_metric_the_probe_only_extended_is_truncated_not_dropped(sft):
    """A key that already existed keeps its earlier values; only the probe's
    own additions go."""
    self = _trainer(sft)
    self._metrics["train"]["mean_token_accuracy"].extend([0.5, 0.75])
    from transformers import Trainer
    Out = collections.namedtuple("Out", "loss logits")

    def fake(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        loss = torch.tensor(1.0)
        out = Out(loss, EmptyLogits())
        return (loss, out) if return_outputs else loss

    original = Trainer.compute_loss
    Trainer.compute_loss = fake
    try:
        sft.SFTTrainer.compute_loss(self, self.model, _inputs(),
                                    num_items_in_batch = None)
    finally:
        Trainer.compute_loss = original
    assert self._metrics["train"]["mean_token_accuracy"] == [0.5, 0.75]


# ---- the public positional call shape -------------------------------------

@pytest.mark.parametrize("sentinel", SENTINELS)
def test_a_positional_return_outputs_is_replaced_not_duplicated(sft, sentinel):
    """`compute_loss(model, inputs, False, ...)` is a legal public call. Adding
    the keyword beside the positional would raise "got multiple values"."""
    from transformers import Trainer
    Out = collections.namedtuple("Out", "loss logits aux_loss")

    def fake(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        loss = torch.tensor(1.0)
        out = Out(loss, sentinel(), torch.tensor(0.25))
        return (loss, out) if return_outputs else loss

    original = Trainer.compute_loss
    Trainer.compute_loss = fake
    try:
        self = _trainer(sft)
        self.aux_loss_enabled = True
        for _ in range(2):
            sft.SFTTrainer.compute_loss(self, self.model, _inputs(), False, None)
    finally:
        Trainer.compute_loss = original
    assert self._metrics["train"]["aux_loss"] == [0.25, 0.25]


# ---- eval and predict, where the caller already asked for outputs ---------

@pytest.mark.parametrize("sentinel", SENTINELS)
def test_aux_loss_survives_an_eval_step_too(sft, sentinel):
    """`Trainer.prediction_step` calls compute_loss(..., return_outputs=True),
    so `force` is False and the outputs side channel would be empty for exactly
    the batches that still want the metric."""
    from transformers import Trainer
    Out = collections.namedtuple("Out", "loss logits aux_loss")

    def fake(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        loss = torch.tensor(1.0)
        out = Out(loss, sentinel(), torch.tensor(0.25))
        return (loss, out) if return_outputs else loss

    original = Trainer.compute_loss
    Trainer.compute_loss = fake
    try:
        self = _trainer(sft)
        self.aux_loss_enabled = True
        for _ in range(3):
            got = sft.SFTTrainer.compute_loss(
                self, self.model, _inputs(), return_outputs = True,
                num_items_in_batch = None)
            assert isinstance(got, tuple) and len(got) == 2, got
    finally:
        Trainer.compute_loss = original
    assert self._metrics["train"]["aux_loss"] == [0.25, 0.25, 0.25]


# ---- a retry that also fails must not latch the trainer -------------------

def test_a_failing_retry_does_not_latch_the_fast_path():
    """If the logits are wanted by something other than the metric block
    (trl's loss_type='dft', a custom loss), the retry raises too. Latching
    before it would keep this trainer in the no-logits path for the rest of the
    process, including a rerun with UNSLOTH_RETURN_LOGITS=1 set.

    Driven through the wrapper directly: the detector reads a frame local
    named `outputs`, and going via the installed trl would only ever raise
    from the metric block, which the retry skips by construction.
    """
    from unsloth_zoo.temporary_patches.misc import _sft_wrap_compute_loss
    Out = collections.namedtuple("Out", "loss logits")
    calls = []

    def original(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        calls.append(self.args.use_liger_kernel)
        outputs = Out(torch.tensor(1.0), EmptyLogits())  # noqa: F841 - read off the frame
        raise RuntimeError("this loss needs the logits itself")

    self = _trainer()
    with pytest.raises(RuntimeError, match = "needs the logits"):
        _sft_wrap_compute_loss(original)(self, self.model, _inputs(),
                                         num_items_in_batch = None)
    assert calls == [False, True], calls
    assert not getattr(self, "_unsloth_logits_are_empty", False)


def test_a_succeeding_retry_still_latches(sft):
    """The other half: the fast path exists so step 2 onwards costs one attempt,
    not two."""
    self, calls = _run_steps(sft, EmptyLogits(), steps = 3)
    assert getattr(self, "_unsloth_logits_are_empty", False)
    assert calls == 4, calls


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
