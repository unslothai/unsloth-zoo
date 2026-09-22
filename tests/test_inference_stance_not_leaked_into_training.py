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
"""The inference-only compiler stance must never be set during training.

`compiler.py` emits a block that flips dynamo's stance to `eager_on_recompile`
once a model has served two label-free forwards. That is a generation
optimisation, and the stance is process-global.

A training loop that also runs label-free forwards must not inherit it. Those
forwards are ordinary: a GKD or distillation teacher scoring a batch, an
evaluation pass, on-policy sampling. Two things go wrong if the stance sticks.

Correctness: `torch.utils.checkpoint` runs its recomputation with dynamo
disabled on purpose, and says why in its own Note, that AOTDispatch may save
different activations than eager and conflict with the strict activation count
checks in `check_recomputed_tensors_match`. `_callback_from_stance` downgrades
that disable from None to False for `eager_on_recompile` alone, so the pack and
the recompute can run in different compile modes and the backward aborts.

Performance: once the stance is set, no further frame is ever compiled, so the
rest of training runs with whatever happened to be cached.

These drive the emitted template directly rather than a real model, because
reaching the generated code needs a checkpoint and a GPU.
"""
import textwrap

import pytest

import unsloth_zoo.compiler as compiler_module


class FakeStance:
    def __init__(self):
        self.stance = "default"
        self.skip_guard_eval_unsafe = False


def _build_forward():
    """Compose the two emitted blocks the way the forward templates do."""
    restore = compiler_module.__DYNAMO__RESTORE__
    recompiling = compiler_module.__DYNAMO__RECOMPILING__

    body = (
        "def forward(labels):\n"
        "    if labels is not None:\n"
        + textwrap.indent(textwrap.dedent(restore), " " * 8) +
        "        return 'train'\n"
        "    else:\n"
        + textwrap.indent(textwrap.dedent(recompiling), " " * 8) +
        "        return 'inference'\n"
    )

    stance = FakeStance()
    calls = []

    class FakeEvalFrame:
        _stance = stance

    def set_stance(stance = None, skip_guard_eval_unsafe = False):
        calls.append(stance)
        FakeEvalFrame._stance.stance = stance

    namespace = {
        "torch_dynamo_eval_frame": FakeEvalFrame,
        "torch_compiler_set_stance": set_stance,
        "logger_compiler": None,
        "UNSLOTH_ENABLE_LOGGING": False,
        "INFERENCE_RUNS": 0,
        "UNSLOTH_SET_INFERENCE_STANCE": False,
        "UNSLOTH_SAW_TRAINING_FORWARD": False,
    }
    exec(compile(body, "<emitted>", "exec"), namespace)
    return namespace["forward"], stance, namespace, calls


def _build_forward_without_the_restore():
    """The pre-fix shape: the inference block alone, with nothing to undo it."""
    recompiling = compiler_module.__DYNAMO__RECOMPILING__
    body = (
        "def forward(labels):\n"
        "    global INFERENCE_RUNS, UNSLOTH_SET_INFERENCE_STANCE\n"
        "    global UNSLOTH_SAW_TRAINING_FORWARD\n"
        "    if labels is not None:\n"
        "        return 'train'\n"
        "    else:\n"
        + textwrap.indent(textwrap.dedent(recompiling), " " * 8) +
        "        return 'inference'\n"
    )
    stance = FakeStance()

    class FakeEvalFrame:
        _stance = stance

    def set_stance(stance = None, skip_guard_eval_unsafe = False):
        FakeEvalFrame._stance.stance = stance

    namespace = {
        "torch_dynamo_eval_frame": FakeEvalFrame,
        "torch_compiler_set_stance": set_stance,
        "logger_compiler": None,
        "UNSLOTH_ENABLE_LOGGING": False,
        "INFERENCE_RUNS": 0,
        "UNSLOTH_SET_INFERENCE_STANCE": False,
        # Never latched, because nothing sets it without the restore block.
        "UNSLOTH_SAW_TRAINING_FORWARD": False,
    }
    exec(compile(body, "<emitted-old>", "exec"), namespace)
    return namespace["forward"], stance


def test_negative_control_the_old_shape_really_did_leak():
    # Guards the tests above against passing vacuously. Drive the inference
    # block with no restore, in the GKD order, and the stance sticks from the
    # second teacher forward onwards and is still set when the backward runs.
    forward, stance = _build_forward_without_the_restore()
    leaked = []
    for step in range(5):
        forward(labels = "student")
        forward(labels = None)          # teacher, label-free
        leaked.append(stance.stance)
    assert leaked[0] == "default", "nothing should flip on the first step"
    assert leaked[1:] == ["eager_on_recompile"] * 4, (
        f"expected the old shape to leak from step 1 onwards, got {leaked}")


def test_the_generation_optimisation_still_engages():
    # Nothing has trained, so this is a real inference process and the stance
    # is exactly what it was written for.
    forward, stance, ns, _ = _build_forward()
    for _ in range(4):
        forward(labels = None)
    assert stance.stance == "eager_on_recompile"


def test_a_training_forward_turns_it_off():
    forward, stance, ns, _ = _build_forward()
    for _ in range(4):
        forward(labels = None)
    assert stance.stance == "eager_on_recompile"

    forward(labels = "some labels")
    assert stance.stance == "default"
    assert ns["UNSLOTH_SAW_TRAINING_FORWARD"] is True


def test_a_teacher_forward_cannot_flip_it_back_within_the_step():
    # This is the ordering that makes restoring alone insufficient. Inside one
    # GKD step the student forward carries labels, the teacher forward does not,
    # and the backward runs last. Without the latch the teacher forward would
    # set the stance again before the recompute.
    forward, stance, ns, _ = _build_forward()
    for step in range(5):
        forward(labels = "student")
        forward(labels = None)          # teacher, label-free
        assert stance.stance == "default", f"stance leaked on step {step}"


def test_the_stance_is_never_set_after_training_has_started():
    forward, stance, ns, calls = _build_forward()
    forward(labels = "student")
    for _ in range(20):
        forward(labels = None)
    assert stance.stance == "default"
    assert "eager_on_recompile" not in calls


def test_a_stance_the_user_chose_is_left_alone():
    forward, stance, ns, calls = _build_forward()
    stance.stance = "force_eager"
    forward(labels = "student")
    # We never set it, so we must not clear it.
    assert stance.stance == "force_eager"
    assert calls == []


def test_the_reset_helper_re_arms_the_optimisation():
    src = compiler_module._license_header
    assert "def reset_inference_stance():" in src, (
        "the emitted preamble must expose a way back to the optimisation")


def test_the_restore_block_is_spliced_into_every_forward_template():
    import inspect
    src = inspect.getsource(compiler_module)
    # Every template that splices the inference block must also splice the
    # restore, or that template leaks the stance.
    n_recompiling = src.count(
        '.replace("__DYNAMO__RECOMPILING__", __DYNAMO__RECOMPILING__)')
    n_restore = src.count(
        '.replace("__DYNAMO__RESTORE__", __DYNAMO__RESTORE__)')
    assert n_recompiling >= 1
    assert n_restore == n_recompiling, (
        f"{n_recompiling} templates set the stance but only {n_restore} "
        "restore it")
