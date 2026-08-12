# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Inductor codegen failures degrade to eager instead of ending the run.

torch 2.12 refuses to generate code for the ragged tail of
``torch.chunk(x, chunks = N)`` under dynamic shapes:

    InductorError: CantSplit: 202048*s47*s87 - 3434816*(((s47*s87 + 17)//18))
    not divisible by s47*s87 - 17*(((s47*s87 + 17)//18))

That is arithmetic which is true by inspection, and which the same predicate
answers correctly on 2.9, 2.10 and 2.11. A backend simplifier gap should cost
speed, not kill a GRPO run, so ``_fall_back_to_eager_on_recompile_limit`` now
catches it the way it already catches cache exhaustion.

These tests drive the wrapper directly with a compiled callable that raises,
rather than trying to provoke Inductor, so they run on CPU and stay valid on
torch versions where the real expression compiles fine.
"""

from __future__ import annotations

import pytest

from unsloth_zoo.temporary_patches import utils as patch_utils


@pytest.fixture(autouse = True)
def _clear_latches():
    """The latch is process-global and keyed by label, so tests must not leak."""
    patch_utils._LATCHED_EAGER_LABELS.clear()
    patch_utils._PENDING_EAGER_LABELS.clear()
    patch_utils._COMPILED_OK_LABELS.clear()
    yield
    patch_utils._LATCHED_EAGER_LABELS.clear()
    patch_utils._PENDING_EAGER_LABELS.clear()
    patch_utils._COMPILED_OK_LABELS.clear()


def _inductor_error(message = "CantSplit: 202048*s47*s87 not divisible by s47*s87"):
    """Build whichever backend exception this torch has.

    The constructors disagree across versions: 2.12's `InductorError` takes
    `(inner_exception, first_useful_frame)`, `BackendCompilerFailed` takes
    `(backend_fn, exc)`, and older builds accept a bare message. Try the
    shapes rather than pinning one, so the test tracks the tuple the wrapper
    actually catches instead of a single release's signature.
    """
    errors = patch_utils._backend_compile_errors()
    if not errors:
        pytest.skip("this torch exposes no backend compile exception")
    cls = errors[0]
    inner = RuntimeError(message)
    for args in ((inner, None), (inner, inner), (message,), (inner,)):
        try:
            return cls(*args)
        except TypeError:
            continue
    pytest.skip(f"cannot construct {cls.__name__} on this torch")


def test_the_backend_error_tuple_is_not_empty_on_this_torch():
    """If this ever goes empty the fallback silently stops existing."""
    assert patch_utils._backend_compile_errors(), (
        "no InductorError or BackendCompilerFailed found; the wrapper would "
        "pass a codegen failure straight through"
    )


def test_a_codegen_failure_falls_back_to_eager():
    calls = {"compiled": 0, "eager": 0}

    def compiled(x):
        calls["compiled"] += 1
        raise _inductor_error()

    def eager(x):
        calls["eager"] += 1
        return x * 2

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, eager, "test-codegen",
    )

    assert wrapped(21) == 42
    assert calls == {"compiled": 1, "eager": 1}


def test_the_fallback_latches_so_the_backend_is_not_retried():
    """A codegen refusal is deterministic; retrying pays the compile twice."""
    calls = {"compiled": 0, "eager": 0}

    def compiled(x):
        calls["compiled"] += 1
        raise _inductor_error()

    def eager(x):
        calls["eager"] += 1
        return x + 1

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, eager, "test-latch",
    )

    assert [wrapped(1), wrapped(2), wrapped(3)] == [2, 3, 4]
    assert calls["compiled"] == 1, "the compiled path was retried after latching"
    assert calls["eager"] == 3


def test_only_this_label_latches():
    """Unlike cache exhaustion, a codegen refusal is local to one graph."""
    def compiled_bad(x):
        raise _inductor_error()

    def compiled_good(x):
        return "compiled"

    bad = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled_bad, lambda x: "eager", "test-bad",
    )
    good = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled_good, lambda x: "eager", "test-good",
    )

    assert bad(1) == "eager"
    assert good(1) == "compiled", "an unrelated region was knocked eager"
    assert "test-good" not in patch_utils._LATCHED_EAGER_LABELS


def test_a_real_runtime_error_still_raises():
    """The net must be narrow: only backend codegen, never the user's bug."""
    def compiled(x):
        raise ValueError("a real bug in the model")

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, lambda x: "eager", "test-runtime",
    )

    with pytest.raises(ValueError, match = "a real bug"):
        wrapped(1)


def test_the_hard_switch_lets_the_error_escape(monkeypatch):
    monkeypatch.setenv("UNSLOTH_HARD_BACKEND_FAILURE", "1")
    errors = patch_utils._backend_compile_errors()
    if not errors:
        pytest.skip("this torch exposes no backend compile exception")

    def compiled(x):
        raise _inductor_error()

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, lambda x: "eager", "test-hard",
    )

    with pytest.raises(errors[0]):
        wrapped(1)


def test_the_switch_is_read_live_not_at_import(monkeypatch):
    """A test setting the variable after import must still be honoured."""
    monkeypatch.delenv("UNSLOTH_HARD_BACKEND_FAILURE", raising = False)
    assert patch_utils._wants_hard_backend_failure() is False
    monkeypatch.setenv("UNSLOTH_HARD_BACKEND_FAILURE", "1")
    assert patch_utils._wants_hard_backend_failure() is True


# ---- the checkpoint distinction -------------------------------------------
#
# `_give_up` must re-raise inside a non-reentrant checkpoint, because it latches
# every borrower and some of them packed activations compiled. The backend path
# latches one label, so the only pack that can desynchronise is this function's
# own -- and a first-call codegen refusal means it packed nothing compiled.


def test_a_first_compile_refusal_falls_back_even_under_a_checkpoint(monkeypatch):
    """The Muse Glimmer case: Inductor refuses before anything was ever packed."""
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint", lambda: True)
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", True)

    def compiled(x):
        raise _inductor_error()

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, lambda x: "eager", "test-first-compile",
    )

    assert wrapped(1) == "eager", "a first-call refusal should not end the step"


def test_a_later_refusal_after_a_successful_compile_still_raises(monkeypatch):
    """A new dynamic shape refusing after an earlier one packed compiled.

    Here the pack was compiled and the recompute would be eager, which either
    aborts the backward or returns wrong gradients, so the step must end.
    """
    calls = {"n": 0}

    def compiled(x):
        calls["n"] += 1
        if calls["n"] == 1:
            return "compiled"
        raise _inductor_error()

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, lambda x: "eager", "test-later-refusal",
    )

    assert wrapped(1) == "compiled"

    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint", lambda: True)
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", True)
    errors = patch_utils._backend_compile_errors()
    with pytest.raises(errors):
        wrapped(2)


def test_outside_a_checkpoint_a_later_refusal_still_falls_back(monkeypatch):
    """Nothing is packed, so there is no pack/recompute pair to protect."""
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint", lambda: False)
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", False)
    calls = {"n": 0}

    def compiled(x):
        calls["n"] += 1
        if calls["n"] == 1:
            return "compiled"
        raise _inductor_error()

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, lambda x: "eager", "test-no-checkpoint",
    )

    assert wrapped(1) == "compiled"
    assert wrapped(2) == "eager"


# ---- what the review round found ------------------------------------------
#
# Four holes in the checkpoint gate above, all of them cases where the wrapper
# either re-raises when the fallback was safe, or falls back when it was not.


def test_a_refusal_in_a_later_step_falls_back(monkeypatch):
    """`compiled_ok` must not answer for a step that is already over.

    Held per wrapper it stays true forever, so the first refusal in ANY later
    step re-raises even though nothing compiled has been packed in that step.
    That defeats the fallback for exactly the long-running GRPO case it exists
    for.
    """
    calls = {"n": 0}

    def compiled(x):
        calls["n"] += 1
        if calls["n"] == 1:
            return "compiled"
        raise _inductor_error()

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, lambda x: "eager", "test-later-step",
    )
    assert wrapped(1) == "compiled"

    # The step boundary. This is what `apply_pending_eager_fallbacks` does.
    patch_utils._COMPILED_OK_LABELS.clear()

    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint", lambda: True)
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", True)
    assert wrapped(2) == "eager", "a new step re-raised on its first refusal"


def test_a_successful_budget_retry_is_recorded_as_compiled():
    """A retry that compiles has packed compiled activations, so record it.

    The retry returns from the exception branch and never reaches the `else:`
    clause that normally does the recording, which left the wrapper believing
    nothing compiled had run.

    Worth being precise about the consequence, because it is narrower than it
    looks: the retry also latches this label eager, so the SAME wrapper never
    makes another compiled call and cannot itself hit a later backend refusal.
    The record still has to be right -- it is the process-wide answer to "did
    this label pack something compiled in this step" -- but the pack/recompute
    divergence this guards against is not reachable through this path today.
    """
    calls = {"n": 0}

    def compiled(x):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _recompile_limit_error()
        return "retried"

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, lambda x: "eager", "test-retry-counts",
    )
    if wrapped(1) != "retried":
        pytest.skip("this torch does not take the budget-retry path")
    assert "test-retry-counts" in patch_utils._COMPILED_OK_LABELS, (
        "a retry that compiled and returned was not recorded as compiled"
    )


def test_the_compiled_pack_marker_is_restored_after_a_refusal(monkeypatch):
    """Nothing compiled ran, so the process-wide marker must not stay set.

    Left set, an unrelated wrapper exhausting its budget in the same checkpoint
    reads it as evidence of a compiled pack and ends the step for no reason.
    """
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", False)
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint", lambda: True)
    # The real one sets the marker, which is precisely what has to be undone.
    monkeypatch.setattr(
        patch_utils, "_note_packed_under_checkpoint",
        lambda: setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", True),
    )

    def compiled(x):
        raise _inductor_error()

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, lambda x: "eager", "test-marker-restored",
    )
    assert wrapped(1) == "eager"
    assert patch_utils._PACKED_COMPILED_IN_CHECKPOINT is False, (
        "a refusal that ran no compiled code left the compiled-pack marker set"
    )


def test_a_non_inductor_backend_failure_still_raises():
    """`BackendCompilerFailed` wraps any backend, not only Inductor.

    `torch_compile_with_fallback(..., backend = custom)` is supported, and
    swallowing a custom backend's exception would hide a configuration or
    programming error behind a permanent silent switch to eager.
    """
    try:
        from torch._dynamo.exc import BackendCompilerFailed
    except Exception:
        pytest.skip("this torch has no BackendCompilerFailed")

    inner = ValueError("my custom backend is misconfigured")
    built = None
    def _backend_fn(gm, example_inputs): return gm
    for args in (
        (_backend_fn, inner, None),   # 2.9+: (backend_fn, inner, first_useful_frame)
        (_backend_fn, inner),
        (inner, None),
        (inner,),
    ):
        try:
            built = BackendCompilerFailed(*args)
        except TypeError:
            continue
        break
    if built is None:
        pytest.skip("cannot construct BackendCompilerFailed on this torch")
    if patch_utils._is_inductor_codegen_failure(built):
        pytest.skip("this torch gives the wrapper no usable backend identity")

    def compiled(x):
        raise built

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, lambda x: "eager", "test-custom-backend",
    )
    with pytest.raises(BackendCompilerFailed):
        wrapped(1)
    assert "test-custom-backend" not in patch_utils._LATCHED_EAGER_LABELS


def _recompile_limit_error():
    errors = patch_utils._recompile_limit_errors()
    if not errors:
        pytest.skip("this torch exposes no recompile-limit exception")
    cls = errors[0]
    for args in (("recompile limit reached",), ()):
        try:
            return cls(*args)
        except TypeError:
            continue
    pytest.skip(f"cannot construct {cls.__name__}")


def test_restoring_the_marker_cannot_erase_an_earlier_compiled_pack(monkeypatch):
    """The restore writes a process-wide global, so it must not clear too much.

    If another region already packed something compiled inside this checkpoint,
    the marker was true BEFORE this call and has to stay true: clearing it would
    tell a later `_give_up` the checkpoint is clean when it is not, and that is
    the direction that returns wrong gradients rather than merely ending a step.
    Restoring the prior value rather than assigning False is what makes this
    safe, and this test is the difference between the two.
    """
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint", lambda: True)
    # An earlier region in this same checkpoint already packed compiled.
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", True)
    monkeypatch.setattr(
        patch_utils, "_note_packed_under_checkpoint",
        lambda: setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", True),
    )

    def compiled(x):
        raise _inductor_error()

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, lambda x: "eager", "test-marker-not-erased",
    )
    # This label itself never compiled, so it falls back rather than raising.
    assert wrapped(1) == "eager"
    assert patch_utils._PACKED_COMPILED_IN_CHECKPOINT is True, (
        "the restore erased an earlier region's compiled pack"
    )
