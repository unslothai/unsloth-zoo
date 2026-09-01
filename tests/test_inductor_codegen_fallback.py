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
import torch

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


def _inductor_backend_fn(gm, example_inputs):
    """Stands in for the backend object torch hands `BackendCompilerFailed`.

    `BackendCompilerFailed.__init__` stores `getattr(backend_fn, "__name__",
    "?")` as `backend_name`, and torch passes `OutputGraph.compiler_fn`, whose
    `__name__` is `inductor`. Verified by execution on torch 2.9.1: making
    `torch._inductor.compile_fx.compile_fx` raise a bare `RuntimeError` under
    `torch.compile(backend = "inductor")` produces a `BackendCompilerFailed`
    with `backend_name == "inductor"`.
    """
    return gm
_inductor_backend_fn.__name__ = "inductor"


def _inductor_error(message = "CantSplit: 202048*s47*s87 not divisible by s47*s87"):
    """Build whichever backend exception this torch has, WITH an Inductor identity.

    The constructors disagree across versions: 2.12's `InductorError` takes
    `(inner_exception, first_useful_frame)`, `BackendCompilerFailed` takes
    `(backend_fn, inner_exception)` up to 2.8 and `(backend_fn,
    inner_exception, first_useful_frame)` from 2.9. Try the shapes rather than
    pinning one, so the test tracks the tuple the wrapper actually catches
    instead of a single release's signature.

    Every candidate has to carry an Inductor identity, and the result is
    checked for one. On torch 2.6 `InductorError` does not exist, so
    `_backend_compile_errors()[0]` is `BackendCompilerFailed` and the old
    `(inner, None)` shape fitted its two-argument signature -- producing an
    exception whose backend was the RuntimeError (`backend_name == "?"`) and
    whose `inner_exception` was `None`. `_is_inductor_codegen_failure` rightly
    rejects that, so every fallback test below asserted the OPPOSITE of the
    behaviour it was written for on exactly the version where the wrapped form
    is the only form.
    """
    errors = patch_utils._backend_compile_errors()
    if not errors:
        pytest.skip("this torch exposes no backend compile exception")
    cls = errors[0]
    inner = RuntimeError(message)
    for args in (
        (inner, None),                              # InductorError, 2.7+
        (_inductor_backend_fn, inner, None),        # BackendCompilerFailed, 2.9+
        (_inductor_backend_fn, inner),              # BackendCompilerFailed, <= 2.8
    ):
        try:
            built = cls(*args)
        except TypeError:
            continue
        if patch_utils._is_inductor_codegen_failure(built):
            return built
    pytest.skip(f"cannot construct an Inductor-identified {cls.__name__}")


class _Torch26BackendCompilerFailed(RuntimeError):
    """torch 2.6's `BackendCompilerFailed`, reproduced attribute for attribute.

    2.6 is in the supported range and is the version with no `InductorError`,
    so the wrapped form is the ONLY form the fallback can see there. Standing
    it in is the only way to exercise that path from a newer torch, and it is a
    faithful copy of the 2.6.0 source rather than an approximation.
    """
    def __init__(self, backend_fn, inner_exception):
        self.backend_name = getattr(backend_fn, "__name__", "?")
        self.inner_exception = inner_exception
        super().__init__(
            f"backend={self.backend_name!r} raised:\n"
            f"{type(inner_exception).__name__}: {inner_exception}"
        )


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
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        lambda **_: True)
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

    The successful call has to happen INSIDE the checkpoint, which is what
    makes it a compiled pack. Entering the region only for the refusal, as this
    test used to, describes a different situation entirely -- a compiled call
    somewhere else in the step -- and that one is safe to fall back from.
    """
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        lambda **_: True)
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", True)
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

    errors = patch_utils._backend_compile_errors()
    with pytest.raises(errors):
        wrapped(2)


def test_outside_a_checkpoint_a_later_refusal_still_falls_back(monkeypatch):
    """Nothing is packed, so there is no pack/recompute pair to protect."""
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        lambda **_: False)
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

    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        lambda **_: True)
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", True)
    assert wrapped(2) == "eager", "a new step re-raised on its first refusal"


def test_a_successful_budget_retry_is_recorded_as_compiled(monkeypatch):
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

    Under a checkpoint, because that is the only place a compiled return packs
    something that will be recomputed, and therefore the only place the record
    means anything.
    """
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        lambda **_: True)
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", True)
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
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        lambda **_: True)
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
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        lambda **_: True)
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


# ---- what the second review round found ------------------------------------
#
# Five more, all of them the same shape as the first four: a path where the
# wrapper either re-raises where the fallback was safe, or never sees the
# refusal at all.


def test_the_wrapped_form_is_recognised_by_its_recorded_backend_name():
    """`BackendCompilerFailed` records `backend_name`; nothing records `backend`.

    This is the shape torch actually produces when Inductor raises something
    that is not an `InductorError` -- confirmed on 2.9.1 by making `compile_fx`
    raise a bare `RuntimeError`: `backend_name == "inductor"`, no `backend`
    attribute, and `inner_exception` a `builtins.RuntimeError` whose module
    tells the caller nothing. Reading the attribute that never exists made this
    branch dead code, so the wrapped form `_backend_compile_errors()` goes out
    of its way to catch was re-raised anyway.
    """
    built = _Torch26BackendCompilerFailed(
        _inductor_backend_fn, RuntimeError("CantSplit: not divisible"))
    assert built.backend_name == "inductor"
    assert not hasattr(built, "backend"), "no torch has ever set `backend`"
    assert patch_utils._is_inductor_codegen_failure(built), (
        "an Inductor refusal recorded under `backend_name` was not recognised"
    )


def test_a_non_inductor_backend_name_is_still_rejected():
    """The `backend_name` read must not widen the net to every backend."""
    def my_backend(gm, example_inputs): return gm
    built = _Torch26BackendCompilerFailed(
        my_backend, ValueError("my custom backend is misconfigured"))
    assert built.backend_name == "my_backend"
    assert not patch_utils._is_inductor_codegen_failure(built)


def test_a_torch_2_6_shaped_refusal_falls_back_to_eager(monkeypatch):
    """2.6 has no `InductorError`, so the wrapped form is the only form.

    Stand in the 2.6 exception and drive the real wrapper with it: without the
    `backend_name` read this re-raises, which on 2.6 means the fallback does
    not exist at all.
    """
    monkeypatch.setattr(
        patch_utils, "_backend_compile_errors",
        lambda: (_Torch26BackendCompilerFailed,),
    )

    def compiled(x):
        raise _Torch26BackendCompilerFailed(
            _inductor_backend_fn, RuntimeError("CantSplit: not divisible"))

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, lambda x: "eager", "test-torch26-shape",
    )
    assert wrapped(1) == "eager"


def test_a_refusal_from_the_budget_retry_falls_back(monkeypatch):
    """Cache exhaustion first, then Inductor refuses once the budget is bumped.

    `_retry_with_more_budget` calls `compiled_func` inside its own `try`, and
    its `except BaseException` re-raises. Python does not route that to the
    wrapper's sibling `except backend_errors`, so the refusal escaped and ended
    the run -- the exact outcome the fallback exists to prevent, reached by the
    one path where the compiler had to be given more budget before it got as
    far as refusing.
    """
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        lambda **_: False)
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", False)
    calls = {"n": 0}

    def compiled(x):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _recompile_limit_error()
        raise _inductor_error()

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, lambda x: "eager", "test-retry-refusal",
    )
    if not patch_utils._recompile_limit_errors():
        pytest.skip("this torch exposes no recompile-limit exception")
    assert wrapped(1) == "eager", (
        "a codegen refusal raised by the budget retry escaped the wrapper"
    )
    assert "test-retry-refusal" in patch_utils._LATCHED_EAGER_LABELS


def test_a_refusal_from_the_budget_retry_is_not_recorded_as_compiled(monkeypatch):
    """It ran EAGER, so it must not count as a compiled pack.

    Guards the sentinel: returning the eager value down the ordinary
    retry-succeeded path would record this label as having packed something
    compiled, which is the lie that ends a later step for no reason -- or, in a
    checkpoint, the one that recomputes in the wrong mode.
    """
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        lambda **_: True)
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", True)
    calls = {"n": 0}

    def compiled(x):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _recompile_limit_error()
        raise _inductor_error()

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, lambda x: "eager", "test-retry-refusal-record",
    )
    if not patch_utils._recompile_limit_errors():
        pytest.skip("this torch exposes no recompile-limit exception")
    assert wrapped(1) == "eager"
    assert "test-retry-refusal-record" not in patch_utils._COMPILED_OK_LABELS


def test_a_legacy_budget_retry_is_recorded_as_compiled(monkeypatch):
    """torch 2.4 reaches the retry through the graph-break arm, not the typed one.

    Cache exhaustion has no exception class there, so it arrives as
    `Unsupported` and is matched by message. That arm returned the retry's
    result without recording it, so 2.4 took the opposite decision from 2.7+
    when a backend refusal followed in the same step.
    """
    graph_break_errors = patch_utils._disabled_hook_graph_break_error()
    if not graph_break_errors:
        pytest.skip("this torch exposes no graph-break exception")
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        lambda **_: True)
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", True)
    calls = {"n": 0}

    def compiled(x):
        calls["n"] += 1
        if calls["n"] == 1:
            try:
                raise graph_break_errors[0]("recompile_limit reached")
            except TypeError:
                raise graph_break_errors[0]()
        return "retried"

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, lambda x: "eager", "test-legacy-retry",
    )
    if wrapped(1) != "retried":
        pytest.skip("this torch does not take the legacy budget-retry path")
    assert "test-legacy-retry" in patch_utils._COMPILED_OK_LABELS, (
        "a legacy retry that compiled and returned was not recorded as compiled"
    )


def test_a_compiled_call_outside_a_checkpoint_is_not_a_compiled_pack(monkeypatch):
    """GRPO generates with the same patched kernels it then trains with.

    A compiled call made outside a non-reentrant checkpoint packs nothing that
    is ever recomputed, so it cannot desynchronise a pack from a recompute and
    must not stop the fallback. Recording every compiled return made the first
    refusal of the training pass end the step because the generation pass had
    succeeded earlier in the same step.
    """
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        lambda **_: False)
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", False)
    calls = {"n": 0}

    def compiled(x):
        calls["n"] += 1
        if calls["n"] == 1:
            return "compiled"
        raise _inductor_error()

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, lambda x: "eager", "test-uncheckpointed-success",
    )
    assert wrapped(1) == "compiled"           # generation: no region open
    assert "test-uncheckpointed-success" not in patch_utils._COMPILED_OK_LABELS

    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        lambda **_: True)
    assert wrapped(2) == "eager", (
        "a compiled call made outside any checkpoint blocked the fallback"
    )


def test_an_unknown_checkpoint_answer_still_counts_as_packed(monkeypatch):
    """`None` is "torch cannot say", and that has to count as yes.

    Before 2.8 there is no hook accessor, and a user's own
    `saved_tensors_hooks` can sit above ours on any version. Over-recording
    ends a step that `force_eager_fallback` retries; under-recording recomputes
    a compiled pack eagerly and hands back wrong gradients, so the unknown
    answer must take the expensive side.

    Only the RECORD is asserted here, which is what `_note_compiled_ok` owns.
    Whether `_give_up_on_backend` then raises is a separate decision, and on an
    unknown answer it leans on the marker -- so the marker is set for the
    refusal, matching a step where some region is known to have packed.
    """
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        lambda **_: None)
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", False)
    calls = {"n": 0}

    def compiled(x):
        calls["n"] += 1
        if calls["n"] == 1:
            return "compiled"
        raise _inductor_error()

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, lambda x: "eager", "test-unknown-checkpoint",
    )
    assert wrapped(1) == "compiled"
    assert "test-unknown-checkpoint" in patch_utils._COMPILED_OK_LABELS, (
        "an unknown checkpoint answer was read as 'no region', which is the "
        "direction that recomputes a compiled pack eagerly"
    )
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", True)
    errors = patch_utils._backend_compile_errors()
    with pytest.raises(errors):
        wrapped(2)


def test_a_backend_refusal_is_evidence_for_force_eager_fallback(monkeypatch):
    """The transition has to be recorded where `force_eager_fallback` reads it.

    It deliberately does not read `_LATCHED_EAGER_LABELS` -- that set is
    permanent, so a discarded model's labels would answer for this one -- and
    otherwise asks the LIVE wrappers. GRPO's `accumulate_chunk` is built inside
    the forward and collected before the backward that calls this, so with only
    the permanent latch recorded the call returns 0, the caller reads that as
    "no compile-mode flip happened" and re-raises the failure it was asked to
    retry past. Every other give-up path already writes `_RECENT_EAGER_LABELS`.
    """
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        lambda **_: True)
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", True)
    monkeypatch.setattr(patch_utils, "_RECENT_EAGER_LABELS", set())
    monkeypatch.setattr(patch_utils, "_EAGER_FALLBACK_WRAPPERS", [])

    def _transient():
        """Built inside the forward and dropped before backward, like GRPO's."""
        calls = {"n": 0}
        def compiled(x):
            calls["n"] += 1
            if calls["n"] == 1:
                return "compiled"
            raise _inductor_error()
        w = patch_utils._fall_back_to_eager_on_recompile_limit(
            compiled, lambda x: "eager", "test-transient-evidence",
        )
        assert w(1) == "compiled"
        with pytest.raises(patch_utils._backend_compile_errors()):
            w(2)

    _transient()
    import gc; gc.collect()

    assert "test-transient-evidence" in patch_utils._RECENT_EAGER_LABELS
    assert patch_utils.force_eager_fallback() > 0, (
        "the backend fallback left no evidence, so the caller would re-raise"
    )


def test_the_error_builder_produces_an_inductor_identity_on_a_2_6_shaped_torch(
        monkeypatch):
    """The builder must not hand the tests an unrecognisable exception.

    On a torch with no `InductorError`, `_backend_compile_errors()[0]` is
    `BackendCompilerFailed`, whose two-argument signature happily accepts
    `(inner, None)` -- producing `backend_name == "?"` and
    `inner_exception is None`. Every fallback test above then drove the wrapper
    with something `_is_inductor_codegen_failure` correctly rejects, so on 2.6
    they asserted eager fallback against an exception that re-raises by design.
    """
    monkeypatch.setattr(
        patch_utils, "_backend_compile_errors",
        lambda: (_Torch26BackendCompilerFailed,),
    )
    built = _inductor_error()
    assert isinstance(built, _Torch26BackendCompilerFailed)
    assert built.backend_name == "inductor", (
        f"the builder produced backend_name={built.backend_name!r}"
    )
    assert isinstance(built.inner_exception, RuntimeError)
    assert patch_utils._is_inductor_codegen_failure(built)


# ---- what the SECOND review round found -----------------------------------
#
# The post-call `_note_compiled_ok` probe runs on the SUCCESS path, so unlike
# every probe above it is paid by runs that never fall back at all. Two
# consequences, one a cost and one a correctness hole, and the same one-line
# marker latch closes both.


def test_a_successful_call_does_not_reprobe_once_the_marker_has_latched(
        monkeypatch):
    """The success-path probe must read the cheap global BEFORE the walk.

    `_note_compiled_ok` asked `_in_non_reentrant_checkpoint() is False and not
    _PACKED_COMPILED_IN_CHECKPOINT`, and Python evaluates the left operand of
    `and` first, so the probe was paid on EVERY successful compiled call even
    after the marker had latched and the answer could not change. On a torch
    with no saved-tensor-hook accessor that probe is an uncapped walk to the
    root of the Python stack, which the module itself prices at ~15us against
    ~0.1us for the wrapper -- a per-call cost on runs that compile cleanly and
    never touch the fallback at all.
    """
    probes = {"n": 0}

    def _counting_probe(**_):
        probes["n"] += 1
        return True

    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        _counting_probe)
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", True)

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        lambda x: "compiled", lambda x: "eager", "test-no-reprobe",
    )
    for _ in range(20):
        assert wrapped(1) == "compiled"

    assert "test-no-reprobe" in patch_utils._COMPILED_OK_LABELS, \
        "the observation itself must survive the short-circuit"
    assert probes["n"] == 0, (
        f"the marker was already latched, so the checkpoint probe should never "
        f"have run; it ran {probes['n']} times"
    )


def test_the_success_probe_latches_the_marker_when_it_finds_a_checkpoint(
        monkeypatch):
    """A definite yes from the post-call probe is process-wide news, not local.

    The pre-call probe can miss a region this one then finds -- on a torch with
    no accessor its `_probe_walk` budget may already be spent by earlier
    uncheckpointed calls. Recording only the label left
    `_PACKED_COMPILED_IN_CHECKPOINT` false, so a later refusal by this same
    wrapper OUTSIDE the region read `packed` as false, latched eager, and left
    the compiled activations this call had just packed to be recomputed
    eagerly: a checkpoint abort, or silently wrong gradients.
    """
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", False)
    inside = {"yes": True}
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        lambda **_: inside["yes"])
    calls = {"n": 0}

    def compiled(x):
        calls["n"] += 1
        if calls["n"] == 1:
            return "compiled"
        raise _inductor_error()

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        compiled, lambda x: "eager", "test-latch-marker",
    )

    assert wrapped(1) == "compiled"
    assert patch_utils._PACKED_COMPILED_IN_CHECKPOINT is True, (
        "a definite yes from the post-call probe must latch the process-wide "
        "marker, not only the per-label history"
    )

    # The layer has returned, so the region is closed -- but its compiled pack
    # is still owed a recompute.
    inside["yes"] = False
    with pytest.raises(patch_utils._backend_compile_errors()):
        wrapped(2)


def test_an_unknown_answer_does_not_latch_the_process_wide_marker(monkeypatch):
    """`None` counts as packed for THIS label, but must not speak for others.

    The marker is read by every other wrapper's give-up path, so latching it on
    a guess would end steps that had nothing compiled packed anywhere.
    """
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", False)
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        lambda **_: None)

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        lambda x: "compiled", lambda x: "eager", "test-unknown-no-latch",
    )
    assert wrapped(1) == "compiled"

    assert "test-unknown-no-latch" in patch_utils._COMPILED_OK_LABELS
    assert patch_utils._PACKED_COMPILED_IN_CHECKPOINT is False, \
        "an unknown must not latch the process-wide marker"


def test_a_spent_walk_budget_does_not_invent_checkpoint_history(monkeypatch):
    """An exhausted probe budget must not mark uncheckpointed calls as packed.

    A budgeted post-call probe has to answer something once the budget is gone,
    and both answers are wrong somewhere. Answering None and recording it is the
    dangerous one: on a torch with no hook accessor, a generation-heavy GRPO
    step spends the budget on uncheckpointed inference calls, every later
    success gets recorded as a possible compiled pack, and the first codegen
    refusal inside the training checkpoint is then re-raised as though an
    earlier compiled pack existed -- which ends the run this module exists to
    keep alive. So the probe stays unbudgeted and answers definitely.
    """
    monkeypatch.setattr(patch_utils, "_saved_tensor_hook_accessor",
                        lambda: None)
    # No checkpoint anywhere: the walk always answers a definite False.
    monkeypatch.setattr(patch_utils, "_walk_for_checkpoint_frame",
                        lambda: False)
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", False)
    # The budget is already spent, as it would be after a long generation pass.
    monkeypatch.setattr(patch_utils, "_CHECKPOINT_PROBE_MISSES",
                        patch_utils._CHECKPOINT_PROBE_MISS_BUDGET)

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        lambda x: "compiled", lambda x: "eager", "test-spent-budget",
    )
    for _ in range(5):
        assert wrapped(1) == "compiled"

    assert "test-spent-budget" not in patch_utils._COMPILED_OK_LABELS, (
        "an exhausted budget must not turn definitely-uncheckpointed calls "
        "into checkpoint history"
    )
    assert patch_utils._in_non_reentrant_checkpoint() is False, \
        "the probe must keep giving a definite answer, not None"


def test_releasing_borrowed_budget_keeps_compiled_pack_history(monkeypatch):
    """`_restore_recompile_limits` is not a step boundary, so it must not clear.

    `_release_borrowed_budget` calls it mid-step, after a caught exception, as
    soon as no live wrapper still needs its bump. Clearing the history there
    erases another wrapper's record of having packed compiled under a still-open
    checkpoint, and that wrapper's next refusal then latches eager and lets the
    pack be recomputed eagerly. Only `apply_pending_eager_fallbacks`, the real
    step boundary, may clear it.
    """
    patch_utils._COMPILED_OK_LABELS.add("test-packed-earlier")
    patch_utils._restore_recompile_limits()
    assert "test-packed-earlier" in patch_utils._COMPILED_OK_LABELS, (
        "handing back the recompile budget mid-step erased a wrapper's record "
        "of having packed compiled activations"
    )
    # The genuine step boundary still clears it.
    patch_utils.apply_pending_eager_fallbacks()
    assert "test-packed-earlier" not in patch_utils._COMPILED_OK_LABELS, \
        "the step boundary must still reset the history"

def test_generation_does_not_pay_the_success_probe(monkeypatch):
    """Inference mode saves nothing for backward, so there is nothing to ask.

    This is the O(1) exit that keeps the frame walk off GRPO's generation pass
    on a torch with no hook accessor, where the walk is the only signal.
    Inference mode and NOT `is_grad_enabled()`: `autograd.Function.forward`
    runs with grad off and Unsloth's gradient checkpointing IS one, so grad-off
    is true in the exact place the record has to be made.
    """
    monkeypatch.setattr(patch_utils, "_saved_tensor_hook_accessor",
                        lambda: None)
    walks = {"n": 0}

    def _counting_walk():
        walks["n"] += 1
        return False

    monkeypatch.setattr(patch_utils, "_walk_for_checkpoint_frame",
                        _counting_walk)
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", False)
    monkeypatch.setattr(patch_utils, "_CHECKPOINT_PROBE_MISSES", 0)

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        lambda x: "compiled", lambda x: "eager", "test-generation-free",
    )
    with torch.inference_mode():
        for _ in range(200):
            assert wrapped(1) == "compiled"

    assert walks["n"] == 0, (
        f"generation walked the stack {walks['n']} times for an answer that "
        f"cannot matter: inference mode packs nothing that is recomputed"
    )
    assert "test-generation-free" not in patch_utils._COMPILED_OK_LABELS


def test_a_recorded_label_is_not_probed_again(monkeypatch):
    """The record is idempotent, so the second probe cannot change anything.

    With the marker branch, this is what bounds the success-path cost: a label
    is probed until its answer is known, not once per call forever.
    """
    monkeypatch.setattr(patch_utils, "_PACKED_COMPILED_IN_CHECKPOINT", False)
    probes = {"n": 0}

    def _counting_probe(*_, **__):
        probes["n"] += 1
        return None                      # unknown: records without latching

    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        _counting_probe)

    wrapped = patch_utils._fall_back_to_eager_on_recompile_limit(
        lambda x: "compiled", lambda x: "eager", "test-probe-once",
    )
    for _ in range(50):
        assert wrapped(1) == "compiled"

    assert "test-probe-once" in patch_utils._COMPILED_OK_LABELS
    assert probes["n"] == 1, (
        f"an already-recorded label was probed {probes['n']} times"
    )
