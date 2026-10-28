# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Inductor codegen refusal (torch 2.12 CantSplit on torch.chunk ragged tail under dynamic shapes; 2.9 to 2.11 answer
correctly) must fall back to eager in _fall_back_to_eager_on_recompile_limit, since a simplifier gap should cost
speed, not kill a GRPO run"""

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
    """BackendCompilerFailed stores backend_fn's __name__ as backend_name; torch passes OutputGraph.compiler_fn, named
    inductor (verified on torch 2.9.1 by raising a bare RuntimeError from compile_fx)."""
    return gm
_inductor_backend_fn.__name__ = "inductor"


def _inductor_error(message = "CantSplit: 202048*s47*s87 not divisible by s47*s87"):
    """Signatures differ by torch (InductorError in 2.12; BackendCompilerFailed takes an extra frame arg from 2.9), so
    try shapes and require an Inductor identity; on 2.6 the old (inner, None) build gives backend_name ? and
    inner_exception None, which is rejected."""
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
    """Faithful copy of torch 2.6.0 BackendCompilerFailed: 2.6 has no InductorError, so the wrapped form is the only
    form the fallback sees there; newer torch needs this stand-in."""
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
# Under a non-reentrant checkpoint _give_up must re-raise, as it latches every borrower; a first-call codegen refusal
# still falls back, having packed nothing compiled.


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
    """A refusal after a successful compiled call inside the checkpoint must still raise, since recomputing that pack
    eagerly aborts backward or gives wrong gradients."""
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
# Four holes in the checkpoint gate above: the wrapper either re-raised when the fallback was safe or fell back when it was not.


def test_a_refusal_in_a_later_step_falls_back(monkeypatch):
    """compiled_ok must reset per step: held per wrapper it stays true forever, so a refusal in a later step would
    re-raise and defeat the fallback for long GRPO runs."""
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
    """A compiled budget retry must be recorded as packed under a checkpoint, since it skips the else recording; the
    eager latch makes that divergence unreachable through this path today."""
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
    """After a refusal with nothing compiled, the process-wide marker must be restored; left set, an unrelated wrapper
    exhausting its budget in the same checkpoint ends the step."""
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
    """BackendCompilerFailed also wraps custom backends; swallowing a custom backend's failure would hide a config or
    programming error behind a silent permanent switch to eager."""
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
    """Restoring the prior marker value, not False, keeps an earlier region's compiled pack visible to _give_up;
    clearing it would return wrong gradients rather than just end a step."""
    monkeypatch.setattr(patch_utils, "_in_non_reentrant_checkpoint",
                        lambda **_: True)
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
# Five more holes of the same shape: a path where the wrapper either re-raises where fallback was safe, or never sees the refusal at all.


def test_the_wrapped_form_is_recognised_by_its_recorded_backend_name():
    """BackendCompilerFailed records backend_name and never backend; reading the missing attribute made the wrapped-form branch dead, so it re-raised anyway."""
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
    """Drives the real wrapper with the torch 2.6-shaped exception: without the backend_name read it re-raises, so the fallback would not exist at all on 2.6."""
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
    """A refusal inside _retry_with_more_budget must fall back: its except BaseException re-raises past the sibling
    backend_errors handler, so the refusal escaped the fallback."""
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
    """An eager result from the budget retry must not be recorded as a compiled pack, or a later step ends for no reason
    and a checkpoint recomputes in the wrong mode."""
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
    """On torch 2.4 the retry arrives through the graph-break Unsupported arm, matched by message; that arm must record the retry as compiled too, as 2.7+ does."""
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
    """A compiled call outside a non-reentrant checkpoint packs nothing recomputed, so it must not count as a compiled
    pack, or a training refusal would end the step."""
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
    """An unknown (None) checkpoint answer counts as packed, since under-recording gives wrong gradients; before torch
    2.8 there is no hook accessor, and user hooks can sit above ours."""
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
    """A backend refusal must be recorded in _RECENT_EAGER_LABELS, which force_eager_fallback reads; not the permanent
    _LATCHED_EAGER_LABELS (a discarded model would answer) nor live wrappers (GRPO collects them before backward, so
    the query returns 0 and the retry re-raises)"""
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
    """The builder must not yield (inner, None) on 2.6: that gives backend_name ? and inner_exception None, which the predicate correctly rejects."""
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
# The post-call _note_compiled_ok probe runs on the success path, so runs that never fall back still pay its cost; one
# marker latch closes both that cost and a correctness hole.


def test_a_successful_call_does_not_reprobe_once_the_marker_has_latched(
        monkeypatch):
    """Test the cheap _PACKED_COMPILED_IN_CHECKPOINT marker before the frame walk: otherwise the walk is paid on every
    successful compiled call, even after the marker latched."""
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
    """A definite yes from the post-call probe must latch the process-wide _PACKED_COMPILED_IN_CHECKPOINT, not just the
    label; otherwise a later refusal recomputes compiled packs eagerly."""
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

    # The layer has returned so the region is closed, but its compiled pack still owes a recompute.
    inside["yes"] = False
    with pytest.raises(patch_utils._backend_compile_errors()):
        wrapped(2)


def test_an_unknown_answer_does_not_latch_the_process_wide_marker(monkeypatch):
    """None counts as packed for this label only; latching the process-wide marker on a guess would end steps that had nothing compiled packed anywhere."""
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
    """The post-call probe stays unbudgeted and answers definitely: recording None once the walk budget is spent would
    make a later checkpoint refusal re-raise as if a pack existed."""
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
    """_restore_recompile_limits runs mid-step, so it must not clear _COMPILED_OK_LABELS; only
    apply_pending_eager_fallbacks, the real step boundary, may clear that history."""
    patch_utils._COMPILED_OK_LABELS.add("test-packed-earlier")
    patch_utils._restore_recompile_limits()
    assert "test-packed-earlier" in patch_utils._COMPILED_OK_LABELS, (
        "handing back the recompile budget mid-step erased a wrapper's record "
        "of having packed compiled activations"
    )
    patch_utils.apply_pending_eager_fallbacks()
    assert "test-packed-earlier" not in patch_utils._COMPILED_OK_LABELS, \
        "the step boundary must still reset the history"

def test_generation_does_not_pay_the_success_probe(monkeypatch):
    """Inference mode saves nothing for backward, so it skips the frame walk; test inference mode rather than
    is_grad_enabled(), since checkpointed Function.forward runs with grad off."""
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
    """The record is idempotent, and a label is probed only until its answer is known, not on every call."""
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
