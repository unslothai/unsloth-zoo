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

"""A frame that keeps a caught exception while sitting on that exception's own traceback is a reference cycle: every
tensor on the failed call stack then lives until the cycle collector runs instead of being freed by refcount."""
import contextlib
import gc
import warnings
import weakref

import pytest
import torch
from torch.utils import checkpoint

import unsloth_zoo.fused_losses.fused_linear as FL
import unsloth_zoo.temporary_patches.moe_triton_kernels as K
import unsloth_zoo.temporary_patches.utils as U

_STOP = getattr(checkpoint, "_StopRecomputationError", None)


@contextlib.contextmanager
def _refcount_only():
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


def _big(refs):
    t = torch.empty(1 << 20)
    refs.append(weakref.ref(t))
    return t


def _assert_freed(refs):
    assert refs, "the failing call never ran"
    alive = sum(r() is not None for r in refs)
    assert alive == 0, f"{alive} tensor(s) from the failed call kept alive by a traceback cycle"


@pytest.mark.skipif(_STOP is None, reason = "this torch has no checkpoint early stop")
def test_moe_triton_disable_reraise_frees_tensors(monkeypatch):
    monkeypatch.setattr(K, "_DISABLED_REASON", None)
    monkeypatch.setattr(K, "moe_triton_kernels_available", lambda device = None: True)
    refs = []

    def stop(*args):
        big = _big(refs)  # noqa: F841
        raise _STOP()
    monkeypatch.setattr(K._WeightedUnpermute, "apply", stop)
    with _refcount_only():
        try:
            K.weighted_unpermute(torch.randn(8, 4), torch.arange(8), torch.rand(8), 4, 2)
        except _STOP:
            pass
        else:
            pytest.fail("the early stop signal must propagate")
        _assert_freed(refs)


@contextlib.contextmanager
def _give_up_raises(monkeypatch, label):
    """Make the eager-fallback wrapper's give-up path re-raise: no budget to borrow, and a compiled pack this step."""
    monkeypatch.setattr(U, "_bump_recompile_limits", lambda: False)
    monkeypatch.setattr(U, "_PACKED_COMPILED_IN_CHECKPOINT", True)
    monkeypatch.setattr(U, "_RAISED_INSIDE_CHECKPOINT", False)
    monkeypatch.setattr(U, "_EAGER_FALLBACK_WRAPPERS", [])
    monkeypatch.setattr(U, "_wants_hard_recompile_failure", lambda: False)
    monkeypatch.setattr(U, "_wants_hard_backend_failure", lambda: False)
    monkeypatch.setattr(U, "_note_packed_under_checkpoint", lambda: None)
    try:
        yield
    finally:
        for labels in (U._LATCHED_EAGER_LABELS, U._RECENT_EAGER_LABELS, U._PENDING_EAGER_LABELS, U._COMPILED_OK_LABELS):
            labels.discard(label)


def _run_wrapper(wrapper, refs, expected):
    with _refcount_only():
        try:
            wrapper(torch.randn(4))
        except expected:
            pass
        else:
            pytest.fail("a give-up inside a compiled checkpoint pack must end the step")
        _assert_freed(refs)


def test_give_up_on_recompile_limit_frees_tensors(monkeypatch):
    errors = U._recompile_limit_errors()
    if not errors:
        pytest.skip("this torch exposes no Dynamo recompile-limit exception")
    label = "test_give_up_on_recompile_limit_frees_tensors"
    refs = []

    def compiled(x):
        big = _big(refs)  # noqa: F841
        raise errors[0]("recompile limit")
    with _give_up_raises(monkeypatch, label):
        wrapper = U._fall_back_to_eager_on_recompile_limit(compiled, lambda x: x, label)
        _run_wrapper(wrapper, refs, errors)


def test_give_up_on_backend_frees_tensors(monkeypatch):
    try:
        from torch._inductor.exc import InductorError
    except ImportError:
        pytest.skip("this torch has no InductorError")
    if InductorError not in U._backend_compile_errors():
        pytest.skip("InductorError is not a handled backend refusal on this torch")
    label = "test_give_up_on_backend_frees_tensors"
    refs = []

    def compiled(x):
        big = _big(refs)  # noqa: F841
        raise InductorError(RuntimeError("CantSplit"), "")
    with _give_up_raises(monkeypatch, label):
        U._COMPILED_OK_LABELS.add(label)
        wrapper = U._fall_back_to_eager_on_recompile_limit(compiled, lambda x: x, label)
        _run_wrapper(wrapper, refs, InductorError)


def test_fused_linear_guarded_fallback_frees_tensors(monkeypatch):
    monkeypatch.setattr(FL, "_use_triton", lambda device: True)
    refs = []
    calls = {"n": 0}

    def body(hidden, weight):
        calls["n"] += 1
        big = _big(refs)  # noqa: F841
        if calls["n"] <= 2:
            raise RuntimeError("triton launch failed")
        return hidden @ weight.t()
    guarded = FL._guarded(body)
    hidden, weight = torch.randn(2, 4), torch.randn(3, 4)
    try:
        with _refcount_only(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = guarded(hidden, weight)
            torch.testing.assert_close(out, hidden @ weight.t())
            _assert_freed(refs)
    finally:
        FL._TRITON_BROKEN.discard(weight.device)


def test_fused_linear_scale_rows_fallback_frees_tensors(monkeypatch):
    monkeypatch.setattr(FL, "_use_triton", lambda device: True)
    refs = []

    def launch(device, fn):
        big = _big(refs)  # noqa: F841
        raise RuntimeError("triton launch failed")
    monkeypatch.setattr(FL, "_launch", launch)
    u, g = torch.randn(4, 3), torch.rand(4)
    try:
        with _refcount_only(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = FL._scale_rows_impl(u, g, torch.float32)
            torch.testing.assert_close(out, u * g[:, None])
            _assert_freed(refs)
    finally:
        FL._TRITON_BROKEN.discard(u.device)
