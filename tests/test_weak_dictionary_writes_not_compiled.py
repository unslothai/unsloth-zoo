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

"""Dynamo must not compile the standard library's weak-dictionary writes.

Fine-tuning gemma-4-E2B-it on a T4 dies in the second step with

    AssertionError: Something went unexpectedly wrong in activation checkpoint

and the recompile budget is why, but not the way it looks. The budget that runs
out belongs to `weakref.__setitem__`, compiled 1030 times in that one step
against a limit of 1024, while the gemma4 RMSNorm kernel Unsloth does compile
was compiled six times in the whole run. Activation checkpointing's saved-tensor
bookkeeping reaches those writes from inside a compiled region and hands Dynamo
a fresh key object per checkpointed region, so every one of them is a
compilation that can never be reused.

Kaggle T4 `danielhanchen/unsloth-t4-ci-e29ca7f0`: with these four code objects
marked skipped and nothing else changed, the run that had been asserting
completes, still compiled.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unsloth_zoo import patching_utils as P  # noqa: E402


def test_every_weak_dictionary_writer_is_marked():
    """All four, because all four are what the T4 run was measured with."""
    marked = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("torch._dynamo.eval_frame.skip_code",
                      lambda code: marked.append(code))
        count = P.stop_compiling_weak_dictionary_writes()
    import weakref
    expected = {
        weakref.WeakKeyDictionary.__setitem__.__code__,
        weakref.WeakKeyDictionary.__delitem__.__code__,
        weakref.WeakValueDictionary.__setitem__.__code__,
        weakref.WeakValueDictionary.__delitem__.__code__,
    }
    assert count == 4
    assert set(marked) == expected


def test_the_marked_frames_are_never_compiled():
    """No Dynamo cache entry for any of the four, through the real accessor.

    This is a weaker statement than it looks anywhere but a T4. Whether Dynamo
    intercepts these frames at all depends on how they are reached -- on the
    failing kernel it is from the saved-tensor bookkeeping of a checkpoint
    recompute, on an autograd worker thread, with a compiled region on the
    stack, and that arrangement has not been reproduced anywhere else. What this
    pins is the invariant either way: after the mark, nothing compiles them.
    """
    import weakref
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list

    class Key:
        pass

    assert P.stop_compiling_weak_dictionary_writes() == 4
    box, keys = weakref.WeakKeyDictionary(), [Key() for _ in range(3)]
    compiled = torch.compile(weakref.WeakKeyDictionary.__setitem__,
                             backend = "eager")
    for index, key in enumerate(keys):
        compiled(box, key, index)
    assert len(box) == 3, "the skipped frame did not run"
    for owner_name, method_name in P._WEAK_DICTIONARY_WRITERS:
        code = getattr(getattr(weakref, owner_name), method_name).__code__
        assert _debug_get_cache_entry_list(code) == []


def test_a_skipped_write_still_works():
    """A skipped frame runs in the interpreter, which is what it did before
    anyone thought to compile it. Nothing about the dictionary changes."""
    import weakref

    class Key:
        pass

    P.stop_compiling_weak_dictionary_writes()
    kept, box = Key(), weakref.WeakKeyDictionary()
    box[kept] = "value"
    assert box[kept] == "value"
    dropped = Key()
    box[dropped] = "gone"
    del box[dropped]
    assert len(box) == 1


def test_it_is_safe_to_call_twice():
    """`patch_torch_compile` is not guaranteed to run once, and marking an
    already-marked code object must not be an error."""
    assert P.stop_compiling_weak_dictionary_writes() == 4
    assert P.stop_compiling_weak_dictionary_writes() == 4


def test_a_torch_without_the_accessor_is_not_an_error():
    """This runs from `patch_torch_compile`, at import. A torch that cannot be
    asked has to mean "nothing marked", never a failed import."""
    with pytest.MonkeyPatch.context() as patch:
        patch.delattr("torch._dynamo.eval_frame.skip_code")
        assert P.stop_compiling_weak_dictionary_writes() == 0


def test_one_refusal_does_not_stop_the_others():
    refused = []

    def flaky(code):
        if not refused:
            refused.append(code)
            raise RuntimeError("no")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("torch._dynamo.eval_frame.skip_code", flaky)
        assert P.stop_compiling_weak_dictionary_writes() == 3


def test_a_missing_weak_dictionary_is_skipped_quietly():
    """Named through `weakref` rather than imported, so a python that has moved
    one of them leaves the other three marked."""
    import weakref
    with pytest.MonkeyPatch.context() as patch:
        patch.delattr(weakref, "WeakValueDictionary")
        assert P.stop_compiling_weak_dictionary_writes() == 2


def test_patching_torch_compile_marks_them():
    """The wiring, not just the function: nothing else calls it, so a
    `patch_torch_compile` that forgets it is the bug back again."""
    called = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(P, "stop_compiling_weak_dictionary_writes",
                      lambda: called.append(True) or 4)
        P.patch_torch_compile()
    assert called == [True]
