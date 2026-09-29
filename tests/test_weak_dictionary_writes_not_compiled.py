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

Fine-tuning gemma-4-E2B-it on a T4 dies in the second step with "AssertionError:
Something went unexpectedly wrong in activation checkpoint". The exhausted
recompile budget is `weakref.__setitem__`'s -- 1030 compiles against a limit of
1024 in that step -- not the gemma4 RMSNorm kernel the warning names, which
compiles six times in the whole run. Checkpointing's saved-tensor bookkeeping
reaches those writes from inside a compiled region with a fresh key object per
region, so none of those compilations can ever be reused.

Kaggle T4 `danielhanchen/unsloth-t4-ci-e29ca7f0`: with these four code objects
skipped and nothing else changed, the asserting run completes, still compiled.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unsloth_zoo import patching_utils as P  # noqa: E402


def test_every_weak_dictionary_writer_is_marked():
    """All four, as measured on the T4 run."""
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

    Weaker than it looks off a T4: whether Dynamo intercepts these frames at all
    depends on how they are reached, and the failing arrangement (checkpoint
    recompute on an autograd thread under a compiled region) has not been
    reproduced elsewhere. The invariant holds either way.
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
    """A skipped frame runs in the interpreter; the dictionary is unchanged."""
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
    """`patch_torch_compile` is not guaranteed to run only once."""
    assert P.stop_compiling_weak_dictionary_writes() == 4
    assert P.stop_compiling_weak_dictionary_writes() == 4


def test_a_torch_without_the_accessor_is_not_an_error():
    """This runs at import, so a torch that cannot be asked must mean
    "nothing marked", never a failed import."""
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
    """Named through `weakref`, so a python missing one still marks the rest."""
    import weakref
    with pytest.MonkeyPatch.context() as patch:
        patch.delattr(weakref, "WeakValueDictionary")
        assert P.stop_compiling_weak_dictionary_writes() == 2


def test_patching_torch_compile_marks_them():
    """Nothing else calls it, so a `patch_torch_compile` that forgets it is the
    bug back again."""
    called = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(P, "stop_compiling_weak_dictionary_writes",
                      lambda: called.append(True) or 4)
        P.patch_torch_compile()
    assert called == [True]
