# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Name a half-installed torchvision instead of re-raising its ImportError.

Found by running `Kaggle-Llama3.2_(11B)-Vision`: a venv installed over the base
image left torchvision 0.25.0+cu128 partly overwritten, so `io/__init__.py`
still imports a `video` module that is gone and `import unsloth` dies on a bare
`No module named 'torchvision.io.video'`. Not a version boundary: 0.25 ships
`io/video.py` and 0.26 removed it with its importer, so no release raises this.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

SOURCE = (Path(__file__).resolve().parents[1]
          / "unsloth_zoo" / "temporary_patches" / "utils.py").read_text(encoding = "utf-8")


def _handler_body():
    """The ImportError arm that classifies a failed `Unpack` import."""
    start = SOURCE.index("    from transformers.processing_utils import Unpack")
    end = SOURCE.index("except Exception as e:", start)
    return SOURCE[start:end]


def test_the_removed_video_module_is_recognised():
    body = _handler_body()
    assert "torchvision.io.video" in body, \
        "a torchvision without io.video still falls through to a bare Exception"
    # Must precede the catch-all, which would otherwise swallow it.
    assert body.index("torchvision.io.video") < body.index('elif "Unpack" not in e')


@pytest.mark.parametrize("message", [
    "No module named 'torchvision.io.video'",
    "No module named 'torchvision.io._video'",
])
def test_the_message_is_actionable_and_names_the_cause(message):
    """A RuntimeError a user can act on, not the original ImportError text."""
    arm = _handler_body()
    assert "install is incomplete" in arm, "the message does not name the cause"
    assert "force-reinstall --no-cache-dir torchvision" in arm, \
        "the message does not say what to run"
    # Substring test, so both spellings must reach the arm.
    assert re.search(r'"torchvision\.io\.video" in e or "torchvision\.io\._video" in e', arm)
    assert message.split("'")[1].rsplit(".", 1)[0] in ("torchvision.io", "torchvision")


def test_an_unrelated_import_error_still_falls_through():
    """The new arm must not swallow errors it does not explain."""
    arm = _handler_body()
    assert 'elif "Unpack" not in e' in arm, "the catch-all arm was removed"
    assert "raise Exception(e)" in arm, "unrecognised errors no longer surface"


def test_every_named_arm_raises_runtime_error():
    """RuntimeError, not Exception: the caller tells a diagnosis it can show the
    user from an error it could not classify."""
    arm = _handler_body()
    # Sliced to the next branch, not a fixed window, or a comment in an arm
    # pushes its `raise` out of view and the test passes blind.
    bounds = [m.start() for m in re.finditer(r"\n    (?:elif |raise )", arm)] + [len(arm)]
    for kind in ("numpy._core.umath", "torchvision::nms", "torchvision.io.video", "PIL"):
        i = arm.index(kind)
        end = next(b for b in bounds if b > i)
        assert "RuntimeError" in arm[i:end], f"{kind} does not raise RuntimeError"


def _fallback_arm():
    """The `except Exception` arm, which sees what the ImportError arm cannot.

    Anchored after the Unpack handler, since the file has earlier
    `except Exception` blocks that have nothing to do with this.
    """
    anchor = SOURCE.index("    from transformers.processing_utils import Unpack")
    start = SOURCE.index("except Exception as e:", anchor)
    return SOURCE[start:SOURCE.index("KWARGS_TYPE", start)]


def test_the_nms_break_is_caught_where_it_actually_lands():
    """A torchvision whose ops do not match torch fails inside
    `_meta_registrations` at `register_fake("torchvision::nms")`, which raises
    RuntimeError. The ImportError arm can never see it."""
    arm = _fallback_arm()
    assert "torchvision::nms does not exist" in arm, (
        "the nms check exists only on the ImportError arm, where this error never arrives"
    )
    # And it must be diagnosed before the bare re-raise at the end swallows it.
    assert arm.index("torchvision::nms") < arm.rindex("raise")


def test_both_arms_give_the_same_instruction():
    """One message, so the two arms cannot drift apart."""
    assert SOURCE.count("_TORCHVISION_BROKE") >= 3, "the shared message is not used by both arms"
    assert "force-reinstall --no-cache-dir torchvision" in SOURCE


def test_an_unrelated_runtime_error_still_surfaces():
    arm = _fallback_arm()
    assert "raise" in arm.split("torchvision::nms")[-1], (
        "unrecognised errors no longer reach the bare re-raise"
    )
