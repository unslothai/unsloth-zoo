# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""torchvision retired `torchvision.io.video`; say so instead of re-raising it.

Found by running the Kaggle notebook variants: `Kaggle-Llama3.2_(11B)-Vision`
fails where its Colab twin passes, because the Kaggle install cell takes an
unpinned torchvision off the cu128 index while Colab keeps the preinstalled one.
The user sees `Exception: No module named 'torchvision.io.video'` raised from
`import unsloth`, with nothing pointing at the real cause.
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
    # Before the named arms is wrong: the catch-all would swallow it first.
    assert body.index("torchvision.io.video") < body.index('elif "Unpack" not in e')


@pytest.mark.parametrize("message", [
    "No module named 'torchvision.io.video'",
    "No module named 'torchvision.io._video'",
])
def test_the_message_is_actionable_and_names_the_cause(message):
    """A RuntimeError a user can act on, not the original ImportError text."""
    arm = _handler_body()
    assert "torchcodec" in arm, "the message does not say where the module went"
    assert "upgrade transformers" in arm, "the message does not say what to run"
    # The classifier is a substring test, so both spellings must reach the arm.
    assert re.search(r'"torchvision\.io\.video" in e or "torchvision\.io\._video" in e', arm)
    assert message.split("'")[1].rsplit(".", 1)[0] in ("torchvision.io", "torchvision")


def test_an_unrelated_import_error_still_falls_through():
    """The new arm must not swallow errors it does not explain."""
    arm = _handler_body()
    assert 'elif "Unpack" not in e' in arm, "the catch-all arm was removed"
    assert "raise Exception(e)" in arm, "unrecognised errors no longer surface"


def test_every_named_arm_raises_runtime_error():
    """RuntimeError, not Exception: the caller distinguishes a diagnosis it can
    show the user from an error it could not classify."""
    arm = _handler_body()
    # Sliced to the next branch, not a fixed window: a comment inside an arm
    # would otherwise push its `raise` out of view and pass the test blind.
    bounds = [m.start() for m in re.finditer(r"\n    (?:elif |raise )", arm)] + [len(arm)]
    for kind in ("numpy._core.umath", "torchvision::nms", "torchvision.io.video", "PIL"):
        i = arm.index(kind)
        end = next(b for b in bounds if b > i)
        assert "RuntimeError" in arm[i:end], f"{kind} does not raise RuntimeError"
