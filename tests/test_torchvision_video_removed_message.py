# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Name a half-installed torchvision instead of re-raising its ImportError.

Found by running `Kaggle-Llama3.2_(11B)-Vision`: a venv installed over the base
image left torchvision 0.25.0+cu128 partly overwritten, so `io/__init__.py`
still imports a `video` module that is gone and `import unsloth` dies on a bare
`No module named 'torchvision.io.video'`. Not a version boundary: 0.25 ships
`io/video.py` and 0.26 removed it with its importer, so no release raises this.

These tests drive the real guard, they do not read the source. The source-grep
version of this file passed with the whole feature deleted, because the prose
explaining an arm contains every string the arm does. So each case plants a
`transformers.processing_utils` whose `Unpack` raises the exact error a broken
install raises, re-imports `unsloth_zoo.temporary_patches.utils` in a
subprocess, and reads back what came out.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# (case name, exception class to raise, message)
CASES = [
    ("io_video",      "ImportError",  "No module named 'torchvision.io.video'"),
    ("io_video_priv", "ImportError",  "No module named 'torchvision.io._video'"),
    ("nms_import",    "ImportError",  "operator torchvision::nms does not exist"),
    ("nms_runtime",   "RuntimeError",
     "Failed to register operator torchvision::nms. operator torchvision::nms does not exist"),
    ("unrelated_rt",  "RuntimeError", "cuBLAS workspace allocation failed"),
    ("unrelated_imp", "ImportError",  "No module named 'flash_attn'"),
    ("unpack_moved",  "ImportError",
     "cannot import name 'Unpack' from 'transformers.processing_utils'"),
    ("pillow",        "ImportError",  "cannot import name '_Ink' from 'PIL'"),
    ("numpy_uv",      "ImportError",  "cannot import name '_center' from 'numpy._core.umath'"),
    ("numpy_stale",   "RuntimeError", "numpy.core._multiarray_umath failed to import"),
    # Same substring, module present: a missing SYMBOL, which reinstalling does not fix.
    ("video_symbol",  "ImportError",
     "cannot import name 'read_video' from 'torchvision.io.video'"),
]

_DRIVER = r'''
import importlib, json, sys, types

MOD = "unsloth_zoo.temporary_patches.utils"
cases = json.loads(sys.argv[1])
try:
    importlib.import_module(MOD)      # warm every dependency of the module first
except BaseException as e:
    # This environment cannot import the module at all (no accelerator, no
    # unsloth installed). Nothing to say about the guard here.
    sys.stdout.write("<<<WARMFAIL>>>%s: %s" % (type(e).__name__, e))
    raise SystemExit(0)
import transformers
real = sys.modules.get("transformers.processing_utils")
out = {}
for name, kind, message in cases:
    sys.modules.pop(MOD, None)
    fake = types.ModuleType("transformers.processing_utils")
    def _raise(attr, _m = message, _e = {"ImportError": ImportError,
                                         "RuntimeError": RuntimeError}[kind]):
        raise _e(_m)
    fake.__getattr__ = _raise
    sys.modules["transformers.processing_utils"] = fake
    transformers.processing_utils = fake
    try:
        importlib.import_module(MOD)
        out[name] = {"type": None, "message": None}
    except BaseException as e:
        out[name] = {"type": type(e).__name__, "message": str(e)}
    finally:
        if real is not None:
            sys.modules["transformers.processing_utils"] = real
            transformers.processing_utils = real
sys.stdout.write("<<<RESULTS>>>" + json.dumps(out))
'''

PIP = "pip install --upgrade --force-reinstall --no-cache-dir torchvision"


@pytest.fixture(scope = "module")
def raised():
    """What each planted failure actually produces, from one warm subprocess."""
    pytest.importorskip("transformers")
    # NOT `importorskip` on the module under test: it catches an ImportError
    # raised while INITIALISING the module too, so a real init regression would
    # skip all of these and leave the gate this file is listed in green. Skip
    # only on the two absent-prerequisite guards `unsloth_zoo/__init__.py`
    # raises; anything else is re-raised and fails.
    try:
        importlib.import_module("unsloth_zoo.temporary_patches.utils")
    except BaseException as exc:
        text = f"{type(exc).__name__}: {exc}"
        if "pip install unsloth" in text or "accelerator" in text:
            pytest.skip("prerequisite absent here -- " + text)
        raise
    root = Path(__file__).resolve().parents[1]
    # The child gets the parent's environment, so it imports the same tree under
    # the same accelerator settings; only the path to this checkout is added.
    env = dict(os.environ)
    env.setdefault("UNSLOTH_ALLOW_CPU", "1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER, json.dumps(CASES)],
        capture_output = True, text = True, timeout = 900, cwd = str(root), env = env,
    )
    if "<<<WARMFAIL>>>" in proc.stdout:
        # A FAILURE, not a skip. The parent just imported the module, so a child
        # that cannot is a real initialization regression -- and a skip reads as
        # success to the CI gate this file is listed in, which would leave both
        # handlers unexercised and every required check green.
        pytest.fail("the child could not import what this process already did: "
                    + proc.stdout.split("<<<WARMFAIL>>>", 1)[1][:400])
    if "<<<RESULTS>>>" not in proc.stdout:
        pytest.fail(
            f"driver failed (rc={proc.returncode})\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
        )
    return json.loads(proc.stdout.split("<<<RESULTS>>>", 1)[1])


@pytest.mark.parametrize("case, missing", [
    ("io_video",      "torchvision.io.video"),
    ("io_video_priv", "torchvision.io._video"),
])
def test_a_missing_io_video_is_named_as_an_incomplete_install(raised, case, missing):
    got = raised[case]
    assert got["type"] == "RuntimeError", got
    assert "install is incomplete" in got["message"]
    assert PIP in got["message"]
    # The original text is kept, so the report still shows what was missing.
    assert missing in got["message"]


def test_a_mismatched_torchvision_is_caught_on_the_runtimeerror_arm(raised):
    """`register_fake("torchvision::nms")` raises RuntimeError, not ImportError,
    so the ImportError arm could never fire for it."""
    got = raised["nms_runtime"]
    assert got["type"] == "RuntimeError", got
    assert "reinstall torchvision" in got["message"]
    assert PIP in got["message"]
    # Not the bare error the user used to see.
    assert "does not exist" not in got["message"]


def test_both_arms_give_the_same_instruction(raised):
    assert raised["nms_runtime"]["message"] == raised["nms_import"]["message"]


def test_an_unrelated_runtime_error_is_re_raised_untouched(raised):
    got = raised["unrelated_rt"]
    assert got["type"] == "RuntimeError"
    assert got["message"] == "cuBLAS workspace allocation failed"


def test_an_unrelated_import_error_still_falls_through(raised):
    got = raised["unrelated_imp"]
    assert got["type"] == "Exception", got
    assert "flash_attn" in got["message"]


def test_a_missing_symbol_is_not_called_an_incomplete_install(raised):
    """`cannot import name 'read_video' from 'torchvision.io.video'` holds the
    same substring while the module is right there, so the incomplete-install
    message would be false and its force-reinstall would not fix it."""
    got = raised["video_symbol"]
    assert "install is incomplete" not in (got["message"] or "")
    assert got["type"] == "Exception", got
    assert "read_video" in got["message"]


def test_the_untouched_arms_still_answer(raised):
    """The diagnoses that were already here must be unchanged."""
    assert "Unpack has been moved" in raised["unpack_moved"]["message"]
    assert "Pillow (PIL) version is incompatible" in raised["pillow"]["message"]
    assert "they broke numpy" in raised["numpy_uv"]["message"]
    assert "numpy C extensions cannot be reloaded" in raised["numpy_stale"]["message"]
