# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.
"""A bitsandbytes that raises AttributeError on import (mismatched torch wheel) must not break import unsloth_zoo;
patching_utils and vllm_utils take the no-bnb path, healthy bnb still patched; each case runs in a fresh interpreter."""

import os
import pathlib
import subprocess
import sys

import pytest


_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Verbatim AttributeError from a mismatched wheel, so a guard narrowed back to ImportError fails here instead of passing vacuously.
_BREAKAGE = "module 'torch._C' has no attribute '_cuda_getCurrentRawStream'"

_STUB = f'raise AttributeError("{_BREAKAGE}")\n'


def _run(code, *, poisoned):
    """Runs code in a fresh interpreter, optionally with the broken dep first on sys.path; repo root goes on PYTHONPATH
    so the tree under test is imported, not an installed unsloth_zoo."""
    path = [str(_ROOT)]
    if poisoned:
        path.insert(0, str(poisoned))
    if os.environ.get("PYTHONPATH"):
        path.append(os.environ["PYTHONPATH"])
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output = True,
        text = True,
        env = dict(
            os.environ,
            PYTHONPATH = os.pathsep.join(path),
            UNSLOTH_ALLOW_CPU = "1",
        ),
        timeout = 900,
    )


@pytest.fixture
def poisoned_bitsandbytes(tmp_path):
    (tmp_path / "bitsandbytes.py").write_text(_STUB)
    return tmp_path


def test_the_stub_really_breaks_the_import(poisoned_bitsandbytes):
    """Without this, other cases could pass on a host that resolved the real bitsandbytes and never saw the stub at all."""
    out = _run(
        "try:\n"
        "    import bitsandbytes\n"
        "except AttributeError as e:\n"
        "    print('RAISED', type(e).__name__, e)\n"
        "else:\n"
        "    print('NOT_RAISED', bitsandbytes.__file__)\n",
        poisoned = poisoned_bitsandbytes,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert "RAISED AttributeError" in out.stdout, out.stdout
    assert _BREAKAGE in out.stdout, out.stdout


def test_patching_utils_imports_when_bitsandbytes_cannot(poisoned_bitsandbytes):
    """Regression: patching_utils is on the path import unsloth walks via unsloth/models/_utils.py, so its failure takes the whole import down."""
    out = _run(
        "import sys, importlib\n"
        # Precondition: transformers must really import the poisoned bitsandbytes on this version, or patching_utils was
        # never at risk and the case proves nothing.
        "try:\n"
        "    importlib.import_module('transformers.integrations.bitsandbytes')\n"
        "except AttributeError:\n"
        "    print('PRECONDITION poisoned')\n"
        "except Exception as e:\n"
        "    print('PRECONDITION other', type(e).__name__)\n"
        "else:\n"
        "    print('PRECONDITION unreached')\n"
        "import unsloth_zoo.patching_utils as p\n"
        "print('IMPORT_OK')\n"
        "print('PATCH_APPLIED', '_unsloth_replace_with_bnb_linear' in vars(p))\n"
        "print('INTEGRATION_LOADED', 'transformers.integrations.bitsandbytes' in sys.modules)\n",
        poisoned = poisoned_bitsandbytes,
    )
    if "PRECONDITION unreached" in out.stdout:
        pytest.skip("transformers did not import bitsandbytes, so nothing could break")
    assert "PRECONDITION poisoned" in out.stdout, out.stdout
    assert out.returncode == 0, out.stderr[-3000:]
    assert "IMPORT_OK" in out.stdout, out.stdout
    # The AttributeError must not merely be caught somewhere and re-raised later.
    assert _BREAKAGE not in out.stderr, out.stderr[-3000:]
    # On the no-bnb path the dynamic-4bit patch needs source of a function it cannot reach, so it is skipped rather than half-applied.
    assert "PATCH_APPLIED False" in out.stdout, out.stdout
    assert "INTEGRATION_LOADED False" in out.stdout, out.stdout


def test_vllm_utils_takes_the_no_bnb_path_when_bitsandbytes_cannot_import(poisoned_bitsandbytes):
    """vllm_utils gated bitsandbytes on find_spec, which only proves the wheel is on disk, so a mismatched wheel passed
    and raised; its else branch no-op stand-ins are the fallback. peft is preloaded before the stub since it dies
    first on a broken wheel, upstream."""
    out = _run(
        "import importlib, sys\n"
        "import peft\n"
        "for _m in [m for m in sys.modules if m == 'bitsandbytes' or m.startswith('bitsandbytes.')]:\n"
        "    del sys.modules[_m]\n"
        f"sys.path.insert(0, {str(poisoned_bitsandbytes)!r})\n"
        "importlib.invalidate_caches()\n"
        "try:\n"
        "    import bitsandbytes\n"
        "except AttributeError:\n"
        "    print('PRECONDITION poisoned')\n"
        "else:\n"
        "    print('PRECONDITION unreached')\n"
        "for _m in [m for m in sys.modules if m == 'bitsandbytes' or m.startswith('bitsandbytes.')]:\n"
        "    del sys.modules[_m]\n"
        "import unsloth_zoo.vllm_utils as v\n"
        "print('IMPORT_OK')\n"
        "print('LINEAR4BIT_DEFINED', hasattr(v, 'Linear4bit'))\n"
        # The no-op stand-in is a bare return touching no globals; the real one reassigns bitsandbytes.functional.QuantState.from_dict and Linear4bit.
        "print('HELPER_TOUCHES_BNB', bool(v.patch_bitsandbytes_quant_state.__code__.co_names))\n"
        "v.patch_bitsandbytes_quant_state()\n"
        "print('HELPER_CALLABLE_OK')\n",
        poisoned = None,
    )
    if "PRECONDITION unreached" in out.stdout:
        pytest.skip("the stub did not take over bitsandbytes, so nothing could break")
    assert out.returncode == 0, out.stderr[-3000:]
    assert "IMPORT_OK" in out.stdout, out.stdout
    assert _BREAKAGE not in out.stderr, out.stderr[-3000:]
    # The gate must close rather than half-open: Linear4bit subclasses a missing bitsandbytes class, so patch helper callers get the no-op stand-ins.
    assert "LINEAR4BIT_DEFINED False" in out.stdout, out.stdout
    assert "HELPER_TOUCHES_BNB False" in out.stdout, out.stdout
    # The stand-in must survive being called, since callers at vllm_utils.py:803 and :3629 invoke it unconditionally.
    assert "HELPER_CALLABLE_OK" in out.stdout, out.stdout


def test_a_healthy_bitsandbytes_still_gets_the_dynamic_4bit_patch():
    """The guard must degrade only when bitsandbytes is genuinely unusable; with a working wheel the dynamic 4bit patch must still land."""
    out = _run(
        "import unsloth_zoo.patching_utils as p\n"
        "import transformers.integrations.bitsandbytes as tib\n"
        "print('PATCH_APPLIED', '_unsloth_replace_with_bnb_linear' in vars(p))\n"
        "print('HOOKED', getattr(getattr(tib, '_replace_with_bnb_linear', None),"
        " '__name__', 'ABSENT'))\n",
        poisoned = None,
    )
    if out.returncode != 0:
        pytest.skip("no usable bitsandbytes on this host: " + out.stderr[-500:])
    if "HOOKED ABSENT" in out.stdout:
        # transformers 5.x dropped _replace_with_bnb_linear; the should_convert_module patch covers that branch instead, not this case.
        pytest.skip("transformers has no _replace_with_bnb_linear to patch")
    assert "PATCH_APPLIED True" in out.stdout, out.stdout
    assert "HOOKED _unsloth_replace_with_bnb_linear" in out.stdout, out.stdout


def test_a_healthy_bitsandbytes_still_opens_the_vllm_utils_gate():
    """Replacing find_spec with a real import probe must not close the vllm_utils gate on a host where bitsandbytes works."""
    out = _run(
        "import unsloth_zoo.vllm_utils as v\n"
        "import bitsandbytes.nn.modules\n"
        "print('LINEAR4BIT_DEFINED', hasattr(v, 'Linear4bit'))\n"
        "print('LINEAR4BIT_BASE', v.Linear4bit.__mro__[1].__module__)\n"
        "print('HELPER_TOUCHES_BNB', bool(v.patch_bitsandbytes_quant_state.__code__.co_names))\n"
        "v.patch_bitsandbytes_quant_state()\n"
        "print('QUANT_STATE_PATCHED', bitsandbytes.nn.modules.Linear4bit is v.Linear4bit)\n",
        poisoned = None,
    )
    if out.returncode != 0:
        pytest.skip("no usable bitsandbytes on this host: " + out.stderr[-500:])
    assert "LINEAR4BIT_DEFINED True" in out.stdout, out.stdout
    assert "LINEAR4BIT_BASE bitsandbytes.nn.modules" in out.stdout, out.stdout
    assert "HELPER_TOUCHES_BNB True" in out.stdout, out.stdout
    # The gate being open is only worth anything if the patch behind it still lands.
    assert "QUANT_STATE_PATCHED True" in out.stdout, out.stdout
