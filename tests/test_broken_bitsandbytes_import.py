# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.
"""A bitsandbytes that fails its own import with something other than ImportError
must not take the whole unsloth_zoo import down with it.

A bitsandbytes wheel built against a different torch raises out of its own module
scope, and not as an ImportError:

  bitsandbytes/backends/cuda/ops.py:69
  _get_raw_stream = torch._C._cuda_getCurrentRawStream
  AttributeError: module 'torch._C' has no attribute '_cuda_getCurrentRawStream'

Two module-scope sites let that through. `patching_utils` reaches bitsandbytes second
hand: its dynamic-4bit patch imports `transformers.integrations.bitsandbytes`, and that
transformers module does a bare module-scope `import bitsandbytes as bnb` of its own.
`vllm_utils` reaches it directly, behind an `importlib.util.find_spec` gate that only
proves the wheel is on disk. Either one put the AttributeError on the `import unsloth`
path, breaking 16bit and full finetuning on a host whose only real problem was a 4bit dep.

Both have a no-bnb path already, and that is where a broken wheel now goes. The healthy
path is pinned here too, so a guard cannot buy the fix by turning 4bit off for everybody.

The dep is faked with a one-line module on `sys.path`, never a real wheel and never the
network, and every case runs in a fresh interpreter so the poisoned `bitsandbytes` cannot
leak into the rest of the suite.
"""

import os
import pathlib
import subprocess
import sys

import pytest


_ROOT = pathlib.Path(__file__).resolve().parents[1]

# The message the mismatched wheel actually dies with, reproduced verbatim so a
# guard narrowed back to ImportError fails here rather than passing vacuously.
_BREAKAGE = "module 'torch._C' has no attribute '_cuda_getCurrentRawStream'"

_STUB = f'raise AttributeError("{_BREAKAGE}")\n'


def _run(code, *, poisoned):
    """Run `code` in a fresh interpreter, optionally with the broken dep first
    on the path. The repo root goes on PYTHONPATH so the child imports the tree
    under test rather than an installed unsloth_zoo."""
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
    """Without this, every other case in the file could pass on a host that
    simply resolved the real bitsandbytes and never saw the stub at all."""
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
    """The regression. `patching_utils` is on the path `import unsloth` walks,
    via `unsloth/models/_utils.py`, so this failing takes the whole import down."""
    out = _run(
        "import sys, importlib\n"
        # Establish that transformers really does try to load the poisoned dep on
        # this version. Where it does not, patching_utils was never at risk and
        # the case has nothing to prove.
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
    # The no-bnb path: the dynamic-4bit patch needs the source of a function it
    # cannot reach, so it is skipped rather than half-applied.
    assert "PATCH_APPLIED False" in out.stdout, out.stdout
    assert "INTEGRATION_LOADED False" in out.stdout, out.stdout


def test_vllm_utils_takes_the_no_bnb_path_when_bitsandbytes_cannot_import(poisoned_bitsandbytes):
    """`vllm_utils` gated its bitsandbytes block on `importlib.util.find_spec`, which only
    proves the wheel is on disk. A mismatched wheel is on disk, so the gate opened and the
    module-scope `import bitsandbytes.functional` behind it raised. The `else:` branch of
    that same gate already defines no-op stand-ins, so there was a no-bnb path to take.

    peft carries an `except ImportError` guard of its own and dies first on a genuinely
    broken wheel. That is upstream and not what this pins, so the child loads peft while
    the real wheel is still reachable and only then puts the stub on the path.
    """
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
        # The no-op stand-in is a bare `return` and so touches no globals at all; the
        # real one reassigns bitsandbytes.functional.QuantState.from_dict and Linear4bit.
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
    # The gate has to close, not half-open: Linear4bit subclasses a bitsandbytes class
    # that is not there, and the callers of the patch helpers get the no-op stand-ins.
    assert "LINEAR4BIT_DEFINED False" in out.stdout, out.stdout
    assert "HELPER_TOUCHES_BNB False" in out.stdout, out.stdout
    # The stand-in has to survive being called, which is what callers at vllm_utils.py:803
    # and :3629 do unconditionally.
    assert "HELPER_CALLABLE_OK" in out.stdout, out.stdout


def test_a_healthy_bitsandbytes_still_gets_the_dynamic_4bit_patch():
    """The guard degrades only when the dep is genuinely unusable. With the real
    wheel installed the patch still has to land, or the guard has quietly turned
    dynamic 4bit off for everybody."""
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
        # transformers 5.x dropped _replace_with_bnb_linear; the should_convert_module
        # patch covers that branch instead and is not what this case pins.
        pytest.skip("transformers has no _replace_with_bnb_linear to patch")
    assert "PATCH_APPLIED True" in out.stdout, out.stdout
    assert "HOOKED _unsloth_replace_with_bnb_linear" in out.stdout, out.stdout


def test_a_healthy_bitsandbytes_still_opens_the_vllm_utils_gate():
    """The other half of the same bargain: swapping `find_spec` for a real import probe
    must not close the gate on a host where bitsandbytes works."""
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
