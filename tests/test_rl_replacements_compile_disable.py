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

"""UNSLOTH_COMPILE_DISABLE did not reach the GRPO helpers.

Three functions in `rl_replacements.py` carried a bare `@torch.compile(...)`.
A decorator runs at import, before any of the compiler's gates, so the flag
never applied to them -- while `compiler.py` and `temporary_patches/utils.py`
both consult it. That matters because the flag is the documented escape hatch
for precisely the failure they produce: re-running `NeMo-Gym-Sudoku` with
UNSLOTH_COMPILE_DISABLE=1 changed nothing, since the compile had already
happened at import.

This makes the flag work. It does NOT fix the shape bug.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FUNCS = [
    "chunked_selective_log_softmax",
    "chunked_hidden_states_selective_log_softmax",
]


def _probe(value):
    """Import rl_replacements in a subprocess with the flag set and report
    whether each helper came out compiled. A subprocess because the flag is read
    at import and the module is then cached in sys.modules."""
    script = textwrap.dedent("""
        import json, torch
        import unsloth_zoo.rl_replacements as R
        out = {}
        for name in %r:
            f = getattr(R, name, None)
            if f is None:
                out[name] = "missing"
            else:
                out[name] = bool(getattr(f, "_torchdynamo_orig_callable", None)
                                 or type(f).__name__ == "OptimizedModule")
        print("PROBE " + json.dumps(out))
    """ % (FUNCS,))
    env = dict(os.environ, PYTHONPATH=str(ROOT), UNSLOTH_COMPILE_DISABLE=value)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, timeout=600, env=env)
    line = [l for l in r.stdout.splitlines() if l.startswith("PROBE ")]
    assert line, (r.stdout[-2000:], r.stderr[-3000:])
    import json
    return json.loads(line[0][len("PROBE "):])


def test_the_helpers_are_compiled_by_default():
    """The change adds an off switch, not off-by-default. If this fails, every
    GRPO run just got slower."""
    got = _probe("0")
    assert all(v is True for v in got.values()), got


def test_the_flag_turns_them_off():
    got = _probe("1")
    assert all(v is False for v in got.values()), got


def test_partial_also_turns_them_off():
    """compiler.py treats "partial" as compile-off (it only skips the source
    rewrites for "1"), so the helpers must honour it as well."""
    got = _probe("partial")
    assert all(v is False for v in got.values()), got


def test_an_unset_flag_behaves_like_zero():
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    script = ("import unsloth_zoo.rl_replacements as R;"
              "f=R.chunked_hidden_states_selective_log_softmax;"
              "print('COMPILED', bool(getattr(f,'_torchdynamo_orig_callable',None)))")
    r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, timeout=600, env=env)
    assert "COMPILED True" in r.stdout, (r.stdout[-1500:], r.stderr[-2000:])


# ---- the source, so the fix cannot be half-applied -----------------------

def _src():
    return (ROOT / "unsloth_zoo" / "rl_replacements.py").read_text(encoding="utf-8")


def test_no_bare_torch_compile_decorator_remains():
    """A fourth helper with a bare decorator would reintroduce the gap."""
    src = _src()
    bare = [n for n, line in enumerate(src.splitlines(), 1)
            if line.strip().startswith("@torch.compile")]
    assert bare == [], f"bare @torch.compile at lines {bare}"


def test_the_generated_code_string_is_left_alone():
    """One `@torch.compile(...)` lives inside a string emitted as generated
    source; rewriting it would call a helper that module does not import."""
    assert ('"@torch_compile_with_fallback(dynamic = True, fullgraph = True, '
            'options = torch_compile_options)\\n"') in _src()


def _common_src():
    return (ROOT / "unsloth_zoo" / "temporary_patches" / "common.py").read_text(encoding="utf-8")


def test_the_helper_falls_back_to_identity_not_to_none():
    """Returning None would replace the function with None and fail at call
    time rather than at import."""
    src = _common_src()
    i = src.index("def _maybe_compile(")
    body = src[i:]
    assert "return lambda fn: fn" in body, body[:400]


def test_the_flag_comes_from_the_shared_definition():
    """Re-reading os.environ in either place would drift from compiler.py."""
    assert "UNSLOTH_COMPILE_DISABLE" not in _src(), "read it via _maybe_compile"
    common = _common_src()
    i = common.index("def _maybe_compile(")
    assert "UNSLOTH_COMPILE_DISABLE" in common[i:]
    assert "os.environ" not in common[i:]


def test_the_helper_is_not_defined_beside_its_callers():
    """It was, and the compiler copies decorated source into generated trainer
    modules, where a name from rl_replacements does not resolve -- every SFT run
    then fell back to plain trl, silently."""
    assert "def _maybe_compile(" not in _src()
    assert "def _maybe_compile(" in _common_src()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
