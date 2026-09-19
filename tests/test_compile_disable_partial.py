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

"""UNSLOTH_COMPILE_DISABLE=partial must also switch off the temporary patches.

compiler.py reads the flag as `in ("1", "partial")` but common.py read it as `== "1"`,
so "partial" left every patch_function(fullgraph = ...) compiling and the escape hatch
could not work around a compile-only crash. The flag is read at import, so each value
is probed in a subprocess.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_SCRIPT = textwrap.dedent("""
    import json
    try:
        import transformers.models.gemma3n.modeling_gemma3n as M
    except Exception:
        print("PROBE " + json.dumps({"skip": True})); raise SystemExit
    from unsloth_zoo.temporary_patches.common import UNSLOTH_COMPILE_DISABLE
    from unsloth_zoo.temporary_patches.gemma3n import (
        patch_Gemma3nMultimodalEmbedder_forward,
    )
    patch_Gemma3nMultimodalEmbedder_forward()
    print("PROBE " + json.dumps({
        "skip"     : False,
        "flag"     : bool(UNSLOTH_COMPILE_DISABLE),
        "compiled" : hasattr(M.Gemma3nMultimodalEmbedder.forward, "get_compiler_config"),
    }))
""")


def _probe_env(value):
    # Without the `unsloth` package, unsloth_zoo/__init__ raises "Please install Unsloth"
    # before the probe prints; this env var takes the light import path, same flag.
    return dict(
        os.environ,
        PYTHONPATH = str(ROOT),
        UNSLOTH_COMPILE_DISABLE = value,
        UNSLOTH_ZOO_DISABLE_GPU_INIT = "1",
    )


def _probe(value):
    env = _probe_env(value)
    r = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output = True, text = True, timeout = 900, env = env,
    )
    line = [l for l in r.stdout.splitlines() if l.startswith("PROBE ")]
    assert line, (r.stdout[-2000:], r.stderr[-3000:])
    out = json.loads(line[0][len("PROBE "):])
    if out["skip"]: pytest.skip("transformers has no gemma3n")
    return out


def test_probe_survives_a_zoo_only_checkout():
    """CI installs `unsloth` with `|| true`, so it can legitimately be absent."""
    env = _probe_env("0")
    assert env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] == "1"


def test_unset_still_compiles():
    out = _probe("0")
    assert out["flag"] is False
    assert out["compiled"] is True, "the patch stopped compiling altogether"


def test_one_disables_compile():
    out = _probe("1")
    assert out["flag"] is True
    assert out["compiled"] is False


def test_partial_disables_compile():
    out = _probe("partial")
    assert out["flag"] is True
    assert out["compiled"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
