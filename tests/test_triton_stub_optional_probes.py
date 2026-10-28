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

"""Optional Triton modules torch probes must stay unimportable under the stub.

torch 2.15 builds ``torch._inductor.config`` with
``try: from triton._torchtlx_default import DEFAULT_MODE except ImportError``.
The Apple Silicon / no-GPU-init triton stub resolved every ``triton.*`` name, so
that probe got a ``_Noop`` back, the config module rejected it, and the failed
``torch._inductor`` import left ``torch._dynamo`` unimportable for the rest of the
process: every transformers model, generation or ``set_seed`` import then died
with ``Could not import module ...``. Each check runs in a fresh interpreter so the
stub never leaks into this one.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys
import textwrap

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
STUB_PATH = REPO_ROOT / "unsloth_zoo" / "stubs" / "triton_stub.py"

# Loads the stub standalone (no unsloth_zoo import) and installs it the way
# unsloth_zoo/__init__.py does on an MLX host.
_INJECT = textwrap.dedent(f'''
    import importlib.util, sys
    _spec = importlib.util.spec_from_file_location("_triton_stub_under_test", {str(STUB_PATH)!r})
    _stub = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _stub
    _spec.loader.exec_module(_stub)
    for _k in [k for k in sys.modules if k == "triton" or k.startswith("triton.")]:
        del sys.modules[_k]
    _stub.inject_into_sys_modules()
''')


def _run(body):
    return subprocess.run(
        [sys.executable, "-c", _INJECT + textwrap.dedent(body)],
        capture_output = True, text = True, timeout = 600,
    )


def test_the_torchtlx_probe_is_not_stubbed():
    proc = _run('''
        try:
            from triton._torchtlx_default import DEFAULT_MODE
        except ImportError:
            pass
        else:
            raise SystemExit(f"stub answered the probe with {DEFAULT_MODE!r}")
        # Everything else keeps resolving to the permissive stub.
        import triton.language.extra.some_new_module as mod
        assert type(mod).__name__ == "_PermissiveModule", type(mod)
        print("OK")
    ''')
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stdout + proc.stderr


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason = "torch not installed")
def test_dynamo_and_inductor_import_under_the_stub():
    """The user-facing break: torch._dynamo (pulled in by transformers' masking_utils)
    must import after the stub is installed."""
    proc = _run('''
        import torch
        import torch._inductor.config
        import torch._dynamo
        print("OK", torch.__version__)
    ''')
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stdout + proc.stderr[-4000:]
