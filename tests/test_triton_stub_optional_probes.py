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

"""Modules stock Triton lacks must stay unimportable under the stub.

torch 2.15 probes ``triton._torchtlx_default`` while building
``torch._inductor.config``; a stubbed answer broke ``torch._dynamo`` and so every
transformers import. Each check runs in a fresh interpreter.
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


@pytest.mark.parametrize("stmt", [
    "from triton._torchtlx_default import DEFAULT_MODE",
    "import triton.language.extra.tlx",
    "from triton.language.extra.tlx.inductor import registry",
])
def test_absent_modules_raise_import_error(stmt):
    proc = _run(f'''
        try:
            {stmt}
        except ImportError:
            pass
        else:
            raise SystemExit("stub answered: {stmt}")
        import triton.language.extra.some_new_module as mod
        assert type(mod).__name__ == "_PermissiveModule", type(mod)
        print("OK")
    ''')
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stdout + proc.stderr


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason = "torch not installed")
def test_dynamo_and_inductor_import_under_the_stub():
    proc = _run('''
        import torch
        import torch._inductor.config
        import torch._dynamo
        assert torch._inductor.config.triton.tlx_mode is None
        f = torch.compile(lambda x: x.sin() + 1, backend = "eager")
        assert torch.allclose(f(torch.ones(3)), torch.ones(3).sin() + 1)
        print("OK", torch.__version__)
    ''')
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stdout + proc.stderr[-4000:]
