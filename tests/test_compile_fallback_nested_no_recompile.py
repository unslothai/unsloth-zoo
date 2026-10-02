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

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ZOO = Path(__file__).resolve().parents[1]


def _run(body):
    # Fresh interpreter: the compile helpers read UNSLOTH_COMPILE_DISABLE at import time.
    env = dict(os.environ)
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output = True, text = True, env = env, cwd = str(ZOO),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


def test_enclosing_compile_does_not_recompile_on_the_fallback_state():
    out = _run("""
        import torch
        from torch._dynamo.testing import CompileCounter
        from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback

        for inner_fullgraph in (True, False):
            torch._dynamo.reset()
            counter = CompileCounter()

            def inner(x):
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + 1e-6)
            inner = torch_compile_with_fallback(
                fullgraph = inner_fullgraph, backend = counter, dynamic = True)(inner)

            def outer(x):
                return inner(x).sin() + 1
            outer = torch_compile_with_fallback(fullgraph = False, backend = counter, dynamic = True)(outer)

            def reference(x):
                return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + 1e-6)).sin() + 1

            for n in (8, 16, 32):
                x = torch.randn(n, 64)
                with torch.no_grad():
                    torch.testing.assert_close(outer(x), reference(x))
            assert counter.frame_count == 1, (inner_fullgraph, counter.frame_count)

            x = torch.randn(8, 64, requires_grad = True)
            outer(x).sum().backward()
            ref = x.detach().clone().requires_grad_()
            reference(ref).sum().backward()
            torch.testing.assert_close(x.grad, ref.grad)
            # One more graph for grad mode, none for the wrapper's own bookkeeping.
            assert counter.frame_count == 2, (inner_fullgraph, counter.frame_count)

            # Called on its own, the inner wrapper still compiles and runs.
            with torch.no_grad():
                torch.testing.assert_close(
                    inner(torch.ones(4, 64)), torch.ones(4, 64) * torch.rsqrt(torch.tensor(1.0 + 1e-6)))
        print("OK")
    """)
    assert "OK" in out
