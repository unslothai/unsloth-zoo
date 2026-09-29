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


def test_compiled_regions_run_eager_only_inside_eager_decode():
    out = _run("""
        import torch
        from unsloth_zoo.temporary_patches.utils import (
            torch_compile_with_fallback, torch_compiler_disable_unless_decode,
            unsloth_eager_decode, eager_decode_active,
        )
        from unsloth_zoo.temporary_patches.common import unwrap_already_compiled

        frames = torch._dynamo.utils.counters["stats"]
        def calls():
            return frames["calls_captured"], frames["unique_graphs"]

        for fullgraph in (True, False):
            def f(x): return x * 2 + 1
            g = torch_compile_with_fallback(fullgraph = fullgraph, dynamic = True)(f)
            assert hasattr(g, "get_compiler_config"), "lost the compiled marker"
            assert unwrap_already_compiled(g) is f, "unwrap no longer reaches the original"
            x = torch.ones(3)
            torch._dynamo.reset(); frames.clear()
            with torch.no_grad(), unsloth_eager_decode():
                assert eager_decode_active()
                y = g(x)
            assert not eager_decode_active()
            assert frames["unique_graphs"] == 0, "compiled during eager decode"
            with torch.no_grad():
                g(x)
            assert frames["unique_graphs"] == 1, "compiled path no longer taken outside"
            with unsloth_eager_decode():  # grad on: training, never bypassed
                assert not eager_decode_active()
            assert torch.equal(y, x * 2 + 1)

        # An outer compile traces the compiled path even inside the scope.
        def h(x): return x - 1
        h = torch_compile_with_fallback(fullgraph = True, dynamic = True)(h)
        outer = torch.compile(lambda x: h(x) * 3, backend = "eager", fullgraph = True)
        with torch.no_grad(), unsloth_eager_decode():
            assert torch.equal(outer(torch.ones(2)), torch.zeros(2))

        # Another thread never sees this thread's scope, and nested scopes restore their own.
        import threading
        other = []
        with torch.no_grad(), unsloth_eager_decode():
            t = threading.Thread(target = lambda: other.append(eager_decode_active()))
            t.start(); t.join()
            with unsloth_eager_decode():
                pass
            assert eager_decode_active()
        assert other == [False] and not eager_decode_active()

        # Dynamo never reads the flag, so entering and leaving the scope does not recompile.
        calls = torch.compile(lambda x: h(x) + 1, backend = "eager", fullgraph = True)
        torch._dynamo.reset(); frames.clear()
        with torch.no_grad():
            for _ in range(3):
                calls(torch.ones(2))
                with unsloth_eager_decode():
                    calls(torch.ones(2))
        assert frames["unique_graphs"] == 1, frames

        seen = []
        @torch_compiler_disable_unless_decode
        def d(x):
            seen.append(torch.compiler.is_compiling())
            return x + 1
        with torch.no_grad(), unsloth_eager_decode():
            d(torch.ones(1))
        assert seen == [False]
        print("OK")
    """)
    assert "OK" in out
