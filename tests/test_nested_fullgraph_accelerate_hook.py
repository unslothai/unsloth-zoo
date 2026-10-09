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

"""unslothai/unsloth#3419: a fullgraph = False region calling a fullgraph = True one whose Conv2d
carries accelerate's disabled `AlignDevicesHook` (InternVL on a split device map) must not raise."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ZOO = Path(__file__).resolve().parents[1]


def _run(body):
    env = dict(os.environ)
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    env["PYTHONPATH"] = str(ZOO) + os.pathsep + env.get("PYTHONPATH", "")
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output = True, text = True, env = env, cwd = str(ZOO),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


@pytest.mark.parametrize("hooked", [False, True])
def test_nested_fullgraph_region_calls_hooked_child(hooked):
    pytest.importorskip("accelerate")
    out = _run(f"""
        import torch, torch.nn as nn
        from accelerate.hooks import AlignDevicesHook, add_hook_to_module
        from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback

        @torch_compile_with_fallback(fullgraph = True, dynamic = True)
        def patch_forward(self, x):
            return self.projection(x).flatten(2).transpose(1, 2)

        @torch_compile_with_fallback(fullgraph = False, dynamic = True)
        def embeddings_forward(self, x):
            e = self.patch(x)
            return torch.cat([self.cls.expand(e.shape[0], -1, -1), e], 1) * 2

        class Patch(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Conv2d(3, 8, 4, 4)
            def forward(self, x):
                return patch_forward(self, x)

        class Embeddings(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch = Patch()
                self.cls = nn.Parameter(torch.randn(1, 1, 8))
            def forward(self, x):
                return embeddings_forward(self, x)

        torch.manual_seed(0)
        model = Embeddings()
        if {hooked}:
            add_hook_to_module(model.patch.projection, AlignDevicesHook(execution_device = "cpu"))
        x = torch.randn(2, 3, 16, 16, requires_grad = True)
        with torch.no_grad():
            reference = torch.cat(
                [model.cls.expand(2, -1, -1), model.patch.projection(x).flatten(2).transpose(1, 2)], 1) * 2
        out = model(x)
        torch.testing.assert_close(out, reference)
        out.sum().backward()
        assert x.grad is not None and model.patch.projection.weight.grad is not None

        # Called on its own, the fullgraph region still compiles and runs.
        with torch.no_grad():
            torch.testing.assert_close(model.patch(x), model.patch.projection(x).flatten(2).transpose(1, 2))
        print("OK")
    """)
    assert "OK" in out
