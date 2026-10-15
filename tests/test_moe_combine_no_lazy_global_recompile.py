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

"""The MoE combine runs inside compiled MoE forwards. Its Triton entry points must not
resolve module state on first call: Dynamo guards on the global it read as None, the
first call sets it, and every enclosing graph compiles a second time."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

ZOO = Path(__file__).resolve().parents[1]


def _run(body):
    # Fresh interpreter: the bug only shows on the first call in a process.
    env = dict(os.environ)
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    env.pop("UNSLOTH_MOE_TRITON_KERNELS", None)
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output = True, text = True, env = env, cwd = str(ZOO),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


_BODY = """
    import torch
    from torch._dynamo.testing import CompileCounter
    from unsloth_zoo.temporary_patches import moe_utils

    device = "{device}"
    counter = CompileCounter()

    def combine(y, sorted_indices, w):
        out = moe_utils.weighted_unpermute(y, sorted_indices, w, 4, 2, out_dtype = torch.bfloat16)
        if out is None:
            out = torch.zeros(4, y.shape[1], dtype = torch.float32, device = y.device)
            out.index_add_(0, sorted_indices // 2, y.float() * w.float()[:, None])
            out = out.to(torch.bfloat16)
        return out
    compiled = torch.compile(combine, backend = counter, dynamic = False)

    g = torch.Generator().manual_seed(0)
    y = torch.randn(8, 64, generator = g).to(device, torch.bfloat16)
    sorted_indices = torch.randperm(8, generator = g).to(device)
    w = torch.rand(8, generator = g).to(device)
    expected = torch.zeros(4, 64, dtype = torch.float32, device = device)
    expected.index_add_(0, sorted_indices // 2, y.float() * w[:, None])
    for _ in range(3):
        torch.testing.assert_close(compiled(y, sorted_indices, w), expected.to(torch.bfloat16))
    assert counter.frame_count == 1, counter.frame_count
    print("OK", device, moe_utils.weighted_unpermute(y, sorted_indices, w, 4, 2) is not None)
"""


def test_combine_compiles_once_cpu():
    out = _run(_BODY.format(device = "cpu"))
    assert "OK cpu False" in out


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
def test_combine_compiles_once_cuda():
    out = _run(_BODY.format(device = "cuda"))
    assert "OK cuda True" in out


def test_experts_interface_lookups_filled_before_compile():
    out = _run("""
        import torch
        from torch._dynamo.testing import CompileCounter
        from unsloth_zoo.temporary_patches import moe_experts_interface as mei

        if mei._experts_interface() is None:
            print("SKIP no experts interface")
            raise SystemExit(0)
        mei.patch_experts_interface()
        assert {"default_apply_gate", "layout_is_standard", "moe_utils"} <= set(mei._LAZY), mei._LAZY

        counter = CompileCounter()

        @torch.compile(backend = counter, dynamic = False)
        def route(x):
            # The lookups _dense_standard_experts / _has_custom_gate make inside a traced MoE forward.
            if mei._experts_layout_is_standard_or_none() is not None:
                x = x + 1
            if mei._default_apply_gate_or_none() is not None:
                x = x * 2
            return mei._moe_utils_module() is not None and x.sum()

        x = torch.ones(4)
        for _ in range(3):
            route(x)
        assert counter.frame_count == 1, counter.frame_count
        print("OK")
    """)
    assert "OK" in out or "SKIP" in out
