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

"""The MXFP4 kernel self-checks (``torch.equal`` / ``torch.allclose``) are data dependent: the first call of a
process can come from inside a fullgraph trace (a cold compiled backward decodes the weight), which must not fail
and must not cache a verdict. Each case runs in a fresh process so nothing is warm."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")

_COMMON = """
import torch
import unsloth_zoo.mxfp4_dequant as md
import unsloth_zoo.mxfp4_gemm as mg
device = torch.device("cuda", torch.cuda.current_device())
g = torch.Generator(device = device).manual_seed(0)
blocks = torch.randint(0, 256, (64, 8, 16), dtype = torch.uint8, device = device, generator = g)
scales = torch.randint(118, 128, (64, 8), dtype = torch.uint8, device = device, generator = g)
"""

_DEQUANT = _COMMON + """
assert not md._KERNEL_VERIFIED
f = torch.compile(lambda b, s: md.mxfp4_dequantize(b, s, dtype = torch.float16), fullgraph = True)
got = f(blocks, scales)
want = md.mxfp4_dequantize_torch(blocks, scales, dtype = torch.float16)
assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
assert not md._KERNEL_VERIFIED, md._KERNEL_VERIFIED  # the trace decided nothing
md.mxfp4_dequantize(blocks, scales, dtype = torch.float16)
assert md._KERNEL_VERIFIED, "the first eager call still runs the self-check"
print("OK")
"""

_GRAD = _COMMON + """
w = md.mxfp4_dequantize_torch(blocks, scales, dtype = torch.float16)

class F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x @ w.t()

    @staticmethod
    def backward(ctx, gy):
        return gy @ md.mxfp4_dequantize(blocks, scales, dtype = torch.float16)

x = torch.randn(3, 256, dtype = torch.float16, device = device, requires_grad = True)
f = torch.compile(lambda a: F.apply(a), fullgraph = True)
f(x).float().sum().backward()
want = torch.ones(3, 64, dtype = torch.float16, device = device) @ w
torch.testing.assert_close(x.grad, want, rtol = 2e-3, atol = 2e-3 * want.abs().max().item())
print("OK")
"""

_GEMM = _COMMON + """
assert not mg._AVAILABLE_DTYPE
x = torch.randn(3, 256, dtype = torch.float16, device = device)
f = torch.compile(lambda a: a * 2 if mg.mxfp4_gemm_available(a.device, a.dtype) else a * 3, fullgraph = True)
f(x)
assert not mg._AVAILABLE_DTYPE, mg._AVAILABLE_DTYPE  # nothing cached from inside the trace
eager = mg.mxfp4_gemm_available(device, torch.float16)
assert mg._AVAILABLE_DTYPE
print("OK", eager)
"""


@pytest.mark.parametrize("script", [_DEQUANT, _GRAD, _GEMM], ids = ["dequant", "backward", "gemm_available"])
def test_cold_self_check_inside_fullgraph(script):
    env = dict(os.environ)
    env.setdefault("UNSLOTH_IS_PRESENT", "1")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env = env, capture_output = True, text = True, timeout = 600,
    )
    if result.returncode != 0 or "OK" not in result.stdout:
        pytest.fail(result.stdout[-2000:] + "\n" + result.stderr[-4000:], pytrace = False)
