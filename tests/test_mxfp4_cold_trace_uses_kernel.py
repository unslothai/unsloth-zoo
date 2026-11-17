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

"""A cold compiled graph must already use the verified fused MXFP4 kernel, not bake in the slow torch reference; each case runs in a fresh process."""
import os, subprocess, sys, textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")

_SCRIPT = """
import sys, torch
import unsloth_zoo.mxfp4_dequant as md
import unsloth_zoo.mxfp4_gemm as mg
from torch.profiler import profile, ProfilerActivity
fullgraph = sys.argv[1] == "1"
device = torch.device("cuda", torch.cuda.current_device())
g = torch.Generator(device = device).manual_seed(0)
blocks = torch.randint(0, 256, (4, 256, 8, 16), dtype = torch.uint8, device = device, generator = g)
scales = torch.randint(118, 128, (4, 256, 8), dtype = torch.uint8, device = device, generator = g)
assert not md._KERNEL_VERIFIED
f = torch.compile(lambda b, s: md.mxfp4_dequantize(b, s, dtype = torch.bfloat16, transpose = True), fullgraph = fullgraph)
got = f(blocks, scales)
want = md.mxfp4_dequantize_torch(blocks, scales, dtype = torch.bfloat16, transpose = True)
assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
with profile(activities = [ProfilerActivity.CUDA]) as p:
    f(blocks, scales); torch.cuda.synchronize()
names = {e.name for e in p.events() if e.device_type.name == "CUDA"}
assert any("_mxfp4_dequant_kernel" in n for n in names), names
x = torch.randn(3, 256, dtype = torch.bfloat16, device = device)
h = torch.compile(lambda a: a * 2 if mg.mxfp4_gemm_available(a.device, a.dtype) else a * 3, fullgraph = fullgraph)
y = h(x)
eager = mg.mxfp4_gemm_available(device, torch.bfloat16)
assert torch.equal(y, x * (2 if eager else 3)), "traced verdict differs from the eager one"
print("OK")
"""


@pytest.mark.parametrize("fullgraph", ["1", "0"])
def test_cold_trace_already_uses_the_verified_kernel(fullgraph):
    env = dict(os.environ)
    env.setdefault("UNSLOTH_IS_PRESENT", "1")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(_SCRIPT), fullgraph], env = env,
                       capture_output = True, text = True, timeout = 600)
    if r.returncode != 0 or "OK" not in r.stdout:
        pytest.fail(r.stdout[-2000:] + "\n" + r.stderr[-3000:], pytrace = False)
