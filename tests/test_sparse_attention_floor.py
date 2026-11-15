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

"""Sparse attention dispatch across Triton / torch versions and under torch.compile: gather is only selected where
its Triton kernels import, and traced dispatch does not recompile on lazily-filled state."""

import os
import sys

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth_zoo import sparse_attention as sa  # noqa: E402

_VENDORED_TRITON = ("unsloth_zoo._vendored.attn_gym._backends.triton",
                    "unsloth_zoo._vendored.attn_gym.sparse.gather_attn.impl.triton")


def test_gather_needs_its_triton_kernels_not_only_the_api(monkeypatch):
    # torch 2.4 - 2.8 bundle Triton 3.0 - 3.4, which lack triton.constexpr_function: the api and op registrations
    # import there, the kernels (loaded on the first call) do not, so gather must not be selected.
    triton = pytest.importorskip("triton")
    monkeypatch.setattr(sa, "_GATHER", None)
    if hasattr(triton, "constexpr_function"):
        assert sa._gather_importable()   # api + ops cached; their op definitions never run twice below
        monkeypatch.delattr(triton, "constexpr_function")
        for name in [m for m in sys.modules if m.startswith(_VENDORED_TRITON)]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setattr(sa, "_GATHER", None)
    assert sa._gather_importable() is False


_COLD = r"""
import sys, torch
sys.path.insert(0, sys.argv[1])
case = sys.argv[2]
torch._dynamo.config.error_on_recompile = True
if hasattr(torch.fx.experimental._config, "use_duck_shape"):   # torch >= 2.5
    torch.fx.experimental._config.use_duck_shape = False
if case != "import_in_trace":
    from unsloth_zoo import sparse_attention as sa
dev, dt, h, d = "cuda", torch.bfloat16, 16, 512

def index_default(q, kv, sink, idx, ckv, cidx):
    return sa.sparse_attn(q, kv, sink, idx, d ** -0.5)

def window_triton(q, kv, sink, idx, ckv, cidx):
    return sa.sparse_attn_window(q, kv, ckv, cidx, sink, d ** -0.5, 64, backend = "triton")

def import_in_trace(q, kv, sink, idx, ckv, cidx):
    from unsloth_zoo import sparse_attention as sa
    return sa.sparse_attn_window(q, kv, ckv, cidx[..., :0], sink, d ** -0.5, 64)

fn = torch.compile({"index_default": index_default, "window_triton": window_triton,
                    "import_in_trace": import_in_trace}[case], fullgraph = True, dynamic = True)
for s in (256, 300):
    g = torch.Generator(device = dev).manual_seed(s)
    r = lambda *shape: (torch.randn(*shape, device = dev, generator = g) * 0.3).to(dt).requires_grad_()
    q, kv, ckv = r(1, s, h, d), r(1, s, d), r(1, s // 4 + 3, d)
    sink = torch.randn(h, device = dev, generator = g).requires_grad_()
    idx = torch.randint(-1, s, (1, s, 64), device = dev, dtype = torch.int32, generator = g)
    cidx = torch.randint(-1, s // 4 + 3, (1, s, 32), device = dev, dtype = torch.int32, generator = g)
    fn(q, kv, sink, idx, ckv, cidx).float().square().mean().backward()
from unsloth_zoo import sparse_mqa_kernel as sk
print("COLD_OK", sk.SPARSE_MQA_STATS["fwd"] + sk.SPARSE_MQA_STATS["window_fwd"])
"""


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8, reason = "bf16 CUDA kernels")
@pytest.mark.parametrize("case", ["index_default", "window_triton", "import_in_trace"])
def test_cold_compiled_dispatch_does_not_recompile_across_lengths(case):
    # A fresh process whose first sparse call is compiled (dynamic shapes, two lengths): dispatch state filled during
    # the first trace (shared-memory caches, the gather import) must not become a guard that recompiles the second.
    import subprocess
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[1])
    proc = subprocess.run([sys.executable, "-c", _COLD, root, case], capture_output = True, text = True, timeout = 900,
                          env = dict(os.environ, PYTHONPATH = root + os.pathsep + os.environ.get("PYTHONPATH", "")))
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("COLD_OK")), None)
    assert line is not None, proc.stderr[-3000:]
    if case == "index_default" and torch.cuda.get_device_capability()[0] in (10, 11):
        # Blackwell picks gather first; a cold trace used to skip it (the import had not run yet) and take Triton.
        assert int(line.split()[1]) == 0, line
