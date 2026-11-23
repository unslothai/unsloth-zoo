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

"""Routed MoE kernels read expert ids at IDX + p. A strided route index (a column slice of the router's top-k:
reshape(-1) keeps stride 2) must pick the same experts as its contiguous copy. Runs the real Triton kernels, in
interpreter mode on CPU (a subprocess, since TRITON_INTERPRET is read at import) or natively on CUDA."""

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("triton")

_SCRIPT = r"""
import contextlib, json, os, sys
import torch
if not torch.cuda.is_available():
    torch.cuda.device = lambda device: contextlib.nullcontext()  # the launchers enter the operand's CUDA device
from unsloth_zoo.temporary_patches import moe_routed as MR

dev = "cuda" if torch.cuda.is_available() else "cpu"
ids = torch.tensor([[0, 1, 2, 3]], device = dev)[:, ::2]
assert ids.reshape(-1).tolist() == [0, 2] and ids.reshape(-1).stride() == (2,)
result = {"file": MR.__file__}

# NF4 select dequant (own-gate decode): expert e stores byte 0x11 * (e + 1), so code 1 + e is its signature.
w = torch.tensor([[0x11] * 32, [0x22] * 32, [0x33] * 32], dtype = torch.uint8, device = dev).reshape(-1)
ones = torch.ones(3, device = dev)
tb = {"w": w, "a": ones, "a2": ones, "c2": ones, "off": torch.zeros(1, device = dev),
      "lut": torch.arange(16, dtype = torch.float32, device = dev), "N": 1, "K": 64,
      "blocksize": 64, "blocksize2": 256, "nested": False}
got = MR.nf4_select_dequant(tb, ids.reshape(-1), torch.float32)
ref = MR.nf4_select_dequant(tb, ids.reshape(-1).contiguous(), torch.float32)
result["nf4_strided"] = got[:, 0, 0].tolist()
result["nf4_contiguous"] = ref[:, 0, 0].tolist()

# Routed LoRA A: row p = x[p] @ a[idx[p]].
x = torch.ones(2, 16, device = dev)
a = torch.arange(4, dtype = torch.float32, device = dev).view(4, 1, 1).expand(4, 16, 2).contiguous()
result["lora_a_strided"] = MR.routed_lora_a(x, ids, a)[:, 0].tolist()
result["lora_a_contiguous"] = MR.routed_lora_a(x, ids.contiguous(), a)[:, 0].tolist()
print("RESULT " + json.dumps(result))
"""


def test_strided_route_index_selects_the_same_experts():
    env = dict(os.environ, UNSLOTH_IS_PRESENT = "1")
    try:
        import torch
        on_cuda = torch.cuda.is_available()
    except ImportError:
        on_cuda = False
    if not on_cuda:
        env.update(TRITON_INTERPRET = "1", CUDA_VISIBLE_DEVICES = "")
    proc = subprocess.run([sys.executable, "-c", _SCRIPT], env = env, capture_output = True, text = True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT "))
    result = json.loads(line[len("RESULT "):])
    from unsloth_zoo.temporary_patches import moe_routed
    assert result.pop("file") == moe_routed.__file__
    # Experts 0 and 2: NF4 signatures 1 and 3, LoRA A rows 0 * 16 and 2 * 16.
    assert result["nf4_contiguous"] == [1.0, 3.0]
    assert result["nf4_strided"] == result["nf4_contiguous"]
    assert result["lora_a_contiguous"] == [0.0, 32.0]
    assert result["lora_a_strided"] == result["lora_a_contiguous"]
