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
interpreter mode on CPU (a subprocess, since TRITON_INTERPRET is read at import) or natively on CUDA, and checks
that every public routed entry point hands its kernel launch a unit-stride index."""

import json
import os
import subprocess
import sys

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
pytest.importorskip("triton")
import torch

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


def case(name, fn):
    try:
        got, ref = fn(ids.reshape(-1)), fn(ids.reshape(-1).contiguous())
        result[name] = {"strided": got, "contiguous": ref}
    except Exception as e:  # an op this Triton's interpreter cannot run
        result[name] = {"error": f"{type(e).__name__}: {e}"[:500]}


# NF4 tables: expert e stores byte 0x11 * (e + 1), so code 1 + e is its signature.
def nf4_table(N):
    w = torch.tensor([0x11] * (N * 32) + [0x22] * (N * 32) + [0x33] * (N * 32), dtype = torch.uint8, device = dev)
    ones = torch.ones(12, device = dev)
    return {"w": w, "a": ones, "a2": ones, "c2": ones, "off": torch.zeros(1, device = dev),
            "lut": torch.arange(16, dtype = torch.float32, device = dev), "N": N, "K": 64,
            "blocksize": 64, "blocksize2": 256, "nested": False, "stacked": True, "bias": None, "bias_kind": 0}


# NF4 select dequant (own-gate decode).
case("nf4", lambda idx: MR.nf4_select_dequant(nf4_table(1), idx, torch.float32)[:, 0, 0].tolist())
# NF4 routed down (GPT-OSS decode): one token, top-2, unit routing weights, so out = 64 * (code_0 + code_1).
case("nf4_down", lambda idx: MR.routed_down(
    torch.ones(2, 64, device = dev), idx, torch.ones(1, 2, device = dev), False, nf4_table(4), 2,
    torch.float32)[:, 0].tolist())
# Routed LoRA A: row p = x[p] @ a[idx[p]].
x = torch.ones(2, 16, device = dev)
a = torch.arange(4, dtype = torch.float32, device = dev).view(4, 1, 1).expand(4, 16, 2).contiguous()
case("lora_a", lambda idx: MR.routed_lora_a(x, idx, a)[:, 0].tolist())
print("RESULT " + json.dumps(result))
"""

# Experts 0 and 2: NF4 signatures 1 and 3; down 64 * (1 + 3); LoRA A rows 0 * 16 and 2 * 16.
_EXPECTED = {"nf4": [1.0, 3.0], "nf4_down": [256.0], "lora_a": [0.0, 32.0]}


@pytest.fixture(scope = "module")
def interpreted():
    env = dict(os.environ, UNSLOTH_IS_PRESENT = "1")
    if not torch.cuda.is_available():
        env.update(TRITON_INTERPRET = "1", CUDA_VISIBLE_DEVICES = "")
    proc = subprocess.run([sys.executable, "-c", _SCRIPT], env = env, capture_output = True, text = True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT "))
    result = json.loads(line[len("RESULT "):])
    from unsloth_zoo.temporary_patches import moe_routed
    assert result.pop("file") == moe_routed.__file__
    return result


@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_strided_route_index_selects_the_same_experts(interpreted, name):
    got = interpreted[name]
    if "error" in got:
        if torch.cuda.is_available():
            pytest.fail(got["error"])
        pytest.skip(f"this Triton's CPU interpreter cannot run the kernel: {got['error']}")
    assert got["contiguous"] == _EXPECTED[name]
    assert got["strided"] == got["contiguous"]


def _launched_index(monkeypatch, entry):
    """The index tensor every Triton launch of `entry` receives, captured instead of launching."""
    from unsloth_zoo.temporary_patches import moe_routed as MR
    seen = []

    def fast_launch(kernel, grid, args, *rest, **kwargs):
        seen.append(args[1])

    def lora_h_launch(x, idx, a_ptrs, kind, r, row_div, out):
        seen.append(idx)
        return out

    monkeypatch.setattr(MR, "_fast_launch", fast_launch)
    monkeypatch.setattr(MR, "_lora_h_launch", lora_h_launch)
    monkeypatch.setattr(MR, "lora_pointer_table", lambda weights: (torch.zeros(len(weights), dtype = torch.int64), 1))
    ids = torch.tensor([[0, 1, 2, 3]])[:, ::2].reshape(-1)
    assert ids.stride() == (2,)
    entry(MR, ids)
    assert seen, "no kernel launch reached"
    return seen


def _nf4_table():
    one = torch.ones(12)
    return {"w": torch.zeros(4 * 32 * 3, dtype = torch.uint8), "a": one, "a2": one, "c2": one, "off": one,
            "lut": torch.arange(16, dtype = torch.float32), "N": 4, "K": 64, "blocksize": 64, "blocksize2": 256,
            "nested": False, "stacked": True, "bias": None, "bias_kind": 0}


_ENTRIES = {
    # The GPT-OSS NF4 decode chain (gpt_oss_routed): routed_lora_h, routed_gate_up, routed_down.
    "routed_gate_up": lambda MR, ids: MR.routed_gate_up(torch.ones(1, 64), ids, _nf4_table(), 2, 1.702, 7.0),
    "routed_down": lambda MR, ids: MR.routed_down(
        torch.ones(2, 64), ids, torch.ones(1, 2), False, _nf4_table(), 2, torch.float32),
    "routed_lora_h": lambda MR, ids: MR.routed_lora_h(torch.ones(1, 16), ids, [torch.ones(2, 16)] * 4, 2),
}


@pytest.mark.parametrize("name", sorted(_ENTRIES))
def test_routed_entry_launches_with_a_unit_stride_index(monkeypatch, name):
    for idx in _launched_index(monkeypatch, _ENTRIES[name]):
        assert idx.tolist() == [0, 2]
        assert idx.is_contiguous(), f"{name} launched with index stride {idx.stride()}"
