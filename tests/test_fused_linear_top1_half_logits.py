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


"""return_is_top1 with is_top1_half_logits=True is TRL <= 1.14's token accuracy: the argmax of the model's half logits,
ties to the first column. With fp32 GEMM logits (the CUDA default) it must read them rounded to the half dtype, else
DPO's mean_token_accuracy drifts from TRL's (L4, Qwen3-0.6B bf16, TRL 1.14: 0.7030 vs 0.7107 at step 1). The default
keeps the fp32 argmax, which is TRL 1.15's chunked kernel definition."""

import inspect
import json
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

import unsloth_zoo.fused_losses.fused_linear as fl
from unsloth_zoo.fused_losses.fused_linear import fused_linear_logprobs


def _tied_case(dtype):
    # Column 1's fp32 logit is 1 + 2^-12, column 0's is 1: both round to 1.0 in bf16 / fp16, so TRL's argmax ties
    # to column 0, while the fp32 GEMM output says column 1.
    V = 32
    hidden = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype = dtype)
    weight = torch.zeros(V, 2, dtype = dtype)
    weight[:, 0] = -4.0
    weight[0] = torch.tensor([1.0, 0.0])
    weight[1] = torch.tensor([1.0, 2.0 ** -12])
    targets = torch.tensor([0, 1])
    return hidden, weight, targets


def _trl_top1(hidden, weight, targets):
    return F.linear(hidden, weight).argmax(-1) == targets


def _half_flag():
    # Older zoo without the keyword: run its default, so the test fails on the result rather than the signature.
    has = "is_top1_half_logits" in inspect.signature(fused_linear_logprobs).parameters
    return {"is_top1_half_logits": True} if has else {}


def _fp32_gemm(monkeypatch):
    # CUDA's default logits mode (fp32 GEMM output, UNSLOTH_LOGPROB_FP32_LOGITS=1), on the torch row path.
    monkeypatch.setattr(fl, "_logits_mode", lambda device, dtype = torch.float16: (True, False))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_top1_reads_half_logits_with_fp32_gemm(monkeypatch, dtype):
    hidden, weight, targets = _tied_case(dtype)
    assert (hidden.float() @ weight.float().T).argmax(-1).tolist() == [1, 1]
    expect = _trl_top1(hidden, weight, targets)
    assert expect.tolist() == [True, False]
    _fp32_gemm(monkeypatch)
    *_, top1 = fused_linear_logprobs(hidden, weight, targets, return_entropy = True, return_is_top1 = True,
                                     **_half_flag())
    assert top1.tolist() == expect.tolist()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_top1_default_keeps_fp32_argmax(monkeypatch, dtype):
    # TRL 1.15's chunked kernel: argmax of the fp32 GEMM logits (first max), the default for every other caller.
    hidden, weight, targets = _tied_case(dtype)
    _fp32_gemm(monkeypatch)
    *_, top1 = fused_linear_logprobs(hidden, weight, targets, return_entropy = True, return_is_top1 = True)
    assert top1.tolist() == ((hidden.float() @ weight.float().T).argmax(-1) == targets).tolist() == [False, True]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_top1_half_logits_mode_matches_trl(dtype):
    # Half logits (CPU) and fp32 models were already TRL's argmax: unchanged.
    hidden, weight, targets = _tied_case(dtype)
    *_, top1 = fused_linear_logprobs(hidden, weight, targets, return_is_top1 = True, **_half_flag())
    assert top1.tolist() == _trl_top1(hidden, weight, targets).tolist()


_KERNEL_SCRIPT = r"""
import inspect, json, sys, types, torch
import unsloth_zoo.fused_losses as pkg
path = pkg.__path__[0] + "/fused_linear.py"
# The interpreter has no libdevice and fused_linear disables Triton under it: load the kernels anyway.
src = open(path).read().replace('_HAS_TRITON = os.environ.get("TRITON_INTERPRET", "0") != "1"', "_HAS_TRITON = True")
src = src.replace("from triton.language.extra import libdevice as _libdevice", "raise ImportError")
fl = types.ModuleType("unsloth_zoo.fused_losses.fused_linear")
fl.__file__ = path
sys.modules[fl.__name__] = fl
exec(compile(src, path, "exec"), fl.__dict__)
fl._use_triton = lambda device: True
fl._launch = lambda device, fn: fn()
V = 64
x = torch.full((2, V), -4.0)
x[:, 3] = 1.0
x[:, 5] = 1.0 + 2.0 ** -10
tgt = torch.tensor([3, 5])
out = {"file": path}
for split in (0, 16):  # one row kernel, and four vocab splits through the partial + combine kernels
    if split:
        fl._SPLIT_COLS = split
    has = "compute_dtype" in inspect.signature(fl._row_forward).parameters
    for half in (False, True):  # compute_dtype None: the fp32 argmax (the default); bf16: the half argmax
        calls = fl.CALLS["triton"]
        kw = {"compute_dtype": torch.bfloat16} if half and has else {}
        st, _ = fl._row_forward(torch.device("cpu"), x.clone(), None, tgt, None, None, torch.full((), -1.0),
                                (1.0, 1.0, 0.0, 1.0, 0.0), (True, False, False, True), False, None, **kw)
        assert fl.CALLS["triton"] == calls + 1
        out[f"{split}_{'half' if half else 'fp32'}"] = st["top1"].tolist()
print("RESULT " + json.dumps(out))
"""


def test_top1_kernels_read_half_logits():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET = "1", CUDA_VISIBLE_DEVICES = "", UNSLOTH_IS_PRESENT = "1")
    proc = subprocess.run([sys.executable, "-c", _KERNEL_SCRIPT], env = env, capture_output = True, text = True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(s for s in proc.stdout.splitlines() if s.startswith("RESULT "))
    result = json.loads(line[len("RESULT "):])
    assert result.pop("file") == fl.__file__
    # bf16 ties column 3 and 5 at 1.0: the first one wins, as torch.argmax on TRL's bf16 logits.
    assert result == {"0_fp32": [False, True], "0_half": [True, False], "16_fp32": [False, True],
                      "16_half": [True, False]}
