# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.
"""An Ascend host must get DEVICE_TYPE "npu" instead of raising at import."""

import os
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("torch")

_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Import under UNSLOTH_ALLOW_CPU=1 for GPU-less runners, then drop it or the none cell answers "cuda".
_CHILD = r"""
import os, sys, types
import unsloth_zoo.device_type as dt
import torch

os.environ.pop("UNSLOTH_ALLOW_CPU", None)
for cell in sys.argv[1:]:
    torch.cuda.is_available = lambda cell=cell: cell in ("cuda", "hip")
    torch.version.hip = "7.2.0" if cell == "hip" else None
    dt.is_hip.cache_clear()
    if hasattr(torch, "xpu"):
        torch.xpu.is_available = lambda cell=cell: cell in ("xpu", "xpu+npu")
    if cell in ("npu", "xpu+npu"):
        npu = types.ModuleType("torch.npu")
        npu.is_available = lambda: True
        torch.npu = npu
    elif hasattr(torch, "npu"):
        del torch.npu
    if hasattr(dt, "npu_is_available"):
        dt.npu_is_available.cache_clear()
    acc = {"npu": "npu", "xpu+npu": "xpu", "cuda": "cuda", "hip": "cuda"}.get(cell)
    torch.accelerator.is_available = lambda acc=acc: acc is not None
    torch.accelerator.current_accelerator = lambda acc=acc: acc
    try:
        print("CELL", cell, "OK", dt.get_device_type.__wrapped__())
    except NotImplementedError:
        print("CELL", cell, "RAISE")

# The helpers must reach torch.npu once npu is selected, not fall through to a no-op.
calls = []
npu = types.ModuleType("torch.npu")
npu.is_available = lambda: True
npu.synchronize = lambda: calls.append("synchronize")
npu.empty_cache = lambda: calls.append("empty_cache")
npu.is_bf16_supported = lambda: True
torch.npu = npu
dt.npu_is_available.cache_clear()
dt.DEVICE_TYPE = "npu"
dt.device_synchronize(); dt.device_empty_cache()
print("HELPERS", ",".join(calls), dt.device_is_bf16_supported())
"""

_CELLS = ("cuda", "hip", "xpu", "npu", "xpu+npu", "none")


@pytest.fixture(scope = "module")
def answers():
    env = {k: v for k, v in os.environ.items() if k != "UNSLOTH_ZOO_DISABLE_GPU_INIT"}
    env["UNSLOTH_ALLOW_CPU"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_ROOT), env.get("PYTHONPATH")]))
    out = subprocess.run(
        [sys.executable, "-c", _CHILD, *_CELLS],
        capture_output = True,
        text = True,
        env = env,
        timeout = 900,
    )
    got = {}
    for line in out.stdout.splitlines():
        if line.startswith("CELL "):
            _, cell, *rest = line.split()
            got[cell] = " ".join(rest)
        elif line.startswith("HELPERS "):
            got["helpers"] = line[len("HELPERS "):]
    assert set(got) >= set(_CELLS), out.stderr[-2000:]
    return got


def test_an_ascend_host_gets_npu(answers):
    assert answers["npu"] == "OK npu"


def test_npu_helpers_reach_torch_npu(answers):
    assert answers.get("helpers") == "synchronize,empty_cache True"


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("cuda", "OK cuda"),
        ("hip", "OK hip"),
        ("xpu", "OK xpu"),
        ("xpu+npu", "OK xpu"),
        ("none", "RAISE"),
    ],
)
def test_every_other_host_answers_as_before(answers, cell, expected):
    assert answers[cell] == expected
