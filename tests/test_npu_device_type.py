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
if dt._IS_MLX:  # Apple Silicon with mlx answers "mlx" first and never imports torch here
    print("MLX")
    sys.exit(0)
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
    if hasattr(torch, "accelerator"):  # absent before torch 2.6
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

# Gradient checkpointing init on npu; buffers land on cpu since the stub registers no device.
npu.stream = lambda s: s
import unsloth_zoo.gradient_checkpointing as gc
npu.device_count = lambda: 1
npu.Event = lambda *a, **k: "event"
npu.Stream = lambda device = None, **k: ("stream", device)
npu.default_stream = lambda i: "main"
gc.DEVICE_TYPE, gc.DEVICE_TYPE_TORCH = "npu", "cpu"
gc.initialize_unsloth_gradient_checkpointing()
amp = getattr(gc.torch_amp_custom_fwd, "keywords", {}).get("device_type")
try:
    import unsloth_zoo.vllm_utils as vu
except ModuleNotFoundError as e:  # e.g. no triton wheel on Windows
    vu = None
    print("NODEP", e.name)
before = calls.count("empty_cache")
if vu is not None:
    npu.mem_get_info = lambda *a: (1, 2)
    vu.DEVICE_TYPE = "npu"
    print("VLLM", vu.get_mem_info())
    print("FLASHINFER", vu._clear_flashinfer_env_on_hip())
    vu._device_empty_cache()
gc.reset_unsloth_gradient_checkpointing_buffers()
print("CACHE", calls.count("empty_cache") - before)
print("GC", len(gc.GPU_BUFFERS), gc.GPU_BUFFERS[0].dtype, gc.MAIN_STREAMS, gc.EXTRA_STREAMS, amp)
"""

_CELLS = ("cuda", "hip", "xpu", "npu", "xpu+npu", "none")


@pytest.fixture(scope = "module")
def answers():
    env = {k: v for k, v in os.environ.items() if k != "UNSLOTH_ZOO_DISABLE_GPU_INIT"}
    env["UNSLOTH_ALLOW_CPU"] = "1"
    env["UNSLOTH_DISABLE_PINNED_MEMORY"] = "1"  # stub npu has no pinned allocator
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_ROOT), env.get("PYTHONPATH")]))
    out = subprocess.run(
        [sys.executable, "-c", _CHILD, *_CELLS],
        capture_output = True,
        text = True,
        env = env,
        timeout = 900,
    )
    if out.stdout.splitlines()[-1:] == ["MLX"]:
        pytest.skip("an mlx host answers \"mlx\" before any accelerator")
    got = {}
    for line in out.stdout.splitlines():
        if line.startswith("CELL "):
            _, cell, *rest = line.split()
            got[cell] = " ".join(rest)
        elif line.startswith(("HELPERS ", "GC ", "VLLM ", "CACHE ", "NODEP ", "FLASHINFER ")):
            key, _, rest = line.partition(" ")
            got[key.lower()] = rest
    assert set(got) >= set(_CELLS), out.stderr[-2000:]
    got["stderr"] = out.stderr[-2000:]
    return got


def test_an_ascend_host_gets_npu(answers):
    assert answers["npu"] == "OK npu"


def test_npu_helpers_reach_torch_npu(answers):
    assert answers.get("helpers") == "synchronize,empty_cache True"


def test_npu_vllm_memory_reads_torch_npu(answers):
    if "nodep" in answers:
        pytest.skip(f"vllm_utils needs {answers['nodep']}")
    assert answers.get("vllm") == "(1, 2)", answers["stderr"]


def test_npu_cache_cleanup_reaches_torch_npu(answers):
    if "nodep" in answers:
        pytest.skip(f"vllm_utils needs {answers['nodep']}")
    assert answers.get("cache") == "2", answers["stderr"]


def test_npu_skips_the_cuda_flashinfer_setup(answers):
    if "nodep" in answers:
        pytest.skip(f"vllm_utils needs {answers['nodep']}")
    assert answers.get("flashinfer") == "True", answers["stderr"]


def test_npu_gradient_checkpointing_initializes(answers):
    assert answers.get("gc") == "1 torch.bfloat16 ('main',) (('stream', 0),) npu", answers["stderr"]


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


def test_unsloth_train_takes_its_device_from_the_backend():
    # Source-level: importing training_utils needs datasets.
    import ast
    tree = ast.parse((_ROOT / "unsloth_zoo" / "training_utils.py").read_text(encoding = "utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "unsloth_train")
    passed = [
        a.value for c in ast.walk(fn) if isinstance(c, ast.Call)
        for a in [*c.args, *(k.value for k in c.keywords)]
        if isinstance(a, ast.Constant)
    ]
    assert not {"cuda", "cuda:0"} & set(passed)
