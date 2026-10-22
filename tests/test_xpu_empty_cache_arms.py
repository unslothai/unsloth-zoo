# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.

"""On XPU the smart gradient checkpointing buffers and patch_model_and_tokenizer release cached
memory through torch.xpu, not torch.cuda (a no-op there). CPU only: both empty_cache calls are counted."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "unsloth_zoo"


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: seen.append("cuda"))
    monkeypatch.setattr(torch.xpu, "empty_cache", lambda: seen.append("xpu"), raising = False)
    return seen


@pytest.mark.parametrize("device_type,expected", [("xpu", "xpu"), ("cuda", "cuda"), ("hip", "cuda")])
def test_reset_buffers_releases_through_the_active_backend(monkeypatch, calls, device_type, expected):
    gc_mod = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
    monkeypatch.setattr(gc_mod, "DEVICE_TYPE", device_type)
    monkeypatch.setattr(gc_mod, "CPU_BUFFERS", [torch.empty(4)], raising = False)
    monkeypatch.setattr(gc_mod, "GPU_BUFFERS", (torch.empty(4),), raising = False)
    monkeypatch.setattr(gc_mod, "GPU_BUFFERS_B", (torch.empty(4),), raising = False)
    monkeypatch.setattr(gc_mod, "NEXT_BUFFER_SLOT", None, raising = False)
    monkeypatch.setattr(gc_mod, "_double_buffer_disabled", lambda: True)
    gc_mod.reset_unsloth_gradient_checkpointing_buffers()
    assert calls == [expected]


def test_unpatch_releases_through_xpu(monkeypatch, calls):
    gc_mod = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
    cp = torch.utils.checkpoint
    UnslothCheckpointFunction = type("UnslothCheckpointFunction", (), {})
    monkeypatch.setattr(cp, "CheckpointFunction", UnslothCheckpointFunction)
    monkeypatch.setattr(cp, "_old_CheckpointFunction", object, raising = False)
    monkeypatch.setattr(gc_mod, "DEVICE_TYPE", "xpu")
    for name, value in {"CPU_BUFFERS": [torch.empty(4)], "GPU_BUFFERS": (torch.empty(4),),
                        "GPU_BUFFERS_B": None, "BUFFER_EVENTS_A": None, "BUFFER_EVENTS_B": None,
                        "NEXT_BUFFER_SLOT": None}.items():
        monkeypatch.setattr(gc_mod, name, value, raising = False)
    gc_mod.unpatch_unsloth_smart_gradient_checkpointing()
    assert calls == ["xpu"]


def _patch_model_final_release():
    """patch_model_and_tokenizer's closing `for _ in range(3)` release loop, sliced with ast."""
    tree = ast.parse((ROOT / "patching_utils.py").read_text(encoding = "utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "patch_model_and_tokenizer")
    loop = next(n for n in fn.body if isinstance(n, ast.For) and "empty_cache" in ast.unparse(n))
    return compile(ast.Module([loop], []), "patching_utils.py", "exec")


@pytest.mark.parametrize("device_type,expected", [("xpu", "xpu"), ("cuda", "cuda"), ("hip", "cuda")])
def test_patch_model_and_tokenizer_releases_through_the_active_backend(device_type, expected):
    seen = []
    fake_torch = SimpleNamespace(
        cuda = SimpleNamespace(empty_cache = lambda: seen.append("cuda")),
        xpu = SimpleNamespace(empty_cache = lambda: seen.append("xpu")),
    )
    exec(_patch_model_final_release(), {"gc": SimpleNamespace(collect = lambda: 0), "torch": fake_torch,
                                        "DEVICE_TYPE": device_type})
    assert seen == [expected] * 3
