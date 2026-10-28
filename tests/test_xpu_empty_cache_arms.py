# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.

"""On XPU the smart gradient checkpointing buffers and patch_model_and_tokenizer release cached
memory through torch.xpu, not torch.cuda (a no-op there). CPU only: both empty_cache calls are counted."""

from __future__ import annotations

import ast
import gc as real_gc
import importlib.util
import threading
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
    """patch_model_and_tokenizer's closing release statement, sliced with ast."""
    tree = ast.parse((ROOT / "patching_utils.py").read_text(encoding = "utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "patch_model_and_tokenizer")
    node = next(
        n for n in fn.body
        if (isinstance(n, ast.For) and "empty_cache" in ast.unparse(n))
        or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            and getattr(n.value.func, "id", None) == "collect_garbage")
    )
    return compile(ast.Module([node], []), "patching_utils.py", "exec")


def _helper(gc):
    """unsloth_zoo/_gc_passes.py loaded on its own (no torch import), reading `gc`."""
    path = ROOT / "_gc_passes.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_gc_passes_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.gc = gc
    return module.collect_garbage


class _FakeGC:
    """`script` items: an int or None is what a pass that runs returns; "skip" returns 0 without collecting
    (another thread's collection is in progress), leaving the full-collection stats unchanged."""

    def __init__(self, script, callbacks = ()):
        self.script = iter(script)
        self.callbacks = list(callbacks)
        self.stats = {"collections": 0, "collected": 0, "uncollectable": 0}

    def get_stats(self):
        return [{}, {}, dict(self.stats)]

    def collect(self):
        result = next(self.script)
        if result == "skip":
            return 0
        self.stats["collections"] += 1
        self.stats["collected"] += result or 0
        return result


def _release(gc, fake_torch, device_type):
    namespace = {"gc": gc, "torch": fake_torch, "DEVICE_TYPE": device_type}
    helper = _helper(gc)
    if helper is not None:
        namespace["collect_garbage"] = helper
    exec(_patch_model_final_release(), namespace)


@pytest.mark.parametrize("device_type,expected", [("xpu", "xpu"), ("cuda", "cuda"), ("hip", "cuda")])
def test_patch_model_and_tokenizer_releases_through_the_active_backend(device_type, expected):
    seen = []
    fake_torch = SimpleNamespace(
        cuda = SimpleNamespace(empty_cache = lambda: seen.append("cuda")),
        xpu = SimpleNamespace(empty_cache = lambda: seen.append("xpu")),
    )
    _release(_FakeGC([1, 1, 1]), fake_torch, device_type)
    assert seen == [expected] * 3


@pytest.mark.parametrize(
    "found,expected",
    [([5, 0, 7], 2), ([0, 9, 9], 1), ([4, 2, 1], 3), ([None, None, None], 3),
     # A pass skipped for another thread's collection returns 0 too; it is not proof the heap is quiet.
     (["skip", 0, 9], 2), (["skip", "skip", "skip"], 3)],
)
def test_patch_model_and_tokenizer_stops_once_a_pass_finds_nothing(found, expected):
    # A full pass that ran and found nothing changed nothing, so the next one cannot find anything either.
    seen = []
    fake_torch = SimpleNamespace(cuda = SimpleNamespace(empty_cache = lambda: seen.append("cuda")))
    _release(_FakeGC(found), fake_torch, "cuda")
    # empty_cache still follows every collect that ran.
    assert seen == ["cuda"] * expected


def test_patch_model_and_tokenizer_keeps_every_pass_when_gc_callbacks_are_registered():
    # A gc callback runs around each pass and may drop references, so a 0 no longer proves the heap is quiet.
    seen = []
    fake_torch = SimpleNamespace(cuda = SimpleNamespace(empty_cache = lambda: seen.append("cuda")))
    _release(_FakeGC([0, 0, 0], callbacks = [print]), fake_torch, "cuda")
    assert seen == ["cuda"] * 3


def test_patch_model_and_tokenizer_keeps_collecting_while_another_thread_collects():
    # Real gc. Thread A is inside a collection, blocked in a finalizer, when the release starts: the first
    # gc.collect() returns 0 without collecting. Stopping there would leave our garbage behind.
    started, release = threading.Event(), threading.Event()

    class Slow:
        def __init__(self):
            self.me = self

        def __del__(self):
            started.set()
            release.wait(30)

    class Ours:
        def __init__(self):
            self.me = self

    seen = []

    def empty_cache():
        # Thread A finishes during the first flush, between our first and second pass.
        release.set()
        a.join(30)
        seen.append("cuda")

    fake_torch = SimpleNamespace(cuda = SimpleNamespace(empty_cache = empty_cache))
    was_enabled = real_gc.isenabled()
    real_gc.disable()
    a = threading.Thread(target = real_gc.collect)
    try:
        real_gc.collect()
        Slow()
        a.start()
        assert started.wait(30)
        for _ in range(5):
            Ours()
        _release(real_gc, fake_torch, "cuda")
        # Skipped, then found ours, then quiet.
        assert seen == ["cuda"] * 3
        assert not any(type(o) is Ours for o in real_gc.get_objects())
    finally:
        release.set()
        a.join(30)
        if was_enabled:
            real_gc.enable()
