# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.

"""The repeated `gc.collect(); empty_cache()` release passes in vllm_utils and tokenizer_utils go through
_gc_passes.collect_garbage, which stops once a pass that really ran finds nothing. Such a pass freed nothing
and ran no finalizer, so the next one cannot find anything either. Anything else keeps every pass: gc.callbacks
registered, a non-zero or None result, or a pass skipped because another thread was collecting."""

from __future__ import annotations

import ast
import gc as real_gc
import importlib.util
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "unsloth_zoo"


def _load():
    # The module only imports gc: load it on its own, without the package's torch imports.
    spec = importlib.util.spec_from_file_location("_gc_passes_under_test", ROOT / "_gc_passes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeGC:
    """`script` items: an int or None is what a pass that runs returns; "skip" returns 0 without collecting."""

    def __init__(self, script, callbacks = ()):
        self.script = iter(script)
        self.callbacks = list(callbacks)
        self.stats = {"collections": 0, "collected": 0, "uncollectable": 0}
        self.calls = 0

    def get_stats(self):
        return [{}, {}, dict(self.stats)]

    def collect(self):
        self.calls += 1
        result = next(self.script)
        if result == "skip":
            return 0
        self.stats["collections"] += 1
        self.stats["collected"] += result or 0
        return result


@pytest.mark.parametrize("passes", [2, 3])
@pytest.mark.parametrize(
    "script,expected",
    [
        ([5, 0, 7], 2),
        ([0, 9, 9], 1),
        ([4, 2, 1], 3),
        ([None, None, None], 3),  # gc.collect swapped for `lambda: None` (vLLM graph capture)
        (["skip", 0, 9], 2),
        (["skip", "skip", "skip"], 3),
    ],
)
def test_stops_after_a_pass_that_ran_and_found_nothing(monkeypatch, passes, script, expected):
    module = _load()
    gc = _FakeGC(script)
    monkeypatch.setattr(module, "gc", gc)
    empties = []
    module.collect_garbage(passes = passes, after = lambda: empties.append(1))
    assert gc.calls == min(expected, passes)
    assert len(empties) == gc.calls


def test_keeps_every_pass_with_gc_callbacks(monkeypatch):
    module = _load()
    gc = _FakeGC([0, 0, 0], callbacks = [print])
    monkeypatch.setattr(module, "gc", gc)
    module.collect_garbage()
    assert gc.calls == 3


def test_a_pass_skipped_for_another_threads_collection_does_not_end_it():
    collect_garbage = _load().collect_garbage
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
        passes = []

        def after():
            # Thread A finishes between our first and second pass.
            release.set()
            a.join(30)
            passes.append(1)

        collect_garbage(after = after)
        # Skipped, then found ours, then quiet.
        assert len(passes) == 3
        assert not any(type(o) is Ours for o in real_gc.get_objects())
    finally:
        release.set()
        a.join(30)
        if was_enabled:
            real_gc.enable()


def _is_collect(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "collect"
        and getattr(node.func.value, "id", None) == "gc"
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        # graph capture v1 + v0 (2 passes), convert_vllm_to_huggingface, _release_failed_vllm_engine, load_vllm,
        # _test_get_vllm_state_dict
        ("vllm_utils.py", [2, 2, 3, 3, 3, 3]),
        ("tokenizer_utils.py", [3, 3]),  # add_new_tokens, fix_untrained_tokens
    ],
)
def test_release_passes_go_through_the_helper(name, expected):
    tree = ast.parse((ROOT / name).read_text(encoding = "utf-8"))
    loops = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id == "_"
        and any(_is_collect(c) for c in ast.walk(n))
    ]
    assert loops == []
    calls = sorted(
        (n.lineno, next((k.value.value for k in n.keywords if k.arg == "passes"), 3))
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "collect_garbage"
    )
    assert [p for _, p in calls] == expected
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "collect_garbage":
            assert [k.arg for k in n.keywords if k.arg == "after"] == ["after"]
