# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.

"""The `for _ in range(N): gc.collect(); empty_cache()` release loops in vllm_utils and tokenizer_utils stop
once a pass finds nothing, like patch_model_and_tokenizer's. A full pass that found nothing freed nothing and
ran no finalizer, so the next one cannot find anything either; gc.callbacks run around every pass, so with any
registered every pass still runs. CPU only: the loops are sliced out with ast and run against a fake gc."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / "unsloth_zoo"
FILES = ("vllm_utils.py", "tokenizer_utils.py")


def _is_collect(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "collect"
        and getattr(node.func.value, "id", None) == "gc"
    )


def _release_loops():
    for name in FILES:
        tree = ast.parse((ROOT / name).read_text(encoding = "utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and node.target.id == "_"
                and isinstance(node.iter, ast.Call)
                and getattr(node.iter.func, "id", None) == "range"
                and any(_is_collect(c) for c in ast.walk(node))
            ):
                yield name, node


CASES = [
    pytest.param(name, loop, id = f"{name}:{i}")
    for i, (name, loop) in enumerate(_release_loops())
]


def test_every_release_loop_is_covered():
    # vllm_utils: graph capture v1 + v0, convert_vllm_to_huggingface, _release_failed_vllm_engine, load_vllm,
    # _test_get_vllm_state_dict; tokenizer_utils: add_new_tokens, fix_untrained_tokens.
    assert len(CASES) == 8


def _run(loop, found, callbacks = ()):
    results = iter(found)
    collects, empties = [], []

    def collect(*args):
        collects.append(args)
        return next(results)

    fake_gc = SimpleNamespace(collect = collect, callbacks = list(callbacks))
    empty = lambda *a, **k: empties.append(1)
    fake_torch = SimpleNamespace(cuda = SimpleNamespace(empty_cache = empty))
    code = compile(ast.Module([loop], []), "loop", "exec")
    exec(code, {"gc": fake_gc, "torch": fake_torch, "_device_empty_cache": empty})
    return len(collects), len(empties)


@pytest.mark.parametrize("name,loop", CASES)
def test_release_loop_stops_once_a_pass_finds_nothing(name, loop):
    n = loop.iter.args[0].value
    assert _run(loop, [0] + [9] * n) == (1, 1)
    if n >= 3:
        assert _run(loop, [4, 0] + [9] * n) == (2, 2)
    assert _run(loop, [4] * n) == (n, n)


@pytest.mark.parametrize("name,loop", CASES)
def test_release_loop_keeps_every_pass_with_gc_callbacks(name, loop):
    n = loop.iter.args[0].value
    assert _run(loop, [0] * n, callbacks = [print]) == (n, n)


@pytest.mark.parametrize("name,loop", CASES)
def test_release_loop_keeps_every_pass_when_collect_is_replaced(name, loop):
    # patch_vllm_graph_capture swaps gc.collect for `lambda: None` during capture; None is not "found nothing".
    n = loop.iter.args[0].value
    assert _run(loop, [None] * n) == (n, n)
