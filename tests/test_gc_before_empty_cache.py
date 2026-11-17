# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.

"""Buffer teardown must run gc.collect() before empty_cache(): a cycle-only tensor is still allocated then, so its block stays cached until the next empty_cache()."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "gradient_checkpointing.py"


def _call_name(node):
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


@pytest.mark.parametrize(
    "function_name",
    ["unpatch_unsloth_smart_gradient_checkpointing", "reset_unsloth_gradient_checkpointing_buffers"],
)
def test_gc_collect_runs_before_empty_cache(function_name):
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    calls = sorted(
        (node.lineno, _call_name(node))
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _call_name(node) in ("collect", "empty_cache")
    )
    collects = [line for line, name in calls if name == "collect"]
    empties = [line for line, name in calls if name == "empty_cache"]
    assert collects and empties, f"{function_name} no longer frees caches"
    # The last teardown pair: the collect must come first.
    assert max(collects) < max(empties), calls
