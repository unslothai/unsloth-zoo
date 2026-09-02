# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`set_additional_modules` used to assign weights via `exec(f"{prefix}{key} = val")`.

The prefixes "new_" and "new_model." were a string trick: "new_" + "model.visual.x" spells
`new_model.visual.x`, so they meant "key with the leading `model.` consumed" and "key as is".
The walk keeps both candidates in that order and additionally handles numeric segments, since
`a.0.weight` is a syntax error for exec and such keys were silently skipped.

CPU-only and network-free.
"""

import pytest
import torch
import torch.nn as nn

from unsloth_zoo.empty_model import _set_module_attribute


def _build_model():
    visual = nn.Module()
    visual.pos_embed = nn.Embedding(4, 8)
    visual.merger = nn.Module()
    visual.merger.ln_q = nn.LayerNorm(8)
    visual.blocks = nn.ModuleList([nn.LayerNorm(8) for _ in range(2)])

    inner = nn.Module()
    inner.visual = visual
    inner.norm = nn.LayerNorm(8)

    root = nn.Module()
    root.model = inner
    root.lm_head = nn.Linear(8, 4, bias = False)
    return root


def _assign(model, key, value):
    """Mirrors the candidate order used by set_additional_modules."""
    candidates = ([key[len("model."):]] if key.startswith("model.") else []) + [key]
    for path in candidates:
        try:
            _set_module_attribute(model, path, value)
            return path
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            continue
    return None


@pytest.mark.parametrize("key", [
    "model.visual.pos_embed.weight",
    "model.visual.merger.ln_q.weight",
    "model.visual.merger.ln_q.bias",
    "model.norm.weight",
    "lm_head.weight",
])
def test_matches_old_exec_placement(key):
    """The value must land where `exec(f"new_model.{key} = val")` used to put it."""
    value = torch.nn.Parameter(torch.randn(3), requires_grad = False)

    reference = _build_model()
    exec(f"new_model.{key} = val", {"new_model": reference, "val": value})

    got = _build_model()
    assert _assign(got, key, value) is not None

    leaf_reference, leaf_got = reference, got
    parts = key.split(".")
    for part in parts[:-1]:
        leaf_reference = getattr(leaf_reference, part)
        leaf_got = getattr(leaf_got, part)
    assert getattr(leaf_reference, parts[-1]) is value
    assert getattr(leaf_got, parts[-1]) is value


def test_numeric_index_now_assigned():
    """`blocks.0.weight` cannot be expressed as an exec assignment."""
    key = "model.visual.blocks.0.weight"
    value = torch.nn.Parameter(torch.randn(8), requires_grad = False)

    with pytest.raises(SyntaxError):
        compile(f"new_model.{key} = val", "<test>", "exec")

    model = _build_model()
    assert _assign(model, key, value) is not None
    assert model.model.visual.blocks[0].weight is value


@pytest.mark.parametrize("path", [
    "does.not.exist",
    "model.visual.blocks.9.weight",
    "visual.blocks.x.weight",
])
def test_unresolvable_path_raises(path):
    """Failures must surface as catchable errors so the caller can report them
    instead of the old bare `except:` swallowing everything."""
    with pytest.raises((AttributeError, IndexError, KeyError, TypeError, ValueError)):
        _set_module_attribute(_build_model(), path, torch.zeros(1))
