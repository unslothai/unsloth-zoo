# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`peft_utils` resolved module paths with `eval(f"model.{path}", globals(), ...)`.

The paths come from `model.named_parameters()`, so they are attribute names registered
on live `nn.Module`s rather than checkpoint keys - but `eval` means the whole path is
evaluated as Python with `peft_utils`' globals in scope, and `nn.Module.add_module`
only rejects `""` and names containing `"."`. `_get_module_attribute` is the read
counterpart of the `_set_module_attribute` walk PR #1083 added, and removes the
question entirely.

The removed `eval` is kept here as the oracle, the same way
`test_set_additional_modules_paths.py` keeps the removed `exec`.

CPU-only and network-free.
"""

import pytest
import torch
import torch.nn as nn

from unsloth_zoo.empty_model import _get_module_attribute, _set_module_attribute


def _build_model():
    visual = nn.Module()
    visual.pos_embed = nn.Embedding(4, 8)
    visual.merger = nn.Module()
    visual.merger.ln_q = nn.LayerNorm(8)
    visual.blocks = nn.ModuleList([nn.LayerNorm(8) for _ in range(2)])

    inner = nn.Module()
    inner.visual = visual
    inner.norm = nn.LayerNorm(8)
    inner.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])

    root = nn.Module()
    root.model = inner
    root.lm_head = nn.Linear(8, 4, bias = False)
    return root


# --- equivalence with the removed eval ---------------------------------------

@pytest.mark.parametrize("path", [
    "model",
    "model.visual",
    "model.visual.merger",
    "model.visual.merger.ln_q",
    "model.visual.pos_embed.weight",
    "model.norm.weight",
    "lm_head",
    "lm_head.weight",
])
def test_matches_the_old_eval(path):
    """The walk must return exactly what `eval(f"model.{path}")` returned."""
    model = _build_model()
    reference = eval(f"model.{path}", {}, {"model": model})
    assert _get_module_attribute(model, path) is reference


def test_empty_path_returns_the_root():
    """`eval("model")` gave the model itself; the walk keeps that."""
    model = _build_model()
    assert _get_module_attribute(model, "") is model


# --- what eval could not do --------------------------------------------------

@pytest.mark.parametrize("path", [
    "model.visual.blocks.0",
    "model.visual.blocks.1.weight",
    "model.layers.2.weight",
])
def test_numeric_segments_resolve(path):
    """`blocks.0.weight` is a syntax error for an attribute expression."""
    with pytest.raises(SyntaxError):
        compile(f"model.{path}", "<test>", "eval")

    model = _build_model()
    assert _get_module_attribute(model, path) is not None


@pytest.mark.parametrize("path", [
    "__import__('os')",
    "__import__('os').system('touch /tmp/pwned')",
    "(lambda: 1)()",
    "1 + 1",
    "model.__class__.__init__.__globals__['os']",
])
def test_a_path_component_is_a_name_not_an_expression(path):
    """The whole point. `eval` ran these; the walk looks each one up as an attribute
    name and fails, so no call, no subscript and no operator ever executes."""
    model = _build_model()
    # eval would have happily executed the first three.
    with pytest.raises((AttributeError, IndexError, KeyError, TypeError, ValueError)):
        _get_module_attribute(model, path)


# --- failure is loud ---------------------------------------------------------

@pytest.mark.parametrize("path", [
    "does.not.exist",
    "model.visual.blocks.9",
    "model.visual.blocks.x",
    "model.nope",
])
def test_unresolvable_path_raises(path):
    with pytest.raises((AttributeError, IndexError, KeyError, TypeError, ValueError)):
        _get_module_attribute(_build_model(), path)


# --- getter and setter agree -------------------------------------------------

@pytest.mark.parametrize("path", [
    "model.norm.weight",
    "model.visual.blocks.0.weight",
    "lm_head.weight",
])
def test_round_trip_with_the_setter(path):
    model = _build_model()
    value = nn.Parameter(torch.randn(8), requires_grad = False)
    _set_module_attribute(model, path, value)
    assert _get_module_attribute(model, path) is value
