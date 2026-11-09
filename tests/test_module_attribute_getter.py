# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests _get_module_attribute, the read side of the PR 1083 attribute walk that replaces the peft_utils eval of module
paths; eval ran them as Python with globals in scope; eval kept as oracle; CPU-only, network-free"""

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
    """Path components are looked up as attribute names, so no call, subscript or operator runs where eval would have run them; each path must raise."""
    model = _build_model()
    with pytest.raises((AttributeError, IndexError, KeyError, TypeError, ValueError)):
        _get_module_attribute(model, path)


@pytest.mark.parametrize("path", [
    "does.not.exist",
    "model.visual.blocks.9",
    "model.visual.blocks.x",
    "model.nope",
])
def test_unresolvable_path_raises(path):
    with pytest.raises((AttributeError, IndexError, KeyError, TypeError, ValueError)):
        _get_module_attribute(_build_model(), path)


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


# Bracket components, from the parallel review of PR #1108.

def test_bracket_components_resolve_like_eval():
    """peft_utils rewrites .0. to [0] and one call site keeps brackets; a bare getattr cannot index, so name[0] must
    resolve or gradient checkpointing breaks for models calling self.blocks[0] directly."""
    import torch
    from unsloth_zoo.empty_model import _get_module_attribute

    class Inner(torch.nn.Module):
        def __init__(self): 
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Inner(), Inner()])

    model = Outer()
    for path in ("blocks[0]", "blocks[1]", "blocks[0].weight"):
        oracle = eval("model." + path, {}, {"model": model})
        assert _get_module_attribute(model, path) is oracle, path

    assert _get_module_attribute(model, "blocks.0") is _get_module_attribute(model, "blocks[0]")


def test_bracket_components_stay_a_name_and_index_walk():
    """Only digits are accepted inside the brackets, so this is not an expression."""
    import torch
    from unsloth_zoo.empty_model import _get_module_attribute

    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Identity()])
    for hostile in ("blocks[0+0]", "blocks[x]", "blocks[-1]", "blocks['a']", "blocks[()]"):
        try:
            _get_module_attribute(model, hostile)
        except Exception:
            continue
        raise AssertionError(f"{hostile!r} resolved, so a component is being evaluated")


def test_a_unicode_named_submodule_resolves():
    """Python identifiers may be non-ASCII and PyTorch accepts them; an ASCII-only pattern for the name in name[0] would
    fall through to getattr on the whole 层[0] string and fail."""
    import torch

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class Unicode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.层 = torch.nn.ModuleList([Block(), Block()])

    from unsloth_zoo.empty_model import _get_module_attribute

    model = Unicode()
    for path in ("层[0]", "层[1]", "层[0].weight"):
        oracle = eval("model." + path, {}, {"model": model})
        assert _get_module_attribute(model, path) is oracle, path
    # The dotted-index spelling too, which eval could never parse at all.
    assert _get_module_attribute(model, "层.0") is model.层[0]
    # And the subscript is still digits only.
    for hostile in ("层[x]", "层[0+0]", "层[-1]", "层['a']"):
        try:
            _get_module_attribute(model, hostile)
        except Exception:
            continue
        raise AssertionError(f"{hostile!r} resolved")
