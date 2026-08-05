# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The gradient-checkpointing hooks must not run while Dynamo is tracing.

requires_grad_pre_hook / requires_grad_post_hook call `requires_grad_()`,
which Dynamo cannot trace:

    Unsupported: Unsupported Tensor.requires_grad_() call

Outside a fullgraph region that is only a graph break, but Gemma 3N puts a
LoRA target (embed_audio.embedding_projection) inside a forward that
temporary_patches/gemma3n.py compiles with fullgraph = True, so the same hook
becomes a hard error and trainer.train() dies. torch._dynamo.disable() does
not rescue it either -- under fullgraph it turns into "Skip calling
torch.compiler.disable()d function". Guarding on
`torch.compiler.is_compiling()` does, and is a no-op in eager.

These tests use the real hooks, pulled off a model that
requires_grad_for_gradient_checkpointing() has actually hooked, so they break
if the hooks are renamed or moved. CPU only.
"""

import ast
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from unsloth_zoo.peft_utils import requires_grad_for_gradient_checkpointing

_ZOO = Path(__file__).resolve().parents[1] / "unsloth_zoo"


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias = False)

    def forward(self, hidden_states):
        return self.proj(hidden_states)


class _PreHookModel(nn.Module):
    """`self.proj(` inside _Inner.forward makes proj a pre-hook target."""
    def __init__(self):
        super().__init__()
        self.vision = _Inner()

    def forward(self, hidden_states):
        return self.vision(hidden_states)


class _PostHookModel(nn.Module):
    """`head` is never called as `self.head(`, so it becomes a fallback
    (post-hook) target instead."""
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(8, 8, bias = False)

    def forward(self, hidden_states):
        return torch.nn.functional.linear(hidden_states, self.head.weight)


def _real_pre_hook():
    model = _PreHookModel()
    model.requires_grad_(False)
    model.vision.proj.weight.requires_grad_(True)
    requires_grad_for_gradient_checkpointing(model)
    hooks = list(model.vision.proj._forward_pre_hooks.values())
    assert len(hooks) == 1, hooks
    assert "requires_grad_pre_hook" in hooks[0].__qualname__
    return hooks[0]


def _real_post_hook():
    model = _PostHookModel()
    model.requires_grad_(False)
    model.head.weight.requires_grad_(True)
    requires_grad_for_gradient_checkpointing(model)
    hooks = list(model.head._forward_hooks.values())
    assert len(hooks) == 1, hooks
    assert "requires_grad_post_hook" in hooks[0].__qualname__
    return hooks[0]


# ---------------------------------------------------------------- eager path

def test_pre_hook_still_fires_eagerly():
    hook = _real_pre_hook()
    x = torch.randn(2, 8)
    hook(None, (x,), {})
    assert x.requires_grad


def test_pre_hook_still_fires_eagerly_on_kwargs():
    hook = _real_pre_hook()
    x = torch.randn(2, 8)
    hook(None, (), {"inputs_embeds" : x})
    assert x.requires_grad


def test_post_hook_still_fires_eagerly():
    hook = _real_post_hook()
    y = torch.randn(2, 8)
    hook(None, None, y)
    assert y.requires_grad


# -------------------------------------------------------------- under compile

def test_pre_hook_no_ops_while_compiling(monkeypatch):
    hook = _real_pre_hook()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    x = torch.randn(2, 8)
    hook(None, (x,), {})
    assert not x.requires_grad


def test_post_hook_no_ops_while_compiling(monkeypatch):
    hook = _real_post_hook()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    y = torch.randn(2, 8)
    hook(None, None, y)
    assert not y.requires_grad


def test_post_hook_does_not_raise_on_unknown_output_while_compiling(monkeypatch):
    # Eagerly this output shape raises "Neither loss, logits, nor
    # last_hidden_state are available"; under compile it must be skipped.
    hook = _real_post_hook()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    hook(None, None, {"not" : "a tensor"})


def test_fullgraph_compiled_module_with_pre_hook_runs():
    # The Gemma 3N failure, reduced: a hooked module inside a
    # fullgraph = True region. Before the guard this raised
    # "Unsupported: Unsupported Tensor.requires_grad_() call".
    torch._dynamo.reset()
    model = _PreHookModel()
    model.requires_grad_(False)
    model.vision.proj.weight.requires_grad_(True)
    requires_grad_for_gradient_checkpointing(model)

    compiled = torch.compile(
        _PreHookModel.forward.__get__(model), fullgraph = True, dynamic = True,
    )
    x = torch.randn(2, 8)
    out = compiled(x)
    assert out.shape == (2, 8)
    # The LoRA-style trainable weight still carries the graph.
    assert out.requires_grad


def test_fullgraph_compiled_module_with_post_hook_runs():
    torch._dynamo.reset()
    model = _PostHookModel()
    model.requires_grad_(False)
    model.head.weight.requires_grad_(True)
    requires_grad_for_gradient_checkpointing(model)

    compiled = torch.compile(
        _PostHookModel.forward.__get__(model), fullgraph = True, dynamic = True,
    )
    out = compiled(torch.randn(2, 8))
    assert out.shape == (2, 8)


# ---------------------------------------------------------------- source shape

def _guarded_functions(path, names):
    """Which of `names` open with `if torch.compiler.is_compiling(): return`?"""
    src = path.read_text(encoding = "utf-8")
    found = OrderedDict()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.FunctionDef) or node.name not in names:
            continue
        body = [n for n in node.body if not (
            isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
        )]
        first = body[0] if body else None
        found[node.name] = (
            isinstance(first, ast.If)
            and "is_compiling" in ast.dump(first.test)
            and any(isinstance(n, ast.Return) for n in first.body)
        )
    return found


def test_both_gradient_checkpointing_hooks_are_guarded():
    names = ("requires_grad_pre_hook", "requires_grad_post_hook")
    guarded = _guarded_functions(_ZOO / "peft_utils.py", names)
    assert set(guarded) == set(names), guarded
    assert all(guarded.values()), guarded


def test_make_inputs_require_grad_is_NOT_guarded():
    """The peft_utils hooks sit on LoRA modules whose output already requires
    grad, so skipping them under tracing changes nothing. This one is the only
    thing making a FROZEN embedding's output require grad, and reentrant
    gradient checkpointing needs at least one grad-carrying input. Guarding it
    would leave that tensor detached and silently stop gradients reaching the
    adapters, which is worse than the graph break it would have avoided.
    """
    guarded = _guarded_functions(
        _ZOO / "training_utils.py", ("make_inputs_require_grad",),
    )
    assert guarded == {"make_inputs_require_grad" : False}, guarded


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
