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

"""The gradient-checkpointing hooks must not flip requires_grad while Dynamo traces.

Dynamo rejects `requires_grad_()` when it would change the flag, and Gemma 3N compiles
a LoRA target with fullgraph = True, so it is a hard error rather than a graph break.
torch._dynamo.disable() does not rescue it; `torch.compiler.is_compiling()` does.

The post hook also lands on `get_input_embeddings()`, where it is the only thing making
a FROZEN embedding's output require grad, so it may skip only the no-op case.

Hooks are pulled off a real hooked model, so these break on a rename. CPU only.
"""

import ast
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

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
    """`head` is never called as `self.head(`, so it is a post-hook target."""
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(8, 8, bias = False)

    def forward(self, hidden_states):
        return torch.nn.functional.linear(hidden_states, self.head.weight)


class _Trunk(nn.Module):
    """Never called as `self.adapter(`, so the fallback post hook lands here, on a
    tuple-returning module like a decoder layer."""
    def __init__(self):
        super().__init__()
        self.adapter = nn.Linear(8, 8, bias = False)

    def forward(self, hidden_states):
        return (torch.nn.functional.linear(hidden_states, self.adapter.weight), None)


class _TupleOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = _Trunk()

    def forward(self, hidden_states):
        return getattr(self, "trunk")(hidden_states)[0]


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Linear(8, 8, bias = False)   # the only trainable weight

    def forward(self, hidden_states):
        return self.adapter(hidden_states)


class _LanguageModel(nn.Module):
    """`for layer in self.layers:` makes this the hook target, and get_input_embeddings()
    puts the post hook on the FROZEN embedding."""
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.layers = nn.ModuleList([_Layer()])

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class _EmbeddingHookModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()

    def forward(self, input_ids):
        return self.language_model(input_ids)


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


def test_post_hook_no_ops_while_compiling_when_output_already_requires_grad(monkeypatch):
    hook = _real_post_hook()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    y = torch.randn(2, 8, requires_grad = True)
    hook(None, None, y)  # nothing to flip, so nothing Dynamo could reject
    assert y.requires_grad


def test_post_hook_still_flips_a_frozen_output_while_compiling(monkeypatch):
    """On `get_input_embeddings()` this is the only thing making a FROZEN embedding's
    output require grad, so only the no-op case may be skipped."""
    hook = _real_post_hook()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    y = torch.randn(2, 8)
    hook(None, None, y)
    assert y.requires_grad


def test_post_hook_still_raises_on_unknown_output_while_compiling(monkeypatch):
    """An unmarkable output must stay a loud error while tracing: is_compiling() is
    constant folded, so a skip is permanent and the region trains with no gradients."""
    hook = _real_post_hook()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(RuntimeError, match = "Failed to make output require gradients"):
        hook(None, None, {"not" : "a tensor"})


def test_compiled_tuple_output_still_raises():
    """End to end: a fallback post-hook target returning a tuple, under compile.
    requires_grad_for_gradient_checkpointing picks this target itself, so the shape is
    one Unsloth produces and the error must survive tracing."""
    torch._dynamo.reset()
    model = _TupleOutputModel()
    model.requires_grad_(False)
    model.trunk.adapter.weight.requires_grad_(True)
    requires_grad_for_gradient_checkpointing(model)
    hooks = list(model.trunk._forward_hooks.values())
    assert len(hooks) == 1, hooks
    assert "requires_grad_post_hook" in hooks[0].__qualname__

    x = torch.randn(2, 8)
    with pytest.raises(RuntimeError, match = "Failed to make output require gradients"):
        model(x)
    with pytest.raises(RuntimeError, match = "Failed to make output require gradients"):
        torch.compile(model, backend = "aot_eager")(x)


def test_fullgraph_compiled_module_with_pre_hook_runs():
    # The Gemma 3N failure, reduced: a hooked module in a fullgraph = True
    # region. Before the guard this raised "Unsupported Tensor.requires_grad_()".
    torch._dynamo.reset()
    model = _PreHookModel()
    model.requires_grad_(False)
    model.vision.proj.weight.requires_grad_(True)
    requires_grad_for_gradient_checkpointing(model)

    # aot_eager, like the other compiles in this file. What is under test is
    # whether Dynamo can trace the hook, not what Inductor emits, and Inductor
    # needs triton: on Apple Silicon it is stubbed out, so the default backend
    # fails here with a NotImplementedError that says nothing about the hook.
    compiled = torch.compile(
        _PreHookModel.forward.__get__(model),
        fullgraph = True, dynamic = True, backend = "aot_eager",
    )
    x = torch.randn(2, 8)
    out = compiled(x)
    assert out.shape == (2, 8)
    assert out.requires_grad


def test_fullgraph_compiled_module_with_post_hook_runs():
    torch._dynamo.reset()
    model = _PostHookModel()
    model.requires_grad_(False)
    model.head.weight.requires_grad_(True)
    requires_grad_for_gradient_checkpointing(model)

    compiled = torch.compile(
        _PostHookModel.forward.__get__(model),
        fullgraph = True, dynamic = True, backend = "aot_eager",
    )
    out = compiled(torch.randn(2, 8))
    assert out.shape == (2, 8)


def test_compiled_frozen_embedding_still_carries_gradients():
    """Compile the frozen embedding, then feed it to a reentrant-checkpointed trainable
    layer. If the hook no-ops while tracing, the adapter gets no gradient at all."""
    torch._dynamo.reset()
    model = _EmbeddingHookModel()
    model.requires_grad_(False)
    layer = model.language_model.layers[0]
    layer.adapter.weight.requires_grad_(True)
    requires_grad_for_gradient_checkpointing(model)

    embedding = model.language_model.embed_tokens
    hooks = list(embedding._forward_hooks.values())
    assert len(hooks) == 1, hooks
    assert "requires_grad_post_hook" in hooks[0].__qualname__
    assert not embedding.weight.requires_grad

    # aot_eager keeps this off inductor's C++ codegen; the hook behaviour is the same.
    hidden_states = torch.compile(embedding, backend = "aot_eager")(
        torch.randint(0, 16, (2, 4))
    )
    assert hidden_states.requires_grad, "compiled frozen embedding lost requires_grad"

    checkpoint(layer, hidden_states, use_reentrant = True).float().sum().backward()
    assert layer.adapter.weight.grad is not None, "no gradient reached the adapter"


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


def test_pre_hook_is_guarded():
    guarded = _guarded_functions(_ZOO / "peft_utils.py", ("requires_grad_pre_hook",))
    assert guarded == {"requires_grad_pre_hook" : True}, guarded


def test_post_hook_guard_is_not_unconditional():
    """A bare `if is_compiling(): return` would drop a frozen embedding's flag, so only
    the no-op case (output already requires grad) may be skipped."""
    src = (_ZOO / "peft_utils.py").read_text(encoding = "utf-8")
    fn = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "requires_grad_post_hook"
    )
    tests = [
        ast.dump(n.test) for n in ast.walk(fn)
        if isinstance(n, ast.If) and "is_compiling" in ast.dump(n.test)
    ]
    assert tests, "the post hook no longer checks is_compiling()"
    assert any("requires_grad" in t for t in tests), tests
    # and it must not bail out of the whole hook before reaching that check
    guarded = _guarded_functions(_ZOO / "peft_utils.py", ("requires_grad_post_hook",))
    assert guarded == {"requires_grad_post_hook" : False}, guarded


def test_make_inputs_require_grad_is_NOT_guarded():
    """Unlike the peft_utils hooks, this is the only thing making a FROZEN embedding's
    output require grad, so guarding it would silently starve the adapters."""
    guarded = _guarded_functions(
        _ZOO / "training_utils.py", ("make_inputs_require_grad",),
    )
    assert guarded == {"make_inputs_require_grad" : False}, guarded


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
