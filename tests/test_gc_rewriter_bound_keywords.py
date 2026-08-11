# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Keywords bound into the checkpointed callable by
``unsloth_zoo.compiler.patch_gradient_checkpointing``.

``patch_gradient_checkpointing`` rewrites ``hidden_states = blk(ARGS)`` into a
``self._gradient_checkpointing_func(blk.__call__, ARGS)`` call. Everything in
``ARGS`` goes to the *checkpoint function*, not to the layer, so a keyword the
layer only accepts through ``**kwargs`` (transformers 5.x passes
``max_seqlen=max_seqlen`` to the Qwen2-VL vision block that way) cannot travel
there: ``self._gradient_checkpointing_func`` is frequently plain
``torch.utils.checkpoint.checkpoint``, which raises ``ValueError: Unexpected
keyword arguments`` under ``use_reentrant = True``.

The optional third element of a ``custom_gradient_checkpointing_replacements``
entry names such keywords, and the rewriter binds them into the callable with
``functools.partial`` - the same thing
``transformers.modeling_layers.GradientCheckpointingLayer.__call__`` does.

These tests pin that contract without needing transformers, a GPU, or a model
download.
"""

import ast
import functools
import inspect
import os
import sys
import textwrap

import pytest

import torch

from unsloth_zoo import compiler
from unsloth_zoo.gradient_checkpointing import (
    Unsloth_Gradient_Checkpointer,
    _TORCH_CHECKPOINT_KEYWORDS,
    _bind_checkpoint_kwargs,
    unsloth_checkpoint,
    unsloth_gradient_checkpoint,
)


# ---------------------------------------------------------------------------
# A miniature model whose forward has the exact shape the rewriter looks for.
# ---------------------------------------------------------------------------

_TOY_SOURCE = '''
class _ToyBlocks(torch.nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(depth)]
        )

    def forward(self, hidden_states, extra, **kwargs):
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                extra=extra,
                sentinel=sentinel,
                **kwargs,
            )
        return hidden_states
'''


@functools.lru_cache(maxsize = None)
def _build_toy():
    # patch_gradient_checkpointing() reads the class through inspect.getsource(),
    # so the toy has to live in a real file on disk.
    import importlib.util
    import tempfile

    directory = tempfile.mkdtemp(prefix = "unsloth_gc_toy_")
    path = os.path.join(directory, "unsloth_gc_toy.py")
    with open(path, "w", encoding = "utf-8") as handle:
        handle.write("import torch\n" + _TOY_SOURCE)
    spec = importlib.util.spec_from_file_location("unsloth_gc_toy", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["unsloth_gc_toy"] = module
    spec.loader.exec_module(module)
    return module._ToyBlocks


_TOY_FIND = """hidden_states = layer(
                hidden_states,
                extra=extra,
                sentinel=sentinel,
                **kwargs,
            )"""
_TOY_REPLACE = """hidden_states = layer(
                hidden_states,
                extra=extra,
                **kwargs,
            )"""
# Same call site with no keyword expansion left at all.
_TOY_REPLACE_NO_KWARGS = """hidden_states = layer(
                hidden_states,
                extra=extra,
            )"""


def _rewrite_toy(monkeypatch, entry):
    """Run patch_gradient_checkpointing over the toy class with one entry."""
    monkeypatch.setattr(
        compiler, "custom_gradient_checkpointing_replacements", [entry],
    )
    cls = _build_toy()
    output = compiler.patch_gradient_checkpointing("_ToyBlocks", cls)
    assert output is not None, "rewriter bailed on the toy class"
    return output[1]


def _for_loop_body(forward_source):
    tree = ast.parse(textwrap.dedent(forward_source))
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            return node
    raise AssertionError("no for-loop in the rewritten forward")


def test_rewriter_two_tuple_entries_declare_no_bound_keyword(monkeypatch):
    """A 2-tuple entry declares no bound keywords, so no extra keyword appears.

    The layer's own ``**kwargs`` expansion still has to move into the callable:
    it is a keyword expansion like any other, and leaving it in the argument
    list hands the layer's runtime keywords to the checkpoint function."""
    forward = _rewrite_toy(monkeypatch, (_TOY_FIND, _TOY_REPLACE))
    assert "sentinel" not in forward
    branch = _for_loop_body(forward).body[0]
    checkpointed = branch.body[0].value
    assert [kw.arg for kw in checkpointed.keywords] == []
    callable_arg = checkpointed.args[0]
    assert ast.unparse(callable_arg.func) == "functools.partial"
    assert [
        (kw.arg, ast.unparse(kw.value)) for kw in callable_arg.keywords
    ] == [(None, "kwargs")]
    ast.parse(textwrap.dedent(forward))


def test_rewriter_leaves_a_kwargless_call_site_alone(monkeypatch):
    """No ``**kwargs`` at the call site and no declared keyword means no
    ``functools.partial`` at all - the pre-existing zero-overhead emission."""
    forward = _rewrite_toy(
        monkeypatch, (_TOY_FIND, _TOY_REPLACE_NO_KWARGS),
    )
    assert "functools.partial" not in forward
    branch = _for_loop_body(forward).body[0]
    checkpointed = branch.body[0].value
    assert ast.unparse(checkpointed.args[0]) == "layer.__call__"
    assert checkpointed.keywords == []
    ast.parse(textwrap.dedent(forward))


@pytest.mark.parametrize("declaration", [("sentinel",), {"sentinel": "sentinel"}])
def test_rewriter_binds_declared_keyword_into_the_callable(monkeypatch, declaration):
    """Sequence and mapping spellings both bind the keyword into the callable on
    the checkpointed branch and pass it directly on the else branch."""
    forward = _rewrite_toy(
        monkeypatch, (_TOY_FIND, _TOY_REPLACE, declaration),
    )
    loop = _for_loop_body(forward)
    branch = loop.body[0]
    assert isinstance(branch, ast.If), "expected the if/else gradient checkpointing branch"

    checkpointed = branch.body[0].value
    assert isinstance(checkpointed, ast.Call)
    assert ast.unparse(checkpointed.func) == "self._gradient_checkpointing_func"
    # NOTHING may be left as a keyword on the checkpoint function itself - not
    # the declared keyword and not the layer's own **kwargs expansion, which is
    # empty often enough to look harmless and then raises `Unexpected keyword
    # arguments` the first time a caller passes anything.
    assert [kw.arg for kw in checkpointed.keywords] == [], (
        "no keyword may reach _gradient_checkpointing_func; "
        f"got {[kw.arg for kw in checkpointed.keywords]}"
    )
    callable_arg = checkpointed.args[0]
    assert isinstance(callable_arg, ast.Call)
    assert ast.unparse(callable_arg.func) == "functools.partial"
    assert ast.unparse(callable_arg.args[0]) == "layer.__call__"
    assert [
        (kw.arg, ast.unparse(kw.value)) for kw in callable_arg.keywords
    ] == [("sentinel", "sentinel"), (None, "kwargs")]

    direct = branch.orelse[0].value
    assert isinstance(direct, ast.Call)
    assert ast.unparse(direct.func) == "layer"
    assert {kw.arg: ast.unparse(kw.value) for kw in direct.keywords if kw.arg} == {
        "sentinel": "sentinel",
    }


def test_rewriter_bound_keyword_supports_a_custom_expression(monkeypatch):
    """A mapping may point the keyword at a different expression."""
    forward = _rewrite_toy(
        monkeypatch, (_TOY_FIND, _TOY_REPLACE, {"sentinel": "extra"}),
    )
    assert "functools.partial(layer.__call__, sentinel = extra, **kwargs)" in forward
    ast.parse(textwrap.dedent(forward))


def test_rewriter_both_branches_call_the_layer_identically(monkeypatch):
    """The checkpointed and the direct branch must hand the layer the same
    arguments - the only difference may be the checkpoint indirection."""
    forward = _rewrite_toy(monkeypatch, (_TOY_FIND, _TOY_REPLACE, ("sentinel",)))
    branch = _for_loop_body(forward).body[0]
    checkpointed = branch.body[0].value
    direct = branch.orelse[0].value

    positional = [ast.unparse(a) for a in checkpointed.args[1:]]
    assert positional == [ast.unparse(a) for a in direct.args]
    # Every keyword the direct branch passes - named or expanded - has to be on
    # the partial instead, and none of them on the checkpoint function.
    assert sorted(
        (str(kw.arg), ast.unparse(kw.value)) for kw in checkpointed.args[0].keywords
    ) == sorted(
        (str(kw.arg), ast.unparse(kw.value)) for kw in direct.keywords
    )
    assert checkpointed.keywords == []


def test_rewritten_forward_survives_non_empty_runtime_kwargs(monkeypatch):
    """Executed proof, not just emitted text.

    With an EMPTY ``**kwargs`` the old emission worked by accident, which is how
    the leak survived review. Give the forward one real runtime keyword and the
    reentrant ``torch.utils.checkpoint.checkpoint`` raises ``ValueError:
    Unexpected keyword arguments`` instead of ever reaching the layer."""
    forward = _rewrite_toy(monkeypatch, (_TOY_FIND, _TOY_REPLACE, ("sentinel",)))

    seen = []

    class _Layer(torch.nn.Module):
        def forward(self, hidden_states, extra, **kwargs):
            seen.append(dict(kwargs))
            return hidden_states * extra

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([_Layer(), _Layer()])
            self.gradient_checkpointing = True
            self._gradient_checkpointing_func = functools.partial(
                torch.utils.checkpoint.checkpoint, use_reentrant = True,
            )

    namespace = {"torch": torch, "functools": functools, "sentinel": 7}
    exec(compile(textwrap.dedent(forward), "<rewritten>", "exec"), namespace)
    _Model.forward = namespace["forward"]

    model = _Model()
    model.train()
    hidden_states = torch.randn(4, requires_grad = True)
    out = model(hidden_states, torch.tensor(2.0), probe = "delivered")
    out.sum().backward()

    assert seen, "the layer was never called"
    for kwargs in seen:
        assert kwargs["sentinel"] == 7, "declared keyword lost"
        assert kwargs["probe"] == "delivered", "runtime keyword never reached the layer"
    assert hidden_states.grad is not None


def test_qwen2_vl_5x_entry_declares_max_seqlen():
    """The shipped transformers 5.x Qwen2-VL entry must keep max_seqlen rather
    than drop it, and must remove it from the positional list (otherwise the
    ``arg=arg`` demotion would push it into a positional slot the block does not
    have)."""
    entries = compiler.custom_gradient_checkpointing_replacements
    matching = [e for e in entries if "max_seqlen=max_seqlen" in e[0]]
    assert len(matching) == 1, "expected exactly one 5.x max_seqlen entry"
    entry = matching[0]
    assert len(entry) == 3, "the 5.x entry must declare its bound keywords"
    assert "max_seqlen" in entry[2]
    assert "max_seqlen" not in entry[1], (
        "max_seqlen must leave the argument list; it comes back bound into the "
        "callable"
    )
    # And the 5.x entry has to stay last, after both 4.x entries, because its
    # replacement text is literally the 4.x call site.
    assert entries.index(entry) == len(entries) - 1


def test_generated_module_imports_functools():
    """``functools`` is not in the generated module's license header nor
    guaranteed to be re-exported by the model module, so create_new_function()
    must import it whenever the emitted source uses it."""
    source = inspect.getsource(compiler.create_new_function)
    assert '"functools." in new_source' in source
    assert '"import functools\\n"' in source


# ---------------------------------------------------------------------------
# Checkpoint shims
# ---------------------------------------------------------------------------

def test_bind_checkpoint_kwargs_is_identity_without_kwargs():
    def fn(x):
        return x
    assert _bind_checkpoint_kwargs(fn, {}) is fn
    # torch's own checkpoint keywords are consumed by the checkpoint machinery
    # and must never be bound into the wrapped function.
    only_torch = {k: object() for k in _TORCH_CHECKPOINT_KEYWORDS}
    assert _bind_checkpoint_kwargs(fn, only_torch) is fn


def test_bind_checkpoint_kwargs_binds_the_rest():
    def fn(x, flag = None):
        return (x, flag)
    bound = _bind_checkpoint_kwargs(fn, {"flag": 7, "preserve_rng_state": False})
    assert isinstance(bound, functools.partial)
    assert bound(1) == (1, 7)


@pytest.mark.parametrize(
    "checkpoint_fn", [unsloth_gradient_checkpoint, unsloth_checkpoint],
)
def test_checkpoint_shims_deliver_extra_kwargs_to_the_function(checkpoint_fn):
    """``unsloth_gradient_checkpoint`` used to drop them silently and
    ``unsloth_checkpoint`` used to raise. Both now behave like torch's
    non-reentrant checkpoint: the keyword reaches the wrapped function."""
    if checkpoint_fn is unsloth_checkpoint:
        pytest.importorskip("torch")
        pytest.skip(
            "unsloth_checkpoint routes through UnslothCheckpointFunction, whose "
            "module globals are only initialised on a real accelerator"
        )
    seen = {}

    def fn(x, flag = None):
        seen["flag"] = flag
        return x * 2

    x = torch.randn(4, requires_grad = True)
    out = checkpoint_fn(fn, x, flag = "delivered")
    out.sum().backward()
    assert seen["flag"] == "delivered"
    assert x.grad is not None


def test_unsloth_checkpoint_binds_instead_of_raising():
    """Source-level guard for the shim we cannot execute on CPU: the leftover
    keyword branch must bind, not raise."""
    source = inspect.getsource(unsloth_checkpoint)
    assert "_bind_checkpoint_kwargs" in source
    assert "raise ValueError" not in source.split("if kwargs and use_reentrant")[1][:400]


def test_unsloth_gradient_checkpointer_accepts_a_bare_tensor_return():
    """Every modern transformers block returns the tensor itself rather than a
    1-tuple. The recompute in backward used to insist on a 1-tuple and raised
    ``too many values to unpack``."""
    def block(x):
        return x * 3

    x = torch.randn(5, requires_grad = True)
    out = Unsloth_Gradient_Checkpointer.apply(block, x)
    assert torch.is_tensor(out)
    out.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, 3.0))


def test_unsloth_gradient_checkpointer_still_accepts_a_one_tuple_return():
    def block(x):
        return (x * 3,)

    x = torch.randn(5, requires_grad = True)
    out = Unsloth_Gradient_Checkpointer.apply(block, x)
    if isinstance(out, tuple):
        (out,) = out
    out.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, 3.0))
