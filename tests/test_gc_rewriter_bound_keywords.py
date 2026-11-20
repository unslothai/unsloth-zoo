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


# A miniature model whose forward has the exact shape the rewriter looks for.

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


# Same shape, but with a named layer class the rewriter can resolve.
_TOY_NAMED_SOURCE = '''
class _ToyLayer(torch.nn.Module):
    def forward(self, hidden_states, extra, **kwargs):
        return hidden_states


class _ToyNamedBlocks(torch.nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.layers = torch.nn.ModuleList([_ToyLayer() for _ in range(depth)])

    def forward(self, hidden_states, extra, **kwargs):
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                extra=extra,
                **kwargs,
            )
        return hidden_states
'''

_TOY_NAMED_FIND = """hidden_states = layer(
                hidden_states,
                extra=extra,
                **kwargs,
            )"""
_TOY_NAMED_REPLACE = """hidden_states = layer(
                hidden_states,
                extra=extra,
                injected=injected,
                **kwargs,
            )"""


@functools.lru_cache(maxsize = None)
def _build_toy():
    # patch_gradient_checkpointing() reads the class through inspect.getsource().
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


@functools.lru_cache(maxsize = None)
def _build_named_toy():
    import importlib.util
    import tempfile

    directory = tempfile.mkdtemp(prefix = "unsloth_gc_named_toy_")
    path = os.path.join(directory, "unsloth_gc_named_toy.py")
    with open(path, "w", encoding = "utf-8") as handle:
        handle.write("import torch\n" + _TOY_NAMED_SOURCE)
    spec = importlib.util.spec_from_file_location("unsloth_gc_named_toy", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["unsloth_gc_named_toy"] = module
    spec.loader.exec_module(module)
    return module._ToyNamedBlocks


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
    """A 2-tuple entry declares no bound keywords, but the layer's own **kwargs expansion must still move into the
    callable, or its runtime keywords reach the checkpoint function."""
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
    """Without **kwargs at the call site or a declared keyword, no functools.partial is emitted, keeping the pre-existing zero-overhead emission."""
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
    """Sequence and mapping spellings both bind the keyword into the callable on the checkpointed branch and pass it directly on the else branch."""
    forward = _rewrite_toy(
        monkeypatch, (_TOY_FIND, _TOY_REPLACE, declaration),
    )
    loop = _for_loop_body(forward)
    branch = loop.body[0]
    assert isinstance(branch, ast.If), "expected the if/else gradient checkpointing branch"

    checkpointed = branch.body[0].value
    assert isinstance(checkpointed, ast.Call)
    assert ast.unparse(checkpointed.func) == "self._gradient_checkpointing_func"
    # No keyword may reach the checkpoint function, not even the layer's own **kwargs, or the first caller passing one gets Unexpected keyword arguments.
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
    """The checkpointed and direct branches must hand the layer identical arguments; the only difference may be the checkpoint indirection."""
    forward = _rewrite_toy(monkeypatch, (_TOY_FIND, _TOY_REPLACE, ("sentinel",)))
    branch = _for_loop_body(forward).body[0]
    checkpointed = branch.body[0].value
    direct = branch.orelse[0].value

    positional = [ast.unparse(a) for a in checkpointed.args[1:]]
    assert positional == [ast.unparse(a) for a in direct.args]
    assert sorted(
        (str(kw.arg), ast.unparse(kw.value)) for kw in checkpointed.args[0].keywords
    ) == sorted(
        (str(kw.arg), ast.unparse(kw.value)) for kw in direct.keywords
    )
    assert checkpointed.keywords == []


def test_rewritten_forward_survives_non_empty_runtime_kwargs(monkeypatch):
    """Executed, not just emitted: with an empty **kwargs the old emission only worked by accident; one runtime keyword
    makes reentrant torch.utils.checkpoint.checkpoint raise 'Unexpected keyword arguments'."""
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


def _rewrite_named_toy(monkeypatch, entry):
    monkeypatch.setattr(
        compiler, "custom_gradient_checkpointing_replacements", [entry],
    )
    output = compiler.patch_gradient_checkpointing(
        "_ToyNamedBlocks", _build_named_toy(),
    )
    assert output is not None, "rewriter bailed on the named toy class"
    return output[1]


def test_rewriter_resolves_the_modulelist_layer_class():
    cls = _build_named_toy()
    init = inspect.getsource(cls.__init__)
    layer_class = compiler._modulelist_layer_class(cls, init, "self.layers")
    assert layer_class is not None and layer_class.__name__ == "_ToyLayer"


def test_rewriter_skips_a_replacement_the_layer_cannot_take(monkeypatch):
    """A replacement may declare layer parameters it needs: Qwen2-VL call sites read the same in transformers 4.53.0-5.9
    and 5.10-5.14, but only the first takes rotary_pos_emb; injecting it on 5.10-5.14 passed one positional too many,
    so the entry is skipped when the parameter is absent."""
    forward = _rewrite_named_toy(
        monkeypatch,
        (_TOY_NAMED_FIND, _TOY_NAMED_REPLACE, None, ("injected",)),
    )
    assert "injected" not in forward, \
        "injected a keyword the layer's forward does not declare"
    ast.parse(textwrap.dedent(forward))


def test_rewriter_applies_a_replacement_the_layer_can_take(monkeypatch):
    """Control for the test above: the same entry fires when the declared parameter is present, so the guard cannot pass by never applying."""
    forward = _rewrite_named_toy(
        monkeypatch,
        (_TOY_NAMED_FIND, _TOY_NAMED_REPLACE, None, ("extra",)),
    )
    assert "injected" in forward
    ast.parse(textwrap.dedent(forward))


def test_rewriter_applies_a_guarded_replacement_when_the_layer_is_unresolvable(
        monkeypatch):
    """Unresolvable layer names must behave as before, never silently stop rewriting; the toy's ModuleList holds
    torch.nn.Identity, which is not a name in its module."""
    forward = _rewrite_toy(
        monkeypatch, (_TOY_FIND, _TOY_REPLACE, None, ("nothing_has_this",)),
    )
    assert "sentinel" not in forward, "the replacement did not fire"
    ast.parse(textwrap.dedent(forward))


def test_qwen2_vl_rotary_entries_declare_the_parameter_they_depend_on():
    """Shipped entries that re-inject rotary_pos_emb must be guarded on it, since transformers 5.10-5.14 spell the call site the same way but dropped the parameter."""
    guarded = [
        entry for entry in compiler.custom_gradient_checkpointing_replacements
        if "rotary_pos_emb=rotary_pos_emb" in entry[1]
    ]
    assert guarded, "no rotary_pos_emb re-injecting entry left"
    for entry in guarded:
        assert len(entry) == 4 and "rotary_pos_emb" in entry[3], (
            "an entry that injects rotary_pos_emb must declare it as required; "
            f"got {entry[2:]}"
        )


def test_qwen2_vl_5x_entry_declares_max_seqlen():
    """The shipped transformers 5.x Qwen2-VL entry must keep max_seqlen and remove it from the positional list, else the
    arg=arg demotion pushes it into a slot the block lacks."""
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
    # The 5.x entry must stay last: its replacement text is the 4.x call site.
    assert entries.index(entry) == len(entries) - 1


def test_generated_module_imports_functools():
    """functools is neither in the generated header nor guaranteed re-exported by the model module, so create_new_function() must import it when the source uses it."""
    source = inspect.getsource(compiler.create_new_function)
    assert '"functools." in new_source' in source
    assert '"import functools\\n"' in source


# Checkpoint shims

def test_bind_checkpoint_kwargs_is_identity_without_kwargs():
    def fn(x):
        return x
    assert _bind_checkpoint_kwargs(fn, {}) == (fn, ())
    # torch's own checkpoint keywords must never be bound into the function.
    only_torch = {k: object() for k in _TORCH_CHECKPOINT_KEYWORDS}
    assert _bind_checkpoint_kwargs(fn, only_torch) == (fn, ())


def test_torch_checkpoint_keywords_covers_every_keyword_only_parameter():
    """torch checkpoint's keyword-only params belong to the checkpoint, not the block: early_stop is ignored under
    use_reentrant=True (forced by the Unsloth shims), so binding it onto the block raised TypeError."""
    signature = inspect.signature(torch.utils.checkpoint.checkpoint)
    keyword_only = {
        name for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    } - {"use_reentrant"}
    missing = keyword_only - _TORCH_CHECKPOINT_KEYWORDS
    assert not missing, f"checkpoint keywords that would be bound to the block: {missing}"


def test_bind_checkpoint_kwargs_drops_early_stop(monkeypatch):
    """Executed: a block without an early_stop parameter must survive a caller passing torch's documented early_stop keyword."""
    if "early_stop" not in inspect.signature(torch.utils.checkpoint.checkpoint).parameters:
        pytest.skip("this torch has no per-call early_stop")

    def block(x):
        return x * 2

    assert _bind_checkpoint_kwargs(block, {"early_stop": False}) == (block, ())
    x = torch.randn(4, requires_grad = True)
    out = unsloth_gradient_checkpoint(block, x, early_stop = False)
    out.sum().backward()
    assert x.grad is not None


def test_bind_checkpoint_kwargs_binds_the_rest():
    def fn(x, flag = None):
        return (x, flag)
    bound, extra = _bind_checkpoint_kwargs(fn, {"flag": 7, "preserve_rng_state": False})
    # Nothing differentiable here, so the cheap partial is kept.
    assert isinstance(bound, functools.partial)
    assert extra == ()
    assert bound(1) == (1, 7)


@pytest.mark.parametrize(
    "checkpoint_fn", [unsloth_gradient_checkpoint, unsloth_checkpoint],
)
def test_checkpoint_shims_deliver_extra_kwargs_to_the_function(checkpoint_fn):
    """Extra keyword arguments must reach the wrapped function in unsloth_gradient_checkpoint and unsloth_checkpoint, as in torch's non-reentrant checkpoint."""
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
    """Source-level guard for the shim, which cannot run on CPU: the leftover keyword branch must bind via _bind_checkpoint_kwargs, not raise ValueError."""
    source = inspect.getsource(unsloth_checkpoint)
    assert "_bind_checkpoint_kwargs" in source
    assert "raise ValueError" not in source.split("if kwargs and use_reentrant")[1][:400]


def test_unsloth_gradient_checkpointer_accepts_a_bare_tensor_return():
    """Modern transformers blocks return a bare tensor, not a 1-tuple; the backward recompute must accept it instead of raising 'too many values to unpack'."""
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


# Tensor keywords must stay autograd inputs: a non-leaf tensor closed over by functools.partial is not a Function input,
# so a shared one breaks on second backward.


@pytest.fixture
def cpu_offload_globals(monkeypatch):
    """Lets UnslothCheckpointFunction run on CPU: its globals are normally seeded by
    initialize_unsloth_gradient_checkpointing() (needs pinned memory, streams, device count); MINIMUM_SIZE keeps
    offload off."""
    from unsloth_zoo import gradient_checkpointing as gc_module
    for name, value in (
        ("FIRST_PASS", False), ("LAST_GC_INDEX", 0), ("CURRENT_GC_INDEX", 0),
        ("MINIMUM_SIZE", 1 << 40), ("CPU_INDEX", 0),
        ("CPU_BUFFERS", [torch.empty(0)]), ("BACKWARD_PASS", False),
        ("USE_DOUBLE_BUFFER", False), ("GPU_BUFFERS", []), ("GPU_BUFFERS_B", None),
    ):
        monkeypatch.setattr(gc_module, name, value, raising = False)
    return gc_module


@pytest.mark.parametrize("preserve_rng_state", [None, True, False])
def test_offloaded_wrapper_matches_torch_checkpoint(cpu_offload_globals, preserve_rng_state):
    gc_module = cpu_offload_globals
    checkpoint = getattr(
        torch.utils.checkpoint, "_unsloth_pristine_checkpoint", torch.utils.checkpoint.checkpoint,
    )
    kwargs = {} if preserve_rng_state is None else {"preserve_rng_state": preserve_rng_state}

    def run(checkpoint_fn):
        torch.manual_seed(123)
        hidden = torch.randn(2, 16, requires_grad = True)
        side = torch.randn(2, 16, requires_grad = True)

        def block(hidden, side):
            return torch.nn.functional.dropout(hidden + side, p = 0.5, training = True)

        output = checkpoint_fn(block, hidden, side, use_reentrant = True, **kwargs)
        output.sum().backward()
        return output, hidden.grad, side.grad, torch.get_rng_state()

    expected = run(checkpoint)
    actual = run(gc_module.unsloth_offloaded_gradient_checkpoint)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference)


def test_offloaded_wrapper_reinitialises_after_an_unpatch(cpu_offload_globals, monkeypatch):
    """Wrapper must re-initialise CPU_BUFFERS left None by unpatch_unsloth_smart_gradient_checkpointing
    (prepare_model_for_training calls it for non-'unsloth' modes); len(None) would be a TypeError."""
    gc_module = cpu_offload_globals
    monkeypatch.setattr(gc_module, "CPU_BUFFERS", None, raising = False)

    initialised = []
    def fake_initialize(dtype = None):
        initialised.append(dtype)
        gc_module.CPU_BUFFERS = [torch.empty(0)]
    monkeypatch.setattr(gc_module, "initialize_unsloth_gradient_checkpointing", fake_initialize)

    hidden = torch.randn(2, 16, requires_grad = True)
    output = gc_module.unsloth_offloaded_gradient_checkpoint(
        lambda x: x * 2.0, hidden, use_reentrant = True,
    )
    output.sum().backward()

    assert initialised == [hidden.dtype]
    torch.testing.assert_close(hidden.grad, torch.full_like(hidden, 2.0))


class _SideLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(4) * 0.5)

    def forward(self, hidden_states, side = None, flag = False):
        out = hidden_states @ self.weight
        if side is not None:
            out = out + side
        if flag:
            out = out * 1.0
        return out


def _run_shared_keyword(checkpoint_fn):
    """Two checkpointed layers fed the same non-leaf tensor keyword, which also feeds the loss; returns the gradients."""
    torch.manual_seed(0)
    layers = torch.nn.ModuleList([_SideLayer(), _SideLayer()])
    torch.manual_seed(1)
    x    = torch.randn(2, 4, requires_grad = True)
    leaf = torch.randn(2, 4, requires_grad = True)
    side = leaf * 2.0                     # non-leaf, requires grad, shared

    hidden_states = x
    for layer in layers:
        if checkpoint_fn is None:
            hidden_states = layer(hidden_states, side = side, flag = True)
        else:
            hidden_states = checkpoint_fn(layer, hidden_states, side = side, flag = True)
    (hidden_states.sum() + side.sum()).backward()
    return {
        "x": x.grad, "leaf": leaf.grad,
        "w0": layers[0].weight.grad, "w1": layers[1].weight.grad,
    }


@pytest.mark.parametrize(
    "checkpoint_fn", [unsloth_gradient_checkpoint, unsloth_checkpoint],
)
def test_shared_tensor_keyword_matches_the_uncheckpointed_reference(
    checkpoint_fn, cpu_offload_globals,
):
    reference = _run_shared_keyword(None)
    got = _run_shared_keyword(checkpoint_fn)
    for name, expected in reference.items():
        assert got[name] is not None, f"{name} received no gradient"
        assert torch.equal(got[name], expected), (
            f"{name}: {(got[name] - expected).abs().max().item()}"
        )


def test_bind_checkpoint_kwargs_routes_differentiable_tensors_positionally():
    """The differentiable tensor keyword becomes a positional input of the checkpoint Function and is put back as a keyword before the block is called."""
    def block(x, side = None, flag = None):
        return (x, side, flag)

    side = (torch.randn(3, requires_grad = True) * 2.0)
    bound, extra = _bind_checkpoint_kwargs(block, {"side": side, "flag": 7})
    assert not isinstance(bound, functools.partial)
    assert extra == (side,)
    x = torch.zeros(3)
    seen_x, seen_side, seen_flag = bound(x, *extra)
    assert seen_x is x and seen_side is side and seen_flag == 7


def test_non_differentiable_tensor_keyword_keeps_the_cheap_partial():
    """Only tensors that require grad need to be visible to autograd."""
    def block(x, mask = None):
        return x

    mask = torch.ones(3)
    bound, extra = _bind_checkpoint_kwargs(block, {"mask": mask})
    assert isinstance(bound, functools.partial)
    assert extra == ()


def test_reentrant_checkpointer_returns_gradients_for_tensor_args():
    """Positional tensor args are the same case: the recompute ran on the still-attached tensor and the Function returned None as their gradient."""
    def block(hidden_states, side):
        return hidden_states * 2.0 + side

    leaf = torch.randn(4, requires_grad = True)
    side = leaf * 3.0
    x = torch.randn(4, requires_grad = True)
    out = Unsloth_Gradient_Checkpointer.apply(block, x, side)
    (out.sum() + side.sum()).backward()
    assert torch.allclose(x.grad, torch.full_like(x, 2.0))
    # d(loss)/d(leaf) = 3 * (1 from the block + 1 from side.sum())
    assert torch.allclose(leaf.grad, torch.full_like(leaf, 6.0))


# UnslothCheckpointFunction decided it had a gradient to produce from positional input zero alone, so a frozen input
# zero hit the early return and gave one None where a gradient per input was needed; torch's reentrant
# CheckpointFunction saves every tensor input unconditionally.


def _frozen_first_input(checkpoint_fn, as_keyword):
    """activation 0 does not require grad; the second input does."""
    def block(hidden_states, side):
        return hidden_states * 2.0 + side * 3.0

    def block_kw(hidden_states, side = None):
        return block(hidden_states, side)

    torch.manual_seed(3)
    frozen = torch.randn(4)
    side   = torch.randn(4, requires_grad = True)
    if checkpoint_fn is None:
        out = block(frozen, side)
    elif as_keyword:
        out = checkpoint_fn(block_kw, frozen, side = side)
    else:
        out = checkpoint_fn(block, frozen, side)
    out.sum().backward()
    return side.grad


@pytest.mark.parametrize("as_keyword", [False, True])
@pytest.mark.parametrize(
    "checkpoint_fn", [unsloth_checkpoint, unsloth_gradient_checkpoint],
)
def test_frozen_first_activation_still_propagates(
    checkpoint_fn, as_keyword, cpu_offload_globals,
):
    reference = _frozen_first_input(None, as_keyword)
    got = _frozen_first_input(checkpoint_fn, as_keyword)
    assert got is not None, "the differentiable input received no gradient"
    assert torch.equal(got, reference)


def test_frozen_first_activation_matches_torchs_own_reentrant_checkpoint(
        cpu_offload_globals):
    """Pristine torch already behaves this way on the reentrant checkpoint path these shims force."""
    pristine = getattr(
        torch.utils.checkpoint, "_unsloth_pristine_checkpoint", None,
    ) or torch.utils.checkpoint.checkpoint

    def block(hidden_states, side):
        return hidden_states * 2.0 + side * 3.0

    torch.manual_seed(3)
    frozen = torch.randn(4)
    side   = torch.randn(4, requires_grad = True)
    pristine(block, frozen, side, use_reentrant = True).sum().backward()
    assert torch.equal(side.grad, _frozen_first_input(unsloth_checkpoint, False))
