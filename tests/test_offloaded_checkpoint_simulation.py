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
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Behaviour simulation for the offloaded gradient checkpoint shim.

``unsloth_offloaded_gradient_checkpoint`` is public API (re-exported through
``unsloth.models._utils``) that no Unsloth setting installs - the only way in is
to call ``patch_unsloth_gradient_checkpointing()`` yourself. That is why an
argument-order defect survived in it for so long, and why fixing it needs
coverage that does not depend on an accelerator being present.

Everything here is CPU-only and accelerator-free on purpose, so the same
assertions run on Linux, macOS and Windows runners. The GPU-only counterpart
(real pinned-buffer offload) lives in ``test_offloaded_checkpoint_rng.py``.
"""
import functools
import inspect

import pytest
import torch


@pytest.fixture
def gc_module(monkeypatch):
    """Offload globals seeded for CPU, with the offload branch itself disabled.

    ``MINIMUM_SIZE`` above every tensor built here keeps the pinned-buffer
    branch off, so what is under test is the wrapper's argument handling and the
    plain recompute path.
    """
    from unsloth_zoo import gradient_checkpointing as module
    for name, value in (
        ("FIRST_PASS", False), ("LAST_GC_INDEX", 0), ("CURRENT_GC_INDEX", 0),
        ("MINIMUM_SIZE", 1 << 40), ("CPU_INDEX", 0),
        ("CPU_BUFFERS", [torch.empty(0)]), ("BACKWARD_PASS", False),
        ("USE_DOUBLE_BUFFER", False), ("GPU_BUFFERS", []), ("GPU_BUFFERS_B", None),
    ):
        monkeypatch.setattr(module, name, value, raising = False)
    return module


def pristine():
    return getattr(
        torch.utils.checkpoint, "_unsloth_pristine_checkpoint",
        torch.utils.checkpoint.checkpoint,
    )


# ---------------------------------------------------------------------------
# 1. The calling convention itself
# ---------------------------------------------------------------------------

def test_apply_receives_preserve_in_slot_two_and_untouched_args(gc_module, monkeypatch):
    """The regression, stated directly as a contract on ``.apply``.

    Pre-fix the call was ``apply(function, *args)``, so slot 2 held the first
    activation and the activation list was one short. Asserting on the recorded
    call is what makes the failure legible rather than just "numbers differ".
    """
    recorded = {}
    real_apply = gc_module.UnslothCheckpointFunction.apply

    def spy(function, *rest):
        recorded["preserve"] = rest[0]
        recorded["args"]     = rest[1:]
        return real_apply(function, *rest)
    monkeypatch.setattr(gc_module.UnslothCheckpointFunction, "apply", staticmethod(spy))

    a = torch.randn(2, 4, requires_grad = True)
    b = torch.randn(2, 4, requires_grad = True)
    c = torch.randn(2, 4)                       # frozen, still must arrive
    gc_module.unsloth_offloaded_gradient_checkpoint(
        lambda x, y, z: x + y + z, a, b, c, use_reentrant = True,
    )

    assert recorded["preserve"] is True, "default preserve_rng_state must be True"
    assert len(recorded["args"]) == 3
    for got, want in zip(recorded["args"], (a, b, c)):
        assert got is want, "activations must arrive unchanged and in order"


@pytest.mark.parametrize("flag", [True, False])
def test_explicit_preserve_flag_is_forwarded_verbatim(gc_module, monkeypatch, flag):
    seen = {}
    real_apply = gc_module.UnslothCheckpointFunction.apply

    def spy(function, preserve, *args):
        seen["preserve"] = preserve
        return real_apply(function, preserve, *args)
    monkeypatch.setattr(gc_module.UnslothCheckpointFunction, "apply", staticmethod(spy))

    x = torch.randn(2, 4, requires_grad = True)
    gc_module.unsloth_offloaded_gradient_checkpoint(
        lambda t: t * 2, x, use_reentrant = True, preserve_rng_state = flag,
    )
    assert seen["preserve"] is flag


def test_backward_returns_one_gradient_per_input(gc_module):
    """Arity is what an autograd.Function gets wrong when a slot is inserted.

    ``backward`` returns ``(None, None) + grads``; if the wrapper and the
    Function disagreed about the leading slots, autograd would raise
    "returned an incorrect number of gradients".
    """
    tensors = [torch.randn(3, 5, requires_grad = True) for _ in range(4)]
    out = gc_module.unsloth_offloaded_gradient_checkpoint(
        lambda *t: sum(t), *tensors, use_reentrant = True,
    )
    out.sum().backward()
    for i, t in enumerate(tensors):
        assert t.grad is not None, f"input {i} got no gradient"
        torch.testing.assert_close(t.grad, torch.ones_like(t))


def test_frozen_first_input_next_to_trainable_later_one(gc_module):
    """An untrained embedding feeding a trainable tensor is ordinary, and it is
    exactly the shape that made an earlier `requires_grad`-of-input-zero check
    bail out of backward early."""
    frozen    = torch.randn(3, 5)
    trainable = torch.randn(3, 5, requires_grad = True)
    out = gc_module.unsloth_offloaded_gradient_checkpoint(
        lambda a, b: a * b, frozen, trainable, use_reentrant = True,
    )
    out.sum().backward()
    torch.testing.assert_close(trainable.grad, frozen)


# ---------------------------------------------------------------------------
# 2. RNG semantics against the pristine torch reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preserve", [None, True, False])
def test_matches_pristine_torch_checkpoint(gc_module, preserve):
    kwargs = {} if preserve is None else {"preserve_rng_state" : preserve}

    def run(fn):
        torch.manual_seed(4242)
        h = torch.randn(4, 32, requires_grad = True)
        s = torch.randn(4, 32, requires_grad = True)
        block = lambda a, b: torch.nn.functional.dropout(a + b, p = 0.5, training = True)
        out = fn(block, h, s, use_reentrant = True, **kwargs)
        out.sum().backward()
        return out, h.grad, s.grad, torch.get_rng_state()

    for got, want in zip(run(gc_module.unsloth_offloaded_gradient_checkpoint), run(pristine())):
        torch.testing.assert_close(got, want)


def test_preserve_true_reproduces_the_forward_dropout_mask(gc_module):
    """The point of preserving RNG: recompute must see the forward's mask, so a
    checkpointed step equals an uncheckpointed one."""
    def run(fn):
        torch.manual_seed(11)
        h = torch.randn(4, 32, requires_grad = True)
        block = lambda a: torch.nn.functional.dropout(a * 3.0, p = 0.5, training = True)
        out = block(h) if fn is None else fn(block, h, use_reentrant = True, preserve_rng_state = True)
        out.sum().backward()
        return out, h.grad

    for got, want in zip(run(gc_module.unsloth_offloaded_gradient_checkpoint), run(None)):
        torch.testing.assert_close(got, want)


def test_preserve_false_leaves_rng_advanced(gc_module):
    """With the flag off, the recompute consumes RNG instead of forking it.

    This is the observable that proves the value reached ``ctx``; comparing
    gradients cannot distinguish it because the reference moves with it.
    """
    def final_state(preserve):
        torch.manual_seed(5)
        h = torch.randn(4, 32, requires_grad = True)
        block = lambda a: torch.nn.functional.dropout(a, p = 0.5, training = True)
        out = gc_module.unsloth_offloaded_gradient_checkpoint(
            block, h, use_reentrant = True, preserve_rng_state = preserve,
        )
        out.sum().backward()
        return torch.get_rng_state()

    assert not torch.equal(final_state(True), final_state(False))


# ---------------------------------------------------------------------------
# 3. Keyword handling - what must be dropped, bound, or passed through
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("keyword", ["context_fn", "determinism_check", "debug", "early_stop"])
def test_checkpoint_machinery_keywords_are_dropped_not_bound(gc_module, keyword):
    """Binding one of these onto the block would turn a call that works against
    unpatched torch into a TypeError the moment Unsloth patches checkpointing."""
    def block(x):                      # strict signature: one parameter only
        return x * 2
    value = (lambda: None) if keyword == "context_fn" else "default"
    x = torch.randn(2, 4, requires_grad = True)
    out = gc_module.unsloth_offloaded_gradient_checkpoint(
        block, x, use_reentrant = True, **{keyword : value},
    )
    out.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 2.0))


def test_tensor_keyword_requiring_grad_gets_a_gradient(gc_module):
    """`_bind_checkpoint_kwargs` routes such tensors through the positional list
    so autograd can see them; the RNG flag must not disturb that tail."""
    def block(hidden, *, side):
        return hidden + side
    hidden = torch.randn(3, 6, requires_grad = True)
    side   = torch.randn(3, 6, requires_grad = True)
    out = gc_module.unsloth_offloaded_gradient_checkpoint(
        block, hidden, side = side, use_reentrant = True, preserve_rng_state = True,
    )
    out.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))
    torch.testing.assert_close(side.grad,   torch.ones_like(side))


def test_non_tensor_keywords_reach_the_block(gc_module):
    seen = {}
    def block(x, *, scale, label):
        seen["scale"], seen["label"] = scale, label
        return x * scale
    x = torch.randn(2, 4, requires_grad = True)
    out = gc_module.unsloth_offloaded_gradient_checkpoint(
        block, x, scale = 3.0, label = "abc", use_reentrant = True,
    )
    out.sum().backward()
    assert seen == {"scale" : 3.0, "label" : "abc"}
    torch.testing.assert_close(x.grad, torch.full_like(x, 3.0))


def test_mixed_tensor_and_constant_keywords(gc_module):
    def block(x, *, weight, scale):
        return x * scale + weight
    x      = torch.randn(3, 6, requires_grad = True)
    weight = torch.randn(3, 6, requires_grad = True)
    out = gc_module.unsloth_offloaded_gradient_checkpoint(
        block, x, weight = weight, scale = 2.0,
        use_reentrant = True, preserve_rng_state = False,
    )
    out.sum().backward()
    torch.testing.assert_close(x.grad,      torch.full_like(x, 2.0))
    torch.testing.assert_close(weight.grad, torch.ones_like(weight))


def test_caller_keyword_dict_is_not_mutated(gc_module):
    """`kwargs.pop` acts on the fresh `**kwargs` dict, never the caller's.

    A caller reusing one options dict across layers - which is exactly what
    `gradient_checkpointing_kwargs` is - would otherwise silently lose the flag
    after the first layer.
    """
    options = {"preserve_rng_state" : False}
    snapshot = dict(options)
    x = torch.randn(2, 4, requires_grad = True)
    gc_module.unsloth_offloaded_gradient_checkpoint(
        lambda t: t * 2, x, use_reentrant = True, **options,
    )
    assert options == snapshot


def test_functools_partial_bound_flag_still_works(gc_module):
    """How transformers actually delivers it: `gradient_checkpointing_enable`
    bakes `gradient_checkpointing_kwargs` in with `functools.partial`."""
    bound = functools.partial(
        gc_module.unsloth_offloaded_gradient_checkpoint,
        use_reentrant = True, preserve_rng_state = True,
    )
    x = torch.randn(2, 4, requires_grad = True)
    out = bound(lambda t: t * 4, x)
    out.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 4.0))


def test_use_reentrant_false_is_accepted_and_ignored(gc_module):
    """transformers 5.x defaults `use_reentrant=False`; the shim forces the
    reentrant path regardless, and must not choke on the keyword."""
    x = torch.randn(2, 4, requires_grad = True)
    out = gc_module.unsloth_offloaded_gradient_checkpoint(
        lambda t: t * 5, x, use_reentrant = False,
    )
    out.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 5.0))


# ---------------------------------------------------------------------------
# 4. Buffer lifecycle - the state the wrapper finds when it is installed late
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("initial", [None, []], ids = ["unpatched-to-None", "empty-list"])
def test_buffers_are_reinitialised_when_absent(gc_module, monkeypatch, initial):
    """`unpatch_unsloth_smart_gradient_checkpointing` nulls the buffers, and
    `prepare_model_for_training` calls it for every non-"unsloth" setting - so
    None, not [], is the state this shim normally starts from."""
    monkeypatch.setattr(gc_module, "CPU_BUFFERS", initial, raising = False)
    called = []
    def fake_init(dtype = None):
        called.append(dtype)
        gc_module.CPU_BUFFERS = [torch.empty(0)]
    monkeypatch.setattr(gc_module, "initialize_unsloth_gradient_checkpointing", fake_init)

    x = torch.randn(2, 4, requires_grad = True)
    out = gc_module.unsloth_offloaded_gradient_checkpoint(lambda t: t * 2, x, use_reentrant = True)
    out.sum().backward()
    assert called == [x.dtype]


def test_existing_buffers_are_not_reinitialised(gc_module, monkeypatch):
    def boom(dtype = None):
        raise AssertionError("must not re-initialise populated buffers")
    monkeypatch.setattr(gc_module, "initialize_unsloth_gradient_checkpointing", boom)
    x = torch.randn(2, 4, requires_grad = True)
    gc_module.unsloth_offloaded_gradient_checkpoint(lambda t: t * 2, x, use_reentrant = True)


# ---------------------------------------------------------------------------
# 5. Agreement with the sibling shims
# ---------------------------------------------------------------------------

def _rng_run(fn):
    torch.manual_seed(808)
    h = torch.randn(4, 16, requires_grad = True)
    block = lambda a: torch.nn.functional.dropout(a * 1.5, p = 0.3, training = True)
    out = fn(block, h, use_reentrant = True, preserve_rng_state = True)
    out.sum().backward()
    return out, h.grad


def test_offloaded_shim_now_agrees_with_unsloth_checkpoint(gc_module):
    """`unsloth_checkpoint` always honoured the flag; the offloaded wrapper did
    not. Bringing the two into agreement is what this fix is."""
    reference = _rng_run(pristine())
    for shim in (
        gc_module.unsloth_offloaded_gradient_checkpoint,
        gc_module.unsloth_checkpoint,
    ):
        for got, want in zip(_rng_run(shim), reference):
            torch.testing.assert_close(got, want)


def test_plain_shim_still_ignores_the_flag_as_documented(gc_module):
    """Pins the remaining inconsistency so it is visible rather than surprising.

    `unsloth_gradient_checkpoint` routes through `Unsloth_Gradient_Checkpointer`,
    which has no `preserve_rng_state` slot at all, so the recompute draws a fresh
    dropout mask and the input gradient does not match torch. The comment above
    `_TORCH_CHECKPOINT_KEYWORDS_LITERAL` calls that deliberate, and it is out of
    scope here - but it is now the *only* shim that behaves this way, so this
    test fails loudly if someone fixes it and forgets to update the comment.
    """
    reference = _rng_run(pristine())
    actual    = _rng_run(gc_module.unsloth_gradient_checkpoint)
    assert not torch.allclose(actual[1], reference[1]), \
        "unsloth_gradient_checkpoint now preserves RNG - update the deliberate-drop comment"


# ---------------------------------------------------------------------------
# 6. Output and nesting shapes
# ---------------------------------------------------------------------------

def test_tuple_output_with_a_non_tensor_member(gc_module):
    """Decoder blocks return tuples, sometimes with None or metadata in them."""
    def block(x):
        return (x * 2, None, "meta")
    x = torch.randn(2, 4, requires_grad = True)
    out = gc_module.unsloth_offloaded_gradient_checkpoint(block, x, use_reentrant = True)
    assert out[1] is None and out[2] == "meta"
    out[0].sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 2.0))


def test_nested_checkpointing(gc_module):
    inner = lambda t: t * 2
    def outer(t):
        return gc_module.unsloth_offloaded_gradient_checkpoint(inner, t, use_reentrant = True)
    x = torch.randn(2, 4, requires_grad = True)
    out = gc_module.unsloth_offloaded_gradient_checkpoint(outer, x, use_reentrant = True)
    out.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 2.0))


def test_all_inputs_frozen_produces_no_graph(gc_module):
    x = torch.randn(2, 4)
    out = gc_module.unsloth_offloaded_gradient_checkpoint(lambda t: t * 2, x, use_reentrant = True)
    assert not out.requires_grad


def test_repeated_calls_are_stable(gc_module):
    """Module-level FIRST_PASS / CURRENT_GC_INDEX state must not drift into a
    wrong answer over a training loop."""
    for _ in range(5):
        x = torch.randn(2, 4, requires_grad = True)
        out = gc_module.unsloth_offloaded_gradient_checkpoint(
            lambda t: t * 3, x, use_reentrant = True,
        )
        out.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 3.0))


# ---------------------------------------------------------------------------
# 7. Public API stability - the backwards-compatibility guard
# ---------------------------------------------------------------------------

def test_public_signature_is_unchanged(gc_module):
    """The shim is re-exported through `unsloth.models._utils`, so an older
    `unsloth` can call this newer `unsloth_zoo`. The fix is body-only; if the
    signature ever moves, that pairing breaks silently."""
    parameters = inspect.signature(gc_module.unsloth_offloaded_gradient_checkpoint).parameters
    assert list(parameters) == ["function", "args", "use_reentrant", "kwargs"]
    assert parameters["args"].kind  is inspect.Parameter.VAR_POSITIONAL
    assert parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    assert parameters["use_reentrant"].default is None


def test_patch_then_unpatch_restores_torch_exactly():
    """Installing and removing the shim must leave torch as it was, or every
    later consumer in the process inherits Unsloth's checkpointing."""
    from unsloth_zoo import gradient_checkpointing as module
    original = torch.utils.checkpoint.checkpoint
    try:
        module.patch_unsloth_gradient_checkpointing()
        assert torch.utils.checkpoint.checkpoint.__name__ == "unsloth_offloaded_gradient_checkpoint"
        module.unpatch_unsloth_gradient_checkpointing()
        assert torch.utils.checkpoint.checkpoint is original
    finally:
        torch.utils.checkpoint.checkpoint = original


def test_pristine_checkpoint_is_still_recoverable_after_patching():
    """`_capture_pristine_checkpoint_once` is how consumers that must force
    `use_reentrant=False` (the Gemma-4 KV-sharing fix) escape the shim."""
    from unsloth_zoo import gradient_checkpointing as module
    original = torch.utils.checkpoint.checkpoint
    try:
        module.patch_unsloth_gradient_checkpointing()
        recovered = getattr(torch.utils.checkpoint, "_unsloth_pristine_checkpoint", None)
        assert recovered is not None
        assert recovered.__name__ not in module._UNSLOTH_CKPT_SHIM_NAMES
    finally:
        torch.utils.checkpoint.checkpoint = original
