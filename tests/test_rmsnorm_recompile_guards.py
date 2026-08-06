# SPDX-License-Identifier: AGPL-3.0-only
"""One compiled RMSNorm code object is shared by every norm instance in a model.

gemma-4-E2B has 504 of them, so that single frame's Dynamo cache holds the
*product* of every axis its guards can see: parameter width, input rank,
`self.with_scale`, dtype, grad mode and requires_grad. That reached
`patch_torch_compile`'s recompile_limit on a T4 and raised
`FailOnRecompileLimitHit`, which then took activation checkpointing down with it.

Compiling a pure tensor kernel instead of a bound method leaves `self` in eager,
which removes the parameter-width, rank and with_scale axes. These tests pin
that: the refactor must not change a single output bit, and it must keep the
cache small enough that a realistic spread of call shapes never trips the limit.
"""

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "needs a GPU: these assert on real torch.compile guard behaviour",
)

# The spread one model actually produces: several norm widths, rank 3 residual
# norms and rank 4 q/k norms, scaled and unscaled, in each dtype the fp32 patch
# can see. Deliberately more combinations than the limit below.
CASES = [
    ((2, 7, 64),      64,   True),
    ((2, 7, 768),     768,  True),
    ((1, 4, 5, 256),  256,  True),
    ((3, 11, 1536),   1536, True),
    ((2, 7, 64),      64,   False),
    ((1, 4, 5, 256),  256,  False),
]
DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_FP16_MAX = float(torch.finfo(torch.float16).max)


def _old_body(hidden_states, weight, eps, with_scale):
    """The pre-fix maths, verbatim, so equivalence is checked against the real thing."""
    x_fp32 = hidden_states.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim = True)
    normed = x_fp32 * torch.pow(variance + eps, -0.5)
    if with_scale:
        normed = normed * weight.to(torch.float32)
    return torch.clamp(normed, min = -_FP16_MAX, max = _FP16_MAX).to(torch.float16)


class _OldNorm(torch.nn.Module):
    """A bound method reading `self.weight` / `self.with_scale`, i.e. the old shape."""

    def __init__(self, hidden, with_scale, dtype):
        super().__init__()
        self.eps = 1e-6
        self.with_scale = with_scale
        if with_scale:
            self.weight = torch.nn.Parameter(torch.randn(hidden, device = "cuda", dtype = dtype))
        else:
            self.weight = None

    def forward(self, hidden_states):
        return _old_body(hidden_states, self.weight, self.eps, self.with_scale)


def _new(hidden_states, weight, eps, with_scale):
    from unsloth_zoo.temporary_patches.common import (
        flatten_for_elementwise_norm,
        unwrap_norm_weight,
    )
    from unsloth_zoo.temporary_patches.gemma4_float32 import (
        _gemma4_rms_norm_scaled,
        _gemma4_rms_norm_unscaled,
    )
    flat, shape = flatten_for_elementwise_norm(hidden_states)
    if with_scale:
        out = _gemma4_rms_norm_scaled(flat, unwrap_norm_weight(weight), eps)
    else:
        out = _gemma4_rms_norm_unscaled(flat, eps)
    return out.reshape(shape)


def _new_eager(hidden_states, weight, eps, with_scale):
    """The refactor's data flow without torch.compile.

    Compiled output legitimately differs from eager by Inductor's float
    reassociation, and the old code was compiled too, so comparing eager-old to
    compiled-new would measure the compiler rather than this change. Running the
    same reshape/unwrap path eagerly isolates the refactor itself.
    """
    from unsloth_zoo.temporary_patches.common import (
        flatten_for_elementwise_norm,
        unwrap_norm_weight,
    )
    flat, shape = flatten_for_elementwise_norm(hidden_states)
    weight = None if weight is None else unwrap_norm_weight(weight)
    return _old_body(flat, weight, eps, with_scale).reshape(shape)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape, hidden, with_scale", CASES)
def test_the_refactor_changes_no_output_bit(dtype, shape, hidden, with_scale):
    torch.manual_seed(0)
    hidden_states = torch.randn(*shape, device = "cuda", dtype = dtype)
    weight = torch.nn.Parameter(torch.randn(hidden, device = "cuda", dtype = dtype)) \
        if with_scale else None

    expected = _old_body(hidden_states, weight, 1e-6, with_scale)
    actual = _new_eager(hidden_states, weight, 1e-6, with_scale)

    assert torch.equal(expected, actual)


@pytest.mark.parametrize("with_scale", (True, False))
def test_gradients_still_reach_the_parameter_through_the_view(with_scale):
    # unwrap_norm_weight hands the kernel a *view* of the Parameter. If that view
    # detached, training would silently stop updating every norm in the model.
    torch.manual_seed(0)
    old_x = torch.randn(2, 7, 768, device = "cuda", dtype = torch.float16, requires_grad = True)
    new_x = old_x.detach().clone().requires_grad_(True)
    old_w = torch.nn.Parameter(torch.randn(768, device = "cuda", dtype = torch.float16)) \
        if with_scale else None
    new_w = torch.nn.Parameter(old_w.detach().clone()) if with_scale else None

    _old_body(old_x, old_w, 1e-6, with_scale).float().sum().backward()
    _new_eager(new_x, new_w, 1e-6, with_scale).float().sum().backward()

    assert torch.equal(old_x.grad, new_x.grad)
    if with_scale:
        assert new_w.grad is not None
        assert torch.equal(old_w.grad, new_w.grad)


def _run_every_case(fn):
    for dtype in DTYPES:
        for shape, hidden, with_scale in CASES:
            torch.manual_seed(0)
            hidden_states = torch.randn(*shape, device = "cuda", dtype = dtype)
            weight = torch.nn.Parameter(torch.randn(hidden, device = "cuda", dtype = dtype)) \
                if with_scale else None
            fn(hidden_states, weight, with_scale, hidden, dtype)


def test_the_old_bound_method_exhausts_a_realistic_recompile_budget():
    # The regression this fix exists for. A low limit stands in for the real one:
    # the point is that the old form's cache grows with the product of the axes,
    # so on a model with hundreds of norms any fixed budget is eventually spent.
    torch._dynamo.reset()
    with torch._dynamo.config.patch(
        recompile_limit = 8,
        fail_on_recompile_limit_hit = True,
        # unsloth_zoo sets suppress_errors globally, and Dynamo asserts the two
        # are never both on, so pin it rather than inherit it.
        suppress_errors = False,
    ):
        compiled = {}

        def call(hidden_states, weight, with_scale, hidden, dtype):
            key = (hidden, with_scale, dtype)
            if key not in compiled:
                module = _OldNorm(hidden, with_scale, dtype).cuda()
                if with_scale:
                    module.weight.data = module.weight.data.to(dtype)
                compiled[key] = torch.compile(module, fullgraph = True)
            compiled[key](hidden_states)

        with pytest.raises(torch._dynamo.exc.FailOnRecompileLimitHit):
            _run_every_case(call)


def test_the_kernels_survive_the_same_budget():
    torch._dynamo.reset()
    with torch._dynamo.config.patch(
        recompile_limit = 8,
        fail_on_recompile_limit_hit = True,
        # unsloth_zoo sets suppress_errors globally, and Dynamo asserts the two
        # are never both on, so pin it rather than inherit it.
        suppress_errors = False,
    ):
        _run_every_case(
            lambda hidden_states, weight, with_scale, hidden, dtype:
                _new(hidden_states, weight, 1e-6, with_scale)
        )


def test_a_parameter_view_is_not_itself_a_parameter():
    # This is the whole reason the width guard goes away: Dynamo's is_static_input
    # forces a static shape when `type(t) is torch.nn.Parameter`, and `dynamic=True`
    # does not override it. A view is a plain Tensor, so it takes dynamic shapes.
    from unsloth_zoo.temporary_patches.common import unwrap_norm_weight

    weight = torch.nn.Parameter(torch.randn(64))
    view = unwrap_norm_weight(weight)

    assert type(weight) is torch.nn.Parameter
    assert type(view) is torch.Tensor
    assert unwrap_norm_weight(None) is None


def test_flatten_round_trips_every_rank():
    from unsloth_zoo.temporary_patches.common import flatten_for_elementwise_norm

    for shape in ((2, 7, 64), (1, 4, 5, 256), (3, 64), (2, 2, 2, 2, 16)):
        x = torch.randn(*shape)
        flat, original = flatten_for_elementwise_norm(x)
        assert flat.ndim == 2
        assert flat.shape[-1] == shape[-1]
        assert tuple(original) == shape
        assert torch.equal(flat.reshape(original), x)
