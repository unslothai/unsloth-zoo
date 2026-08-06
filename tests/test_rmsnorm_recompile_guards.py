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
"""One compiled RMSNorm code object is shared by every norm instance in a model.

gemma-4-E2B has 504 of them, so that frame's Dynamo cache holds the *product* of every
axis its guards see (width, rank, `self.with_scale`, dtype, grad mode, requires_grad),
which hit the recompile_limit on a T4 and raised `FailOnRecompileLimitHit`, taking
activation checkpointing with it. Compiling a pure tensor kernel instead of a bound
method leaves `self` in eager, dropping the width, rank and with_scale axes. These
tests pin that: no output bit changes, and a realistic spread of shapes stays in budget.
"""

import pytest
import torch


# Dynamo renamed its cache-limit knobs and its exception in torch 2.7, and
# `config.patch` raises on a key this torch does not define, so select by what is
# there. Same approach as tests/test_recompile_limit_fallback.py.
_HAS_NEW_NAMES = hasattr(torch._dynamo.config, "recompile_limit")
_LIMIT_KEY = "recompile_limit" if _HAS_NEW_NAMES else "cache_size_limit"
_FAIL_KEY = ("fail_on_recompile_limit_hit" if _HAS_NEW_NAMES
             else "fail_on_cache_limit_hit")
_LIMIT_ERROR = next(
    (e for e in (getattr(torch._dynamo.exc, n, None)
                 for n in ("FailOnRecompileLimitHit", "RecompileLimitExceeded",
                           "CacheLimitExceeded"))
     if isinstance(e, type) and issubclass(e, BaseException)),
    None,
)


def _limit_patch(limit):
    """`config.patch` kwargs for a hard failure at `limit` recompiles."""
    return {
        _LIMIT_KEY: limit,
        _FAIL_KEY: True,
        # unsloth_zoo sets this globally, and Dynamo asserts it is never on together
        # with the fail-on-limit flag.
        "suppress_errors": False,
    }


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "needs a GPU: these assert on real torch.compile guard behaviour",
)

if _LIMIT_ERROR is None or not hasattr(torch._dynamo.config, _FAIL_KEY):
    pytestmark = [
        pytestmark,
        pytest.mark.skip(reason = "this torch exposes no hard recompile-limit failure"),
    ]

# The spread a real model produces: several widths, rank 3 residual and rank 4 q/k
# norms, scaled and unscaled, every dtype. More combinations than the limit below.
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
    """The pre-fix maths, verbatim."""
    x_fp32 = hidden_states.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim = True)
    normed = x_fp32 * torch.pow(variance + eps, -0.5)
    if with_scale:
        normed = normed * weight.to(torch.float32)
    return torch.clamp(normed, min = -_FP16_MAX, max = _FP16_MAX).to(torch.float16)


class _OldNorm(torch.nn.Module):
    """A bound method reading `self.weight` / `self.with_scale`, i.e. the old form."""

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

    Inductor reassociates floats, so comparing eager-old to compiled-new would
    measure the compiler, not this change.
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
    # A detaching view here would silently stop updating every norm in the model.
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
    # The regression this fix exists for. The low limit stands in for the real one:
    # the old form's cache grows with the product of the axes, so any fixed budget is
    # eventually spent on a model with hundreds of norms.
    torch._dynamo.reset()
    with torch._dynamo.config.patch(**_limit_patch(8)):
        compiled = {}

        def call(hidden_states, weight, with_scale, hidden, dtype):
            key = (hidden, with_scale, dtype)
            if key not in compiled:
                module = _OldNorm(hidden, with_scale, dtype).cuda()
                if with_scale:
                    module.weight.data = module.weight.data.to(dtype)
                compiled[key] = torch.compile(module, fullgraph = True)
            compiled[key](hidden_states)

        with pytest.raises(_LIMIT_ERROR):
            _run_every_case(call)


def test_the_kernels_survive_the_same_budget():
    torch._dynamo.reset()
    with torch._dynamo.config.patch(**_limit_patch(8)):
        _run_every_case(
            lambda hidden_states, weight, with_scale, hidden, dtype:
                _new(hidden_states, weight, 1e-6, with_scale)
        )


def test_a_parameter_view_is_not_itself_a_parameter():
    # Why the width guard goes away: Dynamo forces a static shape when
    # `type(t) is torch.nn.Parameter`, even under `dynamic=True`. A view is a plain
    # Tensor, so it takes dynamic shapes.
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


def test_the_kernels_fall_back_to_eager_instead_of_aborting():
    """Cache exhaustion must latch to eager, not kill the run.

    `patch_function` wraps whatever it compiles with `fullgraph = True` in
    `_fall_back_to_eager_on_recompile_limit`; a bare decorator skips it, turning the
    exact failure this file is about into a hard abort.
    """
    from unsloth_zoo.temporary_patches.gemma4_float32 import (
        _gemma4_rms_norm_scaled, _gemma4_rms_norm_unscaled,
    )
    from unsloth_zoo.temporary_patches.gemma import (
        _gemma3_rms_norm_float32, _gemma3_rms_norm_generic,
    )
    from unsloth_zoo.temporary_patches.qwen3_moe_float32 import _qwen3_moe_rms_norm

    kernels = (_gemma4_rms_norm_scaled, _gemma4_rms_norm_unscaled,
               _gemma3_rms_norm_float32, _gemma3_rms_norm_generic,
               _qwen3_moe_rms_norm)
    for kernel in kernels:
        # The wrapper's own marker. `get_compiler_config` is no signal: the wrapper
        # re-exposes it on purpose, so callers cannot tell the two apart.
        assert hasattr(kernel, "_unsloth_fallback_state"), (
            f"{kernel.__name__} is a bare compiled function, so cache exhaustion "
            f"under fullgraph raises instead of falling back to eager"
        )

    # And it really does run. `fail_on_recompile_limit_hit` is deliberately NOT set:
    # that flag is the user asking for a hard stop and the wrapper re-raises for it,
    # while real runs leave it off, so limit plus `fullgraph = True` is the case that
    # matters. Widths cost nothing under `dynamic = True`; dtype and grad mode do.
    # The latch is module-level, permanent and untouched by reset(), so restore it or
    # every later test in this worker silently gets the eager kernel.
    saved = {k: dict(k._unsloth_fallback_state) for k in kernels}
    try:
        torch._dynamo.reset()
        with torch._dynamo.config.patch(**{_LIMIT_KEY: 1, "suppress_errors": False}):
            for dtype in DTYPES:
                for grad in (True, False):
                    with torch.set_grad_enabled(grad):
                        x = torch.randn(4, 128, device = "cuda", dtype = dtype)
                        w = torch.randn(128, device = "cuda", dtype = dtype)
                        out = _gemma4_rms_norm_scaled(x, w, 1e-6)
                        assert out.shape == x.shape
                        assert torch.isfinite(out).all()

        assert _gemma4_rms_norm_scaled._unsloth_fallback_state["eager"], (
            "the cache was never actually exhausted, so this asserted nothing"
        )
    finally:
        for kernel, state in saved.items():
            kernel._unsloth_fallback_state.clear()
            kernel._unsloth_fallback_state.update(state)
        torch._dynamo.reset()


def test_each_kernel_reports_its_own_fallback_state():
    """Two wrappers sharing a label hide each other in the public diagnostic.

    `eager_fallback_state()` keys by label, so a later non-latched entry could
    overwrite a latched one and report the path as still compiled.
    """
    from unsloth_zoo.temporary_patches.utils import eager_fallback_state
    from unsloth_zoo.temporary_patches.gemma4_float32 import (
        _gemma4_rms_norm_scaled, _gemma4_rms_norm_unscaled,
    )
    from unsloth_zoo.temporary_patches.gemma import (
        _gemma3_rms_norm_float32, _gemma3_rms_norm_generic,
    )

    labels = [k._unsloth_fallback_label for k in
              (_gemma4_rms_norm_scaled, _gemma4_rms_norm_unscaled,
               _gemma3_rms_norm_float32, _gemma3_rms_norm_generic)]
    assert len(set(labels)) == len(labels), f"labels collide: {labels}"

    # One latched, its neighbour not: the dict must show the latched one.
    saved = dict(_gemma4_rms_norm_scaled._unsloth_fallback_state)
    _gemma4_rms_norm_scaled._unsloth_fallback_state["eager"] = True
    try:
        state = eager_fallback_state()
        assert state[_gemma4_rms_norm_scaled._unsloth_fallback_label] is True
        assert state[_gemma4_rms_norm_unscaled._unsloth_fallback_label] is False
    finally:
        _gemma4_rms_norm_scaled._unsloth_fallback_state.update(saved)
