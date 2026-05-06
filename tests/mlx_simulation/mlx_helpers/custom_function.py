# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""mx.custom_function — closure-style differentiable op as torch.autograd.Function.

MLX pattern:

    @mx.custom_function
    def fn(*primals): return primal
    @fn.vjp
    def fn_vjp(primals, cotangents, outputs): return grads

We wrap into a `torch.autograd.Function` whose `forward` calls the
primal and `backward` calls the user's vjp closure.
"""

from __future__ import annotations

import torch


def make_custom_function(primal_fn):
    """Decorator that returns an object exposing .vjp(...) and __call__."""

    class _MLXCustom:
        def __init__(self, primal):
            self.primal_fn = primal
            self.vjp_fn = None

        def vjp(self, vjp_fn):
            """Register the VJP function; returns it (so it can be used as a decorator)."""
            self.vjp_fn = vjp_fn
            return vjp_fn

        def __call__(self, *primals):
            outer = self
            primal_fn = self.primal_fn

            class _AG(torch.autograd.Function):
                @staticmethod
                def forward(ctx, *primals_):
                    out = primal_fn(*primals_)
                    ctx.save_for_backward(*[
                        p for p in primals_ if isinstance(p, torch.Tensor)
                    ])
                    ctx._mlx_outputs = out
                    if isinstance(out, (tuple, list)):
                        return tuple(out)
                    return out

                @staticmethod
                def backward(ctx, *cotangents):
                    if outer.vjp_fn is None:
                        raise RuntimeError(
                            f"mlx-shim: custom_function {primal_fn.__name__!r} "
                            f"has no .vjp() registered."
                        )
                    primals_ = ctx.saved_tensors
                    outputs_ = ctx._mlx_outputs
                    grads = outer.vjp_fn(primals_, cotangents, outputs_)
                    if isinstance(grads, (tuple, list)):
                        return tuple(grads)
                    return (grads,)

            return _AG.apply(*primals)

    return _MLXCustom(primal_fn)
