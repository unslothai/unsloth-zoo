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

"""
Backport of pytorch/pytorch#190965 (commit 93879b11) to torch 2.13.

torch 2.13 added `Layout._stride_expr_ge_or_false`, whose last check is
`guard_or_false(Eq(left % right, 0))`. With compound symbolic strides, Python `%`
on SymPy expressions rewrites the modulo through a reciprocal power (`x**-1`),
and value-range analysis then dies with
`ValueError: Exponent must be non-negative.` (or `ValueRangeError: Invalid
ranges [0:-1]`). Any `torch.compile(dynamic=True)` whose layouts hit
`is_stride_ordered` with such strides fails to compile, e.g. MiniMax-M3 attention
on CUDA (pytorch/pytorch#188323, #188422).

torch 2.14 fixed it by trying `statically_known_multiple_of` first and building the
guard with PyTorch's non-simplifying integer `Mod`. This module installs exactly
that body, and only when the installed function still has the `left % right`
form, so torch 2.12 and older (no such function) and 2.14+ (fixed) are untouched.
"""

import inspect

__all__ = ["patch_inductor_stride_guard"]


def _needs_patch(fn):
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return "left % right" in source


def patch_inductor_stride_guard():
    """Returns True if the backport is installed (now or earlier), else False."""
    try:
        import sympy
        from torch._inductor import ir
        from torch._inductor.virtualized import V
        from torch.utils._sympy.functions import Mod
    except Exception:
        return False

    layout = getattr(ir, "Layout", None)
    original = getattr(layout, "_stride_expr_ge_or_false", None) if layout is not None else None
    if original is None:
        return False
    if getattr(original, "_unsloth_stride_guard", False):
        return True
    if not _needs_patch(original):
        return False

    def _stride_expr_ge_or_false(left, right):
        sizevars = V.graph.sizevars
        if sizevars.guard_or_false(sympy.Eq(right, 0)):
            return True
        if sizevars.guard_or_false(sympy.Eq(left, 0)):
            return False
        return (
            sizevars.guard_or_false(sympy.Ge(left, right))
            or sizevars.statically_known_multiple_of(left, right)
            or sizevars.guard_or_false(sympy.Eq(Mod(left, right), 0))
        )

    _stride_expr_ge_or_false._unsloth_stride_guard = True
    _stride_expr_ge_or_false.__doc__ = original.__doc__
    layout._stride_expr_ge_or_false = staticmethod(_stride_expr_ge_or_false)
    return True


# Apply at import time so it is active before any compilation.
try:
    patch_inductor_stride_guard()
except Exception:
    pass
