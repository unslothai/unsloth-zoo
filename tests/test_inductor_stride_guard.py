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

"""torch 2.13's Layout._stride_expr_ge_or_false builds `left % right`, which SymPy turns into a
reciprocal power for compound symbolic strides, and value-range analysis then raises
`Exponent must be non-negative.` (a dynamic compile of MiniMax-M3 attention on CUDA dies there).
Runs on CPU: the layout check is pure symbolic code, the same strides as pytorch#190965's test."""

from unittest import mock

import pytest

torch = pytest.importorskip("torch")
sympy = pytest.importorskip("sympy")

from torch._inductor import ir  # noqa: E402

pytestmark = pytest.mark.skipif(
    not hasattr(ir.Layout, "_stride_expr_ge_or_false"),
    reason = "torch without Layout._stride_expr_ge_or_false (2.12 and older)",
)


def _strides():
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch.utils._sympy.functions import FloorDiv

    shape_env = ShapeEnv()
    seq = shape_env.create_unbacked_symint().node.expr
    shape_env.constrain_symbol_range(seq, 4, 1024)
    half = FloorDiv(seq, 2)
    reduced = FloorDiv(half**2, half)
    left = 16 * reduced - 30
    right = 16 * (64 * half * reduced - 120 * half - 120 * reduced + 225)
    return shape_env, seq, left, right


def _patched():
    # The package import is what users get; it applies the backport where torch needs it.
    import unsloth_zoo.temporary_patches  # noqa: F401


def test_compound_strides_do_not_crash_stride_order_check():
    _patched()
    from torch._inductor.sizevars import SizeVarAllocator
    from torch._inductor.virtualized import V

    shape_env, seq, left, right = _strides()
    layout = ir.FixedLayout(torch.device("cpu"), torch.float32, size = [seq, 2], stride = [left, right])
    with V.set_graph_handler(mock.Mock(sizevars = SizeVarAllocator(shape_env))):
        # torch 2.13.0 raises ValueError("Exponent must be non-negative.") here.
        assert layout.is_stride_ordered([0, 1]) in (True, False)


def test_divisibility_guard_has_no_reciprocal_power():
    _patched()
    from torch._inductor.virtualized import V

    _, _, left, right = _strides()
    sizevars = mock.Mock()
    sizevars.guard_or_false.return_value = False
    sizevars.statically_known_multiple_of.return_value = False
    with V.set_graph_handler(mock.Mock(sizevars = sizevars)):
        assert ir.Layout._stride_expr_ge_or_false(left, right) is False
    guard = sizevars.guard_or_false.call_args.args[0]
    negative = [p for p in guard.atoms(sympy.Pow) if p.args[1].is_negative is True]
    assert negative == [], guard


def test_patch_only_replaces_the_unfixed_body_and_is_idempotent():
    from unsloth_zoo.temporary_patches import inductor_stride_guard as s
    fn = ir.Layout._stride_expr_ge_or_false
    installed = s.patch_inductor_stride_guard()
    assert s.patch_inductor_stride_guard() == installed
    assert ir.Layout._stride_expr_ge_or_false is fn
    # Installed exactly when the torch body still had `left % right` (2.13); 2.14+ is left alone.
    assert installed == getattr(fn, "_unsloth_stride_guard", False)
