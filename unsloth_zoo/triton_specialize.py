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
"""Triton int specialization helpers for per-call sizes.

Triton compiles one binary per (== 1, % 16, other) class of every int argument. A token / row count is 1 in
batch 1 decode and anything in prefill, so a kernel taking it plainly builds a third binary mid-session the first
time a count is a multiple of 16. jit_any_alignment keeps the == 1 binary (its folded index math is the fast
decode path) and drops only the % 16 class (do_not_specialize_on_alignment)."""

__all__ = ["jit_any_alignment"]

try:
    import triton
except Exception:  # CPU-only installs: kernels are never built
    triton = None


def jit_any_alignment(*names):
    """triton.jit with do_not_specialize_on_alignment = names; plain triton.jit on a Triton without it."""
    def wrap(fn):
        if triton is None:
            return fn
        try:
            return triton.jit(fn, do_not_specialize_on_alignment = list(names))
        except TypeError:  # a Triton without do_not_specialize_on_alignment
            return triton.jit(fn)
    return wrap
