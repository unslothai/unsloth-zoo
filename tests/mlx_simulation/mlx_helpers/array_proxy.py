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

"""Monkey-patch torch.Tensor with the MLX-only methods PR code uses.

Idempotent — calling twice is a no-op.  Called from
simulate_mlx_on_torch() before any unsloth_zoo MLX module is imported.

Methods added:
  * .astype(dtype)      ->  .to(dtype)              [100 sites in PR-A]
  * .expand_dims(axis)  ->  .unsqueeze(axis)        [33 sites]
  * .at[idx].set/add/multiply/...  ->  JAX-style functional update
                                          [used in gated_delta_vjp.py]

Methods overridden (only when the args mismatch torch's expectations):
  * .transpose(*axes)   when given >2 axes -> .permute(*axes)
                        because MLX uses N-axis permutation but
                        torch.Tensor.transpose only swaps two dims.

We do NOT override existing torch behavior unconditionally — only when
the call signature is unambiguous MLX-style.
"""

from __future__ import annotations

import torch


_PATCHED = False


class _AtProxy:
    """JAX-style at[idx].set/add/multiply functional update."""
    __slots__ = ("_tensor", "_idx")

    def __init__(self, tensor, idx):
        self._tensor = tensor
        self._idx = idx

    def set(self, value):
        out = self._tensor.clone()
        out[self._idx] = value
        return out

    def add(self, value):
        out = self._tensor.clone()
        out[self._idx] = out[self._idx] + value
        return out

    def multiply(self, value):
        out = self._tensor.clone()
        out[self._idx] = out[self._idx] * value
        return out

    def divide(self, value):
        out = self._tensor.clone()
        out[self._idx] = out[self._idx] / value
        return out

    def maximum(self, value):
        out = self._tensor.clone()
        out[self._idx] = torch.maximum(out[self._idx], value)
        return out

    def minimum(self, value):
        out = self._tensor.clone()
        out[self._idx] = torch.minimum(out[self._idx], value)
        return out


class _AtAccessor:
    """Returned by `tensor.at` — supports `tensor.at[idx]` indexing."""
    __slots__ = ("_tensor",)

    def __init__(self, tensor):
        self._tensor = tensor

    def __getitem__(self, idx):
        return _AtProxy(self._tensor, idx)


def patch_tensor_with_mlx_methods():
    """Install MLX-flavored methods on torch.Tensor.  Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # .astype(dtype) — MLX's name for .to(dtype)
    def astype(self, dtype):
        return self.to(dtype)
    torch.Tensor.astype = astype

    # .expand_dims(axis) — MLX's name for .unsqueeze(dim)
    def expand_dims(self, axis):
        return self.unsqueeze(axis)
    torch.Tensor.expand_dims = expand_dims

    # .transpose(*axes) — when given >=3 args (MLX-style multi-axis permute),
    # delegate to .permute().  When given 0 or 2 args (torch-compatible),
    # use torch's original behavior.
    if not hasattr(torch.Tensor, "_orig_transpose_for_mlx_shim"):
        torch.Tensor._orig_transpose_for_mlx_shim = torch.Tensor.transpose

        def transpose(self, *args):
            if len(args) >= 3:
                return self.permute(*args)
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                seq = args[0]
                if len(seq) >= 3:
                    return self.permute(*seq)
                # 2-axis tuple form
                return torch.Tensor._orig_transpose_for_mlx_shim(self, *seq)
            if len(args) == 0:
                # MLX no-arg transpose = reverse all axes
                return self.permute(*reversed(range(self.ndim)))
            return torch.Tensor._orig_transpose_for_mlx_shim(self, *args)

        torch.Tensor.transpose = transpose

    # .at JAX-style indexing
    if not hasattr(torch.Tensor, "at"):
        @property
        def at(self):
            return _AtAccessor(self)
        torch.Tensor.at = at
