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

"""Lookup in a host-RAM embedding table as one opaque op a compiled, CUDA graphed region can call.
Inference only: no autograd formula, and it skips the module's forward hooks."""

from typing import Optional

import torch

__all__ = ["offloaded_embedding"]

_OP_NAME = "unsloth_zoo::offloaded_embedding"


def _offloaded_embedding_impl(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    padding_idx: Optional[int] = None,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Blocking copies: a non_blocking copy into the CPU can be read before it lands.
    ids = input_ids.to(weight.device)
    rows = torch.nn.functional.embedding(ids, weight, padding_idx)
    if scale is not None:
        # transformers' ScaledWordEmbedding product, in the table's dtype, so rows match bit for bit.
        rows = rows * scale.to(device = weight.device, dtype = weight.dtype)
    return rows.to(input_ids.device)


def _register():
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:
        return None
    # cudagraph_unsafe runs the lookup outside the CUDA graph; inlined, it read ids not yet copied back.
    tag = getattr(torch._C.Tag, "cudagraph_unsafe", None)
    # torch before 2.8 has no `tags` keyword at all: the last attempt omits it.
    attempts = ([dict(tags = (tag,))] if tag is not None else []) + [dict()]
    for kwargs in attempts:
        try:
            op = custom_op(_OP_NAME, mutates_args = (), **kwargs)(_offloaded_embedding_impl)
            break
        except Exception:
            continue
    else:
        try:
            return torch.ops.unsloth_zoo.offloaded_embedding
        except Exception:
            return None

    @op.register_fake
    def _(input_ids, weight, padding_idx = None, scale = None):
        return input_ids.new_empty((*input_ids.shape, weight.shape[-1]), dtype = weight.dtype)

    return op


# None when this torch has no torch.library.custom_op: callers fall back to nn.Embedding.
offloaded_embedding = _register()
