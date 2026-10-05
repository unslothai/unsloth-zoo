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

"""Embedding lookup against a table offloaded to host RAM, as one opaque op a compiled (and CUDA
graphed) region can call: the ids go to the table's device, the rows come back to the ids' device.
An optional `scale` multiplies the rows as transformers' `ScaledWordEmbedding` does. Inference only
(no autograd formula, and it skips the module's forward hooks): callers keep the module whenever
grad is enabled."""

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
    # Blocking copies both ways: a non_blocking copy into the CPU can be read before it lands.
    # padding_idx only changes the gradient of F.embedding, so the rows are the same either way.
    ids = input_ids.to(weight.device)
    rows = torch.nn.functional.embedding(ids, weight, padding_idx)
    if scale is not None:
        # The transformers `ScaledWordEmbedding` product, on the table's device in its dtype as the
        # module computes it, so the rows match it bit for bit.
        rows = rows * scale.to(device = weight.device, dtype = weight.dtype)
    return rows.to(input_ids.device)


def _register():
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:
        return None
    # cudagraph_unsafe makes Inductor's graph partitioning run the lookup outside the CUDA graph,
    # in order with the rest of the step; inlined, the CPU kernel read token ids the graph had
    # not copied back yet.
    tag = getattr(torch._C.Tag, "cudagraph_unsafe", None)
    for tags in (((tag,),) if tag is not None else ()) + ((),):
        try:
            op = custom_op(_OP_NAME, mutates_args = (), tags = tags)(_offloaded_embedding_impl)
            break
        except Exception:
            continue
    else:
        # Already defined in this process (a module reload): reuse the registered op.
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
