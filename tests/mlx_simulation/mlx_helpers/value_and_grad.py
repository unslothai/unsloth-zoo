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

"""nn.value_and_grad(model, fn) -> functional autograd over a torch model."""

from __future__ import annotations

import torch


def nn_value_and_grad(model, fn):
    """Return a callable (*args) -> (loss, grads_tree).

    Phase 1 implementation: assume model.parameters() returns a flat
    dict of trainable torch tensors; backward via torch.autograd.grad.
    Phase 4 will extend to MLX's nested-dict tree structure.
    """

    def _wrapped(*args, **kwargs):
        # Gather params (flat list with name index)
        params_dict = model.parameters() if callable(getattr(model, "parameters", None)) else {}
        if not isinstance(params_dict, dict):
            params_dict = {}
        names, tensors = [], []
        for k, v in _flatten_params(params_dict).items():
            if isinstance(v, torch.Tensor):
                v.requires_grad_(True)
                names.append(k)
                tensors.append(v)

        out = fn(*args, **kwargs)
        loss = out if isinstance(out, torch.Tensor) else out[0]

        if not tensors:
            return loss, {}

        grads = torch.autograd.grad(
            loss, tensors, allow_unused=True, retain_graph=False, create_graph=False,
        )
        grads_tree = {n: g if g is not None else torch.zeros_like(t)
                      for n, g, t in zip(names, grads, tensors)}
        return loss, _unflatten_params(grads_tree)

    return _wrapped


def _flatten_params(node, prefix=""):
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            sub = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_params(v, prefix=sub))
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            sub = f"{prefix}.{i}" if prefix else str(i)
            out.update(_flatten_params(v, prefix=sub))
    else:
        out[prefix] = node
    return out


def _unflatten_params(flat):
    out = {}
    for k, v in flat.items():
        parts = k.split(".") if k else []
        cur = out
        for p in parts[:-1]:
            if p not in cur:
                cur[p] = {}
            cur = cur[p]
        if parts:
            cur[parts[-1]] = v
    return out
