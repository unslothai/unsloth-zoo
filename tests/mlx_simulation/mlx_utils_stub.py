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

# Unsloth Zoo - Utilities for Unsloth
# MLX utils stub: tree_map, tree_flatten, tree_unflatten, etc.
"""
mlx.utils — tree functions that walk nested dicts/lists/tuples/namedtuples.

MLX's tree_flatten produces flat dot-joined string keys
(`"layers.0.attention.q_proj.weight"`); torch.utils._pytree uses
opaque tree-spec objects.  We wrap torch's pytree to produce
mlx-flavored output.
"""

from __future__ import annotations

import sys
import types
from typing import Any


def _is_dict_like(x):
    return isinstance(x, dict)


def _is_list_like(x):
    return isinstance(x, (list, tuple))


def tree_flatten(tree, prefix: str = "", is_leaf=None) -> list[tuple[str, Any]]:
    """Flatten an mlx-style tree to a list of (dotted_key, leaf) pairs."""
    if is_leaf is not None and is_leaf(tree):
        return [(prefix, tree)]
    if _is_dict_like(tree):
        out = []
        for k, v in tree.items():
            sub_prefix = f"{prefix}.{k}" if prefix else str(k)
            out.extend(tree_flatten(v, prefix=sub_prefix, is_leaf=is_leaf))
        return out
    if _is_list_like(tree):
        out = []
        for i, v in enumerate(tree):
            sub_prefix = f"{prefix}.{i}" if prefix else str(i)
            out.extend(tree_flatten(v, prefix=sub_prefix, is_leaf=is_leaf))
        return out
    return [(prefix, tree)]


def tree_unflatten(items: list[tuple[str, Any]]) -> Any:
    """Inverse of tree_flatten: rebuild a nested dict from flat keys."""
    if not items:
        return {}
    # If all top-level keys are integers, rebuild a list.
    out: dict = {}
    for path, value in items:
        parts = path.split(".") if path else []
        cur = out
        for i, part in enumerate(parts[:-1]):
            if part not in cur:
                cur[part] = {}
            cur = cur[part]
        if parts:
            cur[parts[-1]] = value
    return _maybe_listify(out)


def _maybe_listify(node):
    """Recursively turn dicts with all-integer keys into lists."""
    if isinstance(node, dict):
        keys = list(node.keys())
        all_int = bool(keys) and all(isinstance(k, str) and k.isdigit() for k in keys)
        if all_int:
            sorted_keys = sorted(keys, key=int)
            return [_maybe_listify(node[k]) for k in sorted_keys]
        return {k: _maybe_listify(v) for k, v in node.items()}
    return node


def tree_map(fn, tree, *rest, is_leaf=None):
    """Apply fn to every leaf of `tree`, with parallel leaves from *rest."""
    if is_leaf is not None and is_leaf(tree):
        return fn(tree, *rest)
    if _is_dict_like(tree):
        return {k: tree_map(fn, v,
                            *(r[k] if isinstance(r, dict) else r for r in rest),
                            is_leaf=is_leaf)
                for k, v in tree.items()}
    if _is_list_like(tree):
        return type(tree)(
            tree_map(fn, v,
                     *(r[i] if isinstance(r, (list, tuple)) else r for r in rest),
                     is_leaf=is_leaf)
            for i, v in enumerate(tree)
        )
    return fn(tree, *rest)


def tree_map_with_path(fn, tree, prefix: str = "", is_leaf=None):
    """Like tree_map but the callback receives the dotted path as first arg."""
    if is_leaf is not None and is_leaf(tree):
        return fn(prefix, tree)
    if _is_dict_like(tree):
        return {k: tree_map_with_path(fn, v,
                                      prefix=f"{prefix}.{k}" if prefix else str(k),
                                      is_leaf=is_leaf)
                for k, v in tree.items()}
    if _is_list_like(tree):
        return type(tree)(
            tree_map_with_path(fn, v,
                               prefix=f"{prefix}.{i}" if prefix else str(i),
                               is_leaf=is_leaf)
            for i, v in enumerate(tree)
        )
    return fn(prefix, tree)


def tree_reduce(fn, tree, init, is_leaf=None):
    leaves = [v for _, v in tree_flatten(tree, is_leaf=is_leaf)]
    acc = init
    for v in leaves:
        acc = fn(acc, v)
    return acc


def tree_merge(a, b, merge_fn=None):
    out = {}
    for k in set(a) | set(b):
        if k in a and k in b:
            if isinstance(a[k], dict) and isinstance(b[k], dict):
                out[k] = tree_merge(a[k], b[k], merge_fn=merge_fn)
            elif merge_fn is not None:
                out[k] = merge_fn(a[k], b[k])
            else:
                out[k] = b[k]
        elif k in a:
            out[k] = a[k]
        else:
            out[k] = b[k]
    return out


# ---------------------------------------------------------------------------
def inject_into_sys_modules():
    this = sys.modules[__name__]
    sys.modules["mlx.utils"] = this
    # Also expose as attribute on top-level mlx package.
    if "mlx" in sys.modules:
        setattr(sys.modules["mlx"], "utils", this)
