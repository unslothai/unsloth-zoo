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

"""Encode each distinct image of a vision forward once (GRPO repeats each prompt's images G times).

Copies are row gathers with a deterministic summed backward. Stands down while vision dropout is
active. ``UNSLOTH_GRPO_IMAGE_DEDUP=0`` disables it (read per call).
"""

__all__ = [
    "install_image_dedup",
    "image_dedup_enabled",
    "DEDUP_STATS",
]

import functools
import inspect
import logging
import os
import weakref

import torch

logger = logging.getLogger(__name__)

DEDUP_STATS = {
    "calls": 0,           # wrapped get_image_features calls with 2+ images and the switch on
    "deduped_calls": 0,   # calls that encoded fewer images than they were given
    "images_in": 0,
    "images_encoded": 0,
    "fallbacks": 0,       # outputs this module could not map back: encoded in full instead
    "verified": 0,        # first-use self-checks passed
    "last_fallback": None,
}

_INSTALLED = "_unsloth_image_dedup_installed"
_DISABLED = "_unsloth_image_dedup_disabled"
_VERIFIED = "_unsloth_image_dedup_verified"
_STOCHASTIC = "_unsloth_image_dedup_stochastic"
_SCANNED = "_unsloth_image_dedup_scanned"

# Per-image tensors that may ride along with row-layout pixel_values (one row per image).
_PER_IMAGE_KWARGS = ("image_position_ids", "pixel_position_ids", "pixel_attention_mask", "image_sizes")
_GRID_KWARG = "image_grid_thw"
_LANGUAGE_PREFIXES = ("language_model", "lm_head", "text_model")


def image_dedup_enabled():
    return os.environ.get("UNSLOTH_GRPO_IMAGE_DEDUP", "1") != "0"


class _Unmappable(Exception):
    pass


class _BlockExpand(torch.autograd.Function):
    """src[row_index]; backward sums in fixed order (index_add atomics are nondeterministic)."""

    @staticmethod
    def forward(ctx, src, row_index, blocks):
        ctx.blocks = blocks
        ctx.src_shape = src.shape
        return src.index_select(0, row_index)

    @staticmethod
    def backward(ctx, grad):
        grad_src = grad.new_zeros(ctx.src_shape)
        # Half-precision copies are summed in fp32 and rounded once: a running fp16 sum overflows on partial
        # sums (40000 + 40000 - 40000 - 40000 gave inf) that the unwrapped tower accumulates without.
        wide = torch.float32 if grad.dtype in (torch.float16, torch.bfloat16) else grad.dtype
        for s0, length, copies in ctx.blocks:
            acc = grad[copies[0] : copies[0] + length]
            if len(copies) > 1:
                acc = acc.to(wide)
                for c in copies[1:]:
                    acc = acc + grad[c : c + length]
            grad_src[s0 : s0 + length] = acc
        return grad_src, None, None


def _expand(src, seg_sizes, inverse):
    """Rows of ``src`` = unique blocks of ``seg_sizes`` rows, laid out for ``inverse``."""
    starts = [0]
    for s in seg_sizes:
        starts.append(starts[-1] + s)
    if starts[-1] != src.shape[0]:
        raise _Unmappable()
    index, copies, dst = [], [[] for _ in seg_sizes], 0
    for u in inverse:
        copies[u].append(dst)
        index.extend(range(starts[u], starts[u] + seg_sizes[u]))
        dst += seg_sizes[u]
    row_index = torch.tensor(index, dtype = torch.long, device = src.device)
    if not src.requires_grad:
        return src.index_select(0, row_index)
    blocks = [(starts[u], seg_sizes[u], copies[u]) for u in range(len(seg_sizes)) if copies[u]]
    return _BlockExpand.apply(src, row_index, blocks)


def _groups(flat, extras):
    """(reps, inverse): rows equal in ``flat`` and every ``extras`` tensor share a representative."""
    n = flat.shape[0]
    fp = torch.stack(
        [
            flat.sum(1, dtype = torch.float32),
            flat[:, 1::2].sum(1, dtype = torch.float32),
            flat[:, ::7].sum(1, dtype = torch.float32),
        ],
        dim = 1,
    ).tolist()
    first = {}
    rep_of = []
    for i in range(n):
        key = tuple(fp[i])
        rep_of.append(first.setdefault(key, i))
    members = [i for i in range(n) if rep_of[i] != i]
    if members:
        m = torch.tensor(members, device = flat.device)
        r = torch.tensor([rep_of[i] for i in members], device = flat.device)
        same = (flat.index_select(0, m) == flat.index_select(0, r)).all(1)
        for e in extras:
            e = e.reshape(n, -1)
            same = same & (e.index_select(0, m.to(e.device)) == e.index_select(0, r.to(e.device))).all(1).to(same.device)
        for i, ok in zip(members, same.tolist()):
            if not ok:
                rep_of[i] = i
    reps, slot, inverse = [], {}, []
    for i in range(n):
        j = rep_of[i]
        if j not in slot:
            slot[j] = len(reps)
            reps.append(j)
        inverse.append(slot[j])
    return reps, inverse


def _stochastic_vision(module):
    """Whether a vision-side layer of ``module`` could draw differently for two equal images now."""
    cached = module.__dict__.get(_STOCHASTIC, None)
    if cached is None:
        cached = []
        for name, sub in module.named_modules():
            if any(part in _LANGUAGE_PREFIXES for part in name.split(".")):
                continue
            if isinstance(sub, torch.nn.modules.dropout._DropoutNd):
                cached.append((sub, "p"))
                continue
            # drop_prob: timm / transformers DropPath (stochastic depth) layers, e.g. QianfanOCR's vision tower.
            for attr in ("dropout", "attention_dropout", "attn_dropout", "dropout_p", "drop_path_rate", "drop_prob"):
                value = sub.__dict__.get(attr, None)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    cached.append((sub, attr))
        module.__dict__[_STOCHASTIC] = cached
    return any(sub.training and float(getattr(sub, attr, 0) or 0) > 0 for sub, attr in cached)


def _map_rows(obj, inverse, n_unique):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if torch.is_tensor(obj):
        if obj.ndim == 0 or obj.shape[0] != n_unique:
            raise _Unmappable(f"tensor {tuple(obj.shape)} with {n_unique} unique images")
        return _expand(obj, [1] * n_unique, inverse)
    if isinstance(obj, (tuple, list)):
        return type(obj)(_map_rows(x, inverse, n_unique) for x in obj)
    if hasattr(obj, "keys") and hasattr(obj, "__setitem__"):
        for key in list(obj.keys()):
            obj[key] = _map_rows(obj[key], inverse, n_unique)
        return obj
    raise _Unmappable(f"{type(obj).__name__}")


def _map_grid(obj, inverse, rows_u):
    """Grid layout (Qwen-VL): map outputs back by image segment."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    n_unique = len(rows_u)
    if torch.is_tensor(obj):
        if obj.ndim == 0:
            raise _Unmappable()
        n = obj.shape[0]
        total = sum(rows_u)
        if n == total:
            return _expand(obj, rows_u, inverse)
        if n > 0 and total % n == 0 and all(r % (total // n) == 0 for r in rows_u):
            ratio = total // n
            return _expand(obj, [r // ratio for r in rows_u], inverse)
        if n == n_unique:
            return _expand(obj, [1] * n_unique, inverse)
        raise _Unmappable()
    if isinstance(obj, (tuple, list)):
        if (
            len(obj) == n_unique
            and all(torch.is_tensor(x) and x.ndim >= 1 for x in obj)
        ):
            sizes = [x.shape[0] for x in obj]
            if all(r % s == 0 for r, s in zip(rows_u, sizes) if s) and all(sizes) and len(
                {r // s for r, s in zip(rows_u, sizes)}
            ) == 1:
                return type(obj)(obj[u] for u in inverse)
        return type(obj)(_map_grid(x, inverse, rows_u) for x in obj)
    if hasattr(obj, "keys") and hasattr(obj, "__setitem__"):
        for key in list(obj.keys()):
            obj[key] = _map_grid(obj[key], inverse, rows_u)
        return obj
    raise _Unmappable()


def _flat_tensors(obj, out):
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, (tuple, list)):
        for x in obj:
            _flat_tensors(x, out)
    elif hasattr(obj, "keys") and hasattr(obj, "values"):
        for x in obj.values():
            _flat_tensors(x, out)
    return out


def _outputs_mismatch(a, b, inverse):
    """None if ``a`` has the layout of full encode ``b``, else the reason.

    Structural, not tight: compiled towers vary ~12% with batch size, distinct images ~50%.
    """
    ta, tb = _flat_tensors(a, []), _flat_tensors(b, [])
    if len(ta) != len(tb):
        return f"{len(ta)} tensors vs {len(tb)}"
    n = len(inverse)
    groups = torch.tensor(inverse)
    for k, (x, y) in enumerate(zip(ta, tb)):
        if x.shape != y.shape:
            return f"tensor {k}: shape {tuple(x.shape)} vs {tuple(y.shape)}"
        if not x.is_floating_point():
            if not torch.equal(x, y):
                return f"tensor {k}: integer values differ"
            continue
        x, y = x.detach().float(), y.detach().float()
        if x.ndim >= 2 and x.shape[0] == n and len(set(inverse)) > 1:
            dist = torch.cdist(x.flatten(1), y.flatten(1)).cpu()
            same = groups[:, None] == groups[None, :]
            own = dist.masked_fill(~same, float("inf")).min(1).values
            other = dist.masked_fill(same, float("inf")).min(1).values
            if not bool((own <= other).all()):
                return f"tensor {k} {tuple(x.shape)}: an image is nearer another image's features"
            continue
        err = ((x - y).norm() / y.norm().clamp_min(1e-12)).item()
        if not err < 0.3:
            return f"tensor {k} {tuple(x.shape)}: |d|/|y| = {err:.3e}"
    return None


def _fallback(module, reason):
    module.__dict__[_DISABLED] = True
    DEDUP_STATS["fallbacks"] += 1
    DEDUP_STATS["last_fallback"] = reason
    logger.warning(f"Unsloth: encoding every image again for {type(module).__name__} ({reason}).")


try:
    from torch.distributed.tensor import DTensor as _DTensor
except Exception:
    try:
        from torch.distributed._tensor import DTensor as _DTensor
    except Exception:
        _DTensor = None


def _sharded_parameters(module):
    # FSDP2 (DTensor) or ZeRO-3 (ds_id) parameters all-gather on every forward. Sharding happens inside
    # train(), after install: fully_shard replaces the Parameter objects, so the answer is cached against
    # the first parameter's identity and re-read once it changes.
    first = next(module.parameters(), None)
    cached = module.__dict__.get("_unsloth_image_dedup_sharded", None)
    if cached is not None and cached[0]() is first:
        return cached[1]
    value = any(
        hasattr(p, "ds_id") or (_DTensor is not None and isinstance(p, _DTensor)) for p in module.parameters()
    )
    module.__dict__["_unsloth_image_dedup_sharded"] = (weakref.ref(first) if first is not None else (lambda: None), value)
    return value


def _zero3(module):
    # DeepSpeed ZeRO-3 gathers each partitioned parameter on every forward (a collective on every rank).
    try:
        from unsloth_zoo.temporary_patches.utils import zero3_partitioned_training
        if zero3_partitioned_training():
            return True
    except Exception:
        pass
    return _sharded_parameters(module)


def _collective(module):
    # FSDP2 (Unsloth's resharding layout or any DTensor-sharded tower) and ZeRO-3 towers all-gather per
    # forward; a rank-local self-check or fallback forward would hang the other ranks.
    if _zero3(module):
        return True
    try:
        from unsloth_zoo.rl_replacements import grpo_forwards_are_collective
        return grpo_forwards_are_collective(module)
    except Exception:
        return False


def _make_wrapper(module, original):
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        signature = None

    # transformers >= 5.23 generate pops encoder inputs by this signature; (*args, **kwargs) breaks vision generate.
    @functools.wraps(original)
    def get_image_features(*args, **kwargs):
        if (
            signature is None
            or not image_dedup_enabled()
            or module.__dict__.get(_DISABLED, False)
            or _collective(module)
        ):
            return original(*args, **kwargs)
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            return original(*args, **kwargs)
        named = dict(bound.arguments)
        var_kw = None
        for p in signature.parameters.values():
            if p.kind is inspect.Parameter.VAR_KEYWORD and p.name in named:
                var_kw = p.name
                named.update(named.pop(p.name))
        pixel_values = named.get("pixel_values", None)
        if not torch.is_tensor(pixel_values) or pixel_values.ndim < 2:
            return original(*args, **kwargs)
        grid = named.get(_GRID_KWARG, None)
        n = int(grid.shape[0]) if torch.is_tensor(grid) else int(pixel_values.shape[0])
        if n < 2:
            return original(*args, **kwargs)
        extras = {}
        for key, value in named.items():
            if key in ("pixel_values", _GRID_KWARG) or not torch.is_tensor(value):
                continue
            if grid is None and key in _PER_IMAGE_KWARGS and value.ndim >= 1 and value.shape[0] == n:
                extras[key] = value
                continue
            # Any other tensor argument: no way to know how it splits by image.
            return original(*args, **kwargs)
        if _stochastic_vision(module):
            return original(*args, **kwargs)

        DEDUP_STATS["calls"] += 1
        DEDUP_STATS["images_in"] += n
        if grid is None:
            reps, inverse = _groups(pixel_values.reshape(n, -1), list(extras.values()))
            rows_u = None
        else:
            if grid.ndim != 2:
                return original(*args, **kwargs)
            rows = grid.prod(-1).tolist()
            if sum(rows) != pixel_values.shape[0]:
                return original(*args, **kwargs)
            starts = [0]
            for r in rows:
                starts.append(starts[-1] + r)
            grid_list = grid.tolist()
            reps, inverse = [], [None] * n
            by_shape = {}
            for i in range(n):
                by_shape.setdefault(tuple(grid_list[i]), []).append(i)
            local = {}
            for shape, idx in by_shape.items():
                if len(idx) == 1:
                    local[idx[0]] = idx[0]
                    continue
                seg = torch.stack([pixel_values[starts[i] : starts[i + 1]] for i in idx]).reshape(len(idx), -1)
                g_reps, g_inv = _groups(seg, [])
                for k, i in enumerate(idx):
                    local[i] = idx[g_reps[g_inv[k]]]
            slot = {}
            for i in range(n):
                j = local[i]
                if j not in slot:
                    slot[j] = len(reps)
                    reps.append(j)
                inverse[i] = slot[j]
            rows_u = [rows[j] for j in reps]
        DEDUP_STATS["images_encoded"] += len(reps)
        if len(reps) == n:
            return original(*args, **kwargs)

        if len(reps) == 1:
            # Never encode batch 1: compiled towers specialise it (Gemma-3 SigLIP 4% off on B200).
            reps = reps * 2
            DEDUP_STATS["images_encoded"] += 1
            if rows_u is not None:
                rows_u = rows_u * 2
        reps_t = torch.tensor(reps, dtype = torch.long, device = pixel_values.device)
        new = dict(bound.arguments)
        if var_kw is not None:
            new[var_kw] = dict(new[var_kw])
        def _set(key, value):
            if key in new:
                new[key] = value
            else:
                new[var_kw][key] = value
        if grid is None:
            _set("pixel_values", pixel_values.index_select(0, reps_t))
            for key, value in extras.items():
                _set(key, value.index_select(0, reps_t.to(value.device)))
        else:
            _set("pixel_values", torch.cat([pixel_values[starts[j] : starts[j + 1]] for j in reps]))
            _set(_GRID_KWARG, grid.index_select(0, reps_t.to(grid.device)))
        bound.arguments.clear()
        bound.arguments.update(new)
        out = original(*bound.args, **bound.kwargs)
        try:
            if grid is None:
                mapped = _map_rows(out, inverse, len(reps))
            else:
                mapped = _map_grid(out, inverse, rows_u)
        except _Unmappable as e:
            _fallback(module, f"unmappable output: {e}")
            return original(*args, **kwargs)
        if not module.__dict__.get(_VERIFIED, False) and os.environ.get("UNSLOTH_GRPO_IMAGE_DEDUP_VERIFY", "1") != "0":
            # First use per module: self-check the mapping against a full encode.
            with torch.no_grad():
                full = original(*args, **kwargs)
            reason = _outputs_mismatch(mapped, full, inverse)
            if reason is not None:
                _fallback(module, f"self-check mismatch: {reason}")
                del mapped
                return original(*args, **kwargs)
            module.__dict__[_VERIFIED] = True
            DEDUP_STATS["verified"] += 1
        DEDUP_STATS["deduped_calls"] += 1
        return mapped

    get_image_features._unsloth_image_dedup_original = original
    return get_image_features


def _image_modules(model):
    """Innermost ``get_image_features`` owners (wrapping outer + inner would dedup twice)."""
    found = []
    for name, module in model.named_modules():
        if callable(getattr(type(module), "get_image_features", None)):
            found.append((name, module))
    names = [n for n, _ in found]
    inner = []
    for name, module in found:
        prefix = name + "." if name else ""
        if not any(other != name and other.startswith(prefix) for other in names):
            inner.append(module)
    return inner


def install_image_dedup(model):
    """Wrap the innermost ``get_image_features`` of ``model`` once. Returns the number wrapped."""
    if model is None or not isinstance(model, torch.nn.Module) or not image_dedup_enabled() or _collective(model):
        return 0
    if model.__dict__.get(_SCANNED, False):
        return 0
    model.__dict__[_SCANNED] = True
    count = 0
    for module in _image_modules(model):
        if module.__dict__.get(_INSTALLED, False):
            continue
        original = module.get_image_features
        module.get_image_features = _make_wrapper(module, original)
        module.__dict__[_INSTALLED] = True
        count += 1
    return count
