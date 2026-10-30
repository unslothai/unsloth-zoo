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
Sequence tiling for token-local modules (MLPs, token-local MoE blocks, projectors).

Each tile runs under torch's non-reentrant checkpoint, so parameters are used through normal autograd: the engine sums
every tile's contribution before AccumulateGrad runs (DDP / FSDP2 hooks fire once per backward), layers whose input does
not require grad still get parameter gradients, and every output of every tile is differentiable. Only one tile's
intermediate activations live at once, in forward and in the backward recompute.
"""

import functools
import os
import sys

import torch

__all__ = [
    "tiled_call",
    "bucket_tiles",
]

try:
    from torch.compiler import is_compiling as _is_compiling
except Exception:
    try: from torch._dynamo import is_compiling as _is_compiling
    except Exception: _is_compiling = lambda: False
pass

try:
    from torch._higher_order_ops.wrap import tag_activation_checkpoint as _tag_activation_checkpoint
except Exception:
    _tag_activation_checkpoint = None
pass

_CONTRACTS = ("rows", "rows_flat", "sum", "none")


def _torch_checkpoint_fn():
    # Unsloth replaces torch.utils.checkpoint.checkpoint with shims that force the reentrant path; tiles need the
    # genuine non-reentrant one.
    ck = torch.utils.checkpoint
    fn = getattr(ck, "_unsloth_pristine_checkpoint", None)
    if fn is not None:
        return fn
    fn = ck.checkpoint
    if getattr(fn, "__module__", None) == "torch.utils.checkpoint":
        return fn
    return _IMPORT_TIME_CHECKPOINT
pass

_IMPORT_TIME_CHECKPOINT = torch.utils.checkpoint.checkpoint
if getattr(_IMPORT_TIME_CHECKPOINT, "__module__", None) != "torch.utils.checkpoint":
    _IMPORT_TIME_CHECKPOINT = getattr(torch.utils.checkpoint, "_unsloth_pristine_checkpoint", None)
    if _IMPORT_TIME_CHECKPOINT is None:
        def _IMPORT_TIME_CHECKPOINT(function, *args, preserve_rng_state = True, use_reentrant = False):
            gen = torch.utils.checkpoint._checkpoint_without_reentrant_generator(
                function, preserve_rng_state, torch.utils.checkpoint.noop_context_fn,
                torch.utils.checkpoint._DEFAULT_DETERMINISM_MODE, False, *args,
            )
            next(gen)
            ret = function(*args)
            try: next(gen)
            except StopIteration: pass
            return ret
pass


def _checkpoint(fn, args, preserve_rng_state):
    if _is_compiling():
        # Dynamo recognises checkpoint only by identity with the current torch.utils.checkpoint.checkpoint, which a
        # shim may have replaced: call the higher order op it lowers to directly.
        if _tag_activation_checkpoint is not None:
            return _tag_activation_checkpoint(
                fn, *args, use_reentrant = False, preserve_rng_state = preserve_rng_state,
            )
        return torch.utils.checkpoint.checkpoint(
            fn, *args, use_reentrant = False, preserve_rng_state = preserve_rng_state,
        )
    return _torch_checkpoint_fn()(
        functools.partial(_replay_tile, fn, _tile_gc_flag()), *args,
        use_reentrant = False, preserve_rng_state = preserve_rng_state,
    )
pass


def _gc_tls():
    try:
        from unsloth_zoo.gradient_checkpointing import _GC_RECOMPUTE_TLS
        return _GC_RECOMPUTE_TLS
    except Exception:
        return None
pass


def _tile_gc_flag():
    # Inside a tile, both runs read "not in a checkpoint recompute": a tile's own recompute is immediately followed
    # by its backward, so the MoE policy's rebuild path costs one tile, while pinning in every tile replay raised the
    # eager OLMoE 8k peak by 2.5 GiB on G4. None when the flag is unavailable (nothing to pin).
    return None if _gc_tls() is None else False
pass


def _replay_tile(fn, gc_flag, *args):
    """One eager tile, with the state that picks code paths inside it the same for the forward and the recompute.

    Non-reentrant checkpoint requires the recompute to save the same tensors as the forward. Under Unsloth's outer
    gradient checkpointing, a tile's forward runs inside the outer recompute (in_gradient_checkpoint_recompute() is
    True) but its own recompute runs later in backward (False). Unsloth's MoE kernels read that flag to pin or rebuild
    the dequantized expert stack, which saves a different number of tensors, so backward raised CheckpointError
    (OLMoE: 29 saved in forward, 27 in recompute). Both runs see the same value (see _tile_gc_flag).
    """
    tls = _gc_tls() if gc_flag is not None else None
    previous = getattr(tls, "active", False) if tls is not None else None
    if tls is not None:
        tls.active = gc_flag
    try:
        if getattr(fn, "_unsloth_dynamic_body", False):
            return fn(*args)
        return _dynamic_inner_compiles(fn, *args)
    finally:
        if tls is not None:
            tls.active = previous
pass


def _dynamic_inner_compiles(fn, *args):
    """Run one eager tile with torch.compile'd code inside it compiling dynamic shapes from the first call.

    Non-reentrant checkpoint matches the tensors a tile saved in forward with the ones its recompute saves. A
    compiled function inside the tile (Unsloth's MoE experts, whose per-expert token counts change every tile) first
    compiles a static graph, then, on the next shape, a dynamic one that Dynamo tries first from then on. A tile
    that ran on the static graph is then recomputed on the dynamic graph, which saves different tensors, and
    backward fails with "A different number of tensors was saved". Compiling dynamic from the start means a later
    graph never takes over shapes an earlier one served. Functions compiled with dynamic=False keep one static graph
    per shape, which is consistent too.
    """
    try:
        patch = torch._dynamo.config.patch(assume_static_by_default = False)
    except Exception:
        return fn(*args)
    with patch:
        return fn(*args)
pass


class _WriteRows(torch.autograd.Function):
    """buf[start : start + len] along `dim` = tile, in place. Backward hands each tile a narrow view of the output
    gradient (no copies); the region a later tile overwrote is never read by an earlier node."""
    @staticmethod
    def forward(ctx, buf, tile, dim, start):
        length = tile.shape[dim]
        ctx.dim, ctx.start, ctx.length = dim, start, length
        buf.narrow(dim, start, length).copy_(tile)
        ctx.mark_dirty(buf)
        return buf

    @staticmethod
    def backward(ctx, grad):
        return grad, grad.narrow(ctx.dim, ctx.start, ctx.length), None, None
pass


def _normalise_contract(output_contract):
    if output_contract is None:
        return None
    if isinstance(output_contract, str):
        output_contract = (output_contract,)
    output_contract = tuple(output_contract)
    for c in output_contract:
        if c not in _CONTRACTS:
            raise ValueError(f"Unsloth: unknown tiled output contract {c!r}; expected one of {_CONTRACTS}.")
    return output_contract
pass


def _infer_contract(out, tile_ndim, index):
    if out is None:
        return "none"
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"Unsloth: tiled output {index} is a {type(out).__name__}; pass output_contract explicitly.")
    if out.dim() == tile_ndim:
        return "rows"
    if out.dim() == 2 and tile_ndim == 3:
        return "rows_flat"
    raise ValueError(
        f"Unsloth: tiled output {index} has shape {tuple(out.shape)}; its token axis is ambiguous (a scalar aux loss "
        f"needs output_contract 'sum', and only if it is additive over tokens)."
    )
pass


def tiled_call(fn, x, *token_args, n_tiles, needs_rng = True, output_contract = None):
    """
    Run token-local `fn(x, *token_args)` over `n_tiles` row tiles of `x`, each under non-reentrant checkpoint.

    x: [B, S, H], [1, N, H] or [N, H]. Rows are the flattened leading dims. A 3-D x gives `fn` tiles of shape
    [1, t, H] (MoE blocks unpack three dims); a 2-D x gives [t, H].
    token_args: tensors whose leading dims equal x.shape[:-1] (or None), split with the same row boundaries.
    n_tiles: a Python int (see bucket_tiles); <= 1 calls fn directly. Clamped to the row count.
    needs_rng: preserve and replay the RNG state per tile (dropout inside fn). False skips the RNG snapshot.
    output_contract: per output, "rows" (token axis, width may change, reshaped back to x's leading dims),
    "rows_flat" ([t, E] -> [rows, E], e.g. router logits), "sum" (additive scalars), "none". None infers it.
    """
    n_tiles = int(n_tiles)
    if n_tiles <= 1:
        return fn(x, *token_args)
    if _is_compiling():
        mode = _outside_graph_mode()
        if mode != "unroll":
            return _tiled_call_eager(
                fn, x, *token_args, n_tiles = n_tiles, needs_rng = needs_rng, output_contract = output_contract,
                compile_tile = mode == "compile_tile",
            )
    lead = x.shape[:-1]
    if x.dim() == 2:
        flat, dim = x, 0
    elif x.dim() >= 3:
        flat, dim = x.reshape(1, -1, x.shape[-1]), 1
    else:
        return fn(x, *token_args)
    rows = flat.shape[dim]
    if n_tiles > rows:
        n_tiles = max(1, rows)
        if n_tiles <= 1:
            return fn(x, *token_args)
    x_tiles = torch.tensor_split(flat, n_tiles, dim = dim)
    n_lead = len(lead)
    arg_tiles = []
    for a in token_args:
        if a is None:
            arg_tiles.append(None)
            continue
        a = a.reshape(flat.shape[:-1] + a.shape[n_lead:])
        arg_tiles.append(torch.tensor_split(a, n_tiles, dim = dim))
    pass

    contract = _normalise_contract(output_contract)
    grad = torch.is_grad_enabled()
    compiling = _is_compiling()
    is_tuple = None
    pieces = None
    main_buf = None
    start = 0
    tile_ndim = flat.dim()

    for i in range(n_tiles):
        args = (x_tiles[i],) + tuple(None if t is None else t[i] for t in arg_tiles)
        if grad:
            out = _checkpoint(fn, args, needs_rng)
        else:
            out = fn(*args)
        if is_tuple is None:
            is_tuple = isinstance(out, (tuple, list))
            outs0 = tuple(out) if is_tuple else (out,)
            if contract is None:
                contract = tuple(_infer_contract(o, tile_ndim, j) for j, o in enumerate(outs0))
            elif len(contract) == 1 and len(outs0) > 1:
                raise ValueError(f"Unsloth: tiled fn returned {len(outs0)} outputs but output_contract has 1.")
            if len(contract) != len(outs0):
                raise ValueError(f"Unsloth: tiled fn returned {len(outs0)} outputs, output_contract has {len(contract)}.")
            pieces = [[] if c in ("rows", "rows_flat") else None for c in contract]
        outs = tuple(out) if is_tuple else (out,)
        if isinstance(out, (tuple, list)) != is_tuple or len(outs) != len(contract):
            raise ValueError(f"Unsloth: tile {i} returned {len(outs)} outputs, the first tile {len(contract)}.")
        for j, (c, o) in enumerate(zip(contract, outs)):
            if c in ("rows", "rows_flat") and not compiling:
                want = x_tiles[i].shape[dim]
                got_rows = o.shape[dim] if c == "rows" else o.shape[0]
                if got_rows != want:
                    raise ValueError(f"Unsloth: tiled output {j} of tile {i} has {got_rows} rows, the tile has {want}.")
            if c == "none":
                if o is not None:
                    raise ValueError(f"Unsloth: tiled output {j} has contract 'none' but is not None.")
            elif c == "sum":
                pieces[j] = o if pieces[j] is None else pieces[j] + o
            elif j == 0 and c == "rows" and not compiling:
                # One preallocated buffer, as the old TiledMLP: tile outputs are freed as they are written.
                if main_buf is None:
                    main_buf = o.new_empty(flat.shape[:-1] + o.shape[tile_ndim - 1:])
                main_buf = _WriteRows.apply(main_buf, o, dim, start)
                start = start + o.shape[dim]
            elif c == "rows":
                pieces[j].append(o)
            else:
                pieces[j].append(o)
        pass
    pass

    results = []
    for j, c in enumerate(contract):
        if c == "none":
            results.append(None)
        elif c == "sum":
            results.append(pieces[j])
        elif c == "rows":
            y = main_buf if (j == 0 and main_buf is not None) else torch.cat(pieces[j], dim = dim)
            results.append(y.reshape(tuple(lead) + tuple(y.shape[tile_ndim - 1:])))
        else:
            results.append(torch.cat(pieces[j], dim = 0))
    pass
    if is_tuple:
        return tuple(results)
    return results[0]
pass


_ENV_COMPILE = "UNSLOTH_TILED_COMPILE"


def _outside_graph_mode():
    """How a compiled caller runs the tile loop: "unroll" (one checkpoint per tile inside its graph), "compile_tile"
    (one graph break at tiled_call; an eager loop runs every tile through one shared compiled tile body) or "break"
    (the same loop with eager tiles).

    Unrolling costs compile time linear in the tile count (G4: about 1.1 s per tile, 440 s for a cold step at 192k),
    so inside Unsloth's per-layer compiled blocks (fullgraph=False, where a break is legal) the loop leaves the graph
    and the tile body is compiled once with dynamic rows, shared by every tile and layer. A user's own
    torch.compile may be fullgraph=True, where a break is an error, so it keeps the unrolled loop.
    UNSLOTH_TILED_COMPILE = unroll | compile_tile | break forces a mode.
    """
    mode = os.environ.get(_ENV_COMPILE, "auto").strip().lower()
    if mode in ("unroll", "compile_tile", "break"):
        return mode
    block = sys.modules.get("unsloth.models._block_compile")
    flag = getattr(block, "_IN_COMPILED_REGION", None)
    return "compile_tile" if (bool(flag) and bool(flag[0])) else "unroll"
pass


def _call_tile(func, *args, **kwargs):
    return func(*args, **kwargs)
pass


_COMPILED_TILE = [None]


def _compiled_tile_fn(fn):
    """fn with its body run through one torch.compile'd frame shared by every tile and layer. A functools.partial
    is unpacked so its module argument is a graph input (no per-layer guard on the module's identity)."""
    if _COMPILED_TILE[0] is None:
        _COMPILED_TILE[0] = torch.compile(_call_tile, dynamic = True)
    compiled = _COMPILED_TILE[0]
    if isinstance(fn, functools.partial):
        func, bound, keywords = fn.func, fn.args, fn.keywords or {}
        body = lambda *tile_args: compiled(func, *bound, *tile_args, **keywords)
    else:
        body = lambda *tile_args: compiled(fn, *tile_args)
    # Already dynamic: the per-tile Dynamo config patch in _replay_tile is not needed (it costs ~20 us per call).
    body._unsloth_dynamic_body = True
    return body
pass


def _tiled_call_eager_impl(fn, x, *token_args, n_tiles, needs_rng, output_contract, compile_tile):
    if compile_tile:
        fn = _compiled_tile_fn(fn)
    return tiled_call(fn, x, *token_args, n_tiles = n_tiles, needs_rng = needs_rng, output_contract = output_contract)
pass


# Leaves the caller's graph; inside, the tile loop is eager and each tile runs under the eager checkpoint. Code
# compiled inside a tile compiles with dynamic rows from its first call (_dynamic_inner_compiles).
_tiled_call_eager = torch.compiler.disable(_tiled_call_eager_impl, recursive = True)


def bucket_tiles(rows, tile_rows, cap = 64):
    """Tile count bucket: 1 when rows <= tile_rows, else next_pow2(ceil(rows / tile_rows)), at most cap. Dynamo guards
    on the bucket only, so sequence lengths inside a bucket share one graph."""
    tile_rows = max(1, int(tile_rows))
    if rows <= tile_rows:
        return 1
    n = (rows + tile_rows - 1) // tile_rows
    p = 2
    while p < n and p < cap:
        p *= 2
    return int(min(p, max(1, int(cap))))
pass
