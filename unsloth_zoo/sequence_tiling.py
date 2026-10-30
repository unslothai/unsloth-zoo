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

"""Sequence tiling for token-local modules: each row tile runs under torch's non-reentrant checkpoint."""

import functools
import os
import sys
import types

import torch

from torch.compiler import is_compiling as _is_compiling
from torch._higher_order_ops.wrap import tag_activation_checkpoint as _tag_activation_checkpoint

__all__ = [
    "tiled_call",
]

_CONTRACTS = ("rows", "rows_flat", "sum", "none")


# Unsloth's checkpoint shims force the reentrant path; tiles need torch's genuine non-reentrant one.
_IMPORT_TIME_CHECKPOINT = torch.utils.checkpoint.checkpoint


def _torch_checkpoint_fn():
    ck = torch.utils.checkpoint
    return getattr(ck, "_unsloth_pristine_checkpoint", None) or _IMPORT_TIME_CHECKPOINT
pass


def _owned(fn, x, *args):
    """Checkpointed tile body on its own copy of the rows: a kernel writing dX into its input in backward
    (fused LoRA MLP, inplace=True) would otherwise corrupt the input every other tile's checkpoint saved."""
    return fn(x.clone(), *args)
pass


def _checkpoint(fn, args, preserve_rng_state):
    body = functools.partial(_owned, fn)
    if _is_compiling():
        # Dynamo matches checkpoint by identity, which a shim may have replaced: call its higher order op directly.
        return _tag_activation_checkpoint(
            body, *args, use_reentrant = False, preserve_rng_state = preserve_rng_state,
        )
    return _torch_checkpoint_fn()(
        functools.partial(_replay_tile, body, _tile_gc_flag()), *args,
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
    # Both runs must read "not in a checkpoint recompute" (same saved tensors); pinning in every tile replay instead
    # raised the eager OLMoE 8k peak by 2.5 GiB on G4.
    return None if _gc_tls() is None else False
pass


def _replay_tile(fn, gc_flag, *args):
    """One eager tile whose forward and recompute read the same in_gradient_checkpoint_recompute() value: Unsloth's
    MoE kernels save a different tensor count per value, which raised CheckpointError (OLMoE: 29 vs 27)."""
    tls = _gc_tls() if gc_flag is not None else None
    previous = getattr(tls, "active", False) if tls is not None else None
    if tls is not None:
        tls.active = gc_flag
    try:
        if _is_dynamic_body(fn):
            return fn(*args)
        return _dynamic_inner_compiles(fn, *args)
    except BaseException as exc:
        # Checkpoint ends a recompute early by raising; a helper that re-raises it makes the traceback a reference cycle
        # that keeps every tile's recomputed intermediates alive until the cycle collector runs. Drop it on the raise
        # itself: a bare raise on Python 3.9 / 3.10 re-attaches the traceback stored in the thread state.
        if _STOP_RECOMPUTATION is not None and isinstance(exc, _STOP_RECOMPUTATION):
            raise exc.with_traceback(None)
        raise
    finally:
        if tls is not None:
            tls.active = previous
pass

_STOP_RECOMPUTATION = getattr(torch.utils.checkpoint, "_StopRecomputationError", None)


def _is_dynamic_body(fn):
    while isinstance(fn, functools.partial):
        if getattr(fn, "_unsloth_dynamic_body", False):
            return True
        fn = fn.args[0] if fn.func is _owned and fn.args else fn.func
    return bool(getattr(fn, "_unsloth_dynamic_body", False))
pass


def _dynamic_inner_compiles(fn, *args):
    """Compile code inside a tile dynamic from its first call: a tile forwarded on a static graph and recomputed
    on the later dynamic one saves different tensors ("A different number of tensors was saved")."""
    try:
        patch = torch._dynamo.config.patch(assume_static_by_default = False)
    except Exception:
        return fn(*args)
    with patch:
        return fn(*args)
pass


class _WriteRows(torch.autograd.Function):
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


def _sym_min(a, b):
    # torch.sym_min on two plain ints (static shapes) is not traceable on torch 2.6.
    return min(a, b) if isinstance(a, int) and isinstance(b, int) else torch.sym_min(a, b)


def _sym_max(a, b):
    return max(a, b) if isinstance(a, int) and isinstance(b, int) else torch.sym_max(a, b)


def _tile_bounds(rows, n_tiles, compiling):
    """(start, length) per tile, ceil(rows / n_tiles) rows each. Depending on rows only through that ceil keeps one
    compiled graph per tile-count bucket (tensor_split guards on rows % n_tiles). Compiled callers keep 0-row tiles:
    dropping them would guard on the length."""
    chunk = (rows + n_tiles - 1) // n_tiles
    bounds = []
    for i in range(n_tiles):
        start = i * chunk
        if compiling:
            length = _sym_max(0, _sym_min(chunk, rows - start))
            start = _sym_min(start, rows)
        else:
            length = min(chunk, rows - start)
            if length <= 0:
                break
        bounds.append((start, length))
    return bounds
pass


def tiled_call(fn, x, *token_args, n_tiles, needs_rng = True, output_contract = None):
    """Run token-local `fn(x, *token_args)` over `n_tiles` row tiles of `x`, each under non-reentrant checkpoint.

    x: [B, S, H], [1, N, H] or [N, H]; 3-D gives `fn` [1, t, H] tiles, 2-D gives [t, H]. token_args share x's row
    boundaries. output_contract: per output, "rows", "rows_flat", "sum" or "none"; None infers it."""
    n_tiles = int(n_tiles)
    if n_tiles <= 1:
        return fn(x, *token_args)
    compiling = _is_compiling()
    if compiling:
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
    if not compiling and n_tiles > rows:
        n_tiles = rows
        if n_tiles <= 1:
            return fn(x, *token_args)
    bounds = _tile_bounds(rows, n_tiles, compiling)
    n_lead = len(lead)
    token_flat = [
        None if a is None else a.reshape(flat.shape[:-1] + a.shape[n_lead:])
        for a in token_args
    ]

    contract = _normalise_contract(output_contract)
    grad = torch.is_grad_enabled()
    is_tuple = None
    pieces = None
    main_buf = None
    tile_ndim = flat.dim()

    for i, (start, length) in enumerate(bounds):
        x_tile = flat.narrow(dim, start, length)
        args = (x_tile,) + tuple(None if a is None else a.narrow(dim, start, length) for a in token_flat)
        if grad:
            out = _checkpoint(fn, args, needs_rng)
        else:
            out = fn(*args)
        if is_tuple is None:
            is_tuple = isinstance(out, (tuple, list))
            outs0 = tuple(out) if is_tuple else (out,)
            if contract is None:
                contract = tuple(_infer_contract(o, tile_ndim, j) for j, o in enumerate(outs0))
            elif not is_tuple:
                contract = contract[:1]
            if len(contract) != len(outs0):
                raise ValueError(f"Unsloth: tiled fn returned {len(outs0)} outputs, output_contract has {len(contract)}.")
            pieces = [[] if c in ("rows", "rows_flat") else None for c in contract]
        outs = tuple(out) if is_tuple else (out,)
        if isinstance(out, (tuple, list)) != is_tuple or len(outs) != len(contract):
            raise ValueError(f"Unsloth: tile {i} returned {len(outs)} outputs, the first tile {len(contract)}.")
        for j, (c, o) in enumerate(zip(contract, outs)):
            if c in ("rows", "rows_flat"):
                got_rows = o.shape[dim] if c == "rows" else o.shape[0]
                if compiling:
                    torch._check(got_rows == length)  # no message: Dynamo on torch 2.6 cannot trace one
                elif got_rows != length:
                    raise ValueError(
                        f"Unsloth: tiled output {j} of tile {i} has {got_rows} rows, the tile has {length}."
                    )
            if c == "none":
                if o is not None:
                    raise ValueError(f"Unsloth: tiled output {j} has contract 'none' but is not None.")
            elif c == "sum":
                pieces[j] = o if pieces[j] is None else pieces[j] + o
            elif j == 0 and c == "rows" and not compiling:
                if main_buf is None:
                    main_buf = o.new_empty(flat.shape[:-1] + o.shape[tile_ndim - 1:])
                main_buf = _WriteRows.apply(main_buf, o, dim, start)
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
    """"unroll", "compile_tile" or "break" (UNSLOTH_TILED_COMPILE forces one). Unrolling costs ~1.1 s compile per
    tile on G4, so Unsloth's fullgraph=False blocks leave the graph; a user's compile may be fullgraph=True, so it unrolls."""
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


_COMPILED_TILES = {}


def _compiled_tile_for(key):
    """One compiled tile frame per (class, inner fn): a shared frame guards on the class and hits the recompile limit."""
    compiled = _COMPILED_TILES.get(key)
    if compiled is None:
        code = _call_tile.__code__.replace(co_name = f"_call_tile_{len(_COMPILED_TILES)}")
        frame = types.FunctionType(code, _call_tile.__globals__, code.co_name)
        compiled = _COMPILED_TILES[key] = torch.compile(frame, dynamic = True)
    return compiled
pass


def _compiled_tile_fn(fn):
    if isinstance(fn, functools.partial):
        func, bound, keywords = fn.func, fn.args, fn.keywords or {}
        owner = type(bound[0]) if bound else None
        compiled = _compiled_tile_for((getattr(func, "__code__", func), owner))
        body = lambda *tile_args: compiled(func, *bound, *tile_args, **keywords)
    else:
        compiled = _compiled_tile_for((getattr(fn, "__code__", fn), None))
        body = lambda *tile_args: compiled(fn, *tile_args)
    body._unsloth_dynamic_body = True
    return body
pass


def _tiled_call_eager_impl(fn, x, *token_args, n_tiles, needs_rng, output_contract, compile_tile):
    if compile_tile:
        fn = _compiled_tile_fn(fn)
    return tiled_call(fn, x, *token_args, n_tiles = n_tiles, needs_rng = needs_rng, output_contract = output_contract)
pass


_tiled_call_eager = torch.compiler.disable(_tiled_call_eager_impl, recursive = True)
