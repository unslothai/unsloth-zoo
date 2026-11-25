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
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ROCm: flex_attention compiled with a symbolic sequence length returned inf / NaN gradients on gfx1151 (torch 2.11,
ROCm 7.13 / 7.14: tiny Gemma-3 trained at 24 tokens after a 32 token step), while a static compile per length and
eager flex were both correct. A flex call whose backward will run on a ROCm GPU is therefore compiled with static
shapes: a static compile of its own entry point outside a graph, its Q / K / V sequence lengths specialized inside an
enclosing graph. No-grad calls (inference, decode with a growing KV cache) keep each call site's own compile, so
decode does not recompile per length. UNSLOTH_ROCM_FLEX_STATIC=0 turns it off.

Inside an enclosing graph only the sequence dimensions (dim -2 of Q / K / V) are specialized: the batch dimension
stays symbolic, so a block compiled with maybe_mark_dynamic on its batch keeps one graph across batch sizes instead
of one per (batch, length) pair, which would also use up the block's recompile budget. ROCM_FLEX_STATIC_STATS counts
what happened (specialized, left dynamic, failed) and a failure warns once instead of silently keeping the symbolic
kernel."""

__all__ = [
    "ROCM_FLEX_STATIC_STATS",
    "rocm_flex_static_enabled",
    "needs_static_flex",
    "specialize_flex_shapes",
    "compile_static_flex",
    "rocm_static_flex",
]

import inspect
import logging
import os
import warnings
import torch

try:
    from torch._dynamo.comptime import comptime as _comptime
except Exception:
    _comptime = None

# Every new length is one more static graph; torch's default allows 8 per code object.
_RECOMPILE_FLOOR = 1024

# Engagement counters bumped at trace time (comptime) or eagerly, never in a graph: specialized
# (Q/K/V sizes made concrete), left_dynamic (size derives from a mark_dynamic input), failed (torch
# internals moved or comptime missing), static_calls (eager calls routed to the static compile).
ROCM_FLEX_STATIC_STATS = {"specialized": 0, "left_dynamic": 0, "failed": 0, "static_calls": 0}
_WARNED = set()
_logger = logging.getLogger(__name__)


def _warn_once(kind, message):
    # Also at trace time (comptime): -W error::RuntimeWarning must not fail the trace; log it.
    if kind in _WARNED:
        return
    _WARNED.add(kind)
    try:
        warnings.warn(f"Unsloth: {message}", RuntimeWarning, stacklevel = 3)
    except Exception:
        _logger.warning("Unsloth: %s", message)


def rocm_flex_static_enabled():
    return bool(getattr(torch.version, "hip", None)) and os.environ.get("UNSLOTH_ROCM_FLEX_STATIC", "1") != "0"


def _affected_device(device):
    # ROCm GPU tensors report device type "cuda"; CPU tensors on a ROCm build keep the stock path.
    return device.type == "cuda"


def needs_static_flex(query, key, value):
    """A flex call on a ROCm GPU whose backward will run."""
    if not isinstance(query, torch.Tensor) or not _affected_device(query.device):
        return False
    if not torch.is_grad_enabled():
        return False
    return any(isinstance(t, torch.Tensor) and t.requires_grad for t in (query, key, value))


def _strict_symbols(ctx):
    """Shape symbols of the enclosing graph's inputs marked strictly dynamic (torch._dynamo.mark_dynamic):
    specializing one raises ConstraintViolationError, so those graphs are left as they are."""
    strict = set()
    tx = ctx._ComptimeContext__tx
    for tracked in tx.output.tracked_fakes:
        constraints = getattr(tracked.symbolic_context, "constraint_sizes", None) or ()
        fake = tracked.fake
        if not isinstance(fake, torch.Tensor):
            continue
        for dim, constraint in enumerate(constraints):
            # mark_dynamic: RelaxedUnspecConstraint / StrictMinMaxConstraint with warn_only False; maybe_mark_dynamic: None.
            if constraint is not None and not getattr(constraint, "warn_only", True) and dim < fake.dim():
                size = fake.size(dim)
                if isinstance(size, torch.SymInt):
                    strict |= size.node.expr.free_symbols
    return strict


def _sequence_sizes(ctx):
    """Symbolic sequence sizes (dim -2) of the traced query / key / value. flex_attention takes (B, H, L, E): batch,
    heads and head dim are left as the enclosing graph has them."""
    sizes = []
    for name in ("query", "key", "value"):
        fake = ctx.get_local(name).as_fake()
        if isinstance(fake, torch.Tensor) and fake.dim() >= 2:
            size = fake.size(-2)
            if isinstance(size, torch.SymInt):
                sizes.append(size)
    return sizes


def _force_static_qkv(ctx):
    # Runs at trace time (comptime), not inside the graph.
    try:
        sizes = _sequence_sizes(ctx)
        if not sizes:
            return
        symbols = set().union(*(size.node.expr.free_symbols for size in sizes))
        if symbols & _strict_symbols(ctx):
            ROCM_FLEX_STATIC_STATS["left_dynamic"] += 1
            _warn_once(
                "left_dynamic",
                "a flex_attention sequence length inside a compiled graph derives from a torch._dynamo.mark_dynamic "
                "input, so it stays symbolic. On ROCm (gfx1151) symbolic-length flex gradients were inf / NaN.",
            )
            return
        for size in sizes:
            int(size)  # guard_int: specializes the symbol, as torch._dynamo.mark_static does inside a trace
        ROCM_FLEX_STATIC_STATS["specialized"] += 1
    except Exception as error:
        # Unknown torch internals: the enclosing graph keeps its shape policy, but say so.
        ROCM_FLEX_STATIC_STATS["failed"] += 1
        _warn_once(
            "failed",
            f"could not make flex_attention sequence lengths static inside a compiled graph on ROCm "
            f"(torch {torch.__version__}: {type(error).__name__}: {error}). Symbolic-length flex gradients were "
            f"inf / NaN on gfx1151.",
        )


def specialize_flex_shapes(query, key, value):
    """Inside a Dynamo trace: make the Q / K / V sequence sizes concrete (one graph per length; batch stays symbolic),
    unless they derive from a strictly dynamic input. Outside a trace this does nothing."""
    if _comptime is not None:
        _comptime(_force_static_qkv)


def _note_static_call():
    ROCM_FLEX_STATIC_STATS["static_calls"] += 1


def _per_call_recompile_limit():
    # torch 2.12+: torch.compile(recompile_limit = N) limits that compile alone.
    try:
        return "recompile_limit" in inspect.signature(torch.compile).parameters
    except (TypeError, ValueError):
        return False


_PER_CALL_RECOMPILE_LIMIT = _per_call_recompile_limit()


def _raise_recompile_budget(per_frame = True):
    try:
        from unsloth_zoo.torch_config import set_torch_config
    except Exception:
        set_torch_config = lambda config, name, value: setattr(config, name, value)
    config = torch._dynamo.config
    groups = [("accumulated_recompile_limit", "accumulated_cache_size_limit")]
    if per_frame:
        groups.insert(0, ("recompile_limit", "cache_size_limit"))
    for names in groups:
        for name in names:
            if hasattr(config, name):
                try:
                    if getattr(config, name) < _RECOMPILE_FLOOR:
                        set_torch_config(config, name, _RECOMPILE_FLOOR)
                except Exception:
                    pass
                break


def _configured_recompile_limit():
    config = torch._dynamo.config
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(config, name):
            try:
                return int(getattr(config, name))
            except Exception:
                return 0
    return 0


def compile_static_flex(entry, training = False):
    """torch.compile(entry, dynamic = False) with room for one graph per length. `entry` should be a function of the
    call site's own, so its Dynamo cache is not shared with another compile of flex_attention."""
    options = {}
    if _PER_CALL_RECOMPILE_LIMIT:
        # This compile's recompile_limit is max(configured, floor); process-wide limit is untouched,
        # and the accumulated per-code-object cap, counted across compiles, still needs room.
        options["recompile_limit"] = max(_configured_recompile_limit(), _RECOMPILE_FLOOR)
        _raise_recompile_budget(per_frame = False)
    else:
        _raise_recompile_budget()
    # transformers' torch 2.6 training workaround (pytorch/pytorch#146260), kept.
    if training and torch.__version__.startswith("2.6.0"):
        options["mode"] = "max-autotune-no-cudagraphs"
    return torch.compile(entry, dynamic = False, **options)


def _check_comptime():
    """Eager, at install time: without torch._dynamo.comptime the in-graph specialization cannot run at all."""
    if _comptime is None:
        ROCM_FLEX_STATIC_STATS["failed"] += 1
        _warn_once(
            "failed",
            f"torch._dynamo.comptime is unavailable on torch {torch.__version__}, so flex_attention sequence lengths "
            f"inside compiled graphs stay symbolic on ROCm. Symbolic-length flex gradients were inf / NaN on gfx1151.",
        )


def _without_dynamic_marks(tensor):
    # A caller's mark_dynamic / maybe_mark_dynamic on an input wins over dynamic = False; a view carries no marks.
    if isinstance(tensor, torch.Tensor) and (
        getattr(tensor, "_dynamo_dynamic_indices", None) or getattr(tensor, "_dynamo_weak_dynamic_indices", None)
    ):
        return tensor.view_as(tensor)
    return tensor


def rocm_static_flex(auto, entry):
    """Wrap a call site's compiled flex (`auto`). `entry(query, key, value, *args, **kwargs)` calls the uncompiled flex.
    Off ROCm, or with UNSLOTH_ROCM_FLEX_STATIC=0, `auto` is returned unchanged."""
    if not rocm_flex_static_enabled() or getattr(auto, "_unsloth_rocm_static", False):
        return auto
    _check_comptime()
    static = []

    def flex_attention(query, key, value, *args, **kwargs):
        if needs_static_flex(query, key, value):
            if torch.compiler.is_compiling():
                specialize_flex_shapes(query, key, value)
                return auto(query, key, value, *args, **kwargs)
            if not static:
                static.append(compile_static_flex(entry, training = True))
            query, key, value = map(_without_dynamic_marks, (query, key, value))
            _note_static_call()
            return static[0](query, key, value, *args, **kwargs)
        return auto(query, key, value, *args, **kwargs)

    flex_attention._unsloth_rocm_static = True
    flex_attention.__wrapped__ = auto
    return flex_attention
