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

"""
Memory-efficient custom VJP for GatedDeltaNet (Qwen3.5).

Replaces the T-step Python loop in gated_delta_ops with an mx.custom_function
that recomputes states during backward instead of keeping all T intermediate
states in the autograd graph.

Usage:
    from gated_delta_vjp import patch_gated_delta
    patch_gated_delta()  # monkey-patches mlx_lm's gated_delta module
"""

from functools import lru_cache
from typing import NamedTuple, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn


def _gated_delta_step(q, k, v, g, beta, state, mask=None):
    """Single recurrent step (no @mx.compile — we need it differentiable)."""
    old_state = state
    if g.ndim == 2:
        decay = g[..., None, None]
    elif g.ndim == 3:
        decay = g[..., None, :]
    else:
        raise ValueError(f"Unsupported gating shape {g.shape}")

    state = state * decay
    kv_mem = (state * k[..., None, :]).sum(axis=-1)
    delta = (v - kv_mem) * beta[..., None]
    state = state + k[..., None, :] * delta[..., None]
    y = (state * q[..., None, :]).sum(axis=-1)

    if mask is not None:
        mask = mx.expand_dims(mask, axis=(1, 2, 3))
        state = mx.where(mask, state, old_state)

    return y.astype(q.dtype), state


def gated_delta_ops_efficient(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """Memory-efficient GDN forward+backward via custom VJP.

    Wraps the recurrence in mx.custom_function so backward recomputes states
    on the fly during BPTT instead of keeping T graph nodes alive.
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    # Chunk for checkpointed BPTT: each chunk's forward is recomputed during
    # backward. Memory: O(num_chunks * state_size) instead of O(T * state_size).
    CHUNK_SIZE = max(1, min(64, T))
    num_chunks = (T + CHUNK_SIZE - 1) // CHUNK_SIZE

    @mx.custom_function
    def _chunked_forward(q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk, state_in, mask_chunk):
        """Process one chunk of timesteps."""
        chunk_T = q_chunk.shape[1]
        _has_mask = mask_chunk is not None and mask_chunk.shape[-1] >= chunk_T
        ys = []
        s = state_in
        for t in range(chunk_T):
            m = mask_chunk[:, t:t+1].squeeze(1) if _has_mask else None
            y, s = _gated_delta_step(
                q_chunk[:, t], k_chunk[:, t], v_chunk[:, t],
                g_chunk[:, t], beta_chunk[:, t], s, m,
            )
            ys.append(y)
        return mx.stack(ys, axis=1), s

    @_chunked_forward.vjp
    def _chunked_vjp(primals, cotangents, outputs):
        q_c, k_c, v_c, g_c, beta_c, state_in, mask_chunk = primals
        dy, d_state_out = cotangents
        chunk_T = q_c.shape[1]
        _has_mask = mask_chunk is not None and mask_chunk.shape[-1] >= chunk_T

        # Recompute the chunk's RETURNED states (post-mask: where(mask,
        # state_new, state_prev), so masked steps give states[t+1]==states[t]).
        # Used only as each step's entry `state_prev`; the y-path backward needs
        # state_new, recomputed below.
        states = [state_in]
        s = state_in
        for t in range(chunk_T):
            m = mask_chunk[:, t:t+1].squeeze(1) if _has_mask else None
            _, s = _gated_delta_step(
                q_c[:, t], k_c[:, t], v_c[:, t],
                g_c[:, t], beta_c[:, t], s, m,
            )
            states.append(s)

        # BPTT: `d_state` is the cotangent w.r.t. the RETURNED state at the
        # current step (= input to step t+1). Starts at d_state_out, then
        # propagates through the recurrence + mask.
        # Per-step grads are collected in lists and stacked afterwards: each
        # t is produced exactly once, and mx `.at[:, t].add` scatter-add
        # reads the update tensor with a wrong batch stride (wrong grads for
        # every batch row past the first; verified against plain autodiff on
        # mlx 0.31).
        # Fixed upstream in ml-explore/mlx#3483, not yet in any release.
        d_q_steps = []
        d_k_steps = []
        d_v_steps = []
        d_g_steps = []
        d_beta_steps = []
        d_state = d_state_out

        for t in range(chunk_T - 1, -1, -1):
            state_prev = states[t]
            q_t = q_c[:, t]
            k_t = k_c[:, t]
            v_t = v_c[:, t]
            g_t = g_c[:, t]
            beta_t = beta_c[:, t]
            dy_t = dy[:, t]

            # Cotangent flowing into step t's output (state_returned).
            d_state_returned = d_state

            # Recompute state_new (pre-mask): y always depends on it, but
            # states[t+1] equals state_prev when mask=False, so it can't be used.
            if g_t.ndim == 2:
                decay = g_t[..., None, None]
            else:
                decay = g_t[..., None, :]
            state_decayed = state_prev * decay
            kv_mem = (state_decayed * k_t[..., None, :]).sum(axis=-1)
            delta = (v_t - kv_mem) * beta_t[..., None]
            state_new = state_decayed + k_t[..., None, :] * delta[..., None]

            # Forward: state_returned = where(mask, state_new, state_prev).
            # Split d_state_returned: to d_state_new when mask=True, passthrough
            # to d_state_prev when mask=False.
            if _has_mask:
                m = mask_chunk[:, t]
                m_exp = mx.expand_dims(m, axis=(1, 2, 3))
                zero = mx.zeros_like(d_state_returned)
                d_state_new_from_returned = mx.where(m_exp, d_state_returned, zero)
                d_state_prev_passthrough = mx.where(m_exp, zero, d_state_returned)
            else:
                d_state_new_from_returned = d_state_returned
                d_state_prev_passthrough = mx.zeros_like(d_state_returned)

            # y = (state_new * q).sum(-1) — y is unmasked, contributes always.
            d_state_new = (
                d_state_new_from_returned
                + dy_t[..., None].astype(mx.float32) * q_t[..., None, :].astype(mx.float32)
            )
            d_q_t = (dy_t[..., None].astype(mx.float32) * state_new).sum(axis=-2)
            d_q_steps.append(d_q_t.astype(q_c.dtype))

            # state_new = state_decayed + k[..., None, :] * delta[..., None]
            d_kd = d_state_new
            d_state_decayed = mx.array(d_state_new)

            # d_k / d_delta from the k*delta term
            d_k_t_from_update = (d_kd * delta[..., None].astype(mx.float32)).sum(axis=-2)
            d_delta = (d_kd * k_t[..., None, :].astype(mx.float32)).sum(axis=-1)

            # delta = (v - kv_mem) * beta[..., None]
            d_v_minus_kv = d_delta * beta_t[..., None].astype(mx.float32)
            d_beta_t = (d_delta * (v_t.astype(mx.float32) - kv_mem)).sum(axis=-1)
            d_v_t = d_v_minus_kv
            d_kv_mem = -d_v_minus_kv

            # kv_mem = (state_decayed * k[..., None, :]).sum(-1)
            d_state_decayed = (
                d_state_decayed
                + d_kv_mem[..., None].astype(mx.float32) * k_t[..., None, :].astype(mx.float32)
            )
            d_k_t_from_kv = (d_kv_mem[..., None].astype(mx.float32) * state_decayed).sum(axis=-2)

            # state_decayed = state_prev * decay
            d_state_prev_via_recurrence = d_state_decayed * decay.astype(mx.float32)
            d_decay = (d_state_decayed * state_prev).sum(axis=-2)
            if g_t.ndim == 2:
                d_g_t = d_decay.sum(axis=-1)
            else:
                d_g_t = d_decay

            d_k_t = d_k_t_from_update + d_k_t_from_kv
            d_k_steps.append(d_k_t.astype(k_c.dtype))
            d_v_steps.append(d_v_t.astype(v_c.dtype))
            d_g_steps.append(d_g_t.astype(g_c.dtype))
            d_beta_steps.append(d_beta_t.astype(beta_c.dtype))

            # d_state_prev = recurrence-derived gradient + mask passthrough.
            d_state = d_state_prev_via_recurrence + d_state_prev_passthrough

        for steps in (d_q_steps, d_k_steps, d_v_steps, d_g_steps, d_beta_steps):
            steps.reverse()
        d_q = mx.stack(d_q_steps, axis=1)
        d_k = mx.stack(d_k_steps, axis=1)
        d_v = mx.stack(d_v_steps, axis=1)
        d_g = mx.stack(d_g_steps, axis=1)
        d_beta = mx.stack(d_beta_steps, axis=1)
        d_mask = mx.zeros_like(mask_chunk) if mask_chunk is not None else None
        return d_q, d_k, d_v, d_g, d_beta, d_state, d_mask

    # Run chunked forward
    all_ys = []
    s = state
    for c in range(num_chunks):
        t_start = c * CHUNK_SIZE
        t_end = min(t_start + CHUNK_SIZE, T)
        q_c = q[:, t_start:t_end]
        k_c = k[:, t_start:t_end]
        v_c = v[:, t_start:t_end]
        g_c = g[:, t_start:t_end]
        beta_c = beta[:, t_start:t_end]
        # why: pass per-chunk mask as a primal so chunk-local t maps to the
        # right timesteps. Closure-captured `mask[:, t]` read mask[:,0:CHUNK]
        # for every chunk.
        if mask is None:
            mask_c = mx.ones((q_c.shape[0], q_c.shape[1]), dtype=mx.bool_)
        else:
            mask_c = mask[:, t_start:t_end]
        chunk_y, s = _chunked_forward(q_c, k_c, v_c, g_c, beta_c, s, mask_c)
        all_ys.append(chunk_y)

    y = mx.concatenate(all_ys, axis=1)
    return y, s


_WARNED_FOREIGN_GATED_DELTA: set[str] = set()


def _is_training_call(state, use_kernel):
    """An empty cache does not mark a training call -- prefill has one too. Call
    sites say so with `use_kernel=not self.training`; GLM-5.x's passes neither, so
    an open training window speaks for it."""
    if state is not None:
        return False
    if not use_kernel:
        return True
    # Absolute (test_relative_imports_resolve.py enforces it): custom_object_save
    # copies this file beside a checkpoint, where `mlx.utils.py` does not exist.
    # Lazy: this runs once per layer per decode step, so a pure-inference process
    # should not import `unsloth_zoo.mlx.utils` until a call site is ambiguous.
    from unsloth_zoo.mlx.utils import mlx_training_patches_active
    return mlx_training_patches_active()


def patch_gated_delta():
    """Monkey-patch mlx_lm's gated_delta module to use our efficient VJP.

    Consumers (mlx_lm qwen3_5 / qwen3_next / kimi_linear, mlx_vlm qwen3_5)
    bind ``gated_delta_update`` via ``from .gated_delta import ...`` at import
    time, so rebinding the source module alone never reaches their call sites.
    After patching the source, sweep already-imported consumer modules and
    rebind any stale reference to the original function.
    """
    import sys
    try:
        from mlx_lm.models import gated_delta
    except ImportError:
        # Reachable since layers are matched structurally: an mlx-vlm-defined
        # gated-delta mixer routes here even where mlx_lm has no such module.
        return

    if not getattr(gated_delta, "_unsloth_gated_delta_patched", False):
        original_gated_delta_update = gated_delta.gated_delta_update

        def patched_gated_delta_update(
            q, k, v, a, b, A_log, dt_bias,
            state=None, mask=None, use_kernel=True,
        ):
            is_training_call = _is_training_call(state, use_kernel)
            beta = mx.sigmoid(b)
            g = gated_delta.compute_g(A_log, a, dt_bias)
            if state is None:
                B, _, Hk, Dk = q.shape
                Hv, Dv = v.shape[-2:]
                state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

            # Training: prefer the fused-kernel VJP, ops VJP otherwise.
            if is_training_call:
                if gated_delta_kernel_supported(q, g, mask, v, k):
                    return gated_delta_kernel_efficient(
                        q, k, v, g, beta, state, mask,
                    )
                return gated_delta_ops_efficient(q, k, v, g, beta, state, mask)

            # Cached state: prefer the kernel for speed, else efficient ops.
            if not use_kernel or mx.default_device() != mx.gpu or not mx.metal.is_available():
                return gated_delta_ops_efficient(q, k, v, g, beta, state, mask)
            return gated_delta.gated_delta_kernel(q, k, v, g, beta, state, mask)

        gated_delta._unsloth_gated_delta_original = original_gated_delta_update
        gated_delta.gated_delta_update = patched_gated_delta_update
        gated_delta._unsloth_gated_delta_patched = True
        print("Unsloth: Patched GatedDeltaNet with memory-efficient custom VJP.")

    # Sweep every call: a consumer imported after a prior patch is still stale.
    original = getattr(gated_delta, "_unsloth_gated_delta_original", None)
    if original is None:
        # Older patch set the flag without recording the original; nothing to match.
        return
    patched = gated_delta.gated_delta_update
    rebound = []
    foreign = []
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith(("mlx_lm.models", "mlx_vlm.models")):
            continue
        binding = getattr(module, "gated_delta_update", None)
        if binding is None or binding is patched:
            continue
        if "patch_gated_delta" in getattr(binding, "__qualname__", ""):
            # Owned by a sibling unsloth patch (patch_gated_delta_vlm); not foreign.
            continue
        if binding is original:
            # Stale from-import of the function we replaced; rebind it. Anything
            # else is a foreign implementation (e.g. mlx-vlm >= 0.6's own module).
            module.gated_delta_update = patched
            rebound.append(name)
        else:
            foreign.append(name)
    if rebound:
        print(f"Unsloth: Rebound gated_delta_update in {', '.join(sorted(rebound))}.")
    new_foreign = [name for name in foreign if name not in _WARNED_FOREIGN_GATED_DELTA]
    if new_foreign:
        _WARNED_FOREIGN_GATED_DELTA.update(new_foreign)
        print(
            "Unsloth: WARNING — unrecognized gated_delta_update in "
            f"{', '.join(sorted(new_foreign))}; those modules will train without "
            "the memory-efficient VJP (slow, and long sequences may exhaust "
            "Metal resources)."
        )


def patch_gated_delta_vlm():
    """Patch mlx_vlm >= 0.6's own qwen3_5 gated_delta_update.

    That module ships its own copy of the function (calling the
    non-differentiable gated_delta_kernel directly), so it is a distinct
    object that patch_gated_delta()'s identity sweep deliberately leaves
    alone. Patch it with the same training dispatch (fused-kernel VJP,
    ops fallback) in both namespaces that hold a reference. Older
    mlx_vlm (0.4.x - 0.5.x) from-imports mlx_lm's function instead;
    the sweep in patch_gated_delta() already rebinds those.
    """
    import sys
    # Importing it here would drag the mlx-vlm model packages into a text-only run.
    if not any(name.startswith("mlx_vlm.models.qwen3_5") for name in sys.modules):
        return
    try:
        from mlx_vlm.models.qwen3_5 import gated_delta as vlm_gated_delta
    except ImportError:
        try:
            from mlx_vlm.models.qwen3_5 import language as vlm_language
        except ImportError:
            return
        patch_gated_delta()
        try:
            from mlx_lm.models import gated_delta
        except ImportError:
            return
        if (
            getattr(vlm_language, "gated_delta_update", None)
            is not gated_delta.gated_delta_update
        ):
            vlm_language.gated_delta_update = gated_delta.gated_delta_update
            print(
                "Unsloth: Rebound legacy mlx-vlm GatedDeltaNet to mlx-lm "
                "patched VJP."
            )
        return
    try:
        from mlx_vlm.models.qwen3_5 import language as vlm_language
    except ImportError:
        vlm_language = None
    try:
        from mlx_lm.models import gated_delta
    except ImportError:
        # `compute_g` below still comes from mlx_lm even on 0.6+, which ships its
        # own copy of the update; without it there is nothing to patch with.
        return

    if getattr(vlm_gated_delta, "_unsloth_gated_delta_patched", False):
        return

    original_update = vlm_gated_delta.gated_delta_update

    def patched_vlm_gated_delta_update(
        q, k, v, a, b, A_log, dt_bias,
        state=None, mask=None, use_kernel=True,
    ):
        if not _is_training_call(state, use_kernel):
            return original_update(
                q, k, v, a, b, A_log, dt_bias,
                state=state, mask=mask, use_kernel=use_kernel,
            )
        beta = mx.sigmoid(b)
        g = gated_delta.compute_g(A_log, a, dt_bias)
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        if gated_delta_kernel_supported(q, g, mask, v, k):
            return gated_delta_kernel_efficient(q, k, v, g, beta, state, mask)
        return gated_delta_ops_efficient(q, k, v, g, beta, state, mask)

    vlm_gated_delta.gated_delta_update = patched_vlm_gated_delta_update
    if vlm_language is not None:
        vlm_language.gated_delta_update = patched_vlm_gated_delta_update
    vlm_gated_delta._unsloth_gated_delta_patched = True
    print("Unsloth: Patched mlx-vlm GatedDeltaNet with memory-efficient custom VJP.")


def patch_gated_delta_vlm_shared():
    """Patch mlx_vlm's shared gated-delta update, wherever it lives.

    GLM-5.x linear attention binds this rather than the per-model copy
    `patch_gated_delta_vlm` covers, and never passes `use_kernel=not self.training`,
    so training would reach the fused Metal kernel, which has no VJP."""
    import sys
    # 0.6.5 kept this under `text_models`; 0.6.6 moved it up to `models`.
    vlm_gated_delta = (sys.modules.get("mlx_vlm.models.gated_delta")
                       or sys.modules.get("mlx_vlm.models.text_models.gated_delta"))
    if vlm_gated_delta is None:
        return
    if getattr(vlm_gated_delta, "_unsloth_gated_delta_patched", False):
        return
    original_update = vlm_gated_delta.gated_delta_update

    def patched_shared_gated_delta_update(q, k, v, a, b, A_log, dt_bias,
            state=None, mask=None, use_kernel=True, lower_bound=None):
        if not _is_training_call(state, use_kernel):
            # Pre-0.6.9 mlx-vlm has no `lower_bound` here, and no caller passes one.
            bound = {} if lower_bound is None else {"lower_bound": lower_bound}
            return original_update(q, k, v, a, b, A_log, dt_bias,
                                   state=state, mask=mask, use_kernel=use_kernel, **bound)
        beta = mx.sigmoid(b)
        g = (vlm_gated_delta.compute_g(A_log, a, dt_bias) if lower_bound is None
             else vlm_gated_delta.compute_g_safe(A_log, a, dt_bias, lower_bound))
        B, Dk = q.shape[0], q.shape[-1]
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        if gated_delta_kernel_supported(q, g, mask, v, k):
            return gated_delta_kernel_efficient(q, k, v, g, beta, state, mask)
        return gated_delta_ops_efficient(q, k, v, g, beta, state, mask)

    vlm_gated_delta.gated_delta_update = patched_shared_gated_delta_update
    vlm_gated_delta._unsloth_gated_delta_patched = True
    # Consumers from-import the function; rebind their stale references.
    rebound = []
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("mlx_vlm.models"): continue
        if getattr(module, "gated_delta_update", None) is original_update:
            module.gated_delta_update = patched_shared_gated_delta_update
            rebound.append(name)
    if rebound:
        print(f"Unsloth: Rebound gated_delta_update in {', '.join(sorted(rebound))}.")
    print("Unsloth: Patched shared mlx-vlm GatedDeltaNet with custom VJP.")


# --------------------------------------------------------------------------
# Fused Metal kernels for the training backward pass.
#
# The ops VJP above is memory-correct but dispatch-bound (~T*12 lazy
# primitives per layer per step, unfusable by mx.compile). These kernels work
# at chunk granularity: a blocked-sequential forward, and a backward that
# checkpoints block boundaries and reverse-scans with atomics, leaving low bits
# nondeterministic, as elsewhere on Metal.
#
# Must stay BELOW patch_gated_delta: the pinned-symbol suite (commit 46866ce)
# asserts the patched training branch precedes the first `gated_delta_kernel(`
# here. patch_gated_delta resolves these names at call time, so definition
# order is semantically irrelevant.
# --------------------------------------------------------------------------

# BPTT chunk length, derived per call: longer amortizes the launch and the replay.
_KERNEL_CHUNK_MIN = 64
_KERNEL_CHUNK_MAX = 512
_KERNEL_CHECKPOINT_BUDGET = 128 << 20


# Blocked-sequential thread layout, shared by the three kernels below. mlx-lm
# gives each state row its own simdgroup; here P = Dk/W threads cover a row
# inside one, DB rows share a threadgroup, which stages a TB-step block once.

_BLOCKED_TG_MEMORY = 32768   # Metal threadgroup-memory limit
_BLOCKED_TG_THREADS = 256
_BLOCKED_PAD = 8             # row padding, keeps threadgroup rows bank- and 8-byte-aligned

class _BlockedCfg(NamedTuple):
    dk_rows: int
    dv_rows: int
    dv_f32_rows: int
    red_arrays: int
    max_segment: int
    tb_choices: tuple


# Wide segments give the forward more rows per threadgroup and fewer redundant
# k/q reads; the backward carries three state fragments per thread and replays
# blocks, so it prefers narrow. First fit wins; forcing agreement costs 1.9x.
_BLOCKED_FORWARD_CFGS = (_BlockedCfg(2, 1, 0, 0, 32, (16, 8, 4)),)
_BLOCKED_BACKWARD_CFGS = (_BlockedCfg(2, 2, 1, 2, 16, (8, 4)),
                          _BlockedCfg(2, 2, 1, 2, 32, (8, 4)))


def _blocked_dv_block(Dv, cap, rows_per_simd):
    for DB in range(min(cap, Dv), 0, -1):
        if Dv % DB == 0 and DB % rows_per_simd == 0:
            return DB
    return None


@lru_cache(maxsize=None)
def _blocked_layout(Dk, Dv, in_dtype, vectorized, cfg):
    dk_rows, dv_rows, dv_f32_rows, red_arrays, max_segment, tb_choices = cfg
    if red_arrays and vectorized:
        red_arrays += 1        # per-column d_g reduces the same way d_k does
    if Dk <= 0 or Dv <= 0:
        return None
    # Halving the threadgroup costs far less than losing the fused path entirely.
    threads_cap = _BLOCKED_TG_THREADS
    while threads_cap >= 32:
        P = 1
        while P <= 32:
            W = Dk // P
            # Whole simdgroups: the backward reduces across the 32/P rows one holds.
            DB = (None if Dk % P or W % 4 or W > max_segment else
                  _blocked_dv_block(Dv, min(Dv, threads_cap // P), 32 // P))
            if DB is not None:
                # Cross-row reduction scratch: one Dk-wide row per simdgroup.
                fixed = red_arrays * (P * DB // 32) * Dk * 4
                for TB in tb_choices:
                    used = TB * (
                        in_dtype.size * (dk_rows * (Dk + _BLOCKED_PAD)
                                         + dv_rows * (DB + _BLOCKED_PAD))
                        + 4 * dv_f32_rows * (DB + _BLOCKED_PAD)
                        + 4 * (1 + (Dk + _BLOCKED_PAD if vectorized else 1)))
                    if used + fixed <= _BLOCKED_TG_MEMORY:
                        return P, W, DB, TB, P * DB
            P *= 2
        threads_cap //= 2
    return None


_BLOCKED_PREAMBLE = """
        constexpr int NV = W / 4;
        constexpr int TG = P * DB;
        constexpr int PAD = 8;

        const int tid = thread_position_in_threadgroup.x;
        const int blk = threadgroup_position_in_grid.x;
        const int hv  = threadgroup_position_in_grid.y;
        const int b   = threadgroup_position_in_grid.z;
        const int hk  = hv / (Hv / Hk);
        const int dv0 = blk * DB;

        // thread -> (row in the dv block, W-wide segment of Dk)
        const int dv  = tid / P;
        const int seg = tid % P;
        const int d0  = seg * W;
        const int row_lane = (tid % 32) / P * P;
        const int simd = tid / 32;
        constexpr int NSIMD = TG / 32;

        const size_t krow = (size_t)Hk * Dk;
        const size_t vrow = (size_t)Hv * Dv;
"""

_BLOCKED_FORWARD_SRC = """
        threadgroup InT k_s[TB][Dk + PAD];
        threadgroup InT q_s[TB][Dk + PAD];
        threadgroup InT v_s[TB][DB + PAD];
        threadgroup float b_s[TB];
        {g_decl}

        const device InT* k_base = k + ((size_t)b * T * Hk + hk) * Dk;
        const device InT* q_base = q + ((size_t)b * T * Hk + hk) * Dk;
        auto v_base = v + ((size_t)b * T * Hv + hv) * Dv + dv0;
        auto y_base = y + ((size_t)b * T * Hv + hv) * Dv + dv0;

        float4 st[NV];
        {{
          auto s_in = state_in + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0;
          for (int i = 0; i < NV; ++i) {{
            st[i] = float4(static_cast<float>(s_in[4 * i + 0]),
                           static_cast<float>(s_in[4 * i + 1]),
                           static_cast<float>(s_in[4 * i + 2]),
                           static_cast<float>(s_in[4 * i + 3]));
          }}
        }}

        for (int t0 = 0; t0 < T; t0 += TB) {{
          const int tt = (TB < T - t0) ? TB : (int)(T - t0);
          for (int p = tid; p < tt * Dk; p += TG) {{
            const int r = p / Dk, d = p % Dk;
            k_s[r][d] = static_cast<InT>(k_base[(size_t)(t0 + r) * krow + d]);
            q_s[r][d] = static_cast<InT>(q_base[(size_t)(t0 + r) * krow + d]);
          }}
          for (int p = tid; p < tt * DB; p += TG) {{
            const int r = p / DB, d = p % DB;
            v_s[r][d] = static_cast<InT>(v_base[(size_t)(t0 + r) * vrow + d]);
          }}
          for (int p = tid; p < tt; p += TG) {{
            b_s[p] = static_cast<float>(beta[((size_t)b * T + t0 + p) * Hv + hv]);
          }}
          {g_stage}
          threadgroup_barrier(mem_flags::mem_threadgroup);

          for (int t = 0; t < tt; ++t) {{
            const float bt = b_s[t];
            {g_load}
            const threadgroup vec<InT, 4>* k4 =
                (const threadgroup vec<InT, 4>*)&k_s[t][d0];
            const threadgroup vec<InT, 4>* q4 =
                (const threadgroup vec<InT, 4>*)&q_s[t][d0];

            float4 kf[NV];
            float part4 = 0.0f;
            float4 acc = 0.0f;
            for (int i = 0; i < NV; ++i) {{
              kf[i] = float4(k4[i]);
              st[i] *= {g_decay};
              acc += st[i] * kf[i];
            }}
            part4 = acc.x + acc.y + acc.z + acc.w;
            for (int off = P / 2; off > 0; off >>= 1) {{
              part4 += simd_shuffle_down(part4, off);
            }}
            const float kv_mem = simd_shuffle(part4, row_lane);
            const float delta = (static_cast<float>(v_s[t][dv]) - kv_mem) * bt;

            float4 o4 = 0.0f;
            for (int i = 0; i < NV; ++i) {{
              st[i] += kf[i] * delta;
              o4 += st[i] * float4(q4[i]);
            }}
            float out = o4.x + o4.y + o4.z + o4.w;
            for (int off = P / 2; off > 0; off >>= 1) {{
              out += simd_shuffle_down(out, off);
            }}
            if (seg == 0) {{
              y_base[(size_t)(t0 + t) * vrow + dv] = static_cast<InT>(out);
            }}
          }}
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        {{
          auto s_out = state_out + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0;
          for (int i = 0; i < NV; ++i) {{
            s_out[4 * i + 0] = static_cast<StT>(st[i].x);
            s_out[4 * i + 1] = static_cast<StT>(st[i].y);
            s_out[4 * i + 2] = static_cast<StT>(st[i].z);
            s_out[4 * i + 3] = static_cast<StT>(st[i].w);
          }}
        }}
"""


_BLOCKED_CHECKPOINT_SRC = """
        threadgroup InT k_s[TB][Dk + PAD];
        threadgroup InT v_s[TB][DB + PAD];
        threadgroup float b_s[TB];
        {g_decl}

        const device InT* k_base = k + ((size_t)b * T * Hk + hk) * Dk;
        auto v_base = v + ((size_t)b * T * Hv + hv) * Dv + dv0;

        // ckpt [B, Hv, NB, Dv, Dk]: state per boundary. kv [B, Hv, T, Dv]: k.state.
        constexpr size_t cstride = (size_t)Dv * Dk;
        const int NB = metal::max(1, (T + TB - 1) / TB - 1);
        auto ck_base = ckpt + ((size_t)b * Hv + hv) * NB * cstride
                     + (size_t)(dv0 + dv) * Dk + d0;
        auto kv_base = kv + ((size_t)b * Hv + hv) * (size_t)T * Dv + dv0 + dv;

        float4 st[NV];
        {{
          auto s_in = state_in + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0;
          for (int i = 0; i < NV; ++i) {{
            st[i] = float4(static_cast<float>(s_in[4 * i + 0]),
                           static_cast<float>(s_in[4 * i + 1]),
                           static_cast<float>(s_in[4 * i + 2]),
                           static_cast<float>(s_in[4 * i + 3]));
          }}
        }}

        for (int t0 = 0; t0 < T; t0 += TB) {{
          const int tt = (TB < T - t0) ? TB : (int)(T - t0);
          for (int p = tid; p < tt * Dk; p += TG) {{
            const int r = p / Dk, d = p % Dk;
            k_s[r][d] = static_cast<InT>(k_base[(size_t)(t0 + r) * krow + d]);
          }}
          for (int p = tid; p < tt * DB; p += TG) {{
            const int r = p / DB, d = p % DB;
            v_s[r][d] = static_cast<InT>(v_base[(size_t)(t0 + r) * vrow + d]);
          }}
          for (int p = tid; p < tt; p += TG) {{
            b_s[p] = static_cast<float>(beta[((size_t)b * T + t0 + p) * Hv + hv]);
          }}
          {g_stage}
          threadgroup_barrier(mem_flags::mem_threadgroup);

          for (int t = 0; t < tt; ++t) {{
            const float bt = b_s[t];
            {g_load}
            const threadgroup vec<InT, 4>* k4 =
                (const threadgroup vec<InT, 4>*)&k_s[t][d0];

            float4 kf[NV];
            float4 acc = 0.0f;
            for (int i = 0; i < NV; ++i) {{
              kf[i] = float4(k4[i]);
              st[i] *= {g_decay};
              acc += st[i] * kf[i];
            }}
            float kv_mem = acc.x + acc.y + acc.z + acc.w;
            for (int off = P / 2; off > 0; off >>= 1) {{
              kv_mem += simd_shuffle_down(kv_mem, off);
            }}
            kv_mem = simd_shuffle(kv_mem, row_lane);
            if (seg == 0) {{
              kv_base[(size_t)(t0 + t) * Dv] = kv_mem;
            }}
            const float delta = (static_cast<float>(v_s[t][dv]) - kv_mem) * bt;
            for (int i = 0; i < NV; ++i) {{
              st[i] += kf[i] * delta;
            }}
          }}
          if (t0 + TB < T) {{
            device float4* out = (device float4*)(ck_base + (size_t)(t0 / TB) * cstride);
            for (int i = 0; i < NV; ++i) {{
              out[i] = st[i];
            }}
          }}
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
"""


_BLOCKED_BACKWARD_SRC = """
        threadgroup InT k_s[TB][Dk + PAD];
        threadgroup InT q_s[TB][Dk + PAD];
        threadgroup InT v_s[TB][DB + PAD];
        threadgroup InT dy_s[TB][DB + PAD];
        threadgroup float kv_s[TB][DB + PAD];
        threadgroup float b_s[TB];
        threadgroup float red_q[NSIMD][Dk];
        threadgroup float red_k[NSIMD][Dk];
        {d_g_scratch}
        {g_decl}

        const device InT* k_base = k + ((size_t)b * T * Hk + hk) * Dk;
        const device InT* q_base = q + ((size_t)b * T * Hk + hk) * Dk;
        auto v_base  = v  + ((size_t)b * T * Hv + hv) * Dv + dv0;
        auto dy_base = dy + ((size_t)b * T * Hv + hv) * Dv + dv0;

        constexpr size_t cstride = (size_t)Dv * Dk;
        const int NB = metal::max(1, (T + TB - 1) / TB - 1);
        auto ck_base = ckpt + ((size_t)b * Hv + hv) * NB * cstride
                     + (size_t)(dv0 + dv) * Dk + d0;
        auto kv_base = kv + ((size_t)b * Hv + hv) * (size_t)T * Dv + dv0 + dv;
        auto s_in = state_in + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0;
        const int lane = tid % 32;

        float4 d_state[NV];
        {{
          auto ds = d_state_out + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0;
          for (int i = 0; i < NV; ++i) {{
            d_state[i] = float4(static_cast<float>(ds[4 * i + 0]),
                                static_cast<float>(ds[4 * i + 1]),
                                static_cast<float>(ds[4 * i + 2]),
                                static_cast<float>(ds[4 * i + 3]));
          }}
        }}

        float4 entry[NV];
        float4 sp[NV];

        for (int t0 = ((T - 1) / TB) * TB; t0 >= 0; t0 -= TB) {{
          const int tt = (TB < T - t0) ? TB : (int)(T - t0);
          threadgroup_barrier(mem_flags::mem_threadgroup);
          for (int p = tid; p < tt * Dk; p += TG) {{
            const int r = p / Dk, d = p % Dk;
            k_s[r][d] = static_cast<InT>(k_base[(size_t)(t0 + r) * krow + d]);
            q_s[r][d] = static_cast<InT>(q_base[(size_t)(t0 + r) * krow + d]);
          }}
          for (int p = tid; p < tt * DB; p += TG) {{
            const int r = p / DB, d = p % DB;
            v_s[r][d]  = static_cast<InT>(v_base[(size_t)(t0 + r) * vrow + d]);
            dy_s[r][d] = static_cast<InT>(dy_base[(size_t)(t0 + r) * vrow + d]);
            kv_s[r][d] = kv[((size_t)b * Hv + hv) * (size_t)T * Dv
                            + (size_t)(t0 + r) * Dv + dv0 + d];
          }}
          for (int p = tid; p < tt; p += TG) {{
            b_s[p] = static_cast<float>(beta[((size_t)b * T + t0 + p) * Hv + hv]);
          }}
          {g_stage}
          threadgroup_barrier(mem_flags::mem_threadgroup);

          if (t0 > 0) {{
            const device float4* ck =
                (const device float4*)(ck_base + (size_t)(t0 / TB - 1) * cstride);
            for (int i = 0; i < NV; ++i) {{
              entry[i] = ck[i];
            }}
          }} else {{
            for (int i = 0; i < NV; ++i) {{
              entry[i] = float4(static_cast<float>(s_in[4 * i + 0]),
                                static_cast<float>(s_in[4 * i + 1]),
                                static_cast<float>(s_in[4 * i + 2]),
                                static_cast<float>(s_in[4 * i + 3]));
            }}
          }}

          for (int t = tt - 1; t >= 0; --t) {{
            const int ts = t0 + t;
            const float bt = b_s[t];
            const float v_t = static_cast<float>(v_s[t][dv]);
            const float dy_t = static_cast<float>(dy_s[t][dv]);
            const float kv_mem = kv_s[t][dv];
            const float delta = (v_t - kv_mem) * bt;

            // Replay to t-1 in the checkpoint pass's fma order, so it reproduces it.
            for (int i = 0; i < NV; ++i) {{
              sp[i] = entry[i];
            }}
            for (int u = 0; u < t; ++u) {{
              const threadgroup vec<InT, 4>* ku =
                  (const threadgroup vec<InT, 4>*)&k_s[u][d0];
              {g_load_u}
              const float du = (static_cast<float>(v_s[u][dv]) - kv_s[u][dv]) * b_s[u];
              for (int i = 0; i < NV; ++i) {{
                sp[i] *= {g_decay_u};
                sp[i] += float4(ku[i]) * du;
              }}
            }}

            {g_load}
            const threadgroup vec<InT, 4>* k4 =
                (const threadgroup vec<InT, 4>*)&k_s[t][d0];
            const threadgroup vec<InT, 4>* q4 =
                (const threadgroup vec<InT, 4>*)&q_s[t][d0];

            float4 acc_d = 0.0f;
            for (int i = 0; i < NV; ++i) {{
              const float4 kf = float4(k4[i]);
              acc_d += (d_state[i] + dy_t * float4(q4[i])) * kf;
              float4 dq = dy_t * (sp[i] * {g_decay} + kf * delta);
              for (int off = P; off < 32; off <<= 1) {{
                dq += simd_shuffle_down(dq, off);
              }}
              if (lane < P) {{
                threadgroup float* r = &red_q[simd][d0 + 4 * i];
                r[0] = dq.x; r[1] = dq.y; r[2] = dq.z; r[3] = dq.w;
              }}
            }}
            float d_delta = acc_d.x + acc_d.y + acc_d.z + acc_d.w;
            for (int off = P / 2; off > 0; off >>= 1) {{
              d_delta += simd_shuffle_down(d_delta, off);
            }}
            d_delta = simd_shuffle(d_delta, row_lane);
            const float d_kv = -d_delta * bt;

            {d_g_decl}
            for (int i = 0; i < NV; ++i) {{
              const float4 kf = float4(k4[i]);
              const float4 dS = d_state[i] + dy_t * float4(q4[i]);
              const float4 sd = sp[i] * {g_decay};
              const float4 d_sd = dS + d_kv * kf;
              float4 dkc = dS * delta + d_kv * sd;
              for (int off = P; off < 32; off <<= 1) {{
                dkc += simd_shuffle_down(dkc, off);
              }}
              if (lane < P) {{
                threadgroup float* r = &red_k[simd][d0 + 4 * i];
                r[0] = dkc.x; r[1] = dkc.y; r[2] = dkc.z; r[3] = dkc.w;
              }}
              {d_g_accum}
              d_state[i] = d_sd * {g_decay};
            }}
            {d_g_finalize}

            auto d_q_ = d_q + ((size_t)b * T + ts) * Hk * Dk + (size_t)hk * Dk;
            auto d_k_ = d_k + ((size_t)b * T + ts) * Hk * Dk + (size_t)hk * Dk;
            {d_g_setup}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int c = tid; c < Dk; c += TG) {{
              float sq = 0.0f, sk = 0.0f;
              {d_g_reduce_decl}
              for (int m = 0; m < NSIMD; ++m) {{
                sq += red_q[m][c];
                sk += red_k[m][c];
                {d_g_reduce_accum}
              }}
              atomic_fetch_add_explicit(&d_q_[c], sq, memory_order_relaxed);
              atomic_fetch_add_explicit(&d_k_[c], sk, memory_order_relaxed);
              {d_g_reduce_store}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (seg == 0) {{
              auto d_v_ = d_v + ((size_t)b * T + ts) * Hv * Dv + (size_t)hv * Dv;
              atomic_fetch_add_explicit(&d_v_[dv0 + dv], d_delta * bt,
                                        memory_order_relaxed);
            }}
            float d_beta_part = (seg == 0) ? d_delta * (v_t - kv_mem) : 0.0f;
            d_beta_part = simd_sum(d_beta_part);
            if (lane == 0) {{
              atomic_fetch_add_explicit(&d_beta[((size_t)b * T + ts) * Hv + hv],
                                        d_beta_part, memory_order_relaxed);
            }}
          }}
        }}

        {{
          auto dsi = d_state_in + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0;
          for (int i = 0; i < NV; ++i) {{
            atomic_fetch_add_explicit(&dsi[4 * i + 0], d_state[i].x, memory_order_relaxed);
            atomic_fetch_add_explicit(&dsi[4 * i + 1], d_state[i].y, memory_order_relaxed);
            atomic_fetch_add_explicit(&dsi[4 * i + 2], d_state[i].z, memory_order_relaxed);
            atomic_fetch_add_explicit(&dsi[4 * i + 3], d_state[i].w, memory_order_relaxed);
          }}
        }}
"""


def _blocked_gating(vectorized):
    if vectorized:
        return {
            "g_decl": "threadgroup float g_s[TB][Dk + PAD];",
            "g_stage": """for (int p = tid; p < tt * Dk; p += TG) {
            const int r = p / Dk, d = p % Dk;
            g_s[r][d] = static_cast<float>(
                g[(((size_t)b * T + t0 + r) * Hv + hv) * Dk + d]);
          }""",
            "g_load": "const threadgroup float4* g4 = (const threadgroup float4*)&g_s[t][d0];",
            "g_decay": "g4[i]",
            "g_load_u": "const threadgroup float4* gu = (const threadgroup float4*)&g_s[u][d0];",
            "g_decay_u": "gu[i]",
            "d_g_scratch": "threadgroup float red_g[NSIMD][Dk];",
            "d_g_setup": "auto d_g_ = d_g + (((size_t)b * T + ts) * Hv + hv) * Dk;",
            "d_g_decl": "",
            "d_g_accum": """float4 dgc = d_sd * sp[i];
              for (int off = P; off < 32; off <<= 1) {
                dgc += simd_shuffle_down(dgc, off);
              }
              if (lane < P) {
                threadgroup float* rg = &red_g[simd][d0 + 4 * i];
                rg[0] = dgc.x; rg[1] = dgc.y; rg[2] = dgc.z; rg[3] = dgc.w;
              }""",
            "d_g_finalize": "",
            "d_g_reduce_decl": "float sg = 0.0f;",
            "d_g_reduce_accum": "sg += red_g[m][c];",
            "d_g_reduce_store": "atomic_fetch_add_explicit(&d_g_[c], sg, memory_order_relaxed);",
        }
    return {
        "g_decl": "threadgroup float g_s[TB];",
        "g_stage": """for (int p = tid; p < tt; p += TG) {
            g_s[p] = static_cast<float>(g[((size_t)b * T + t0 + p) * Hv + hv]);
          }""",
        "g_load": "const float gt = g_s[t];",
        "g_decay": "gt",
        "g_load_u": "const float gu = g_s[u];",
        "g_decay_u": "gu",
        "d_g_scratch": "",
        "d_g_setup": "",
        "d_g_decl": "float d_g_part = 0.0f;",
        "d_g_reduce_decl": "",
        "d_g_reduce_accum": "",
        "d_g_reduce_store": "",
        "d_g_accum": "d_g_part += dot(d_sd, sp[i]);",
        "d_g_finalize": """d_g_part = simd_sum(d_g_part);
            if (lane == 0) {
              atomic_fetch_add_explicit(&d_g[((size_t)b * T + ts) * Hv + hv],
                                        d_g_part, memory_order_relaxed);
            }""",
    }


def _make_gd_blocked_forward_kernel(vectorized=False):
    source = _BLOCKED_PREAMBLE + _BLOCKED_FORWARD_SRC.format(**_blocked_gating(vectorized))
    return mx.fast.metal_kernel(
        name=f"unsloth_gd_blocked_fwd{'_vec' if vectorized else ''}",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=source,
    )


def _make_gd_blocked_checkpoint_kernel(vectorized=False):
    source = _BLOCKED_PREAMBLE + _BLOCKED_CHECKPOINT_SRC.format(**_blocked_gating(vectorized))
    return mx.fast.metal_kernel(
        name=f"unsloth_gd_blocked_ckpt{'_vec' if vectorized else ''}",
        input_names=["k", "v", "g", "beta", "state_in", "T"],
        output_names=["ckpt", "kv"],
        source=source,
    )


def _make_gd_blocked_backward_kernel(vectorized=False):
    source = _BLOCKED_PREAMBLE + _BLOCKED_BACKWARD_SRC.format(**_blocked_gating(vectorized))
    return mx.fast.metal_kernel(
        name=f"unsloth_gd_blocked_backward{'_vec' if vectorized else ''}",
        input_names=[
            "q", "k", "v", "g", "beta", "state_in", "ckpt", "kv",
            "dy", "d_state_out", "T",
        ],
        output_names=["d_q", "d_k", "d_v", "d_g", "d_beta", "d_state_in"],
        source=source,
        atomic_outputs=True,
    )


_GD_BLOCKED: dict = {}


def _get_blocked_kernels(vectorized=False):
    if vectorized not in _GD_BLOCKED:
        _GD_BLOCKED[vectorized] = (None, None, None) if not mx.metal.is_available() else (
            _make_gd_blocked_forward_kernel(vectorized),
            _make_gd_blocked_checkpoint_kernel(vectorized),
            _make_gd_blocked_backward_kernel(vectorized),
        )
    return _GD_BLOCKED[vectorized]


def _blocked_layouts(q, g, v):
    """(forward, backward) layouts, or None if either does not fit -- the backward
    replays the forward, so a shape needs both. The checkpoint kernel shares the
    backward's: TB is what its checkpoint boundaries count in."""
    Dk, Dv, vec = q.shape[-1], v.shape[-1], g.ndim == 4
    out = tuple(
        next((lay for cfg in cfgs
              if (lay := _blocked_layout(Dk, Dv, q.dtype, vec, cfg)) is not None), None)
        for cfgs in (_BLOCKED_FORWARD_CFGS, _BLOCKED_BACKWARD_CFGS)
    )
    return None if any(x is None for x in out) else out


def _blocked_shares_q_dtype(q, *others):
    """One Metal template element type stages them all, so another dtype misreads."""
    return all(x.dtype == q.dtype for x in others)


def gated_delta_blocked_forward(q, k, v, g, beta, state):
    """Forward half, or None when it does not apply. gated_delta_kernel's unmasked
    contract: (y [B, T, Hv, Dv] in q.dtype, state_out shaped and typed like state)."""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    layouts = _blocked_layouts(q, g, v)
    kernel = _get_blocked_kernels(g.ndim == 4)[0]
    if layouts is None or kernel is None or not _blocked_shares_q_dtype(q, k, v):
        return None
    P, W, DB, TB, threads = layouts[0]
    return kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[
            ("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv),
            ("Hk", Hk), ("Hv", Hv), ("TB", TB), ("P", P), ("W", W), ("DB", DB),
        ],
        grid=(threads * (Dv // DB), Hv, B),
        threadgroup=(threads, 1, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


def gated_delta_blocked_backward(q, k, v, g, beta, state, dy, d_state_out):
    """Backward half, or None when it does not apply: six fp32 gradients in primal order."""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    vectorized = g.ndim == 4
    layouts = _blocked_layouts(q, g, v)
    _, ckpt_kernel, backward_kernel = _get_blocked_kernels(vectorized)
    if (layouts is None or ckpt_kernel is None
            or not _blocked_shares_q_dtype(q, k, v, dy)):
        return None
    P, W, DB, TB, threads = layouts[1]
    template = [
        ("InT", q.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv),
        ("TB", TB), ("P", P), ("W", W), ("DB", DB),
    ]
    grid, threadgroup = (threads * (Dv // DB), Hv, B), (threads, 1, 1)
    # A single-block chunk would pass an empty input, which arrives as `constant`.
    boundaries = max(1, -(-T // TB) - 1)

    ckpt, kv = ckpt_kernel(
        inputs=[k, v, g, beta, state, T],
        template=template, grid=grid, threadgroup=threadgroup,
        output_shapes=[(B, Hv, boundaries, Dv, Dk), (B, Hv, T, Dv)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return backward_kernel(
        inputs=[q, k, v, g, beta, state, ckpt, kv, dy, d_state_out, T],
        template=template, grid=grid, threadgroup=threadgroup,
        output_shapes=[
            (B, T, Hk, Dk), (B, T, Hk, Dk), (B, T, Hv, Dv),
            (B, T, Hv, Dk) if vectorized else (B, T, Hv), (B, T, Hv), state.shape,
        ],
        output_dtypes=[mx.float32] * 6,
        init_value=0,
    )


def gated_delta_kernel_supported(q, g, mask, v=None, k=None) -> bool:
    """Whether the fused-kernel VJP path can handle this call.

    A caller omitting `v` or `k` gets False, not a yes the dispatch would refuse.

    Device first: _blocked_layouts reads `dtype.size`, which only a real mx.Dtype
    has, so off Metal this has to answer False before reaching it rather than
    raise. tests/mlx_simulation hands out torch dtypes, so a training call under
    the shim died in the layout search instead of falling back to the ops VJP.
    """
    return (
        mx.metal.is_available()
        and mx.default_device() == mx.gpu
        and mask is None
        and v is not None
        and k is not None
        and _blocked_shares_q_dtype(q, k, v)
        and g.ndim in (3, 4)
        and q.shape[-1] % 32 == 0
        and _blocked_layouts(q, g, v) is not None
    )


def _kernel_chunk_size(B, Hv, Dk, Dv, TB):
    """Longest chunk whose checkpoints fit the budget, clamped to the length bounds
    and so overridden by either; n blocks hold n - 1 states."""
    states = _KERNEL_CHECKPOINT_BUDGET // max(1, B * Hv * Dv * Dk * 4)
    return min(_KERNEL_CHUNK_MAX, max(_KERNEL_CHUNK_MIN, (states + 1) * TB))


def gated_delta_kernel_efficient(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """Chunked GDN forward+backward with fused Metal kernels.

    Same contract as gated_delta_ops_efficient, restricted to the
    kernel-eligible case (see gated_delta_kernel_supported).
    """
    if not gated_delta_kernel_supported(q, g, mask, v, k):
        raise ValueError(
            "gated_delta_kernel_efficient called outside kernel support "
            "(requires a Metal GPU, mask=None, Dk % 32 == 0, one dtype shared "
            "by q, k and v, and a thread layout for this (Dk, Dv) -- see "
            "gated_delta_kernel_supported); use gated_delta_ops_efficient "
            "instead."
        )
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    chunk = _kernel_chunk_size(B, Hv, Dk, Dv, _blocked_layouts(q, g, v)[1][3])

    if (repeat_factor := Hv // Hk) > 1:
        # Repeat outside custom_function so autodiff folds the per-group
        # gradient sum back to [B, T, Hk, Dk].
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    @mx.custom_function
    def _chunk(q_c, k_c, v_c, g_c, beta_c, state_in):
        return gated_delta_blocked_forward(q_c, k_c, v_c, g_c, beta_c, state_in)

    @_chunk.vjp
    def _chunk_vjp(primals, cotangents, outputs):
        q_c, k_c, v_c, g_c, beta_c, state_in = primals
        dy, d_state_out = cotangents
        d_q, d_k, d_v, d_g, d_beta, d_state_in = gated_delta_blocked_backward(
            q_c, k_c, v_c, g_c, beta_c, state_in, dy, d_state_out,
        )
        return (
            d_q.astype(q_c.dtype),
            d_k.astype(k_c.dtype),
            d_v.astype(v_c.dtype),
            d_g.astype(g_c.dtype),
            d_beta.astype(beta_c.dtype),
            d_state_in.astype(state_in.dtype),
        )

    all_ys = []
    # fp32 across chunks: rounding per chunk makes the length observable in y.
    s = state if T <= chunk else state.astype(mx.float32)
    for t_start in range(0, T, chunk):
        t_end = min(t_start + chunk, T)
        chunk_y, s = _chunk(
            q[:, t_start:t_end],
            k[:, t_start:t_end],
            v[:, t_start:t_end],
            g[:, t_start:t_end],
            beta[:, t_start:t_end],
            s,
        )
        all_ys.append(chunk_y)

    y = mx.concatenate(all_ys, axis=1) if len(all_ys) > 1 else all_ys[0]
    return y, s.astype(state.dtype)
