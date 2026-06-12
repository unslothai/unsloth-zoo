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

from typing import Optional, Tuple
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
        # t is produced exactly once, and mx `.at[:, t].add` scatters read
        # the update with a wrong batch stride (wrong grads for every batch
        # row past the first; verified against plain autodiff on mlx 0.31).
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

        d_q = mx.stack(d_q_steps[::-1], axis=1)
        d_k = mx.stack(d_k_steps[::-1], axis=1)
        d_v = mx.stack(d_v_steps[::-1], axis=1)
        d_g = mx.stack(d_g_steps[::-1], axis=1)
        d_beta = mx.stack(d_beta_steps[::-1], axis=1)
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


_WARNED_FOREIGN_GATED_DELTA: set = set()


def patch_gated_delta():
    """Monkey-patch mlx_lm's gated_delta module to use our efficient VJP.

    Consumers (mlx_lm qwen3_5 / qwen3_next / kimi_linear, mlx_vlm qwen3_5)
    bind ``gated_delta_update`` via ``from .gated_delta import ...`` at import
    time, so rebinding the source module alone never reaches their call sites.
    After patching the source, sweep already-imported consumer modules and
    rebind any stale reference to the original function.
    """
    import sys
    from mlx_lm.models import gated_delta

    if not getattr(gated_delta, "_unsloth_gated_delta_patched", False):
        original_gated_delta_update = gated_delta.gated_delta_update

        def patched_gated_delta_update(
            q, k, v, a, b, A_log, dt_bias,
            state=None, mask=None, use_kernel=True,
        ):
            # Heuristic: training calls enter with state=None (no cache); inference
            # passes the cached state in. Route training through the efficient ops
            # so the custom VJP runs — gated_delta_kernel has none, so it would keep
            # all T intermediate states alive and defeat this module.
            is_training_call = state is None
            beta = mx.sigmoid(b)
            g = gated_delta.compute_g(A_log, a, dt_bias)
            if state is None:
                B, _, Hk, Dk = q.shape
                Hv, Dv = v.shape[-2:]
                state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

            # No incoming state cache: training. Prefer the fused-kernel VJP
            # (fast, O(chunks) graph); fall back to the ops VJP when the call
            # shape is outside kernel support.
            if is_training_call:
                if gated_delta_kernel_supported(q, g, mask):
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

    # Sweep on every call (not just the first): a consumer module imported
    # after a previous patch still holds a stale from-import binding.
    original = gated_delta._unsloth_gated_delta_original
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
            # A sibling unsloth patch (patch_gated_delta_vlm) already owns
            # this binding; it is not a foreign implementation.
            continue
        if binding is original:
            # Identity match: this is a stale from-import of the function we
            # replaced. Anything else (e.g. mlx-vlm >= 0.6 ships its own
            # gated_delta module) is a foreign implementation we must not touch.
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
    try:
        from mlx_vlm.models.qwen3_5 import gated_delta as vlm_gated_delta
        from mlx_vlm.models.qwen3_5 import language as vlm_language
    except ImportError:
        return
    from mlx_lm.models import gated_delta

    if getattr(vlm_gated_delta, "_unsloth_gated_delta_patched", False):
        return

    original_update = vlm_gated_delta.gated_delta_update

    def patched_vlm_gated_delta_update(
        q, k, v, a, b, A_log, dt_bias,
        state=None, mask=None, use_kernel=True,
    ):
        # state=None means a training call; route it through the
        # differentiable VJP. Inference (state provided) keeps the kernel.
        if state is not None:
            return original_update(
                q, k, v, a, b, A_log, dt_bias,
                state=state, mask=mask, use_kernel=use_kernel,
            )
        beta = mx.sigmoid(b)
        g = gated_delta.compute_g(A_log, a, dt_bias)
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        if gated_delta_kernel_supported(q, g, mask):
            return gated_delta_kernel_efficient(q, k, v, g, beta, state, mask)
        return gated_delta_ops_efficient(q, k, v, g, beta, state, mask)

    vlm_gated_delta.gated_delta_update = patched_vlm_gated_delta_update
    vlm_language.gated_delta_update = patched_vlm_gated_delta_update
    vlm_gated_delta._unsloth_gated_delta_patched = True
    print("Unsloth: Patched mlx-vlm GatedDeltaNet with memory-efficient custom VJP.")


# --------------------------------------------------------------------------
# Fused Metal kernels for the training backward pass.
#
# The ops-based VJP above is memory-correct but builds ~T*12 lazy
# primitives per layer per step (dispatch-bound, unfusable by mx.compile).
# These kernels replace both directions at chunk granularity: forward
# reuses mlx-lm's fused gated_delta_kernel per chunk; backward replays the
# chunk states (K1) and reverse-scans with atomic grad accumulation (K2).
# Eligibility: Metal GPU, no mask (training passes mask=None), and
# Dk % 32 == 0; both scalar (qwen3_5/qwen3_next) and vectorized
# (kimi_linear) gating are supported — anything else falls back to the
# ops VJP. Atomic accumulation makes low-order gradient bits
# nondeterministic, matching Metal reduction-order behavior elsewhere.
#
# NOTE: this section must stay BELOW patch_gated_delta. The pinned-symbol
# suite asserts the patched function's training branch precedes the first
# `gated_delta_kernel(` occurrence in this file (commit 46866ce regression
# net), and the chunked forward below legitimately calls that kernel.
# patch_gated_delta resolves these names at call time, so definition
# order is semantically irrelevant.
# --------------------------------------------------------------------------

_KERNEL_CHUNK_SIZE = 64
_KERNEL_THREADGROUP_X = 32
_KERNEL_THREADGROUP_Y = 4


def _make_gd_chunk_states_kernel(vectorized=False):
    """K1: replay the chunk forward, storing every post-step state.

    Thread layout mirrors mlx-lm's gated_delta_step kernel: grid z = B*Hv,
    grid y = Dv (one state row per simdgroup), 32 lanes each owning
    Dk/32 contiguous state columns held in registers. `vectorized` selects
    per-column gating (g: [B, T, Hv, Dk], kimi_linear) over per-head
    scalar gating (g: [B, T, Hv]).
    """
    if not mx.metal.is_available():
        return None
    if vectorized:
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_decay = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_decay = "g_[hv_idx]"
        g_advance = "g_ += Hv;"
    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        constexpr int n_per_t = Dk / 32;

        // q unused in the state replay; k: [B, T, Hv, Dk] (post GQA-repeat)
        auto k_ = k + b_idx * T * Hv * Dk + hv_idx * Dk;
        // v: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        // states: [B, Hv, T, Dv, Dk] -- post-update state for each step
        auto states_ = states + ((n * T) * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * {g_decay};
            kv_mem += state[i] * k_[s_idx];
          }}
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem)
              * static_cast<float>(beta_[hv_idx]);

          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
            states_[s_idx] = state[i];
          }}

          k_ += Hv * Dk;
          v_ += Hv * Dv;
          {g_advance}
          beta_ += Hv;
          states_ += Dv * Dk;
        }}
    """
    suffix = "_vec" if vectorized else ""
    return mx.fast.metal_kernel(
        name=f"unsloth_gd_chunk_states{suffix}",
        input_names=["k", "v", "g", "beta", "state_in", "T"],
        output_names=["states"],
        source=source,
    )


def _make_gd_chunk_backward_kernel(vectorized=False):
    """K2: reverse-time scan over one chunk, accumulating input gradients.

    Same thread layout as K1. The per-row d_state slice lives in registers;
    cross-row reductions (d_q, d_k, d_g, d_beta) go through atomic float
    adds into zero-initialized outputs. With vectorized gating d_g is
    per-column ([B, T, Hv, Dk]) and needs no simdgroup reduction.
    """
    if not mx.metal.is_available():
        return None
    if vectorized:
        g_setup = "auto g_ = g + ((b_idx * T + (T - 1)) * Hv + hv_idx) * Dk;"
        d_g_setup = "auto d_g_ = d_g + ((b_idx * T + (T - 1)) * Hv + hv_idx) * Dk;"
        g_step_decl = "float gcol[n_per_t];"
        g_load = "gcol[i] = static_cast<float>(g_[s_idx]);"
        g_col = "gcol[i]"
        d_g_accum = """atomic_fetch_add_explicit(&d_g_[s_idx], d_sd * s_prev[i],
                                      memory_order_relaxed);"""
        d_g_finalize = ""
        g_retreat = "g_ -= Hv * Dk;"
        d_g_retreat = "d_g_ -= Hv * Dk;"
    else:
        g_setup = "auto g_ = g + (b_idx * T + (T - 1)) * Hv;"
        d_g_setup = "auto d_g_ = d_g + (b_idx * T + (T - 1)) * Hv;"
        g_step_decl = "float g_t = 0.0f;"
        g_load = "g_t = static_cast<float>(g_[hv_idx]);"
        g_col = "g_t"
        d_g_accum = "d_g_partial += d_sd * s_prev[i];"
        d_g_finalize = """d_g_partial = simd_sum(d_g_partial);
          if (thread_index_in_simdgroup == 0) {
            atomic_fetch_add_explicit(&d_g_[hv_idx], d_g_partial,
                                      memory_order_relaxed);
          }"""
        g_retreat = "g_ -= Hv;"
        d_g_retreat = "d_g_ -= Hv;"
    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // Start every per-timestep pointer at t = T-1.
        auto q_ = q + (b_idx * T + (T - 1)) * Hv * Dk + hv_idx * Dk;
        auto k_ = k + (b_idx * T + (T - 1)) * Hv * Dk + hv_idx * Dk;
        auto v_ = v + (b_idx * T + (T - 1)) * Hv * Dv + hv_idx * Dv;
        {g_setup}
        auto beta_ = beta + (b_idx * T + (T - 1)) * Hv;
        auto dy_ = dy + (b_idx * T + (T - 1)) * Hv * Dv + hv_idx * Dv;

        auto d_q_ = d_q + (b_idx * T + (T - 1)) * Hv * Dk + hv_idx * Dk;
        auto d_k_ = d_k + (b_idx * T + (T - 1)) * Hv * Dk + hv_idx * Dk;
        auto d_v_ = d_v + (b_idx * T + (T - 1)) * Hv * Dv + hv_idx * Dv;
        {d_g_setup}
        auto d_beta_ = d_beta + (b_idx * T + (T - 1)) * Hv;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto states_ = states + ((n * T) * Dv + dv_idx) * Dk;
        auto d_state_out_ = d_state_out + (n * Dv + dv_idx) * Dk;
        auto d_state_in_ = d_state_in + (n * Dv + dv_idx) * Dk;

        // d_state: cotangent w.r.t. the state RETURNED by step t.
        float d_state[n_per_t];
        float s_prev[n_per_t];
        float s_cur[n_per_t];
        // Reduce threadgroup-local Dv rows before the global d_q/d_k atomics.
        threadgroup float tg_dq[TG_ROWS * 32];
        threadgroup float tg_dk[TG_ROWS * 32];
        {g_step_decl}
        auto tg_y = thread_position_in_threadgroup.y;
        auto tg_lane = thread_index_in_simdgroup;
        auto tg_idx = tg_y * 32 + tg_lane;
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          d_state[i] = static_cast<float>(d_state_out_[s_idx]);
        }}

        for (int t = T - 1; t >= 0; --t) {{
          auto cur_row = states_ + t * Dv * Dk;
          float beta_t = static_cast<float>(beta_[hv_idx]);
          float v_t = static_cast<float>(v_[dv_idx]);
          float dy_t = static_cast<float>(dy_[dv_idx]);

          // Recompute Sd = S_prev * g, kv, delta from the stored states.
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            {g_load}
            if (t > 0) {{
              auto prev_row = states_ + (t - 1) * Dv * Dk;
              s_prev[i] = static_cast<float>(prev_row[s_idx]);
            }} else {{
              s_prev[i] = static_cast<float>(i_state[s_idx]);
            }}
            s_cur[i] = cur_row[s_idx];
            kv_mem += s_prev[i] * {g_col} * static_cast<float>(k_[s_idx]);
          }}
          kv_mem = simd_sum(kv_mem);
          float delta = (v_t - kv_mem) * beta_t;

          // y = (S_t * q).sum(-1): dy contributes to d_state and d_q.
          float d_delta_partial = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            float k_c = static_cast<float>(k_[s_idx]);
            float q_c = static_cast<float>(q_[s_idx]);
            float dS_tot = d_state[i] + dy_t * q_c;

            tg_dq[tg_idx] = dy_t * s_cur[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tg_y == 0) {{
              float d_q_group = 0.0f;
              for (int yy = 0; yy < TG_ROWS; ++yy) {{
                d_q_group += tg_dq[yy * 32 + tg_lane];
              }}
              atomic_fetch_add_explicit(&d_q_[s_idx], d_q_group,
                                        memory_order_relaxed);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            d_delta_partial += dS_tot * k_c;
            // Stash dS_tot for the second pass below.
            s_cur[i] = dS_tot;
          }}
          float d_delta = simd_sum(d_delta_partial);

          float d_kv = -d_delta * beta_t;
          float d_g_partial = 0.0f;
          (void)d_g_partial;
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            float k_c = static_cast<float>(k_[s_idx]);
            float dS_tot = s_cur[i];
            float sd = s_prev[i] * {g_col};
            float d_sd = dS_tot + d_kv * k_c;

            tg_dk[tg_idx] = dS_tot * delta + d_kv * sd;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tg_y == 0) {{
              float d_k_group = 0.0f;
              for (int yy = 0; yy < TG_ROWS; ++yy) {{
                d_k_group += tg_dk[yy * 32 + tg_lane];
              }}
              atomic_fetch_add_explicit(&d_k_[s_idx], d_k_group,
                                        memory_order_relaxed);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {d_g_accum}
            d_state[i] = d_sd * {g_col};
          }}
          {d_g_finalize}

          if (thread_index_in_simdgroup == 0) {{
            atomic_fetch_add_explicit(&d_v_[dv_idx], d_delta * beta_t,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&d_beta_[hv_idx],
                                      d_delta * (v_t - kv_mem),
                                      memory_order_relaxed);
          }}

          if (t > 0) {{
            q_ -= Hv * Dk;
            k_ -= Hv * Dk;
            v_ -= Hv * Dv;
            {g_retreat}
            beta_ -= Hv;
            dy_ -= Hv * Dv;
            d_q_ -= Hv * Dk;
            d_k_ -= Hv * Dk;
            d_v_ -= Hv * Dv;
            {d_g_retreat}
            d_beta_ -= Hv;
          }}
        }}

        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          atomic_fetch_add_explicit(&d_state_in_[s_idx], d_state[i],
                                    memory_order_relaxed);
        }}
    """
    suffix = "_vec" if vectorized else ""
    return mx.fast.metal_kernel(
        name=f"unsloth_gd_chunk_backward{suffix}",
        input_names=[
            "q", "k", "v", "g", "beta", "state_in", "states",
            "dy", "d_state_out", "T",
        ],
        output_names=["d_q", "d_k", "d_v", "d_g", "d_beta", "d_state_in"],
        source=source,
        atomic_outputs=True,
    )


_GD_KERNELS: dict = {}


def _get_kernels(vectorized=False):
    if vectorized not in _GD_KERNELS:
        _GD_KERNELS[vectorized] = (
            _make_gd_chunk_states_kernel(vectorized=vectorized),
            _make_gd_chunk_backward_kernel(vectorized=vectorized),
        )
    return _GD_KERNELS[vectorized]


def gated_delta_kernel_supported(q, g, mask) -> bool:
    """Whether the fused-kernel VJP path can handle this call."""
    return (
        mask is None
        and g.ndim in (3, 4)
        and q.shape[-1] % 32 == 0
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    )


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
    assert mask is None
    from mlx_lm.models.gated_delta import gated_delta_kernel

    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (repeat_factor := Hv // Hk) > 1:
        # The repeat is recorded outside the custom_function, so autodiff
        # folds the per-group gradient sum back to [B, T, Hk, Dk] for us.
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    vectorized = g.ndim == 4
    states_kernel, backward_kernel = _get_kernels(vectorized=vectorized)
    d_g_shape = (lambda b, t: (b, t, Hv, Dk)) if vectorized else (lambda b, t: (b, t, Hv))

    @mx.custom_function
    def _chunk(q_c, k_c, v_c, g_c, beta_c, state_in):
        return gated_delta_kernel(q_c, k_c, v_c, g_c, beta_c, state_in)

    @_chunk.vjp
    def _chunk_vjp(primals, cotangents, outputs):
        q_c, k_c, v_c, g_c, beta_c, state_in = primals
        dy, d_state_out = cotangents
        Bc, Tc = q_c.shape[:2]

        (states,) = states_kernel(
            inputs=[k_c, v_c, g_c, beta_c, state_in, Tc],
            template=[("Dk", Dk), ("Dv", Dv), ("Hv", Hv)],
            grid=(32, Dv, Bc * Hv),
            threadgroup=(_KERNEL_THREADGROUP_X, _KERNEL_THREADGROUP_Y, 1),
            output_shapes=[(Bc, Hv, Tc, Dv, Dk)],
            output_dtypes=[mx.float32],
        )
        d_q, d_k, d_v, d_g, d_beta, d_state_in = backward_kernel(
            inputs=[
                q_c, k_c, v_c, g_c, beta_c, state_in, states,
                dy, d_state_out, Tc,
            ],
            template=[
                ("Dk", Dk),
                ("Dv", Dv),
                ("Hv", Hv),
                ("TG_ROWS", _KERNEL_THREADGROUP_Y),
            ],
            grid=(32, Dv, Bc * Hv),
            threadgroup=(_KERNEL_THREADGROUP_X, _KERNEL_THREADGROUP_Y, 1),
            output_shapes=[
                (Bc, Tc, Hv, Dk),
                (Bc, Tc, Hv, Dk),
                (Bc, Tc, Hv, Dv),
                d_g_shape(Bc, Tc),
                (Bc, Tc, Hv),
                (Bc, Hv, Dv, Dk),
            ],
            output_dtypes=[mx.float32] * 6,
            init_value=0,
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
    s = state
    for t_start in range(0, T, _KERNEL_CHUNK_SIZE):
        t_end = min(t_start + _KERNEL_CHUNK_SIZE, T)
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
    return y, s
