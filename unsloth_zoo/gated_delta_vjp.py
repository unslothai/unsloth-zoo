"""
Memory-efficient custom VJP for GatedDeltaNet (Qwen3.5).

Replaces the T-step Python loop in gated_delta_ops with a single
mx.custom_function that recomputes states during backward instead
of storing all T intermediate states in the autograd graph.

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

    Instead of a Python for-loop that creates T graph nodes (each holding
    state references), this wraps the entire recurrence in mx.custom_function.
    The backward recomputes states on-the-fly during BPTT.
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    # Chunk the sequence into segments for checkpointed BPTT.
    # Each chunk's forward is recomputed during backward.
    # Memory: O(num_chunks * state_size) instead of O(T * state_size)
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

        # Recompute the chunk's RETURNED states (post-mask). At step t the
        # returned state is `where(mask, state_new, state_prev)` — so for
        # masked steps `states[t+1] == states[t]`. We don't use these for
        # the y-path backward (that needs state_new, recomputed below); we
        # only need them as the entry point `state_prev` for each step.
        states = [state_in]
        s = state_in
        for t in range(chunk_T):
            m = mask_chunk[:, t:t+1].squeeze(1) if _has_mask else None
            _, s = _gated_delta_step(
                q_c[:, t], k_c[:, t], v_c[:, t],
                g_c[:, t], beta_c[:, t], s, m,
            )
            states.append(s)

        # BPTT: backward through time. `d_state` holds the cotangent w.r.t.
        # the RETURNED state at the current step's output (= input to step
        # t+1). At chunk boundary it's d_state_out; subsequent iterations
        # propagate it through the recurrence + mask.
        d_q = mx.zeros_like(q_c)
        d_k = mx.zeros_like(k_c)
        d_v = mx.zeros_like(v_c)
        d_g = mx.zeros_like(g_c)
        d_beta = mx.zeros_like(beta_c)
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

            # Recompute state_new (pre-mask). y always depends on state_new
            # regardless of mask; using states[t+1] would be wrong when
            # mask=False because states[t+1] equals state_prev there.
            if g_t.ndim == 2:
                decay = g_t[..., None, None]
            else:
                decay = g_t[..., None, :]
            state_decayed = state_prev * decay
            kv_mem = (state_decayed * k_t[..., None, :]).sum(axis=-1)
            delta = (v_t - kv_mem) * beta_t[..., None]
            state_new = state_decayed + k_t[..., None, :] * delta[..., None]

            # Forward had: state_returned = where(mask, state_new, state_prev).
            # Backward splits d_state_returned into:
            #   d_state_new path (recurrence backward): only when mask=True
            #   d_state_prev passthrough:               only when mask=False
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
            d_q = d_q.at[:, t].add(d_q_t.astype(d_q.dtype))

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
            d_k = d_k.at[:, t].add(d_k_t.astype(d_k.dtype))
            d_v = d_v.at[:, t].add(d_v_t.astype(d_v.dtype))
            d_g = d_g.at[:, t].add(d_g_t.astype(d_g.dtype))
            d_beta = d_beta.at[:, t].add(d_beta_t.astype(d_beta.dtype))

            # d_state_prev = recurrence-derived gradient + mask passthrough.
            d_state = d_state_prev_via_recurrence + d_state_prev_passthrough

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


def patch_gated_delta():
    """Monkey-patch mlx_lm's gated_delta module to use our efficient VJP."""
    from mlx_lm.models import gated_delta

    if getattr(gated_delta, "_unsloth_gated_delta_patched", False):
        return

    def patched_gated_delta_update(
        q, k, v, a, b, A_log, dt_bias,
        state=None, mask=None, use_kernel=True,
    ):
        # Heuristic: training calls enter with state=None (start of sequence,
        # no cache); inference with KV cache passes the cached state in.
        # Route training through gated_delta_ops_efficient so the
        # memory-efficient custom VJP actually runs — gated_delta_kernel
        # has no custom VJP, so going through it forces mlx to keep all T
        # intermediate states alive in the autograd graph and defeats the
        # whole point of this module.
        is_training_call = state is None
        beta = mx.sigmoid(b)
        g = gated_delta.compute_g(A_log, a, dt_bias)
        if state is None:
            B, _, Hk, Dk = q.shape
            Hv, Dv = v.shape[-2:]
            state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

        # Training (or any call without an incoming state cache): always
        # use the efficient ops path so backward is memory-efficient.
        if is_training_call:
            return gated_delta_ops_efficient(q, k, v, g, beta, state, mask)

        # Inference with cached state: prefer the kernel for speed when
        # available; fall back to efficient ops otherwise.
        if not use_kernel or mx.default_device() != mx.gpu or not mx.metal.is_available():
            return gated_delta_ops_efficient(q, k, v, g, beta, state, mask)
        return gated_delta.gated_delta_kernel(q, k, v, g, beta, state, mask)

    gated_delta.gated_delta_update = patched_gated_delta_update
    gated_delta._unsloth_gated_delta_patched = True
    print("Unsloth: Patched GatedDeltaNet with memory-efficient custom VJP.")
