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
    def _chunked_forward(q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk, state_in):
        """Process one chunk of timesteps."""
        chunk_T = q_chunk.shape[1]
        ys = []
        s = state_in
        for t in range(chunk_T):
            m = None if mask is None else mask[:, t:t+1].squeeze(1)
            y, s = _gated_delta_step(
                q_chunk[:, t], k_chunk[:, t], v_chunk[:, t],
                g_chunk[:, t], beta_chunk[:, t], s, m,
            )
            ys.append(y)
        return mx.stack(ys, axis=1), s

    @_chunked_forward.vjp
    def _chunked_vjp(primals, cotangents, outputs):
        q_c, k_c, v_c, g_c, beta_c, state_in = primals
        dy, d_state_out = cotangents
        chunk_T = q_c.shape[1]

        # Recompute forward states for this chunk
        states = [state_in]
        s = state_in
        for t in range(chunk_T):
            m = None if mask is None else mask[:, t:t+1].squeeze(1)
            _, s = _gated_delta_step(
                q_c[:, t], k_c[:, t], v_c[:, t],
                g_c[:, t], beta_c[:, t], s, m,
            )
            states.append(s)

        # BPTT: backward through time
        d_q = mx.zeros_like(q_c)
        d_k = mx.zeros_like(k_c)
        d_v = mx.zeros_like(v_c)
        d_g = mx.zeros_like(g_c)
        d_beta = mx.zeros_like(beta_c)
        d_state = d_state_out

        for t in range(chunk_T - 1, -1, -1):
            state_prev = states[t]
            state_cur = states[t + 1]
            q_t = q_c[:, t]
            k_t = k_c[:, t]
            v_t = v_c[:, t]
            g_t = g_c[:, t]
            beta_t = beta_c[:, t]
            dy_t = dy[:, t]

            # y = (state_cur * q_t[..., None, :]).sum(-1)
            # d_state from y: d_state += dy_t[..., None] * q_t[..., None, :]
            d_state = d_state + dy_t[..., None].astype(mx.float32) * q_t[..., None, :].astype(mx.float32)
            # d_q from y: d_q_t = (state_cur * dy_t[..., None]).sum(-2)... no
            # y = sum over Dk: state[..., dv, dk] * q[..., dk]  → y[..., dv]
            # dy/dq = state summed over dv? No.
            # y[b,h,dv] = sum_dk state[b,h,dv,dk] * q[b,h,dk]
            # dy/dq[b,h,dk] = sum_dv dy[b,h,dv] * state[b,h,dv,dk]
            d_q_t = (dy_t[..., None].astype(mx.float32) * state_cur).sum(axis=-2)
            d_q = d_q.at[:, t].add(d_q_t.astype(d_q.dtype))

            # state_cur = state_decayed + k * delta[..., None]
            # state_decayed = state_prev * decay
            # delta = (v - kv_mem) * beta[..., None]
            # kv_mem = (state_decayed * k[..., None, :]).sum(-1)

            if g_t.ndim == 2:
                decay = g_t[..., None, None]
            else:
                decay = g_t[..., None, :]

            state_decayed = state_prev * decay
            kv_mem = (state_decayed * k_t[..., None, :]).sum(axis=-1)
            delta = (v_t - kv_mem) * beta_t[..., None]

            # d_state → d_state_decayed + d from k*delta term
            # state_cur = state_decayed + k[...,None,:] * delta[...,None]
            d_kd = d_state  # gradient through the sum
            d_state_decayed = mx.array(d_state)

            # d_k from k[...,None,:] * delta[...,None]
            # shape: [B, H, Dv, Dk] = k[B,H,1,Dk] * delta[B,H,Dv,1]
            d_k_t_from_update = (d_kd * delta[..., None].astype(mx.float32)).sum(axis=-2)
            d_delta = (d_kd * k_t[..., None, :].astype(mx.float32)).sum(axis=-1)

            # delta = (v - kv_mem) * beta[..., None]
            d_v_minus_kv = d_delta * beta_t[..., None].astype(mx.float32)
            d_beta_t = (d_delta * (v_t.astype(mx.float32) - kv_mem)).sum(axis=-1)
            d_v_t = d_v_minus_kv
            d_kv_mem = -d_v_minus_kv

            # kv_mem = (state_decayed * k[..., None, :]).sum(-1)
            # d_state_decayed += d_kv_mem[..., None] * k[..., None, :]
            d_state_decayed = d_state_decayed + d_kv_mem[..., None].astype(mx.float32) * k_t[..., None, :].astype(mx.float32)
            d_k_t_from_kv = (d_kv_mem[..., None].astype(mx.float32) * state_decayed).sum(axis=-2)

            # state_decayed = state_prev * decay
            # d_state_prev = d_state_decayed * decay
            d_state = d_state_decayed * decay.astype(mx.float32)
            # d_decay = d_state_decayed * state_prev
            d_decay = (d_state_decayed * state_prev).sum(axis=-2)  # sum over Dv
            if g_t.ndim == 2:
                d_g_t = d_decay.sum(axis=-1)  # sum over Dk too
            else:
                d_g_t = d_decay  # [B, H, Dk]

            d_k_t = d_k_t_from_update + d_k_t_from_kv
            d_k = d_k.at[:, t].add(d_k_t.astype(d_k.dtype))
            d_v = d_v.at[:, t].add(d_v_t.astype(d_v.dtype))
            d_g = d_g.at[:, t].add(d_g_t.astype(d_g.dtype))
            d_beta = d_beta.at[:, t].add(d_beta_t.astype(d_beta.dtype))

            # Handle mask
            if mask is not None:
                m = mask[:, t]
                m_exp = mx.expand_dims(m, axis=(1, 2, 3))
                d_state = mx.where(m_exp, d_state, d_state + d_state_out)

        return d_q, d_k, d_v, d_g, d_beta, d_state

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
        chunk_y, s = _chunked_forward(q_c, k_c, v_c, g_c, beta_c, s)
        all_ys.append(chunk_y)

    y = mx.concatenate(all_ys, axis=1)
    return y, s


def patch_gated_delta():
    """Monkey-patch mlx_lm's gated_delta module to use our efficient VJP."""
    from mlx_lm.models import gated_delta

    original_ops = gated_delta.gated_delta_ops

    def patched_gated_delta_update(
        q, k, v, a, b, A_log, dt_bias,
        state=None, mask=None, use_kernel=True,
    ):
        beta = mx.sigmoid(b)
        g = gated_delta.compute_g(A_log, a, dt_bias)
        if state is None:
            B, _, Hk, Dk = q.shape
            Hv, Dv = v.shape[-2:]
            state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

        # Always use our efficient ops path for training (use_kernel=False)
        # and for inference when kernel isn't available
        if not use_kernel or mx.default_device() != mx.gpu or not mx.metal.is_available():
            return gated_delta_ops_efficient(q, k, v, g, beta, state, mask)
        return gated_delta.gated_delta_kernel(q, k, v, g, beta, state, mask)

    gated_delta.gated_delta_update = patched_gated_delta_update
    print("Unsloth: Patched GatedDeltaNet with memory-efficient custom VJP.")
