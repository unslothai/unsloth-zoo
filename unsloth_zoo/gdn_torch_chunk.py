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

"""fla's chunk_gated_delta_rule in torch, for GPUs where Triton has no tensor-core dot (sm75, Triton >= 3.3).

The chunk-parallel parts (decays, the UT / WY triangular solve, intra-chunk attention, outputs) are batched
matmuls over every chunk at once (cuBLAS tensor cores in float16). Only the inter-chunk state recurrence is
sequential: one opaque custom op with a hand-written backward, so torch.compile never unrolls it (no recompile
per sequence length)."""

__all__ = ["torch_chunk_gated_delta_rule", "scan_ops"]

import torch

CHUNK = 64
_OPS = {}
# Engagement: scan forwards / backwards actually run (inside the op, so compiled callers count too).
CALLS = {"scan": 0, "scan_bwd": 0}


def _bmm_add(c, a, b, alpha = 1.0):
    """c + alpha * a @ b over the leading [B, H] dims, one cuBLAS call (float32)."""
    return torch.baddbmm(c.flatten(0, 1), a.flatten(0, 1), b.flatten(0, 1), alpha = alpha).view(c.shape)


def _scan_fwd_impl(w, u, kd, d, s0):
    """w [B,H,N,C,K], u [B,H,N,C,V], kd [B,H,N,K,C], d [B,H,N] (per-chunk decay), s0 [B,H,K,V]; float32.
    states[i] = S entering chunk i; vn[i] = u_i - w_i S_i; S_{i+1} = d_i S_i + kd_i vn_i."""
    CALLS["scan"] += 1
    n = w.shape[2]
    s = s0
    states, vns = [], []
    ws, us, kds, ds = w.unbind(2), u.unbind(2), kd.unbind(2), d.unbind(2)
    for i in range(n):
        states.append(s)
        vn = _bmm_add(us[i], ws[i], s, alpha = -1.0)
        vns.append(vn)
        s = _bmm_add(s * ds[i][..., None, None], kds[i], vn)
    return torch.stack(states, 2), torch.stack(vns, 2), s


def _scan_bwd_impl(g_states, g_vn, g_final, w, kd, d, states, vn):
    """Reverse of _scan_fwd_impl. G = grad of the state leaving chunk i:
    g_vn_i = dvn_i + kd_i^T G; G_i = dS_i + d_i G - w_i^T g_vn_i; then g_w, g_kd, g_d over all chunks at once."""
    CALLS["scan_bwd"] += 1
    n = w.shape[2]
    G = g_final
    ws, kds, ds = w.unbind(2), kd.unbind(2), d.unbind(2)
    gss, gvs = g_states.unbind(2), g_vn.unbind(2)
    g_vns, g_next = [None] * n, [None] * n
    for i in range(n - 1, -1, -1):
        gv = _bmm_add(gvs[i], kds[i].transpose(-1, -2), G)
        g_vns[i], g_next[i] = gv, G
        G = _bmm_add(gss[i] + ds[i][..., None, None] * G, ws[i].transpose(-1, -2), gv, alpha = -1.0)
    g_vn_all = torch.stack(g_vns, 2)
    g_next_all = torch.stack(g_next, 2)
    g_w = -torch.matmul(g_vn_all, states.transpose(-1, -2))
    g_kd = torch.matmul(g_next_all, vn.transpose(-1, -2))
    g_d = (g_next_all * states).sum((-1, -2))
    return g_w, g_vn_all, g_kd, g_d, G


def scan_ops():
    """(scan, scan_bwd) custom ops, registered once; None when this torch has no torch.library.custom_op."""
    if "ops" in _OPS:
        return _OPS["ops"]
    _OPS["ops"] = None
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:
        return None
    try:
        @custom_op("unsloth_zoo::gdn_chunk_scan", mutates_args = ())
        def scan(w: torch.Tensor, u: torch.Tensor, kd: torch.Tensor, d: torch.Tensor, s0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return _scan_fwd_impl(w, u, kd, d, s0)

        @scan.register_fake
        def _(w, u, kd, d, s0):
            B, H, N, C, K = w.shape
            V = u.shape[-1]
            return s0.new_empty(B, H, N, K, V), u.new_empty(B, H, N, C, V), s0.new_empty(B, H, K, V)

        @custom_op("unsloth_zoo::gdn_chunk_scan_bwd", mutates_args = ())
        def scan_bwd(
            g_states: torch.Tensor, g_vn: torch.Tensor, g_final: torch.Tensor, w: torch.Tensor, kd: torch.Tensor,
            d: torch.Tensor, states: torch.Tensor, vn: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return _scan_bwd_impl(g_states, g_vn, g_final, w, kd, d, states, vn)

        @scan_bwd.register_fake
        def _(g_states, g_vn, g_final, w, kd, d, states, vn):
            return torch.empty_like(w), torch.empty_like(g_vn), torch.empty_like(kd), torch.empty_like(d), torch.empty_like(g_final)

        def setup_context(ctx, inputs, output):
            w, u, kd, d, s0 = inputs
            states, vn, _ = output
            ctx.save_for_backward(w, kd, d, states, vn)

        def backward(ctx, g_states, g_vn, g_final):
            w, kd, d, states, vn = ctx.saved_tensors
            # Unused outputs arrive as None; the op wants tensors.
            if g_states is None:
                g_states = torch.zeros_like(states)
            if g_vn is None:
                g_vn = torch.zeros_like(vn)
            if g_final is None:
                g_final = torch.zeros_like(states[:, :, 0])
            g_w, g_u, g_kd, g_d, g_s0 = scan_bwd(
                g_states.contiguous(), g_vn.contiguous(), g_final.contiguous(), w, kd, d, states, vn,
            )
            return g_w, g_u, g_kd, g_d, g_s0

        scan.register_autograd(backward, setup_context = setup_context)
    except Exception:
        return None
    _OPS["ops"] = (scan, scan_bwd)
    return _OPS["ops"]


# Registered at import: registering inside a torch.compile trace fails (infer_schema is not traceable).
scan_ops()


def _chunked(q, k, v, g, beta, scale, initial_state, output_final_state, mm_dtype, seg = None):
    """q/k [B,T,H,K] (l2-normalised if asked, H == HV), v [B,T,H,V], g/beta [B,T,H]; float32 o
    [B,T,H,V], state [B,H,K,V] or None; seg [B,T] packs docs (B=1), masks state; no host sync."""
    scan, _ = scan_ops()
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = CHUNK
    N = (T + C - 1) // C
    pad = N * C - T
    # float32 math (float64 inputs keep float64: the oracle tests run the same code exactly).
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v = (x.transpose(1, 2).to(acc) for x in (q, k, v))
    beta, g = beta.transpose(1, 2).to(acc), g.transpose(1, 2).to(acc)
    if pad:
        # Zeros concatenated and the pad as N * C - T, not F.pad / (-T) % C: torch 2.4's dynamic shapes fail on the modulo.
        q, k, v = (torch.cat([x, x.new_zeros(B, H, pad, x.shape[-1])], dim = 2) for x in (q, k, v))
        beta, g = (torch.cat([x, x.new_zeros(B, H, pad)], dim = 2) for x in (beta, g))
    q = (q * scale).reshape(B, H, N, C, K)
    k = k.reshape(B, H, N, C, K)
    v = v.reshape(B, H, N, C, V)
    beta = beta.reshape(B, H, N, C, 1)
    cum = g.reshape(B, H, N, C).cumsum(-1)
    blocked = torch.ones(C, C, dtype = torch.bool, device = q.device).triu(1)
    if seg is not None:
        if pad:
            # Padding is a document of its own.
            seg = torch.cat([seg, seg.new_full((B, pad), -2)], dim = 1)
        seg = seg.reshape(B, 1, N, C)
        blocked = blocked | (seg.unsqueeze(-1) != seg.unsqueeze(-2))
        # The state entering chunk n belongs to the document of the token before it (none before chunk 0).
        before = torch.cat([seg.new_full((B, 1, 1), -1), seg[..., :-1, -1]], dim = 2)
        reads_state = (seg == before.unsqueeze(-1)).to(acc)  # token continues the carried document
        in_last = (seg == seg[..., -1:]).to(acc)              # token belongs to the document the chunk hands on
        keeps_state = reads_state[..., -1]                    # that document is the carried one
    decay = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).masked_fill(blocked, float("-inf")).exp()

    mm_ = mm_dtype if acc == torch.float32 else acc

    def mm(a, b):
        return torch.matmul(a.to(mm_), b.to(mm_)).to(acc)

    k_beta = k * beta
    kt = k.transpose(-1, -2)
    system = mm(k_beta, kt) * decay
    attn = mm(q, kt) * decay
    read = cum.exp()
    kd_scale = (cum[..., -1:] - cum).exp()
    d = cum[..., -1].exp()
    if seg is not None:
        read = read * reads_state
        kd_scale = kd_scale * in_last
        d = d * keeps_state
    # (I + strictly_lower(system)) [u | w] = [v beta | k beta exp(cum)]: the chunk's UT / WY transform.
    rhs = torch.cat([v * beta, k_beta * read.unsqueeze(-1)], dim = -1)
    sol = torch.linalg.solve_triangular(system, rhs, upper = False, unitriangular = True)
    u, w = sol[..., :V], sol[..., V:]
    kd = (k * kd_scale.unsqueeze(-1)).transpose(-1, -2)
    s0 = q.new_zeros(B, H, K, V) if initial_state is None else initial_state.to(acc)
    states, vn, final = scan(w.contiguous(), u.contiguous(), kd.contiguous(), d.contiguous(), s0.contiguous())
    # Outputs of every chunk at once: read of the state entering the chunk + intra-chunk attention.
    o = mm(q * read.unsqueeze(-1), states) + mm(attn, vn)
    o = o.reshape(B, H, N * C, V)[:, :, :T].transpose(1, 2)
    return o, (final if output_final_state else None)


def torch_chunk_gated_delta_rule(
    q, k, v, g, beta, scale = None, initial_state = None, output_final_state = False,
    use_qk_l2norm_in_kernel = False, cu_seqlens = None, mm_dtype = None,
):
    """fla's chunk_gated_delta_rule contract (g is the per-token log decay, q / k may have fewer heads than v,
    initial_state [B, HV, K, V]); returns (o in q's dtype, final state float32 or None). Packed documents
    (``cu_seqlens``, B = 1) take neither an initial state nor a final state: fla's per-document states."""
    if cu_seqlens is not None and (q.shape[0] != 1 or initial_state is not None or output_final_state):
        raise ValueError("Unsloth: packed gated delta (cu_seqlens) runs with B = 1 and no initial / final state.")
    out_dtype = q.dtype
    if mm_dtype is None:
        # float16 inputs use float16 matmuls; others float32 (bf16 has no sm75 tensor cores).
        mm_dtype = torch.float16 if q.dtype == torch.float16 else torch.float32
    groups = v.shape[2] // q.shape[2]
    if groups > 1:
        q = q.repeat_interleave(groups, dim = 2)
        k = k.repeat_interleave(groups, dim = 2)
    if use_qk_l2norm_in_kernel:
        # fla's l2norm: x * rsqrt(sum(x^2) + 1e-6), reduced in float32.
        acc = torch.float64 if q.dtype == torch.float64 else torch.float32
        q = q.to(acc)
        k = k.to(acc)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim = True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim = True) + 1e-6)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    with torch.autocast(device_type = q.device.type, enabled = False):
        seg = None
        if cu_seqlens is not None:
            positions = torch.arange(q.shape[1], device = q.device)
            seg = torch.bucketize(positions, cu_seqlens[1:].to(device = q.device, dtype = positions.dtype), right = True)[None]
        o, final = _chunked(q, k, v, g, beta, scale, initial_state, output_final_state, mm_dtype, seg)
    return o.to(out_dtype), final
