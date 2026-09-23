# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Triton kernels for the MoE expert path, each with the eager path as fallback.

nf4_dequant_triton
    NF4 (bitsandbytes, blocksize 64, optionally double-quantized absmax) to the
    quant_state dtype, bit-identical to bitsandbytes.functional.dequantize_4bit:
    codebook value times the fp32 absmax, rounded once. It writes each output
    pair contiguously, which is where bitsandbytes' own kernel loses half its
    bandwidth on an expert stack (measured 0.26 ms vs 0.49 ms for a
    (128, 1408, 2816) bf16 stack on a B200, both bit-identical).

weighted_unpermute
    The MoE combine: out[t] = sum_k w[slot(t, k)] * Y[slot(t, k)] accumulated in
    fp32 and rounded once to out_dtype, with a matching backward, no atomics and
    a fixed reduction order. Replaces the routing-weight multiply of the whole
    permuted output, the gather through the inverse permutation, the sum over
    top_k and the cast (four full passes over a [tokens * top_k, hidden] buffer,
    plus the same again in backward).

Both kernels are launched with enable_fp_fusion=False: the eager paths they
replace materialise every product before adding it (two roundings), and a
contracted fma would round once and drift. That flag is honoured by the CUDA and
the HIP backend alike, unlike libdevice's mul_rn/add_rn which the HIP backend
does not ship, so the kernels compile on ROCm too.

Both are used only when a CUDA/ROCm device and Triton are available and
UNSLOTH_MOE_TRITON_KERNELS is not "0"; every caller keeps its eager path. If a
launch ever fails (a Triton build without a working compiler, an unsupported
backend), the kernels switch themselves off for the process with one warning
and the caller falls back, rather than taking the training run down.
"""
import os
import logging
import torch

__all__ = [
    "moe_triton_kernels_available",
    "nf4_dequant_triton",
    "weighted_unpermute",
]

logger = logging.getLogger(__name__)

_TRITON = None          # None: not probed yet; True/False: importable on a CUDA/ROCm device
_DISABLED_REASON = None # set on the first failed launch; everything falls back afterwards
_K = None               # compiled kernels, built on first use


def moe_triton_kernels_available(device = None) -> bool:
    """Triton importable, an accelerator Triton can target, not disabled, and no
    launch has failed in this process."""
    global _TRITON
    if _DISABLED_REASON is not None or os.environ.get("UNSLOTH_MOE_TRITON_KERNELS", "1") == "0":
        return False
    if device is not None and getattr(device, "type", None) != "cuda":
        return False
    if _TRITON is None:
        try:
            import triton  # noqa: F401
            import triton.language  # noqa: F401
            _TRITON = torch.cuda.is_available()
        except Exception:
            _TRITON = False
    return _TRITON


def _disable(where, exc):
    """Switch the kernels off for the rest of the process after a failed launch."""
    global _DISABLED_REASON
    _DISABLED_REASON = f"{where}: {type(exc).__name__}: {exc}"
    logger.warning(
        "Unsloth: the Triton MoE kernels failed to compile or launch and are disabled "
        "for this process; the bitsandbytes / eager path is used instead. "
        f"Set UNSLOTH_MOE_TRITON_KERNELS=0 to silence this. Reason: {_DISABLED_REASON}"
    )


def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _nf4_dequant_kernel(
        q_ptr, out_ptr, code_ptr,
        absmax_ptr, code2_ptr, absmax2_ptr, offset_ptr,
        n_bytes,
        NESTED: tl.constexpr, BLOCKSIZE2: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < n_bytes
        qw = tl.load(q_ptr + offs, mask = mask, other = 0)
        blk = offs // 32                      # 64 elements = 32 bytes per absmax block
        if NESTED:
            aq = tl.load(absmax_ptr + blk, mask = mask, other = 0).to(tl.int32)
            am2 = tl.load(absmax2_ptr + blk // BLOCKSIZE2, mask = mask, other = 0.0).to(tl.float32)
            # bitsandbytes dequantizes the absmax as a blockwise multiply and then
            # adds the offset: two roundings (fp fusion is off for this kernel).
            am = tl.load(code2_ptr + aq) * am2 + tl.load(offset_ptr).to(tl.float32)
        else:
            am = tl.load(absmax_ptr + blk, mask = mask, other = 0.0).to(tl.float32)
        hi = (qw >> 4).to(tl.int32)
        lo = (qw & 15).to(tl.int32)
        vh = (tl.load(code_ptr + hi) * am).to(out_ptr.dtype.element_ty)
        vl = (tl.load(code_ptr + lo) * am).to(out_ptr.dtype.element_ty)
        w = tl.reshape(tl.join(vh, vl), (2 * BLOCK,))
        offs2 = pid * (2 * BLOCK) + tl.arange(0, 2 * BLOCK).to(tl.int64)
        tl.store(out_ptr + offs2, w, mask = offs2 < 2 * n_bytes)

    @triton.jit
    def _combine_fwd_kernel(y_ptr, inv_ptr, w_ptr, out_ptr, T, H: tl.constexpr, TOPK: tl.constexpr,
                            BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr, ROUND_PRODUCT: tl.constexpr):
        # A tile of BLOCK_T tokens by BLOCK_H hidden columns per program: the
        # per-token loads then coalesce across the tile instead of one token
        # per program (measured 0.074 -> 0.04 ms on 2048 x 8 x 2816).
        pid_t = tl.program_id(0).to(tl.int64)
        hb = tl.program_id(1)
        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
        t_mask = offs_t < T
        offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = offs_h < H
        # torch.sum over the top_k axis (the eager path) accumulates with four
        # interleaved partial sums, acc[k % 4] += p_k, then ((a0 + a1) + a2) + a3.
        # Same order here, so the fp32 result and its single rounding are
        # bit-identical to the eager combine for any top_k.
        acc0 = tl.zeros((BLOCK_T, BLOCK_H), dtype = tl.float32)
        acc1 = tl.zeros((BLOCK_T, BLOCK_H), dtype = tl.float32)
        acc2 = tl.zeros((BLOCK_T, BLOCK_H), dtype = tl.float32)
        acc3 = tl.zeros((BLOCK_T, BLOCK_H), dtype = tl.float32)
        for k in tl.static_range(TOPK):
            slot = tl.load(inv_ptr + offs_t * TOPK + k, mask = t_mask, other = 0).to(tl.int64)
            wk = tl.load(w_ptr + slot, mask = t_mask, other = 0.0).to(tl.float32)
            y = tl.load(y_ptr + slot[:, None] * H + offs_h[None, :], mask = t_mask[:, None] & h_mask[None, :], other = 0.0)
            p = wk[:, None] * y.to(tl.float32)
            if ROUND_PRODUCT:
                # A low-precision router multiplies in the activation dtype in
                # the eager path; match that rounding, then sum in fp32.
                p = p.to(y_ptr.dtype.element_ty).to(tl.float32)
            if k % 4 == 0:
                acc0 += p
            elif k % 4 == 1:
                acc1 += p
            elif k % 4 == 2:
                acc2 += p
            else:
                acc3 += p
        acc = ((acc0 + acc1) + acc2) + acc3
        tl.store(out_ptr + offs_t[:, None] * H + offs_h[None, :], acc.to(out_ptr.dtype.element_ty),
                 mask = t_mask[:, None] & h_mask[None, :])

    @triton.jit
    def _combine_bwd_dy_kernel(dout_ptr, sorted_ptr, w_ptr, dy_ptr, N, H: tl.constexpr, TOPK: tl.constexpr,
                               BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        hb = tl.program_id(1)
        offs_i = pid * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
        i_mask = offs_i < N
        offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = offs_h < H
        t = tl.load(sorted_ptr + offs_i, mask = i_mask, other = 0).to(tl.int64) // TOPK
        wk = tl.load(w_ptr + offs_i, mask = i_mask, other = 0.0).to(tl.float32)
        g = tl.load(dout_ptr + t[:, None] * H + offs_h[None, :], mask = i_mask[:, None] & h_mask[None, :], other = 0.0).to(tl.float32)
        tl.store(dy_ptr + offs_i[:, None] * H + offs_h[None, :], (wk[:, None] * g).to(dy_ptr.dtype.element_ty),
                 mask = i_mask[:, None] & h_mask[None, :])

    @triton.jit
    def _combine_bwd_dw_kernel(y_ptr, dout_ptr, sorted_ptr, dw_ptr, H: tl.constexpr, TOPK: tl.constexpr, BLOCK_H: tl.constexpr,
                               ROUND_PRODUCT: tl.constexpr):
        i = tl.program_id(0).to(tl.int64)
        t = tl.load(sorted_ptr + i).to(tl.int64) // TOPK
        acc = tl.zeros((BLOCK_H,), dtype = tl.float32)
        for hb in range(0, tl.cdiv(H, BLOCK_H)):
            offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
            h_mask = offs_h < H
            y = tl.load(y_ptr + i * H + offs_h, mask = h_mask, other = 0.0)
            g = tl.load(dout_ptr + t * H + offs_h, mask = h_mask, other = 0.0)
            p = y.to(tl.float32) * g.to(tl.float32)
            if ROUND_PRODUCT:
                # The eager mul-backward forms grad * Y in the activation
                # dtype before its fp32-accumulated reduction; match it.
                p = p.to(y_ptr.dtype.element_ty).to(tl.float32)
            acc += p
        tl.store(dw_ptr + i, tl.sum(acc, axis = 0))

    return triton, _nf4_dequant_kernel, _combine_fwd_kernel, _combine_bwd_dy_kernel, _combine_bwd_dw_kernel


def _get_kernels():
    global _K
    if _K is None:
        _K = _kernels()
    return _K


def _build_kernels_eagerly():
    """Define the @triton.jit functions once, outside any compiled region.

    Defining them is cheap (no compilation happens until the first launch), but
    doing it lazily from inside an autograd Function meant the first call under
    torch.compile(fullgraph=True) traced the decorator itself, which Dynamo
    cannot do. Built here at import when Triton is importable; anything that
    fails just leaves the lazy path, and a launch failure still disables the
    kernels through _disable.
    """
    global _K
    if _K is not None:
        return
    try:
        import triton  # noqa: F401
    except Exception:
        return
    try:
        _K = _kernels()
    except Exception:
        _K = None


_build_kernels_eagerly()


# ---------------------------------------------------------------------------
# NF4 dequant
# ---------------------------------------------------------------------------

def _on_device(device, *tensors):
    return all(isinstance(t, torch.Tensor) and t.device == device for t in tensors)


def nf4_dequant_triton(packed, quant_state, out_shape = None):
    """Dequantize bitsandbytes NF4 `packed` (any shape; any quant_storage dtype,
    the bytes are viewed as uint8) with `quant_state` to a tensor of `out_shape`
    (default quant_state.shape) in quant_state.dtype. Returns None when this is
    not a state the kernel handles (not NF4, blocksize other than 64, a padded
    odd element count, state tensors on another device), or when the Triton
    path is unavailable, so the caller keeps the bitsandbytes path."""
    if not moe_triton_kernels_available(packed.device):
        return None
    if (getattr(quant_state, "quant_type", None) != "nf4" or quant_state.blocksize != 64
            or quant_state.code.numel() != 16):
        return None
    shape = tuple(out_shape) if out_shape is not None else tuple(quant_state.shape)
    numel = 1
    for s in shape:
        numel *= s
    q = packed.contiguous().reshape(-1)
    if q.dtype != torch.uint8:
        q = q.view(torch.uint8)
    n_bytes = q.numel()
    if numel != 2 * n_bytes:
        return None
    device = packed.device
    code = quant_state.code
    absmax = quant_state.absmax
    nested = bool(quant_state.nested)
    if nested:
        s2 = quant_state.state2
        code2, absmax2 = s2.code, s2.absmax
        offset = quant_state.offset
        if not isinstance(offset, torch.Tensor):   # older bitsandbytes kept a python float
            offset = torch.tensor(float(offset), dtype = torch.float32, device = device)
        offset = offset.reshape(1)
        if not _on_device(device, code, absmax, code2, absmax2, offset):
            return None
    else:
        if not _on_device(device, code, absmax):
            return None
        code2 = absmax2 = offset = absmax   # unused pointers, must still be valid
    if code.dtype != torch.float32:
        code = code.float()
    if nested and code2.dtype != torch.float32:
        code2 = code2.float()
    BLOCK = 1024
    try:
        triton, kernel, _, _, _ = _get_kernels()
        out = torch.empty(numel, dtype = quant_state.dtype, device = device)
        kernel[(triton.cdiv(n_bytes, BLOCK),)](
            q, out, code, absmax, code2, absmax2, offset, n_bytes,
            NESTED = nested, BLOCKSIZE2 = (quant_state.state2.blocksize if nested else 1),
            BLOCK = BLOCK, num_warps = 4, enable_fp_fusion = False,
        )
    except Exception as exc:
        _disable("nf4_dequant_triton", exc)
        return None
    return out.view(shape)


# ---------------------------------------------------------------------------
# Weighted unpermute (combine)
# ---------------------------------------------------------------------------

def _round_product(w, y) -> bool:
    """The eager path forms w * y in torch's promoted dtype; it rounds each
    product only when that dtype is narrower than fp32 (then it equals y's)."""
    return torch.promote_types(w.dtype, y.dtype) != torch.float32


class _WeightedUnpermute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, sorted_indices, w_perm, num_tokens, top_k, out_dtype):
        triton, _, fwd, _, _ = _get_kernels()
        H = y.shape[1]
        inv = torch.empty_like(sorted_indices)
        inv.scatter_(0, sorted_indices, torch.arange(sorted_indices.numel(), device = y.device, dtype = sorted_indices.dtype))
        out = torch.empty((num_tokens, H), device = y.device, dtype = out_dtype or y.dtype)
        BLOCK_T, BLOCK_H = 8, 512
        fwd[(triton.cdiv(num_tokens, BLOCK_T), triton.cdiv(H, BLOCK_H))](
            y, inv, w_perm, out, num_tokens, H, top_k, BLOCK_T, BLOCK_H,
            ROUND_PRODUCT = _round_product(w_perm, y), num_warps = 4, enable_fp_fusion = False,
        )
        # y is only read by the routing-weight gradient. When the router is
        # frozen or detached (the gate-grad identity path hands us a detached
        # weight on purpose, so that Y is not pinned on the tape), keep only
        # its shape; saving it unconditionally would undo that memory saving.
        ctx.needs_w = w_perm.requires_grad
        ctx.save_for_backward(*((y, sorted_indices, w_perm) if ctx.needs_w else (sorted_indices, w_perm)))
        ctx.y_meta = (y.shape, y.dtype, y.device)
        ctx.top_k = top_k
        ctx.round_product = _round_product(w_perm, y)
        return out

    @staticmethod
    def backward(ctx, dout):
        if ctx.needs_w:
            y, sorted_indices, w_perm = ctx.saved_tensors
        else:
            y = None
            sorted_indices, w_perm = ctx.saved_tensors
        (T, H), y_dtype, y_device = ctx.y_meta
        top_k = ctx.top_k
        dout = dout.contiguous()
        # The backward kernels compile on first use, inside autograd and outside the forward's
        # fallback, so a failure here also switches the path off and finishes in eager.
        if moe_triton_kernels_available(dout.device):
            try:
                triton, _, _, bwd_dy, bwd_dw = _get_kernels()
                BLOCK_T, BLOCK_H = 16, 512
                dy = torch.empty((T, H), dtype = y_dtype, device = y_device)
                bwd_dy[(triton.cdiv(T, BLOCK_T), triton.cdiv(H, BLOCK_H))](
                    dout, sorted_indices, w_perm, dy, T, H, top_k, BLOCK_T, BLOCK_H,
                    num_warps = 4, enable_fp_fusion = False)
                dw = None
                if ctx.needs_w:
                    dw32 = torch.empty((T,), device = y_device, dtype = torch.float32)
                    bwd_dw[(T,)](y, dout, sorted_indices, dw32, H, top_k, 1024,
                                 ROUND_PRODUCT = ctx.round_product, enable_fp_fusion = False)
                    dw = dw32.to(w_perm.dtype)
                return dy, None, dw, None, None, None
            except Exception as exc:
                _disable("weighted_unpermute backward", exc)
        return _weighted_unpermute_backward_eager(ctx, dout, y, sorted_indices, w_perm, y_dtype, top_k)


def _weighted_unpermute_backward_eager(ctx, dout, y, sorted_indices, w_perm, y_dtype, top_k):
    # Slot i of the permuted rows belongs to token sorted_indices[i] // top_k.
    token_rows = dout.float().index_select(0, torch.div(sorted_indices, top_k, rounding_mode = "floor"))
    dy = (token_rows * w_perm.float().unsqueeze(-1)).to(y_dtype)
    dw = None
    if ctx.needs_w:
        dw = (y.float() * token_rows).sum(-1).to(w_perm.dtype)
    return dy, None, dw, None, None, None


def weighted_unpermute(permuted_output, sorted_indices, permuted_weights, num_tokens, top_k, out_dtype = None):
    """out[t] = sum over the top_k routed slots i of token t of
    permuted_weights[i] * permuted_output[i], fp32 accumulate, one rounding.
    permuted_output: (num_tokens * top_k, hidden) in expert-sorted order;
    sorted_indices: the argsort that produced that order; permuted_weights:
    (num_tokens * top_k,) routing weights in the same order. Returns None when
    the Triton path is unavailable so the caller keeps the eager reduction."""
    if not moe_triton_kernels_available(permuted_output.device):
        return None
    if permuted_output.dim() != 2 or sorted_indices.numel() != permuted_output.shape[0]:
        return None
    try:
        return _WeightedUnpermute.apply(
            permuted_output.contiguous(), sorted_indices.contiguous(),
            permuted_weights.reshape(-1).contiguous(), int(num_tokens), int(top_k), out_dtype,
        )
    except Exception as exc:
        _disable("weighted_unpermute", exc)
        return None
