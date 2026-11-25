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

"""W8A16 grouped GEMM over frozen block-FP8 routed experts.

Block-FP8 checkpoints (transformers FP8Experts: e4m3 codes ``(E, N, K)`` plus an fp32 ``weight_scale_inv`` grid
``(E, ceil(N / bn), ceil(K / bk))``) used to be dequantized to a whole bf16 stack on every experts call: the first
pass, the checkpoint replay, and the refill before backward. Here every weight tile is decoded in the GEMM main loop
as ``bf16(fp32(code) * scale)``, the value the whole-stack dequant writes, so the GEMM sees bit-identical weights and
no bf16 stack is ever allocated. Forward is ``x @ W[e]^T``, backward the input gradient ``g @ W[e]`` only (LoRA /
QLoRA: the base experts are frozen).

UNSLOTH_MOE_FP8_FUSED: ``auto`` (default: CUDA sm89+, not ROCm), ``0`` the dequant path. A build or launch failure
falls back to dequant + grouped GEMM for that shape, and a per-device self-check runs before the first use."""

__all__ = [
    "FP8_FUSED_STATS",
    "fp8_fused_enabled",
    "fp8_block_grouped_mm",
    "fp8_block_grouped_linear",
    "fp8_experts_as_packed",
    "fp8_block_base_and_lora",
]

import logging
import os

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

# Engagement counters (tests and benchmarks read them): forward / dX launches, replays served from a kept base,
# calls that fell back to dequantize + grouped GEMM, launches with the expert LoRA folded in.
FP8_FUSED_STATS = {"fwd": 0, "dx": 0, "kept": 0, "fallback": 0, "lora_fold": 0}
FAILED_SHAPES = {}

_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)


if triton is not None:

    @triton.jit
    def _fp8_block_grouped_kernel(
        A, W, S, C, COUNTS, A2, B2,
        E, N, K, K2, a2_scale,
        stride_am, stride_cm,
        stride_we, stride_wk, stride_wn,
        stride_se, stride_sk, stride_sn,
        stride_a2m, stride_b2e, stride_b2k, stride_b2n,
        E_POW2: tl.constexpr, BSK: tl.constexpr, BSN: tl.constexpr,  # num experts pow2 / scale block along K / N
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  # tile M / N / K (BSK % BLOCK_K == 0)
        GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,  # L2 row-tile group (0: 2D grid) / K % BLOCK_K == 0
        HAS_A2: tl.constexpr, BLOCK_K2: tl.constexpr,  # LoRA segment: C += a2_scale * A2 @ B2[e] / its K tile (K2 <= BLOCK_K2)
    ):
        # C[m, n] = sum_k A[m, k] * bf16(fp32(W[e, k, n]) * S[e, k // BSK, n // BSN]) for the rows m of expert e. W and
        # S are logical (E, K, N) views of the stored stacks (any strides), so forward and dX are the same kernel.
        # HAS_A2 folds a LoRA term into the same fp32 accumulator, as moe_lora_fold does for 16-bit stacks.
        if GROUP_M > 0:
            pid = tl.program_id(0)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_m = tl.num_programs(0) // num_pid_n
            width = GROUP_M * num_pid_n
            first_m = (pid // width) * GROUP_M
            size_m = tl.minimum(num_pid_m - first_m, GROUP_M)
            pid_m = first_m + (pid % width) % size_m
            pid_n = (pid % width) // size_m
        else:
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
        # Tile -> expert without a host sync: per-expert row tiles from the device counts, prefix-summed.
        e_offs = tl.arange(0, E_POW2)
        counts = tl.load(COUNTS + e_offs, mask = e_offs < E, other = 0).to(tl.int32)
        tiles = (counts + BLOCK_M - 1) // BLOCK_M
        tile_end = tl.cumsum(tiles, 0)
        e = tl.sum((tile_end <= pid_m).to(tl.int32), 0)
        if e >= E:
            return
        sel = e_offs == e
        row_end = tl.sum(tl.where(sel, tl.cumsum(counts, 0), 0), 0)
        row_start = row_end - tl.sum(tl.where(sel, counts, 0), 0)
        tile_start = tl.sum(tl.where(sel, tile_end - tiles, 0), 0)
        offs_m = row_start + (pid_m - tile_start) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < row_end
        offs_m64 = offs_m.to(tl.int64)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        e64 = e.to(tl.int64)
        w_base = W + e64 * stride_we
        # One scale row per K step (BLOCK_K divides BSK): the N blocks of this tile, as a [BLOCK_N] vector.
        s_base = S + e64 * stride_se + (offs_n // BSN).to(tl.int64) * stride_sn
        a_rows = A + offs_m64[:, None] * stride_am
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            if EVEN_K:
                a = tl.load(a_rows + offs_k[None, :], mask = mask_m[:, None], other = 0.0)
                q = tl.load(w_base + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                            mask = mask_n[None, :], other = 0.0)
            else:
                mask_k = offs_k < K
                a = tl.load(a_rows + offs_k[None, :], mask = mask_m[:, None] & mask_k[None, :], other = 0.0)
                q = tl.load(w_base + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                            mask = mask_k[:, None] & mask_n[None, :], other = 0.0)
            s = tl.load(s_base + (k0 // BSK) * stride_sk, mask = mask_n, other = 0.0)
            # One fp32 product rounded once to the activation dtype: the whole-stack dequant's value.
            b = (q.to(tl.float32) * s[None, :]).to(A.dtype.element_ty)
            acc = tl.dot(a, b, acc)
        if HAS_A2:
            offs_k2 = tl.arange(0, BLOCK_K2)
            mask_k2 = offs_k2 < K2
            a2 = tl.load(A2 + offs_m64[:, None] * stride_a2m + offs_k2[None, :],
                         mask = mask_m[:, None] & mask_k2[None, :], other = 0.0)
            b2 = tl.load(B2 + e64 * stride_b2e + offs_k2[:, None] * stride_b2k + offs_n[None, :] * stride_b2n,
                         mask = mask_k2[:, None] & mask_n[None, :], other = 0.0)
            a2 = (a2.to(tl.float32) * a2_scale).to(b2.dtype)
            acc = tl.dot(a2, b2, acc)
        c = C + offs_m64[:, None] * stride_cm + offs_n[None, :]
        tl.store(c, acc.to(C.dtype.element_ty), mask = mask_m[:, None] & mask_n[None, :])


def fp8_fused_mode() -> str:
    mode = os.environ.get("UNSLOTH_MOE_FP8_FUSED", "auto").strip().lower()
    if mode in ("0", "false", "off", "no", "dequant"):
        return "0"
    return "auto"


def _pow2(n):
    return 1 << max(int(n) - 1, 0).bit_length()


# Tiles by average rows per expert (fixed, never autotuned: configs that lower tl.dot differently round
# differently, so a timing-picked config would make the bits depend on the run). At most two per arch family.
_CONFIGS = {
    # (max rows per expert or None, (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages, GROUP_M))
    # L4 (sm89), same shapes: (64, 128, 128) is the best or within 2% of it at 512-8192 tokens of 9 candidates.
    "sm89": ((None, (64, 128, 128, 4, 3, 8)),),
    # RTX PRO 6000 (sm120), Qwen3-30B-A3B experts, fwd + dX: 512 tokens 1.04 ms, 2048 1.53, 4096 2.53, 8192 4.77
    # (dequant + grouped_mm GEMMs alone: 1.86 / 2.40 / 2.50 / 3.90, plus 1.35 ms per whole-stack dequant).
    "sm120": ((32, (32, 128, 128, 4, 3, 0)), (None, (64, 128, 64, 4, 3, 8))),
    "sm90": ((32, (32, 128, 128, 4, 4, 0)), (None, (128, 128, 64, 8, 4, 8))),
}
_CAPABILITY = {}


def _capability(device):
    cap = _CAPABILITY.get(device.index)
    if cap is None:
        index = device.index if device.index is not None else torch.cuda.current_device()
        cap = _CAPABILITY[device.index] = tuple(torch.cuda.get_device_capability(index))
    return cap


def _family(device):
    major = _capability(device)[0]
    if major == 12:
        return "sm120"
    if major >= 9:
        return "sm90"
    return "sm89"


def _config(M, E, N, K, BSK, device):
    rows = M / max(E, 1)
    table = _CONFIGS[_family(device)]
    cfg = table[-1][1]
    for hi, c in table:
        if hi is None or rows <= hi:
            cfg = c
            break
    BM, BN, BK, warps, stages, group_m = cfg
    BN = max(16, min(BN, _pow2(N)))
    BK = max(16, min(BK, _pow2(K), BSK))
    while BSK % BK:
        BK //= 2
    return BM, BN, BK, warps, stages, group_m


def _logical(w, s, transpose_b):
    """(N, K, strides of W and S as logical (E, K, N)) for the stored (E, R, C) stack: transpose_b reduces over C."""
    if transpose_b:
        return (w.shape[1], w.shape[2], (w.stride(0), w.stride(2), w.stride(1)), (s.stride(0), s.stride(2), s.stride(1)))
    return (w.shape[2], w.shape[1], (w.stride(0), w.stride(1), w.stride(2)), (s.stride(0), s.stride(1), s.stride(2)))


def _launch(x, w, s, counts, out, transpose_b, block, launcher, lora_in = None, lora_w = None, scale = 1.0):
    """out = x @ W[e]^T (transpose_b) or x @ W[e] (+ scale * lora_in @ lora_w[e]), rows sorted by expert;
    w (E, R, C) e4m3, s (E, R / br, C / bc) fp32, lora_in (M, r), lora_w logical (E, r, N), any strides."""
    E = w.shape[0]
    M = x.shape[0]
    N, K, sw, ss = _logical(w, s, transpose_b)
    br, bc = block
    BSK, BSN = (bc, br) if transpose_b else (br, bc)
    BM, BN, BK, warps, stages, group_m = _config(M, E, N, K, BSK, x.device)
    n_m = -(-M // BM) + torch.sym_min(E, M)
    n_n = -(-N // BN)
    grid = (n_m * n_n,) if group_m else (n_m, n_n)
    has_a2 = lora_in is not None
    K2 = lora_in.shape[1] if has_a2 else 0
    launcher(_fp8_block_grouped_kernel)[grid](
        x, w, s, out, counts, lora_in if has_a2 else x, lora_w if has_a2 else x,
        E, N, K, K2, float(scale),
        x.stride(0), out.stride(0),
        sw[0], sw[1], sw[2],
        ss[0], ss[1], ss[2],
        lora_in.stride(0) if has_a2 else 0, *(lora_w.stride() if has_a2 else (0, 0, 0)),
        E_POW2 = max(16, _pow2(E)), BSK = BSK, BSN = BSN,
        BLOCK_M = BM, BLOCK_N = BN, BLOCK_K = BK, GROUP_M = group_m, EVEN_K = K % BK == 0,
        HAS_A2 = has_a2, BLOCK_K2 = max(16, _pow2(K2)),
        num_warps = warps, num_stages = stages, enable_fp_fusion = True,
    )
    return out


def _out_cols(w, transpose_b):
    return w.shape[1] if transpose_b else w.shape[2]


def _dequant_stack(w, s, block, dtype):
    """Whole-stack decode in torch, chunked by expert: bf16(fp32(code) * scale), (E, R, C)."""
    E, R, C = w.shape
    br, bc = block
    out = torch.empty((E, R, C), dtype = dtype, device = w.device)
    step = max(1, (1 << 26) // max(R * C, 1))
    for e0 in range(0, E, step):
        e1 = min(E, e0 + step)
        scale = s[e0:e1].float().repeat_interleave(br, dim = 1)[:, :R].repeat_interleave(bc, dim = 2)[:, :, :C]
        out[e0:e1] = (w[e0:e1].float() * scale).to(dtype)
    return out


def _fallback(x, w, s, counts, transpose_b, block, lora_in = None, lora_w = None, scale = 1.0):
    """Dequantize + grouped GEMM: the path this kernel replaces."""
    from unsloth_zoo.temporary_patches.moe_utils import _grouped_mm_with_backward_fix
    dense = _dequant_stack(w, s, block, x.dtype)
    rhs = dense.transpose(-2, -1) if transpose_b else dense
    offsets = torch.cumsum(counts, dim = 0, dtype = torch.int32)
    with torch.no_grad():
        out = _grouped_mm_with_backward_fix(x.contiguous(), rhs, offsets)
        if lora_in is not None:
            out = out + _grouped_mm_with_backward_fix(lora_in.contiguous(), lora_w, offsets) * scale
        return out.contiguous()


def _passthrough_errors():
    try:
        from unsloth_zoo.temporary_patches.moe_triton_kernels import _autograd_control_flow_errors
        flow = _autograd_control_flow_errors()
    except ImportError:
        flow = ()
    return flow + (torch.cuda.OutOfMemoryError,)


def _grouped_eager(x, w, s, counts, transpose_b, block, lora_in = None, lora_w = None, scale = 1.0):
    x = x.contiguous()
    if lora_in is not None:
        lora_in = lora_in.contiguous()
    out = x.new_empty((x.shape[0], _out_cols(w, transpose_b)))
    if x.shape[0] == 0:
        return out
    key = (tuple(w.shape), tuple(s.shape), tuple(block), bool(transpose_b), x.dtype, str(x.device), lora_in is not None)
    if key in FAILED_SHAPES:
        FP8_FUSED_STATS["fallback"] += 1
        return _fallback(x, w, s, counts, transpose_b, block, lora_in, lora_w, scale)
    try:
        with torch.cuda.device(x.device):
            _launch(x, w, s, counts, out, transpose_b, block, lambda kernel: kernel, lora_in, lora_w, scale)
    except Exception as exc:
        if isinstance(exc, _passthrough_errors()):
            raise
        FAILED_SHAPES[key] = f"{type(exc).__name__}: {exc}"[:800]
        logger.warning(f"Unsloth: the FP8 W8A16 grouped GEMM failed for {key}; dequantizing instead: {FAILED_SHAPES[key]}")
        FP8_FUSED_STATS["fallback"] += 1
        return _fallback(x, w, s, counts, transpose_b, block, lora_in, lora_w, scale)
    FP8_FUSED_STATS["fwd" if transpose_b else "dx"] += 1
    if lora_in is not None:
        FP8_FUSED_STATS["lora_fold"] += 1
    return out


# Custom op: torch.compile traces it (no graph break) and Inductor launches the same kernel.
_OP = None
if triton is not None:
    try:
        from torch.library import triton_op as _triton_op, wrap_triton as _wrap_triton

        @_triton_op("unsloth_zoo::fp8_block_grouped_mm_v1", mutates_args = ())
        def _fp8_block_grouped_mm_op(
            x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, counts: torch.Tensor,
            transpose_b: bool, block_r: int, block_c: int,
        ) -> torch.Tensor:
            x = x.contiguous()
            out = torch.empty((x.shape[0], _out_cols(w, transpose_b)), dtype = x.dtype, device = x.device)
            return _launch(x, w, s, counts, out, transpose_b, (block_r, block_c), _wrap_triton)

        def _op_setup(ctx, inputs, output):
            _, w, s, counts, transpose_b, block_r, block_c = inputs
            ctx.save_for_backward(w, s, counts)
            ctx.transpose_b, ctx.block = transpose_b, (block_r, block_c)

        def _op_backward(ctx, grad):
            w, s, counts = ctx.saved_tensors
            dx = None
            if ctx.needs_input_grad[0]:
                dx = torch.ops.unsloth_zoo.fp8_block_grouped_mm_v1(
                    grad.contiguous(), w, s, counts, not ctx.transpose_b, ctx.block[0], ctx.block[1],
                )
            return dx, None, None, None, None, None, None

        _fp8_block_grouped_mm_op.register_autograd(_op_backward, setup_context = _op_setup)
        _OP = torch.ops.unsloth_zoo.fp8_block_grouped_mm_v1.default
    except Exception:  # pragma: no cover
        _OP = None


class _Fp8BlockGroupedMM(torch.autograd.Function):
    """Eager twin of the custom op (skips its dispatch cost); backward is dX only, the experts are frozen."""

    @staticmethod
    def forward(ctx, x, w, s, counts, transpose_b, block):
        ctx.save_for_backward(w, s, counts)
        ctx.transpose_b, ctx.block = transpose_b, block
        return _grouped_eager(x, w, s, counts, transpose_b, block)

    @staticmethod
    def backward(ctx, grad):
        w, s, counts = ctx.saved_tensors
        dx = None
        if ctx.needs_input_grad[0]:
            dx = _grouped_eager(grad, w, s, counts, not ctx.transpose_b, ctx.block)
        return dx, None, None, None, None, None


def fp8_block_grouped_mm(x, w, s, counts, transpose_b = True, block = (128, 128)):
    """``x @ W[e]^T`` (``transpose_b``) or ``x @ W[e]`` for a frozen block-FP8 stack ``w`` (E, R, C) e4m3 with fp32
    scales ``s`` (E, ceil(R / block[0]), ceil(C / block[1])); rows of ``x`` sorted by expert, ``counts`` int32 (E,)."""
    if torch.compiler.is_compiling() and _OP is not None:
        return _OP(x, w, s, counts, bool(transpose_b), int(block[0]), int(block[1]))
    if torch.is_grad_enabled() and x.requires_grad:
        return _Fp8BlockGroupedMM.apply(x, w, s, counts, bool(transpose_b), tuple(block))
    return _grouped_eager(x, w, s, counts, bool(transpose_b), tuple(block))


class _KeptBase(torch.autograd.Function):
    """The first pass's base output for the checkpoint replay (handed over in a list, not an autograd input);
    backward is dX through the frozen stack, as fp8_block_grouped_mm's."""

    @staticmethod
    def forward(ctx, x, w, s, counts, transpose_b, block, box):
        ctx.save_for_backward(w, s, counts)
        ctx.transpose_b, ctx.block = transpose_b, block
        return box.pop()

    @staticmethod
    def backward(ctx, grad):
        w, s, counts = ctx.saved_tensors
        dx = None
        if ctx.needs_input_grad[0]:
            dx = _grouped_eager(grad, w, s, counts, not ctx.transpose_b, ctx.block)
        return dx, None, None, None, None, None, None


def fp8_block_grouped_linear(inputs, param, counts, transpose_b, keep_slot = None, proj = None):
    """Expert-sorted ``inputs`` @ the frozen Fp8BlockExpertParam stack. Inside an Unsloth checkpoint (``keep_slot``)
    the no-grad first pass keeps its output and the replay reuses it (moe_gc_keep_base), so a step runs the base
    GEMM twice (first pass, dX) instead of three times."""
    w = param.data.view(_FP8_DTYPE)
    s = param.packed_scales
    if s.device != w.device:
        s = param.packed_scales = s.to(w.device)
    counts = counts.to(device = inputs.device, dtype = torch.int32)
    block = tuple(param.packed_block)
    if keep_slot is not None and not torch.compiler.is_compiling():
        if keep_slot.replaying and torch.is_grad_enabled():
            base = keep_slot.take(proj, inputs.shape[0], inputs.dtype, inputs.device)
            if base is not None:
                FP8_FUSED_STATS["kept"] += 1
                if inputs.requires_grad:
                    return _KeptBase.apply(inputs, w, s, counts, bool(transpose_b), block, [base])
                return base
        out = fp8_block_grouped_mm(inputs, w, s, counts, transpose_b, block)
        if not keep_slot.replaying and not torch.is_grad_enabled():
            keep_slot.store(proj, out)
        return out
    return fp8_block_grouped_mm(inputs, w, s, counts, transpose_b, block)


class _Fp8LoraFold(torch.autograd.Function):
    """Y = X @ W^T + s (X @ A) @ B over the frozen block-FP8 stack in one kernel (moe_lora_fold's algebra):
    dX = dY @ W + s (dY @ B^T) @ A^T in one kernel, dA = s X^T U, dB = s T^T dY."""

    @staticmethod
    def forward(ctx, inputs, first, second, w, s, counts, offsets, transpose_b, block, scaling):
        from unsloth_zoo.temporary_patches.moe_lora_fold import _rank_mm
        x = inputs.contiguous()
        t = _rank_mm(x, first, offsets)
        with torch.no_grad():
            out = _grouped_eager(x, w, s, counts, transpose_b, block, t, second, scaling)
        need_x, need_a, need_b = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        ctx.save_for_backward(
            x if need_a else None, t if need_b else None, first if (need_x or need_a) else None,
            second if (need_x or need_a) else None, w, s, counts, offsets,
        )
        ctx.transpose_b, ctx.block, ctx.scaling = transpose_b, block, scaling
        return out

    @staticmethod
    def backward(ctx, grad):
        from unsloth_zoo.temporary_patches.moe_lora_fold import _rank_mm, _wgrad
        x, t, first, second, w, s, counts, offsets = ctx.saved_tensors
        need_x, need_a, need_b = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        sc = ctx.scaling
        g = grad.contiguous()
        if torch.is_grad_enabled():
            # create_graph=True: differentiable pieces (fused dX is itself an autograd op, the rank terms torch's).
            from unsloth_zoo.temporary_patches import moe_utils as mu
            u = mu._grouped_mm_eager(g, second.transpose(-2, -1), offsets) if (need_x or need_a) else None
            gx = ga = gb = None
            if need_x:
                gx = fp8_block_grouped_mm(g, w, s, counts, not ctx.transpose_b, ctx.block) \
                    + mu._grouped_mm_eager(u, first.transpose(-2, -1), offsets) * sc
            if need_a:
                ga = mu._grouped_mm_fp16_wgrad_eager(x, u, offsets) * sc
            if need_b:
                gb = mu._grouped_mm_fp16_wgrad_eager(t, g, offsets) * sc
            return gx, ga, gb, None, None, None, None, None, None, None
        gx = ga = gb = None
        with torch.no_grad():
            u = _rank_mm(g, second.transpose(-2, -1), offsets) if (need_x or need_a) else None
            if need_x:
                gx = _grouped_eager(g, w, s, counts, not ctx.transpose_b, ctx.block, u, first.transpose(-2, -1), sc)
            if need_a:
                ga = _wgrad(x, u, offsets).mul_(sc)
            if need_b:
                gb = _wgrad(t, g, offsets).mul_(sc)
        return gx, ga, gb, None, None, None, None, None, None, None


def _fold_lora(inputs, lora):
    """(first, second, scaling) ready for the folded kernel, or None (the separated LoRA path applies)."""
    if lora is None or torch.compiler.is_compiling():
        return None
    try:
        from unsloth_zoo.temporary_patches.moe_lora_fold import lora_fold_enabled, _prepare_lora
    except ImportError:
        return None
    if not lora_fold_enabled():
        return None
    first, second, _ = lora
    if not all(type(t) in (torch.Tensor, torch.nn.Parameter) for t in (first, second)):
        return None
    if first.dim() != 3 or second.dim() != 3 or first.shape[1] != inputs.shape[1] or second.shape[1] != first.shape[2]:
        return None
    if first.device != inputs.device or second.device != inputs.device or first.shape[2] > 128:
        return None
    return _prepare_lora(inputs, lora)


def fp8_block_base_and_lora(inputs, param, counts, offsets, transpose_b, lora, keep_slot = None, proj = None):
    """(output, LoRA still to add) for one projection over a frozen Fp8BlockExpertParam stack: the expert LoRA folded
    into the W8A16 kernel when it can be, else the base GEMM alone (kept for the checkpoint replay) and the LoRA left
    to the separated path."""
    prepared = _fold_lora(inputs, lora)
    if prepared is None:
        return fp8_block_grouped_linear(inputs, param, counts, transpose_b, keep_slot = keep_slot, proj = proj), lora
    first, second, scaling = prepared
    w = param.data.view(_FP8_DTYPE)
    s = param.packed_scales
    if s.device != w.device:
        s = param.packed_scales = s.to(w.device)
    counts = counts.to(device = inputs.device, dtype = torch.int32)
    out = _Fp8LoraFold.apply(
        inputs, first, second, w, s, counts, offsets, bool(transpose_b), tuple(param.packed_block), scaling,
    )
    return out, None


_AVAILABLE = {}


def _triton_too_old(capability):
    # Triton 3.0 (torch 2.4) aborts the whole process compiling some kernels, and Triton 3.2 asserts (abort, past any
    # try / except) lowering tl.dot for sm_100 / sm_120.
    try:
        major, minor = (int(v) for v in triton.__version__.split(".")[:2])
    except Exception:
        return False
    return (major, minor) < ((3, 3) if capability[0] >= 10 else (3, 1))


def _probe(device):
    """Fused forward and dX against dequantize + fp32 matmul on a small ragged problem (empty expert, odd rows)."""
    gen = torch.Generator(device = device).manual_seed(0)
    E, R, C, block = 3, 256, 384, (128, 128)
    w = (torch.randn(E, R, C, device = device, generator = gen) * 0.5).to(_FP8_DTYPE)
    s = torch.rand(E, 2, 3, device = device, generator = gen) * 0.02 + 0.001
    counts = torch.tensor([5, 0, 19], dtype = torch.int32, device = device)
    rows = torch.repeat_interleave(torch.arange(E, device = device), counts.long())
    dense = _dequant_stack(w, s, block, torch.bfloat16).float()
    x = torch.randn(24, C, device = device, generator = gen).to(torch.bfloat16)
    got = _grouped_eager(x, w, s, counts, True, block).float()
    want = torch.bmm(x.float().unsqueeze(1), dense[rows].transpose(1, 2)).squeeze(1)
    g = torch.randn(24, R, device = device, generator = gen).to(torch.bfloat16)
    got_t = _grouped_eager(g, w, s, counts, False, block).float()
    want_t = torch.bmm(g.float().unsqueeze(1), dense[rows]).squeeze(1)
    return bool(
        torch.allclose(got, want, rtol = 2e-2, atol = 2e-2 * want.abs().max().item())
        and torch.allclose(got_t, want_t, rtol = 2e-2, atol = 2e-2 * want_t.abs().max().item())
    )


def fp8_fused_enabled(device) -> bool:
    """CUDA sm89+ (native e4m3 converts), not ROCm, Triton >= 3.1, self-checked once per device."""
    if fp8_fused_mode() == "0" or triton is None or _FP8_DTYPE is None:
        return False
    if getattr(device, "type", None) != "cuda" or torch.version.hip is not None:
        return False
    key = str(device)
    ok = _AVAILABLE.get(key)
    if ok is not None:
        return ok
    ok = False
    try:
        if _capability(device) < (8, 9) or _triton_too_old(_capability(device)):
            ok = False
        elif torch.cuda.is_current_stream_capturing():
            return False  # the self-check syncs: decide on the next uncaptured call
        else:
            before = dict(FP8_FUSED_STATS)
            with torch.cuda.device(device), torch.autocast(device.type, enabled = False), torch.no_grad():
                ok = _probe(device)
            FP8_FUSED_STATS.update(before)
    except torch.cuda.OutOfMemoryError:
        return False  # transient: dequantize this call, probe again on the next one
    except Exception as exc:
        logger.warning(f"Unsloth: FP8 W8A16 grouped GEMM unavailable ({type(exc).__name__}: {exc}); dequantizing instead.")
        ok = False
    _AVAILABLE[key] = ok
    return ok


def _block_of(weight, scale, block):
    """(rows, cols) block size of a 3-D stack: the attached block_size, else inferred from the scale grid."""
    E, R, C = weight.shape
    p, q = scale.shape[1], scale.shape[2]
    if block is not None and len(block) == 2:
        br, bc = int(block[0]), int(block[1])
    else:
        br, bc = -(-R // max(p, 1)), -(-C // max(q, 1))
    if p != -(-R // br) or q != -(-C // bc):
        return None
    return br, bc


def _stack_view(module, name, cache):
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import _get_moe_weight_and_quant_info
    weight, scale, kind = _get_moe_weight_and_quant_info(module, name)
    if (
        not isinstance(weight, torch.Tensor) or weight.dtype != _FP8_DTYPE or weight.dim() != 3
        # FP8Experts allocates its codes with requires_grad=True; PEFT freezes them. Under grad keep the dequant path
        # for a stack still marked trainable, so nothing changes for it.
        or (weight.requires_grad and torch.is_grad_enabled())
        or not weight.is_contiguous() or weight.device.type != "cuda"
        or kind != "weight_scale_inv" or not isinstance(scale, torch.Tensor) or scale.dim() != 3
        or scale.shape[0] != weight.shape[0] or not scale.dtype.is_floating_point
        or scale.device != weight.device
    ):
        return None
    E, R, C = weight.shape
    # The kernel indexes one expert's (R, C) codes in int32.
    if R * C >= 2**31:
        return None
    block = _block_of(weight, scale, getattr(weight, "block_size", None) or getattr(scale, "block_size", None))
    if block is None or min(block) < 16 or block[0] % 16 or block[1] % 16:
        return None
    # _version: an in-place copy_ / load_state_dict keeps the storage but must drop the converted scales.
    key = (name, weight.data_ptr(), scale.data_ptr(), tuple(weight.shape), weight._version, scale._version, block)
    view = cache.get(key)
    if view is None:
        from unsloth_zoo.packed_experts import Fp8BlockExpertParam
        with torch.inference_mode(False), torch.no_grad():
            s = scale.detach().to(torch.float32).contiguous()
            view = Fp8BlockExpertParam(weight.detach().view(torch.uint8), packed_scales = s, packed_block = block)
        for stale in [k for k in cache if k[0] == name]:
            cache.pop(stale, None)
        cache[key] = view
    return view


def fp8_experts_as_packed(module, hidden_states):
    """(gate_up, down) Fp8BlockExpertParam views of frozen block-FP8 experts for the fused grouped GEMM, else None
    (UNSLOTH_MOE_FP8_FUSED=0, ROCm, below sm89, fp16 / fp32 compute, trainable or non-block FP8 experts)."""
    if fp8_fused_mode() == "0" or torch.compiler.is_compiling():
        return None
    from unsloth_zoo.temporary_patches.moe_utils import moe_compute_dtype
    if moe_compute_dtype(hidden_states) != torch.bfloat16:
        return None
    cache = module.__dict__.setdefault("_unsloth_fp8_block_views", {})
    views = []
    for name in ("gate_up_proj", "down_proj"):
        view = _stack_view(module, name, cache)
        if view is None:
            return None
        views.append(view)
    if not fp8_fused_enabled(views[0].device):
        return None
    return tuple(views)
