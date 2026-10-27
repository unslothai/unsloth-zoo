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

"""MXFP4 (e2m1 nibbles + e8m0 block scales) to bf16 / fp16, matching transformers' convert_moe_packed_tensors; Triton kernel with a torch fallback."""

import os

import torch

__all__ = [
    "mxfp4_dequantize",
    "mxfp4_dequantize_torch",
    "mxfp4_kernel_available",
    "Mxfp4ExpertParam",
    "is_mxfp4_expert_param",
]

_FP4_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# Bounds the torch path's temporaries to ~256 MB (a whole gpt-oss-120b stack would be 8.5 GB).
_TORCH_CHUNK_BYTES = 256 << 20

# ROCm Triton (gfx11, Triton 3.6) lowers bf16 x bf16 to v_dot2_bf16_bf16(a, b, +0) (flushes subnormals,
# -0 -> +0) and fp32 multiply then fp16 cast to v_fma_mixlo_f16(a, b, +0) (-0 -> +0). There the kernel
# multiplies in fp32 and rounds to 16 bits with integer ops instead.
_IS_HIP = torch.version.hip is not None

# torch < 2.12's ldexp forms 2^e in the input dtype: inf from 2^16 in fp16, and from 2^128 (e8m0 255) in fp32 (0 -> NaN).
_LDEXP_EXACT = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 12)


def mxfp4_dequantize_torch(blocks, scales, dtype = torch.bfloat16, transpose = False, out = None, max_rows = None):
    """Reference: transformers' convert_moe_packed_tensors arithmetic, in row chunks."""
    from unsloth_zoo.utils import device_guard

    # torch.ldexp launches on the current device: operands on another GPU fault without this.
    with device_guard(blocks):
        return _mxfp4_dequantize_torch(blocks, scales, dtype, transpose, out, max_rows)


def _mxfp4_dequantize_torch(blocks, scales, dtype, transpose, out, max_rows):
    # float64 holds 2^128; routing it through fp32 would overflow.
    work = dtype if _LDEXP_EXACT or dtype == torch.float64 else torch.float32
    lut = torch.tensor(_FP4_VALUES, dtype = work, device = blocks.device)
    *prefix, G, B = blocks.shape
    if transpose and not prefix:
        raise ValueError("Unsloth: transpose needs MXFP4 blocks with a row dim")
    N = prefix[-1] if prefix else 1
    R = blocks.numel() // (G * B)
    shape = (*prefix[:-1], G * B * 2, N) if transpose else (*prefix, G * B * 2)
    buf = out if out is not None and out.is_contiguous() and out.dtype == dtype else torch.empty(shape, dtype = dtype, device = blocks.device)
    rows_in = blocks.reshape(R, G, B)
    scales_in = scales.reshape(R, G)
    # Transposed: row r lands in column r % N of expert r // N; a chunk never crosses an expert.
    dst = buf.view(R // N, G * B * 2, N) if transpose else buf.view(R, G * B * 2)
    # Live bytes per value: 8 native; the fp32 detour (float32 output too) also holds ldexp/where outputs and the cast (~18).
    per_value = 8 if _LDEXP_EXACT or dtype == torch.float64 else 20
    step = max(1, min(N if transpose else R, _TORCH_CHUNK_BYTES // (G * B * 2 * per_value)))
    if max_rows is not None:
        step = max(1, min(step, max_rows))
    r0 = 0
    while r0 < R:
        # Advance to r1, not by step: a chunk cut at an expert edge is shorter.
        r1 = min(r0 + step, (r0 // N + 1) * N if transpose else R)
        b = rows_in[r0:r1]
        piece = torch.empty(r1 - r0, G, B * 2, dtype = work, device = blocks.device)
        piece[..., 0::2] = lut[(b & 0x0F).long()]
        piece[..., 1::2] = lut[(b >> 4).long()]
        exponent = (scales_in[r0:r1].to(torch.int32) - 127).unsqueeze(-1)
        if _LDEXP_EXACT:
            piece = torch.ldexp(piece, exponent).to(dtype)
        elif work == torch.float64:
            # Old float64 ldexp rounds on CUDA; 2^e from its float64 bits is exact for every e8m0 scale.
            piece.mul_(((exponent.to(torch.int64) + 1023) << 52).view(torch.float64))
        else:
            # Exact in fp32, then one rounding; 255 as 2^127 then 2.
            piece = torch.ldexp(piece, exponent.clamp(max = 127))
            piece = torch.where(exponent == 128, piece * 2, piece).to(dtype)
        piece = piece.reshape(r1 - r0, G * B * 2)
        if transpose:
            e, i0 = divmod(r0, N)
            dst[e, :, i0:i0 + (r1 - r0)] = piece.T
        else:
            dst[r0:r1] = piece
        r0 = r1
    if out is not None and buf is not out:
        out.copy_(buf)
        return out
    return buf


if _HAS_TRITON:

    @triton.jit
    def _e2m1_to_f32(nib):
        # e2m1: sign bit 3, exponent bits 2..1, mantissa bit 0; exponent 0 is the subnormal 0 / 0.5.
        sign = tl.where((nib & 8) != 0, -1.0, 1.0)
        e = (nib >> 1) & 3
        m = (nib & 1).to(tl.float32)
        mag = tl.where(e == 0, m * 0.5, (1.0 + 0.5 * m) * tl.exp2((e - 1).to(tl.float32)))
        return sign * mag

    @triton.jit
    def _f32_to_f16_rne(x):
        u = x.to(tl.int32, bitcast = True)
        sign = (u >> 16) & 0x8000
        a = u & 0x7FFFFFFF
        c = tl.minimum(a, 0x477FF000)
        normal = (c + 0xFFF + ((c >> 13) & 1) - 0x38000000) >> 13
        # Clamped to [1, 31]: lanes >= 2^-14 take `normal` but must not shift by a negative count.
        shift = tl.minimum(tl.maximum(126 - (a >> 23), 1), 31)
        m = (a & 0x7FFFFF) | 0x800000
        q = m >> shift
        rem = m - (q << shift)
        half = 1 << (shift - 1)
        q = q + ((rem > half) | ((rem == half) & ((q & 1) == 1))).to(tl.int32)
        h = tl.where(a < 0x38800000, q, normal)
        h = tl.where(a >= 0x477FF000, 0x7C00, h)
        h = tl.where(a > 0x7F800000, 0x7E00 | ((a >> 13) & 0x3FF), h)
        return (h | sign).to(tl.uint16).to(tl.float16, bitcast = True)

    @triton.jit
    def _f32_to_bf16_rne(x):
        # Matches c10::BFloat16 rounding; NaN -> 0x7FC0 | sign.
        u = x.to(tl.uint32, bitcast = True)
        r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
        r = tl.where((u & 0x7FFFFFFF) > 0x7F800000, ((u >> 16) & 0x8000) | 0x7FC0, r)
        return r.to(tl.uint16).to(tl.bfloat16, bitcast = True)

    @triton.jit
    def _mxfp4_dequant_kernel(
        blocks_ptr, scales_ptr, out_ptr, experts_ptr, counts_ptr,
        N, G,
        stride_out_e, stride_out_n, stride_out_k,
        BLOCK_N: tl.constexpr, BLOCK_G: tl.constexpr, USE_EXPERTS: tl.constexpr, USE_COUNTS: tl.constexpr,
        BF16_BITS: tl.constexpr, LONG_INDEXING: tl.constexpr, EXACT_ROUND: tl.constexpr,
    ):
        pid_e = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_g = tl.program_id(2)
        if USE_EXPERTS:
            e = tl.load(experts_ptr + pid_e)
        else:
            e = pid_e
        e = e.to(tl.int64) if LONG_INDEXING else e.to(tl.int32)
        if USE_COUNTS:
            if tl.load(counts_ptr + e) == 0:
                return
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_g = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
        offs_b = tl.arange(0, 16)
        mask_ng = (offs_n[:, None] < N) & (offs_g[None, :] < G)
        row = e * N + offs_n
        byte_offs = (row[:, None, None] * G + offs_g[None, :, None]) * 16 + offs_b[None, None, :]
        packed = tl.load(blocks_ptr + byte_offs, mask = mask_ng[:, :, None], other = 0).to(tl.int32)
        scale = tl.load(scales_ptr + row[:, None] * G + offs_g[None, :], mask = mask_ng, other = 127).to(tl.int32)
        # e8m0 255 = 2^128 overflows: scale by 2^127 then 2, so 0 stays 0 and 0.5 finite, as ldexp.
        top = scale == 255
        scale = tl.where(top, 254, scale)
        double = tl.where(top, 2.0, 1.0)
        # 2^(scale - 127) built from its fp32 bits; scale 0 is the fp32 subnormal 2^-127.
        factor = tl.where(scale == 0, 5.877471754111438e-39, ((scale << 23).to(tl.int32, bitcast = True)).to(tl.float32, bitcast = True))
        if BF16_BITS:
            # bf16 bits straight from the nibble (triton_kernels' portable upcast).
            em0 = packed & 0x07
            em1 = packed & 0x70
            x0 = (em0 << 6) | ((packed & 0x08) << 12)
            x1 = (em1 << 2) | ((packed & 0x80) << 8)
            x0 = tl.where((em0 & 0x06) != 0, x0 + (126 << 7), x0)
            x1 = tl.where((em1 & 0x60) != 0, x1 + (126 << 7), x1)
            x0 = tl.where(em0 == 0x01, 16128 | (x0 & 0x8000), x0)
            x1 = tl.where(em1 == 0x10, 16128 | (x1 & 0x8000), x1)
            lo = x0.to(tl.uint16).to(tl.bfloat16, bitcast = True)
            hi = x1.to(tl.uint16).to(tl.bfloat16, bitcast = True)
            # scale 0 is the bf16 subnormal 2^-127.
            s16 = tl.maximum(scale << 7, 0x0040).to(tl.uint16).to(tl.bfloat16, bitcast = True)
            lo = lo * s16[:, :, None]
            hi = hi * s16[:, :, None]
            double16 = double.to(tl.bfloat16)[:, :, None]
            lo = lo * double16
            hi = hi * double16
        else:
            lo = _e2m1_to_f32(packed & 15) * factor[:, :, None] * double[:, :, None]
            hi = _e2m1_to_f32(packed >> 4) * factor[:, :, None] * double[:, :, None]
        vals = tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_G * 32))
        if EXACT_ROUND and not BF16_BITS:
            if out_ptr.dtype.element_ty == tl.float16:
                vals = _f32_to_f16_rne(vals)
            elif out_ptr.dtype.element_ty == tl.bfloat16:
                vals = _f32_to_bf16_rne(vals)
        vals = vals.to(out_ptr.dtype.element_ty)
        offs_k = pid_g * BLOCK_G * 32 + tl.arange(0, BLOCK_G * 32)
        mask = (offs_n[:, None] < N) & (offs_k[None, :] < G * 32)
        out_offs = e * stride_out_e + offs_n[:, None] * stride_out_n + offs_k[None, :] * stride_out_k
        tl.store(out_ptr + out_offs, vals, mask = mask)


_KERNEL_FAILED = set()
_KERNEL_VERIFIED = {}


def _kernel_failed():
    return _exact_round() in _KERNEL_FAILED


def _triton_usable(blocks):
    return _HAS_TRITON and not _kernel_failed() and blocks.is_cuda and blocks.dtype == torch.uint8


def _kernel_fail(exception):
    # Per rounding path: the other variant can still be tried.
    _KERNEL_FAILED.add(_exact_round())
    import warnings
    warnings.warn(f"Unsloth: fused MXFP4 dequant unavailable ({exception}); using the torch path.")


def _tracing():
    try:
        if torch.compiler.is_compiling():
            return True
        from torch._guards import detect_fake_mode
        return detect_fake_mode() is not None
    except Exception:
        return False


def _outside_trace(fn):
    """``fn`` run for real while a graph is traced: dynamo calls an ``assume_constant_result`` function eagerly and
    bakes in its result; fake mode is lifted so the self-check's own tensors are real."""
    def run(*args):
        from torch._subclasses.fake_tensor import unset_fake_temporarily
        with unset_fake_temporarily():
            return fn(*args)
    try:
        return torch._dynamo.assume_constant_result(run)
    except Exception:
        return run


# Verdicts reached inside a trace. Never read while tracing (a dict the traced frame read must not change under
# it), only by the next eager call, which moves them into _KERNEL_VERIFIED.
_TRACE_VERIFIED = {}


def _kernel_verified_in_trace(device, dtype, transpose, exact_round):
    key = (device, dtype, transpose, exact_round)
    try:
        verified = _probe_kernel(device, dtype, transpose)
    except Exception:
        verified = False  # the eager call repeats the probe and records the failure
    _TRACE_VERIFIED[key] = verified
    return verified


_kernel_verified_outside_trace = _outside_trace(_kernel_verified_in_trace)


def _probe_kernel(device, dtype, transpose):
    E, N, G = 2, 131, 9
    blocks = ((torch.arange(E * N * G * 16, device = device) * 7) % 256).to(torch.uint8).reshape(E, N, G, 16)
    if dtype == torch.float16:
        scales = 112 + (torch.arange(E * N * G, device = device) * 5) % 24
    else:
        scales = (torch.arange(E * N * G, device = device) * 37) % 255
    scales = scales.to(torch.uint8).reshape(E, N, G)
    edges = [0, 127, 254, 255] if dtype != torch.float16 else [0, 127, 135, 255]
    scales.view(-1)[: len(edges)] = torch.tensor(edges, dtype = torch.uint8)
    want = mxfp4_dequantize_torch(blocks, scales, dtype = dtype, transpose = transpose)
    got = _kernel_dequantize(blocks, scales, dtype, transpose, None, None, None)
    return bool(torch.equal(got.view(torch.uint8), want.view(torch.uint8)))


def _kernel_verified(device, dtype, transpose):
    """Once per (device, dtype, layout, rounding path): kernel must match the torch reference bit for bit, else torch is used."""
    key = (str(device), dtype, bool(transpose), _exact_round())
    verified = _KERNEL_VERIFIED.get(key)
    if verified is not None:
        return verified
    if _tracing():
        # The probe's torch.equal is data dependent (fullgraph refuses it, e.g. a cold compiled backward): run it
        # outside the trace, so the traced graph already gets the verdict (the kernel) instead of the reference.
        return _kernel_verified_outside_trace(*key)
    if _TRACE_VERIFIED.get(key) is True:
        _KERNEL_VERIFIED[key] = True
        return True
    try:
        verified = _probe_kernel(device, dtype, transpose)
    except torch.cuda.OutOfMemoryError:
        raise  # a full GPU says nothing about the kernel: decide on a later call
    except Exception as exception:
        _kernel_fail(exception)
        verified = False
    if not verified and not _kernel_failed():
        import warnings
        warnings.warn("Unsloth: fused MXFP4 dequant does not match the reference on this device; using the torch path.")
    _KERNEL_VERIFIED[key] = verified
    return verified


def mxfp4_kernel_available(device = None, dtype = torch.bfloat16) -> bool:
    """Whether the fused kernel runs, and is exact, for the GPT-OSS transposed layout on ``device``."""
    if not _HAS_TRITON or _kernel_failed() or not torch.cuda.is_available():
        return False
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    device = torch.device(device)
    if device.type != "cuda":
        return False
    try:
        return _kernel_verified(device, dtype, True)
    except torch.cuda.OutOfMemoryError:
        return False  # a predicate: no room for the probe now, nothing cached, a later call decides


def mxfp4_dequantize(
    blocks, scales, dtype = torch.bfloat16, transpose = False, experts = None, token_counts = None, out = None,
):
    """``experts`` / ``token_counts`` fill only those slices; the rest stay uninitialised (grouped GEMMs skip them)."""
    if blocks.shape[:-1] != scales.shape or blocks.shape[-1] != 16:
        raise ValueError(f"Unsloth: MXFP4 blocks {tuple(blocks.shape)} do not match scales {tuple(scales.shape)}")
    from unsloth_zoo.utils import device_guard

    # Triton and torch.ldexp launch on the current device, not the operand's.
    with device_guard(blocks):
        return _mxfp4_dequantize(blocks, scales, dtype, transpose, experts, token_counts, out)


def _mxfp4_dequantize(blocks, scales, dtype, transpose, experts, token_counts, out):
    if (
        not _triton_usable(blocks) or blocks.dim() < 3
        or not _kernel_verified(blocks.device, dtype, transpose)
    ):
        return mxfp4_dequantize_torch(blocks, scales, dtype = dtype, transpose = transpose, out = out)
    try:
        return _kernel_dequantize(blocks, scales, dtype, transpose, experts, token_counts, out)
    except torch.cuda.OutOfMemoryError:
        raise  # not a kernel failure: keep the kernel, let the caller shrink its chunk
    except Exception as exception:
        _kernel_fail(exception)
        return mxfp4_dequantize_torch(blocks, scales, dtype = dtype, transpose = transpose, out = out)


def _kernel_dequantize(blocks, scales, dtype, transpose, experts, token_counts, out):
    *prefix, G, _ = blocks.shape
    E = 1
    for size in prefix[:-1]:
        E *= size
    N = prefix[-1]
    K = G * 32
    shape = (*prefix[:-1], K, N) if transpose else (*prefix, K)
    if out is None:
        out = torch.empty(shape, dtype = dtype, device = blocks.device)
    elif tuple(out.shape) != shape or not out.is_contiguous():
        raise ValueError(f"Unsloth: MXFP4 out buffer must be contiguous {shape}, got {tuple(out.shape)}")
    blocks = blocks.contiguous()
    scales = scales.contiguous()
    if transpose:
        stride_e, stride_n, stride_k = K * N, 1, N
    else:
        stride_e, stride_n, stride_k = N * K, K, 1
    if token_counts is not None:
        if experts is not None or token_counts.numel() != E or token_counts.device != blocks.device:
            token_counts = None
        else:
            token_counts = token_counts.contiguous()
    if experts is not None:
        experts = experts.to(device = blocks.device, dtype = torch.int32).contiguous()
        n_programs_e = experts.numel()
        if n_programs_e == 0:
            return out
    else:
        n_programs_e = E
    # Swept on B200 over gpt-oss-20b shapes: a taller n tile keeps transposed stores coalesced.
    exact_round = _exact_round()
    bf16_bits = dtype == torch.bfloat16 and not exact_round
    if bf16_bits:
        BLOCK_N, BLOCK_G = (128, 4) if transpose else (32, 8)
    else:
        BLOCK_N, BLOCK_G = (128, 1) if transpose else (64, 2)
    grid = (n_programs_e, triton.cdiv(N, BLOCK_N), triton.cdiv(G, BLOCK_G))
    # int32 offsets only when transposed: untransposed expert-subset dequant ran ~5% slower in int32 (B200).
    long_indexing = not transpose or E * N * K + BLOCK_N * K > 2**31
    _launch(grid, blocks, scales, out, experts, token_counts, N, G, stride_e, stride_n, stride_k,
            BLOCK_N, BLOCK_G, bf16_bits, long_indexing, exact_round)
    return out


def _exact_round():
    # UNSLOTH_MXFP4_EXACT_ROUND=1 / 0 forces the fp32-multiply + integer-rounding path on / off (default: ROCm only).
    flag = os.environ.get("UNSLOTH_MXFP4_EXACT_ROUND", "")
    return _IS_HIP if flag == "" else flag == "1"


def _launch(grid, blocks, scales, out, experts, token_counts, N, G, stride_e, stride_n, stride_k,
            BLOCK_N, BLOCK_G, bf16_bits, long_indexing, exact_round = False):
    with torch.cuda.device(blocks.device):
        _mxfp4_dequant_kernel[grid](
            blocks, scales, out, experts if experts is not None else blocks,
            token_counts if token_counts is not None else blocks,
            N, G,
            stride_e, stride_n, stride_k,
            BLOCK_N = BLOCK_N, BLOCK_G = BLOCK_G, USE_EXPERTS = experts is not None,
            USE_COUNTS = token_counts is not None,
            BF16_BITS = bf16_bits, LONG_INDEXING = long_indexing, EXACT_ROUND = exact_round, num_warps = 4,
            enable_fp_fusion = True,
        )


class Mxfp4ExpertParam(torch.nn.Parameter):
    """Frozen MXFP4 expert stack: blocks ``(E, N, G, 16)`` + ``mxfp4_scales``; rebuildable as ``cls(data, requires_grad, **__dict__)`` (accelerate)."""

    def __new__(
        cls, data, requires_grad = False, mxfp4_scales = None, mxfp4_transposed = True,
        mxfp4_dtype = torch.bfloat16, **_ignored,
    ):
        if mxfp4_scales is None or data.dtype != torch.uint8 or data.dim() < 3 or data.shape[-1] != 16:
            raise ValueError("Unsloth: Mxfp4ExpertParam needs uint8 blocks (..., G, 16) and their scales")
        param = torch.Tensor._make_subclass(cls, data, False)
        param.mxfp4_scales = mxfp4_scales
        param.mxfp4_transposed = bool(mxfp4_transposed)
        param.mxfp4_dtype = mxfp4_dtype
        *prefix, N, G, _ = data.shape
        param._original_shape = torch.Size((*prefix, G * 32, N) if mxfp4_transposed else (*prefix, N, G * 32))
        return param

    def _init_kwargs(self):
        return dict(
            mxfp4_scales = self.mxfp4_scales, mxfp4_transposed = self.mxfp4_transposed,
            mxfp4_dtype = self.mxfp4_dtype,
        )

    def dequantize(self, dtype = None, token_counts = None, out = None, transpose = None):
        """Logical layout by default; ``transpose`` overrides it (False -> (E, N, K) packed-row order)."""
        scales = self.mxfp4_scales
        if scales.device != self.device:
            scales = self.mxfp4_scales = scales.to(self.device)
        return mxfp4_dequantize(
            self.data, scales, dtype = dtype or self.mxfp4_dtype,
            transpose = self.mxfp4_transposed if transpose is None else bool(transpose),
            token_counts = token_counts, out = out,
        )

    def to(self, *args, **kwargs):
        # Only a device move applies to the packed bytes; a dtype cast would destroy them.
        device, _, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is None or torch.device(device) == self.device:
            return self
        kwargs = self._init_kwargs()
        kwargs["mxfp4_scales"] = self.mxfp4_scales.to(device, non_blocking = non_blocking)
        return Mxfp4ExpertParam(self.data.to(device, non_blocking = non_blocking), **kwargs)

    def copy_(self, src, non_blocking = False):
        if not isinstance(src, Mxfp4ExpertParam):
            raise RuntimeError(
                "Unsloth: cannot copy bare MXFP4 blocks into a packed expert stack, their scales "
                "would be left stale. Load a state_dict taken from a packed model, or reload the "
                "model with UNSLOTH_MXFP4_KEEP_PACKED=0."
            )
        if src.mxfp4_transposed != self.mxfp4_transposed or src.mxfp4_scales.shape != self.mxfp4_scales.shape:
            raise RuntimeError("Unsloth: MXFP4 expert stacks with different layouts cannot be copied")
        torch.Tensor.copy_(self, src, non_blocking = non_blocking)
        self.mxfp4_scales = src.mxfp4_scales.to(self.device, non_blocking = non_blocking, copy = True)
        self.mxfp4_dtype = src.mxfp4_dtype
        return self

    def detach(self):
        # nn.Module._apply re-wraps meta moves as a Parameter, which needs detach() to keep the subtype.
        return Mxfp4ExpertParam(torch.Tensor.detach(self), **self._init_kwargs())

    def __deepcopy__(self, memo):
        kwargs = self._init_kwargs()
        kwargs["mxfp4_scales"] = self.mxfp4_scales.clone()
        return Mxfp4ExpertParam(self.data.clone(), **kwargs)

    def __reduce_ex__(self, protocol):
        return (_rebuild_mxfp4_expert_param, (self.data, self._init_kwargs()))

    def __repr__(self):
        return f"Mxfp4ExpertParam(logical_shape={tuple(self._original_shape)}, device={self.device})"


def _rebuild_mxfp4_expert_param(data, kwargs):
    return Mxfp4ExpertParam(data, **kwargs)


def is_mxfp4_expert_param(param) -> bool:
    return isinstance(param, Mxfp4ExpertParam)
