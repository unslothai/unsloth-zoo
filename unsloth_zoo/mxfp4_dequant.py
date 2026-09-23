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

"""MXFP4 (e2m1 nibbles + e8m0 block scales, 32 values per block) to bf16 / fp16.

``mxfp4_dequantize(blocks, scales)`` takes the checkpoint layout, blocks ``(..., G, 16)`` uint8
and scales ``(..., G)`` uint8, and returns ``(..., G * 32)``, the layout
``transformers.integrations.mxfp4.convert_moe_packed_tensors`` returns. ``transpose = True``
writes the last two dims swapped in the same pass (what GPT-OSS stores as ``gate_up_proj``),
and ``experts`` restricts the work to those leading-dim slices. One Triton pass, no fp32 or
int64 temporaries; the torch path is the fallback on CPU, MPS and without Triton.
"""

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


def mxfp4_dequantize_torch(blocks, scales, dtype = torch.bfloat16, transpose = False):
    """Reference: the same arithmetic as transformers' convert_moe_packed_tensors."""
    lut = torch.tensor(_FP4_VALUES, dtype = dtype, device = blocks.device)
    *prefix, G, B = blocks.shape
    out = torch.empty(*prefix, G, B * 2, dtype = dtype, device = blocks.device)
    out[..., 0::2] = lut[(blocks & 0x0F).long()]
    out[..., 1::2] = lut[(blocks >> 4).long()]
    out = torch.ldexp(out, (scales.to(torch.int32) - 127).unsqueeze(-1)).to(dtype)
    out = out.reshape(*prefix, G * B * 2)
    if transpose:
        out = out.transpose(-2, -1).contiguous()
    return out


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
    def _mxfp4_dequant_kernel(
        blocks_ptr, scales_ptr, out_ptr, experts_ptr, counts_ptr,
        N, G,
        stride_out_e, stride_out_n, stride_out_k,
        BLOCK_N: tl.constexpr, BLOCK_G: tl.constexpr, USE_EXPERTS: tl.constexpr, USE_COUNTS: tl.constexpr,
        BF16_BITS: tl.constexpr,
    ):
        pid_e = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_g = tl.program_id(2)
        if USE_EXPERTS:
            e = tl.load(experts_ptr + pid_e).to(tl.int64)
        else:
            e = pid_e.to(tl.int64)
        if USE_COUNTS:
            # An expert no token was routed to is never read by the grouped GEMM: skip it.
            if tl.load(counts_ptr + e) == 0:
                return
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_g = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
        offs_b = tl.arange(0, 16)
        mask_ng = (offs_n[:, None] < N) & (offs_g[None, :] < G)
        row = e * N + offs_n
        # bytes [BLOCK_N, BLOCK_G, 16]
        byte_offs = (row[:, None, None] * G + offs_g[None, :, None]) * 16 + offs_b[None, None, :]
        packed = tl.load(blocks_ptr + byte_offs, mask = mask_ng[:, :, None], other = 0).to(tl.int32)
        scale = tl.load(scales_ptr + row[:, None] * G + offs_g[None, :], mask = mask_ng, other = 127).to(tl.int32)
        # 2^(scale - 127) built from its fp32 bits; scale 0 is the fp32 subnormal 2^-127.
        factor = tl.where(scale == 0, 5.877471754111438e-39, ((scale << 23).to(tl.int32, bitcast = True)).to(tl.float32, bitcast = True))
        if BF16_BITS:
            # bf16 bit patterns straight from the nibble (triton_kernels' portable upcast): sign to bit
            # 15, exponent + mantissa shifted into place and rebiased, 0.5 (subnormal e2m1) set directly.
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
            # 2^(scale - 127) as a bf16 bit pattern; scale 0 is the bf16 subnormal 2^-127. Exact product.
            s16 = tl.maximum(scale << 7, 0x0040).to(tl.uint16).to(tl.bfloat16, bitcast = True)
            lo = lo * s16[:, :, None]
            hi = hi * s16[:, :, None]
        else:
            lo = _e2m1_to_f32(packed & 15) * factor[:, :, None]
            hi = _e2m1_to_f32(packed >> 4) * factor[:, :, None]
        vals = tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_G * 32)).to(out_ptr.dtype.element_ty)
        offs_k = pid_g * BLOCK_G * 32 + tl.arange(0, BLOCK_G * 32)
        mask = (offs_n[:, None] < N) & (offs_k[None, :] < G * 32)
        out_offs = e * stride_out_e + offs_n[:, None] * stride_out_n + offs_k[None, :] * stride_out_k
        tl.store(out_ptr + out_offs, vals, mask = mask)


_KERNEL_FAILED = False
_KERNEL_VERIFIED = {}


def _triton_usable(blocks):
    return _HAS_TRITON and not _KERNEL_FAILED and blocks.is_cuda and blocks.dtype == torch.uint8


def _kernel_fail(exception):
    # A Triton build that cannot compile the kernel (an old ROCm stack) falls back for good.
    global _KERNEL_FAILED
    _KERNEL_FAILED = True
    import warnings
    warnings.warn(f"Unsloth: fused MXFP4 dequant unavailable ({exception}); using the torch path.")


def _kernel_verified(device, dtype, transpose):
    """Once per (device, dtype, layout): the kernel must reproduce the torch reference bit for bit
    on every byte value, the edge e8m0 scales and partial tiles, else the torch path is used."""
    key = (str(device), dtype, bool(transpose))
    verified = _KERNEL_VERIFIED.get(key)
    if verified is not None:
        return verified
    try:
        E, N, G = 2, 131, 9
        blocks = ((torch.arange(E * N * G * 16, device = device) * 7) % 256).to(torch.uint8).reshape(E, N, G, 16)
        if dtype == torch.float16:
            scales = 112 + (torch.arange(E * N * G, device = device) * 5) % 24
        else:
            scales = (torch.arange(E * N * G, device = device) * 37) % 255
        scales = scales.to(torch.uint8).reshape(E, N, G)
        scales.view(-1)[:3] = torch.tensor([0, 127, 254 if dtype != torch.float16 else 135], dtype = torch.uint8)
        want = mxfp4_dequantize_torch(blocks, scales, dtype = dtype, transpose = transpose)
        got = _kernel_dequantize(blocks, scales, dtype, transpose, None, None, None)
        verified = bool(torch.equal(got.view(torch.uint8), want.view(torch.uint8)))
    except Exception as exception:
        _kernel_fail(exception)
        verified = False
    if not verified and not _KERNEL_FAILED:
        import warnings
        warnings.warn("Unsloth: fused MXFP4 dequant does not match the reference on this device; using the torch path.")
    _KERNEL_VERIFIED[key] = verified
    return verified


def mxfp4_kernel_available(device = None, dtype = torch.bfloat16) -> bool:
    """Whether the fused kernel runs, and is exact, for the GPT-OSS transposed layout on ``device``."""
    if not _HAS_TRITON or _KERNEL_FAILED or not torch.cuda.is_available():
        return False
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    device = torch.device(device)
    if device.type != "cuda":
        return False
    return _kernel_verified(device, dtype, True)


def mxfp4_dequantize(
    blocks, scales, dtype = torch.bfloat16, transpose = False, experts = None, token_counts = None, out = None,
):
    """Dequantize MXFP4 ``blocks`` / ``scales`` to ``dtype``.

    ``transpose`` swaps the last two output dims in the same pass. ``experts`` (1-D int tensor
    of leading-dim indices) fills only those slices of a full-size output; the rest is left
    uninitialised, which is safe for grouped GEMMs that never read an expert with no tokens.
    ``token_counts`` (per-expert token counts, on device) does the same without a host sync:
    experts with a zero count are skipped inside the kernel. ``out`` reuses a buffer."""
    if blocks.shape[:-1] != scales.shape or blocks.shape[-1] != 16:
        raise ValueError(f"Unsloth: MXFP4 blocks {tuple(blocks.shape)} do not match scales {tuple(scales.shape)}")
    if (
        not _triton_usable(blocks) or blocks.dim() < 3
        or not _kernel_verified(blocks.device, dtype, transpose)
    ):
        full = mxfp4_dequantize_torch(blocks, scales, dtype = dtype, transpose = transpose)
        if out is not None:
            out.copy_(full)
            return out
        return full
    try:
        return _kernel_dequantize(blocks, scales, dtype, transpose, experts, token_counts, out)
    except Exception as exception:
        _kernel_fail(exception)
        full = mxfp4_dequantize_torch(blocks, scales, dtype = dtype, transpose = transpose)
        if out is not None:
            out.copy_(full)
            return out
        return full


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
    bf16_bits = dtype == torch.bfloat16
    if bf16_bits:
        BLOCK_N, BLOCK_G = (128, 4) if transpose else (32, 8)
    else:
        BLOCK_N, BLOCK_G = (128, 1) if transpose else (64, 2)
    grid = (n_programs_e, triton.cdiv(N, BLOCK_N), triton.cdiv(G, BLOCK_G))
    _launch(grid, blocks, scales, out, experts, token_counts, N, G, stride_e, stride_n, stride_k,
            BLOCK_N, BLOCK_G, bf16_bits)
    return out


def _launch(grid, blocks, scales, out, experts, token_counts, N, G, stride_e, stride_n, stride_k,
            BLOCK_N, BLOCK_G, bf16_bits):
    with torch.cuda.device(blocks.device):
        _mxfp4_dequant_kernel[grid](
            blocks, scales, out, experts if experts is not None else blocks,
            token_counts if token_counts is not None else blocks,
            N, G,
            stride_e, stride_n, stride_k,
            BLOCK_N = BLOCK_N, BLOCK_G = BLOCK_G, USE_EXPERTS = experts is not None,
            USE_COUNTS = token_counts is not None,
            BF16_BITS = bf16_bits, num_warps = 4,
        )


class Mxfp4ExpertParam(torch.nn.Parameter):
    """A frozen MoE expert stack kept as MXFP4: ``.data`` holds the blocks ``(E, N, G, 16)``
    uint8, ``mxfp4_scales`` the ``(E, N, G)`` e8m0 scales. ``_original_shape`` is the logical
    stack (``(E, G * 32, N)`` when ``mxfp4_transposed``, GPT-OSS's ``(E, in, out)`` layout),
    read wherever a packed parameter's logical shape is needed, as for bnb 4-bit experts.

    Built as ``Mxfp4ExpertParam(blocks, mxfp4_scales = scales)`` and re-buildable as
    ``cls(data, requires_grad, **param.__dict__)``, which is how accelerate re-creates a
    parameter on another device. uint8 data is never touched by ``module.to(dtype)`` /
    ``.half()``; the scales follow the blocks lazily in ``dequantize``."""

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

    def dequantize(self, dtype = None, token_counts = None, out = None):
        scales = self.mxfp4_scales
        if scales.device != self.device:
            scales = self.mxfp4_scales = scales.to(self.device)
        return mxfp4_dequantize(
            self.data, scales, dtype = dtype or self.mxfp4_dtype,
            transpose = self.mxfp4_transposed, token_counts = token_counts, out = out,
        )

    def to(self, *args, **kwargs):
        # Only a device move applies to the packed bytes; a dtype cast would destroy them.
        device, _, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is None or torch.device(device) == self.device:
            return self
        kwargs = self._init_kwargs()
        kwargs["mxfp4_scales"] = self.mxfp4_scales.to(device, non_blocking = non_blocking)
        return Mxfp4ExpertParam(self.data.to(device, non_blocking = non_blocking), **kwargs)

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
