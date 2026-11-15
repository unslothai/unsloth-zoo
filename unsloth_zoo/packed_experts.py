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

"""Frozen routed-expert stacks kept packed for LoRA: NVFP4 (e2m1 codes, e4m3 1x16 block scales, fp32 global scales)
and MXFP8 (e4m3 codes, e8m0 1x32 block scales).

Codes are stored ``(E, N, K / 2)`` (NVFP4) or ``(E, N, K)`` (MXFP8) with the blocks along the contraction dim K, the
checkpoint orientation. The logical (module) shape is ``(E, N, K)``, or ``(E, K, N)`` when ``packed_transposed``.
Forward and dX dequantize a chunk of experts at a time into a 16-bit stack (``_chunked_grouped_mm``); the base is
frozen, so there is no dW. The decode is one fp32 rounding, the same arithmetic as Unsloth's ``nvfp4_dequantize``
(``e2m1 * (scale / global)``; ModelOpt's ``weight_scale_2`` multiplies instead) and the MXFP8 scale-grid decode
(``e4m3 * 2 ** (e8m0 - 127)``), so the packed path is bitwise equal to a grouped GEMM on the dense dequantized stack.
"""

import os

import torch

__all__ = [
    "Nvfp4ExpertParam",
    "Mxfp8ExpertParam",
    "is_packed_expert_param",
    "is_new_packed_expert_param",
    "nvfp4_experts_dequantize",
    "nvfp4_experts_dequantize_torch",
    "mxfp8_experts_dequantize",
    "mxfp8_experts_dequantize_torch",
    "packed_expert_grouped_linear",
    "quantize_experts_nvfp4",
    "quantize_experts_mxfp4",
    "quantize_experts_mxfp8",
    "PACKED_EXPERT_STATS",
]

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False

NVFP4_GROUP = 16
MXFP8_GROUP = 32
MX_GROUP = 32
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MAX = 6.0
_E4M3_MAX = 448.0

# Engagement counters (eager calls of the grouped op, chunks dequantized); tests read them.
PACKED_EXPERT_STATS = {"calls": 0, "chunks": 0}

# Torch reference row chunk: bounds its fp32 temporaries to ~64 MB.
_TORCH_CHUNK_ELEMS = 1 << 24


def _use_triton(t):
    return _HAS_TRITON and t.device.type == "cuda" and os.environ.get("UNSLOTH_PACKED_EXPERTS_TRITON", "1") != "0"


if _HAS_TRITON:
    @triton.jit
    def _e2m1_value(code):
        # fp32 bits of an E2M1 code: mag >= 2 is (1 + m / 2) * 2^(e - 1), mag 1 is 0.5, mag 0 is 0; bit 3 is the sign.
        mag = code & 7
        bits = tl.where(
            mag >= 2, (((mag >> 1) + 126) << 23) | ((mag & 1) << 22), tl.where(mag == 1, 126 << 23, 0)
        )
        return (bits | ((code & 8) << 28)).to(tl.float32, bitcast = True)

    @triton.jit
    def _e4m3_value(b):
        # E4M3 bytes by hand (no float8e4nv before sm_89): bias 7, exp 0 subnormal, 0x7F / 0xFF NaN, as torch.
        exp = (b >> 3) & 15
        man = b & 7
        normal = (((exp + 120) << 23) | (man << 20)).to(tl.float32, bitcast = True)
        value = tl.where(exp == 0, man.to(tl.float32) * 0.001953125, normal)
        value = tl.where((b & 127) == 127, float("nan"), value)
        return (value.to(tl.int32, bitcast = True) | ((b & 128) << 24)).to(tl.float32, bitcast = True)

    @triton.jit
    def _e8m0_value(s):
        # 2 ** (s - 127) as fp32 bits, what exp2(s - 127.0) gives: s = 0 is the subnormal 2^-127, s = 255 is inf.
        bits = tl.where(s == 0, 0x00400000, tl.where(s == 255, 0x7F800000, s << 23))
        return bits.to(tl.float32, bitcast = True)

    @triton.jit
    def _nvfp4_experts_kernel(
        codes_ptr, scale_ptr, global_ptr, out_ptr, counts_ptr,
        N, HALF, SCALE_COLS, ROWS_PER_PART, PARTS,
        s_oe, s_or, s_oc,
        BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, USE_COUNTS: tl.constexpr, MULTIPLY: tl.constexpr,
    ):
        e = tl.program_id(0).to(tl.int64)
        if USE_COUNTS:
            if tl.load(counts_ptr + e) == 0:
                return
        r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
        c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
        g = tl.program_id(2) * (BLOCK_C // 8) + tl.arange(0, BLOCK_C // 8)
        row_ok = r[:, None] < N
        byte = tl.load(
            codes_ptr + e * N * HALF + r[:, None].to(tl.int64) * HALF + c[None, :],
            mask = row_ok & (c[None, :] < HALF), other = 0,
        ).to(tl.int32)
        scale = tl.load(
            scale_ptr + e * N * SCALE_COLS + r[:, None].to(tl.int64) * SCALE_COLS + g[None, :],
            mask = row_ok & (g[None, :] < SCALE_COLS), other = 0,
        ).to(tl.int32)
        scale = _e4m3_value(scale)
        glob = tl.load(global_ptr + e * PARTS + r // ROWS_PER_PART, mask = r < N, other = 1.0).to(tl.float32)
        if MULTIPLY:
            group = scale * glob[:, None]
        else:
            group = tl.math.div_rn(scale, glob[:, None])
        values = tl.reshape(tl.join(_e2m1_value(byte & 15), _e2m1_value(byte >> 4)), (BLOCK_R, BLOCK_C // 8, 16))
        values = tl.reshape(values * group[:, :, None], (BLOCK_R, 2 * BLOCK_C))
        oc = tl.program_id(2) * (2 * BLOCK_C) + tl.arange(0, 2 * BLOCK_C)
        tl.store(
            out_ptr + e * s_oe + r[:, None].to(tl.int64) * s_or + oc[None, :].to(tl.int64) * s_oc,
            values.to(out_ptr.dtype.element_ty),
            mask = row_ok & (oc[None, :] < 2 * HALF),
        )

    @triton.jit
    def _mxfp8_experts_kernel(
        codes_ptr, scale_ptr, out_ptr, counts_ptr,
        N, K, SCALE_COLS,
        s_oe, s_or, s_oc,
        BLOCK_R: tl.constexpr, BLOCK_K: tl.constexpr, USE_COUNTS: tl.constexpr,
    ):
        e = tl.program_id(0).to(tl.int64)
        if USE_COUNTS:
            if tl.load(counts_ptr + e) == 0:
                return
        r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
        k = tl.program_id(2) * BLOCK_K + tl.arange(0, BLOCK_K)
        g = tl.program_id(2) * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)
        row_ok = r[:, None] < N
        byte = tl.load(
            codes_ptr + e * N * K + r[:, None].to(tl.int64) * K + k[None, :],
            mask = row_ok & (k[None, :] < K), other = 0,
        ).to(tl.int32)
        s = tl.load(
            scale_ptr + e * N * SCALE_COLS + r[:, None].to(tl.int64) * SCALE_COLS + g[None, :],
            mask = row_ok & (g[None, :] < SCALE_COLS), other = 127,
        ).to(tl.int32)
        values = tl.reshape(_e4m3_value(byte), (BLOCK_R, BLOCK_K // 32, 32)) * _e8m0_value(s)[:, :, None]
        values = tl.reshape(values, (BLOCK_R, BLOCK_K))
        tl.store(
            out_ptr + e * s_oe + r[:, None].to(tl.int64) * s_or + k[None, :].to(tl.int64) * s_oc,
            values.to(out_ptr.dtype.element_ty),
            mask = row_ok & (k[None, :] < K),
        )


def _out_buffer(E, N, K, dtype, device, transpose, out):
    shape = (E, K, N) if transpose else (E, N, K)
    if out is not None and tuple(out.shape) == shape and out.dtype == dtype and out.is_contiguous():
        return out
    return torch.empty(shape, dtype = dtype, device = device)


def _counts_arg(token_counts, E, device):
    if token_counts is None or token_counts.numel() != E or token_counts.device != device:
        return None
    return token_counts.contiguous()


def _parts(global_scale, E, N):
    parts = global_scale.numel() // E
    if parts < 1 or global_scale.numel() != E * parts or N % parts:
        raise ValueError(
            f"Unsloth: NVFP4 expert global scale {tuple(global_scale.shape)} does not split {E} experts x {N} rows"
        )
    return parts


def nvfp4_experts_dequantize_torch(codes, scales, global_scale, dtype = torch.bfloat16, transpose = False,
                                   multiply = False, out = None):
    """Reference decode, same rounding as the kernel: ``e2m1 * (scale / global)`` (or ``* global``) in fp32, one cast."""
    E, N, half = codes.shape
    K = half * 2
    parts = _parts(global_scale, E, N)
    buf = _out_buffer(E, N, K, dtype, codes.device, transpose, out)
    lut = torch.tensor(_E2M1, dtype = torch.float32, device = codes.device)
    glob = global_scale.reshape(E, parts).to(torch.float32).repeat_interleave(N // parts, dim = 1)  # (E, N)
    rows = max(1, _TORCH_CHUNK_ELEMS // max(1, K))
    for e in range(E):
        for r0 in range(0, N, rows):
            r1 = min(N, r0 + rows)
            byte = codes[e, r0:r1]
            code = torch.stack((byte & 0x0F, byte >> 4), dim = -1).reshape(r1 - r0, K)
            values = lut[(code & 7).long()]
            values = torch.where((code & 8).bool(), -values, values)
            scale = scales[e, r0:r1].view(torch.float8_e4m3fn).to(torch.float32) if scales.dtype == torch.uint8 \
                else scales[e, r0:r1].to(torch.float32)
            g = glob[e, r0:r1, None]
            group = scale * g if multiply else scale / g
            dense = (values.view(r1 - r0, K // NVFP4_GROUP, NVFP4_GROUP) * group[:, :, None]).reshape(r1 - r0, K)
            if transpose:
                buf[e, :, r0:r1] = dense.t().to(dtype)
            else:
                buf[e, r0:r1] = dense.to(dtype)
    return buf


def nvfp4_experts_dequantize(codes, scales, global_scale, dtype = torch.bfloat16, transpose = False,
                             multiply = False, token_counts = None, out = None):
    """Dense ``(E, N, K)`` (``transpose``: ``(E, K, N)``) stack of packed ``(E, N, K / 2)`` NVFP4 codes; ``token_counts``
    leaves experts with no tokens undecoded."""
    if codes.dim() != 3 or codes.dtype != torch.uint8:
        raise ValueError("Unsloth: NVFP4 expert codes must be uint8 (E, N, K / 2)")
    E, N, half = codes.shape
    if tuple(scales.shape) != (E, N, half * 2 // NVFP4_GROUP):
        raise ValueError(f"Unsloth: NVFP4 expert scales {tuple(scales.shape)} do not match codes {tuple(codes.shape)}")
    if not _use_triton(codes):
        return nvfp4_experts_dequantize_torch(codes, scales, global_scale, dtype, transpose, multiply, out)
    K = half * 2
    parts = _parts(global_scale, E, N)
    buf = _out_buffer(E, N, K, dtype, codes.device, transpose, out)
    if E == 0 or N == 0 or K == 0:
        return buf
    counts = _counts_arg(token_counts, E, codes.device)
    s_oe = N * K
    s_or, s_oc = (1, N) if transpose else (K, 1)
    BLOCK_R, BLOCK_C = (32, 64) if transpose else (8, 128)
    sc = scales.view(torch.uint8) if scales.dtype != torch.uint8 else scales
    with torch.cuda.device(codes.device):
        _nvfp4_experts_kernel[(E, triton.cdiv(N, BLOCK_R), triton.cdiv(half, BLOCK_C))](
            codes.contiguous(), sc.contiguous(), global_scale.to(torch.float32).contiguous(), buf,
            counts if counts is not None else codes,
            N, half, K // NVFP4_GROUP, N // parts, parts, s_oe, s_or, s_oc,
            BLOCK_R = BLOCK_R, BLOCK_C = BLOCK_C, USE_COUNTS = counts is not None, MULTIPLY = bool(multiply),
            num_warps = 4,
        )
    return buf


def mxfp8_experts_dequantize_torch(codes, scales, dtype = torch.bfloat16, transpose = False, out = None):
    """Reference decode: ``float(e4m3) * 2 ** (e8m0 - 127)`` in fp32, one cast (the scale-grid decode's arithmetic)."""
    E, N, K = codes.shape
    buf = _out_buffer(E, N, K, dtype, codes.device, transpose, out)
    rows = max(1, _TORCH_CHUNK_ELEMS // max(1, K))
    for e in range(E):
        for r0 in range(0, N, rows):
            r1 = min(N, r0 + rows)
            q = codes[e, r0:r1].view(torch.float8_e4m3fn).to(torch.float32)
            s = (scales[e, r0:r1].to(torch.float32) - 127.0).exp2()
            dense = (q.view(r1 - r0, K // MXFP8_GROUP, MXFP8_GROUP) * s[:, :, None]).reshape(r1 - r0, K)
            if transpose:
                buf[e, :, r0:r1] = dense.t().to(dtype)
            else:
                buf[e, r0:r1] = dense.to(dtype)
    return buf


def mxfp8_experts_dequantize(codes, scales, dtype = torch.bfloat16, transpose = False, token_counts = None,
                             out = None):
    """Dense ``(E, N, K)`` (``transpose``: ``(E, K, N)``) stack of ``(E, N, K)`` e4m3 bytes with e8m0 1x32 scales."""
    if codes.dim() != 3 or codes.dtype != torch.uint8:
        raise ValueError("Unsloth: MXFP8 expert codes must be uint8 (E, N, K) e4m3 bytes")
    E, N, K = codes.shape
    if tuple(scales.shape) != (E, N, K // MXFP8_GROUP) or K % MXFP8_GROUP:
        raise ValueError(f"Unsloth: MXFP8 expert scales {tuple(scales.shape)} do not match codes {tuple(codes.shape)}")
    if not _use_triton(codes):
        return mxfp8_experts_dequantize_torch(codes, scales, dtype, transpose, out)
    buf = _out_buffer(E, N, K, dtype, codes.device, transpose, out)
    if E == 0 or N == 0 or K == 0:
        return buf
    counts = _counts_arg(token_counts, E, codes.device)
    s_oe = N * K
    s_or, s_oc = (1, N) if transpose else (K, 1)
    BLOCK_R, BLOCK_K = (32, 64) if transpose else (8, 256)
    with torch.cuda.device(codes.device):
        _mxfp8_experts_kernel[(E, triton.cdiv(N, BLOCK_R), triton.cdiv(K, BLOCK_K))](
            codes.contiguous(), scales.contiguous(), buf, counts if counts is not None else codes,
            N, K, K // MXFP8_GROUP, s_oe, s_or, s_oc,
            BLOCK_R = BLOCK_R, BLOCK_K = BLOCK_K, USE_COUNTS = counts is not None, num_warps = 4,
        )
    return buf


class _PackedExpertParam(torch.nn.Parameter):
    """Frozen packed expert stack; rebuildable as ``cls(data, requires_grad, **__dict__)`` (accelerate)."""

    packed_format = None

    def __new__(
        cls, data, requires_grad = False, packed_scales = None, packed_global = None, packed_transposed = False,
        packed_dtype = torch.bfloat16, packed_global_mode = "divide", **_ignored,
    ):
        if packed_scales is None and isinstance(data, _PackedExpertParam):
            # accelerate's set_module_tensor_to_device rebuilds as cls(value.to(device), requires_grad = ...).
            kwargs = data._init_kwargs()
            packed_scales, packed_global = kwargs["packed_scales"], kwargs["packed_global"]
            packed_transposed, packed_dtype = kwargs["packed_transposed"], kwargs["packed_dtype"]
            packed_global_mode = kwargs["packed_global_mode"]
        if data.dtype != torch.uint8 or data.dim() != 3 or packed_scales is None:
            raise ValueError(f"Unsloth: {cls.__name__} needs uint8 (E, N, *) codes and their block scales")
        param = torch.Tensor._make_subclass(cls, data, False)
        param.packed_scales = packed_scales
        param.packed_global = packed_global
        param.packed_transposed = bool(packed_transposed)
        param.packed_dtype = packed_dtype
        param.packed_global_mode = str(packed_global_mode)
        cls._validate(param)
        E, N, K = param.packed_dims()
        param._original_shape = torch.Size((E, K, N) if packed_transposed else (E, N, K))
        return param

    @classmethod
    def _validate(cls, param):
        pass

    def packed_dims(self):
        raise NotImplementedError

    def _init_kwargs(self):
        return dict(
            packed_scales = self.packed_scales, packed_global = self.packed_global,
            packed_transposed = self.packed_transposed, packed_dtype = self.packed_dtype,
            packed_global_mode = self.packed_global_mode,
        )

    def _companions(self):
        # accelerate rebuilds the param on a new device but leaves the companion tensors where they were.
        if self.packed_scales.device != self.device:
            self.packed_scales = self.packed_scales.to(self.device)
        if self.packed_global is not None and self.packed_global.device != self.device:
            self.packed_global = self.packed_global.to(self.device)
        return self.packed_scales, self.packed_global

    def _decode(self, codes, scales, glob, dtype, transpose, token_counts):
        raise NotImplementedError

    def dequantize(self, dtype = None, token_counts = None, out = None, transpose = None):
        """Logical layout by default; ``transpose`` overrides it (False -> (E, N, K) packed-row order)."""
        scales, glob = self._companions()
        transpose = self.packed_transposed if transpose is None else bool(transpose)
        return self._decode(self.data, scales, glob, dtype or self.packed_dtype, transpose, token_counts, out)

    def dequantize_experts(self, start, end, dtype = None, transpose = None, token_counts = None):
        """Experts ``[start, end)`` only (the chunked grouped GEMM's unit)."""
        scales, glob = self._companions()
        transpose = self.packed_transposed if transpose is None else bool(transpose)
        g = None if glob is None else glob[start:end]
        counts = None if token_counts is None else token_counts
        return self._decode(self.data[start:end], scales[start:end], g, dtype or self.packed_dtype, transpose, counts)

    def nbytes_packed(self):
        total = self.data.numel() * self.data.element_size() + self.packed_scales.numel() * self.packed_scales.element_size()
        if self.packed_global is not None:
            total += self.packed_global.numel() * self.packed_global.element_size()
        return total

    def to(self, *args, **kwargs):
        # Only a device move applies to the packed bytes; a dtype cast would destroy them. _parse_to refuses
        # copy = (compressed-tensors' offload passes it on the first forward of a compressed checkpoint).
        copy = bool(kwargs.pop("copy", False))
        device, _, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        same = device is None or torch.device(device) == self.device
        if same and not copy:
            return self
        device = self.device if same else device
        move = (lambda t: t.clone()) if same else (lambda t: t.to(device, non_blocking = non_blocking))
        kwargs = self._init_kwargs()
        kwargs["packed_scales"] = move(self.packed_scales)
        if self.packed_global is not None:
            kwargs["packed_global"] = move(self.packed_global)
        return type(self)(move(self.data), **kwargs)

    def copy_(self, src, non_blocking = False):
        if type(src) is not type(self):
            raise RuntimeError(
                f"Unsloth: cannot copy {type(src).__name__} into a packed {self.packed_format} expert stack: its "
                "scales would be left stale. Load a state_dict taken from a packed model."
            )
        if src.packed_transposed != self.packed_transposed or src.packed_scales.shape != self.packed_scales.shape:
            raise RuntimeError(f"Unsloth: {self.packed_format} expert stacks with different layouts cannot be copied")
        torch.Tensor.copy_(self, src, non_blocking = non_blocking)
        self.packed_scales = src.packed_scales.to(self.device, non_blocking = non_blocking, copy = True)
        if src.packed_global is not None:
            self.packed_global = src.packed_global.to(self.device, non_blocking = non_blocking, copy = True)
        self.packed_dtype = src.packed_dtype
        self.packed_global_mode = src.packed_global_mode
        return self

    def detach(self):
        # nn.Module._apply re-wraps meta moves as a Parameter, which needs detach() to keep the subtype.
        return type(self)(torch.Tensor.detach(self), **self._init_kwargs())

    def __deepcopy__(self, memo):
        kwargs = self._init_kwargs()
        kwargs["packed_scales"] = self.packed_scales.clone()
        if self.packed_global is not None:
            kwargs["packed_global"] = self.packed_global.clone()
        return type(self)(self.data.clone(), **kwargs)

    def __reduce_ex__(self, protocol):
        return (_rebuild_packed_expert_param, (type(self), self.data, self._init_kwargs()))

    def __repr__(self):
        return f"{type(self).__name__}(logical_shape={tuple(self._original_shape)}, device={self.device})"


def _rebuild_packed_expert_param(cls, data, kwargs):
    return cls(data, **kwargs)


class Nvfp4ExpertParam(_PackedExpertParam):
    """NVFP4 stack: codes ``(E, N, K / 2)`` (low nibble = even column), e4m3 ``packed_scales`` ``(E, N, K / 16)``,
    fp32 ``packed_global`` ``(E, P)`` (scales held as their uint8 bytes) (P = 1 per expert, 2 for unfused gate / up, N per row). ``packed_global_mode``:
    ``divide`` (compressed-tensors ``weight_global_scale``) or ``multiply`` (ModelOpt ``weight_scale_2``)."""

    packed_format = "nvfp4"

    @classmethod
    def _validate(cls, param):
        E, N, half = param.data.shape
        scales = param.packed_scales
        if tuple(scales.shape) != (E, N, half * 2 // NVFP4_GROUP) or (half * 2) % NVFP4_GROUP:
            raise ValueError(f"Unsloth: NVFP4 scales {tuple(scales.shape)} do not match codes {tuple(param.data.shape)}")
        if scales.dtype not in (torch.float8_e4m3fn, torch.uint8):
            raise ValueError("Unsloth: NVFP4 block scales must be float8_e4m3fn")
        if scales.dtype == torch.float8_e4m3fn:
            # Kept as the raw bytes: plain pickle of an fp8 tensor fails (torch 2.14), and the kernels read bytes.
            param.packed_scales = scales.view(torch.uint8)
        glob = param.packed_global
        if glob is None or glob.dtype != torch.float32:
            raise ValueError("Unsloth: NVFP4 expert stacks need fp32 global scales")
        if glob.dim() == 1:
            param.packed_global = glob = glob.reshape(E, -1) if E else glob.reshape(0, 1)
        _parts(glob, E, N)
        if param.packed_global_mode not in ("divide", "multiply"):
            raise ValueError(f"Unsloth: unknown NVFP4 global scale mode {param.packed_global_mode!r}")

    def packed_dims(self):
        E, N, half = self.data.shape
        return E, N, half * 2

    def _decode(self, codes, scales, glob, dtype, transpose, token_counts, out = None):
        return nvfp4_experts_dequantize(
            codes, scales, glob, dtype = dtype, transpose = transpose,
            multiply = self.packed_global_mode == "multiply", token_counts = token_counts, out = out,
        )


class Mxfp8ExpertParam(_PackedExpertParam):
    """MXFP8 stack: e4m3 bytes ``(E, N, K)`` (stored uint8, so ``.half()`` never casts them) and e8m0 ``packed_scales``
    ``(E, N, K / 32)``."""

    packed_format = "mxfp8"

    @classmethod
    def _validate(cls, param):
        E, N, K = param.data.shape
        if K % MXFP8_GROUP or tuple(param.packed_scales.shape) != (E, N, K // MXFP8_GROUP):
            raise ValueError(
                f"Unsloth: MXFP8 scales {tuple(param.packed_scales.shape)} do not match codes {tuple(param.data.shape)}"
            )
        if param.packed_scales.dtype != torch.uint8:
            raise ValueError("Unsloth: MXFP8 block scales must be uint8 e8m0 exponents")

    def packed_dims(self):
        return tuple(self.data.shape)

    def _decode(self, codes, scales, glob, dtype, transpose, token_counts, out = None):
        return mxfp8_experts_dequantize(codes, scales, dtype = dtype, transpose = transpose,
                                        token_counts = token_counts, out = out)


def is_new_packed_expert_param(param) -> bool:
    return isinstance(param, _PackedExpertParam)


def is_packed_expert_param(param) -> bool:
    """Any frozen packed expert stack that dequantizes itself: NVFP4, MXFP8 or MXFP4."""
    if isinstance(param, _PackedExpertParam):
        return True
    from unsloth_zoo.mxfp4_dequant import is_mxfp4_expert_param
    return is_mxfp4_expert_param(param)


def packed_param_transposed(param) -> bool:
    if isinstance(param, _PackedExpertParam):
        return param.packed_transposed
    return bool(getattr(param, "mxfp4_transposed", True))


def packed_param_dtype(param):
    if isinstance(param, _PackedExpertParam):
        return param.packed_dtype
    return getattr(param, "mxfp4_dtype", torch.bfloat16)


# ---------------------------------------------------------------------------------------------------------------------
# Grouped GEMM (forward + dX) through the chunked dequantize of mxfp4_stacked_experts.
# ---------------------------------------------------------------------------------------------------------------------

def _rebuild(fmt, codes, scales, glob, transposed, multiply):
    if fmt == "nvfp4":
        return Nvfp4ExpertParam(codes, packed_scales = scales, packed_global = glob, packed_transposed = transposed,
                                packed_global_mode = "multiply" if multiply else "divide")
    return Mxfp8ExpertParam(codes, packed_scales = scales, packed_transposed = transposed)


def _out_dim(codes, fmt, transpose_b):
    N = codes.shape[1]
    K = codes.shape[2] * 2 if fmt == "nvfp4" else codes.shape[2]
    return N if transpose_b else K


def _grouped_impl(inputs, codes, scales, glob, counts, fmt, transpose_b, multiply):
    from unsloth_zoo.mxfp4_stacked_experts import _chunked_grouped_mm
    param = _rebuild(fmt, codes, scales, glob, False, multiply)
    # Contiguous like the fake: an unaligned N comes back as a column slice of a padded GEMM output.
    return _chunked_grouped_mm(inputs.contiguous(), param, counts, None, inputs.dtype, transpose_b).contiguous()


def _grouped_fake(inputs, codes, scales, glob, counts, fmt, transpose_b, multiply):
    return inputs.new_empty((inputs.shape[0], _out_dim(codes, fmt, transpose_b)))


_OP = None


def _register_op():
    global _OP
    if _OP is not None:
        return _OP or None
    _OP = False
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None or not hasattr(torch.library, "register_autograd"):
        return None
    try:
        op = torch.ops.unsloth_zoo.packed_experts_mm.default
    except (AttributeError, RuntimeError):
        op = None
    if op is None:
        try:
            @custom_op("unsloth_zoo::packed_experts_mm", mutates_args = ())
            def packed_experts_mm(
                inputs: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, glob: torch.Tensor,
                counts: torch.Tensor, fmt: str, transpose_b: bool, multiply: bool,
            ) -> torch.Tensor:
                return _grouped_impl(inputs, codes, scales, None if glob.numel() == 0 else glob, counts, fmt,
                                     transpose_b, multiply)

            @packed_experts_mm.register_fake
            def _(inputs, codes, scales, glob, counts, fmt, transpose_b, multiply):
                return _grouped_fake(inputs, codes, scales, glob, counts, fmt, transpose_b, multiply)

            def _setup(ctx, inputs, output):
                _, codes, scales, glob, counts, fmt, transpose_b, multiply = inputs
                ctx.save_for_backward(codes, scales, glob, counts)
                ctx.fmt, ctx.transpose_b, ctx.multiply = fmt, transpose_b, multiply

            def _backward(ctx, grad):
                codes, scales, glob, counts = ctx.saved_tensors
                dx = torch.ops.unsloth_zoo.packed_experts_mm(
                    grad.contiguous(), codes, scales, glob, counts, ctx.fmt, not ctx.transpose_b, ctx.multiply,
                )
                return dx, None, None, None, None, None, None, None

            packed_experts_mm.register_autograd(_backward, setup_context = _setup)
            op = torch.ops.unsloth_zoo.packed_experts_mm.default
        except Exception:
            return None
    _OP = op
    return op


# Registered at import, not on first use: a trace must never be the first caller (global writes break the graph).
try:
    _register_op()
except Exception:  # pragma: no cover
    _OP = False


class _PackedGroupedLinear(torch.autograd.Function):
    # Fallback when torch.library.custom_op is unavailable.
    @staticmethod
    def forward(ctx, inputs, codes, scales, glob, counts, fmt, transpose_b, multiply):
        ctx.save_for_backward(codes, scales, glob, counts)
        ctx.fmt, ctx.transpose_b, ctx.multiply = fmt, transpose_b, multiply
        with torch.no_grad():
            return _grouped_impl(inputs, codes, scales, glob, counts, fmt, transpose_b, multiply)

    @staticmethod
    def backward(ctx, grad):
        codes, scales, glob, counts = ctx.saved_tensors
        dx = None
        if ctx.needs_input_grad[0]:
            with torch.no_grad():
                dx = _grouped_impl(grad.contiguous(), codes, scales, glob, counts, ctx.fmt, not ctx.transpose_b,
                                   ctx.multiply)
        return dx, None, None, None, None, None, None, None


def packed_expert_grouped_linear(inputs, param, counts, transpose_b):
    """``inputs`` (expert-sorted rows) @ W_e for a frozen NVFP4 / MXFP8 stack; ``transpose_b``: the grouped GEMM's
    (E, K, N) operand is the transpose of the packed (E, N, K) rows. Forward and dX only (frozen base)."""
    if not torch.compiler.is_compiling():
        PACKED_EXPERT_STATS["calls"] += 1
    scales, glob = param._companions()
    fmt = param.packed_format
    multiply = param.packed_global_mode == "multiply"
    codes = param.data
    counts = counts.to(device = inputs.device)
    op = _register_op()
    if op is not None:
        g = glob if glob is not None else codes.new_empty((0,), dtype = torch.float32)
        return op(inputs, codes, scales, g, counts, fmt, bool(transpose_b), multiply)
    return torch.compiler.disable(_PackedGroupedLinear.apply)(
        inputs, codes, scales, glob, counts, fmt, bool(transpose_b), multiply,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Round-to-nearest quantizers for 16-bit expert stacks (load-time, opt-in). Each returns the packed param.
# Rows are quantized along K, the last dim of the (E, N, K) row layout; pass the stack in that layout.
# ---------------------------------------------------------------------------------------------------------------------

def _round_e2m1(y):
    """Nearest E2M1 code of y (|y| <= 6 after clamping), ties to the even code; returns int codes 0..15."""
    a = y.abs().clamp(max = _E2M1_MAX)
    grid = torch.tensor(_E2M1, dtype = a.dtype, device = a.device)
    # Index of the nearest grid point; ties go to the lower index unless it is odd (round half to even code).
    idx = torch.bucketize(a, (grid[1:] + grid[:-1]) / 2, right = False)
    mid = (grid[1:] + grid[:-1]) / 2
    tie = (idx < 7) & (a == mid[idx.clamp(max = 6)]) & (idx % 2 == 1)
    idx = torch.where(tie, idx + 1, idx)
    sign = (y < 0) & (idx > 0)
    return (idx | (sign.to(idx.dtype) << 3)).to(torch.uint8)


def _pack_nibbles(codes):
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def quantize_experts_nvfp4(rows, dtype = torch.bfloat16, transposed = False, chunk_experts = 8):
    """RTN NVFP4 of an (E, N, K) stack, one global scale per expert, compressed-tensors convention
    (``global = 448 * 6 / amax``, ``scale = e4m3(amax_block / 6 * global)``, decode ``e2m1 * scale / global``)."""
    E, N, K = rows.shape
    if K % NVFP4_GROUP:
        raise ValueError(f"Unsloth: NVFP4 needs K divisible by {NVFP4_GROUP}, got {K}")
    codes = torch.empty((E, N, K // 2), dtype = torch.uint8, device = rows.device)
    scales = torch.empty((E, N, K // NVFP4_GROUP), dtype = torch.float8_e4m3fn, device = rows.device)
    glob = torch.empty((E, 1), dtype = torch.float32, device = rows.device)
    for e0 in range(0, E, chunk_experts):
        e1 = min(E, e0 + chunk_experts)
        w = rows[e0:e1].to(torch.float32)
        amax = w.abs().amax(dim = (1, 2))
        g = torch.where(amax > 0, (_E4M3_MAX * _E2M1_MAX) / amax, torch.ones_like(amax))
        blocks = w.view(e1 - e0, N, K // NVFP4_GROUP, NVFP4_GROUP)
        bmax = blocks.abs().amax(dim = -1)
        s = (bmax / _E2M1_MAX * g[:, None, None]).clamp(max = _E4M3_MAX).to(torch.float8_e4m3fn)
        decode = s.to(torch.float32) / g[:, None, None]
        y = torch.where(decode[..., None] > 0, blocks / decode[..., None], torch.zeros_like(blocks))
        q = _round_e2m1(y).reshape(e1 - e0, N, K)
        codes[e0:e1] = _pack_nibbles(q)
        scales[e0:e1] = s
        glob[e0:e1, 0] = g
    return Nvfp4ExpertParam(codes, packed_scales = scales, packed_global = glob, packed_transposed = transposed,
                            packed_dtype = dtype)


def _mx_exponent(bmax, emax_elem):
    # OCP MX: shared exponent floor(log2(amax)) - emax of the element type, clamped to e8m0 [-127, 127].
    e = torch.floor(torch.log2(bmax.clamp(min = 2.0 ** -126))) - emax_elem
    e = torch.where(bmax > 0, e, torch.full_like(e, -127.0))
    return e.clamp(-127, 127)


def quantize_experts_mxfp4(rows, dtype = torch.bfloat16, transposed = False, chunk_experts = 8):
    """RTN MXFP4 (OCP: e2m1 + e8m0 per 32) of an (E, N, K) stack as an ``Mxfp4ExpertParam``."""
    from unsloth_zoo.mxfp4_dequant import Mxfp4ExpertParam
    E, N, K = rows.shape
    if K % MX_GROUP:
        raise ValueError(f"Unsloth: MXFP4 needs K divisible by {MX_GROUP}, got {K}")
    G = K // MX_GROUP
    blocks = torch.empty((E, N, G, 16), dtype = torch.uint8, device = rows.device)
    scales = torch.empty((E, N, G), dtype = torch.uint8, device = rows.device)
    for e0 in range(0, E, chunk_experts):
        e1 = min(E, e0 + chunk_experts)
        w = rows[e0:e1].to(torch.float32).view(e1 - e0, N, G, MX_GROUP)
        e = _mx_exponent(w.abs().amax(dim = -1), 2)
        q = _round_e2m1(w / torch.exp2(e)[..., None])
        blocks[e0:e1] = _pack_nibbles(q)
        scales[e0:e1] = (e + 127).to(torch.uint8)
    return Mxfp4ExpertParam(blocks, mxfp4_scales = scales, mxfp4_transposed = transposed, mxfp4_dtype = dtype)


def quantize_experts_mxfp8(rows, dtype = torch.bfloat16, transposed = False, chunk_experts = 8):
    """RTN MXFP8 (OCP: e4m3 + e8m0 per 32) of an (E, N, K) stack."""
    E, N, K = rows.shape
    if K % MXFP8_GROUP:
        raise ValueError(f"Unsloth: MXFP8 needs K divisible by {MXFP8_GROUP}, got {K}")
    G = K // MXFP8_GROUP
    codes = torch.empty((E, N, K), dtype = torch.uint8, device = rows.device)
    scales = torch.empty((E, N, G), dtype = torch.uint8, device = rows.device)
    for e0 in range(0, E, chunk_experts):
        e1 = min(E, e0 + chunk_experts)
        w = rows[e0:e1].to(torch.float32).view(e1 - e0, N, G, MXFP8_GROUP)
        e = _mx_exponent(w.abs().amax(dim = -1), 8)
        y = (w / torch.exp2(e)[..., None]).clamp(-_E4M3_MAX, _E4M3_MAX)
        codes[e0:e1] = y.to(torch.float8_e4m3fn).view(torch.uint8).reshape(e1 - e0, N, K)
        scales[e0:e1] = (e + 127).to(torch.uint8)
    return Mxfp8ExpertParam(codes, packed_scales = scales, packed_transposed = transposed, packed_dtype = dtype)


# ---------------------------------------------------------------------------------------------------------------------
# Model-level helpers: find fused 3-D expert stacks, quantize them at load (opt-in), or install checkpoint stacks.
# ---------------------------------------------------------------------------------------------------------------------

EXPERTS_QUANTIZE_ENV = "UNSLOTH_MOE_EXPERTS_QUANTIZE"
EXPERTS_QUANTIZE_FORMATS = ("nvfp4", "mxfp4", "nf4")
_STACK_NAMES = (("gate_up_proj", "gate_up"), ("down_proj", "down"))


def experts_quantize_format():
    """UNSLOTH_MOE_EXPERTS_QUANTIZE = nvfp4 | mxfp4 | nf4 (off when unset / 0): RTN of 16-bit routed-expert stacks."""
    value = os.environ.get(EXPERTS_QUANTIZE_ENV, "").strip().lower()
    if value in ("", "0", "off", "false", "none", "no"):
        return None
    if value not in EXPERTS_QUANTIZE_FORMATS:
        raise ValueError(
            f"Unsloth: {EXPERTS_QUANTIZE_ENV}={value!r} is not one of {', '.join(EXPERTS_QUANTIZE_FORMATS)}"
        )
    return value


def stack_rows_are_transposed(experts, name, proj_type, param = None):
    """True when the module's stack is (E, K, N) (contraction on dim 1, gpt-oss), False for (E, N, K) (F.linear,
    Qwen / Mixtral), None when the layout cannot be told."""
    from unsloth_zoo.temporary_patches.moe_utils import preprocess_weight
    param = getattr(experts, name, None) if param is None else param
    shape = tuple(getattr(param, "_original_shape", None) or param.shape)
    hidden = getattr(experts, "hidden_dim", None) or getattr(experts, "hidden_size", None)
    if hidden is None:
        down = getattr(experts, "down_proj", None)
        dshape = tuple(getattr(down, "_original_shape", None) or down.shape)
        hidden = dshape[1] if proj_type == "down" else None
        if hidden is None:
            return None
    meta = torch.empty(shape, device = "meta")
    try:
        weight = preprocess_weight(meta, proj_type, int(hidden), getattr(experts, "_unsloth_model_type", None),
                                   experts_module = experts)
    except Exception:
        return None
    if weight is meta or (weight.shape == meta.shape and weight.stride() == meta.stride()):
        return True
    flipped = meta.transpose(-2, -1)
    if weight.shape == flipped.shape and weight.stride() == flipped.stride():
        return False
    return None


def iter_dense_expert_stacks(model):
    """(module name, experts module) for every module holding 3-D 16 / 32-bit gate_up_proj and down_proj stacks."""
    for name, module in model.named_modules():
        params = module._parameters
        stacks = [params.get(n) for n, _ in _STACK_NAMES]
        if any(p is None for p in stacks):
            continue
        if all(
            type(p) is torch.nn.Parameter and p.dim() == 3 and p.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and p.device.type != "meta"
            for p in stacks
        ):
            yield name, module


def _quantize_nf4_stack(value, dtype, module):
    from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import _make_expert_params4bit
    p = _make_expert_params4bit(
        value.to(dtype).contiguous(), requires_grad = False, blocksize = 64, quant_type = "nf4",
        compress_statistics = False, quant_storage = torch.uint8, module = module,
    )
    p._original_shape = value.shape
    return p


def quantize_moe_expert_stacks(model, fmt, dtype = None):
    """Round-to-nearest quantize every frozen 16-bit routed-expert stack in place (opt-in, load time).

    NVFP4 / MXFP4 quantize along each stack's contraction dim (one global scale per expert for NVFP4); NF4 uses the
    bitsandbytes 4-bit expert path. Returns the number of stacks replaced; layouts that cannot be told are left
    alone."""
    if fmt not in EXPERTS_QUANTIZE_FORMATS:
        raise ValueError(f"Unsloth: unknown expert quantization {fmt!r}")
    replaced = 0
    for _, experts in list(iter_dense_expert_stacks(model)):
        plan = []
        for name, proj in _STACK_NAMES:
            param = experts._parameters[name]
            transposed = stack_rows_are_transposed(experts, name, proj)
            if transposed is None:
                plan = []
                break
            plan.append((name, param, transposed))
        if not plan:
            continue
        if fmt == "nf4" and not plan[0][1].is_cuda:
            continue   # bitsandbytes quantizes on the GPU
        if fmt != "nf4":
            mult = 16 if fmt == "nvfp4" else 32
            if any((p.shape[-2] if t else p.shape[-1]) % mult for _, p, t in plan):
                continue
        new = {}
        with torch.no_grad():
            for name, param, transposed in plan:
                compute = dtype or param.dtype
                if fmt == "nf4":
                    new[name] = _quantize_nf4_stack(param.data, compute, experts)
                    continue
                rows = param.data.transpose(-2, -1) if transposed else param.data
                quantize = quantize_experts_nvfp4 if fmt == "nvfp4" else quantize_experts_mxfp4
                new[name] = quantize(rows, compute, transposed = transposed)
        for name, value in new.items():
            experts._parameters[name] = value
        if fmt == "nf4":
            try:
                from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import _route_generic_bnb4bit_experts_class
                _route_generic_bnb4bit_experts_class(experts)
            except Exception:
                pass
        replaced += len(new)
    return replaced


def packed_expert_memory(model):
    """{format: bytes} of the packed expert stacks in ``model`` (diagnostics / tests)."""
    out = {}
    for module in model.modules():
        for param in module._parameters.values():
            if isinstance(param, _PackedExpertParam):
                out[param.packed_format] = out.get(param.packed_format, 0) + param.nbytes_packed()
    return out
