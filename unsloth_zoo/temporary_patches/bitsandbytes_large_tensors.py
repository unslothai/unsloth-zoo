# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
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
"""4-bit quantize / dequantize of tensors with 2^31 elements or more.

bitsandbytes passes the element count of `quantize_4bit` and `dequantize_4bit`
to its CUDA kernels as a C `int32`. A fused expert stack larger than that,
such as Inkling-Small's gate_up_proj (256 experts x 4096 x 4096 = 4.29e9
values), wraps the count negative and the load stops with
"Error invalid argument at line 74 in file /src/csrc/ops.cu".

Block-wise quantization is independent per block, so a tensor is quantized
here in pieces whose sizes are multiples of the block size and below the
limit, and the pieces' packed bytes and absmax are concatenated. The nested
absmax compression (`compress_statistics`) is applied once over the joined
absmax, exactly as bitsandbytes does for a small tensor, so the result is
bit-identical to what one kernel call would produce. Dequantization resolves
the (possibly nested) absmax once and dequantizes the same pieces. Tensors
below the limit take bitsandbytes' own path untouched.
"""
import math

import torch

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import logger

__all__ = ["patch_bitsandbytes_large_tensors", "BNB_INT32_ELEMENT_LIMIT"]

BNB_INT32_ELEMENT_LIMIT = 2**31 - 1
# Pieces of at most 2^30 elements: half the limit, so the packed row count and
# absmax offsets of every piece stay far inside int32 as well.
_PIECE_ELEMENTS = 2**30


def _piece_elements(blocksize: int, quant_storage_itemsize: int) -> int:
    unit = blocksize * quant_storage_itemsize * 2
    return max(unit, (_PIECE_ELEMENTS // unit) * unit)


def patch_bitsandbytes_large_tensors():
    try:
        import bitsandbytes as bnb
        from bitsandbytes import functional as F
        from bitsandbytes.functional import QuantState
    except Exception:
        return
    if getattr(F.quantize_4bit, "_unsloth_large_tensor_patch", False):
        return

    original_quantize = F.quantize_4bit
    original_dequantize = F.dequantize_4bit

    def quantize_4bit(
        A, absmax = None, out = None, blocksize = None, compress_statistics = False,
        quant_type = "fp4", quant_storage = torch.uint8,
    ):
        n = A.numel()
        if n <= BNB_INT32_ELEMENT_LIMIT:
            return original_quantize(
                A, absmax = absmax, out = out, blocksize = blocksize,
                compress_statistics = compress_statistics, quant_type = quant_type, quant_storage = quant_storage,
            )
        if blocksize is None:
            blocksize = 64
        input_shape = A.shape
        flat = A.contiguous().reshape(-1)
        piece = _piece_elements(blocksize, quant_storage.itemsize)
        packed, absmaxes = [], []
        for start in range(0, n, piece):
            part = flat[start : start + piece]
            data, state = original_quantize(
                part, blocksize = blocksize, compress_statistics = False,
                quant_type = quant_type, quant_storage = quant_storage,
            )
            packed.append(data.reshape(-1))
            absmaxes.append(state.absmax)
            code = state.code
        del flat
        data = torch.cat(packed).reshape(-1, 1)
        full_absmax = torch.cat(absmaxes)
        del packed, absmaxes
        if compress_statistics:
            offset = full_absmax.mean()
            qabsmax, state2 = F.quantize_blockwise(full_absmax - offset, blocksize = 256)
            del full_absmax
            state = QuantState(
                absmax = qabsmax, shape = input_shape, dtype = A.dtype, blocksize = blocksize,
                code = code, quant_type = quant_type, offset = offset, state2 = state2,
            )
        else:
            state = QuantState(
                absmax = full_absmax, shape = input_shape, dtype = A.dtype, blocksize = blocksize,
                code = code, quant_type = quant_type,
            )
        if out is not None:
            out = out.copy_(data)
        else:
            out = data
        if absmax is not None:
            absmax.copy_(state.absmax)
            state.absmax = absmax
        return out, state

    def dequantize_4bit(A, quant_state = None, absmax = None, out = None, blocksize = None, quant_type = "fp4"):
        if quant_state is None:
            n = out.numel() if out is not None else 0
        else:
            n = math.prod(quant_state.shape)
        if n <= BNB_INT32_ELEMENT_LIMIT:
            return original_dequantize(
                A, quant_state = quant_state, absmax = absmax, out = out, blocksize = blocksize, quant_type = quant_type,
            )
        if quant_state is None:
            quant_state = QuantState(
                absmax = absmax, shape = out.shape, dtype = out.dtype, blocksize = blocksize or 64,
                code = F.get_4bit_type(quant_type, device = A.device), quant_type = quant_type,
            )
        if quant_state.nested:
            full_absmax = F.dequantize_blockwise(quant_state.absmax, quant_state.state2)
            full_absmax = full_absmax + quant_state.offset
            if full_absmax.dtype != torch.float32:
                full_absmax = full_absmax.float()
        else:
            full_absmax = quant_state.absmax
        bs = quant_state.blocksize
        storage_itemsize = A.element_size()
        values_per_storage = storage_itemsize * 2
        piece = _piece_elements(bs, storage_itemsize)
        flat_packed = A.reshape(-1)
        # A non-contiguous out would make reshape(-1) a copy, so fill a scratch buffer and copy back.
        in_place = out is not None and out.is_contiguous()
        result = out.view(-1) if in_place else torch.empty(n, device = A.device, dtype = quant_state.dtype)
        for start in range(0, n, piece):
            count = min(piece, n - start)
            sub_state = QuantState(
                absmax = full_absmax[start // bs : (start + count + bs - 1) // bs],
                shape = torch.Size((count,)), dtype = quant_state.dtype, blocksize = bs,
                code = quant_state.code, quant_type = quant_state.quant_type,
            )
            sub_packed = flat_packed[start // values_per_storage : (start + count + values_per_storage - 1) // values_per_storage]
            result[start : start + count] = original_dequantize(sub_packed.reshape(-1, 1), quant_state = sub_state).reshape(-1)
        result = result.reshape(quant_state.shape)
        if out is not None:
            if not in_place:
                out.copy_(result.reshape(out.shape))
            result = out
        # Same orientation as bitsandbytes, which transposes a row-packed A with or without out.
        if A.shape[0] == 1:
            return result.t()
        return result

    quantize_4bit._unsloth_large_tensor_patch = True
    dequantize_4bit._unsloth_large_tensor_patch = True
    quantize_4bit.__wrapped__ = original_quantize
    dequantize_4bit.__wrapped__ = original_dequantize
    F.quantize_4bit = quantize_4bit
    F.dequantize_4bit = dequantize_4bit
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched bitsandbytes quantize_4bit / dequantize_4bit for tensors with 2^31 elements or more.")
pass

TEMPORARY_PATCHES.append(patch_bitsandbytes_large_tensors)
