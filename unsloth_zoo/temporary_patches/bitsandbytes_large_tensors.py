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

"""Chunked 4-bit quantize / dequantize for tensors past bitsandbytes' int32 element count."""
import inspect
import math

import torch

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import logger

__all__ = ["patch_bitsandbytes_large_tensors", "BNB_INT32_ELEMENT_LIMIT"]

BNB_INT32_ELEMENT_LIMIT = 2**31 - 1
# 2^30 elements: half the int32 limit, so packed rows and absmax offsets stay in range too.
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

    # Below the limit pass args through untouched: bitsandbytes 0.45 / 0.46 reject blocksize=None.
    quantize_signature = inspect.signature(original_quantize)
    dequantize_signature = inspect.signature(original_dequantize)

    def quantize_4bit(A, *args, **kwargs):
        n = A.numel()
        if n <= BNB_INT32_ELEMENT_LIMIT:
            return original_quantize(A, *args, **kwargs)
        bound = quantize_signature.bind(A, *args, **kwargs)
        bound.apply_defaults()
        return _quantize_large(A, **{k: v for k, v in bound.arguments.items() if k != "A"})

    def _quantize_large(
        A, absmax = None, out = None, blocksize = None, compress_statistics = False,
        quant_type = "fp4", quant_storage = torch.uint8,
    ):
        n = A.numel()
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

    def dequantize_4bit(A, *args, **kwargs):
        quant_state = kwargs["quant_state"] if "quant_state" in kwargs else (args[0] if len(args) > 0 else None)
        if quant_state is not None:
            n = math.prod(quant_state.shape)
        else:
            out = kwargs["out"] if "out" in kwargs else (args[2] if len(args) > 2 else None)
            n = out.numel() if out is not None else 0
        if n <= BNB_INT32_ELEMENT_LIMIT:
            return original_dequantize(A, *args, **kwargs)
        bound = dequantize_signature.bind(A, *args, **kwargs)
        bound.apply_defaults()
        return _dequantize_large(A, **{k: v for k, v in bound.arguments.items() if k != "A"})

    def _dequantize_large(A, quant_state = None, absmax = None, out = None, blocksize = None, quant_type = "fp4"):
        if quant_state is None:
            n = out.numel()
        else:
            n = math.prod(quant_state.shape)
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
        # reshape(-1) of a non-contiguous out copies: fill a scratch buffer, then copy back.
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
