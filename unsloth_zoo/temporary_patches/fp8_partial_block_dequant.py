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
"""FP8 dequant of partial edge blocks (huggingface/transformers#50603): stock derives block size
from scale grid; keep this file self-contained, the CPU backend loads it alone"""
import math

import torch

from .common import TEMPORARY_PATCHES

__all__ = [
    "patch_fp8_dequantize_partial_blocks",
]

_FP8_DTYPES = tuple(getattr(torch, n) for n in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, n))
# transformers 5.17 casts a checkpoint's FP8 codes to the parameter dtype before Fp8Dequantize (5.23 keeps FP8);
# the cast is exact, so widened codes take the same block fix. Packed FP4 codes are integer and stay excluded.
_CODE_DTYPES = _FP8_DTYPES + (torch.float32, torch.bfloat16, torch.float16)


def _configured_block(op):
    config = getattr(getattr(op, "hf_quantizer", None), "quantization_config", None)
    block = config.get("weight_block_size") if isinstance(config, dict) else getattr(config, "weight_block_size", None)
    if not isinstance(block, (list, tuple)) or len(block) != 2:
        return None
    try:
        block_m, block_n = int(block[0]), int(block[1])
    except (TypeError, ValueError):
        return None
    return (block_m, block_n) if block_m > 0 and block_n > 0 else None


def _partial_block(op, quantized, scales):
    """(block_m, block_n) when the configured blocks tile `quantized` with a partial edge matching `scales`, else None."""
    if not isinstance(quantized, torch.Tensor) or quantized.dtype not in _CODE_DTYPES or quantized.dim() < 2:
        return None
    if not isinstance(scales, torch.Tensor) or scales.dim() < 2:
        return None
    block = _configured_block(op)
    if block is None:
        return None
    rows, cols = quantized.shape[-2:]
    if tuple(scales.shape[-2:]) != (math.ceil(rows / block[0]), math.ceil(cols / block[1])):
        return None
    if rows % block[0] == 0 and cols % block[1] == 0:
        return None  # whole blocks: the stock derivation is right
    if scales.shape[:-2] != quantized.shape[:-2]:
        return None
    return block


def patch_fp8_dequantize_partial_blocks():
    try:
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize
    except ImportError:
        return
    original = getattr(Fp8Dequantize, "_dequantize_one", None)
    if original is None or getattr(original, "_unsloth_partial_block_patched", False):
        return

    def _dequantize_one(self, quantized, scales, output_dtype = None, *args, **kwargs):
        block = _partial_block(self, quantized, scales)
        if block is None:
            return original(self, quantized, scales, output_dtype, *args, **kwargs)
        rows, cols = quantized.shape[-2:]
        if output_dtype is None:  # same default as the stock path
            output_dtype = scales.dtype if scales.dtype.is_floating_point and scales.element_size() >= 2 else torch.bfloat16
        s = (scales.to(torch.float32) - 127.0).exp2() if scales.dtype == torch.uint8 else scales.to(torch.float32)
        s = s.repeat_interleave(block[0], dim = -2)[..., :rows, :].repeat_interleave(block[1], dim = -1)[..., :cols]
        return (quantized.to(torch.float32) * s).to(output_dtype)

    _dequantize_one._unsloth_partial_block_patched = True
    _dequantize_one._unsloth_original = original
    Fp8Dequantize._dequantize_one = _dequantize_one
pass


TEMPORARY_PATCHES.append(patch_fp8_dequantize_partial_blocks)
