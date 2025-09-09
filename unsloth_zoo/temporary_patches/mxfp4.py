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
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import re
from typing import Union, List, Optional, Tuple
import inspect
import torch
import torch.nn as nn
import os
import logging
import math

from .common import TEMPORARY_PATCHES, torch_compile_options, UNSLOTH_ENABLE_LOGGING

logger = logging.getLogger(__name__)



def patch_convert_moe_packed_tensors_cpu():
    """
    Add a new CPU-optimized version of convert_moe_packed_tensors with smaller default chunk size.
    """
    try:
        import transformers.integrations.mxfp4
        from transformers.integrations.mxfp4 import FP4_VALUES
    except:
        return

    def convert_moe_packed_tensors_cpu(
        blocks,
        scales,
        *,
        dtype: torch.dtype = torch.bfloat16,
        rows_per_chunk: int = 1024 * 1024,  # CPU-optimized default (~2.6GB temp memory)
    ) -> torch.Tensor:
        """
        Convert the mxfp4 weights again, dequantizing and makes them compatible with the forward
        pass of GPT_OSS. CPU-optimized version with smaller default chunk size.

        Args:
            blocks: Packed quantized weights
            scales: Quantization scales
            dtype: Output data type
            rows_per_chunk: Number of rows to process per chunk. CPU-optimized default: 1M rows.
                           Memory usage per chunk (assuming B=128):
                           - 8192: ~22 MB
                           - 1048576 (1M): ~2.6 GB
                           - 33554432 (32M): ~90 GB
        """
        # Ensure tensors are on CPU
        if blocks.is_cuda:
            blocks = blocks.cpu()
        if scales.is_cuda:
            scales = scales.cpu()

        scales = scales.to(torch.int32) - 127

        assert blocks.shape[:-1] == scales.shape, f"{blocks.shape[:-1]=} does not match {scales.shape=}"

        # Create LUT on CPU
        lut = torch.tensor(FP4_VALUES, dtype=dtype, device='cpu')

        *prefix_shape, G, B = blocks.shape
        rows_total = math.prod(prefix_shape) * G

        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        # Create output tensor on CPU
        out = torch.empty(rows_total, B * 2, dtype=dtype, device='cpu')

        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)

            blk = blocks[r0:r1]
            exp = scales[r0:r1]

            # nibble indices -> int64
            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)

            sub = out[r0:r1]
            sub[:, 0::2] = lut[idx_lo]
            sub[:, 1::2] = lut[idx_hi]

            torch.ldexp(sub, exp, out=sub)
            del idx_lo, idx_hi, blk, exp, sub

        out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
        del blocks, scales, lut
        return out.transpose(1, 2).contiguous()

    # Add the new CPU function to the mxfp4 module
    if hasattr(transformers.integrations.mxfp4, 'convert_moe_packed_tensors'):
        transformers.integrations.mxfp4.convert_moe_packed_tensors_cpu = convert_moe_packed_tensors_cpu
        if UNSLOTH_ENABLE_LOGGING:
            print("Unsloth: Successfully added convert_moe_packed_tensors_cpu function.")
    else:
        if UNSLOTH_ENABLE_LOGGING:
            print("Unsloth: Failed to add convert_moe_packed_tensors_cpu - original function not found.")
    return
pass
TEMPORARY_PATCHES.append(patch_convert_moe_packed_tensors_cpu)
