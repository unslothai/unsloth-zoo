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
import math
from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING, logger
from .utils import patch_function, raise_error

def patch_convert_moe_packed_tensors():
    """
    Pin the original GPU-optimized version of convert_moe_packed_tensors with smaller default chunk size.
    """
    try:
        import transformers.integrations.mxfp4
        from transformers.integrations.mxfp4 import FP4_VALUES
    except Exception as e:
        return raise_error("transformers.integrations.mxfp4", e)

    def convert_moe_packed_tensors(
        blocks,
        scales,
        *,
        dtype: torch.dtype = torch.bfloat16,
        rows_per_chunk: int = 32768 * 1024,
    ) -> torch.Tensor:
        """
        Convert the mxfp4 weights again, dequantizing and makes them compatible with the forward
        pass of GPT_OSS.

        Args:
            blocks: Packed quantized weights
            scales: Quantization scales
            dtype: Output data type
            rows_per_chunk: Number of rows to process per chunk. .
        """
        # Check if blocks and scales are on CPU, and move to GPU if so
        if not blocks.is_cuda and torch.cuda.is_available():
            blocks = blocks.cuda()
            scales = scales.cuda()

        scales = scales.to(torch.int32) - 127

        assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"

        lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

        *prefix_shape, G, B = blocks.shape
        rows_total = math.prod(prefix_shape) * G

        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

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
        return out
    patch_function(transformers.integrations.mxfp4, "convert_moe_packed_tensors", convert_moe_packed_tensors)

    """
    Transformers 4.55.4 did dequantized.transpose(1, 2).contiguous().to(target_device)
    but new versions > 4.56.0 removed the transpose(1, 2) and moved it into patch_convert_moe_packed_tensors
    """
    try:
        import transformers.integrations.mxfp4
        from transformers.integrations.tensor_parallel import shard_and_distribute_module
    except Exception as e:
        return raise_error("transformers.integrations.mxfp4.dequantize", e)

    def dequantize(module, param_name, param_value, target_device, dq_param_name, **kwargs):
        model = kwargs.get("model", None)
        empty_param = kwargs.get("empty_param", None)
        casting_dtype = kwargs.get("casting_dtype", None)
        to_contiguous = kwargs.get("to_contiguous", None)
        rank = kwargs.get("rank", None)
        device_mesh = kwargs.get("device_mesh", None)

        for proj in ["gate_up_proj", "down_proj"]:
            if proj in param_name:
                if device_mesh is not None:
                    param_value = shard_and_distribute_module(
                        model,
                        param_value,
                        empty_param,
                        dq_param_name,
                        casting_dtype,
                        to_contiguous,
                        rank,
                        device_mesh,
                        set_param=False,
                    )
                blocks_attr = f"{proj}_blocks"
                scales_attr = f"{proj}_scales"
                setattr(module, param_name.rsplit(".", 1)[1], param_value)
                if hasattr(module, blocks_attr) and hasattr(module, scales_attr):
                    dequantized = convert_moe_packed_tensors(getattr(module, blocks_attr), getattr(module, scales_attr))
                    # [HERE] we must do transpose(1, 2)
                    dequantized = dequantized.transpose(1, 2).contiguous().to(target_device)
                    # TODO: this is perhaps necessary since if target_device is cpu, and the param was on gpu
                    if target_device == "cpu" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    setattr(module, proj, torch.nn.Parameter(dequantized))
                    delattr(module, blocks_attr)
                    delattr(module, scales_attr)
    patch_function(transformers.integrations.mxfp4, "dequantize", dequantize)

    """
    Add a new CPU-optimized version of convert_moe_packed_tensors with smaller default chunk size.
    """
    try:
        import transformers.integrations.mxfp4
        from transformers.integrations.mxfp4 import FP4_VALUES
    except Exception as e:
        return raise_error("transformers.integrations.mxfp4_CPU", e)

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
        return out

    # Add the new CPU function to the mxfp4 module
    if hasattr(transformers.integrations.mxfp4, 'convert_moe_packed_tensors'):
        transformers.integrations.mxfp4.convert_moe_packed_tensors_cpu = convert_moe_packed_tensors_cpu
        if UNSLOTH_ENABLE_LOGGING:
            logger.info("Unsloth: Successfully added convert_moe_packed_tensors_cpu function.")
    else:
        if UNSLOTH_ENABLE_LOGGING:
            logger.info("Unsloth: Failed to add convert_moe_packed_tensors_cpu - original function not found.")
pass
TEMPORARY_PATCHES.append(patch_convert_moe_packed_tensors)
