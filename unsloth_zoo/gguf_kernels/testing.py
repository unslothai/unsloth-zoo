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

"""Random GGUF blocks for tests and benchmarks: random codes, finite float16 scales."""

import torch

from .formats import GGML_BLOCK, GGMLType as T, ggml_type_id

__all__ = ["random_packed", "SCALE_FIELDS"]

# byte offsets of float16 scale fields inside one block
SCALE_FIELDS = {
    T.Q4_0: (0,), T.Q4_1: (0, 2), T.Q5_0: (0,), T.Q5_1: (0, 2), T.Q8_0: (0,),
    T.Q2_K: (80, 82), T.Q3_K: (108,), T.Q4_K: (0, 2), T.Q5_K: (0, 2), T.Q6_K: (208,),
    T.IQ2_XXS: (0,), T.IQ2_XS: (0,), T.IQ2_S: (0,), T.IQ3_XXS: (0,), T.IQ3_S: (0,), T.IQ1_S: (0,),
    T.IQ4_NL: (0,), T.IQ4_XS: (0,), T.TQ1_0: (52,), T.TQ2_0: (64,), T.Q1_0: (0,), T.Q2_0: (0,),
}


def _scales(n, generator, special):
    """float16 scales around 1e-3 (real weight scale range), plus subnormal / zero / max / negative ones."""
    mag = torch.empty(n).uniform_(-12.0, -4.0, generator = generator).exp2()
    sign = torch.where(torch.rand(n, generator = generator) < 0.25, -1.0, 1.0)
    d = (mag * sign).to(torch.float16)
    if special and n >= 8:
        d[:6] = torch.tensor([0.0, -0.0, 6e-8, -6.1e-5, 65504.0, 1.0], dtype = torch.float16)
    return d.view(torch.uint8).reshape(n, 2)


def random_packed(ggml_type, rows, row_length, seed = 0, special = True, device = "cpu"):
    """uint8 tensor (rows, row_bytes) of random blocks of ``ggml_type`` with finite scales."""
    ggml_type = ggml_type_id(ggml_type)
    qk, bs = GGML_BLOCK[ggml_type]
    if row_length % qk:
        raise ValueError(f"row_length {row_length} is not a multiple of {qk}")
    g = torch.Generator().manual_seed(seed)
    nblk = rows * (row_length // qk)
    if ggml_type in (T.F32, T.F16, T.BF16):
        dt = {T.F32: torch.float32, T.F16: torch.float16, T.BF16: torch.bfloat16}[ggml_type]
        vals = (torch.randn(rows, row_length, generator = g) * 0.05).to(dt)
        if special and vals.numel() >= 8:
            flat = vals.view(-1)
            flat[:6] = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), 1e-30, -2.5]).to(dt)
        return vals.contiguous().view(torch.uint8).reshape(rows, -1).to(device)
    blocks = torch.randint(0, 256, (nblk, bs), dtype = torch.uint8, generator = g)
    for off in SCALE_FIELDS.get(ggml_type, ()):
        blocks[:, off:off + 2] = _scales(nblk, g, special)
    if ggml_type == T.IQ1_M:
        # the float16 scale is spread over the top nibbles of four uint16 words
        d = _scales(nblk, g, special).view(torch.int16).to(torch.int32) & 0xFFFF
        words = blocks[:, 48:56].contiguous().view(torch.int16).to(torch.int32) & 0x0FFF
        for i in range(4):
            words[:, i] |= ((d[:, 0] >> (4 * i)) & 0xF) << 12
        words = words - ((words >> 15) << 16)
        blocks[:, 48:56] = words.to(torch.int16).contiguous().view(torch.uint8).reshape(nblk, 8)
    if ggml_type == T.F8_E4M3 and not special:
        nan = (blocks & 0x7F) == 0x7F
        blocks[nan] = blocks[nan] - 1  # 0x7F / 0xFF are NaN codes
    if ggml_type == T.MXFP4:
        blocks[:, 0] = torch.randint(110, 135, (nblk,), dtype = torch.uint8, generator = g)
        if special and nblk >= 4:
            blocks[:4, 0] = torch.tensor([0, 1, 2, 254], dtype = torch.uint8)
    return blocks.reshape(rows, -1).to(device)
