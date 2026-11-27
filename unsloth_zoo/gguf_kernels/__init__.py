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

"""GGUF quant kernels: keep GGUF weights packed and run them through Triton dequant / fused GEMM kernels.

    from unsloth_zoo.gguf_kernels import dequantize_gguf, gguf_linear, gguf_grouped_mm, GGMLType
    y = gguf_linear(x, packed, GGMLType.Q4_K, (out_features, in_features), bias)
"""

from .formats import (
    GGML_BLOCK,
    GGML_TYPE_NAMES,
    GGMLType,
    SUPPORTED_TYPES,
    ggml_packed_bytes,
    ggml_row_bytes,
    ggml_type_id,
)
from .reference import dequantize_reference
from .ops import (
    dequantize_gguf,
    gguf_grouped_mm,
    gguf_kernels_available,
    gguf_linear,
    gguf_linear_backward_input,
)

__all__ = [
    "GGML_BLOCK",
    "GGML_TYPE_NAMES",
    "GGMLType",
    "SUPPORTED_TYPES",
    "ggml_packed_bytes",
    "ggml_row_bytes",
    "ggml_type_id",
    "dequantize_reference",
    "dequantize_gguf",
    "gguf_grouped_mm",
    "gguf_kernels_available",
    "gguf_linear",
    "gguf_linear_backward_input",
]
