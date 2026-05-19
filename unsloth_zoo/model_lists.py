# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Shared cross-platform model lists.

Lives in its own dependency-free module so it can be imported during
unsloth_zoo's early init without dragging in compiler.py's torch/triton
imports.
"""

__all__ = ["FORCE_FLOAT32"]


# Architectures whose activations exceed fp16's finite range (~6.5e4) and
# must run in bf16 or fp32. Loaded as float16 they silently NaN/Inf at
# training time. Shared source of truth for the CUDA loader
# (unsloth/models/loader.py) and the MLX loader (unsloth_zoo/mlx/loader.py).
FORCE_FLOAT32 = [
    "gemma3,",     # trailing comma so "gemma3" doesn't match "gemma3n"
    "gemma3text",  # EmbeddingGemma / standalone text-only Gemma3
    "gemma3n",
    "gpt_oss",
    "qwen3_5",     # Qwen3.5 GDN layers NaN on fp16
]
