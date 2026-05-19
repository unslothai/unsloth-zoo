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
#
# Entries are HuggingFace ``config.json`` ``model_type`` strings (the same
# values returned by ``unsloth_zoo.hf_utils.get_transformers_model_type``).
# Examples on the Hub:
#   google/gemma-3-*            -> "gemma3"
#   google/embeddinggemma-*     -> "gemma3_text"   (matches "gemma3text" after _ stripping)
#   google/gemma-3n-*           -> "gemma3n"
#   openai/gpt-oss-*            -> "gpt_oss"
#   Qwen/Qwen3.5-*              -> "qwen3_5"
# The CUDA loader matches via substring against a comma-joined model_types
# string, so the trailing comma on "gemma3," keeps it from prefix-matching
# "gemma3n". The MLX helper ``_is_force_float32_arch`` strips ``-``/``_``
# and the trailing comma before comparing, so ``"gpt-oss"`` / ``"gpt_oss"``
# and ``"gemma3_text"`` / ``"gemma3text"`` all resolve correctly.
FORCE_FLOAT32 = [
    "gemma3,",     # trailing comma is a substring-path delimiter for the CUDA loader
    "gemma3text",  # EmbeddingGemma / standalone text-only Gemma3 (config: "gemma3_text")
    "gemma3n",
    "gpt_oss",
    "qwen3_5",     # Qwen3.5 GDN layers NaN on fp16
]
