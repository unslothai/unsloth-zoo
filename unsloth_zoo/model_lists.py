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

"""Shared cross-platform model lists; dependency-free, so early init can import it without pulling in compiler.py's torch or triton imports."""

__all__ = ["FORCE_FLOAT32"]


# Architectures whose activations overflow fp16 (~6.5e4) and need bf16 or fp32, else NaN/Inf in training; entries are
# config.json model_type strings shared by CUDA and MLX loaders; trailing comma stops gemma3 prefix-matching gemma3n;
# MLX strips -, _ and the comma.
FORCE_FLOAT32 = [
    "embedding_gemma2",  # fp16 unsupported upstream; text-only loads have no gemma4 sub-config to match
    "embedding_gemma2_text",  # explicit entry for the MLX exact-match helper
    "gemma3,",     # trailing comma is a substring-path delimiter for the CUDA loader
    "gemma3text",  # EmbeddingGemma / standalone text-only Gemma3 (config: "gemma3_text")
    "gemma3n",
    "gemma4",      # substring-matches "gemma4" and "gemma4_text" on CUDA; fp16 NaNs the grad_norm in the backward (targeted float32 via gemma4_float32.py)
    "gemma4text",  # standalone text-only Gemma4 (config: "gemma4_text") for the MLX exact-match helper
    "glm4_moe",    # GLM-4.x MoE (glm4_moe / glm4_moe_lite on CUDA): fp16 NaNs grad_norm; bf16 load is dtype-consistent
    "glm4_moe_lite",  # explicit entry for the MLX exact-match helper
    "gpt_oss",
    "qwen3_5",     # Qwen3.5 GDN layers NaN on fp16
    "qwen3_moe",   # Qwen3-MoE (Qwen3-30B-A3B): fp16 NaNs grad_norm in the backward (targeted float32 via qwen3_moe_float32.py)
]
