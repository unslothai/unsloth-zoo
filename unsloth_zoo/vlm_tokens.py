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

"""Single source of truth for VLM placeholder tokens.

These are the image/video/audio special tokens that stand in for media content
and must never contribute to the language-modeling loss. Both backends consume
these lists so they cannot drift apart: the CUDA collator via
``unsloth_zoo.vision_utils`` and the MLX trainer via ``unsloth_zoo.mlx.utils``.
Keep this module import-light (no torch / mlx) so either backend can import it.
"""

__all__ = [
    "IMAGE_TOKENS",
    "AUDIO_TOKENS",
    "VLM_PLACEHOLDER_TOKENS",
]

# Image / vision placeholder tokens, keyed by the models that use them.
IMAGE_TOKENS = [
    "<|image|>",          # Llama 3.2 Vision, Phi 3.5, Gemma 4
    "<|vision_start|>",   # Qwen
    "<|vision_end|>",     # Qwen
    "<|vision_pad|>",     # Qwen
    "<|image_pad|>",      # Qwen
    "<|video_pad|>",      # Qwen
    "<image>",            # PaliGemma, Llava, InternVL
    "</image>",           # InternVL
    "[IMG]",              # Mistral
    "[IMG_BREAK]",        # Mistral
    "[IMG_END]",          # Mistral
    "<image_soft_token>", # Gemma 3 / 3n
    "<start_of_image>",   # Gemma 3 / 3n
    "<end_of_image>",     # Gemma 3 / 3n
    "<|image>",           # Gemma 4 (begin image; <|image|> already listed above)
    "<image|>",           # Gemma 4 (end image)
    "<|video|>",          # Gemma 4
    "<|START_OF_IMG|>",   # Cohere
    "<|END_OF_IMG|>",     # Cohere
    "<|IMG_LINE_BREAK|>", # Cohere
    "<|IMG_PATCH|>",      # Cohere
    "<img>",              # InternVL / Nemotron Nano Omni (begin image)
    "</img>",             # InternVL / Nemotron Nano Omni (end image)
    "<video>",            # Nemotron Nano Omni
    "<|IMAGE|>",          # Qwen2.5-Omni
    "<|VIDEO|>",          # Qwen2.5-Omni
    "<|vision_bos|>",     # Qwen2.5-Omni (begin vision)
    "<|vision_eos|>",     # Qwen2.5-Omni (end vision)
]

# Audio placeholder tokens.
AUDIO_TOKENS = [
    "<|audio|>",          # Gemma 4
    "<|audio>",           # Gemma 4 (begin audio)
    "<audio|>",           # Gemma 4 (end audio)
    "<audio_soft_token>", # Gemma 3n
    "<start_of_audio>",   # Gemma 3n (begin audio)
    "<end_of_audio>",     # Gemma 3n (end audio)
    "<so_embedding>",     # Nemotron Nano Omni (sound)
    "<so_start>",         # Nemotron Nano Omni (begin sound)
    "<so_end>",           # Nemotron Nano Omni (end sound)
    "<|AUDIO|>",          # Qwen2.5-Omni
    "<|audio_bos|>",      # Qwen2.5-Omni (begin audio)
    "<|audio_eos|>",      # Qwen2.5-Omni (end audio)
]

# Combined list for callers that mask every media placeholder at once.
VLM_PLACEHOLDER_TOKENS = IMAGE_TOKENS + AUDIO_TOKENS
