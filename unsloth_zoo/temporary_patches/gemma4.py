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

import torch
import os
from .common import TEMPORARY_PATCHES
from .utils import raise_error


# ============================================================================
# Gemma4 AudioAttention patch - fix attention_invalid_logits_value overflow in fp16
# The config value is -1e9 which overflows fp16 max (65504).
# On Tesla T4, autocast can downcast attn_weights to fp16, causing masked_fill to fail.
# ============================================================================

def patch_Gemma4AudioAttention():
    try:
        import transformers.models.gemma4.modeling_gemma4
        Gemma4AudioAttention = getattr(
            transformers.models.gemma4.modeling_gemma4, "Gemma4AudioAttention", None
        )
        if Gemma4AudioAttention is None:
            return
    except Exception as e:
        return raise_error("Gemma4AudioAttention.forward", e)

    _original_audio_attn_forward = Gemma4AudioAttention.forward

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        # Clamp attention_invalid_logits_value to dtype-safe range before attention
        # Only needed for fp16 (Tesla T4) where -1e9 overflows. bf16 supports up to ~3.4e38.
        original_value = getattr(self.config, "attention_invalid_logits_value", -1e9)
        needs_clamp = hidden_states.dtype == torch.float16 and abs(original_value) > 65000.0
        if needs_clamp:
            self.config.attention_invalid_logits_value = -65000.0
        try:
            result = _original_audio_attn_forward(self, hidden_states, position_embeddings, attention_mask, **kwargs)
        finally:
            if needs_clamp:
                self.config.attention_invalid_logits_value = original_value
        return result
    pass
    Gemma4AudioAttention.forward = forward
pass
TEMPORARY_PATCHES.append(patch_Gemma4AudioAttention)
