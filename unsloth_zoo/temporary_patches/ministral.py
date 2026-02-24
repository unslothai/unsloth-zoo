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
from typing import Optional, Callable
from .common import TEMPORARY_PATCHES
from .utils import (
    patch_function,
    KWARGS_TYPE,
    raise_error,
    Cache,
)

def patch_MinistralAttention():
    """
    Fix dtype mismatch in MinistralAttention where RoPE is applied to Q/K
    but not V. In 4-bit (BNB) mode, cos/sin can promote Q/K to a different
    dtype than V, causing scaled_dot_product_attention to fail with:
      ValueError: Expected query, key, and value to have the same dtype
    """
    try:
        import transformers.models.ministral.modeling_ministral
        from transformers.models.ministral.modeling_ministral import (
            apply_rotary_pos_emb,
            eager_attention_forward,
            ALL_ATTENTION_FUNCTIONS,
        )
    except Exception as e:
        return raise_error("MinistralAttention", e)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: KWARGS_TYPE,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Fix dtype mismatch: RoPE may promote Q/K dtype via cos/sin multiplication
        # while V stays in the original dtype from the linear projection.
        # In 4-bit BNB mode, this causes Q/K to be float32 while V is float16/bfloat16.
        if value_states.dtype != query_states.dtype:
            value_states = value_states.to(query_states.dtype)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    patch_function(
        transformers.models.ministral.modeling_ministral.MinistralAttention,
        "forward",
        forward,
        match_level="relaxed",
    )
pass
TEMPORARY_PATCHES.append(patch_MinistralAttention)


def patch_MinistralModel_forward():
    """
    Fix MinistralModel.forward to handle sliding_window=None gracefully.
    When the config has no sliding_window set (or it is None), the model
    should skip creating a sliding window causal mask since calling
    create_sliding_window_causal_mask raises ValueError.
    This also handles the case where all layer_types are 'full_attention'.
    """
    try:
        from transformers.models.ministral.modeling_ministral import MinistralModel
    except Exception as e:
        return raise_error("MinistralModel", e)

    original_forward = MinistralModel.forward

    def patched_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        *args,
        **kwargs,
    ):
        # Check if sliding_window is properly set
        sw = getattr(self.config, "sliding_window", None)
        has_sliding_layers = any(
            lt == "sliding_attention"
            for lt in (getattr(self.config, "layer_types", None) or [])
        )
        if sw is None and not has_sliding_layers:
            # All layers are full_attention and no sliding_window is set.
            # Set sliding_window to max_position_embeddings so
            # create_sliding_window_causal_mask does not raise. The mask will
            # not actually be used by full_attention layers, but the value must
            # be large enough to avoid unintended attention truncation.
            self.config.sliding_window = getattr(
                self.config, "max_position_embeddings", 32768
            )
            try:
                return original_forward(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    *args,
                    **kwargs,
                )
            finally:
                self.config.sliding_window = None
        return original_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            *args,
            **kwargs,
        )

    patch_function(
        MinistralModel,
        "forward",
        patched_forward,
        match_level="relaxed",
    )
pass
TEMPORARY_PATCHES.append(patch_MinistralModel_forward)
