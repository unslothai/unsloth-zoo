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

from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import torch
import torch.nn as nn
import os
from .common import TEMPORARY_PATCHES, torch_compile
from .utils import (
    patch_function,
    process_output_options,
    KWARGS_TYPE,
    raise_error,
    ImageInput,
    PreTokenizedInput,
    TextInput,
    Cache,
    StaticCache,
    HybridCache,
    Unpack,
    patch_function_past_key_values,
    dedent,
)
import inspect

_UNSLOTH_FLEX_ATTENTION_DISABLED = os.environ.get("UNSLOTH_ENABLE_FLEX_ATTENTION", "1") == "0"


def _make_gemma4_attn_forwards(forward_function, has_cache_position):
    """Build past_key_value / past_key_values forward variants for Gemma4TextAttention."""
    functions = []
    if has_cache_position:
        def forward_past_key_value(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_value=None, cache_position=None, **kwargs):
            return forward_function(self, hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)
        def forward_past_key_values(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, cache_position=None, **kwargs):
            return forward_function(self, hidden_states, position_embeddings, attention_mask, past_key_values, cache_position, **kwargs)
    else:
        def forward_past_key_value(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_value=None, **kwargs):
            return forward_function(self, hidden_states, position_embeddings, attention_mask, past_key_value, kwargs.pop("cache_position", None), **kwargs)
        def forward_past_key_values(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
            return forward_function(self, hidden_states, position_embeddings, attention_mask, past_key_values, kwargs.pop("cache_position", None), **kwargs)
    functions.append(forward_past_key_value)
    functions.append(forward_past_key_values)
    return functions


# ============================================================================
# Gemma4 RMSNorm patches
# Key difference from Gemma3: weight * normed (init=1), NOT (1+weight) * normed (init=0)
# Uses torch.pow(mean_sq, -0.5) not torch.rsqrt
# ============================================================================

def patch_Gemma4RMSNorm():
    """FORCE_FLOAT32 path: fp32 internals, clamp to fp16 range, return fp16."""
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.gemma4.modeling_gemma4
        transformers.models.gemma4.modeling_gemma4.Gemma4RMSNorm
    except Exception as e:
        return raise_error("Gemma4RMSNorm.forward", e)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x_fp32 = hidden_states.to(torch.float32)
        mean_squared = x_fp32.pow(2).mean(-1, keepdim=True) + self.eps
        normed = x_fp32 * torch.pow(mean_squared, -0.5)
        if self.with_scale:
            output_fp32 = normed * self.weight.to(torch.float32)
        else:
            output_fp32 = normed
        fp16_max = torch.finfo(torch.float16).max
        fp16_min = torch.finfo(torch.float16).min
        clamped = torch.clamp(output_fp32, min=fp16_min, max=fp16_max)
        return clamped.to(torch.float16)
    pass
    patch_function(transformers.models.gemma4.modeling_gemma4.Gemma4RMSNorm, "forward", forward, fullgraph=True)
pass
TEMPORARY_PATCHES.append(patch_Gemma4RMSNorm)


def patch_Gemma4RMSNorm_generic():
    """Generic path (bf16/non-FORCE_FLOAT32): fp32 internals, return in input dtype."""
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1": return
    try:
        import transformers.models.gemma4.modeling_gemma4
        transformers.models.gemma4.modeling_gemma4.Gemma4RMSNorm
    except Exception as e:
        return raise_error("Gemma4RMSNorm.forward", e)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x_fp32 = hidden_states.to(torch.float32)
        mean_squared = x_fp32.pow(2).mean(-1, keepdim=True) + self.eps
        normed = x_fp32 * torch.pow(mean_squared, -0.5)
        if self.with_scale:
            output_fp32 = normed * self.weight.to(torch.float32)
        else:
            output_fp32 = normed
        return output_fp32.to(hidden_states.dtype)
    pass
    patch_function(transformers.models.gemma4.modeling_gemma4.Gemma4RMSNorm, "forward", forward, fullgraph=True)
pass
TEMPORARY_PATCHES.append(patch_Gemma4RMSNorm_generic)


# ============================================================================
# Gemma4 TextScaledWordEmbedding patch (FORCE_FLOAT32 only)
# ============================================================================

def patch_Gemma4TextScaledWordEmbedding():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.gemma4.modeling_gemma4
        transformers.models.gemma4.modeling_gemma4.Gemma4TextScaledWordEmbedding
    except Exception as e:
        return raise_error("Gemma4TextScaledWordEmbedding.forward", e)

    def forward(self, input_ids: torch.Tensor):
        input_embeds = torch.nn.functional.embedding(
            input_ids,
            weight=self.weight,
            padding_idx=self.padding_idx,
        )
        # Compute in fp32 to avoid overflow (embed_scale can be ~53 for E2B),
        # then clamp and return fp16 to match all downstream Linear layers.
        result_fp32 = input_embeds.to(torch.float32) * self.embed_scale
        fp16_max = torch.finfo(torch.float16).max
        return torch.clamp(result_fp32, min=-fp16_max, max=fp16_max).to(torch.float16)
    pass
    patch_function(transformers.models.gemma4.modeling_gemma4.Gemma4TextScaledWordEmbedding, "forward", forward, fullgraph=True)
pass
TEMPORARY_PATCHES.append(patch_Gemma4TextScaledWordEmbedding)


# ============================================================================
# Gemma4 Attention patches
# Key differences from Gemma3:
#   - KV sharing: is_kv_shared_layer -> fetch from past_key_values.shared_layers
#   - k_eq_v: v_proj is None -> value_states = key_states (BEFORE k_norm)
#   - v_norm: new RMSNorm applied to value_states
#   - variable head_dim (global_head_dim for non-sliding)
#   - store_full_length_kv for shared layer cache
# ============================================================================

def patch_Gemma4TextAttention():
    """FORCE_FLOAT32 path: upcast to fp32 for attention math, return fp16."""
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.gemma4.modeling_gemma4
        transformers.models.gemma4.modeling_gemma4.Gemma4TextAttention
        from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward
    except Exception as e:
        return raise_error("Gemma4TextAttention.forward", e)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
    scaled_dot_product_attention = torch.compiler.disable(scaled_dot_product_attention, recursive=True)
    torch_jit_is_tracing = torch.jit.is_tracing

    def forward_function(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: KWARGS_TYPE,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        num_heads = getattr(self, "num_heads", None) or self.config.num_attention_heads
        use_alt = getattr(self, "use_alternative_attention", False)
        num_kv_heads = (
            self.config.num_global_key_value_heads if use_alt else self.config.num_key_value_heads
        )

        cos, sin = position_embeddings

        # Q projection + norm + RoPE
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states_fp32 = query_states.to(torch.float32)  # q_norm may return fp16
        cos_fp32 = cos.to(torch.float32)
        sin_fp32 = sin.to(torch.float32)
        query_states_fp32 = apply_rotary_pos_emb(query_states_fp32, cos_fp32, sin_fp32, unsqueeze_dim=2)
        query_states_fp32 = query_states_fp32.transpose(1, 2)

        # KV: shared layer reuse or compute fresh
        if getattr(self, "is_kv_shared_layer", False) and past_key_value is not None:
            key_states_fp32, value_states_fp32 = past_key_value.shared_layers[self.kv_shared_layer_index]
            key_states_fp32 = key_states_fp32.to(torch.float32).to(query_states_fp32.device)
            value_states_fp32 = value_states_fp32.to(torch.float32).to(query_states_fp32.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

            key_states_fp32 = key_states.to(torch.float32)
            key_states_fp32 = self.k_norm(key_states_fp32)
            key_states_fp32 = key_states_fp32.to(torch.float32)  # k_norm may return fp16
            key_states_fp32 = apply_rotary_pos_emb(key_states_fp32, cos_fp32, sin_fp32, unsqueeze_dim=2)
            key_states_fp32 = key_states_fp32.transpose(1, 2)

            value_states_fp32 = value_states.to(torch.float32)
            value_states_fp32 = self.v_norm(value_states_fp32)
            value_states_fp32 = value_states_fp32.to(torch.float32)  # v_norm may return fp16
            value_states_fp32 = value_states_fp32.transpose(1, 2)

        # Cache update (match original: no extra cache_kwargs)
        if past_key_value is not None:
            if not getattr(self, "is_kv_shared_layer", False):
                key_states_fp32, value_states_fp32 = past_key_value.update(
                    key_states_fp32, value_states_fp32, self.layer_idx
                )
            if getattr(self, "store_full_length_kv", False):
                if not hasattr(past_key_value, "shared_layers"):
                    past_key_value.shared_layers = {}
                past_key_value.shared_layers[self.layer_idx] = key_states_fp32, value_states_fp32

        # Attention mask
        attn_mask_for_sdpa = attention_mask
        if isinstance(attn_mask_for_sdpa, torch.Tensor) and attn_mask_for_sdpa.dtype != torch.bool:
            attn_mask_for_sdpa = attn_mask_for_sdpa.to(torch.float32)

        attn_impl = getattr(self.config, "_attn_implementation", "sdpa")
        if _UNSLOTH_FLEX_ATTENTION_DISABLED:
            attn_impl = "sdpa"
        if attn_impl == "flex_attention":
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]
            attn_output_fp32, attn_weights = attention_interface(
                self,
                query_states_fp32,
                key_states_fp32,
                value_states_fp32,
                attn_mask_for_sdpa,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=getattr(self, "scaling", None),
                sliding_window=getattr(self, "sliding_window", None),
                **kwargs,
            )
        else:
            is_causal = query_states_fp32.shape[2] > 1 and attn_mask_for_sdpa is None and getattr(self, "is_causal", True)
            if torch_jit_is_tracing() and isinstance(is_causal, torch.Tensor): is_causal = is_causal.item()
            attn_output_fp32 = scaled_dot_product_attention(
                query_states_fp32.contiguous(),
                key_states_fp32.contiguous(),
                value_states_fp32.contiguous(),
                attn_mask=attn_mask_for_sdpa,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
                scale=getattr(self, "scaling", None),
                enable_gqa=getattr(self, "num_key_value_groups", 1) != 1,
            )
            attn_weights = None

        if attn_impl != "flex_attention":
            attn_output_fp32 = attn_output_fp32.transpose(1, 2).contiguous()

        attn_output_fp32 = attn_output_fp32.reshape(*input_shape, -1)
        attn_output_fp16 = attn_output_fp32.to(torch.float16)
        attn_output = self.o_proj(attn_output_fp16)
        return attn_output, attn_weights
    pass

    has_cache_position = "cache_position" in inspect.signature(
        transformers.models.gemma4.modeling_gemma4.Gemma4TextAttention.forward
    ).parameters
    functions = _make_gemma4_attn_forwards(forward_function, has_cache_position)
    patch_function_past_key_values(transformers.models.gemma4.modeling_gemma4.Gemma4TextAttention, "forward", functions, match_level="relaxed")
pass
TEMPORARY_PATCHES.append(patch_Gemma4TextAttention)


def patch_Gemma4TextAttention_generic():
    """Generic path (bf16): no dtype casts, just ensure correct Gemma4 attention flow."""
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1": return
    try:
        import transformers.models.gemma4.modeling_gemma4
        transformers.models.gemma4.modeling_gemma4.Gemma4TextAttention
        from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward
    except Exception as e:
        return raise_error("Gemma4TextAttention.forward", e)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
    scaled_dot_product_attention = torch.compiler.disable(scaled_dot_product_attention, recursive=True)
    torch_jit_is_tracing = torch.jit.is_tracing

    def forward_function(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: KWARGS_TYPE,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        # Q projection + norm + RoPE
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        # KV: shared layer reuse or compute fresh
        if getattr(self, "is_kv_shared_layer", False) and past_key_value is not None:
            key_states, value_states = past_key_value.shared_layers[self.kv_shared_layer_index]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        # Cache update
        if past_key_value is not None:
            if not getattr(self, "is_kv_shared_layer", False):
                try:
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, self.layer_idx
                    )
                except (IndexError, AttributeError):
                    pass  # Cache not yet initialized for this layer; use raw states
            if getattr(self, "store_full_length_kv", False):
                if not hasattr(past_key_value, "shared_layers"):
                    past_key_value.shared_layers = {}
                past_key_value.shared_layers[self.layer_idx] = key_states, value_states

        # Ensure all Q/K/V have the same dtype (apply_rotary_pos_emb may upcast Q/K via fp32 cos/sin)
        target_dtype = query_states.dtype
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

        # Attention mask
        attn_mask_for_sdpa = attention_mask
        if isinstance(attn_mask_for_sdpa, torch.Tensor) and attn_mask_for_sdpa.dtype != torch.bool:
            attn_mask_for_sdpa = attn_mask_for_sdpa.to(target_dtype)

        attn_impl = getattr(self.config, "_attn_implementation", "sdpa")
        if _UNSLOTH_FLEX_ATTENTION_DISABLED:
            attn_impl = "sdpa"
        if attn_impl == "flex_attention":
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attn_mask_for_sdpa,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=getattr(self, "scaling", None),
                sliding_window=getattr(self, "sliding_window", None),
                **kwargs,
            )
        else:
            is_causal = query_states.shape[2] > 1 and attn_mask_for_sdpa is None and getattr(self, "is_causal", True)
            if torch_jit_is_tracing() and isinstance(is_causal, torch.Tensor): is_causal = is_causal.item()
            attn_output = scaled_dot_product_attention(
                query_states.contiguous(),
                key_states.contiguous(),
                value_states.contiguous(),
                attn_mask=attn_mask_for_sdpa,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
                scale=getattr(self, "scaling", None),
                enable_gqa=getattr(self, "num_key_value_groups", 1) != 1,
            )
            attn_weights = None

        if attn_impl != "flex_attention":
            attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    pass

    has_cache_position = "cache_position" in inspect.signature(
        transformers.models.gemma4.modeling_gemma4.Gemma4TextAttention.forward
    ).parameters
    functions = _make_gemma4_attn_forwards(forward_function, has_cache_position)
    patch_function_past_key_values(transformers.models.gemma4.modeling_gemma4.Gemma4TextAttention, "forward", functions, match_level="relaxed")
pass
TEMPORARY_PATCHES.append(patch_Gemma4TextAttention_generic)


# ============================================================================
# Gemma4 causal mask patch (FORCE_FLOAT32 only) - uses mm_token_type_ids
# ============================================================================

def patch_Gemma4ForConditionalGeneration_causal_mask():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.gemma4.modeling_gemma4
        transformers.models.gemma4.modeling_gemma4.Gemma4Model
    except Exception as e:
        return raise_error("Gemma4Model._update_causal_mask", e)

    # Check if Gemma4Model has _update_causal_mask (older transformers style)
    if not hasattr(transformers.models.gemma4.modeling_gemma4.Gemma4Model, "_update_causal_mask"):
        return

    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids,
        past_key_values,
        cache_position,
        input_tensor,
        is_training: bool = False,
    ):
        if self.config.text_config._attn_implementation == "flash_attention_2":
            return attention_mask

        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask

        using_static_cache = isinstance(past_key_values, StaticCache)
        min_dtype = torch.finfo(torch.float16).min
        inputs_lead_dim, sequence_length = input_tensor.shape[:2]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        elif isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=torch.float16, device=cache_position.device
        )

        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)

        # Gemma4 uses mm_token_type_ids, but the causal mask interface uses token_type_ids
        if token_type_ids is not None and sequence_length != 1:
            token_type_mask = token_type_ids.unsqueeze(1) == token_type_ids.unsqueeze(2)
            token_type_mask[token_type_ids == 0] = False
            token_type_mask = token_type_mask.unsqueeze(1).to(causal_mask.device, dtype=torch.bool)
            causal_mask = causal_mask.clone()
            causal_mask[:, :, :, :sequence_length] = causal_mask[:, :, :, :sequence_length].masked_fill(
                token_type_mask, 0.0
            )

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

        return causal_mask
    pass
    if hasattr(transformers.models.gemma4.modeling_gemma4, "Gemma4Model"):
        patch_function(transformers.models.gemma4.modeling_gemma4.Gemma4Model, "_update_causal_mask", _update_causal_mask)
pass
TEMPORARY_PATCHES.append(patch_Gemma4ForConditionalGeneration_causal_mask)


# ============================================================================
# Gemma4 MLP patch (FORCE_FLOAT32 only)
# ============================================================================

def patch_Gemma4TextMLP():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.gemma4.modeling_gemma4
        transformers.models.gemma4.modeling_gemma4.Gemma4TextMLP
    except Exception as e:
        return raise_error("Gemma4TextMLP.forward", e)

    def forward(self, x):
        gate_proj_out = self.gate_proj(x)
        up_proj_out = self.up_proj(x)

        gate_proj_fp32 = gate_proj_out.to(torch.float32)
        up_proj_fp32 = up_proj_out.to(torch.float32)
        activated_fp32 = self.act_fn(gate_proj_fp32)
        intermediate_fp32 = activated_fp32 * up_proj_fp32

        intermediate_fp16 = intermediate_fp32.to(torch.float16)
        down_proj_out = self.down_proj(intermediate_fp16)
        return down_proj_out
    pass
    patch_function(transformers.models.gemma4.modeling_gemma4.Gemma4TextMLP, "forward", forward, fullgraph=False)
pass
TEMPORARY_PATCHES.append(patch_Gemma4TextMLP)


# ============================================================================
# Gemma4 DecoderLayer patch (FORCE_FLOAT32 only)
# The per_layer_input_gate and per_layer_projection are plain nn.Linear (fp16 weights).
# During gradient checkpointing backward recompute, hidden_states can arrive as fp32
# (due to autocast context differences). Fix: cast hidden_states to the gate weight dtype.
# ============================================================================

def patch_Gemma4TextDecoderLayer():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.gemma4.modeling_gemma4
        Gemma4TextDecoderLayer = transformers.models.gemma4.modeling_gemma4.Gemma4TextDecoderLayer
    except Exception as e:
        return raise_error("Gemma4TextDecoderLayer.forward", e)

    _original_decoder_forward = Gemma4TextDecoderLayer.forward

    def forward(self, hidden_states, per_layer_input=None, position_embeddings=None,
                attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        # Ensure hidden_states is fp16 before entering the decoder layer.
        # During gradient checkpointing backward recompute, it may arrive as fp32.
        if hidden_states.dtype == torch.float32 and hasattr(self, "hidden_size_per_layer_input") and self.hidden_size_per_layer_input:
            hidden_states = hidden_states.to(torch.float16)
        if per_layer_input is not None and per_layer_input.dtype == torch.float32:
            per_layer_input = per_layer_input.to(torch.float16)
        return _original_decoder_forward(
            self, hidden_states, per_layer_input, position_embeddings,
            attention_mask, position_ids, past_key_values, **kwargs)
    pass
    Gemma4TextDecoderLayer.forward = forward
pass
TEMPORARY_PATCHES.append(patch_Gemma4TextDecoderLayer)



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


# ============================================================================
# Gemma4 project_per_layer_inputs patch - fix fp32/fp16 dtype mismatch
# When FORCE_FLOAT32 is active, embed_tokens returns fp32 inputs_embeds
# but per_layer_model_projection.weight stays in fp16, causing F.linear to fail.
# Also needed for generic path since RMSNorm or embed may return different dtype.
# ============================================================================

def patch_Gemma4TextModel_project_per_layer_inputs():
    try:
        import transformers.models.gemma4.modeling_gemma4
        Gemma4TextModel = getattr(
            transformers.models.gemma4.modeling_gemma4, "Gemma4TextModel", None
        )
        if Gemma4TextModel is None:
            return
        # Only patch if the model actually uses per_layer_model_projection
        if not hasattr(Gemma4TextModel, "project_per_layer_inputs"):
            return
    except Exception as e:
        return raise_error("Gemma4TextModel.project_per_layer_inputs", e)

    _original_project = Gemma4TextModel.project_per_layer_inputs

    def project_per_layer_inputs(self, inputs_embeds, per_layer_inputs=None):
        # Fix dtype mismatch: FORCE_FLOAT32 makes embed_tokens return fp32,
        # but per_layer_model_projection weight stays in fp16/bf16.
        # Only cast for actual float dtype mismatches (not quantized uint8 weights).
        if hasattr(self, "per_layer_model_projection"):
            weight_dtype = self.per_layer_model_projection.weight.dtype
            # Only cast if both are floating point and they differ
            # (don't cast if weight is quantized uint8/int8 from bitsandbytes)
            if (weight_dtype.is_floating_point and inputs_embeds.dtype.is_floating_point
                    and inputs_embeds.dtype != weight_dtype):
                original_dtype = inputs_embeds.dtype
                inputs_embeds_cast = inputs_embeds.to(weight_dtype)
                result = _original_project(self, inputs_embeds_cast, per_layer_inputs)
                # Return in the original dtype to maintain FORCE_FLOAT32 consistency
                return result.to(original_dtype)
        return _original_project(self, inputs_embeds, per_layer_inputs)
    pass
    Gemma4TextModel.project_per_layer_inputs = project_per_layer_inputs
pass
TEMPORARY_PATCHES.append(patch_Gemma4TextModel_project_per_layer_inputs)
