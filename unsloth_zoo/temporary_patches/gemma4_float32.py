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
#
# ============================================================================
# Gemma-4 targeted float32 patches for fp16 (UNSLOTH_FORCE_FLOAT32) training.
#
# Direct analog of the Gemma3 float32 patches in gemma.py. When gemma4 is in
# unsloth_zoo.model_lists.FORCE_FLOAT32, a fp16 request loads bf16 weights,
# down-casts them to fp16 (do_forced_float32), and relies on these per-module
# patches to keep the residual stream in float32 while running the matmuls in
# fp16. Without them fp16 NaNs the grad_norm in the backward (forward peaks at
# ~350, far below fp16's 65504, so it is a backward gradient overflow), and a
# generic fp32 RoPE upcast of q/k while v stays fp16 makes SDPA raise a dtype
# mismatch. All three patches gate on UNSLOTH_FORCE_FLOAT32 == "1" so bf16 /
# fp32 runs (and fp16 runs on other archs) are untouched.
#
# What each patch does (only the precision-sensitive ops upcast; the rest of
# the model stays fp16, so this is targeted, not a whole-model float32 wrap):
#   * ScaledWordEmbedding -> returns float32, keeping the residual highway in
#     float32 so gradients accumulate without fp16 overflow.
#   * RMSNorm             -> computes in float32, returns fp16 (clamped), so
#     each sub-layer entry down-casts the fp32 residual back to fp16 for the
#     fp16 projections; residual adds (fp32 + fp16) stay fp32.
#   * TextAttention       -> q/k/v + scores all float32 (consistent dtype, no
#     SDPA mismatch), attention output down-cast to fp16 for o_proj.
# The MLP / MoE-expert path is handled by the existing fp16 overflow clamp in
# gemma4.py (patch_Gemma4TextMLP) plus the grouped-GEMM expert forward, both of
# which already run gate*up in float32 for fp16 inputs, so no extra MLP patch is
# needed here.
# ============================================================================

import os
import inspect
import torch
from .common import TEMPORARY_PATCHES
from .utils import patch_function, raise_error


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
            weight = self.weight,
            padding_idx = self.padding_idx,
        )
        # float32 residual stream (embed_scale ~ sqrt(hidden_size))
        return input_embeds.to(torch.float32) * self.embed_scale.to(torch.float32)
    pass
    patch_function(transformers.models.gemma4.modeling_gemma4.Gemma4TextScaledWordEmbedding, "forward", forward, fullgraph = True)
pass
TEMPORARY_PATCHES.append(patch_Gemma4TextScaledWordEmbedding)


def patch_Gemma4RMSNorm():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.gemma4.modeling_gemma4
        transformers.models.gemma4.modeling_gemma4.Gemma4RMSNorm
    except Exception as e:
        return raise_error("Gemma4RMSNorm.forward", e)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: # fp32 (residual) or fp16 (sub-layer)
        # Gemma4 scales by `weight` directly (no 1.0 + weight) and only when with_scale.
        x_fp32 = hidden_states.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        normed_fp32 = x_fp32 * torch.pow(variance + self.eps, -0.5)
        if self.with_scale:
            normed_fp32 = normed_fp32 * self.weight.to(torch.float32)

        # Clamp to fp16 range before casting back so a large residual never becomes inf.
        fp16_max = torch.finfo(torch.float16).max
        return torch.clamp(normed_fp32, min = -fp16_max, max = fp16_max).to(torch.float16)
    pass
    patch_function(transformers.models.gemma4.modeling_gemma4.Gemma4RMSNorm, "forward", forward, fullgraph = True, match_level = "relaxed")
pass
TEMPORARY_PATCHES.append(patch_Gemma4RMSNorm)


def patch_Gemma4TextAttention():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.gemma4.modeling_gemma4
        transformers.models.gemma4.modeling_gemma4.Gemma4TextAttention
        from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb
    except Exception as e:
        return raise_error("Gemma4TextAttention.forward", e)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
    scaled_dot_product_attention = torch.compiler.disable(scaled_dot_product_attention, recursive = True)

    def forward(
        self,
        hidden_states,
        position_embeddings = None,
        attention_mask = None,
        past_key_values = None,
        **kwargs,
    ):
        # Shared-KV carrier (Gemma-4 E-series, num_kv_shared_layers > 0); a no-op
        # for 31B / 26B-A4B where num_kv_shared_layers == 0 and no carrier is attached.
        carrier = getattr(self, "_unsloth_shared_kv_carrier", None)
        if carrier is not None and past_key_values is None:
            past_key_values = carrier

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)

        # q: fp16 proj -> fp16 q_norm -> float32 -> RoPE (float32) -> (b, h, q, d)
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states).to(torch.float32)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim = 2)
        query_states = query_states.transpose(1, 2).contiguous()

        if self.is_kv_shared_layer and past_key_values is not None:
            key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
            key_states = key_states.to(query_states.device, torch.float32)
            value_states = value_states.to(query_states.device, torch.float32)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            # attention_k_eq_v global layers have v_proj is None -> value shares the
            # raw k_proj output (before k_norm / RoPE rebind key_states below).
            value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

            key_states = self.k_norm(key_states).to(torch.float32)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim = 2)
            key_states = key_states.transpose(1, 2).contiguous()

            value_states = self.v_norm(value_states).to(torch.float32)
            value_states = value_states.transpose(1, 2).contiguous()

        if past_key_values is not None:
            if not self.is_kv_shared_layer:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            if self.store_full_length_kv:
                if not hasattr(past_key_values, "shared_layers"):
                    past_key_values.shared_layers = {}
                past_key_values.shared_layers[self.layer_idx] = key_states, value_states

        # Core attention in float32. Gemma4 uses scaling = 1.0 (QK-norm bakes the scale in).
        attn_mask = attention_mask
        if isinstance(attn_mask, torch.Tensor) and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(torch.float32)
        is_causal = attn_mask is None and query_states.shape[2] > 1 and getattr(self, "is_causal", True)
        attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask = attn_mask,
            dropout_p = self.attention_dropout if self.training else 0.0,
            is_causal = is_causal,
            scale = self.scaling,
            enable_gqa = getattr(self, "num_key_value_groups", 1) != 1,
        )

        # (b, h, q, d) -> (b, q, h*d), down-cast to fp16 for o_proj.
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_output = attn_output.to(torch.float16)
        attn_output = self.o_proj(attn_output)
        return attn_output, None
    pass
    # force = True: this patch runs after gemma4.py's shared-KV carrier has already
    # wrapped forward as (self, *args, **kwargs); a signature check would reject the
    # replacement, so the compiler would capture the carrier/original attention (q/k
    # float32 vs v fp16 -> SDPA dtype mismatch) instead of this float32 forward. The
    # carrier is handled inline above, so replacing it outright is safe.
    patch_function(transformers.models.gemma4.modeling_gemma4.Gemma4TextAttention, "forward", forward, force = True, match_level = "relaxed")
pass
TEMPORARY_PATCHES.append(patch_Gemma4TextAttention)
