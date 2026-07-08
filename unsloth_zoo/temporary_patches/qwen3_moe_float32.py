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
# Qwen3-MoE targeted float32 patches for fp16 (UNSLOTH_FORCE_FLOAT32) training.
#
# Direct analog of the Gemma4 float32 patches in gemma4_float32.py. When
# qwen3_moe is in unsloth_zoo.model_lists.FORCE_FLOAT32, a fp16 request loads
# bf16 weights, down-casts them to fp16 and runs under autocast(fp16); these
# patches keep the residual stream in float32 while the matmuls / experts stay
# fp16. Without them fp16 NaNs the grad_norm in backward (forward peaks ~350,
# far below fp16's 65504, so it is a backward gradient overflow), and
# do_forced_float32 upcasts the RMSNorm weights to float32, making q/k float32
# while v stays fp16 -> SDPA raises "float != Half". All patches gate on
# UNSLOTH_FORCE_FLOAT32 == "1", so bf16 / fp32 (and fp16 on other archs) are untouched.
#
# Only the precision-sensitive ops upcast (targeted, not a whole-model float32):
#   * DecoderLayer -> residual stream upcast to float32 at each sub-layer entry.
#     Qwen3's embedding is PLAIN (unscaled), so we hold the residual highway fp32
#     here; RMSNorm down-casts it back to fp16 and residual adds (fp32 + fp16)
#     stay fp32. Not torch.compiled (the compiler only rewrites sub-modules).
#   * RMSNorm      -> computes in float32, returns fp16 (clamped). Qwen3 scales by
#     `weight` directly (no 1.0 + weight) with variance_epsilon.
#   * Attention    -> q/k/v + scores all float32 (no SDPA mismatch), output
#     down-cast to fp16 for o_proj.
# MoE experts / router are untouched: qwen3_moe.py's block-forward already does the
# router softmax in float32 and the grouped expert GEMM stays fp16.
# ============================================================================

import os
import torch
from .common import TEMPORARY_PATCHES
from .utils import patch_function, raise_error

# Mirrors gemma.py: flex dispatch can be turned off globally.
_UNSLOTH_FLEX_ATTENTION_DISABLED = os.environ.get("UNSLOTH_ENABLE_FLEX_ATTENTION", "1") == "0"


def patch_Qwen3MoeDecoderLayer_float32():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.qwen3_moe.modeling_qwen3_moe
        transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeDecoderLayer
    except Exception as e:
        return raise_error("Qwen3MoeDecoderLayer.forward", e)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        use_cache = False,
        position_embeddings = None,
        **kwargs,
    ):
        # float32 residual highway (qwen3 embedding is unscaled -> upcast here).
        # No-op when hidden_states is already float32 (layers >= 1).
        residual = hidden_states.to(torch.float32)
        hidden_states = self.input_layernorm(hidden_states)  # -> fp16
        hidden_states, _ = self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            use_cache = use_cache,
            position_embeddings = position_embeddings,
            **kwargs,
        )  # -> fp16
        hidden_states = residual + hidden_states  # fp32 + fp16 -> fp32

        residual = hidden_states  # fp32
        hidden_states = self.post_attention_layernorm(hidden_states)  # -> fp16
        hidden_states = self.mlp(hidden_states)  # MoE / MLP -> fp16
        # Old-transformers sparse MoE blocks return (hidden_states, router_logits);
        # mirror the upstream decoder's unpack.
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = residual + hidden_states  # fp32 + fp16 -> fp32
        return hidden_states
    pass
    # force = True: guarantee the residual-float32 forward wins even if a wrapper
    # is pre-installed; the layer is not compiled so this stays eager Python.
    patch_function(transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeDecoderLayer, "forward", forward, force = True, match_level = "relaxed")
pass
TEMPORARY_PATCHES.append(patch_Qwen3MoeDecoderLayer_float32)


def patch_Qwen3MoeRMSNorm_float32():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.qwen3_moe.modeling_qwen3_moe
        transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeRMSNorm
    except Exception as e:
        return raise_error("Qwen3MoeRMSNorm.forward", e)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: # fp32 (residual) or fp16 (sub-layer)
        # Qwen3 scales by `weight` directly (no 1.0 + weight) with variance_epsilon.
        x_fp32 = hidden_states.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim = True)
        normed_fp32 = x_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        normed_fp32 = normed_fp32 * self.weight.to(torch.float32)

        # Clamp to fp16 range before casting back so a large residual never becomes inf.
        fp16_max = torch.finfo(torch.float16).max
        return torch.clamp(normed_fp32, min = -fp16_max, max = fp16_max).to(torch.float16)
    pass
    patch_function(transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeRMSNorm, "forward", forward, fullgraph = True, match_level = "relaxed")
pass
TEMPORARY_PATCHES.append(patch_Qwen3MoeRMSNorm_float32)


def patch_Qwen3MoeAttention_float32():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.qwen3_moe.modeling_qwen3_moe
        transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention
        from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS
    except Exception as e:
        return raise_error("Qwen3MoeAttention.forward", e)
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
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings
        cos_fp32 = cos.to(torch.float32)
        sin_fp32 = sin.to(torch.float32)

        # q/k: fp16 proj -> fp16 q_norm/k_norm (returns fp16) -> float32 -> (b, h, s, d) -> RoPE (float32)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).to(torch.float32).transpose(1, 2)
        key_states   = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).to(torch.float32).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).to(torch.float32).transpose(1, 2)

        # Qwen3 applies RoPE to both q and k together, unsqueeze_dim = 1 (post-transpose layout).
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos_fp32, sin_fp32)

        if past_key_values is not None:
            # 4.57.x static caches need cache_position to write the right slot;
            # 5.x layer caches absorb the extra dict harmlessly.
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": kwargs.get("cache_position", None)}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Core attention in float32 (consistent q/k/v dtype -> no SDPA mismatch).
        attn_impl = getattr(self.config, "_attn_implementation", "sdpa") or "sdpa"
        if attn_impl == "flex_attention" and _UNSLOTH_FLEX_ATTENTION_DISABLED:
            attn_impl = "sdpa"
        attention_interface = None
        if attn_impl not in ("sdpa", "eager"):
            try:
                attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]
            except KeyError:
                attention_interface = None  # unknown backend -> raw SDPA below
        attn_weights = None

        if attn_impl == "eager":
            # Mirror upstream eager (additive mask, fp32 softmax) and return weights.
            g = getattr(self, "num_key_value_groups", 1) or 1
            k_r = key_states.repeat_interleave(g, dim = 1) if g != 1 else key_states
            v_r = value_states.repeat_interleave(g, dim = 1) if g != 1 else value_states
            attn_weights = torch.matmul(query_states, k_r.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                m = attention_mask[..., : k_r.shape[-2]]
                if m.dtype == torch.bool:
                    m = torch.zeros_like(attn_weights).masked_fill_(~m, torch.finfo(torch.float32).min)
                attn_weights = attn_weights + m.to(torch.float32)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim = -1)
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p = self.attention_dropout if self.training else 0.0, training = self.training,
            )
            attn_output = torch.matmul(attn_weights, v_r).transpose(1, 2)
        elif attention_interface is not None:
            # Registered backend (flex / flash) owns the mask semantics; pass the
            # mask exactly as upstream prepared it (FA uses a 2-D padding mask,
            # flex a BlockMask). Flash kernels need fp16 inputs and accumulate in
            # float32 internally, so stability is preserved; flex handles fp32.
            q, k, v = query_states, key_states, value_states
            if attn_impl != "flex_attention":
                q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
            attn_output, attn_weights = attention_interface(
                self,
                q,
                k,
                v,
                attention_mask,
                dropout = self.attention_dropout if self.training else 0.0,
                scaling = self.scaling,
                sliding_window = getattr(self, "sliding_window", None),
                **kwargs,
            )  # returns (b, q, h, d), already transposed
        else:
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
            attn_output = attn_output.transpose(1, 2)  # (b, h, q, d) -> (b, q, h, d)

        # (b, q, h, d) -> (b, q, h*d), down-cast to fp16 for o_proj.
        attn_output = attn_output.contiguous().reshape(*input_shape, -1)
        attn_output = attn_output.to(torch.float16)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    pass
    # force = True: win over the compiled / dtype-aligned attention forward that
    # unsloth installs (q/k float32 vs v fp16 -> SDPA mismatch otherwise).
    patch_function(transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention, "forward", forward, force = True, match_level = "relaxed")
pass
TEMPORARY_PATCHES.append(patch_Qwen3MoeAttention_float32)
