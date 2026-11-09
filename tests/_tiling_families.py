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


import importlib

import torch

_COMMON = dict(
    vocab_size = 97,
    hidden_size = 32,
    intermediate_size = 64,
    num_hidden_layers = 2,
    num_attention_heads = 4,
    num_key_value_heads = 2,
    max_position_embeddings = 256,
)

_MOE = dict(num_experts = 4, num_experts_per_tok = 2, moe_intermediate_size = 48)

FAMILIES = {
    "llama": ("LlamaConfig", "LlamaForCausalLM", dict(_COMMON)),
    "qwen3": ("Qwen3Config", "Qwen3ForCausalLM", dict(_COMMON, head_dim = 8)),
    "gemma3": ("Gemma3TextConfig", "Gemma3ForCausalLM", dict(_COMMON, head_dim = 8, sliding_window = 16)),
    "phi3": ("Phi3Config", "Phi3ForCausalLM", dict(_COMMON, pad_token_id = 0)),
    "qwen2_moe": ("Qwen2MoeConfig", "Qwen2MoeForCausalLM", dict(
        _COMMON, **_MOE, shared_expert_intermediate_size = 40, decoder_sparse_step = 1,
    )),
    "qwen3_moe": ("Qwen3MoeConfig", "Qwen3MoeForCausalLM", dict(_COMMON, **_MOE, head_dim = 8, decoder_sparse_step = 1)),
    "qwen3_5_moe": ("Qwen3_5MoeTextConfig", "Qwen3_5MoeForCausalLM", dict(
        _COMMON, **_MOE, head_dim = 8, shared_expert_intermediate_size = 40,
        linear_num_value_heads = 4, linear_num_key_heads = 2, linear_key_head_dim = 8, linear_value_head_dim = 8,
        layer_types = ["linear_attention", "full_attention"],
    )),
    "mixtral": ("MixtralConfig", "MixtralForCausalLM", dict(_COMMON, num_local_experts = 4, num_experts_per_tok = 2)),
    "olmoe": ("OlmoeConfig", "OlmoeForCausalLM", dict(_COMMON, num_experts = 4, num_experts_per_tok = 2)),
    "gpt_oss": ("GptOssConfig", "GptOssForCausalLM", dict(
        _COMMON, num_local_experts = 4, num_experts_per_tok = 2, head_dim = 8, sliding_window = 16,
        layer_types = ["sliding_attention", "full_attention"],
    )),
    "llama4": ("Llama4TextConfig", "Llama4ForCausalLM", dict(
        _COMMON, num_local_experts = 4, num_experts_per_tok = 1, head_dim = 8, intermediate_size_mlp = 64,
        interleave_moe_layer_step = 1, moe_layers = [0, 1], no_rope_layers = [1, 1], attention_chunk_size = 64,
    )),
    "qwen3_next": ("Qwen3NextConfig", "Qwen3NextForCausalLM", dict(
        _COMMON, **_MOE, head_dim = 8, shared_expert_intermediate_size = 40, decoder_sparse_step = 1,
        linear_num_value_heads = 4, linear_num_key_heads = 2, linear_key_head_dim = 8, linear_value_head_dim = 8,
        layer_types = ["linear_attention", "full_attention"],
    )),
    "nemotron_h": ("NemotronHConfig", "NemotronHForCausalLM", dict(
        _COMMON, num_hidden_layers = 4, hybrid_override_pattern = "M*-E", layers_block_type = None,
        mamba_num_heads = 4, mamba_head_dim = 8, ssm_state_size = 8, n_groups = 1, mamba_d_conv = 4,
        mamba_expand = 1, chunk_size = 16, n_routed_experts = 4, num_experts_per_tok = 2,
        moe_intermediate_size = 48, moe_shared_expert_intermediate_size = 40, n_group = 1, topk_group = 1,
        head_dim = 8,
    )),
    "deepseek_v3": ("DeepseekV3Config", "DeepseekV3ForCausalLM", dict(
        _COMMON, num_key_value_heads = 4, n_routed_experts = 4, num_experts_per_tok = 2, moe_intermediate_size = 48,
        n_shared_experts = 1,
        first_k_dense_replace = 1, n_group = 1, topk_group = 1, kv_lora_rank = 16, q_lora_rank = 16,
        qk_rope_head_dim = 8, qk_nope_head_dim = 8, v_head_dim = 8,
    )),
    "switch": ("SwitchTransformersConfig", "SwitchTransformersForConditionalGeneration", dict(
        vocab_size = 97, d_model = 32, d_kv = 8, d_ff = 64, num_layers = 2, num_decoder_layers = 2, num_heads = 4,
        num_experts = 4, expert_capacity = 4, num_sparse_encoder_layers = 1, num_sparse_decoder_layers = 1,
        decoder_start_token_id = 0, pad_token_id = 0,
    )),
}


def build(family, dtype = torch.float32, seed = 0, **overrides):
    import transformers
    cfg_name, model_name, kwargs = FAMILIES[family]
    cfg_cls = getattr(transformers, cfg_name, None)
    model_cls = getattr(transformers, model_name, None)
    if cfg_cls is None or model_cls is None:
        return None
    kwargs = dict(kwargs, **overrides)
    if family == "nemotron_h":
        # 5.x reads layers_block_type, older spellings read hybrid_override_pattern.
        kwargs.pop("layers_block_type", None)
    config = cfg_cls(**kwargs)
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = model_cls(config).to(dtype)
    # Random init leaves router weights tiny, so every expert gets similar scores and top-k ties flip on rounding.
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.2)
    model.eval()
    return model


def available():
    return [f for f in FAMILIES if build_ok(f)]


def build_ok(family):
    try:
        return build(family) is not None
    except Exception:
        return False
