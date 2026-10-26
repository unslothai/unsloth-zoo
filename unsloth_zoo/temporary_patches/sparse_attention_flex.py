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

"""Long-context sparse attention on flex attention for MiniMax M3 and DSA models.

Above the indexer's selection budget, upstream eager / SDPA expand the selection into a
dense mask and run dense masked attention. These patches keep the model's own path below
the budget and run flex attention over a BlockMask built from the selection above it
(unsloth_zoo.flex_attention.sparse_blocks). The indexers also score queries in chunks so
their fp32 score buffers stay bounded.

Engaged automatically on CUDA sm80+ / ROCm with Triton, bf16 / fp16, no attention dropout,
no output_attentions, q_len >= 16. DSA keeps dense masked SDPA (still with the chunked
indexer) where torch picks cuDNN for that masked call, which beats flex there (H100, torch 2.14). A flex failure latches that shape class back to the
dense path. Kill switches: UNSLOTH_SPARSE_ATTN_FLEX=0, UNSLOTH_SPARSE_INDEXER_CHUNKED=0.
The patched forwards are installed on transformers >= 5.17 when the installed source matches
the layout they reimplement (DSA caches expanded K / V on 5.17, the latent on 5.18+);
otherwise the model is left untouched.
"""

__all__ = [
    "patch_minimax_m3_sparse_flex",
    "patch_dsa_sparse_flex",
]

import inspect
import os

import torch

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING, publish_to_modeling_module
from .utils import logger
from . import sparse_attention as _skip
from .sparse_attention import _transformers_tested
from ..flex_attention import sparse_blocks as _sb
from ..flex_attention.sparse import sparse_indexer_chunked_enabled

# family -> modeling module, bound at patch time
_MODELING = {}


def _modeling(name):
    return _skip._modeling_module(name)


def _indexer_base(cls, model_type, skip_forward, skip_installer):
    """(upstream forward, forward to delegate within-budget calls to), or None.

    The exact skip (sparse_attention.py) is installed first, so within the budget its
    bitwise-identical path runs whichever patch set loads first; above the budget this module
    takes over. Its installer only patches an upstream forward, so it no-ops on ours later."""
    skip_installer()
    current = cls.forward
    if current is skip_forward:
        upstream = _skip._REFERENCE.get(f"{model_type}.indexer")
        return (upstream, current) if upstream is not None else None
    if _skip._is_transformers_function(current, model_type):
        return current, current
    return None


def _source_has(function, needles):
    try:
        source = inspect.getsource(function)
    except Exception:
        return False
    return all(n in source for n in needles)


def _dsa_min_keys():
    """UNSLOTH_SPARSE_ATTN_DSA_MIN_KEYS override, else None (budget + cuDNN probe decide)."""
    env = os.environ.get("UNSLOTH_SPARSE_ATTN_DSA_MIN_KEYS")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    return None


# --------------------------------------------------------------------------------------
# MiniMax M3
# --------------------------------------------------------------------------------------
_MINIMAX_ATTN_NEEDLES = (
    "block_indices = self.indexer(hidden_states, position_embeddings, past_key_values, position_ids)",
    "attention_mask = self.indexer.build_block_mask(",
    "past_key_values.update(key_states, value_states, self.layer_idx)",
)
_MINIMAX_INDEXER_NEEDLES = (
    "idx_k = past_key_values.layers[self.layer_idx].update_index(idx_k)",
    "block_scores = scores.amax(dim=-1)",
    "return topk_indices.masked_fill(topk_scores == float(\"-inf\"), -1)",
)


def unsloth_minimax_m3_indexer_forward(self, hidden_states, position_embeddings, past_key_values, position_ids):
    if sparse_indexer_chunked_enabled():
        k_len = _skip._minimax_index_key_len(self, hidden_states.shape[1], past_key_values)
        if k_len is not None and -(-k_len // self.block_size) > self.topk_blocks:
            m = _MODELING["minimax_m3_vl"]
            return _sb.minimax_m3_select_blocks(
                self, hidden_states, position_embeddings, past_key_values, position_ids, m.apply_rotary_pos_emb,
            )
    # Within the budget: the exact skip (or upstream when the skip is off / not installed).
    return self._unsloth_inner_forward(hidden_states, position_embeddings, past_key_values, position_ids)


def unsloth_minimax_m3_sparse_attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values = None, **kwargs):
    m = _MODELING["minimax_m3_vl"]
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = m.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attention_interface = m.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, m.eager_attention_forward
    )
    block_indices = None
    if self.indexer is not None:
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            position_ids = torch.arange(
                key_states.shape[2] - query_states.shape[2], key_states.shape[2], device = query_states.device
            )
        position_ids = (position_ids if position_ids.ndim > 1 else position_ids.unsqueeze(0)).expand(
            query_states.shape[0], -1
        )
        block_indices = self.indexer(hidden_states, position_embeddings, past_key_values, position_ids)
        if _sb.minimax_m3_flex_eligible(self, query_states, key_states, attention_mask, kwargs):
            try:
                attn_output = _sb.minimax_m3_flex_attention(
                    query_states, key_states, value_states, block_indices, attention_mask,
                    position_ids, self.scaling, self.indexer.block_size,
                )
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                return self.o_proj(attn_output), None
            except Exception as error:
                _sb.record_failure("minimax_m3", query_states, key_states, key_states, error)
                logger.warning(f"Unsloth: MiniMax M3 sparse flex attention failed, using the dense path: {error}")
        if self.config._attn_implementation in ("eager", "sdpa"):
            attention_mask = self.indexer.build_block_mask(
                block_indices,
                attention_mask,
                key_states.shape[2],
                query_states.dtype,
                query_states.device,
                position_ids,
            )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout = 0.0 if not self.training else self.attention_dropout,
        scaling = self.scaling,
        block_indices = block_indices,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def patch_minimax_m3_sparse_flex():
    if not _transformers_tested():
        return
    m = _modeling("minimax_m3_vl")
    if m is None:
        return
    attn = getattr(m, "MiniMaxM3VLAttention", None)
    indexer = getattr(m, "MiniMaxM3VLIndexer", None)
    if attn is None or indexer is None or getattr(attn.forward, "_unsloth_sparse_flex", False):
        return
    base = _indexer_base(indexer, "minimax_m3_vl", _skip.minimax_m3_indexer_forward, _skip.patch_minimax_m3_sparse_skip)
    if base is None or not _source_has(attn.forward, _MINIMAX_ATTN_NEEDLES) or not _source_has(base[0], _MINIMAX_INDEXER_NEEDLES):
        if UNSLOTH_ENABLE_LOGGING:
            logger.info("Unsloth: MiniMax M3 attention source differs from the supported layout; sparse flex not installed.")
        return
    _MODELING["minimax_m3_vl"] = m
    publish_to_modeling_module(
        m,
        unsloth_minimax_m3_sparse_attention_forward = unsloth_minimax_m3_sparse_attention_forward,
        unsloth_minimax_m3_indexer_forward = unsloth_minimax_m3_indexer_forward,
    )

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values = None, **kwargs):
        return unsloth_minimax_m3_sparse_attention_forward(
            self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs
        )

    def indexer_forward(self, hidden_states, position_embeddings, past_key_values, position_ids):
        return unsloth_minimax_m3_indexer_forward(self, hidden_states, position_embeddings, past_key_values, position_ids)

    indexer._unsloth_original_forward, indexer._unsloth_inner_forward = base
    forward._unsloth_sparse_flex = True
    indexer_forward._unsloth_sparse_flex = True
    attn.forward = forward
    indexer.forward = indexer_forward
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: MiniMax M3 sparse attention runs on flex attention above the selection budget.")


TEMPORARY_PATCHES.append(patch_minimax_m3_sparse_flex)


# --------------------------------------------------------------------------------------
# DSA: DeepSeek V3.2 and GLM-MoE-DSA
# --------------------------------------------------------------------------------------
_DSA_ATTN_NEEDLES = (
    "key_states, value_states = self.expand_kv(k_pass, k_rot)",
    "attention_mask[:, 0, :, :],",
    ".scatter(-1, topk_indices.long(), False)",
)
# Where the KV cache is written: expanded K / V (5.17) or the latent before expand_kv (5.18+).
_DSA_CACHE_EXPANDED = "key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)"
_DSA_CACHE_LATENT = "k_pass, k_rot = past_key_values.update(k_pass, k_rot, self.layer_idx)"
_DSA_LATENT_CACHE = {}
_DSA_INDEXER_NEEDLES = (
    "k = past_key_values.update_indexer(k, self.layer_idx)",
    "index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)",
    "return index_scores.topk(topk, dim=-1).indices.to(torch.int32)",
)
# Indexer RoPE: DeepSeek V3.2 half-split, GLM-MoE-DSA interleaved.
_DSA_INDEXER_ROPE = {"deepseek_v32": "apply_rotary_pos_emb", "glm_moe_dsa": "apply_rotary_pos_emb_interleave"}


def unsloth_dsa_indexer_forward(self, hidden_states, q_resid, position_embeddings, attention_mask, position_ids = None, past_key_values = None):
    family = self._unsloth_dsa_family
    if sparse_indexer_chunked_enabled() and attention_mask.shape[-1] > self.index_topk:
        rope = getattr(_MODELING[family], _DSA_INDEXER_ROPE[family])
        return _sb.dsa_select_topk(
            self, hidden_states, q_resid, position_embeddings, attention_mask, past_key_values, rope, family,
        )
    # Within the budget: the exact skip (or upstream when the skip is off / not installed).
    return self._unsloth_inner_forward(
        hidden_states, q_resid, position_embeddings, attention_mask, position_ids, past_key_values = past_key_values,
    )


def _dsa_attention(self, family, hidden_states, position_embeddings, attention_mask, past_key_values, position_ids, prev_topk_indices, kwargs):
    m = _MODELING[family]
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim = -1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim = -1)
    k_pass = self.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, self.kv_lora_rank)
    k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
    cos, sin = position_embeddings
    q_rot, k_rot = m.apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)

    latent_cache = _DSA_LATENT_CACHE[family]
    if past_key_values is not None and latent_cache:
        k_pass, k_rot = past_key_values.update(k_pass, k_rot, self.layer_idx)

    query_states = torch.cat((q_pass, q_rot), dim = -1)
    key_states, value_states = self.expand_kv(k_pass, k_rot)

    if past_key_values is not None and not latent_cache:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    if getattr(self, "indexer", None) is not None:
        topk_indices = self.indexer(
            hidden_states,
            q_resid,
            position_embeddings,
            attention_mask[:, 0, :, :],
            position_ids,
            past_key_values = past_key_values,
        )
    else:
        if prev_topk_indices is None:
            raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
        topk_indices = prev_topk_indices

    if _sb.dsa_flex_eligible(self, query_states, key_states, value_states, attention_mask, kwargs, _dsa_min_keys()):
        try:
            attn_output = _sb.dsa_flex_attention(
                query_states, key_states, value_states, topk_indices, attention_mask, self.scaling, family,
            )
            attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
            return self.o_proj(attn_output), None, topk_indices
        except Exception as error:
            _sb.record_failure("dsa", query_states, key_states, value_states, error)
            logger.warning(f"Unsloth: {family} sparse flex attention failed, using the dense path: {error}")

    sparse_indices = None
    if self.config._attn_implementation in ("eager", "sdpa"):
        index_mask = (
            topk_indices.new_ones((batch_size, seq_length, key_states.shape[2]), dtype = torch.bool)
            .scatter(-1, topk_indices.long(), False)
            .unsqueeze(1)
        )
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask & ~index_mask
        else:
            attention_mask = attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min)
    else:
        sparse_indices = topk_indices

    attention_interface = m.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, m.eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout = 0.0 if not self.training else self.attention_dropout,
        scaling = self.scaling,
        indices = sparse_indices,
        **kwargs,
    )
    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, topk_indices


def unsloth_deepseek_v32_sparse_attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values = None, position_ids = None, **kwargs):
    out, weights, _ = _dsa_attention(
        self, "deepseek_v32", hidden_states, position_embeddings, attention_mask, past_key_values, position_ids, None, kwargs,
    )
    return out, weights


def unsloth_glm_moe_dsa_sparse_attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values = None, position_ids = None, prev_topk_indices = None, **kwargs):
    return _dsa_attention(
        self, "glm_moe_dsa", hidden_states, position_embeddings, attention_mask, past_key_values, position_ids, prev_topk_indices, kwargs,
    )


def _patch_dsa_family(family, attn_name, indexer_name, attention_forward):
    if not _transformers_tested():
        return False
    m = _modeling(family)
    if m is None:
        return False
    attn = getattr(m, attn_name, None)
    indexer = getattr(m, indexer_name, None)
    if attn is None or indexer is None or getattr(attn.forward, "_unsloth_sparse_flex", False):
        return False
    skip_forward, skip_installer = {
        "deepseek_v32": (_skip.deepseek_v32_indexer_forward, _skip.patch_deepseek_v32_sparse_skip),
        "glm_moe_dsa": (_skip.glm_moe_dsa_indexer_forward, _skip.patch_glm_moe_dsa_sparse_skip),
    }[family]
    base = _indexer_base(indexer, family, skip_forward, skip_installer)
    rope = getattr(m, _DSA_INDEXER_ROPE[family], None)
    if (
        base is None
        or rope is None
        or not hasattr(m, "apply_rotary_pos_emb_interleave")
        or not _source_has(attn.forward, _DSA_ATTN_NEEDLES)
        or not _source_has(base[0], _DSA_INDEXER_NEEDLES)
        or _DSA_INDEXER_ROPE[family] + "(q_rot, k_rot, cos, sin, unsqueeze_dim=2)" not in inspect.getsource(base[0])
    ):
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(f"Unsloth: {family} attention source differs from the supported layout; sparse flex not installed.")
        return False
    source = inspect.getsource(attn.forward)
    expanded, latent = _DSA_CACHE_EXPANDED in source, _DSA_CACHE_LATENT in source
    if expanded == latent:
        return False
    _DSA_LATENT_CACHE[family] = latent
    _MODELING[family] = m
    publish_to_modeling_module(
        m,
        **{f"unsloth_{family}_sparse_attention_forward": attention_forward},
        unsloth_dsa_indexer_forward = unsloth_dsa_indexer_forward,
    )
    if family == "deepseek_v32":
        def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values = None, position_ids = None, **kwargs):
            return unsloth_deepseek_v32_sparse_attention_forward(
                self, hidden_states, position_embeddings, attention_mask, past_key_values, position_ids, **kwargs
            )
    else:
        def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values = None, position_ids = None, prev_topk_indices = None, **kwargs):
            return unsloth_glm_moe_dsa_sparse_attention_forward(
                self, hidden_states, position_embeddings, attention_mask, past_key_values, position_ids, prev_topk_indices, **kwargs
            )

    def indexer_forward(self, hidden_states, q_resid, position_embeddings, attention_mask, position_ids = None, past_key_values = None):
        return unsloth_dsa_indexer_forward(self, hidden_states, q_resid, position_embeddings, attention_mask, position_ids, past_key_values)

    indexer._unsloth_original_forward, indexer._unsloth_inner_forward = base
    attn._unsloth_upstream_forward = attn.forward
    indexer._unsloth_dsa_family = family
    forward._unsloth_sparse_flex = True
    indexer_forward._unsloth_sparse_flex = True
    attn.forward = forward
    indexer.forward = indexer_forward
    return True


def patch_dsa_sparse_flex():
    for family, attn_name, indexer_name, fn in (
        ("deepseek_v32", "DeepseekV32Attention", "DeepseekV32Indexer", unsloth_deepseek_v32_sparse_attention_forward),
        ("glm_moe_dsa", "GlmMoeDsaAttention", "GlmMoeDsaIndexer", unsloth_glm_moe_dsa_sparse_attention_forward),
    ):
        try:
            if _patch_dsa_family(family, attn_name, indexer_name, fn) and UNSLOTH_ENABLE_LOGGING:
                logger.info(f"Unsloth: {family} sparse attention runs on flex attention above the selection budget.")
        except Exception as error:
            logger.warning(f"Unsloth: could not install {family} sparse flex attention: {error}")


TEMPORARY_PATCHES.append(patch_dsa_sparse_flex)
