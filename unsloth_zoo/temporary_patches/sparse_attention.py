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

"""Skip sparse-attention indexers when top-k keeps every key anyway (exact).

Within the budget (DSA `index_topk`, MiniMax-M3 `index_topk_blocks`, DeepSeek-V4 CSA `index_topk`)
top-k selects every visible key, so the indexer's fp32 multi-head scoring is wasted. The patched
indexers return the same selection without scoring and still update their own key cache.
UNSLOTH_SPARSE_ATTN_SKIP_FULL=0 restores upstream.
"""

__all__ = [
    "patch_deepseek_v32_sparse_skip",
    "patch_glm_moe_dsa_sparse_skip",
    "patch_minimax_m3_sparse_skip",
    "patch_deepseek_v4_sparse_skip",
]

import importlib.util
import inspect

import torch

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import logger

# Original forwards; patched forwards reach them through a lazy import (compiler copies them).
_REFERENCE = {}
_DSA_KEY_ROPE = {}
_DSA_CACHE_OK = {}


def _transformers_tested():
    """Checked on transformers 5.17-5.20; older 5.x indexers differ (no mask for unpadded SDPA,
    per-query MiniMax blocks), so those keep upstream."""
    try:
        import transformers
        from packaging.version import Version
        return Version(transformers.__version__).release >= (5, 17)
    except Exception:
        return False


def _modeling_module(model_type):
    name = f"transformers.models.{model_type}.modeling_{model_type}"
    if not _transformers_tested():
        return None
    try:
        if importlib.util.find_spec(name) is None:
            return None
    except Exception:
        return None
    # Written-out imports: the risky-loader lint rejects import_module on a composed name.
    try:
        if model_type == "deepseek_v32":
            from transformers.models.deepseek_v32 import modeling_deepseek_v32 as modeling
        elif model_type == "glm_moe_dsa":
            from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as modeling
        elif model_type == "minimax_m3_vl":
            from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as modeling
        elif model_type == "deepseek_v4":
            from transformers.models.deepseek_v4 import modeling_deepseek_v4 as modeling
        else:
            return None
        return modeling
    except Exception:
        return None


def _is_transformers_function(function, model_type):
    """By code file: a compiled copy of the class keeps transformers' `__module__` but not its file."""
    try:
        function = inspect.unwrap(function)  # `@torch.no_grad()` wraps the DSA indexers
    except Exception:
        pass
    code = getattr(function, "__code__", None)
    filename = str(getattr(code, "co_filename", "")).replace("\\", "/")
    return f"/transformers/models/{model_type}/" in filename


def _source(function):
    try:
        return inspect.getsource(function)
    except Exception:
        return ""


# DeepSeek-V3.2 / GLM-MoE-DSA token indexer
_DSA_TOPK_SOURCE = "topk = min(self.index_topk, index_scores.shape[-1])"
_DSA_KEY_SOURCE = (
    "k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)",
    "k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)",
    "k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)",
    "k = past_key_values.update_indexer(k, self.layer_idx)",
)


def _dsa_indexer_forward(self, model_type, hidden_states, q_resid, position_embeddings, attention_mask,
                         position_ids, past_key_values):
    from unsloth_zoo.flex_attention.sparse import record_sparse_attn, sparse_skip_full_enabled

    key_len = attention_mask.shape[-1]
    if (
        0 < key_len <= self.index_topk
        and sparse_skip_full_enabled()
        and (past_key_values is None or _DSA_CACHE_OK.get(model_type, False))
    ):
        if past_key_values is not None:
            # Later steps past the budget score against this cache.
            cos, sin = position_embeddings
            k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)
            k_rot, k_pass = torch.split(
                k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
            )
            _, k_rot = _DSA_KEY_ROPE[model_type](k_rot, k_rot, cos, sin, unsqueeze_dim=2)
            k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)
            past_key_values.update_indexer(k, self.layer_idx)
        record_sparse_attn("skip_full", model_type)
        batch_size, seq_len = hidden_states.shape[:2]
        return torch.arange(key_len, device=hidden_states.device, dtype=torch.int32).expand(
            batch_size, seq_len, key_len
        )
    return _REFERENCE[f"{model_type}.indexer"](
        self, hidden_states, q_resid, position_embeddings, attention_mask, position_ids,
        past_key_values=past_key_values,
    )


def deepseek_v32_indexer_forward(self, hidden_states, q_resid, position_embeddings, attention_mask,
                                 position_ids=None, past_key_values=None):
    # Copied into the compiler cache: only torch, self, the arguments and the lazy import.
    from unsloth_zoo.temporary_patches.sparse_attention import _dsa_indexer_forward
    with torch.no_grad():
        return _dsa_indexer_forward(
            self, "deepseek_v32", hidden_states, q_resid, position_embeddings, attention_mask,
            position_ids, past_key_values,
        )


def glm_moe_dsa_indexer_forward(self, hidden_states, q_resid, position_embeddings, attention_mask,
                                position_ids=None, past_key_values=None):
    from unsloth_zoo.temporary_patches.sparse_attention import _dsa_indexer_forward
    with torch.no_grad():
        return _dsa_indexer_forward(
            self, "glm_moe_dsa", hidden_states, q_resid, position_embeddings, attention_mask,
            position_ids, past_key_values,
        )


def _patch_dsa(model_type, indexer_name, rope_name, patched_forward):
    modeling = _modeling_module(model_type)
    if modeling is None:
        return
    indexer_cls = getattr(modeling, indexer_name, None)
    if indexer_cls is None or indexer_cls.forward is patched_forward:
        return
    original = indexer_cls.forward
    if not _is_transformers_function(original, model_type):
        return
    source = _source(original)
    if _DSA_TOPK_SOURCE not in source or ".indices.to(torch.int32)" not in source:
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(f"Unsloth: {indexer_name}.forward changed upstream; sparse skip not applied.")
        return
    rope = getattr(modeling, rope_name, None)
    _DSA_CACHE_OK[model_type] = (
        rope is not None
        and all(line in source for line in _DSA_KEY_SOURCE)
        and f"q_rot, k_rot = {rope_name}(q_rot, k_rot, cos, sin, unsqueeze_dim=2)" in source
    )
    if rope is not None:
        _DSA_KEY_ROPE[model_type] = rope
    _REFERENCE[f"{model_type}.indexer"] = original
    indexer_cls.forward = patched_forward
    if UNSLOTH_ENABLE_LOGGING:
        logger.info(f"Unsloth: {indexer_name} skips scoring when top-k keeps every key.")


def patch_deepseek_v32_sparse_skip():
    _patch_dsa("deepseek_v32", "DeepseekV32Indexer", "apply_rotary_pos_emb", deepseek_v32_indexer_forward)


def patch_glm_moe_dsa_sparse_skip():
    _patch_dsa(
        "glm_moe_dsa", "GlmMoeDsaIndexer", "apply_rotary_pos_emb_interleave", glm_moe_dsa_indexer_forward
    )


# MiniMax-M3 block indexer
_MINIMAX_ROPE = {}
_MINIMAX_SOURCE = (
    "idx_k = self.k_proj(hidden_states).view(batch, q_len, 1, self.head_dim)",
    "idx_k = self.k_norm(idx_k).transpose(1, 2)",
    "idx_q, idx_k = apply_rotary_pos_emb(idx_q, idx_k, cos[..., : self.head_dim], sin[..., : self.head_dim])",
    "idx_k = past_key_values.layers[self.layer_idx].update_index(idx_k)",
    "topk = min(self.topk_blocks, num_key_blocks)",
    "q_block = position_ids // self.block_size",
)
_MINIMAX_MASK_SOURCE = (
    "safe = block_indices.masked_fill(block_indices < 0, num_key_blocks)",
    "bias.scatter_(-1, safe, 0.0)",
)


def _minimax_index_key_len(self, q_len, past_key_values):
    """Indexer key count after this step's update, or None when the cache layout is unknown."""
    if past_key_values is None:
        return q_len
    layer = past_key_values.layers[self.layer_idx]
    if hasattr(layer, "idx_cumulative_length"):
        return getattr(layer, "max_cache_len", None)
    if not hasattr(layer, "idx_keys"):
        return None
    cached = layer.idx_keys
    return q_len + (0 if cached is None else cached.shape[-2])


def _minimax_indexer_forward(self, hidden_states, position_embeddings, past_key_values, position_ids):
    from unsloth_zoo.flex_attention.sparse import record_sparse_attn, sparse_skip_full_enabled

    batch, q_len, _ = hidden_states.shape
    if (
        self.topk_blocks > 0
        and getattr(self.config, "_attn_implementation", None) in ("eager", "sdpa")
        and sparse_skip_full_enabled()
    ):
        # Decided before any cache mutation, so the reference can still run unchanged.
        k_len = _minimax_index_key_len(self, q_len, past_key_values)
        if k_len is not None and 0 < k_len and -(-k_len // self.block_size) <= self.topk_blocks:
            if past_key_values is not None:
                idx_k = self.k_proj(hidden_states).view(batch, q_len, 1, self.head_dim)
                idx_k = self.k_norm(idx_k).transpose(1, 2)
                cos, sin = position_embeddings
                _, idx_k = _MINIMAX_ROPE["rope"](
                    idx_k, idx_k, cos[..., : self.head_dim], sin[..., : self.head_dim]
                )
                past_key_values.layers[self.layer_idx].update_index(idx_k)
            record_sparse_attn("skip_full", "minimax_m3_vl")
            # Exactly the blocks with a non-future key; `build_block_mask` reads them as a set.
            num_key_blocks = -(-k_len // self.block_size)
            blocks = torch.arange(num_key_blocks, device=hidden_states.device)
            q_block = position_ids // self.block_size
            selected = torch.where(blocks <= q_block[..., None], blocks, -1)
            return selected[:, None].expand(batch, self.num_heads, q_len, num_key_blocks)
    return _REFERENCE["minimax_m3_vl.indexer"](self, hidden_states, position_embeddings, past_key_values, position_ids)


def minimax_m3_indexer_forward(self, hidden_states, position_embeddings, past_key_values, position_ids):
    from unsloth_zoo.temporary_patches.sparse_attention import _minimax_indexer_forward
    with torch.no_grad():
        return _minimax_indexer_forward(self, hidden_states, position_embeddings, past_key_values, position_ids)


def patch_minimax_m3_sparse_skip():
    modeling = _modeling_module("minimax_m3_vl")
    if modeling is None:
        return
    indexer_cls = getattr(modeling, "MiniMaxM3VLIndexer", None)
    rope = getattr(modeling, "apply_rotary_pos_emb", None)
    if indexer_cls is None or rope is None or indexer_cls.forward is minimax_m3_indexer_forward:
        return
    # Only `forward` is replaced: the compiler copies every other method from upstream's source.
    forward, build = indexer_cls.forward, getattr(indexer_cls, "build_block_mask", None)
    if build is None or not _is_transformers_function(forward, "minimax_m3_vl"):
        return
    if not (
        all(line in _source(forward) for line in _MINIMAX_SOURCE)
        and all(line in _source(build) for line in _MINIMAX_MASK_SOURCE)
    ):
        if UNSLOTH_ENABLE_LOGGING:
            logger.info("Unsloth: MiniMaxM3VLIndexer changed upstream; sparse skip not applied.")
        return
    _MINIMAX_ROPE["rope"] = rope
    _REFERENCE["minimax_m3_vl.indexer"] = forward
    indexer_cls.forward = minimax_m3_indexer_forward
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: MiniMaxM3VLIndexer skips scoring when top-k keeps every block.")


# DeepSeek-V4 CSA indexer
_V4_SOURCE = (
    "index_scores = self.scorer(q, compressed_kv, hidden_states)",
    "top_k = min(self.index_topk, compressed_len)",
    "future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)",
    "invalid = top_k_indices >= causal_threshold.unsqueeze(-1)",
)


def _deepseek_v4_indexer_forward(self, hidden_states, q_residual, position_ids, past_key_values, layer_idx):
    if getattr(self.scorer, "_unsloth_index_topk", None) != self.index_topk:
        self.scorer._unsloth_index_topk = self.index_topk
    return _REFERENCE["deepseek_v4.indexer"](self, hidden_states, q_residual, position_ids, past_key_values, layer_idx)


def deepseek_v4_indexer_forward(self, hidden_states, q_residual, position_ids, past_key_values, layer_idx):
    from unsloth_zoo.temporary_patches.sparse_attention import _deepseek_v4_indexer_forward
    with torch.no_grad():
        return _deepseek_v4_indexer_forward(self, hidden_states, q_residual, position_ids, past_key_values, layer_idx)


def _deepseek_v4_scorer_forward(self, q, compressed_kv, hidden_states):
    from unsloth_zoo.flex_attention.sparse import record_sparse_attn, sparse_skip_full_enabled

    index_topk = getattr(self, "_unsloth_index_topk", None)
    compressed_len = compressed_kv.shape[1]
    if index_topk is not None and compressed_len <= index_topk and sparse_skip_full_enabled():
        # top-k keeps every entry; the indexer still marks future ones -1.
        record_sparse_attn("skip_full", "deepseek_v4")
        return torch.zeros((q.shape[0], q.shape[1], compressed_len), dtype=torch.float32, device=q.device)
    return _REFERENCE["deepseek_v4.scorer"](self, q, compressed_kv, hidden_states)


def deepseek_v4_indexer_scorer_forward(self, q, compressed_kv, hidden_states):
    from unsloth_zoo.temporary_patches.sparse_attention import _deepseek_v4_scorer_forward
    return _deepseek_v4_scorer_forward(self, q, compressed_kv, hidden_states)


def patch_deepseek_v4_sparse_skip():
    modeling = _modeling_module("deepseek_v4")
    if modeling is None:
        return
    indexer_cls = getattr(modeling, "DeepseekV4Indexer", None)
    scorer_cls = getattr(modeling, "DeepseekV4IndexerScorer", None)
    if indexer_cls is None or scorer_cls is None or indexer_cls.forward is deepseek_v4_indexer_forward:
        return
    forward, scorer = indexer_cls.forward, scorer_cls.forward
    if not (_is_transformers_function(forward, "deepseek_v4") and _is_transformers_function(scorer, "deepseek_v4")):
        return
    if not all(line in _source(forward) for line in _V4_SOURCE):
        if UNSLOTH_ENABLE_LOGGING:
            logger.info("Unsloth: DeepseekV4Indexer changed upstream; sparse skip not applied.")
        return
    _REFERENCE["deepseek_v4.indexer"] = forward
    _REFERENCE["deepseek_v4.scorer"] = scorer
    indexer_cls.forward = deepseek_v4_indexer_forward
    scorer_cls.forward = deepseek_v4_indexer_scorer_forward
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: DeepseekV4Indexer skips scoring when top-k keeps every compressed entry.")


TEMPORARY_PATCHES.append(patch_deepseek_v32_sparse_skip)
TEMPORARY_PATCHES.append(patch_glm_moe_dsa_sparse_skip)
TEMPORARY_PATCHES.append(patch_minimax_m3_sparse_skip)
TEMPORARY_PATCHES.append(patch_deepseek_v4_sparse_skip)
