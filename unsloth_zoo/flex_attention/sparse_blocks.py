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

"""Long-context sparse attention on flex attention for indexer-selected models.

Upstream eager / SDPA paths expand an indexer's selection into a dense [B, H, S, T] mask and
run dense masked attention over it. Here the selection becomes a flex BlockMask instead:
128 x 128 tiles no query row selects are skipped, tiles every row keeps run without a
mask_mod, and boundary tiles read the exact per-row selection. Outputs match the dense path
up to floating-point reduction order.

- MiniMax M3: block-sparse selection, [B, H_idx, S, topk_blocks] block ids (-1 = none).
- DSA (DeepSeek V3.2, GLM-MoE-DSA): token selection, [B, S, topk] key ids.

Block indices are always passed to BlockMask.from_kv_blocks in ascending order: flex's
backward transposes the kv order into q order and silently miscomputes on unsorted lists.
"""

__all__ = [
    "sparse_flex_device_ok",
    "tile_block_mask",
    "minimax_m3_flex_attention",
    "minimax_m3_select_blocks",
    "dsa_flex_attention",
    "dsa_select_topk",
    "masked_sdpa_picks_cudnn",
]

import functools
import os

import torch
import torch.nn.functional as F

from .sparse import record_sparse_attn, sparse_flex_enabled

_TILE = 128
# (kind, device index, dtype, head dims) -> repr of the first failure; later calls skip flex.
_FAILED = {}
# fp32 elements per indexer score chunk (1 GiB); smaller problems run in one shot like upstream.
_SCORE_BUDGET = 1 << 28


def _count(kind, model_type):
    record_sparse_attn(kind, model_type)


def _q_chunk(per_row = 1):
    env = os.environ.get("UNSLOTH_SPARSE_INDEXER_Q_CHUNK")
    if env:
        try:
            return max(16, int(env))
        except ValueError:
            pass
    return max(256, _SCORE_BUDGET // max(1, per_row))


@functools.lru_cache(maxsize = None)
def _device_supported(index):
    if torch.version.hip is not None:
        return True  # first failure latches the fallback
    return torch.cuda.get_device_capability(index) >= (8, 0)


@functools.lru_cache(maxsize = None)
def _flex_available():
    try:
        from ..flex_attention.utils import HAS_FLEX_ATTENTION
        import triton  # noqa: F401
    except Exception:
        return False
    if os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") == "1":
        return False  # uncompiled flex is the dense math fallback
    return bool(HAS_FLEX_ATTENTION)


def sparse_flex_device_ok(query):
    return (
        sparse_flex_enabled()
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and _flex_available()
        and _device_supported(query.device.index or 0)
    )


def _failure_key(kind, query, key, value):
    return (kind, query.device.index, query.dtype, query.shape[-1], value.shape[-1], query.shape[1], key.shape[1])


def _virtual_kv_heads(query, key, value):
    """Repeat KV heads so each holds at most 4 query heads: flex's GQA kernels lose most of their
    parallelism at high group counts (one K/V head per 16+ query heads). Exact: query head h still
    reads original KV head h // (H_q / H_kv)."""
    n_q, n_kv = query.shape[1], key.shape[1]
    if os.environ.get("UNSLOTH_SPARSE_FLEX_VIRTUAL_KV", "1") == "0" or n_q // n_kv <= 4:
        return key, value
    target = n_q // 4
    if n_q % target or target % n_kv:
        return key, value
    rep = target // n_kv
    return key.repeat_interleave(rep, dim = 1), value.repeat_interleave(rep, dim = 1)


def _flex(query, key, value, block_mask, scale):
    from ..flex_attention.utils import flex_attention
    key, value = _virtual_kv_heads(query, key, value)
    return flex_attention(
        query, key, value, block_mask = block_mask, scale = scale,
        enable_gqa = query.shape[1] != key.shape[1],
    )


def _pack_ascending(selected):
    """(num, ids) for BlockMask.from_kv_blocks: active block ids first, ascending."""
    n_kv = selected.shape[-1]
    ar = torch.arange(n_kv, device = selected.device, dtype = torch.int32)
    num = selected.sum(dim = -1, dtype = torch.int32)
    ids = torch.where(selected, ar, n_kv).argsort(dim = -1).to(torch.int32)
    return num, ids


def tile_block_mask(any_tiles, all_tiles, mask_mod, q_len, kv_len):
    """BlockMask from per-tile flags [B, H, n_q_tiles, n_kv_tiles]: all -> full, any -> partial."""
    from torch.nn.attention.flex_attention import BlockMask
    full = all_tiles
    partial = any_tiles & ~all_tiles
    kv_num, kv_ids = _pack_ascending(partial)
    full_num, full_ids = _pack_ascending(full)
    return BlockMask.from_kv_blocks(
        kv_num, kv_ids, full_num, full_ids,
        BLOCK_SIZE = _TILE, mask_mod = mask_mod, seq_lengths = (q_len, kv_len),
    )


def _pad_to(x, dim, size, value):
    extra = size - x.shape[dim]
    if extra <= 0:
        return x
    pad = [0, 0] * (x.dim() - 1 - dim % x.dim()) + [0, extra]
    return F.pad(x, pad, value = value)


def _base_keep(attention_mask, q_pad, kv_pad):
    """Padded bool keep mask [B or 1, 1, q_pad, kv_pad] from a 4D bool / additive mask."""
    keep = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    if keep.shape[1] != 1:
        keep = keep[:, :1]  # head-shared masks only (callers check)
    keep = _pad_to(_pad_to(keep, 2, q_pad, False), 3, kv_pad, False)
    return keep


def _tiles_any_all(keep, n_q, n_kv):
    """[..., q_pad, kv_pad] bool -> any / all over each 128 x 128 tile."""
    shape = keep.shape[:-2]
    t = keep.reshape(*shape, n_q, _TILE, n_kv, _TILE)
    return t.any(dim = -1).any(dim = -2), t.all(dim = -1).all(dim = -2)


# --------------------------------------------------------------------------------------
# MiniMax M3 (block-sparse)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def minimax_m3_select_blocks(indexer, hidden_states, position_embeddings, past_key_values, position_ids, apply_rotary_pos_emb):
    """MiniMaxM3VLIndexer.forward with query-chunked scoring: [B, H_idx, S_q, topk] block ids, -1 padded.

    Same arithmetic per query row as upstream; only the [B, H_idx, S_q, S_k] fp32 score buffer
    is bounded to one chunk of queries."""
    batch, q_len, _ = hidden_states.shape
    idx_q = indexer.q_proj(hidden_states).view(batch, q_len, -1, indexer.head_dim)
    idx_q = indexer.q_norm(idx_q).transpose(1, 2)
    idx_k = indexer.k_proj(hidden_states).view(batch, q_len, 1, indexer.head_dim)
    idx_k = indexer.k_norm(idx_k).transpose(1, 2)
    cos, sin = position_embeddings
    idx_q, idx_k = apply_rotary_pos_emb(idx_q, idx_k, cos[..., : indexer.head_dim], sin[..., : indexer.head_dim])
    if past_key_values is not None:
        idx_k = past_key_values.layers[indexer.layer_idx].update_index(idx_k)

    k_len = idx_k.shape[2]
    block = indexer.block_size
    num_key_blocks = -(-k_len // block)
    pad = num_key_blocks * block - k_len
    topk = min(indexer.topk_blocks, num_key_blocks)
    k_positions = torch.arange(k_len, device = idx_q.device)
    idx_k_t = idx_k.float().transpose(-1, -2)
    chunk = _q_chunk(batch * indexer.num_heads * num_key_blocks * block)
    out = []
    for start in range(0, q_len, chunk):
        end = min(start + chunk, q_len)
        pos = position_ids[:, start:end]
        scores = torch.matmul(idx_q[:, :, start:end].float(), idx_k_t)
        token_future = k_positions[None, None, None, :] > pos[:, None, :, None]
        scores = scores.masked_fill(token_future, float("-inf"))
        if pad:
            scores = F.pad(scores, (0, pad), value = float("-inf"))
        block_scores = scores.view(batch, indexer.num_heads, end - start, num_key_blocks, block).amax(dim = -1)
        del scores
        q_block = pos // block
        if indexer.local_blocks > 0:
            local = torch.arange(indexer.local_blocks, device = idx_q.device)
            local_idx = (q_block[..., None] - local.view(1, 1, -1)).clamp(min = 0)
            local_idx = local_idx.unsqueeze(1).expand(-1, indexer.num_heads, -1, -1)
            block_scores.scatter_(-1, local_idx, float("inf"))
        topk_scores, topk_indices = block_scores.topk(topk, dim = -1)
        out.append(topk_indices.masked_fill(topk_scores == float("-inf"), -1))
    if len(out) > 1:
        _count("indexer_chunked", "minimax_m3_vl")
    return torch.cat(out, dim = 2) if len(out) > 1 else out[0]


def minimax_m3_flex_attention(query, key, value, block_indices, attention_mask, position_ids, scaling, block_size):
    """Flex attention under MiniMax M3's block selection; returns [B, S, H, D] like sdpa_attention_forward.

    Keeps exactly the (query, key) pairs of MiniMaxM3VLIndexer.build_block_mask: the query's
    selected key blocks for its indexer head (h // (H_q / H_idx)), intersected with the base mask
    (or key slot <= position id when the base mask is None)."""
    batch, n_heads, q_len, _ = query.shape
    kv_len = key.shape[2]
    n_idx = block_indices.shape[1]
    group = n_heads // n_idx
    n_q = -(-q_len // _TILE)
    n_kv = -(-kv_len // _TILE)
    q_pad, kv_pad = n_q * _TILE, n_kv * _TILE
    num_key_blocks = -(-kv_len // block_size)
    per_block = block_size // _TILE

    # Per-row selection, one spare column absorbing the -1 slots.
    safe = block_indices.masked_fill(block_indices < 0, num_key_blocks).long()
    sel = torch.zeros(batch, n_idx, q_pad, num_key_blocks + 1, dtype = torch.bool, device = query.device)
    sel[:, :, :q_len].scatter_(-1, safe, True)
    sel_tiles = sel[..., :num_key_blocks].repeat_interleave(per_block, dim = -1)[..., :n_kv]
    sel_tiles = sel_tiles.view(batch, n_idx, n_q, _TILE, n_kv)
    sel_any, sel_all = sel_tiles.any(dim = 3), sel_tiles.all(dim = 3)

    if attention_mask is None:
        pos = _pad_to(position_ids.to(torch.int64), 1, q_pad, -1)  # padded rows keep nothing
        pos_t = pos.view(batch, n_q, _TILE)
        q_max, q_min = pos_t.max(dim = -1).values, pos_t.min(dim = -1).values
        tile_start = torch.arange(n_kv, device = query.device) * _TILE
        tile_end = tile_start + _TILE - 1
        base_any = (tile_start[None, None, :] <= q_max[..., None]).unsqueeze(1)
        base_all = ((tile_end[None, None, :] <= q_min[..., None]) & (tile_end < kv_len)[None, None, :]).unsqueeze(1)
        base = None
    else:
        base = _base_keep(attention_mask, q_pad, kv_pad)
        base_any, base_all = _tiles_any_all(base, n_q, n_kv)
        pos = None

    any_tiles = (sel_any & base_any).repeat_interleave(group, dim = 1)
    all_tiles = (sel_all & base_all).repeat_interleave(group, dim = 1)

    if base is None:
        def mask_mod(b, h, q_idx, kv_idx):
            return sel[b, h // group, q_idx, kv_idx // block_size] & (kv_idx <= pos[b, q_idx])
    else:
        # [B, S, T] with one index per dim: torch 2.10's flex tracing breaks on constant subscripts.
        base3 = base[:, 0].expand(batch, -1, -1).contiguous()

        def mask_mod(b, h, q_idx, kv_idx):
            return sel[b, h // group, q_idx, kv_idx // block_size] & base3[b, q_idx, kv_idx]

    block_mask = tile_block_mask(any_tiles, all_tiles, mask_mod, q_len, kv_len)
    out = _flex(query, key, value, block_mask, scaling)
    _count("flex", "minimax_m3_vl")
    return out.transpose(1, 2).contiguous()


def minimax_m3_flex_eligible(module, query, key, attention_mask, kwargs):
    indexer = module.indexer
    if indexer is None or module.config._attn_implementation not in ("eager", "sdpa"):
        return False
    if key.shape[2] <= indexer.topk_blocks * indexer.block_size or query.shape[2] < 16:
        return False
    if indexer.block_size % _TILE or kwargs.get("output_attentions"):
        return False
    if module.training and module.attention_dropout:
        return False
    if attention_mask is not None and (attention_mask.dim() != 4 or attention_mask.shape[1] != 1):
        return False
    if not sparse_flex_device_ok(query):
        return False
    return _failure_key("minimax_m3", query, key, key) not in _FAILED


# --------------------------------------------------------------------------------------
# DSA (DeepSeek V3.2, GLM-MoE-DSA): token-level top-k
# --------------------------------------------------------------------------------------
@torch.no_grad()
def dsa_select_topk(indexer, hidden_states, q_resid, position_embeddings, attention_mask, past_key_values, apply_rope, model_type = "dsa"):
    """DeepseekV32Indexer / GlmMoeDsaIndexer forward with query-chunked scoring: [B, S, topk] int32.

    Upstream materialises [B, S, H_idx, T] fp32 scores (16 GiB at S = T = 8192). Each query row's
    arithmetic is unchanged; the scores exist one chunk of queries at a time."""
    batch_size, seq_len, _ = hidden_states.shape
    cos, sin = position_embeddings
    q = indexer.wq_b(q_resid).view(batch_size, seq_len, indexer.n_heads, indexer.head_dim)
    q_rot, q_pass = torch.split(q, [indexer.qk_rope_head_dim, indexer.head_dim - indexer.qk_rope_head_dim], dim = -1)
    k = indexer.k_norm(indexer.wk(hidden_states)).unsqueeze(2)
    k_rot, k_pass = torch.split(k, [indexer.qk_rope_head_dim, indexer.head_dim - indexer.qk_rope_head_dim], dim = -1)
    q_rot, k_rot = apply_rope(q_rot, k_rot, cos, sin, unsqueeze_dim = 2)
    q = torch.cat([q_rot, q_pass], dim = -1)
    k = torch.cat([k_rot, k_pass], dim = -1).squeeze(2)
    if past_key_values is not None:
        k = past_key_values.update_indexer(k, indexer.layer_idx)
    k_t = k.transpose(-1, -2).float().unsqueeze(1)  # [B, 1, D, T]
    weights = indexer.weights_proj(hidden_states.to(indexer.weights_proj.weight.dtype)).float() * (indexer.n_heads ** -0.5)
    topk = min(indexer.index_topk, k.shape[1])
    chunk = _q_chunk(batch_size * indexer.n_heads * k.shape[1])
    out = []
    for start in range(0, seq_len, chunk):
        end = min(start + chunk, seq_len)
        scores = torch.matmul(q[:, start:end].float(), k_t) * indexer.softmax_scale
        scores = F.relu(scores)
        index_scores = torch.matmul(weights[:, start:end].unsqueeze(-2), scores).squeeze(-2)
        del scores
        mask = attention_mask[:, start:end]
        if mask.dtype == torch.bool:
            index_scores = index_scores.masked_fill(~mask, float("-inf"))
        else:
            index_scores = index_scores + mask
        out.append(index_scores.topk(topk, dim = -1).indices.to(torch.int32))
    if len(out) > 1:
        _count("indexer_chunked", model_type)
    return torch.cat(out, dim = 1) if len(out) > 1 else out[0]


def dsa_flex_attention(query, key, value, topk_indices, attention_mask, scaling, model_type = "dsa"):
    """Flex attention over the indexer's selected keys; returns [B, S, H, D_v].

    Keeps exactly the pairs of upstream's eager / SDPA path: selected by top-k AND allowed by
    the 4D base mask. The bitmap and BlockMask are stored on `topk_indices`, so GLM's shared
    layers, which receive the producer's tensor object, reuse them within the step."""
    cached = getattr(topk_indices, "_unsloth_dsa_flex", None)
    if cached is not None and cached[0] is attention_mask and cached[1] == key.shape[2]:
        _count("flex_reused", model_type)
        block_mask = cached[2]
    else:
        batch, _, q_len, _ = query.shape
        kv_len = key.shape[2]
        n_q, n_kv = -(-q_len // _TILE), -(-kv_len // _TILE)
        q_pad, kv_pad = n_q * _TILE, n_kv * _TILE
        keep = torch.zeros(batch, q_pad, kv_pad, dtype = torch.bool, device = query.device)
        keep[:, :q_len].scatter_(-1, topk_indices.long(), True)
        base = _base_keep(attention_mask, q_pad, kv_pad)[:, 0]
        keep &= base
        any_tiles, all_tiles = _tiles_any_all(keep, n_q, n_kv)

        def mask_mod(b, h, q_idx, kv_idx):
            return keep[b, q_idx, kv_idx]

        block_mask = tile_block_mask(any_tiles.unsqueeze(1), all_tiles.unsqueeze(1), mask_mod, q_len, kv_len)
        try:
            topk_indices._unsloth_dsa_flex = (attention_mask, kv_len, block_mask)
        except Exception:
            pass
    out = _flex(query, key, value, block_mask, scaling)
    _count("flex", model_type)
    return out.transpose(1, 2).contiguous()


_SDPA_PICKS_CUDNN = {}


def masked_sdpa_picks_cudnn(query, key, value):
    """Whether torch's own SDPA choice for a bool-masked call of this shape class is cuDNN.

    Where it is (H100 on torch 2.14), dense masked SDPA beats flex on DSA's scattered selections
    (GLM-5 4K keys 25 vs 34 ms, 8K 95 vs 136 ms); elsewhere flex wins (A100 on 2.10 / 2.14,
    H100 on 2.10). Probed once per (device, dtype, heads, head dims, grad, cuDNN enabled)."""
    needs_grad = torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad)
    cache_key = (
        query.device.index, query.dtype, query.shape[1], key.shape[1], query.shape[-1], value.shape[-1],
        needs_grad, torch.backends.cuda.cudnn_sdp_enabled(),
    )
    picks = _SDPA_PICKS_CUDNN.get(cache_key)
    if picks is None:
        picks = False
        try:
            from torch.nn.attention import SDPBackend
            cudnn = int(SDPBackend.CUDNN_ATTENTION)
            dev, dt = query.device, query.dtype
            q = torch.empty(1, query.shape[1], 128, query.shape[-1], device = dev, dtype = dt)
            k = torch.empty(1, query.shape[1], 128, key.shape[-1], device = dev, dtype = dt)
            v = torch.empty(1, query.shape[1], 128, value.shape[-1], device = dev, dtype = dt)
            if needs_grad:
                q, k, v = (t.requires_grad_() for t in (q, k, v))
            mask = torch.ones(1, 1, 128, 128, device = dev, dtype = torch.bool)
            with torch.set_grad_enabled(needs_grad):
                picks = int(torch._fused_sdp_choice(q, k, v, attn_mask = mask, dropout_p = 0.0, is_causal = False)) == cudnn
        except Exception:
            picks = False
        _SDPA_PICKS_CUDNN[cache_key] = picks
    return picks


def dsa_flex_eligible(module, query, key, value, attention_mask, kwargs, min_keys):
    """Flex above the selection budget, unless torch would run this masked call on cuDNN.

    `min_keys` is UNSLOTH_SPARSE_ATTN_DSA_MIN_KEYS when set (forces flex at or above it),
    else None: the budget plus the cuDNN probe decide."""
    if module.config._attn_implementation not in ("eager", "sdpa"):
        return False
    budget = int(module.config.index_topk)
    if key.shape[2] <= budget or query.shape[2] < 16 or kwargs.get("output_attentions"):
        return False
    if min_keys is not None and key.shape[2] < min_keys:
        return False
    if module.training and module.attention_dropout:
        return False
    if attention_mask is None or attention_mask.dim() != 4 or attention_mask.shape[1] != 1:
        return False
    if not sparse_flex_device_ok(query):
        return False
    if _failure_key("dsa", query, key, value) in _FAILED:
        return False
    if min_keys is None and module.config._attn_implementation == "sdpa" and masked_sdpa_picks_cudnn(query, key, value):
        return False
    return True


def record_failure(kind, query, key, value, error):
    """Latch a flex failure for this shape class; the caller falls back to the dense path."""
    _FAILED[_failure_key(kind, query, key, value)] = repr(error)[:500]
    _count("fallback", kind)
