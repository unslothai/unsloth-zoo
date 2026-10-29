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

"""GLM-5.3-Flash (glm5_next) pooled DSA: exact skip, chunked indexer, flex BlockMask.

Upstream scores every complete k-pool (index_kpool tokens, aligned at the row's first valid key)
with an fp32 [B, S, H_idx, S / kpool] buffer, keeps the top index_topk // kpool pools whose last
token is visible, appends the query's incomplete tail pool, then expands that into a dense
[B, S, T] int32 count + bool mask for eager / SDPA (no causal mask besides it).

- Within the budget (T < index_topk + kpool) every complete visible pool is selected: the
  selection is built without scoring (UNSLOTH_SPARSE_ATTN_SKIP_FULL=0 restores upstream).
- Above it the pools are scored in query chunks (UNSLOTH_SPARSE_INDEXER_CHUNKED=0 restores upstream).
- Opt-in only (UNSLOTH_SPARSE_ATTN_FLEX_GLM5_NEXT=1): above the budget on CUDA sm80+ / ROCm, attention
  runs on flex attention over a BlockMask whose mask_mod reads a [B, S, T / kpool] pool table plus the
  tail. Off by default: on A100 its layer step was 4.7-5.3x slower than upstream's dense SDPA (4K-32K)
  for 18-48% less peak memory. Without the opt-in (or with UNSLOTH_SPARSE_ATTN_FLEX=0) attention is
  upstream's dense mask + SDPA / eager over the patched indexer's selection.

Both patched indexer paths return the same key set per query as upstream in a fixed layout:
index_topk pool slots (whole pools, -1 padded) then index_kpool - 1 tail slots. The flex path
reads that layout, so it engages only when the patched indexer produced the indices.
"""

__all__ = ["patch_glm5_next_sparse"]

import importlib.util
import inspect

import torch
import torch.nn.functional as F

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING, publish_to_modeling_module
from .utils import logger

_FAMILY = "glm5_next"
_REFERENCE = {}
_MODELING = {}

_ATTN_NEEDLES = (
    "k_pass, k_rot = past_key_values.update(k_pass, k_rot, self.layer_idx)",
    "key_states, value_states = self.expand_kv(k_pass, k_rot)",
    "topk_indices = prev_topk_indices",
    "attention_mask = self.build_attention_mask_from_topk(",
    "return attn_output, attn_weights, topk_indices if self.next_skip_topk else None",
)
_INDEXER_NEEDLES = (
    "k = self.k_norm(self.wk(hidden_states)).view(hidden_shape).squeeze(2)",
    "gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)",
    "packed_states = torch.cat([k, gate_scores, valid_channel], dim=-1)",
    "packed_states = past_key_values.update_indexer(packed_states, self.layer_idx)",
    "pool_keys, pool_indices, pool_valid = self.get_pooled_states(packed_states=packed_states)",
    "scores = F.relu(scores * self.softmax_scale)",
    "select_k = min(self.index_topk // self.index_kpool, index_scores.shape[-1])",
    "topk_indices = self.append_visible_tail(topk_indices, visible_tokens, valid_keys)",
    "topk_indices = topk_indices.masked_fill(~attention_mask[..., None], -1)",
)
_MASK_NEEDLES = (
    "selected_counts.scatter_add_(-1, safe_indices, topk_valid.to(torch.int32))",
    "mask = selected_counts.ne(0).unsqueeze(1)",
)


def _stats():
    from unsloth_zoo.flex_attention import sparse
    return sparse


def _count(kind):
    _stats().record_sparse_attn(kind, _FAMILY)


def _tail_width(indexer):
    return indexer.index_kpool - 1 if indexer.index_kpool_always_select_tail else 0


def _predicted_key_len(indexer, q_len, past_key_values):
    if past_key_values is None:
        return q_len
    try:
        layer = past_key_values.layers[indexer.layer_idx]
    except Exception:
        return q_len
    cached = getattr(layer, "indexer_keys", None)
    # Static: the full allocation is pooled, also on the first prefill before the cache exists.
    max_len = getattr(layer, "max_cache_len", None)
    if isinstance(max_len, int) and max_len > 0:
        return max(max_len, q_len)
    if cached is None or cached.dim() < 2:
        return q_len
    return cached.shape[1] + q_len


# --------------------------------------------------------------------------------------
# Indexer
# --------------------------------------------------------------------------------------
@torch.no_grad()
def glm5_next_select(indexer, hidden_states, q_resid, attention_mask, past_key_values, allow_skip = True):
    """Glm5NextTextIndexer.forward with query-chunked scoring, or no scoring when every complete
    visible pool is selected anyway. [B, S, index_topk + tail] int32, the pool slots then the tail."""
    from unsloth_zoo.flex_attention import sparse_blocks as sb

    batch_size, seq_len = hidden_states.shape[:2]
    head_dim, kpool = indexer.head_dim, indexer.index_kpool
    k = indexer.k_norm(indexer.wk(hidden_states)).view(batch_size, seq_len, -1, head_dim).squeeze(2)
    gate_scores = F.linear(hidden_states, indexer.index_kpool_compress_gate)
    valid_channel = attention_mask.to(k.dtype)[..., None]
    packed_states = torch.cat([k, gate_scores, valid_channel], dim = -1)

    kv_len = seq_len
    current_length = seq_len
    if past_key_values is not None:
        cache_layer = past_key_values.layers[indexer.layer_idx]
        packed_states = past_key_values.update_indexer(packed_states, indexer.layer_idx)
        kv_len = cache_layer.keys.shape[-2]
        current_length = cache_layer.get_seq_length()

    valid_keys = packed_states[..., -1].bool()
    pool_keys, pool_indices, pool_valid = indexer.get_pooled_states(packed_states = packed_states)
    # Complete in-range pools per row <= packed length // kpool: within the budget all are selected.
    score = not (allow_skip and packed_states.shape[1] // kpool <= indexer.index_topk // kpool)
    num_pools = pool_indices.shape[1]
    select_k = min(indexer.index_topk // kpool, num_pools)
    device = hidden_states.device
    batch_idx = torch.arange(batch_size, device = device)[:, None, None]
    pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)
    kv_positions = torch.arange(valid_keys.shape[-1], device = device)
    q_positions = current_length - seq_len + torch.arange(seq_len, device = device)
    tail = _tail_width(indexer)
    width = indexer.index_topk + tail

    if score:
        q = indexer.wq_b(q_resid).view(batch_size, seq_len, -1, head_dim)
        keys_t = pool_keys.transpose(-1, -2).float().unsqueeze(1)
        w_dtype = indexer.weights_proj.weight.dtype
        x = hidden_states if hidden_states.dtype == w_dtype else hidden_states.to(w_dtype)
        weights = indexer.weights_proj(x).float() * (indexer.n_heads ** -0.5)
        per_row = batch_size * (indexer.n_heads * num_pools + valid_keys.shape[-1] // 4)
    else:
        per_row = batch_size * valid_keys.shape[-1]

    out = []
    ranges = sb._q_ranges(seq_len, per_row)
    for start, end in ranges:
        visible = (kv_positions[None, None, :] <= q_positions[None, start:end, None]) & valid_keys[:, None, :]
        pool_visible = visible.gather(-1, pool_end[:, None, :].expand(batch_size, end - start, -1))
        valid_candidates = pool_visible & pool_valid[:, None]
        if score:
            scores = torch.matmul(q[:, start:end].float(), keys_t)
            scores = F.relu(scores * indexer.softmax_scale)
            index_scores = torch.matmul(weights[:, start:end].unsqueeze(-2), scores).squeeze(-2)
            del scores
            index_scores = index_scores.masked_fill(~valid_candidates, torch.finfo(index_scores.dtype).min)
            selected = index_scores.topk(select_k, dim = -1).indices
            del index_scores
            selected_valid = valid_candidates.gather(-1, selected)
            selected_indices = pool_indices[batch_idx, selected]
        else:
            selected_valid = valid_candidates
            selected_indices = pool_indices[:, None].expand(-1, end - start, -1, -1)
        topk_indices = selected_indices.flatten(-2)
        topk_indices = topk_indices.masked_fill(
            ~selected_valid[..., None].expand_as(selected_indices).flatten(-2), -1,
        )
        # Fixed layout: index_topk pool slots, then the tail (upstream pads after the tail).
        topk_indices = F.pad(topk_indices, (0, indexer.index_topk - topk_indices.shape[-1]), value = -1)
        if tail:
            topk_indices = indexer.append_visible_tail(topk_indices, visible, valid_keys)
        del visible
        topk_indices = topk_indices[..., :width]
        topk_indices = topk_indices.masked_fill(~attention_mask[:, start:end, None], -1)
        out.append(topk_indices.to(torch.int32))
    _count("indexer_chunked" if score and len(out) > 1 else ("skip_full" if not score else "indexer_scored"))
    return torch.cat(out, dim = 1) if len(out) > 1 else out[0]


_select_outside = torch.compiler.disable(glm5_next_select)


def unsloth_glm5_next_indexer_forward(self, hidden_states, q_resid, attention_mask, past_key_values):
    from unsloth_zoo.flex_attention import sparse
    from unsloth_zoo.flex_attention import sparse_blocks as sb
    from unsloth_zoo.temporary_patches.glm5_next_sparse_flex import _REFERENCE, _select_outside, glm5_next_select

    skip, chunked = sparse.sparse_skip_full_enabled(), sparse.sparse_indexer_chunked_enabled()
    below = _predicted_key_len(self, hidden_states.shape[1], past_key_values) < self.index_topk + self.index_kpool
    if (below and not skip) or (not below and not chunked):
        return _REFERENCE["indexer"](self, hidden_states, q_resid, attention_mask, past_key_values)
    select = _select_outside if torch.compiler.is_compiling() and sb.in_unsloth_block() else glm5_next_select
    return select(self, hidden_states, q_resid, attention_mask, past_key_values, skip)


# --------------------------------------------------------------------------------------
# Flex attention
# --------------------------------------------------------------------------------------
def _pool_starts(topk_indices, kpool, index_topk):
    """(first token of each selected pool or -1 [B, S, index_topk // kpool], valid, row phase [B])."""
    starts = topk_indices[..., 0:index_topk:kpool].long()
    valid = starts >= 0
    # Every pool of a row starts at first_key + j * kpool: one phase per row.
    phase = torch.where(valid, starts.remainder(kpool), -1).amax(dim = (1, 2)).clamp(min = 0)
    return starts, valid, phase


def glm5_next_flex_attention(query, key, value, topk_indices, scaling, kpool, index_topk, tail):
    """Flex attention over exactly the keys in `topk_indices` (pool slots then tail); [B, S, H, D_v]."""
    from unsloth_zoo.flex_attention import sparse_blocks as sb

    sb._eager_only("glm5_next_flex_attention")
    kv_len = key.shape[2]
    cached = getattr(topk_indices, "_unsloth_glm5_next_flex", None)
    if cached is not None and cached[0] == kv_len:
        _count("flex_reused")
        block_mask = cached[1]
    else:
        block_mask = glm5_next_block_mask(topk_indices, kv_len, kpool, index_topk, tail)
        try:
            topk_indices._unsloth_glm5_next_flex = (kv_len, block_mask)
        except Exception:
            pass
    out = sb._flex(query, key, value, block_mask, scaling)
    _count("flex")
    return out.transpose(1, 2).contiguous()


def glm5_next_block_mask(topk_indices, kv_len, kpool, index_topk, tail):
    from unsloth_zoo.flex_attention import sparse_blocks as sb

    tile = sb._TILE
    batch, q_len, _ = topk_indices.shape
    device = topk_indices.device
    n_q, n_kv = -(-q_len // tile), -(-kv_len // tile)
    q_pad, kv_pad = n_q * tile, n_kv * tile

    starts, valid, phase = _pool_starts(topk_indices, kpool, index_topk)
    # Pool grid shifted by the row's phase: key t is in grid cell (t - phase) // kpool (+1 so t < phase -> 0).
    n_grid = (kv_len + kpool - 1) // kpool + 2
    cell = torch.where(valid, (starts - phase[:, None, None]) // kpool + 1, n_grid)
    pools = torch.zeros(batch, q_pad, n_grid + 1, dtype = torch.bool, device = device)
    pools[:, :q_len].scatter_(-1, cell, True)
    pools = pools[..., :n_grid].contiguous()
    pools[..., 0] = False  # the partial cell before the first full pool is never a whole pool
    if tail:
        tail_ids = _pad_rows(topk_indices[..., index_topk:index_topk + tail].long(), q_pad, -1)
        offsets = torch.arange(tail, device = device)
        tail_base = (tail_ids - offsets).amax(dim = -1).contiguous()
        tail_ids = tail_ids.contiguous()

    # Tile tables from the indices themselves, in query chunks (bounded memory).
    any_tiles = torch.empty(batch, n_q, n_kv, dtype = torch.bool, device = device)
    all_tiles = torch.empty_like(any_tiles)
    rows = max(tile, (sb._SCORE_BUDGET // max(1, batch * (kv_pad + 1))) // tile * tile)
    for r0 in range(0, q_pad, rows):
        r1 = min(r0 + rows, q_pad)
        keep = torch.zeros(batch, r1 - r0, kv_pad + 1, dtype = torch.bool, device = device)
        if r0 < q_len:
            idx = topk_indices[:, r0:min(r1, q_len)].long()
            keep[:, : idx.shape[1]].scatter_(-1, torch.where(idx >= 0, idx, kv_pad), True)
        t = keep[..., :kv_pad].view(batch, (r1 - r0) // tile, tile, n_kv, tile)
        any_tiles[:, r0 // tile:r1 // tile] = t.any(dim = -1).any(dim = -2)
        all_tiles[:, r0 // tile:r1 // tile] = t.all(dim = -1).all(dim = -2)
        del keep, t

    if tail:
        def mask_mod(b, h, q_idx, kv_idx):
            rel = kv_idx - phase[b]
            in_pool = (rel >= 0) & pools[b, q_idx, torch.where(rel >= 0, rel // kpool + 1, 0)]
            t = kv_idx - tail_base[b, q_idx]
            in_tail = (t >= 0) & (t < tail) & (tail_ids[b, q_idx, t.clamp(0, tail - 1)] == kv_idx)
            return in_pool | in_tail
    else:
        def mask_mod(b, h, q_idx, kv_idx):
            rel = kv_idx - phase[b]
            return (rel >= 0) & pools[b, q_idx, torch.where(rel >= 0, rel // kpool + 1, 0)]

    return sb.tile_block_mask(any_tiles.unsqueeze(1), all_tiles.unsqueeze(1), mask_mod, q_len, kv_len)


def _pad_rows(x, rows, value):
    extra = rows - x.shape[1]
    return x if extra <= 0 else F.pad(x, (0, 0, 0, extra), value = value)


def glm5_next_flex_eligible(module, query, key, value, kwargs):
    from unsloth_zoo.flex_attention import sparse, sparse_blocks as sb
    from unsloth_zoo.temporary_patches.sparse_attention_flex import _dsa_min_keys

    if not sparse.sparse_flex_opted_in(_FAMILY):
        return False
    config = module.config
    if config._attn_implementation not in ("eager", "sdpa"):
        return False
    # The layout the mask reads comes from the patched indexer (off with UNSLOTH_SPARSE_INDEXER_CHUNKED=0).
    if not sparse.sparse_indexer_chunked_enabled():
        return False
    if key.shape[2] < config.index_topk + config.index_kpool or query.shape[2] < 16:
        return False
    if sb._wants_attentions(module, kwargs) or sb._compiled_without_break():
        return False
    min_keys = _dsa_min_keys()
    if min_keys is not None and key.shape[2] < min_keys:
        return False
    if module.training and module.attention_dropout:
        return False
    if not sb.sparse_flex_device_ok(query):
        return False
    return sb._failure_key(_FAMILY, query, key, value) not in sb._FAILED


def unsloth_glm5_next_attention_forward(self, hidden_states, attention_mask, past_key_values = None, prev_topk_indices = None, **kwargs):
    from unsloth_zoo.temporary_patches.glm5_next_sparse_flex import (
        _MODELING, glm5_next_flex_attention, glm5_next_flex_eligible,
    )
    from unsloth_zoo.temporary_patches.sparse_attention_flex import _run_flex

    m = _MODELING["modeling"]
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    query_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim = -1)
    k_pass = self.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, self.kv_lora_rank)
    k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

    if past_key_values is not None:
        k_pass, k_rot = past_key_values.update(k_pass, k_rot, self.layer_idx)

    key_states, value_states = self.expand_kv(k_pass, k_rot)

    if self.indexer is not None:
        topk_indices = self.indexer(
            hidden_states = hidden_states,
            q_resid = q_resid,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
        )
    else:
        if prev_topk_indices is None:
            raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
        topk_indices = prev_topk_indices
    shared = topk_indices if self.next_skip_topk else None

    if glm5_next_flex_eligible(self, query_states, key_states, value_states, kwargs):
        config = self.config
        tail = config.index_kpool - 1 if config.index_kpool_always_select_tail else 0
        attn_output = _run_flex(
            _FAMILY, "GLM-5.3-Flash", glm5_next_flex_attention, query_states, key_states, value_states,
            query_states, key_states, value_states, topk_indices, self.scaling,
            config.index_kpool, config.index_topk, tail,
        )
        if attn_output is not None:
            attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
            return self.o_proj(attn_output), None, shared

    attention_mask = self.build_attention_mask_from_topk(
        topk_indices = topk_indices,
        query_states = query_states,
        kv_length = key_states.shape[2],
    )
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
        **kwargs,
    )
    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, shared


# --------------------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------------------
def _source(function):
    try:
        return inspect.getsource(inspect.unwrap(function))
    except Exception:
        return ""


def _is_transformers_function(function):
    try:
        function = inspect.unwrap(function)
    except Exception:
        pass
    filename = str(getattr(getattr(function, "__code__", None), "co_filename", "")).replace("\\", "/")
    return f"/transformers/models/{_FAMILY}/" in filename


def patch_glm5_next_sparse():
    from .sparse_attention import _transformers_tested

    if not _transformers_tested():
        return
    try:
        if importlib.util.find_spec("transformers.models.glm5_next.modeling_glm5_next") is None:
            return
        from transformers.models.glm5_next import modeling_glm5_next as m
    except Exception:
        return
    attn = getattr(m, "Glm5NextTextAttention", None)
    indexer = getattr(m, "Glm5NextTextIndexer", None)
    if attn is None or indexer is None or getattr(attn.forward, "_unsloth_sparse_flex", False):
        return
    if not (_is_transformers_function(attn.forward) and _is_transformers_function(indexer.forward)):
        return
    attn_source, indexer_source = _source(attn.forward), _source(indexer.forward)
    mask_source = _source(getattr(attn, "build_attention_mask_from_topk", None))
    if (
        not all(n in attn_source for n in _ATTN_NEEDLES)
        or not all(n in indexer_source for n in _INDEXER_NEEDLES)
        or not all(n in mask_source for n in _MASK_NEEDLES)
        or not all(hasattr(indexer, n) for n in ("get_pooled_states", "append_visible_tail"))
    ):
        if UNSLOTH_ENABLE_LOGGING:
            logger.info("Unsloth: GLM-5.3-Flash DSA source differs from the supported layout; sparse paths not installed.")
        return
    _REFERENCE["indexer"] = indexer.forward
    _REFERENCE["attention"] = attn.forward
    _MODELING["modeling"] = m
    publish_to_modeling_module(
        m,
        unsloth_glm5_next_attention_forward = unsloth_glm5_next_attention_forward,
        unsloth_glm5_next_indexer_forward = unsloth_glm5_next_indexer_forward,
    )

    def forward(self, hidden_states, attention_mask, past_key_values = None, prev_topk_indices = None, **kwargs):
        return unsloth_glm5_next_attention_forward(
            self, hidden_states, attention_mask, past_key_values, prev_topk_indices, **kwargs
        )

    def indexer_forward(self, hidden_states, q_resid, attention_mask, past_key_values):
        with torch.no_grad():
            return unsloth_glm5_next_indexer_forward(self, hidden_states, q_resid, attention_mask, past_key_values)

    forward._unsloth_sparse_flex = True
    indexer_forward._unsloth_sparse_flex = True
    attn.forward = forward
    indexer.forward = indexer_forward
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: GLM-5.3-Flash DSA: exact skip within the budget, chunked indexer above it (flex opt-in).")


TEMPORARY_PATCHES.append(patch_glm5_next_sparse)
