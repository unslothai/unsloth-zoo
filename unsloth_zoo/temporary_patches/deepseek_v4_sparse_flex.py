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

"""DeepSeek-V4 attention on flex attention instead of eager.

transformers runs V4 eager only: one softmax over [sliding-window tokens | compressed entries] plus a
learned per-head sink, so every layer materialises [B, 64, S, S + C] logits. The additive mask the
attention receives (sliding window + padding, then the compressor's block_bias of causal / indexer
selection over the compressed entries) is turned into a block-sparse BlockMask, and the sink is applied
exactly afterwards from the logsumexp: softmax([l, s]) @ v == (softmax(l) @ v) * sigmoid(lse(l) - s).
Prefill and training only (q_len >= 2048); decode, short prompts and anything else stay eager.
UNSLOTH_SPARSE_ATTN_FLEX (read per call): 0 = always eager, 1 = flex on any sm80+ part with enough shared
memory, unset = flex only on the GPU generations the tiles were measured on.
"""

__all__ = ["patch_deepseek_v4_sparse_flex", "DEEPSEEK_V4_FLEX_STATS"]

import importlib.util
import os

import torch

from .common import TEMPORARY_PATCHES, UNSLOTH_COMPILE_DISABLE, UNSLOTH_ENABLE_LOGGING
from .utils import logger

DEEPSEEK_V4_FLEX_STATS = {"flex": 0, "eager": 0}

_BLOCK = 128
# Below about 2048 query tokens eager is as fast (one V4-Flash attention layer, fwd + bwd, B200:
# 1024 tokens eager 7-11 ms vs flex 9-15 ms; 2048 tokens 19-25 vs 16-26 ms; 4096 tokens 60-76 vs
# 28-63 ms with 2.5x less memory), and decode (one query) never benefits.
_MIN_Q_LEN = 2048
# head_dim 512 needs small tiles. Measured fwd + bwd at the V4-Flash attention shape (64 heads, S 4096,
# B200, bf16): 32 x 32 tiles with 8 warps and 1 stage take 35 ms, the same tiles with 4 warps 205 ms,
# eager 55 ms. 64-row forward tiles and 64-wide backward tiles fault (misaligned address) at this head
# dim, so they are not used.
_KERNEL_OPTIONS = {
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "BLOCK_M1": 32,
    "BLOCK_N1": 32,
    "BLOCK_M2": 32,
    "BLOCK_N2": 32,
    "num_warps": 8,
    "num_stages": 1,
}
# The single KV head is expanded (stride 0, no copy) to one per query head: with grouped-query packing
# (64 query heads per KV head) the same call took 180-210 ms. Autograd sums the expanded gradient back.
# Capabilities the tiles above were run on; UNSLOTH_SPARSE_ATTN_FLEX=1 forces flex on any sm80+ part
# with enough shared memory.
_MEASURED_CAPABILITIES = ((10, 0),)
_MIN_SHARED_MEMORY = 160 * 1024

_FLEX = None
_DEVICE_OK = {}


def _flex_setting():
    return os.environ.get("UNSLOTH_SPARSE_ATTN_FLEX", "").strip()


def _count(key):
    if not torch.compiler.is_compiling():
        DEEPSEEK_V4_FLEX_STATS[key] += 1


def _compiled_flex():
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention
        # dynamic=None: the first length compiles static, the next one recompiles with dynamic shapes, so
        # varying training lengths do not recompile per step.
        _FLEX = torch.compile(flex_attention)
    return _FLEX


def _device_ok(device, forced):
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, forced)
    ok = _DEVICE_OK.get(key)
    if ok is None:
        ok = False
        try:
            if torch.version.hip is None and importlib.util.find_spec("triton") is not None:
                props = torch.cuda.get_device_properties(index)
                capability = (props.major, props.minor)
                smem = getattr(props, "shared_memory_per_block_optin", 0)
                if forced:
                    ok = capability >= (8, 0) and smem >= _MIN_SHARED_MEMORY
                else:
                    ok = capability in _MEASURED_CAPABILITIES
        except Exception:
            ok = False
        _DEVICE_OK[key] = ok
    return ok


def _flex_applies(module, query, key, value, attention_mask, dropout, kwargs):
    setting = _flex_setting()
    # Uncompiled flex attention materialises every score, which is worse than eager.
    if setting == "0" or UNSLOTH_COMPILE_DISABLE or kwargs.get("output_attentions"):
        return False
    if key is not value or dropout:
        return False
    if not (query.is_cuda and query.dtype in (torch.bfloat16, torch.float16)):
        return False
    if query.dim() != 4 or key.dim() != 4 or key.shape[1] != 1 or query.shape[2] < _MIN_Q_LEN:
        return False
    if getattr(module, "sinks", None) is None:
        return False
    if not isinstance(attention_mask, torch.Tensor) or not attention_mask.is_floating_point():
        return False
    bsz, _, q_len, _ = query.shape
    if attention_mask.dim() != 4 or attention_mask.shape[1] != 1:
        return False
    if attention_mask.shape[-2] != q_len or attention_mask.shape[-1] != key.shape[2]:
        return False
    if attention_mask.shape[0] not in (1, bsz):
        return False
    return _device_ok(query.device, setting == "1")


def _pack(selected, num_blocks):
    """Ascending block indices of `selected` [B, nq, nk] -> (count [B,1,nq], indices [B,1,nq,nk])."""
    count = selected.sum(dim = -1, dtype = torch.int32)
    order = torch.arange(num_blocks, device = selected.device)
    indices = torch.where(selected, order, num_blocks).argsort(dim = -1, stable = True).to(torch.int32)
    return count.unsqueeze(1), indices.unsqueeze(1)


def deepseek_v4_block_mask(attention_mask, batch):
    """BlockMask over the additive [B|1, 1, S, KV] mask (0 = attend). Keys a row may not see stay
    masked at token level; whole 128 x 128 tiles with nothing visible are skipped."""
    from torch.nn.attention.flex_attention import BlockMask

    q_len, kv_len = attention_mask.shape[-2], attention_mask.shape[-1]
    nq, nk = -(-q_len // _BLOCK), -(-kv_len // _BLOCK)
    # Lookup padded to whole tiles with False: mask_mod may be evaluated on any index of a tile.
    table = attention_mask.new_zeros((attention_mask.shape[0], nq * _BLOCK, nk * _BLOCK), dtype = torch.bool)
    table[:, :q_len, :kv_len] = attention_mask[:, 0] == 0
    if table.shape[0] != batch:
        table = table.expand(batch, -1, -1)
    tiles = table.view(batch, nq, _BLOCK, nk, _BLOCK)
    any_visible = tiles.any(dim = 4).any(dim = 2)
    all_visible = tiles.all(dim = 4).all(dim = 2)
    partial_count, partial_index = _pack(any_visible & ~all_visible, nk)
    full_count, full_index = _pack(all_visible, nk)

    def mask_mod(b, h, q_idx, kv_idx):
        return table[b, q_idx, kv_idx]

    return BlockMask.from_kv_blocks(
        partial_count,
        partial_index,
        full_count,
        full_index,
        BLOCK_SIZE = _BLOCK,
        mask_mod = mask_mod,
        seq_lengths = (q_len, kv_len),
    )


def deepseek_v4_flex_attention(module, query, key, attention_mask, scaling):
    """Flex attention + exact sink over K == V. query [B, H, S, D], key [B, 1, KV, D]."""
    from unsloth_zoo.flex_attention.utils import FLEX_LSE_KWARGS, flex_output_and_lse

    bsz, n_heads, _, head_dim = query.shape
    block_mask = deepseek_v4_block_mask(attention_mask, bsz)
    kv = key.expand(bsz, n_heads, key.shape[2], head_dim)
    out, lse = flex_output_and_lse(_compiled_flex()(
        query,
        kv,
        kv,
        block_mask = block_mask,
        scale = scaling,
        kernel_options = _KERNEL_OPTIONS,
        **FLEX_LSE_KWARGS,
    ))
    sink_scale = torch.sigmoid(lse.float() - module.sinks.float().view(1, -1, 1))
    out = (out.float() * sink_scale.unsqueeze(-1)).to(query.dtype)
    return out.transpose(1, 2).contiguous()


def _make_attention_forward(original):
    def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout = 0.0, **kwargs):
        if _flex_applies(module, query, key, value, attention_mask, dropout, kwargs):
            _count("flex")
            return deepseek_v4_flex_attention(module, query, key, attention_mask, scaling), None
        _count("eager")
        return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

    eager_attention_forward._unsloth_deepseek_v4_flex = True
    eager_attention_forward._unsloth_original = original
    return eager_attention_forward


def patch_deepseek_v4_sparse_flex():
    if importlib.util.find_spec("transformers.models.deepseek_v4") is None:
        return
    try:
        from transformers.models.deepseek_v4 import modeling_deepseek_v4 as modeling
        from torch.nn.attention.flex_attention import BlockMask  # noqa: F401
    except Exception:
        return
    original = getattr(modeling, "eager_attention_forward", None)
    if original is None or getattr(original, "_unsloth_deepseek_v4_flex", False):
        return
    # DeepseekV4Attention resolves the module global per call, so the dispatcher sees every call that
    # transformers would have sent to eager.
    modeling.eager_attention_forward = _make_attention_forward(original)
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: DeepSeek-V4 prefill / training attention runs on flex attention.")
pass

TEMPORARY_PATCHES.append(patch_deepseek_v4_sparse_flex)
