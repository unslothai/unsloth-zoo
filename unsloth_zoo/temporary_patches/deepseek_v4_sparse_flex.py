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

"""DeepSeek-V4 attention on flex attention instead of eager (transformers runs V4 eager only).

The additive mask (sliding window + padding, then the compressor's block_bias) becomes a block-sparse
BlockMask; the sink is applied exactly from the logsumexp:
softmax([l, s]) @ v == (softmax(l) @ v) * sigmoid(lse(l) - s). Prefill / training only, from a measured
per-GPU key length (_MIN_KEYS). UNSLOTH_SPARSE_ATTN_FLEX: 0 = eager, 1 = also other sm80+ parts with
>= 160 KB shared memory, unset = measured generations only.
"""

__all__ = ["patch_deepseek_v4_sparse_flex", "DEEPSEEK_V4_FLEX_STATS"]

import importlib.util
import os

import torch

from .common import TEMPORARY_PATCHES, UNSLOTH_COMPILE_DISABLE, UNSLOTH_ENABLE_LOGGING
from .utils import logger

DEEPSEEK_V4_FLEX_STATS = {"flex": 0, "eager": 0}

_BLOCK = 128
# Decode (one query) never benefits; chunked prefill of a few queries over a long cache does.
_MIN_Q_LEN = 16
# head_dim 512: 32 x 32 tiles, 8 warps, 1 stage (B200 S 4096: 35 ms vs 205 ms with 4 warps, eager 55 ms);
# 64-wide tiles fault (misaligned address) at this head dim.
_KERNEL_OPTIONS = {
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "BLOCK_M1": 32,
    "BLOCK_N1": 32,
    "BLOCK_M2": 32,
    "BLOCK_N2": 32,
    "num_warps": 8,
    "num_stages": 1,
    # torch 2.10 sends queries under 128 to the decoding kernel, which asserts S * groups <= BLOCK_M (32).
    "FORCE_USE_FLEX_ATTENTION": True,
}
# Stride-0 expansion of the single KV head: grouped-query packing (64:1) took 180-210 ms.
_CSA = "compressed_sparse_attention"
# Minimum key length (sliding + compressed) from one V4-Flash layer fwd + bwd, eager vs flex:
# H100 / B200 4096 for every layer type; A100 4096 for HCA / sliding, 8192 for CSA (8-14% slower
# below). ~101 KB shared-memory cards, T4 and ROCm stay eager. UNSLOTH_SPARSE_ATTN_FLEX_MIN_KEYS=N
# replaces every threshold.
_MIN_KEYS = {
    9: {None: 4096},
    10: {None: 4096},
    8: {None: 4096, _CSA: 8192},
}
_FORCED_MIN_KEYS = 4096
_MIN_SHARED_MEMORY = 160 * 1024

_FLEX = None
_DEVICE_OK = {}
# Looked up once: find_spec is skipped by dynamo, and the first call can come from inside a compiled graph.
_HAS_TRITON = importlib.util.find_spec("triton") is not None


def _transformers_tested():
    """Checked on transformers 5.17-5.20; the V4 attention / compressor contract is newer than 5.x's first
    DeepSeek-V4 release."""
    try:
        import transformers
        from packaging.version import Version
        return Version(transformers.__version__).release >= (5, 17)
    except Exception:
        return False


def _torch_tested():
    """Measured on torch 2.10-2.14; older releases lack `BlockMask.from_kv_blocks(seq_lengths=)`."""
    try:
        import inspect
        from torch.nn.attention.flex_attention import BlockMask
        release = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        return release >= (2, 10) and "seq_lengths" in inspect.signature(BlockMask.from_kv_blocks).parameters
    except Exception:
        return False


def _flex_setting():
    return os.environ.get("UNSLOTH_SPARSE_ATTN_FLEX", "").strip()


def _count(key):
    if not torch.compiler.is_compiling():
        DEEPSEEK_V4_FLEX_STATS[key] += 1


def _compiled_flex():
    # Built at install: set lazily inside a trace, the `_FLEX is None` guard recompiled the next call.
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention
        # dynamic=None: a fixed length (packed / padded runs) keeps a static kernel, 20-30% faster than the
        # dynamic one (CSA 8192: 129 vs 188 ms on B200); varying lengths settle on dynamic after a few graphs.
        _FLEX = torch.compile(flex_attention)
    return _FLEX


def _device_thresholds(device, forced):
    """{layer_type or None: minimum key length} for this device, or None when flex is off there."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, forced)
    if key not in _DEVICE_OK:
        thresholds = None
        try:
            if torch.version.hip is None and _HAS_TRITON:
                props = torch.cuda.get_device_properties(index)
                smem = getattr(props, "shared_memory_per_block_optin", 0)
                # sm86 / sm89 report major 8 but have ~101 KB: the shared memory check keeps them out.
                if smem >= _MIN_SHARED_MEMORY:
                    thresholds = _MIN_KEYS.get(props.major)
                    if thresholds is None and forced and props.major >= 8:
                        thresholds = {None: _FORCED_MIN_KEYS}
        except Exception:
            thresholds = None
        # Cached from eager calls only: filled inside a trace, the dict guard recompiled the next call.
        if torch.compiler.is_compiling():
            return thresholds
        _DEVICE_OK[key] = thresholds
    return _DEVICE_OK[key]


def _min_keys(module, device, forced):
    thresholds = _device_thresholds(device, forced)
    if thresholds is None:
        return None
    override = os.environ.get("UNSLOTH_SPARSE_ATTN_FLEX_MIN_KEYS", "").strip()
    if override.isdigit():
        return int(override)
    return thresholds.get(getattr(module, "layer_type", None), thresholds[None])


def _flex_applies(module, query, key, value, attention_mask, dropout, kwargs):
    setting = _flex_setting()
    # Uncompiled flex attention materialises every score, which is worse than eager.
    # transformers records attention weights when config.output_attentions is set without passing the kwarg.
    output_attentions = kwargs.get("output_attentions")
    if output_attentions is None:
        output_attentions = getattr(getattr(module, "config", None), "output_attentions", False)
    if setting == "0" or UNSLOTH_COMPILE_DISABLE or output_attentions:
        return False
    if key is not value or dropout:
        return False
    # Flex under no_grad inside an enclosing compiled graph fails to lower (torch 2.14); eager is correct there.
    if torch.compiler.is_compiling() and not torch.is_grad_enabled():
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
    min_keys = _min_keys(module, query.device, setting == "1")
    return min_keys is not None and key.shape[2] >= min_keys


def _pack(selected, num_blocks):
    """Ascending block indices of `selected` [B, nq, nk] -> (count [B,1,nq], indices [B,1,nq,nk])."""
    count = selected.sum(dim = -1, dtype = torch.int32)
    order = torch.arange(num_blocks, device = selected.device)
    indices = torch.where(selected, order, num_blocks).argsort(dim = -1, stable = True).to(torch.int32)
    return count.unsqueeze(1), indices.unsqueeze(1)


def deepseek_v4_block_mask(attention_mask, batch):
    """(BlockMask, bias or None) over the additive [B|1, 1, S, KV] mask. Entries at or below half the dtype minimum
    (and -inf) are hard-masked; every other value is added to the score like eager does, so finite custom
    biases are kept. Whole 128 x 128 tiles with nothing visible are skipped."""
    from torch.nn.attention.flex_attention import BlockMask
    from unsloth_zoo.flex_attention.sparse import additive_bias_present

    q_len, kv_len = attention_mask.shape[-2], attention_mask.shape[-1]
    nq, nk = -(-q_len // _BLOCK), -(-kv_len // _BLOCK)
    # Padded to whole tiles: mask_mod / score_mod may be evaluated on any index of a tile.
    visible = attention_mask[:, 0] > torch.finfo(attention_mask.dtype).min / 2
    table = attention_mask.new_zeros((attention_mask.shape[0], nq * _BLOCK, nk * _BLOCK), dtype = torch.bool)
    table[:, :q_len, :kv_len] = visible
    # transformers' own masks are 0 / min only; reading a bias per score costs 5-15%, so only when present.
    # Inside a compiled graph the check would be a data-dependent branch, so the bias is always kept there.
    bias = None
    if torch.compiler.is_compiling() or additive_bias_present(attention_mask[:, 0], visible, "deepseek_v4"):
        bias = attention_mask.new_zeros((attention_mask.shape[0], nq * _BLOCK, nk * _BLOCK))
        bias[:, :q_len, :kv_len] = torch.where(visible, attention_mask[:, 0], 0)
        if bias.shape[0] != batch:
            bias = bias.expand(batch, -1, -1)
    if table.shape[0] != batch:
        table = table.expand(batch, -1, -1)
    tiles = table.view(batch, nq, _BLOCK, nk, _BLOCK)
    any_visible = tiles.any(dim = 4).any(dim = 2)
    all_visible = tiles.all(dim = 4).all(dim = 2)
    partial_count, partial_index = _pack(any_visible & ~all_visible, nk)
    full_count, full_index = _pack(all_visible, nk)

    def mask_mod(b, h, q_idx, kv_idx):
        return table[b, q_idx, kv_idx]

    block_mask = BlockMask.from_kv_blocks(
        partial_count,
        partial_index,
        full_count,
        full_index,
        BLOCK_SIZE = _BLOCK,
        mask_mod = mask_mod,
        seq_lengths = (q_len, kv_len),
    )
    return block_mask, bias


def deepseek_v4_flex_attention(module, query, key, attention_mask, scaling):
    """Flex attention + exact sink over K == V. query [B, H, S, D], key [B, 1, KV, D]."""
    from unsloth_zoo.flex_attention.utils import FLEX_LSE_KWARGS, flex_output_and_lse

    bsz, n_heads, _, head_dim = query.shape
    block_mask, bias = deepseek_v4_block_mask(attention_mask, bsz)
    kv = key.expand(bsz, n_heads, key.shape[2], head_dim)

    score_mod = None
    if bias is not None:
        def score_mod(score, b, h, q_idx, kv_idx):
            return score + bias[b, q_idx, kv_idx]

    out, lse = flex_output_and_lse(_compiled_flex()(
        query,
        kv,
        kv,
        score_mod = score_mod,
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
    if not (_transformers_tested() and _torch_tested()) or importlib.util.find_spec("transformers.models.deepseek_v4") is None:
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
    _compiled_flex()
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: DeepSeek-V4 prefill / training attention runs on flex attention.")
pass

TEMPORARY_PATCHES.append(patch_deepseek_v4_sparse_flex)
