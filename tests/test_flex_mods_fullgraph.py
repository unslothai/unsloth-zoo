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

"""Every score_mod / mask_mod family zoo hands to flex_attention traces under torch.compile(fullgraph = True).

torch 2.15 (pytorch/pytorch#185738) rejects flex_attention bodies that mutate their inputs or whose mask_mod returns
an alias of a captured tensor: fullgraph compiles raise, other compiles graph-break into eager dense attention (the
whole score matrix). Each family is built by its real builder and checked with backend = "aot_eager" (Dynamo's
HOP checks run there; no Inductor codegen, so the test is fast) against eager flex_attention."""

import types

import pytest
import torch

flex = pytest.importorskip("torch.nn.attention.flex_attention")
if not hasattr(flex, "BlockMask") or "seq_lengths" not in flex.BlockMask.__init__.__code__.co_varnames:
    pytest.skip("BlockMask without seq_lengths (torch < 2.6)", allow_module_level = True)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _utils():
    from unsloth_zoo.flex_attention import utils
    if not utils.HAS_FLEX_ATTENTION:
        pytest.skip("zoo flex_attention utils are CUDA only")
    return utils


def _qkv(batch, heads, q_len, kv_len, kv_heads = None, dim = 16):
    g = torch.Generator(device = "cpu").manual_seed(0)
    q = torch.randn(batch, heads, q_len, dim, generator = g).to(DEVICE)
    k = torch.randn(batch, kv_heads or heads, kv_len, dim, generator = g).to(DEVICE)
    v = torch.randn(batch, kv_heads or heads, kv_len, dim, generator = g).to(DEVICE)
    return q, k, v


def _check(block_mask = None, score_mod = None, *, batch = 2, heads = 4, kv_heads = None, q_len = 128, kv_len = 128):
    q, k, v = _qkv(batch, heads, q_len, kv_len, kv_heads)
    gqa = kv_heads is not None and kv_heads != heads

    def attend(q, k, v):
        return flex_attention(q, k, v, score_mod = score_mod, block_mask = block_mask, enable_gqa = gqa)

    torch._dynamo.reset()
    compiled = torch.compile(attend, fullgraph = True, backend = "aot_eager")(q, k, v)
    torch.testing.assert_close(compiled, attend(q, k, v), atol = 1e-5, rtol = 1e-5)


def _mask(mask_mod, batch, q_len, kv_len, heads = None):
    return create_block_mask(mask_mod, batch, heads, q_len, kv_len, device = DEVICE)


def _left_padding(batch = 2):
    return torch.tensor([0, 37][:batch], device = DEVICE)


# ---- unsloth_zoo.flex_attention.utils

def test_utils_causal_and_sliding():
    u = _utils()
    for mask_mod in (u.causal_mask, u.generate_sliding_window_mask(64)):
        _check(_mask(mask_mod, None, 128, 128))


def test_utils_left_padding_masks():
    u = _utils()
    pad = _left_padding()
    for mask_mod in (
        u.generate_causal_mask_with_padding(pad),
        u.generate_decoding_causal_mask_with_padding(pad),
        u.generate_sliding_window_mask_with_padding(64, pad),
    ):
        _check(_mask(mask_mod, 2, 128, 128))


def test_utils_offset_mods():
    u = _utils()
    from unsloth_zoo.flex_attention.decode import _softcap_score_mod
    offset = torch.tensor(100, device = DEVICE)
    mask_mod = u.get_mask_mod_w_offset(u.causal_mask, offset)
    score_mod = u.get_score_mod_w_offset(_softcap_score_mod(30.0), offset)
    _check(_mask(mask_mod, None, 1, 128), score_mod, q_len = 1)


# ---- unsloth_zoo.flex_attention.attention_sink (gpt-oss training)

def test_attention_sink_mods():
    _utils()
    from unsloth_zoo.flex_attention import attention_sink as s
    sinks = torch.randn(4, device = DEVICE)
    score_mod = s.generate_sink_score_mod(sinks)
    for mask_mod in (s.causal_mask_with_sink, s.generate_sliding_window_with_sink(64)):
        _check(_mask(mask_mod, None, 128, 129), score_mod, kv_len = 129)


# ---- unsloth_zoo.flex_attention.decode (one-token flex decode)

@pytest.mark.parametrize("window", [None, 64])
@pytest.mark.parametrize("padded", [False, True])
def test_decode_mask_mods(window, padded):
    from unsloth_zoo.flex_attention import decode as d
    q_o, kv_o = torch.tensor(150, device = DEVICE), torch.tensor(0, device = DEVICE)
    pad = None
    if padded:
        pad = torch.ones(2, 256, dtype = torch.bool, device = DEVICE)
        pad[1, :37] = False
    mask_mod = d._make_mask_mod(d.causal_window_mask_function(window), q_o, kv_o, pad)
    _check(_mask(mask_mod, 2, 1, 256), d._softcap_score_mod(30.0), q_len = 1, kv_len = 256, kv_heads = 2)


# ---- unsloth_zoo.temporary_patches.misc (transformers mask functions, multi-GPU copies)

def test_transformers_mask_function_wrappers():
    masking_utils = pytest.importorskip("transformers.masking_utils")
    from unsloth_zoo.temporary_patches.misc import patch_transformers_masks
    patch_transformers_masks()
    causal = getattr(masking_utils, "causal_mask_function", None)
    if causal is None:
        pytest.skip("transformers without masking_utils.causal_mask_function")
    mods = []
    if hasattr(masking_utils, "padding_mask_function"):
        padding = torch.ones(2, 128, dtype = torch.bool, device = DEVICE)
        padding[1, :37] = False
        mods.append(masking_utils.and_masks(causal, masking_utils.padding_mask_function(padding)))
    if hasattr(masking_utils, "packed_sequence_mask_function"):
        packed = torch.zeros(2, 128, dtype = torch.long, device = DEVICE)
        packed[:, 50:] = 1
        mods.append(masking_utils.and_masks(causal, masking_utils.packed_sequence_mask_function(packed)))
    if hasattr(masking_utils, "add_offsets_to_mask_function"):
        mods.append(masking_utils.add_offsets_to_mask_function(causal, torch.tensor(0, device = DEVICE), torch.tensor(0, device = DEVICE)))
    if not mods:
        pytest.skip("no wrapped transformers mask functions")
    for mask_mod in mods:
        _check(_mask(mask_mod, 2, 128, 128))


# ---- sparse attention families (DeepSeek-V3.2 / GLM DSA, DeepSeek-V4, GLM-5 Next, MiniMax-M3)

def test_sparse_attention_membership_mask():
    from unsloth_zoo import sparse_attention as sa
    topk = torch.randint(0, 256, (2, 128, 16), device = DEVICE)
    topk[0, :, -2:] = -1
    _check(sa._membership_block_mask(topk, 256), kv_len = 256)


def test_deepseek_v4_mask_and_bias_score_mod(monkeypatch):
    from unsloth_zoo.temporary_patches import deepseek_v4_sparse_flex as d
    mask = torch.zeros(2, 1, 128, 256, device = DEVICE)
    mask[:, :, :, 200:] = torch.finfo(mask.dtype).min
    mask[1, :, :, :10] = -0.5  # finite additive bias: score_mod path
    seen = {}

    class Captured(Exception):
        pass

    def capture(query, key, value, score_mod = None, block_mask = None, **kwargs):
        seen.update(block_mask = block_mask, score_mod = score_mod)
        raise Captured
    monkeypatch.setattr(d, "_compiled_flex", lambda: capture)
    module = types.SimpleNamespace(sinks = torch.zeros(4, device = DEVICE))
    q, k, _ = _qkv(2, 4, 128, 256, kv_heads = 1)
    with pytest.raises(Captured):
        d.deepseek_v4_flex_attention(module, q, k, mask, 0.25)
    assert seen["score_mod"] is not None
    _check(seen["block_mask"], seen["score_mod"], kv_len = 256)


def _capture_sparse_blocks(monkeypatch):
    from unsloth_zoo.flex_attention import sparse_blocks as sb
    seen = {}

    class Captured(Exception):
        pass

    def capture(query, key, value, block_mask, scale, score_mod = None):
        seen.update(block_mask = block_mask, score_mod = score_mod)
        raise Captured
    monkeypatch.setattr(sb, "_flex", capture)
    return sb, seen, Captured


def test_dsa_mask_and_bias_score_mod(monkeypatch):
    sb, seen, Captured = _capture_sparse_blocks(monkeypatch)
    q, k, v = _qkv(2, 4, 128, 256)
    topk = torch.randint(0, 200, (2, 128, 32), device = DEVICE)
    mask = torch.zeros(2, 1, 128, 256, device = DEVICE)
    mask[..., 200:] = torch.finfo(mask.dtype).min
    mask[1, ..., :10] = -0.5
    with pytest.raises(Captured):
        sb.dsa_flex_attention(q, k, v, topk, mask, 0.25)
    assert seen["score_mod"] is not None
    _check(seen["block_mask"], seen["score_mod"], kv_len = 256)


@pytest.mark.parametrize("with_mask", [False, True])
def test_minimax_m3_block_masks(monkeypatch, with_mask):
    sb, seen, Captured = _capture_sparse_blocks(monkeypatch)
    q, k, v = _qkv(2, 4, 128, 256)
    block_indices = torch.tensor([0, 1], device = DEVICE).view(1, 1, 1, 2).expand(2, 2, 128, 2).contiguous()
    position_ids = torch.arange(128, device = DEVICE).expand(2, -1) + 128
    mask = None
    if with_mask:
        mask = torch.ones(2, 1, 128, 256, dtype = torch.bool, device = DEVICE).tril(128)
    with pytest.raises(Captured):
        sb.minimax_m3_flex_attention(q, k, v, block_indices, mask, position_ids, 0.25, 128)
    _check(seen["block_mask"], seen["score_mod"], kv_len = 256)


@pytest.mark.parametrize("tail", [0, 8])
def test_glm5_next_block_mask(tail):
    from unsloth_zoo.temporary_patches import glm5_next_sparse_flex as g
    kpool, index_topk = 4, 16
    pools = (torch.arange(index_topk // kpool, device = DEVICE) * 2 * kpool).repeat_interleave(kpool)
    pools = pools + torch.arange(index_topk, device = DEVICE) % kpool
    topk = pools.expand(2, 128, -1)
    if tail:
        tail_ids = torch.arange(128, device = DEVICE).view(1, 128, 1) - torch.arange(tail, device = DEVICE).flip(0)
        topk = torch.cat([topk, tail_ids.clamp(min = -1).expand(2, -1, -1)], -1)
    _check(g.glm5_next_block_mask(topk.contiguous(), 256, kpool, index_topk, tail), kv_len = 256)


# ---- gpt-oss inference prefill mask

@pytest.mark.parametrize("padded", [False, True])
def test_gpt_oss_prefill_mask(padded):
    _utils()
    masking_utils = pytest.importorskip("transformers.masking_utils")
    from unsloth_zoo.temporary_patches import gpt_oss
    attention_mask = None
    if padded:
        attention_mask = torch.ones(2, 128, dtype = torch.long, device = DEVICE)
        attention_mask[1, :37] = 0
    block_mask = gpt_oss._gpt_oss_flex_mask(
        2, q_length = 128, kv_length = 128, mask_function = masking_utils.causal_mask_function,
        attention_mask = attention_mask, device = DEVICE,
    )
    _check(block_mask)
