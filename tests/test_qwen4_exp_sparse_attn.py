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

"""Above its token budget, Qwen4Exp QSA attention must keep exactly the reference indexer's tokens and give the
reference attention output and gradients, while selecting vectorised and attending to the selection only."""

import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")

from unsloth_zoo import sparse_attention as sa  # noqa: E402
from unsloth_zoo.temporary_patches import qwen4_exp as zoo_qwen4_exp  # noqa: E402

BUDGET, RATIO = 16, 4


def _module(attn = "eager", seed = 0):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    config = Qwen4ExpTextConfig(
        hidden_size = 64, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 32,
        indexer_budget = BUDGET, indexer_compress_ratio = RATIO, indexer_head_dim = 16, indexer_n_heads = 2,
        indexer_kv_heads = 1, partial_rotary_factor = 0.25, num_hidden_layers = 4, vocab_size = 128,
        layer_types = ["linear_attention"] * 3 + ["full_attention"],
    )
    config._attn_implementation = attn
    torch.manual_seed(seed)
    attention = modeling.Qwen4ExpTextAttention(config, layer_idx = 3).double()
    rotary = modeling.Qwen4ExpTextRotaryEmbedding(config)
    return config, attention, rotary


def _reference_forward():
    forward = modeling.Qwen4ExpTextAttention.forward
    return getattr(zoo_qwen4_exp, "_reference_attention_forward", None) if forward is zoo_qwen4_exp.qwen4_exp_attention_forward else forward


def _masks(kind, b, s):
    causal = torch.tril(torch.ones(s, s, dtype = torch.bool))
    if kind == "causal":
        m = causal.expand(b, s, s).clone()
    elif kind == "right_pad":
        m = causal.expand(b, s, s).clone()
        m[1, :, s - 9 :] = False  # keys of the padded tail are invisible to every query of row 1
    elif kind == "packed":
        doc = torch.tensor([0] * 23 + [1] * (s - 23))
        m = (causal & (doc[:, None] == doc[None, :])).expand(b, s, s).clone()
    elif kind == "sliding":
        idx = torch.arange(s)
        m = (causal & (idx[:, None] - idx[None, :] < 20)).expand(b, s, s).clone()
    return m.unsqueeze(1)


def _inputs(b, s, rotary, seed = 1):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(b, s, 64, generator = g, dtype = torch.float64)
    pos = torch.arange(s).unsqueeze(0).expand(b, -1)
    cos, sin = rotary(hidden.float(), pos)
    return hidden, (cos.double(), sin.double())


def _float(mask):
    return torch.zeros(mask.shape, dtype = torch.float64).masked_fill(~mask, torch.finfo(torch.float64).min)


def _selection_mask(selected, s):
    out = torch.zeros(*selected.shape[:2], s + 1, dtype = torch.bool)
    out.scatter_(-1, torch.where(selected >= 0, selected, s).long(), True)
    return out[..., :s]


def _zero_index_keys(indexer):
    # Every pooled block key 0 -> every block score exactly 0: each top-k boundary is a tie.
    with torch.no_grad():
        indexer.index_qk_proj.weight[indexer.index_n_heads * indexer.index_head_dim:] = 0
        if getattr(indexer.index_qk_proj, "bias", None) is not None:
            indexer.index_qk_proj.bias[indexer.index_n_heads * indexer.index_head_dim:] = 0


@pytest.mark.parametrize("ties", [False, True], ids = ["random", "all_tied"])
@pytest.mark.parametrize("kind", ["causal", "right_pad", "packed"])
def test_vectorised_selection_equals_reference_loop(kind, ties):
    # Token identity, not only scores: the same tokens as transformers' untouched per-query loop, ties included.
    config, attention, rotary = _module()
    if ties:
        _zero_index_keys(attention.indexer)
    b, s = 2, 61
    hidden, pe = _inputs(b, s, rotary)
    mask = _masks(kind, b, s)
    float_mask = _float(mask)
    with torch.no_grad():
        reference = modeling.Qwen4ExpTextQSAIndexer.forward
        reference = getattr(zoo_qwen4_exp, "_reference_qsa_forward", None) or reference
        ref = reference(attention.indexer, hidden, pe, mask, None)
        ref = ref if ref.dtype == torch.bool else ref == 0
        ours = sa.qsa_select_tokens(attention.indexer, hidden, pe, float_mask, modeling.apply_rotary_pos_emb)
    assert ours is not None
    ours_mask, ref_mask = _selection_mask(ours, s), ref[:, 0] & mask[:, 0]
    torch.testing.assert_close(ours_mask, ref_mask, rtol = 0, atol = 0)


def test_sliding_window_mask_is_not_taken():
    config, attention, rotary = _module()
    hidden, pe = _inputs(1, 61, rotary)
    mask = _masks("sliding", 1, 61)
    assert sa.qsa_select_tokens(attention.indexer, hidden, pe, mask, modeling.apply_rotary_pos_emb) is None
    assert zoo_qwen4_exp._qwen4_exp_sparse_attention(attention, hidden, pe, mask, None, {}) is None


@pytest.mark.parametrize("kind", ["causal", "right_pad", "packed"])
@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_sparse_attention_matches_reference_forward_and_backward(monkeypatch, kind, attn):
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "torch")
    config, attention, rotary = _module(attn)
    b, s = 2, 61
    hidden, pe = _inputs(b, s, rotary)
    mask = _masks(kind, b, s)
    if attn == "eager":
        mask = _float(mask)
    params = [p for p in attention.parameters() if p.requires_grad]
    go = torch.randn(b, s, 64, dtype = torch.float64)

    def run(fn):
        x = hidden.clone().requires_grad_()
        out = fn(x)
        grads = torch.autograd.grad(out, [x] + params, go, allow_unused = True)
        return out.detach(), grads

    ours_out, ours_g = run(lambda x: zoo_qwen4_exp._qwen4_exp_sparse_attention(attention, x, pe, mask, None, {})[0])
    # The untouched reference: transformers' attention with its own indexer selection.
    ref_out, ref_g = run(lambda x: _reference_forward()(attention, x, pe, mask, None)[0])
    # Rows that see no key at all (padded queries) are undefined in the reference's SDPA; compare real rows.
    rows = _masks(kind, b, s)[:, 0].any(-1) if kind != "right_pad" else torch.ones(b, s, dtype = torch.bool)
    # transformers' eager softmax runs in fp32 even for fp64 inputs (~1e-8); SDPA stays fp64.
    tol = dict(rtol = 1e-5, atol = 2e-6) if attn == "eager" else dict(rtol = 1e-10, atol = 1e-10)
    torch.testing.assert_close(ours_out[rows], ref_out[rows], **tol)
    for name, x, y in zip(["hidden"] + [n for n, p in attention.named_parameters() if p.requires_grad], ours_g, ref_g):
        if x is None and y is None:
            continue
        x = torch.zeros_like(y) if x is None else x
        y = torch.zeros_like(x) if y is None else y
        torch.testing.assert_close(x, y, **(tol if attn == "eager" else dict(rtol = 1e-9, atol = 1e-10)), msg = name)


def test_empty_cache_prefill_is_sparse_and_fills_the_cache_like_the_reference(monkeypatch):
    from transformers import DynamicCache
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "torch")
    config, attention, rotary = _module("sdpa")
    hidden, pe = _inputs(1, 40, rotary)
    mask = _masks("causal", 1, 40)
    ours_cache, ref_cache = DynamicCache(config = config), DynamicCache(config = config)
    with torch.no_grad():
        ours = zoo_qwen4_exp._qwen4_exp_sparse_attention(attention, hidden, pe, mask, ours_cache, {})
        selected = _selection_mask(sa.qsa_select_tokens(attention.indexer, hidden, pe, mask, modeling.apply_rotary_pos_emb), 40)
        monkeypatch.setattr(attention.indexer, "forward", lambda h, p, m, c: (c.update_indexer(
            attention.indexer.index_qk_proj(h)[..., -attention.indexer.index_head_dim:], 3), selected.unsqueeze(1))[1])
        ref = _reference_forward()(attention, hidden, pe, mask, ref_cache)
    assert ours is not None
    torch.testing.assert_close(ours[0], ref[0], rtol = 1e-10, atol = 1e-10)
    a, b = ours_cache.layers[3], ref_cache.layers[3]
    for name in ("keys", "values", "indexer_keys"):
        torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol = 0, atol = 0, msg = name)
    # A cache that already holds tokens (decode / chunked prefill) keeps the reference.
    assert zoo_qwen4_exp._qwen4_exp_sparse_attention(attention, hidden, pe, mask, ours_cache, {}) is None


def test_below_budget_and_kill_switch_keep_the_reference(monkeypatch):
    config, attention, rotary = _module()
    hidden, pe = _inputs(1, BUDGET + RATIO - 1, rotary)
    mask = _masks("causal", 1, BUDGET + RATIO - 1)
    assert zoo_qwen4_exp._qwen4_exp_sparse_attention(attention, hidden, pe, mask, None, {}) is None
    hidden, pe = _inputs(1, 40, rotary)
    mask = _masks("causal", 1, 40)
    monkeypatch.setenv("UNSLOTH_QWEN4_EXP_SPARSE_ATTN", "0")
    assert zoo_qwen4_exp._qwen4_exp_sparse_attention(attention, hidden, pe, mask, None, {}) is None


def test_patch_installs_and_counts(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "torch")
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE", "gpu")
    zoo_qwen4_exp.patch_qwen4_exp()
    assert modeling.Qwen4ExpTextAttention.forward is zoo_qwen4_exp.qwen4_exp_attention_forward
    config, attention, rotary = _module()
    hidden, pe = _inputs(1, 40, rotary)
    mask = _masks("causal", 1, 40)
    before = dict(zoo_qwen4_exp.QWEN4_EXP_SPARSE_STATS)
    attention(hidden, pe, _float(mask))
    assert zoo_qwen4_exp.QWEN4_EXP_SPARSE_STATS["sparse"] == before["sparse"] + 1


def test_mask_geometry_is_scanned_once_per_mask_version(monkeypatch):
    # The interval scan and the block-start lists sync with the host: once per forward, not once per QSA layer.
    config, attention, rotary = _module()
    hidden, pe = _inputs(1, 40, rotary)
    mask = _masks("causal", 1, 40)
    calls = []
    original = sa._contiguous_rows
    monkeypatch.setattr(sa, "_contiguous_rows", lambda v: calls.append(1) or original(v))
    with torch.no_grad():
        first = sa.qsa_select_tokens(attention.indexer, hidden, pe, mask, modeling.apply_rotary_pos_emb)
        second = sa.qsa_select_tokens(attention.indexer, hidden, pe, mask, modeling.apply_rotary_pos_emb)
        assert len(calls) == 1 and torch.equal(first, second)
        mask[0, 0, 39, 0] = True  # an in-place edit bumps the version: rescanned
        sa.qsa_select_tokens(attention.indexer, hidden, pe, mask, modeling.apply_rotary_pos_emb)
    assert len(calls) == 2
