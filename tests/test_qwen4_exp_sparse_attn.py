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
    # Text positions on all three mrope grids: older transformers' rotary indexes (3, batch, seq) and newer ones
    # expand to it, so this is the one shape every supported version accepts.
    pos = torch.arange(s).unsqueeze(0).expand(3, b, -1)
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


@pytest.mark.parametrize("exact", [True, False], ids = ["exact_ties", "default"])
@pytest.mark.parametrize("ties", [False, True], ids = ["random", "all_tied"])
@pytest.mark.parametrize("kind", ["causal", "right_pad", "packed"])
def test_vectorised_selection_equals_reference_loop(monkeypatch, kind, ties, exact):
    # UNSLOTH_QWEN4_EXP_EXACT_TIES=1: token identity with transformers' untouched per-query loop, ties included.
    # Default (off): at a k-th / (k+1)-th score tie another, equally-scored block may be kept: same score multiset.
    monkeypatch.setenv("UNSLOTH_QWEN4_EXP_EXACT_TIES", "1" if exact else "0")
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
    if exact:
        torch.testing.assert_close(ours_mask, ref_mask, rtol = 0, atol = 0)
        return
    for b_i, t in (ours_mask != ref_mask).any(-1).nonzero().tolist():
        scores = _block_scores(attention.indexer, hidden, pe, mask[b_i, 0, t], b_i, t)

        def picked(m):
            return sorted(round(v, 9) for start, v in scores.items() if bool(m[b_i, t, start]))
        assert picked(ours_mask) == picked(ref_mask), (b_i, t)
        # Tails (the incomplete last block) are always identical.
        assert ours_mask[b_i, t].sum() == ref_mask[b_i, t].sum()


def _block_scores(indexer, hidden, pe, visible_row, b_i, t):
    """Reference math for one query: {block start: score}."""
    import math
    full_cos, full_sin = pe
    qk = indexer.index_qk_proj(hidden)
    d = indexer.index_head_dim
    q, k = torch.split(qk, [indexer.index_n_heads * d, d], dim = -1)
    q = indexer.q_layernorm(q.reshape(*hidden.shape[:2], -1, d))
    q = modeling.apply_rotary_pos_emb(q, cos = full_cos, sin = full_sin, unsqueeze_dim = 2)
    vis = visible_row.nonzero().flatten()
    nb = vis.numel() // RATIO
    blocks = vis[: nb * RATIO].view(nb, RATIO)
    pooled = indexer.k_layernorm(k[b_i].index_select(0, blocks.flatten()).view(nb, RATIO, d).float().mean(1).to(k.dtype))
    keys = modeling.apply_rotary_pos_emb(pooled.unsqueeze(1), cos = full_cos[b_i, blocks[:, 0]], sin = full_sin[b_i, blocks[:, 0]]).squeeze(1)
    sc = torch.relu(torch.matmul(q[b_i, t].float(), keys.float().T).T).sum(-1) / math.sqrt(d)
    return {int(st): float(v) for st, v in zip(blocks[:, 0], sc)}



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
    monkeypatch.setenv("UNSLOTH_QWEN4_EXP_EXACT_TIES", "1")   # the untouched reference's own selection, ties included
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


@pytest.mark.parametrize("kind", ["causal", "packed"])
def test_compiled_attention_keeps_the_eager_dispatch(monkeypatch, kind):
    # Under torch.compile attention leaves the graph with the selection: eager dispatch, same break count, same result.
    import torch._dynamo
    from torch._dynamo.utils import counters
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "torch")
    config, attention, rotary = _module("sdpa")
    b, s = 2, 41
    hidden, pe = _inputs(b, s, rotary)
    mask = _masks(kind, b, s)
    params = [p for p in attention.parameters() if p.requires_grad]
    go = torch.randn(b, s, 64, dtype = torch.float64)

    def fn(x):
        return zoo_qwen4_exp._qwen4_exp_sparse_attention(attention, x, pe, mask, None, {})[0]

    def run(f):
        x = hidden.clone().requires_grad_()
        out = f(x)
        return out.detach(), torch.autograd.grad(out, [x] + params, go, allow_unused = True)

    ref_out, ref_g = run(fn)
    torch._dynamo.reset()
    counters.clear()
    before = sa.SPARSE_ATTN_STATS["torch"]
    out, grads = run(torch.compile(fn, backend = "aot_eager"))
    assert sa.SPARSE_ATTN_STATS["torch"] == before + 1, "the attention was traced, not dispatched eagerly"
    # One break site, as before the change (there: the selection; here: selection + attention).
    assert len(counters["graph_break"]) == 1, dict(counters["graph_break"])
    torch.testing.assert_close(out, ref_out, rtol = 1e-10, atol = 1e-10)
    for x, y in zip(grads, ref_g):
        if x is None or y is None:
            assert x is None and y is None
            continue
        torch.testing.assert_close(x, y, rtol = 1e-9, atol = 1e-10)
    torch._dynamo.reset()


def test_static_cache_decode_compiles_fullgraph(monkeypatch):
    # Decode must be rejected before the cache probe (StaticCache returns a tensor: graph break + host sync).
    from transformers import StaticCache
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "torch")
    config, attention, rotary = _module("sdpa")
    first, first_pe = _inputs(1, 1, rotary, seed = 2)
    hidden, pe = _inputs(1, 1, rotary)
    mask = torch.zeros((1, 1, 1, 16), dtype = torch.bool)
    mask[..., :2] = True

    def warm():
        cache = StaticCache(config, max_cache_len = 16)
        with torch.no_grad():
            _reference_forward()(attention, first, first_pe, mask, cache)
        return cache

    with torch.no_grad():
        want = _reference_forward()(attention, hidden, pe, mask, warm())[0]
        cache = warm()
        torch._dynamo.reset()
        got = torch.compile(lambda x: zoo_qwen4_exp.qwen4_exp_attention_forward(attention, x, pe, mask, cache)[0],
                            backend = "eager", fullgraph = True)(hidden)
    # fp64 rounding only: transformers 5.23's compiled SDPA dispatch differs from eager in the last bit.
    torch.testing.assert_close(got, want, rtol = 1e-12, atol = 1e-14)


def _text_model_run(model, ids, use_cache):
    # Prefill, then (with a cache) two decode steps through the model's own cache.
    with torch.no_grad():
        if not use_cache:
            return model(input_ids = ids, use_cache = False).last_hidden_state
        out = model(input_ids = ids[:, :-2], use_cache = True)
        hidden, past = [out.last_hidden_state], out.past_key_values
        for t in (-2, -1):
            out = model(input_ids = ids[:, t:][:, :1], past_key_values = past, use_cache = True)
            hidden.append(out.last_hidden_state)
            past = out.past_key_values
        return torch.cat(hidden, dim = 1)


@pytest.fixture
def _few_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(min(threads, 4))
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("use_cache", [False, True], ids = ["no_cache", "prefill_then_decode"])
def test_text_model_forward_matches_the_reference_end_to_end(monkeypatch, use_cache, _few_threads):
    # Through the whole text model (own mRoPE ids and cache), patched equals untouched and engages on prefill.
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "torch")
    monkeypatch.setenv("UNSLOTH_QWEN4_EXP_EXACT_TIES", "1")
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE", "gpu")
    zoo_qwen4_exp.patch_qwen4_exp()
    # CPU run: use torch linear-attention / MoE paths, since the GPU host's Triton kernels reject CPU tensors.
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                 "causal_conv1d_update"):
        original = getattr(getattr(modeling, name, None), "__wrapped__", None)
        if original is not None:
            monkeypatch.setattr(modeling, name, original)
    config, _, _ = _module("sdpa")
    torch.manual_seed(0)
    model = modeling.Qwen4ExpTextModel(config).double().eval()
    for layer in model.layers:
        layer.mlp = layer.mlp.shared_expert
    ids = torch.randint(0, config.vocab_size, (2, 45), generator = torch.Generator().manual_seed(3))
    before = dict(zoo_qwen4_exp.QWEN4_EXP_SPARSE_STATS)
    got = _text_model_run(model, ids, use_cache)
    assert zoo_qwen4_exp.QWEN4_EXP_SPARSE_STATS["sparse"] > before["sparse"]
    monkeypatch.setattr(modeling.Qwen4ExpTextAttention, "forward", _reference_forward())
    want = _text_model_run(model, ids, use_cache)
    torch.testing.assert_close(got, want, rtol = 1e-10, atol = 1e-12)
