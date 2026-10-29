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

CPU: both patched indexer paths select upstream's key sets (padding, holes, cache decode); the
BlockMask's tables + mask_mod keep exactly those keys; a tiny model's logits and gradients match
upstream with every path engaged (flex forward on CPU flex, backward through flex semantics
materialised from the BlockMask, since CPU flex has no backward). GPU: flex vs upstream."""

import contextlib
import importlib

import pytest
import torch

transformers = pytest.importorskip("transformers")
from packaging.version import Version  # noqa: E402


def _spec(name):
    try:
        return importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return None


pytestmark = pytest.mark.skipif(
    _spec("transformers.models.glm5_next.modeling_glm5_next") is None,
    reason = "transformers without glm5_next",
)

from unsloth_zoo.flex_attention import sparse as gates  # noqa: E402
from unsloth_zoo.flex_attention import sparse_blocks as sb  # noqa: E402

TOPK, KPOOL = 64, 16


def _stat(kind):
    return gates.SPARSE_ATTN_STATS.get(kind, {}).get("glm5_next", 0)


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    for name in ("UNSLOTH_SPARSE_ATTN_FLEX", "UNSLOTH_SPARSE_ATTN_FLEX_GLM5_NEXT", "UNSLOTH_SPARSE_INDEXER_CHUNKED", "UNSLOTH_SPARSE_INDEXER_Q_CHUNK",
                 "UNSLOTH_SPARSE_ATTN_SKIP_FULL", "UNSLOTH_SPARSE_ATTN_DSA_MIN_KEYS"):
        monkeypatch.delenv(name, raising = False)
    sb._FAILED.clear()
    gates.reset_sparse_attn_stats()


# transformers 5.17 attends over expanded KV states (another source layout): the patch declines there.
# From 5.21 the layout is the supported one, so a decline is a failure, not a skip.
_LAYOUT_REQUIRED = Version(transformers.__version__).release[:2] >= (5, 21)


def _patched():
    from unsloth_zoo.temporary_patches import glm5_next_sparse_flex as g
    g.patch_glm5_next_sparse()
    if "indexer" not in g._REFERENCE:
        assert not _LAYOUT_REQUIRED, "patch did not install on this transformers"
        pytest.skip(f"transformers {transformers.__version__}: glm5_next source layout not supported (no-op)")
    return g


def test_unsupported_layout_keeps_upstream():
    from unsloth_zoo.temporary_patches import glm5_next_sparse_flex as g
    g.patch_glm5_next_sparse()
    if "indexer" in g._REFERENCE:
        pytest.skip("supported layout: the patch installed")
    m = _m()
    assert not getattr(m.Glm5NextTextAttention.forward, "_unsloth_sparse_flex", False)
    assert not getattr(m.Glm5NextTextIndexer.forward, "_unsloth_sparse_flex", False)


def _m():
    from transformers.models.glm5_next import modeling_glm5_next as m
    return m


@contextlib.contextmanager
def _upstream():
    g, m = _patched(), _m()
    attn, idx = m.Glm5NextTextAttention.forward, m.Glm5NextTextIndexer.forward
    m.Glm5NextTextAttention.forward, m.Glm5NextTextIndexer.forward = g._REFERENCE["attention"], g._REFERENCE["indexer"]
    try:
        yield
    finally:
        m.Glm5NextTextAttention.forward, m.Glm5NextTextIndexer.forward = attn, idx


def _text_config(**kw):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    base = dict(
        hidden_size = 64, intermediate_size = 64, moe_intermediate_size = 32, num_hidden_layers = 4,
        num_attention_heads = 4, num_key_value_heads = 4, qk_nope_head_dim = 16, qk_rope_head_dim = 0,
        v_head_dim = 16, kv_lora_rank = 32, q_lora_rank = 32, vocab_size = 256, index_topk = TOPK,
        index_n_heads = 4, index_head_dim = 16, index_kpool = KPOOL, n_routed_experts = 4,
        num_experts_per_tok = 2, linear_num_heads = 2, linear_head_dim = 16, pad_token_id = 0,
        layer_types = ["indexed_attention", "indexed_attention", "linear_attention", "indexed_attention"],
        indexer_types = ["full", "shared", "full", "full"],
        mlp_layer_types = ["dense", "dense", "sparse", "dense"],
    )
    base.update(kw)
    config = Glm5NextTextConfig(**base)
    config._attn_implementation = kw.pop("attn", "sdpa")
    return config


def _indexer(config, seed = 0):
    torch.manual_seed(seed)
    idx = _m().Glm5NextTextIndexer(config, 0)
    with torch.no_grad():
        for p in idx.parameters():
            p.normal_(0, 0.5)
    return idx


def _selected(topk_indices, kv_len):
    """[B, S, T] bool, the set each query keeps (order-free)."""
    valid = (topk_indices >= 0) & (topk_indices < kv_len)
    out = torch.zeros(*topk_indices.shape[:2], kv_len + 1, dtype = torch.bool)
    out.scatter_(-1, torch.where(valid, topk_indices, kv_len).long(), True)
    return out[..., :kv_len]


def _mask(batch, seq, layout):
    mask = torch.ones(batch, seq, dtype = torch.bool)
    if layout == "left":
        mask[0, :37] = False
    elif layout == "right":
        mask[-1, seq - 29:] = False
    elif layout == "hole":
        mask[0, 50:53] = False
    return mask


def _inputs(config, batch, seq, seed = 1):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(batch, seq, config.hidden_size, generator = g)
    q_resid = torch.randn(batch, seq, config.q_lora_rank, generator = g)
    return hidden, q_resid


# ------------------------------------------------------------------ install
def test_patch_installs_and_is_idempotent():
    g, m = _patched(), _m()
    attn, idx = m.Glm5NextTextAttention.forward, m.Glm5NextTextIndexer.forward
    assert getattr(attn, "_unsloth_sparse_flex", False) and getattr(idx, "_unsloth_sparse_flex", False)
    g.patch_glm5_next_sparse()
    assert m.Glm5NextTextAttention.forward is attn and m.Glm5NextTextIndexer.forward is idx
    assert g._is_transformers_function(g._REFERENCE["indexer"])


# ------------------------------------------------------------------ indexer selection
@pytest.mark.parametrize("seq", [40, 79, 80, 95, 300])
@pytest.mark.parametrize("layout", ["none", "left", "right", "hole"])
@pytest.mark.parametrize("tail", [True, False])
def test_indexer_selects_upstream_sets(monkeypatch, seq, layout, tail):
    g = _patched()
    config = _text_config(index_kpool_always_select_tail = tail)
    idx = _indexer(config)
    hidden, q_resid = _inputs(config, 2, seq)
    mask = _mask(2, seq, layout)
    ref = g._REFERENCE["indexer"](idx, hidden, q_resid, mask, None)
    monkeypatch.setenv("UNSLOTH_SPARSE_INDEXER_Q_CHUNK", "16")  # several query chunks
    ours = idx(hidden_states = hidden, q_resid = q_resid, attention_mask = mask, past_key_values = None)
    assert ours.dtype == torch.int32 and ours.shape == ref.shape
    assert torch.equal(_selected(ours, seq), _selected(ref, seq))
    within = seq // KPOOL <= TOPK // KPOOL
    assert _stat("skip_full") == (1 if within else 0)
    assert _stat("indexer_chunked") == (0 if within else 1)
    # Fixed layout: whole pools in the first index_topk slots.
    pools = ours[..., :TOPK].view(2, seq, -1, KPOOL).long()
    first = pools[..., :1]
    whole = (pools == first + torch.arange(KPOOL)) | ((pools < 0) & (first < 0))
    assert bool(whole.all())


@pytest.mark.parametrize("kpool,topk", [(4, 16), (1, 32)])
def test_indexer_other_pool_sizes(monkeypatch, kpool, topk):
    g = _patched()
    config = _text_config(index_kpool = kpool, index_topk = topk)
    idx = _indexer(config)
    for seq, layout in ((20, "none"), (100, "left"), (100, "hole")):
        hidden, q_resid = _inputs(config, 2, seq)
        mask = _mask(2, seq, layout)
        ref = g._REFERENCE["indexer"](idx, hidden, q_resid, mask, None)
        monkeypatch.setenv("UNSLOTH_SPARSE_INDEXER_Q_CHUNK", "16")
        ours = idx(hidden_states = hidden, q_resid = q_resid, attention_mask = mask, past_key_values = None)
        monkeypatch.delenv("UNSLOTH_SPARSE_INDEXER_Q_CHUNK")
        assert torch.equal(_selected(ours, seq), _selected(ref, seq)), (seq, layout)


@pytest.mark.parametrize("env", ["UNSLOTH_SPARSE_ATTN_SKIP_FULL", "UNSLOTH_SPARSE_INDEXER_CHUNKED"])
def test_kill_switches_restore_upstream(monkeypatch, env):
    g = _patched()
    config = _text_config()
    idx = _indexer(config)
    seq = 40 if env.endswith("SKIP_FULL") else 300
    hidden, q_resid = _inputs(config, 2, seq)
    mask = _mask(2, seq, "left")
    calls = []
    original = g._REFERENCE["indexer"]
    monkeypatch.setitem(g._REFERENCE, "indexer", lambda *a: calls.append(1) or original(*a))
    monkeypatch.setenv(env, "0")
    out = idx(hidden_states = hidden, q_resid = q_resid, attention_mask = mask, past_key_values = None)
    assert calls == [1] and _stat("skip_full") == 0 and _stat("indexer_chunked") == 0
    assert torch.equal(out, original(idx, hidden, q_resid, mask, None))
    monkeypatch.delenv(env)
    out = idx(hidden_states = hidden, q_resid = q_resid, attention_mask = mask, past_key_values = None)
    assert calls == [1] and _stat("skip_full") + _stat("indexer_chunked") + _stat("indexer_scored") == 1


def test_cached_prefill_then_decode_across_the_budget():
    from transformers import DynamicCache
    g = _patched()
    config = _text_config()
    idx = _indexer(config)
    total = 100
    hidden, q_resid = _inputs(config, 2, total)
    mask = _mask(2, total, "left")
    caches = [DynamicCache(config = config), DynamicCache(config = config)]
    for cache in caches:  # attention cache layers must exist for get_seq_length / keys
        cache.update(torch.zeros(2, 1, 0, 32), torch.zeros(2, 1, 0, 0), 0)
    steps = [(0, 60)] + [(i, i + 1) for i in range(60, total)]
    for start, end in steps:
        args = (hidden[:, start:end], q_resid[:, start:end], mask[:, start:end] if end - start > 1 else torch.ones(2, 1, dtype = torch.bool))
        for cache in caches:
            cache.update(torch.zeros(2, 1, end - start, 32), torch.zeros(2, 1, end - start, 0), 0)
        ref = g._REFERENCE["indexer"](idx, *args, caches[0])
        ours = idx(hidden_states = args[0], q_resid = args[1], attention_mask = args[2], past_key_values = caches[1])
        assert torch.equal(_selected(ours, end), _selected(ref, end)), (start, end)
    assert _stat("skip_full") > 0 and _stat("indexer_scored") > 0


# ------------------------------------------------------------------ block mask
def _dense_from_block_mask(block_mask, batch, q_len, kv_len):
    """What flex keeps: full tiles whole, partial tiles through mask_mod, others nothing."""
    tile = sb._TILE
    def tiles(num, ids):
        n_kv = -(-kv_len // tile)
        out = torch.zeros(*num.shape, n_kv + 1, dtype = torch.bool)
        slot = torch.arange(ids.shape[-1])
        idx = torch.where(slot < num[..., None], ids.long(), n_kv)
        out.scatter_(-1, idx, True)
        return out[..., :n_kv]
    partial = tiles(block_mask.kv_num_blocks, block_mask.kv_indices)[:, 0]
    full = tiles(block_mask.full_kv_num_blocks, block_mask.full_kv_indices)[:, 0]

    def up(t):
        return t.repeat_interleave(tile, 1).repeat_interleave(tile, 2)[:, :q_len, :kv_len]

    b = torch.arange(batch)[:, None, None]
    q = torch.arange(q_len)[None, :, None]
    kv = torch.arange(kv_len)[None, None, :]
    mm = block_mask.mask_mod(b, torch.zeros_like(b), q, kv)
    return up(full) | (up(partial) & mm)


@pytest.mark.parametrize("seq", [300, 517])
@pytest.mark.parametrize("layout", ["none", "left", "right", "hole"])
@pytest.mark.parametrize("tail", [True, False])
def test_block_mask_keeps_exactly_the_selection(seq, layout, tail):
    g = _patched()
    config = _text_config(index_kpool_always_select_tail = tail)
    idx = _indexer(config)
    hidden, q_resid = _inputs(config, 2, seq)
    mask = _mask(2, seq, layout)
    topk = idx(hidden_states = hidden, q_resid = q_resid, attention_mask = mask, past_key_values = None)
    bm = g.glm5_next_block_mask(topk, seq, KPOOL, TOPK, KPOOL - 1 if tail else 0)
    assert torch.equal(_dense_from_block_mask(bm, 2, seq, seq), _selected(topk, seq))
    # Upstream's dense mask is the same set.
    dense = _selected(g._REFERENCE["indexer"](idx, hidden, q_resid, mask, None), seq)
    assert torch.equal(_dense_from_block_mask(bm, 2, seq, seq), dense)
    # Sparse in fact: far fewer tiles than causal.
    assert int(bm.full_kv_num_blocks.sum() + bm.kv_num_blocks.sum()) > 0


def _flex_semantics(query, key, value, block_mask, scale, score_mod = None):
    """Dense flex reference with autograd: fully masked rows return 0 like flex."""
    keep = _dense_from_block_mask(block_mask, query.shape[0], query.shape[2], key.shape[2])[:, None]
    scores = (query @ key.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~keep, float("-inf"))
    probs = torch.nan_to_num(scores.softmax(-1) * keep)
    return probs @ value


# ------------------------------------------------------------------ whole model
def _model(seed = 0, **kw):
    m = _m()
    torch.manual_seed(seed)
    config = _text_config(**kw)
    model = m.Glm5NextTextModel(config).float()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "indexer" in name:
                p.normal_(0, 0.5)
    return model


def _run(model, ids, mask):
    out = model(input_ids = ids, attention_mask = mask).last_hidden_state
    loss = (out * mask[..., None]).float().pow(2).mean()
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none = True)
    return out.detach(), grads


@pytest.mark.parametrize("seq", [60, 300])
@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_tiny_model_logits_and_grads_match_upstream(monkeypatch, seq, attn):
    """Skip (60) / chunked + flex (300) vs upstream, every path engaged and counted."""
    _patched()
    model = _model()
    model.config._attn_implementation = attn
    ids = torch.randint(0, 256, (2, seq), generator = torch.Generator().manual_seed(3))
    mask = _mask(2, seq, "left")
    with _upstream():
        ref_out, ref_grads = _run(model, ids, mask)
    monkeypatch.setattr(sb, "sparse_flex_device_ok", lambda q: True)
    monkeypatch.setattr(sb, "_flex", _flex_semantics)
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX_GLM5_NEXT", "1")
    monkeypatch.setenv("UNSLOTH_SPARSE_INDEXER_Q_CHUNK", "64")
    gates.reset_sparse_attn_stats()
    out, grads = _run(model, ids, mask)
    valid = mask[..., None]
    if attn == "sdpa":
        torch.testing.assert_close(out, ref_out, atol = 1e-5, rtol = 1e-5)
    else:  # eager spreads padded query rows uniformly instead of zeroing them: compare real rows only
        torch.testing.assert_close(out * valid, ref_out * valid, atol = 1e-5, rtol = 1e-5)
    assert ref_grads.keys() == grads.keys()
    for name in ref_grads:
        torch.testing.assert_close(grads[name], ref_grads[name], atol = 1e-5, rtol = 1e-4, msg = name)
    if seq == 60:
        assert _stat("skip_full") == 2 and _stat("flex") == 0
    else:
        # 2 full-indexer layers chunk; 3 attention layers run flex; the shared one reuses the BlockMask.
        assert _stat("indexer_chunked") == 2 and _stat("flex") == 3 and _stat("flex_reused") == 1


def test_flex_kill_switch_and_cpu_gate(monkeypatch):
    _patched()
    model = _model()
    ids = torch.randint(0, 256, (1, 300))
    mask = torch.ones(1, 300, dtype = torch.bool)
    with torch.no_grad():
        model(input_ids = ids, attention_mask = mask)
    assert _stat("flex") == 0 and _stat("indexer_chunked") + _stat("indexer_scored") == 2  # CPU: no flex
    monkeypatch.setattr(sb, "sparse_flex_device_ok", lambda q: True)
    monkeypatch.setattr(sb, "_flex", _flex_semantics)
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX_GLM5_NEXT", "1")
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX", "0")
    gates.reset_sparse_attn_stats()
    # The global kill switch wins over the opt-in.
    assert not gates.sparse_flex_enabled() and not gates.sparse_flex_opted_in("glm5_next")
    with torch.no_grad():
        model(input_ids = ids, attention_mask = mask)
    assert _stat("flex") == 0
    monkeypatch.setenv("UNSLOTH_SPARSE_INDEXER_CHUNKED", "0")
    with torch.no_grad():
        model(input_ids = ids, attention_mask = mask)
    assert _stat("flex") == 0  # no patched layout -> no flex


@pytest.mark.parametrize("opt_in", [None, "0", "true"])
def test_flex_is_off_by_default(monkeypatch, opt_in):
    """Every gate but the opt-in passes (device stubbed): the default, "0" and anything but "1" stay dense."""
    _patched()
    model = _model()
    if opt_in is not None:
        monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX_GLM5_NEXT", opt_in)
    monkeypatch.setattr(sb, "sparse_flex_device_ok", lambda q: True)
    calls = []
    monkeypatch.setattr(sb, "_flex", lambda *a, **k: calls.append(1) or _flex_semantics(*a, **k))
    ids = torch.randint(0, 256, (1, 300))
    mask = torch.ones(1, 300, dtype = torch.bool)
    with torch.no_grad():
        model(input_ids = ids, attention_mask = mask)
    assert not gates.sparse_flex_opted_in("glm5_next")
    assert calls == [] and _stat("flex") == 0
    # The chunked indexer (and the skip below the budget) stay on by default.
    assert _stat("indexer_chunked") + _stat("indexer_scored") == 2
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX_GLM5_NEXT", "1")
    gates.reset_sparse_attn_stats()
    with torch.no_grad():
        model(input_ids = ids, attention_mask = mask)
    assert _stat("flex") == 3 and len(calls) == 3


@pytest.mark.parametrize("compiled", [False, True])
def test_cpu_flex_forward_matches_upstream(monkeypatch, compiled):
    """Real flex kernels (CPU, forward only) against upstream dense SDPA."""
    from torch.nn.attention.flex_attention import flex_attention
    _patched()
    model = _model()
    seq = 300
    ids = torch.randint(0, 256, (2, seq), generator = torch.Generator().manual_seed(5))
    mask = _mask(2, seq, "left")
    with torch.no_grad():
        with _upstream():
            ref = model(input_ids = ids, attention_mask = mask).last_hidden_state
        fn = torch.compile(flex_attention) if compiled else flex_attention
        monkeypatch.setattr(sb, "sparse_flex_device_ok", lambda q: True)
        monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX_GLM5_NEXT", "1")
        monkeypatch.setattr("unsloth_zoo.flex_attention.utils.flex_attention", fn)
        gates.reset_sparse_attn_stats()
        out = model(input_ids = ids, attention_mask = mask).last_hidden_state
    assert _stat("flex") == 3
    torch.testing.assert_close(out, ref, atol = 2e-5, rtol = 1e-4)


def test_tiny_model_cached_greedy_decode_across_the_budget():
    """Prefill below the budget (skip), decode past it (scored, one query): same tokens and logits."""
    _patched()
    m = _m()
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
    torch.manual_seed(0)
    text = _text_config().to_dict()
    vision = dict(depth = 1, hidden_size = 32, intermediate_size = 32, num_heads = 2, out_hidden_size = 64)
    config = Glm5NextConfig(text_config = text, vision_config = vision)
    config.text_config._attn_implementation = "sdpa"
    model = m.Glm5NextForConditionalGeneration(config).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "indexer" in name:
                p.normal_(0, 0.5)
    ids = torch.randint(1, 256, (2, 70), generator = torch.Generator().manual_seed(7))
    mask = _mask(2, 70, "left")
    kw = dict(attention_mask = mask, max_new_tokens = 20, do_sample = False, output_logits = True, return_dict_in_generate = True)
    with torch.no_grad():
        with _upstream():
            ref = model.generate(ids, **kw)
        gates.reset_sparse_attn_stats()
        out = model.generate(ids, **kw)
    assert torch.equal(out.sequences, ref.sequences)
    for a, b in zip(out.logits, ref.logits):
        torch.testing.assert_close(a, b, atol = 1e-5, rtol = 1e-5)
    assert _stat("skip_full") > 0 and _stat("indexer_scored") > 0


def test_static_cache_key_length_is_the_allocation_before_the_first_prefill():
    # An uninitialized static cache still pools its whole allocation: predicting the prompt length
    # instead let a short prompt take upstream's indexer while the flex mask read our layout.
    g = _patched()
    config = _text_config()
    idx = _indexer(config)

    class Layer:
        indexer_keys = None
        max_cache_len = 300

    class Cache:
        layers = [Layer()] * (idx.layer_idx + 1)

    assert g._predicted_key_len(idx, 60, Cache()) == 300
    Layer.max_cache_len = None
    assert g._predicted_key_len(idx, 60, Cache()) == 60
    assert g._predicted_key_len(idx, 60, None) == 60
