# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Qwen4Exp QSA indexer beyond the budget without a sparse attention kernel (T4 / sm75, CPU): the dense reference
attention stays, and its selection mask is built vectorised instead of transformers' per-(batch, query) loop.
Token identity with the loop (ties included), the same losses and gradients through the text model, and a kill
switch (UNSLOTH_QWEN4_EXP_VECTORISED_QSA=0) that restores the loop."""
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")

from unsloth_zoo.temporary_patches import qwen4_exp as zoo_qwen4_exp  # noqa: E402

BUDGET, RATIO = 16, 4


def _config(attn = "sdpa"):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    config = Qwen4ExpTextConfig(
        hidden_size = 64, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 32,
        indexer_budget = BUDGET, indexer_compress_ratio = RATIO, indexer_head_dim = 16, indexer_n_heads = 2,
        indexer_kv_heads = 1, partial_rotary_factor = 0.25, num_hidden_layers = 4, vocab_size = 128,
        layer_types = ["linear_attention"] * 3 + ["full_attention"],
    )
    config._attn_implementation = attn
    return config


def _reference():
    zoo_qwen4_exp.patch_qwen4_exp()
    reference = zoo_qwen4_exp._reference_qsa_forward
    assert reference is not None and reference.__module__.startswith("transformers."), reference
    return reference


def _masks(kind, b, s):
    causal = torch.tril(torch.ones(s, s, dtype = torch.bool))
    if kind == "causal":
        m = causal.expand(b, s, s).clone()
    elif kind == "right_pad":
        m = causal.expand(b, s, s).clone()
        m[1, :, s - 9 :] = False
    elif kind == "packed":  # documents of 23 / 7 / rest tokens, as padding-free packing builds them
        doc = torch.tensor([0] * 23 + [1] * 7 + [2] * (s - 30))
        m = (causal & (doc[:, None] == doc[None, :])).expand(b, s, s).clone()
    elif kind == "sliding":
        idx = torch.arange(s)
        m = (causal & (idx[:, None] - idx[None, :] < 20)).expand(b, s, s).clone()
    return m.unsqueeze(1)


def _inputs(config, b, s, seed = 1):
    rotary = modeling.Qwen4ExpTextRotaryEmbedding(config)
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(b, s, config.hidden_size, generator = g, dtype = torch.float64)
    pos = torch.arange(s).unsqueeze(0).expand(3, b, -1)
    cos, sin = rotary(hidden.float(), pos)
    return hidden, (cos.double(), sin.double())


def _indexer(config, ties = False, seed = 0):
    torch.manual_seed(seed)
    indexer = modeling.Qwen4ExpTextQSAIndexer(config, layer_idx = 3).double()
    for p in indexer.parameters():
        torch.nn.init.normal_(p, 0, 0.5)
    if ties:  # every pooled block key 0 -> every block score 0: each top-k boundary is a tie
        with torch.no_grad():
            indexer.index_qk_proj.weight[indexer.index_n_heads * indexer.index_head_dim:] = 0
    return indexer


def _as_float(mask):
    return torch.zeros(mask.shape, dtype = torch.float64).masked_fill(~mask, torch.finfo(torch.float64).min)


@pytest.mark.parametrize("floaty", [False, True], ids = ["bool", "float"])
@pytest.mark.parametrize("ties", [False, True], ids = ["random", "all_tied"])
@pytest.mark.parametrize("kind", ["causal", "right_pad", "packed"])
def test_indexer_mask_equals_the_reference_loop(monkeypatch, kind, ties, floaty):
    monkeypatch.delenv("UNSLOTH_QWEN4_EXP_EXACT_TIES", raising = False)
    monkeypatch.delenv("UNSLOTH_QWEN4_EXP_VECTORISED_QSA", raising = False)
    reference = _reference()
    config = _config()
    indexer = _indexer(config, ties)
    b, s = 2, 61
    hidden, pe = _inputs(config, b, s)
    mask = _masks(kind, b, s)
    mask = _as_float(mask) if floaty else mask
    calls = []
    monkeypatch.setattr(zoo_qwen4_exp, "_reference_qsa_forward", lambda *a, **k: calls.append(1) or reference(*a, **k))
    before = zoo_qwen4_exp.QWEN4_EXP_INDEXER_STATS["vectorised"]
    with torch.no_grad():
        want = reference(indexer, hidden, pe, mask, None)
        got = zoo_qwen4_exp.qwen4_exp_qsa_indexer_forward(indexer, hidden, pe, mask, None)
    assert calls == [], "the per-query loop ran"
    assert zoo_qwen4_exp.QWEN4_EXP_INDEXER_STATS["vectorised"] == before + 1
    assert got.dtype == want.dtype and got.shape == want.shape
    assert torch.equal(got, want)
    # The selection really drops blocks here (otherwise this would only test the within-budget case).
    visible = mask if mask.dtype == torch.bool else mask == 0
    assert not torch.equal(got if got.dtype == torch.bool else got == 0, visible)


def test_kill_switch_and_non_interval_masks_keep_the_loop(monkeypatch):
    reference = _reference()
    config = _config()
    indexer = _indexer(config)
    b, s = 2, 61
    hidden, pe = _inputs(config, b, s)
    calls = []
    monkeypatch.setattr(zoo_qwen4_exp, "_reference_qsa_forward", lambda *a, **k: calls.append(1) or reference(*a, **k))
    with torch.no_grad():
        # Sliding windows are not interval-shaped per document: the loop.
        sliding = _masks("sliding", b, s)
        assert torch.equal(zoo_qwen4_exp.qwen4_exp_qsa_indexer_forward(indexer, hidden, pe, sliding, None),
                           reference(indexer, hidden, pe, sliding, None))
        assert len(calls) == 1
        monkeypatch.setenv("UNSLOTH_QWEN4_EXP_VECTORISED_QSA", "0")
        causal = _masks("causal", b, s)
        assert torch.equal(zoo_qwen4_exp.qwen4_exp_qsa_indexer_forward(indexer, hidden, pe, causal, None),
                           reference(indexer, hidden, pe, causal, None))
        assert len(calls) == 2


def _plain_gdn(monkeypatch):
    # CPU run: torch linear-attention paths (a GPU host's Triton kernels reject CPU tensors).
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                 "causal_conv1d_update"):
        original = getattr(getattr(modeling, name, None), "__wrapped__", None)
        if original is not None:
            monkeypatch.setattr(modeling, name, original)


def _loss_and_grads(model, ids, mask):
    model.zero_grad(set_to_none = True)
    out = model(input_ids = ids, attention_mask = mask, use_cache = False).last_hidden_state
    loss = (out.float() ** 2).mean() + out.float().sum(-1).std()
    loss.backward()
    return loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def test_text_model_losses_and_grads_match_the_loop(monkeypatch):
    # fp32 text model over 2 x 61 tokens, beyond the budget, the reference (dense) attention: the vectorised mask
    # gives the loop's loss and gradients (the selection is discrete, so they are equal, not only close).
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "")
    monkeypatch.setenv("UNSLOTH_QWEN4_EXP_SPARSE_ATTN", "0")  # the T4 / CPU route: dense reference attention
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE", "gpu")
    monkeypatch.delenv("UNSLOTH_QWEN4_EXP_EXACT_TIES", raising = False)
    _plain_gdn(monkeypatch)
    _reference()
    config = _config()
    torch.manual_seed(0)
    model = modeling.Qwen4ExpTextModel(config).float().train()
    for layer in model.layers:
        layer.mlp = layer.mlp.shared_expert
    ids = torch.randint(0, config.vocab_size, (2, 61), generator = torch.Generator().manual_seed(3))
    mask = torch.ones_like(ids)
    mask[1, -9:] = 0
    before = zoo_qwen4_exp.QWEN4_EXP_INDEXER_STATS["vectorised"]
    got_loss, got = _loss_and_grads(model, ids, mask)
    assert zoo_qwen4_exp.QWEN4_EXP_INDEXER_STATS["vectorised"] > before
    monkeypatch.setenv("UNSLOTH_QWEN4_EXP_VECTORISED_QSA", "0")
    after = zoo_qwen4_exp.QWEN4_EXP_INDEXER_STATS["vectorised"]
    want_loss, want = _loss_and_grads(model, ids, mask)
    assert zoo_qwen4_exp.QWEN4_EXP_INDEXER_STATS["vectorised"] == after
    assert torch.equal(got_loss, want_loss)
    assert got.keys() == want.keys() and len(got) > 0
    for name in want:
        assert torch.equal(got[name], want[name]), name
