# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The shared one-token flex decode contract (unsloth_zoo.flex_attention.decode) against float64 references: the
mask against an explicit visibility rule, sinks against the official gpt-oss attention, softcap, a BlockMask refilled
the way CUDA-graph decode refills it, and the graph identity."""

import os
import warnings

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

import pytest
import torch

flex = pytest.importorskip("torch.nn.attention.flex_attention")

from unsloth_zoo.flex_attention import decode as D  # noqa: E402

if not D._HAS_SEQ_LENGTHS:
    pytest.skip("BlockMask without seq_lengths (torch < 2.6)", allow_module_level = True)

warnings.filterwarnings("ignore", message = ".*flex_attention.*")


# Official gpt-oss attention, verbatim apart from the dtype of the mask constant: openai/gpt-oss,
# gpt_oss/torch/model.py `sdpa` (learned sink logit per head appended to the softmax, then dropped; sliding window
# masks keys more than sliding_window - 1 tokens back, so W keys including the query).
def official_gpt_oss_sdpa(Q, K, V, S, sm_scale, sliding_window = 0):
    # sliding_window == 0 means no sliding window
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal = 1)
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal = -sliding_window
        )
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim = -1)
    W = torch.softmax(QK, dim = -1)
    W = W[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


def visible(q_pos, kv_pos, window, real):
    """The rule every decode mask must follow: causal, within W tokens including the query, a real token."""
    ok = (kv_pos <= q_pos) and real
    if window:
        ok = ok and kv_pos > q_pos - window
    return ok


def effective_mask(block_mask, rows, kv_length):
    """[rows, kv_length] what flex attends to: a key must sit in a listed block AND pass mask_mod."""
    block = block_mask.BLOCK_SIZE[1]
    listed = torch.zeros(rows, kv_length, dtype = torch.bool)
    for b in range(rows):
        n = int(block_mask.kv_num_blocks[b, 0, 0])
        for i in block_mask.kv_indices[b, 0, 0, :n].tolist():
            listed[b, i * block:(i + 1) * block] = True
    dense = D.dense_decode_mask(block_mask, kv_length, torch.device("cpu"))[:, 0, 0, :]
    return listed & dense


@pytest.mark.parametrize("window", [0, 128])
def test_mask_follows_the_visibility_rule_across_blocks_padding_and_unwritten_slots(window):
    # Full layer: a 512-slot static cache (slots past the query are unwritten); sliding layer: 128 slots rolling.
    pads = [0, 37, 170]
    rows = len(pads)
    for q in [0, 5, 127, 128, 129, 200, 255, 256, 300, 400]:
        if window:
            kv_length, kv_offset = window, max(0, q - window + 1)
        else:
            kv_length, kv_offset = 512, 0
        width = q + 1
        mask_2d = torch.ones(rows, width, dtype = torch.long)
        for b, p in enumerate(pads):
            mask_2d[b, :min(p, width)] = 0
        bm = D.decode_block_mask(
            rows, kv_length, torch.tensor(q), torch.tensor(kv_offset), attention_mask = mask_2d, window = window,
            device = "cpu",
        )
        got = effective_mask(bm, rows, kv_length)
        want = torch.tensor([
            [visible(q, kv_offset + j, window, (kv_offset + j) < width and bool(mask_2d[b, kv_offset + j]))
             for j in range(kv_length)]
            for b in range(rows)
        ])
        assert torch.equal(got, want), (q, window)


def test_mask_matches_transformers_mask_functions():
    from transformers.masking_utils import sliding_window_causal_mask_function, causal_mask_function
    for window, fn in ((0, causal_mask_function), (128, sliding_window_causal_mask_function(128))):
        for q in [3, 127, 128, 333]:
            kv_length, kv_offset = (128, max(0, q - 127)) if window else (384, 0)
            ours = D.decode_block_mask(2, kv_length, torch.tensor(q), torch.tensor(kv_offset), window = window, device = "cpu")
            theirs = D.decode_block_mask(2, kv_length, torch.tensor(q), torch.tensor(kv_offset), mask_function = fn, device = "cpu")
            assert torch.equal(effective_mask(ours, 2, kv_length), effective_mask(theirs, 2, kv_length))


def _decode_inputs(n_tokens, n_kv, q_mult, d, seed, sink_scale = 1.0):
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(n_tokens, n_kv, q_mult, d, generator = g, dtype = torch.float64)
    K = torch.randn(n_tokens, n_kv, d, generator = g, dtype = torch.float64)
    V = torch.randn(n_tokens, n_kv, d, generator = g, dtype = torch.float64)
    S = torch.randn(n_kv * q_mult, generator = g, dtype = torch.float64) * sink_scale
    return Q, K, V, S


@pytest.mark.parametrize("window", [0, 16])
@pytest.mark.parametrize("sink_scale", [1.0, 30.0])
def test_sinks_match_the_official_gpt_oss_attention(window, sink_scale):
    # Decode the last token of a 40-token sequence with flex + sinks; compare with the official full computation.
    n_tokens, n_kv, q_mult, d = 40, 2, 4, 16
    Q, K, V, S = _decode_inputs(n_tokens, n_kv, q_mult, d, seed = 3 + window, sink_scale = sink_scale)
    scale = d ** -0.5
    want = official_gpt_oss_sdpa(Q, K, V, S, scale, window)[-1].view(n_kv * q_mult, d)
    q = Q[-1].reshape(1, n_kv * q_mult, 1, d)
    k = K.permute(1, 0, 2).unsqueeze(0)
    v = V.permute(1, 0, 2).unsqueeze(0)
    bm = D.decode_block_mask(1, n_tokens, torch.tensor(n_tokens - 1), window = window, device = "cpu")
    got = D.flex_decode_attention(q, k, v, bm, scale, sinks = S)[0, :, 0, :]
    torch.testing.assert_close(got, want, rtol = 1e-10, atol = 1e-12)


def test_an_all_masked_row_returns_exact_zeros_with_and_without_sinks():
    d, n_kv, q_mult, S_len = 16, 2, 2, 64
    g = torch.Generator().manual_seed(0)
    q = torch.randn(2, n_kv * q_mult, 1, d, generator = g, dtype = torch.float64)
    k = torch.randn(2, n_kv, S_len, d, generator = g, dtype = torch.float64)
    v = torch.randn(2, n_kv, S_len, d, generator = g, dtype = torch.float64)
    mask_2d = torch.ones(2, S_len, dtype = torch.long)
    mask_2d[1] = 0  # row 1 is all padding
    bm = D.decode_block_mask(2, S_len, torch.tensor(S_len - 1), attention_mask = mask_2d, device = "cpu")
    for sinks in (None, torch.randn(n_kv * q_mult, generator = g, dtype = torch.float64)):
        out = D.flex_decode_attention(q, k, v, bm, d ** -0.5, sinks = sinks)
        assert torch.isfinite(out).all()
        assert torch.equal(out[1], torch.zeros_like(out[1]))
        # The official attention gives a fully masked query all weight on the sink, i.e. zeros.
        ref = D._eager_attention(q, k, v, mask_2d.bool().view(2, 1, 1, S_len), d ** -0.5, sinks = sinks)
        torch.testing.assert_close(out, ref, rtol = 1e-10, atol = 1e-12)


def test_sinks_with_unequal_heads_and_left_padding_match_the_reference():
    d, n_kv, q_mult, S_len = 32, 2, 4, 200
    g = torch.Generator().manual_seed(7)
    q = torch.randn(3, n_kv * q_mult, 1, d, generator = g, dtype = torch.float64)
    k = torch.randn(3, n_kv, S_len, d, generator = g, dtype = torch.float64) * 3
    v = torch.randn(3, n_kv, S_len, d, generator = g, dtype = torch.float64)
    sinks = torch.linspace(-20, 20, n_kv * q_mult, dtype = torch.float64)
    mask_2d = torch.ones(3, S_len, dtype = torch.long)
    mask_2d[1, :90] = 0
    mask_2d[2, :199] = 0
    for window in (0, 128):
        bm = D.decode_block_mask(3, S_len, torch.tensor(S_len - 1), attention_mask = mask_2d, window = window, device = "cpu")
        dense = torch.tensor([[visible(S_len - 1, j, window, bool(mask_2d[b, j])) for j in range(S_len)] for b in range(3)])
        ref = D._eager_attention(q, k, v, dense.view(3, 1, 1, S_len), d ** -0.5, sinks = sinks)
        out = D.flex_decode_attention(q, k, v, bm, d ** -0.5, sinks = sinks)
        torch.testing.assert_close(out, ref, rtol = 1e-10, atol = 1e-12)


def test_softcap_never_leaks_through_masked_keys():
    # Masked keys get enormous scores: capping them must not make them visible (cap first, then mask).
    d, S_len, cap = 16, 96, 30.0
    g = torch.Generator().manual_seed(11)
    q = torch.randn(1, 4, 1, d, generator = g, dtype = torch.float64)
    k = torch.randn(1, 4, S_len, d, generator = g, dtype = torch.float64)
    v = torch.randn(1, 4, S_len, d, generator = g, dtype = torch.float64)
    k[:, :, :50] = q[:, :, :1] * 1e4  # left padding with dominant scores
    mask_2d = torch.ones(1, S_len, dtype = torch.long)
    mask_2d[:, :50] = 0
    bm = D.decode_block_mask(1, S_len, torch.tensor(S_len - 1), attention_mask = mask_2d, device = "cpu")
    out = D.flex_decode_attention(q, k, v, bm, d ** -0.5, softcap = cap)
    # Independent reference: only the 46 real keys, capped scores, plain softmax.
    scores = (q @ k[..., 50:, :].transpose(-1, -2)) * d ** -0.5
    want = torch.softmax(cap * torch.tanh(scores / cap), dim = -1) @ v[..., 50:, :]
    torch.testing.assert_close(out, want, rtol = 1e-10, atol = 1e-12)


def _refill(static, fresh):
    """What CUDA-graph decode does each step: copy the new mask's tensors into the captured mask's buffers."""
    for a, b in zip(static["fields"], (fresh.kv_num_blocks, fresh.kv_indices)):
        a.copy_(b)
    for a, b in zip(static["closure"], fresh._unsloth_graph_closure[0]):
        a.copy_(b)


@pytest.mark.parametrize("window", [0, 128])
def test_one_block_mask_refilled_for_300_steps_equals_a_fresh_mask_each_step(window):
    from torch.nn.attention.flex_attention import BlockMask
    rows, d, start = 2, 16, 20
    kv_length = window or 512
    g = torch.Generator().manual_seed(5)
    k = torch.randn(rows, 2, kv_length, d, generator = g, dtype = torch.float64)
    v = torch.randn(rows, 2, kv_length, d, generator = g, dtype = torch.float64)
    mask_2d = torch.ones(rows, start + 300, dtype = torch.long)
    mask_2d[1, :13] = 0

    def fresh_at(t):
        q = start + t
        kv_offset = max(0, q - window + 1) if window else 0
        return D.decode_block_mask(rows, kv_length, torch.tensor(q), torch.tensor(kv_offset),
                                   attention_mask = mask_2d[:, :q + 1], window = window, device = "cpu")

    first = fresh_at(0)
    tensors, rebuild = first._unsloth_graph_closure
    static = {
        "fields": (first.kv_num_blocks.clone(), first.kv_indices.clone()),
        "closure": tuple(t.clone() for t in tensors),
    }
    captured = BlockMask.from_kv_blocks(
        static["fields"][0], static["fields"][1], BLOCK_SIZE = first.BLOCK_SIZE, mask_mod = rebuild(static["closure"]),
        seq_lengths = first.seq_lengths,
    )
    for t in range(300):
        fresh = fresh_at(t)
        assert fresh._unsloth_mask_identity == first._unsloth_mask_identity
        _refill(static, fresh)
        q = torch.randn(rows, 4, 1, d, generator = g, dtype = torch.float64)
        torch.testing.assert_close(
            D.flex_decode_attention(q, k, v, captured, d ** -0.5),
            D.flex_decode_attention(q, k, v, fresh, d ** -0.5),
            rtol = 0, atol = 0,
        )


def test_graph_identity_follows_window_and_captured_scalars():
    ident = lambda **kw: getattr(D.decode_block_mask(1, 256, torch.tensor(100), device = "cpu", **kw), "_unsloth_mask_identity", None)
    assert ident(window = 128) is not None
    assert ident(window = 128) == ident(window = 128)
    assert ident(window = 128) != ident(window = 64)
    assert ident() != ident(window = 128)
    table = torch.ones(4, dtype = torch.bool)
    # A mask_function reading a tensor cannot be refilled by the graph: no contract.
    bm = D.decode_block_mask(1, 256, torch.tensor(100), mask_function = lambda b, h, q, kv: table[b] & (kv <= q), device = "cpu")
    assert getattr(bm, "_unsloth_mask_identity", None) is None
    assert getattr(bm, "_unsloth_graph_closure", None) is None
    plain = D.decode_block_mask(1, 256, torch.tensor(100), window = 128, device = "cpu", graph_contract = False)
    assert getattr(plain, "_unsloth_graph_closure", None) is None


def test_the_offsets_are_copies_of_a_live_cache_counter():
    # StaticLayer bumps cumulative_length in place before attention reads the mask.
    counter = torch.tensor(99)
    bm = D.decode_block_mask(1, 256, counter, device = "cpu")
    before = effective_mask(bm, 1, 256).clone()
    counter.add_(50)
    assert torch.equal(effective_mask(bm, 1, 256), before)


def test_attention_interface_falls_back_to_the_same_visibility_without_flex_decoding():
    # On CPU flex_decoding is unsupported: the interface evaluates the BlockMask densely and runs SDPA.
    d, S_len = 16, 150
    g = torch.Generator().manual_seed(2)
    q = torch.randn(2, 4, 1, d, generator = g, dtype = torch.float64)
    k = torch.randn(2, 2, S_len, d, generator = g, dtype = torch.float64)
    v = torch.randn(2, 2, S_len, d, generator = g, dtype = torch.float64)
    mask_2d = torch.ones(2, S_len, dtype = torch.long)
    mask_2d[0, :40] = 0
    bm = D.decode_block_mask(2, S_len, torch.tensor(S_len - 1), attention_mask = mask_2d, window = 128, device = "cpu")
    assert not D.flex_decode_supported(d, 4, 2, torch.float64, "cpu")

    class M(torch.nn.Module):
        num_key_value_groups = 2
        is_causal = True

    out, _ = D.unsloth_flex_decode_attention(M(), q, k, v, bm, scaling = d ** -0.5)
    dense = torch.tensor([[visible(S_len - 1, j, 128, bool(mask_2d[b, j])) for j in range(S_len)] for b in range(2)])
    want = D._eager_attention(q, k, v, dense.view(2, 1, 1, S_len), d ** -0.5)
    torch.testing.assert_close(out, want.transpose(1, 2), rtol = 1e-10, atol = 1e-12)


def test_flex_decode_support_rules(monkeypatch):
    monkeypatch.setattr(D, "_device_capability", lambda index: (7, 5))
    D._supported.cache_clear()
    try:
        assert D._supported(64, 64, 8, torch.float16, "cuda", 0)
        assert D._supported(256, 16, 4, torch.float16, "cuda", 0)
        assert not D._supported(512, 8, 1, torch.float16, "cuda", 0)  # head_dim 512 needs sm80+
        assert not D._supported(256, 16, 4, torch.bfloat16, "cuda", 0)  # no bf16 below sm80
        assert not D._supported(128, 14, 2, torch.float16, "cuda", 0)  # GQA group 7 is not a power of two
        assert not D._supported(64, 64, 8, torch.float16, "cpu", -1)
        monkeypatch.setattr(D, "_device_capability", lambda index: (8, 0))
        D._supported.cache_clear()
        assert D._supported(512, 16, 2, torch.bfloat16, "cuda", 0)
    finally:
        D._supported.cache_clear()


def test_registration_adds_a_custom_key_that_needs_no_flex_support_flag():
    assert D.register_flex_decode()
    from transformers.modeling_utils import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    assert AttentionInterface._global_mapping[D.FLEX_DECODE_KEY] is D.unsloth_flex_decode_attention
    assert AttentionMaskInterface._global_mapping[D.FLEX_DECODE_KEY] is D.unsloth_flex_decode_mask
    # Not a one-token GPU step: the SDPA mask (CPU here).
    from transformers.masking_utils import causal_mask_function
    mask = D.unsloth_flex_decode_mask(
        batch_size = 1, cache_position = torch.arange(4), kv_length = 4, q_length = 4, q_offset = 0, kv_offset = 0,
        mask_function = causal_mask_function, attention_mask = None, device = torch.device("cpu"),
        allow_is_causal_skip = False,
    )
    assert not isinstance(mask, flex.BlockMask)
