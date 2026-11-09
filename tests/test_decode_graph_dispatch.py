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

# Kernel choices inside a generate() call Unsloth decodes through CUDA graphs (decode_graph_state scopes): the
# routed MoE kernels serve every decode row below sm80, gpt-oss decodes one token over a dense mask. Outside the
# scopes every dispatch decision is the one main makes.
from __future__ import annotations

import itertools
import threading

import pytest
import torch

from unsloth_zoo import decode_graph_state as S


def test_scopes_nest_and_need_the_generate_scope():
    assert not S.in_graph_generate() and not S.in_graph_decode_step()
    with S.graph_decode_step():
        # A step scope alone (no planned generate call) is not a graph decode step.
        assert not S.in_graph_decode_step()
    with S.graph_generate():
        assert S.in_graph_generate() and not S.in_graph_decode_step()
        with S.graph_decode_step():
            assert S.in_graph_decode_step()
            with S.graph_generate():
                assert S.in_graph_decode_step()
            assert S.in_graph_decode_step()
        assert not S.in_graph_decode_step()
    assert not S.in_graph_generate()


def test_scopes_are_per_thread():
    seen = []
    with S.graph_generate(), S.graph_decode_step():
        t = threading.Thread(target = lambda: seen.append((S.in_graph_generate(), S.in_graph_decode_step())))
        t.start(); t.join()
        assert S.in_graph_decode_step()
    assert seen == [(False, False)]


def test_scope_exits_on_exception():
    with pytest.raises(RuntimeError):
        with S.graph_generate():
            raise RuntimeError
    assert not S.in_graph_generate()


@pytest.fixture
def R():
    from unsloth_zoo.temporary_patches import moe_routed
    return moe_routed


def test_bf16_gate_without_the_graph_scope_is_mains(R):
    for one_token, rows, top_k in itertools.product((False, True), (1, 2, 4, 8, 16), (1, 2, 4, 8)):
        slots = rows * top_k
        main = not (slots > R.BF16_MAX_SLOTS or (one_token and rows > 1))
        assert R.bf16_routes(one_token, rows, slots) == main
        assert R.bf16_routes(one_token, rows, slots, graph_step = False) == main


def test_bf16_gate_in_a_graph_step_routes_every_row_below_sm80(R, monkeypatch):
    monkeypatch.setattr(R, "_SLOTS_PINNED", False)
    # Below sm80 (one_token_only): grouped_mm cannot be captured, so up to GRAPH_MAX_SLOTS route.
    assert R.bf16_routes(True, 4, 32, graph_step = True)
    assert R.bf16_routes(True, 16, 128, graph_step = True)
    assert not R.bf16_routes(True, 16, R.GRAPH_MAX_SLOTS + 1, graph_step = True)
    # sm80+ keeps its measured cap: Triton grouped GEMM captures there.
    for rows, top_k in itertools.product((1, 4, 16), (2, 8)):
        slots = rows * top_k
        assert R.bf16_routes(False, rows, slots, graph_step = True) == (slots <= R.BF16_MAX_SLOTS)


def test_explicit_slot_cap_is_kept_in_graph_steps(R, monkeypatch):
    monkeypatch.setattr(R, "_SLOTS_PINNED", True)
    assert R.bf16_routes(True, 4, 32, graph_step = True) == R.bf16_routes(True, 4, 32)


def test_nf4_gate(R, monkeypatch):
    E = 8
    for slots in (1, 8, 9, 64, 128, 129):
        main = slots <= R.nf4_slot_limit(E)
        assert R.nf4_routes(E, slots) == main
        assert R.nf4_routes(E, slots, sub_sm80 = False, graph_step = True) == main
        assert R.nf4_routes(E, slots, sub_sm80 = True, graph_step = False) == main
    if R.NF4_MAX_SLOTS is None:
        assert R.nf4_routes(E, 128, sub_sm80 = True, graph_step = True)
        assert not R.nf4_routes(E, R.GRAPH_MAX_SLOTS + 1, sub_sm80 = True, graph_step = True)


def test_graph_step_flag_reads_the_scopes(R):
    assert not R._graph_decode_step()
    with S.graph_generate():
        assert not R._graph_decode_step()
        with S.graph_decode_step():
            assert R._graph_decode_step()
    assert not R._graph_decode_step()


@pytest.fixture
def G():
    from unsloth_zoo.temporary_patches import gpt_oss
    return gpt_oss


def test_dense_graph_decode_selection(G, monkeypatch):
    monkeypatch.delenv("UNSLOTH_GPT_OSS_GRAPH_DECODE_ATTENTION", raising = False)
    one, many = torch.zeros(2, 1, 8), torch.zeros(2, 5, 8)
    assert not G._gpt_oss_dense_graph_decode(one)
    with S.graph_generate():
        assert G._gpt_oss_dense_graph_decode(one)
        # Prefill and multi-token continuations keep flex.
        assert not G._gpt_oss_dense_graph_decode(many)
        assert not G._gpt_oss_dense_graph_decode(None)
        monkeypatch.setenv("UNSLOTH_GPT_OSS_GRAPH_DECODE_ATTENTION", "flex")
        assert not G._gpt_oss_dense_graph_decode(one)
    assert not G._gpt_oss_dense_graph_decode(one)


class _Attn(torch.nn.Module):
    def __init__(self, heads, sinks):
        super().__init__()
        self.sinks = torch.nn.Parameter(sinks)


def _reference(module, query, key, value, mask, scaling, kv_heads):
    # transformers' gpt-oss eager attention with sinks (repeat_kv, logits + mask, sink column, fp32 softmax stored
    # back in the logits dtype, sink dropped).
    groups = query.shape[1] // kv_heads
    k = key.repeat_interleave(groups, dim = 1)
    v = value.repeat_interleave(groups, dim = 1)
    w = torch.matmul(query, k.transpose(2, 3)) * scaling
    if mask is not None:
        w = w + mask[:, :, :, : k.shape[-2]]
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    logits = torch.cat([w, sinks.to(w.dtype)], dim = -1)
    probs = torch.softmax(logits, dim = -1, dtype = torch.float32).to(logits.dtype)[..., :-1]
    return torch.matmul(probs.to(v.dtype), v).transpose(1, 2).contiguous()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("bsz,heads,kv_heads,kv_len", [(1, 8, 2, 33), (3, 8, 8, 128), (2, 16, 4, 300)])
def test_decode_attention_matches_eager_sink_attention(G, dtype, bsz, heads, kv_heads, kv_len):
    g = torch.Generator().manual_seed(bsz * 1000 + kv_len)
    D = 16
    q = torch.randn(bsz, heads, 1, D, generator = g).to(dtype)
    k = torch.randn(bsz, kv_heads, kv_len, D, generator = g).to(dtype)
    v = torch.randn(bsz, kv_heads, kv_len, D, generator = g).to(dtype)
    sinks = torch.randn(heads, generator = g).to(dtype)
    # Left padding + an unwritten static tail: per row visible keys, additive min-dtype mask.
    visible = torch.ones(bsz, kv_len, dtype = torch.bool)
    for b in range(bsz):
        visible[b, : b * 3] = False
        visible[b, kv_len - 5 :] = False
    mask = torch.zeros(bsz, 1, 1, kv_len, dtype = dtype).masked_fill(~visible[:, None, None, :], torch.finfo(dtype).min)
    module = _Attn(heads, sinks)
    want = _reference(module, q, k, v, mask, 0.25, kv_heads)
    got, weights = G._gpt_oss_decode_attention(module, q, k, v, mask, 0.25)
    assert weights is None and got.shape == (bsz, 1, heads, D) and got.dtype == dtype
    if dtype == torch.float64:
        torch.testing.assert_close(got, want, rtol = 1e-12, atol = 1e-12)
    else:
        torch.testing.assert_close(got, want)


def test_decode_attention_trims_a_longer_cache_to_the_mask(G):
    # Static caches can return more slots than the mask covers (_align_kv_to_mask): extra slots are dropped.
    q = torch.randn(1, 4, 1, 8, dtype = torch.float64)
    k = torch.randn(1, 2, 20, 8, dtype = torch.float64)
    v = torch.randn(1, 2, 20, 8, dtype = torch.float64)
    mask = torch.zeros(1, 1, 1, 12, dtype = torch.float64)
    module = _Attn(4, torch.zeros(4, dtype = torch.float64))
    got, _ = G._gpt_oss_decode_attention(module, q, k, v, mask, 0.5)
    want = _reference(module, q, k[:, :, :12], v[:, :, :12], mask, 0.5, 2)
    torch.testing.assert_close(got, want, rtol = 1e-12, atol = 1e-12)


def _decode_mask(G, monkeypatch, mask_function, q_offset, kv_len = 300):
    pytest.importorskip("torch.nn.attention.flex_attention")
    from torch.nn.attention.flex_attention import create_block_mask
    import unsloth_zoo.flex_attention.utils as U
    # CPU-only hosts leave the CUDA-sized flex helpers undefined; decode never calls this one.
    monkeypatch.setattr(U, "compiled_create_block_mask",
                        lambda m, b, h, M, N, device = "cpu": create_block_mask(m, b, h, M, N, device = device),
                        raising = False)
    padding = torch.ones(2, q_offset + 1, dtype = torch.long)
    padding[1, :4] = 0
    return G._gpt_oss_flex_mask(
        batch_size = 2, q_length = 1, kv_length = kv_len, q_offset = q_offset, kv_offset = 0,
        mask_function = mask_function, attention_mask = padding, device = "cpu",
    )


def test_flex_graph_contract_rebuilds_the_same_mask(G, monkeypatch):
    from torch.nn.attention.flex_attention import create_mask
    from transformers.masking_utils import causal_mask_function
    monkeypatch.setenv("UNSLOTH_GPT_OSS_GRAPH_DECODE_ATTENTION", "flex")
    live = torch.tensor(140)
    assert not hasattr(_decode_mask(G, monkeypatch, causal_mask_function, live), "_unsloth_graph_closure")
    with S.graph_generate():
        block_mask = _decode_mask(G, monkeypatch, causal_mask_function, live)
        tensors, rebuild = block_mask._unsloth_graph_closure
        assert all(isinstance(t, torch.Tensor) for t in tensors)
        copies = tuple(t.clone() for t in tensors)
        rebuilt = rebuild(copies)
        dense = create_mask(block_mask.mask_mod, 2, 1, 1, 300, device = "cpu")
        assert torch.equal(create_mask(rebuilt, 2, 1, 1, 300, device = "cpu"), dense)
        # The rebuilt mask_mod reads the given tensors: refilling them moves the query.
        copies[0].fill_(10)
        moved = create_mask(rebuilt, 2, 1, 1, 300, device = "cpu")
        assert not torch.equal(moved, dense) and int(moved[0, 0, 0].sum()) == 11


def test_flex_graph_contract_refuses_tensor_closures_and_dense_mode(G, monkeypatch):
    doc = torch.zeros(2, 300, dtype = torch.long)
    def packed(b, h, q, kv):
        return (kv <= q) & (doc[b, q] == doc[b, kv])
    from transformers.masking_utils import causal_mask_function
    with S.graph_generate():
        monkeypatch.setenv("UNSLOTH_GPT_OSS_GRAPH_DECODE_ATTENTION", "flex")
        assert not hasattr(_decode_mask(G, monkeypatch, packed, torch.tensor(140)), "_unsloth_graph_closure")
        monkeypatch.setenv("UNSLOTH_GPT_OSS_GRAPH_DECODE_ATTENTION", "dense")
        assert not hasattr(_decode_mask(G, monkeypatch, causal_mask_function, torch.tensor(140)),
                           "_unsloth_graph_closure")


def test_decode_attention_reads_per_head_masks(G):
    # A caller-built [B, H, 1, L] mask (different visible keys per head) must match the eager formula,
    # and shapes the grouped attention cannot read are left to the eager path.
    g = torch.Generator().manual_seed(7)
    bsz, heads, kv_heads, kv_len, D = 2, 8, 2, 40, 16
    q = torch.randn(bsz, heads, 1, D, generator = g, dtype = torch.float64)
    k = torch.randn(bsz, kv_heads, kv_len, D, generator = g, dtype = torch.float64)
    v = torch.randn(bsz, kv_heads, kv_len, D, generator = g, dtype = torch.float64)
    sinks = torch.randn(heads, generator = g, dtype = torch.float64)
    visible = torch.rand(bsz, heads, 1, kv_len, generator = g) > 0.3
    visible[..., 0] = True
    mask = torch.zeros(bsz, heads, 1, kv_len, dtype = torch.float64).masked_fill(~visible, torch.finfo(torch.float64).min)
    module = _Attn(heads, sinks)
    want = _reference(module, q, k, v, mask, 0.25, kv_heads)
    got, _ = G._gpt_oss_decode_attention(module, q, k, v, mask, 0.25)
    torch.testing.assert_close(got, want, rtol = 1e-12, atol = 1e-12)
    assert G._gpt_oss_dense_decode_mask_ok(None, heads)
    assert G._gpt_oss_dense_decode_mask_ok(mask, heads)
    assert G._gpt_oss_dense_decode_mask_ok(mask[:, :1], heads)
    assert not G._gpt_oss_dense_decode_mask_ok(mask[:, :2], heads)
    assert not G._gpt_oss_dense_decode_mask_ok(mask[:, :, 0], heads)
    assert not G._gpt_oss_dense_decode_mask_ok({"full_attention": mask}, heads)


def test_decode_attention_per_head_mask_with_one_kv_head(G):
    # Multi-query attention: one KV head, a [B, H, 1, L] mask with its own row per query head.
    g = torch.Generator().manual_seed(11)
    bsz, heads, kv_heads, kv_len, D = 2, 8, 1, 40, 16
    q = torch.randn(bsz, heads, 1, D, generator = g, dtype = torch.float64)
    k = torch.randn(bsz, kv_heads, kv_len, D, generator = g, dtype = torch.float64)
    v = torch.randn(bsz, kv_heads, kv_len, D, generator = g, dtype = torch.float64)
    visible = torch.rand(bsz, heads, 1, kv_len, generator = g) > 0.3
    visible[..., 0] = True
    for mask in (
        torch.zeros(bsz, heads, 1, kv_len, dtype = torch.float64).masked_fill(~visible, torch.finfo(torch.float64).min),
        torch.zeros(bsz, 1, 1, kv_len, dtype = torch.float64).masked_fill(~visible[:, :1], torch.finfo(torch.float64).min),
    ):
        module = _Attn(heads, torch.randn(heads, generator = g, dtype = torch.float64))
        want = _reference(module, q, k, v, mask, 0.25, kv_heads)
        got, _ = G._gpt_oss_decode_attention(module, q, k, v, mask, 0.25)
        torch.testing.assert_close(got, want, rtol = 1e-12, atol = 1e-12)


def test_flex_graph_contract_names_the_mask_function(G, monkeypatch):
    # The graph key sees the outer mask_mod's code, the same for every mask_function: the published identity is
    # what tells two windows (or causal from sliding) apart, and a mask_function holding anything but scalars
    # and functions (a dict of document ids, a partial, an object) publishes no contract at all.
    import functools
    from transformers.masking_utils import causal_mask_function, sliding_window_causal_mask_function
    monkeypatch.setenv("UNSLOTH_GPT_OSS_GRAPH_DECODE_ATTENTION", "flex")
    docs = {"doc": torch.zeros(2, 300, dtype = torch.long)}

    def packed(b, h, q, kv):
        return (kv <= q) & (docs["doc"][b, q] == docs["doc"][b, kv])

    def windowed(b, h, q, kv, window = 8):
        return (kv <= q) & (kv > q - window)

    with S.graph_generate():
        ident = lambda fn: getattr(_decode_mask(G, monkeypatch, fn, torch.tensor(140)), "_unsloth_mask_identity", None)
        causal = ident(causal_mask_function)
        w8, w8b, w16 = (ident(sliding_window_causal_mask_function(w)) for w in (8, 8, 16))
        assert causal is not None and w8 is not None
        assert w8 == w8b and w8 != w16 and w8 != causal
        assert all(type(x) in (bool, int, float, str) for x in w8)
        assert ident(packed) is None
        assert ident(functools.partial(windowed, window = 16)) is None
        assert ident(windowed) != ident(lambda b, h, q, kv, window = 16: (kv <= q) & (kv > q - window))
        for fn in (packed, functools.partial(windowed, window = 16)):
            assert not hasattr(_decode_mask(G, monkeypatch, fn, torch.tensor(140)), "_unsloth_graph_closure")


@pytest.mark.parametrize("dtype,autocast", [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.float16)])
def test_decode_attention_keeps_its_dtype_under_autocast(G, dtype, autocast):
    # The eager path writes the value matmul with out=, which autocast leaves alone: same dtype, same values.
    g = torch.Generator().manual_seed(3)
    q = torch.randn(2, 8, 1, 16, generator = g).to(dtype)
    k = torch.randn(2, 2, 24, 16, generator = g).to(dtype)
    v = torch.randn(2, 2, 24, 16, generator = g).to(dtype)
    module = _Attn(8, torch.randn(8, generator = g).to(dtype))
    want, _ = G._gpt_oss_decode_attention(module, q, k, v, None, 0.25)
    with torch.autocast("cpu", dtype = autocast):
        got, _ = G._gpt_oss_decode_attention(module, q, k, v, None, 0.25)
    assert got.dtype == dtype
    torch.testing.assert_close(got, want, rtol = 0, atol = 0)
    if dtype == torch.bfloat16:
        # Values past fp16's range stay finite (an fp16-autocast matmul would give inf).
        zq, zk = torch.zeros_like(q), torch.zeros_like(k)
        big = torch.full_like(v, 70000)
        with torch.autocast("cpu", dtype = autocast):
            out, _ = G._gpt_oss_decode_attention(_Attn(8, torch.zeros(8, dtype = dtype)), zq, zk, big, None, 0.25)
        assert torch.isfinite(out).all()


WINDOW = 8
_MASK_STATE = {"window": 8}


def _global_window(b, h, q, kv):
    return (kv <= q) & (kv > q - WINDOW)


def _global_dict(b, h, q, kv):
    return (kv <= q) & (kv > q - _MASK_STATE["window"])


def test_flex_graph_contract_names_module_globals(G, monkeypatch):
    # A mask_function reading module-level state: a scalar global is part of the identity (changing it changes
    # the graph key), one holding anything else publishes no contract.
    monkeypatch.setenv("UNSLOTH_GPT_OSS_GRAPH_DECODE_ATTENTION", "flex")
    global WINDOW
    with S.graph_generate():
        ident = lambda fn: getattr(_decode_mask(G, monkeypatch, fn, torch.tensor(140)), "_unsloth_mask_identity", None)
        first = ident(_global_window)
        assert first is not None and first == ident(_global_window)
        WINDOW = 16
        try:
            assert ident(_global_window) not in (None, first)
        finally:
            WINDOW = 8
        assert ident(_global_dict) is None


def test_flex_graph_contract_tells_same_line_lambdas_apart(G, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPT_OSS_GRAPH_DECODE_ATTENTION", "flex")
    a, b = (lambda b, h, q, kv: (kv <= q) & (kv > q - 8)), (lambda b, h, q, kv: (kv <= q) & (kv > q - 16))
    with S.graph_generate():
        ident = lambda fn: getattr(_decode_mask(G, monkeypatch, fn, torch.tensor(140)), "_unsloth_mask_identity", None)
        assert ident(a) is not None and ident(a) == ident(a) and ident(a) != ident(b)


class _NoHostRead(torch.Tensor):
    # A device offset stand-in: reading it on the host (a sync on CUDA, an error inside a capture) raises.
    def __int__(self):
        raise AssertionError("kv_offset read on the host")

    def item(self):
        raise AssertionError("kv_offset read on the host")


def test_flex_mask_never_reads_a_tensor_kv_offset_on_the_host(G, monkeypatch):
    # transformers passes an int kv_offset today (every cache layer's get_mask_sizes); a tensor one must not be
    # turned into a host int to size the padding, and the mask must be the int offset's mask.
    from torch.nn.attention.flex_attention import create_block_mask, create_mask
    import unsloth_zoo.flex_attention.utils as U
    from transformers.masking_utils import causal_mask_function
    monkeypatch.setattr(U, "compiled_create_block_mask",
                        lambda m, b, h, M, N, device = "cpu": create_block_mask(m, b, h, M, N, device = device),
                        raising = False)
    padding = torch.ones(2, 141, dtype = torch.long)
    padding[1, :4] = 0
    kw = dict(batch_size = 2, q_length = 1, kv_length = 300, q_offset = torch.tensor(140), mask_function = causal_mask_function,
              attention_mask = padding, device = "cpu")
    want = create_mask(G._gpt_oss_flex_mask(kv_offset = 0, **kw).mask_mod, 2, 1, 1, 300, device = "cpu")
    offset = torch.tensor(0).as_subclass(_NoHostRead)
    got = create_mask(G._gpt_oss_flex_mask(kv_offset = offset, **kw).mask_mod, 2, 1, 1, 300, device = "cpu")
    assert torch.equal(got, want)


def _visible_through_blocks(block_mask, kv_len):
    # What flex attends to: keys inside the listed blocks that mask_mod keeps.
    from torch.nn.attention.flex_attention import create_mask
    rows = block_mask.kv_num_blocks.shape[0]
    dense = create_mask(block_mask.mask_mod, rows, 1, 1, kv_len, device = "cpu")[:, 0, 0]
    block = block_mask.BLOCK_SIZE[1]
    listed = torch.zeros(rows, kv_len, dtype = torch.bool)
    for b in range(rows):
        for i in block_mask.kv_indices[b, 0, 0, : int(block_mask.kv_num_blocks[b, 0, 0])].tolist():
            listed[b, i * block : (i + 1) * block] = True
    return dense & listed


@pytest.mark.parametrize("window", [None, 200])
def test_flex_graph_contract_refill_moves_across_blocks(G, monkeypatch, window):
    # A graph captured at position p replays later steps by refilling the BlockMask's tensor fields (block counts
    # included) and the closure tensors, as unsloth's generic decode graphs do; the replayed mask must list the new
    # blocks and match causal + window + padding visibility at p, p + BLOCK and p + 3 * BLOCK.
    import copy
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as BLOCK
    from transformers.masking_utils import causal_mask_function, sliding_window_causal_mask_function
    monkeypatch.setenv("UNSLOTH_GPT_OSS_GRAPH_DECODE_ATTENTION", "flex")
    fn = causal_mask_function if window is None else sliding_window_causal_mask_function(window)
    kv_len, p = 4 * BLOCK + 40, 20

    def mask_at(q):
        return _decode_mask(G, monkeypatch, fn, torch.tensor(q), kv_len = kv_len)

    def expected(q):
        j = torch.arange(kv_len)
        pad = torch.zeros(2, kv_len, dtype = torch.bool)
        pad[:, : q + 1] = True
        pad[1, :4] = False
        keep = (j <= q) & ((j > q - window) if window is not None else True)
        return pad & keep

    with S.graph_generate():
        first = mask_at(p)
        fields = {k: v.clone() for k, v in vars(first).items() if isinstance(v, torch.Tensor)}
        static = copy.copy(first)
        for k, v in fields.items():
            setattr(static, k, v)
        tensors, rebuild = first._unsloth_graph_closure
        closure = tuple(t.clone() for t in tensors)
        static.mask_mod = rebuild(closure)
        assert "kv_num_blocks" in fields, sorted(fields)
        assert torch.equal(_visible_through_blocks(static, kv_len), expected(p))
        counts = []
        for q in (p + BLOCK, p + 3 * BLOCK):
            new = mask_at(q)
            for k, v in fields.items():
                v.copy_(getattr(new, k))
            for dst, src in zip(closure, new._unsloth_graph_closure[0]):
                dst.copy_(src)
            assert torch.equal(_visible_through_blocks(static, kv_len), expected(q)), q
            counts.append(int(static.kv_num_blocks.max()))
        assert counts == [2, 4], counts


def test_auto_graph_decode_falls_back_to_the_dense_mask(G, monkeypatch):
    # "auto": stock masks go flex with a graph contract; a mask function the graph cannot key comes back as the
    # dense eager mask, which the dense decode attention reads.
    from transformers.masking_utils import causal_mask_function, eager_mask
    monkeypatch.setenv("UNSLOTH_GPT_OSS_GRAPH_DECODE_ATTENTION", "auto")
    holder = {"w": 8}

    def custom(b, h, q, kv):
        return (kv <= q) & (kv > q - holder["w"])

    one = torch.zeros(2, 1, 8)
    assert not G._gpt_oss_graph_dense_attention(one)
    with S.graph_generate():
        stock = _decode_mask(G, monkeypatch, causal_mask_function, torch.tensor(140))
        assert hasattr(stock, "_unsloth_graph_closure")
        dense = _decode_mask(G, monkeypatch, custom, torch.tensor(140))
        assert isinstance(dense, torch.Tensor) and dense.shape == (2, 1, 1, 300)
        padding = torch.ones(2, 141, dtype = torch.long)
        padding[1, :4] = 0
        common = dict(batch_size = 2, kv_length = 300, kv_offset = 0, mask_function = custom,
                      attention_mask = padding.bool())
        if "q_length" in eager_mask.__code__.co_varnames:
            want = eager_mask(q_length = 1, q_offset = torch.tensor(140), device = "cpu", **common)
        else:
            want = eager_mask(cache_position = torch.tensor([140]), **common)
        assert torch.equal(dense, want)
        assert G._gpt_oss_dense_decode_mask_ok(dense, 8)
        assert G._gpt_oss_graph_dense_attention(one) and not G._gpt_oss_dense_graph_decode(one)
        monkeypatch.setenv("UNSLOTH_GPT_OSS_GRAPH_DECODE_ATTENTION", "flex")
        assert not G._gpt_oss_graph_dense_attention(one)
        monkeypatch.setenv("UNSLOTH_GPT_OSS_GRAPH_DECODE_ATTENTION", "dense")
        assert G._gpt_oss_graph_dense_attention(one) and G._gpt_oss_dense_graph_decode(one)
