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


# Official gpt-oss attention (openai/gpt-oss gpt_oss/torch/model.py `sdpa`), verbatim apart from the mask dtype.
def official_gpt_oss_sdpa(Q, K, V, S, sm_scale, sliding_window = 0):
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


def visible(mask_2d, window, kv_offset = 0, kv_length = None):
    """[B, kv_length] the rule every decode mask must follow for the query at mask_2d's last column: causal, within W
    tokens including the query, a real (written, unpadded) token."""
    q = mask_2d.shape[1] - 1
    kv = kv_offset + torch.arange(kv_length or q + 1)
    ok = (kv <= q) & torch.nn.functional.pad(mask_2d.bool(), (0, max(0, int(kv[-1]) - q)))[:, kv]
    return ok & (kv > q - window) if window else ok


def _pads(S_len, *pads):
    mask_2d = torch.ones(len(pads), S_len, dtype = torch.long)
    for b, p in enumerate(pads):
        mask_2d[b, :p] = 0
    return mask_2d


def _qkv(B, Hq, Hkv, S_len, d, seed):
    g = torch.Generator().manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, generator = g, dtype = torch.float64)
    return rand, rand(B, Hq, 1, d), rand(B, Hkv, S_len, d), rand(B, Hkv, S_len, d)


class _Attn(torch.nn.Module):
    num_key_value_groups = 2
    is_causal = True


def _close(got, want):
    torch.testing.assert_close(got, want, rtol = 1e-10, atol = 1e-12)


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
def test_mask_follows_the_visibility_rule_and_matches_transformers_mask_functions(window):
    from transformers.masking_utils import sliding_window_causal_mask_function, causal_mask_function
    fn = sliding_window_causal_mask_function(window) if window else causal_mask_function
    pads = [0, 37, 170]
    rows = len(pads)
    for q in [0, 5, 127, 128, 129, 200, 255, 256, 300, 400]:
        kv_length, kv_offset = (window, max(0, q - window + 1)) if window else (512, 0)
        mask_2d = _pads(q + 1, *pads)
        bm = D.decode_block_mask(
            rows, kv_length, torch.tensor(q), torch.tensor(kv_offset), attention_mask = mask_2d, window = window,
            device = "cpu",
        )
        want = visible(mask_2d, window, kv_offset, kv_length)
        assert torch.equal(effective_mask(bm, rows, kv_length), want), (q, window)
        theirs = D.decode_block_mask(rows, kv_length, torch.tensor(q), torch.tensor(kv_offset), mask_function = fn, device = "cpu")
        assert torch.equal(effective_mask(theirs, rows, kv_length), want[:1].expand(rows, -1)), (q, window)


@pytest.mark.parametrize("window", [0, 16])
@pytest.mark.parametrize("sink_scale", [1.0, 30.0])
def test_sinks_match_the_official_gpt_oss_attention(window, sink_scale):
    n_tokens, n_kv, q_mult, d = 40, 2, 4, 16
    g = torch.Generator().manual_seed(3 + window)
    Q, K, V = (torch.randn(n_tokens, n_kv, *s, d, generator = g, dtype = torch.float64) for s in ((q_mult,), (), ()))
    S = torch.randn(n_kv * q_mult, generator = g, dtype = torch.float64) * sink_scale
    want = official_gpt_oss_sdpa(Q, K, V, S, d ** -0.5, window)[-1].view(n_kv * q_mult, d)
    bm = D.decode_block_mask(1, n_tokens, torch.tensor(n_tokens - 1), window = window, device = "cpu")
    got = D.flex_decode_attention(Q[-1].reshape(1, -1, 1, d), K.permute(1, 0, 2)[None], V.permute(1, 0, 2)[None], bm, d ** -0.5, sinks = S)
    _close(got[0, :, 0, :], want)


@pytest.mark.parametrize("window", [0, 128])
def test_sinks_unequal_heads_left_padding_and_an_all_masked_row_match_the_reference(window):
    d, S_len = 32, 200
    _, q, k, v = _qkv(4, 8, 2, S_len, d, seed = 7)
    k = k * 3
    mask_2d = _pads(S_len, 0, 90, 199, S_len)  # row 3 is all padding
    bm = D.decode_block_mask(4, S_len, torch.tensor(S_len - 1), attention_mask = mask_2d, window = window, device = "cpu")
    for sinks in (None, torch.linspace(-20, 20, 8, dtype = torch.float64)):
        out = D.flex_decode_attention(q, k, v, bm, d ** -0.5, sinks = sinks)
        assert torch.isfinite(out).all()
        assert torch.equal(out[3], torch.zeros_like(out[3]))
        _close(out, D._eager_attention(q, k, v, visible(mask_2d, window)[:, None, None, :], d ** -0.5, sinks = sinks))


def test_softcap_never_leaks_through_masked_keys():
    # Masked keys get enormous scores: capping them must not make them visible (cap first, then mask).
    d, S_len, cap = 16, 96, 30.0
    _, q, k, v = _qkv(1, 4, 4, S_len, d, seed = 11)
    k[:, :, :50] = q[:, :, :1] * 1e4  # left padding with dominant scores
    bm = D.decode_block_mask(1, S_len, torch.tensor(S_len - 1), attention_mask = _pads(S_len, 50), device = "cpu")
    out = D.flex_decode_attention(q, k, v, bm, d ** -0.5, softcap = cap)
    scores = (q @ k[..., 50:, :].transpose(-1, -2)) * d ** -0.5
    want = torch.softmax(cap * torch.tanh(scores / cap), dim = -1) @ v[..., 50:, :]
    _close(out, want)


@pytest.mark.parametrize("window", [0, 128])
def test_one_block_mask_refilled_for_300_steps_equals_a_fresh_mask_each_step(window):
    rows, d, start = 2, 16, 20
    kv_length = window or 512
    rand, _, k, v = _qkv(rows, 4, 2, kv_length, d, seed = 5)
    mask_2d = _pads(start + 300, 0, 13)

    def fresh_at(t):
        q = start + t
        kv_offset = max(0, q - window + 1) if window else 0
        return D.decode_block_mask(rows, kv_length, torch.tensor(q), torch.tensor(kv_offset),
                                   attention_mask = mask_2d[:, :q + 1], window = window, device = "cpu")

    first = fresh_at(0)
    tensors, rebuild = first._unsloth_graph_closure
    fields = (first.kv_num_blocks.clone(), first.kv_indices.clone())
    closure = tuple(t.clone() for t in tensors)
    captured = flex.BlockMask.from_kv_blocks(
        *fields, BLOCK_SIZE = first.BLOCK_SIZE, mask_mod = rebuild(closure), seq_lengths = first.seq_lengths,
    )
    for t in range(300):
        fresh = fresh_at(t)
        assert fresh._unsloth_mask_identity == first._unsloth_mask_identity
        # What CUDA-graph decode does each step: copy the new mask's tensors into the captured mask's buffers.
        for a, b in zip(fields + closure, (fresh.kv_num_blocks, fresh.kv_indices, *fresh._unsloth_graph_closure[0])):
            a.copy_(b)
        q = rand(rows, 4, 1, d)
        torch.testing.assert_close(
            D.flex_decode_attention(q, k, v, captured, d ** -0.5), D.flex_decode_attention(q, k, v, fresh, d ** -0.5),
            rtol = 0, atol = 0,
        )


def test_graph_identity_follows_window_and_captured_scalars_and_offsets_are_copied():
    ident = lambda **kw: getattr(D.decode_block_mask(1, 256, torch.tensor(100), device = "cpu", **kw), "_unsloth_mask_identity", None)
    assert ident(window = 128) is not None
    assert ident(window = 128) == ident(window = 128)
    assert ident(window = 128) != ident(window = 64)
    assert ident() != ident(window = 128)
    table = torch.ones(4, dtype = torch.bool)
    bm = D.decode_block_mask(1, 256, torch.tensor(100), mask_function = lambda b, h, q, kv: table[b] & (kv <= q), device = "cpu")
    assert getattr(bm, "_unsloth_mask_identity", None) is None
    assert getattr(bm, "_unsloth_graph_closure", None) is None
    plain = D.decode_block_mask(1, 256, torch.tensor(100), window = 128, device = "cpu", graph_contract = False)
    assert getattr(plain, "_unsloth_graph_closure", None) is None
    # The offsets are copies of a live cache counter.
    counter = torch.tensor(99)
    bm = D.decode_block_mask(1, 256, counter, device = "cpu")
    before = effective_mask(bm, 1, 256).clone()
    counter.add_(50)
    assert torch.equal(effective_mask(bm, 1, 256), before)


@pytest.mark.parametrize("compile_fails", [False, True])
def test_unsloth_attention_serves_the_same_visibility_densely_when_flex_decoding_cannot(compile_fails, monkeypatch):
    # Unsupported shape, or a failing flex compile that must raise once and then be served densely.
    d, S_len, window = 16, 150, (0 if compile_fails else 128)
    _, q, k, v = _qkv(2, 4, 2, S_len, d, seed = 3 if compile_fails else 2)
    mask_2d = _pads(S_len, 0, 30) if compile_fails else _pads(S_len, 40, 0)
    bm = D.decode_block_mask(2, S_len, torch.tensor(S_len - 1), attention_mask = mask_2d, window = window, device = "cpu")
    calls = []
    if compile_fails:
        def failing_compiled_flex(*args, **kwargs):
            calls.append(torch._dynamo.config.suppress_errors)
            raise RuntimeError("No valid triton configs. OutOfMemoryError: out of resource")
        monkeypatch.setattr(D, "_flex_function", lambda device: (failing_compiled_flex, True))
        monkeypatch.setattr(D, "flex_decode_supported", lambda *args: True)
        monkeypatch.setattr(D, "_FAILED_SHAPES", set())
        monkeypatch.setattr(D, "_COMPILED_SHAPES", set())
        monkeypatch.setattr(torch._dynamo.config, "suppress_errors", True)
    else:
        assert not D.flex_decode_supported(d, 4, 2, torch.float64, "cpu")
    want = D._eager_attention(q, k, v, visible(mask_2d, window)[:, None, None, :], d ** -0.5)
    module = _Attn()
    for _ in range(3 if compile_fails else 1):
        out, _ = D.unsloth_flex_decode_attention(module, q, k, v, bm, scaling = d ** -0.5)
        _close(out, want.transpose(1, 2))
    if compile_fails:
        assert calls == [False]
        with pytest.raises(D.FlexDecodeUnavailable):
            D.flex_decode_attention(q, k, v, bm, d ** -0.5)
        assert calls == [False]


def test_flex_decode_support_rules_and_kv_tile_fits_shared_memory(monkeypatch):
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
    # T4: head_dim 256 needs 32 rows; 512 is unsupported.
    assert D._kv_tile(64, (7, 5)) is None and D._kv_tile(128, (7, 5)) is None and D._kv_tile(256, (7, 5)) == 32
    assert D._flex_decode_hd_limit((7, 5)) == 256 and D._flex_decode_hd_limit((8, 0)) == 512
    assert D._kv_tile(256, (8, 0)) is None and D._kv_tile(512, (8, 0)) == 32 and D._kv_tile(512, (8, 9)) == 32
    monkeypatch.setattr(D, "_device_capability", lambda index: (7, 5))
    options = D.flex_decode_kernel_options(256, torch.device("cuda", 0))
    assert options["BLOCK_N"] == 32 and options["num_stages"] == 1


def test_registration_adds_a_custom_key_and_installs_the_device_transfer(monkeypatch):
    from transformers.modeling_utils import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, causal_mask_function
    assert D.register_flex_decode()
    assert AttentionInterface._global_mapping[D.FLEX_DECODE_KEY] is D.unsloth_flex_decode_attention
    assert AttentionMaskInterface._global_mapping[D.FLEX_DECODE_KEY] is D.unsloth_flex_decode_mask
    mask = D.unsloth_flex_decode_mask(
        batch_size = 1, cache_position = torch.arange(4), kv_length = 4, q_length = 4, q_offset = 0, kv_offset = 0,
        mask_function = causal_mask_function, attention_mask = None, device = torch.device("cpu"),
        allow_is_causal_skip = False,
    )
    assert not isinstance(mask, flex.BlockMask)
    # BlockMask.to(device) must rebuild a decode mask for that device.
    monkeypatch.setattr(flex.BlockMask, "to", flex.BlockMask.to)
    assert D.install_block_mask_transfer()
    mask = D.decode_block_mask(1, 16, torch.tensor(3))
    rebuilt = []
    mask._unsloth_on_device = lambda device: rebuilt.append(torch.device(device)) or "rebuilt"
    assert mask.to("cpu") == "rebuilt" and rebuilt == [torch.device("cpu")]
    assert flex.BlockMask.to._unsloth_gpt_oss
    assert mask.contiguous() is mask


class _HostTensorSpy(torch.utils._python_dispatch.TorchDispatchMode):
    """Records every op that reads or makes a CPU tensor. With the work on the meta device, any hit is a tensor built
    on the host or copied from it: under CUDA graph capture that is a pageable host-to-device copy, which raises."""

    def __enter__(self):
        self.hits = []
        return super().__enter__()

    def __torch_dispatch__(self, func, types, args = (), kwargs = None):
        out = func(*args, **(kwargs or {}))
        flat, _ = torch.utils._pytree.tree_flatten((args, kwargs, out))
        if any(isinstance(x, torch.Tensor) and x.device.type == "cpu" for x in flat):
            self.hits.append(str(func))
        return out


@pytest.mark.parametrize("window", [0, 128])
def test_per_step_decode_path_never_touches_host_tensors(window, monkeypatch):
    # Every non-kernel op of a decode step on the meta device: no op may see a CPU tensor.
    from types import SimpleNamespace
    dev = torch.device("meta")
    B, Hq, Hkv, d, S_len = 2, 4, 2, 64, 300
    q, sinks = torch.empty(B, Hq, 1, d, device = dev), torch.empty(Hq, device = dev)
    k, v = torch.empty(B, Hkv, S_len, d, device = dev), torch.empty(B, Hkv, S_len, d, device = dev)
    mask_2d = torch.ones(B, S_len, dtype = torch.long, device = dev)

    def fake_compiled_flex(query, key, value, **kwargs):
        lse = torch.empty(query.shape[:-1], device = query.device)
        return query.new_empty(query.shape), (SimpleNamespace(lse = lse) if "return_aux" in D._LSE_KWARGS else lse)
    monkeypatch.setattr(D, "_flex_function", lambda device: (fake_compiled_flex, True))
    with _HostTensorSpy() as spy:
        for step in range(3):
            q_off = torch.full((), S_len - 1 - step, dtype = torch.int64, device = dev)
            bm = D.decode_block_mask(B, S_len, q_off, attention_mask = mask_2d, window = window, device = dev)
            tensors, rebuild = bm._unsloth_graph_closure
            refilled = flex.BlockMask.from_kv_blocks(
                bm.kv_num_blocks, bm.kv_indices, BLOCK_SIZE = bm.BLOCK_SIZE, mask_mod = rebuild(tensors),
                seq_lengths = bm.seq_lengths,
            )
            dense = D.dense_decode_mask(refilled, S_len, dev)
            D._eager_attention(q, k, v, dense, d ** -0.5, sinks = sinks, softcap = 30.0)
            D.flex_decode_attention(q, k, v, bm, d ** -0.5, sinks = sinks, kernel_options = {})
    assert spy.hits == []
    with _HostTensorSpy() as spy:
        torch.tensor(S_len - 1).to(dev)
    assert spy.hits


def test_padding_layout_is_stable_as_the_mask_grows_and_the_window_rolls():
    # The padding must keep shape, strides and dtype (graph key) as the 2D mask widens each step.
    layouts = []
    for step in range(4):
        width = 300 + step
        mask = _pads(width, 0, 5)
        for offset in (0, width - 128, torch.tensor(width - 128)):
            padding = D.decode_padding(mask, 128, offset)
            layouts.append((tuple(padding.shape), padding.stride(), padding.dtype))
            expected = torch.nn.functional.pad(mask.bool(), (0, 128))[:, int(offset) : int(offset) + 128]
            assert torch.equal(padding, expected)
    assert len(set(layouts)) == 1, set(layouts)


@pytest.mark.parametrize("which", ["sinks", "softcap"])
def test_reference_fallback_keeps_an_elided_causal_mask_and_attention_dropout(which):
    # The float32 reference must apply the causal mask SDPA's is_causal implies.
    module = _Attn()
    extra = {"s_aux": torch.full((1,), -1e4)} if which == "sinks" else {"softcap": 30.0}
    q = k = torch.zeros(1, 1, 3, 4)
    v = torch.arange(3.0).view(1, 1, 3, 1).expand(1, 1, 3, 4).contiguous()
    out, _ = D.unsloth_flex_decode_attention(module, q, k, v, None, scaling = 1.0, **extra)
    torch.testing.assert_close(out[0, :, 0, 0], torch.tensor([0.0, 0.5, 1.0]))
    k, v = torch.zeros(1, 1, 8, 4), torch.zeros(1, 1, 8, 4)
    v[:, :, :3] = torch.arange(3.0).view(1, 1, 3, 1)
    out, _ = D.unsloth_flex_decode_attention(module, q, k, v, None, scaling = 1.0, **extra)
    torch.testing.assert_close(out[0, :, 0, 0], torch.tensor([0.0, 0.5, 1.0]))
    torch.testing.assert_close(out.transpose(1, 2), torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal = True))

    # Training with sinks / softcap keeps attention dropout; eval ignores it.
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)
    extra = {"s_aux": torch.zeros(2)} if which == "sinks" else extra
    module.eval()
    attend = lambda p: D.unsloth_flex_decode_attention(module, q, k, v, None, scaling = 0.5, dropout = p, **extra)[0]
    clean, again = attend(0.0), attend(0.5)
    module.train()
    torch.manual_seed(1)
    dropped = attend(0.5)
    torch.manual_seed(1)
    mask = torch.ones(4, 4, dtype = torch.bool).tril()
    expected = D._eager_attention(q, k, v, mask, 0.5, sinks = extra.get("s_aux"), softcap = extra.get("softcap"), dropout = 0.5)
    torch.testing.assert_close(dropped, expected.transpose(1, 2))
    assert not torch.allclose(dropped, again)
    torch.testing.assert_close(clean, again)


def test_a_new_batch_after_a_compiled_one_still_gets_failure_handling(monkeypatch):
    # A success at one batch size must not send a later specialisation (batch is static) down the unguarded path.
    calls = []

    def compiled_flex(query, key, value, **kwargs):
        calls.append(query.shape[0])
        if query.shape[0] == 2:
            raise RuntimeError("No valid triton configs")
        lse = torch.zeros(query.shape[:-1], dtype = torch.float32)
        return torch.zeros_like(query), lse

    monkeypatch.setattr(D, "_flex_function", lambda device: (compiled_flex, True))
    monkeypatch.setattr(D, "_FAILED_SHAPES", set())
    monkeypatch.setattr(D, "_COMPILED_SHAPES", set())
    monkeypatch.setattr(D, "_output_and_lse", lambda result: result)
    for batch, fails in ((1, False), (2, True)):
        _, q, k, v = _qkv(batch, 4, 2, 64, 16, seed = 0)
        bm = D.decode_block_mask(batch, 64, torch.tensor(63), device = "cpu")
        if fails:
            with pytest.raises(D.FlexDecodeUnavailable):
                D.flex_decode_attention(q, k, v, bm, 0.25)
        else:
            D.flex_decode_attention(q, k, v, bm, 0.25)
    assert calls == [1, 2]


def test_block_starts_are_built_once_per_shape():
    first = D._block_starts(8, 128, "cpu")
    assert D._block_starts(8, 128, torch.device("cpu")) is first
    assert first[0] == torch.iinfo(torch.int64).min and first[1:].tolist() == [128 * i for i in range(1, 8)]


@pytest.mark.parametrize("which", ["sinks", "softcap"])
def test_reference_fallback_keeps_finite_additive_mask_penalties(which):
    # A custom float mask reaches the fallback as given: a finite -1e9 must still hide its key.
    module = torch.nn.Module().eval()
    q = torch.zeros(1, 1, 1, 4)
    k = torch.zeros(1, 1, 2, 4)
    v = torch.tensor([0.0, 10.0]).view(1, 1, 2, 1).expand(1, 1, 2, 4).contiguous()
    mask = torch.tensor([0.0, -1e9]).view(1, 1, 1, 2)
    extra = {"s_aux": torch.full((1,), -1e4)} if which == "sinks" else {"softcap": 30.0}
    out, _ = D.unsloth_flex_decode_attention(module, q, k, v, mask, scaling = 1.0, **extra)
    torch.testing.assert_close(out[0, 0, 0], torch.zeros(4))
    full = torch.full((1, 1, 1, 2), torch.finfo(torch.float32).min)
    out, _ = D.unsloth_flex_decode_attention(module, q, k, v, full, scaling = 1.0, **extra)
    torch.testing.assert_close(out[0, 0, 0], torch.zeros(4))
