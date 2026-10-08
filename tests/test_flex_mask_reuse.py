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

# flex_attention_with_sink reuses one BlockMask per (mask_mod, shape, device) for stateless masks.
from __future__ import annotations

import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "flex attention needs CUDA")


def _mods():
    from unsloth_zoo.flex_attention import utils, attention_sink
    if not utils.HAS_FLEX_ATTENTION:
        pytest.skip("no flex attention")
    return utils, attention_sink


@pytest.fixture
def builds(monkeypatch):
    utils, _ = _mods()
    utils.clear_block_mask_cache()
    calls = []
    real = utils.compiled_create_block_mask
    def counted(*args, **kwargs):
        calls.append(args[1:5])
        return real(*args, **kwargs)
    monkeypatch.setattr(utils, "compiled_create_block_mask", counted)
    from unsloth_zoo.flex_attention import attention_sink
    monkeypatch.setattr(attention_sink, "compiled_create_block_mask", counted)
    yield calls
    utils.clear_block_mask_cache()


def _attn(sliding_window, heads = 4, training = True):
    torch.manual_seed(0)
    return types.SimpleNamespace(
        sinks = torch.randn(heads, device = "cuda", requires_grad = True),
        num_key_value_groups = 2, scaling = 0.125, sliding_window = sliding_window, training = training,
    )


def _qkv(bsz = 2, seq = 256, heads = 4, kv_heads = 2, dim = 64, seed = 1):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    q = torch.randn(bsz, heads,    seq, dim, device = "cuda", dtype = torch.bfloat16, generator = g)
    k = torch.randn(bsz, kv_heads, seq, dim, device = "cuda", dtype = torch.bfloat16, generator = g)
    v = torch.randn(bsz, kv_heads, seq, dim, device = "cuda", dtype = torch.bfloat16, generator = g)
    return [x.requires_grad_() for x in (q, k, v)]


def _run(layers, seed = 1, seq = 256):
    _, sink = _mods()
    outs, grads = [], []
    for attn in layers:
        q, k, v = _qkv(seed = seed, seq = seq)
        out = sink.flex_attention_with_sink(attn, q, k, v)
        attn.sinks.grad = None
        out.float().square().sum().backward()
        outs.append(out.detach())
        grads.append([x.grad.clone() for x in (q, k, v, attn.sinks)])
    return outs, grads


def test_reuse_bitwise_and_one_build_per_kind(builds, monkeypatch):
    layers = [_attn(128), _attn(None), _attn(128), _attn(None)]
    monkeypatch.setenv("UNSLOTH_FLEX_MASK_REUSE", "0")
    ref_out, ref_grads = _run(layers)
    assert len(builds) == 4
    builds.clear()
    monkeypatch.setenv("UNSLOTH_FLEX_MASK_REUSE", "1")
    out, grads = _run(layers)
    out2, _ = _run(layers)  # second step reuses too
    assert len(builds) == 2
    for a, b in zip(ref_out, out): assert torch.equal(a, b)
    for a, b in zip(ref_out, out2): assert torch.equal(a, b)
    for ga, gb in zip(ref_grads, grads):
        for a, b in zip(ga, gb): assert torch.equal(a, b)


def test_new_shape_builds_new_mask(builds):
    utils, _ = _mods()
    _run([_attn(128)], seq = 256)
    _run([_attn(128)], seq = 384)
    _run([_attn(128)], seq = 256)
    assert [c[2] for c in builds] == [256, 384]


def test_inference_mode_mask_kept_apart(builds):
    _, sink = _mods()
    attn = _attn(128, training = False)
    with torch.inference_mode():
        q, k, v = [x.detach() for x in _qkv()]
        sink.flex_attention_with_sink(attn, q, k, v)
    attn.training = True
    _run([attn])  # must not reuse an inference tensor mask for backward
    assert len(builds) == 2


def test_cache_bounded(builds):
    utils, _ = _mods()
    for n in range(utils._BLOCK_MASK_CACHE_SIZE + 4):
        utils.reused_compiled_create_block_mask(utils.causal_mask, 1, 1, 128 * (n + 1), 128 * (n + 1), device = "cuda")
    assert len(utils._BLOCK_MASK_CACHE) == utils._BLOCK_MASK_CACHE_SIZE


def test_padding_prefill_mask_not_cached(builds):
    # Left padding prefill closes over padding_start_idx, so it must stay per call
    _, sink = _mods()
    attn = _attn(None, training = False)
    q, k, v = [x.detach() for x in _qkv()]
    mask = torch.ones(2, 256, dtype = torch.long, device = "cuda")
    mask[0, :17] = 0
    with torch.no_grad():
        sink.flex_attention_with_sink(attn, q.clone(), k.clone(), v.clone(), attention_mask = mask)
        sink.flex_attention_with_sink(attn, q.clone(), k.clone(), v.clone(), attention_mask = mask)
    utils, _ = _mods()
    assert len(utils._BLOCK_MASK_CACHE) == 0
