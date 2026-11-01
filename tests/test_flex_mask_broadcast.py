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

# flex_attention_with_sink builds its block masks with only the dims its mask_mod reads:
# B = H = None for causal / sliding window, B = bsz, H = None for left padding.
from __future__ import annotations

import ast
import inspect
import textwrap
import types

import pytest
import torch


def _mods():
    # attention_sink imports create_block_mask from utils, which exists only when torch has flex attention, so check utils.HAS_FLEX_ATTENTION first
    from unsloth_zoo.flex_attention import utils
    if not utils.HAS_FLEX_ATTENTION:
        pytest.skip("no flex attention")
    from unsloth_zoo.flex_attention import attention_sink
    return utils, attention_sink


def _names_read(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    func = tree.body[0]
    params = [a.arg for a in func.args.args]
    used = {n.id for n in ast.walk(func) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return params, used


def test_mask_mods_read_only_the_dims_they_are_built_with():
    utils, _ = _mods()
    idx = torch.tensor([3, 0])
    stateless = [utils.causal_mask, utils.generate_sliding_window_mask(128)]
    padded = [utils.generate_causal_mask_with_padding(idx), utils.generate_sliding_window_mask_with_padding(128, idx)]
    for fn in stateless:
        (b, h, *_), used = _names_read(fn)
        assert b not in used and h not in used, fn.__name__
    for fn in padded:
        (b, h, *_), used = _names_read(fn)
        assert b in used and h not in used, fn.__name__


def test_mask_mods_ignore_batch_and_head_values():
    utils, _ = _mods()
    q = torch.arange(300)[:, None]
    kv = torch.arange(300)[None, :]
    idx = torch.tensor([3, 0, 17])
    for fn in (utils.causal_mask, utils.generate_sliding_window_mask(128)):
        ref = fn(torch.tensor(0), torch.tensor(0), q, kv)
        for b, h in ((1, 5), (7, 63), (2, 0)):
            assert torch.equal(fn(torch.tensor(b), torch.tensor(h), q, kv), ref)
    for fn in (utils.generate_causal_mask_with_padding(idx), utils.generate_sliding_window_mask_with_padding(128, idx)):
        for b in range(3):
            ref = fn(torch.tensor(b), torch.tensor(0), q, kv)
            for h in (1, 5, 63):
                assert torch.equal(fn(torch.tensor(b), torch.tensor(h), q, kv), ref)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "flex attention needs CUDA")


@pytest.fixture
def builds(monkeypatch):
    utils, attention_sink = _mods()
    utils.clear_block_mask_cache()
    calls = []
    real = utils.compiled_create_block_mask
    def counted(*args, **kwargs):
        calls.append(args[1:3])
        return real(*args, **kwargs)
    monkeypatch.setattr(utils, "compiled_create_block_mask", counted)
    monkeypatch.setattr(attention_sink, "compiled_create_block_mask", counted)
    yield calls
    utils.clear_block_mask_cache()


def _attn(sliding_window, heads = 8, training = True):
    torch.manual_seed(0)
    return types.SimpleNamespace(
        sinks = torch.randn(heads, device = "cuda", requires_grad = True),
        num_key_value_groups = 4, scaling = 0.125, sliding_window = sliding_window, training = training,
    )


def _qkv(bsz = 2, seq = 384, heads = 8, kv_heads = 2, dim = 64, seed = 1):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    q = torch.randn(bsz, heads,    seq, dim, device = "cuda", dtype = torch.bfloat16, generator = g)
    k = torch.randn(bsz, kv_heads, seq, dim, device = "cuda", dtype = torch.bfloat16, generator = g)
    v = torch.randn(bsz, kv_heads, seq, dim, device = "cuda", dtype = torch.bfloat16, generator = g)
    return [x.requires_grad_() for x in (q, k, v)]


def _step(attn, **kwargs):
    _, sink = _mods()
    q, k, v = _qkv()
    out = sink.flex_attention_with_sink(attn, q, k, v, **kwargs)
    attn.sinks.grad = None
    out.float().square().sum().backward()
    return [out.detach()] + [x.grad.clone() for x in (q, k, v, attn.sinks)]


@cuda
@pytest.mark.parametrize("sliding_window", [None, 128])
def test_training_mask_is_broadcast_and_bitwise(builds, monkeypatch, sliding_window):
    utils, sink = _mods()
    attn = _attn(sliding_window)
    got = _step(attn)
    assert builds == [(None, None)]
    # Reference: the same mask_mod materialised for every batch row and head.
    real = utils.compiled_create_block_mask
    def full_dims(mask_mod, bsz, head, M, N, device = "cuda"):
        return real(mask_mod, 2, 8, M, N, device = device)
    monkeypatch.setenv("UNSLOTH_FLEX_MASK_REUSE", "0")
    monkeypatch.setattr(utils, "compiled_create_block_mask", full_dims)
    ref = _step(attn)
    for a, b in zip(got, ref):
        assert torch.equal(a, b)


@cuda
def test_padded_prefill_mask_keeps_batch_only(builds):
    _, sink = _mods()
    attn = _attn(None, training = False)
    q, k, v = [x.detach() for x in _qkv()]
    mask = torch.ones(2, 384, dtype = torch.long, device = "cuda")
    mask[0, :17] = 0
    with torch.no_grad():
        sink.flex_attention_with_sink(attn, q.clone(), k.clone(), v.clone(), attention_mask = mask)
    # The decoding cache's own mask first (unchanged), then the prefill mask: B = bsz, H = None
    assert builds[-1] == (2, None)


@cuda
@pytest.mark.parametrize("sliding_window", [None, 128])
def test_broadcast_mask_equals_full_mask(sliding_window):
    utils, _ = _mods()
    mm = utils.generate_sliding_window_mask(sliding_window) if sliding_window else utils.causal_mask
    full = utils.compiled_create_block_mask(mm, 2, 8, 640, 640, device = "cuda")
    bc = utils.compiled_create_block_mask(mm, None, None, 640, 640, device = "cuda")
    assert tuple(bc.kv_num_blocks.shape[:2]) == (1, 1)
    assert torch.equal(full.to_dense(), bc.to_dense().expand(2, 8, -1, -1))
