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

# GPT-OSS decode must reach flex attention with a static batch: Inductor only picks flex_decoding then,
# and generating at a second batch size would otherwise make the batch dim dynamic (3x slower on T4).
from __future__ import annotations

import importlib
import sys
import types

import pytest
import torch

pytest.importorskip("torch.nn.attention.flex_attention")


@pytest.fixture
def S(monkeypatch):
    # attention_sink on any host: CPU-only hosts leave the CUDA-sized flex utils undefined, so stub the missing names.
    import unsloth_zoo.flex_attention.utils as U
    stub = types.ModuleType(U.__name__)
    stub.__dict__.update(U.__dict__)
    stub.__getattr__ = lambda name: None
    monkeypatch.setitem(sys.modules, U.__name__, stub)
    monkeypatch.delitem(sys.modules, "unsloth_zoo.flex_attention.attention_sink", raising = False)
    module = importlib.import_module("unsloth_zoo.flex_attention.attention_sink")
    monkeypatch.delitem(sys.modules, "unsloth_zoo.flex_attention.attention_sink")
    return module


class _Attn(torch.nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.num_key_value_groups = 2
        self.scaling = 0.125
        self.sinks = torch.nn.Parameter(torch.zeros(heads))


def _block_mask(q_len, kv_len):
    from torch.nn.attention.flex_attention import BlockMask
    n = (kv_len + 127) // 128
    return BlockMask.from_kv_blocks(
        torch.full((1, 1, 1), n, dtype = torch.int32),
        torch.arange(n, dtype = torch.int32).view(1, 1, 1, n),
        seq_lengths = (q_len, kv_len),
    )


def _dynamic(size):
    # mark_static inside the trace specializes the batch symbol, which then resolves to a number.
    if not isinstance(size, torch.SymInt):
        return False
    return not size.node.shape_env.replace(size.node.expr).is_number


def _run(monkeypatch, S, q_len, traced_caller):
    # traced_caller False is the real layout: attention_sink runs eagerly and calls the compiled flex_attention.
    seen = []
    def fake_flex(query, key, value, block_mask = None, score_mod = None, enable_gqa = False, scale = None,
                  kernel_options = None, **kwargs):
        return query * 1, query.float().sum(-1)
    def backend(gm, example_inputs):
        values = [node.meta.get("example_value") for node in gm.graph.nodes if node.op == "placeholder"]
        seen.append([_dynamic(x.shape[0]) for x in values if isinstance(x, torch.Tensor) and x.dim() == 4])
        return gm
    torch._dynamo.reset()
    monkeypatch.setattr(S, "flex_attention", fake_flex if traced_caller else torch.compile(fake_flex, backend = backend))
    monkeypatch.setattr(S, "flex_output_and_lse", lambda output: output)
    attn = _Attn(8)
    def step(q, k, v):
        return S.flex_attention_with_sink_inference(attn, q, k, v, _block_mask(q_len, 128))
    if traced_caller:
        step = torch.compile(step, backend = backend, fullgraph = True)
    for batch in (1, 3, 5):
        q = torch.randn(batch, 8, q_len, 16)
        k = torch.randn(batch, 4, 128, 16)
        out = step(q, k, k.clone())
        assert out.shape == (batch, q_len, 8, 16)
    torch._dynamo.reset()
    return seen


@pytest.mark.parametrize("traced_caller", [False, True])
def test_decode_keeps_batch_static_across_batch_sizes(monkeypatch, S, traced_caller):
    seen = _run(monkeypatch, S, q_len = 1, traced_caller = traced_caller)
    # One graph per batch size, every q / k / v batch dim a plain int.
    assert len(seen) == 3
    assert seen[0] and not any(any(graph) for graph in seen), seen


@pytest.mark.parametrize("traced_caller", [False, True])
def test_prefill_batch_may_go_dynamic(monkeypatch, S, traced_caller):
    # Prefill keeps the main kernel either way, so it keeps Dynamo's automatic dynamic batch (no extra recompiles).
    seen = _run(monkeypatch, S, q_len = 7, traced_caller = traced_caller)
    assert len(seen) == 2 and any(seen[1]), seen


def test_dynamic_cache_decode_static_batch_no_per_step_recompile(monkeypatch, S):
    # DynamicCache (fallback when a static cache does not fit) grows kv length every step: batch stays static, kv length
    # goes dynamic once, later steps reuse that graph.
    seen = []
    def fake_flex(query, key, value, block_mask = None, **kwargs):
        return query * 1, query.float().sum(-1) + key.float().mean()
    def backend(gm, example_inputs):
        values = [node.meta.get("example_value") for node in gm.graph.nodes if node.op == "placeholder"]
        seen.append([_dynamic(x.shape[0]) for x in values if isinstance(x, torch.Tensor) and x.dim() == 4])
        return gm
    torch._dynamo.reset()
    monkeypatch.setattr(S, "flex_attention", torch.compile(fake_flex, backend = backend))
    monkeypatch.setattr(S, "flex_output_and_lse", lambda output: output)
    attn = _Attn(8)
    for batch in (1, 4):
        for kv_len in range(128, 136):
            q, k = torch.randn(batch, 8, 1, 16), torch.randn(batch, 4, kv_len, 16)
            S.flex_attention_with_sink_inference(attn, q, k, k.clone(), _block_mask(1, kv_len))
        graphs_after_batch = len(seen)
        for kv_len in range(136, 140):
            q, k = torch.randn(batch, 8, 1, 16), torch.randn(batch, 4, kv_len, 16)
            S.flex_attention_with_sink_inference(attn, q, k, k.clone(), _block_mask(1, kv_len))
        assert len(seen) == graphs_after_batch, seen
    torch._dynamo.reset()
    assert seen and not any(any(graph) for graph in seen), seen
    assert len(seen) <= 4, seen
