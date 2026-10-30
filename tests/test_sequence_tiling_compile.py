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
"""Compile gate for sequence_tiling: fullgraph=True, 0 graph breaks, recompiles <= tile buckets over 1k..32k rows,
every tile a checkpoint HOP in the graph, numerics equal to eager (forward and backward); CPU only."""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from _sequence_tiling_models import GatedMLP, TopKMoE, lora_mlp, load_balancing_loss, H  # noqa: E402
from unsloth_zoo.sequence_tiling import tiled_call, bucket_tiles  # noqa: E402

TILE_ROWS = 1024
SEQS = [1000, 1024, 1500, 2048, 3000, 4096, 5000, 8192, 12000, 16384, 20000, 32768]
N_BUCKETS = len({bucket_tiles(s, TILE_ROWS) for s in SEQS})


class _Counter:
    def __init__(self, backend):
        from torch._dynamo.testing import CompileCounterWithBackend
        self.inner = CompileCounterWithBackend(backend)
        self.hops = []

    def __call__(self, gm, example_inputs):
        self.hops.append(sum("tag_activation_checkpoint" in str(n.target) for n in gm.graph.nodes))
        return self.inner(gm, example_inputs)


class Layer(torch.nn.Module):
    """Pre-norm decoder layer: SDPA attention + tiled MLP, compiled as a whole."""
    def __init__(self, mlp, tiled = True):
        super().__init__()
        self.n1 = torch.nn.RMSNorm(H)
        self.n2 = torch.nn.RMSNorm(H)
        self.qkv = torch.nn.Linear(H, 3 * H, bias = False)
        self.o = torch.nn.Linear(H, H, bias = False)
        self.mlp = mlp
        self.tiled = tiled

    def forward(self, x):
        b, s, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, s, 3, 2, H // 2).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal = True)
        x = x + self.o(a.transpose(1, 2).reshape(b, s, H))
        h = self.n2(x)
        if self.tiled:
            n = bucket_tiles(h.shape[0] * h.shape[1], TILE_ROWS)
            return x + tiled_call(self.mlp, h, n_tiles = n, needs_rng = False)
        return x + self.mlp(h)


def _mlp_fn(m, contract = None):
    def f(x):
        n = bucket_tiles(x.shape[0] * x.shape[1], TILE_ROWS)
        return tiled_call(m, x, n_tiles = n, needs_rng = False, output_contract = contract)
    return f


def _grads(params):
    return [None if p.grad is None else p.grad.clone() for p in params]


def _loss(out):
    if isinstance(out, tuple):
        return out[0].float().pow(2).mean() + 0.3 * load_balancing_loss(out[1], 4, 2) + out[1].float().pow(2).mean()
    return out.float().pow(2).mean()


def _sweep(fn, params, backend, dynamic, seqs = SEQS, x_grad = True):
    from torch._dynamo.utils import counters
    torch._dynamo.reset()
    counters.clear()
    cnt = _Counter(backend)
    compiled = torch.compile(fn, backend = cnt, fullgraph = True, dynamic = dynamic)
    for s in seqs:
        torch.manual_seed(s)
        x = torch.randn(1, s, H, requires_grad = x_grad)
        for p in params: p.grad = None
        out = compiled(x)
        _loss(out).backward()
        gc = _grads(params) + [x.grad]
        for p in params: p.grad = None
        x2 = x.detach().clone().requires_grad_(x_grad)
        out_e = fn(x2)
        _loss(out_e).backward()
        ge = _grads(params) + [x2.grad]
        torch.testing.assert_close(out, out_e, rtol = 1e-4, atol = 1e-5)
        for a, b in zip(gc, ge):
            if b is None: assert a is None
            else: torch.testing.assert_close(a, b, rtol = 1e-4, atol = 1e-5)
    breaks = sum(counters["graph_break"].values())
    return cnt, breaks, len({bucket_tiles(s, TILE_ROWS) for s in seqs})


def _check(cnt, breaks, n_buckets, dynamic):
    recompiles = cnt.inner.frame_count - 1
    print(f"frames={cnt.inner.frame_count} recompiles={recompiles} buckets={n_buckets} breaks={breaks} "
          f"checkpoint_hops_per_graph={cnt.hops}")
    assert breaks == 0
    assert recompiles <= n_buckets
    # Every multi-tile graph holds one checkpoint HOP per tile.
    assert max(cnt.hops) >= 2


BACKENDS = ["aot_eager", "inductor"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dynamic", [None, True])
def test_dense_mlp(backend, dynamic):
    torch.manual_seed(0)
    m = GatedMLP()
    _check(*_sweep(_mlp_fn(m), list(m.parameters()), backend, dynamic), dynamic)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dynamic", [None, True])
def test_frozen_input_lora(backend, dynamic):
    m = lora_mlp()
    params = [p for p in m.parameters() if p.requires_grad]
    _check(*_sweep(_mlp_fn(m), params, backend, dynamic, x_grad = False), dynamic)


@pytest.mark.parametrize("backend", BACKENDS)
def test_moe_router_logits(backend):
    torch.manual_seed(0)
    m = TopKMoE()
    _check(*_sweep(_mlp_fn(m, ("rows", "rows_flat")), list(m.parameters()), backend, True,
                   seqs = [1000, 2048, 3000, 5000]), True)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dynamic", [None, True])
def test_compiled_decoder_layer(backend, dynamic):
    torch.manual_seed(0)
    layer = Layer(lora_mlp())
    params = [p for p in layer.parameters() if p.requires_grad]
    _check(*_sweep(layer, params, backend, dynamic, seqs = [1000, 1024, 1500, 2048, 3000, 4096, 8192]), dynamic)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shim", ["unsloth_checkpoint", "unsloth_offloaded_gradient_checkpoint", "unsloth_gradient_checkpoint"])
def test_with_unsloth_checkpoint_shim(backend, shim, monkeypatch):
    # Unsloth replaces torch.utils.checkpoint.checkpoint with a dynamo-disabled reentrant shim; the checkpoint HOP
    # re-enters it while AOT autograd traces. Still no graph break, and the shim hands the call to torch's own.
    import unsloth_zoo.gradient_checkpointing as gcm
    gcm._capture_pristine_checkpoint_once()
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", getattr(gcm, shim))
    m = lora_mlp()
    params = [p for p in m.parameters() if p.requires_grad]
    _check(*_sweep(_mlp_fn(m), params, backend, True, seqs = [1000, 2048, 3000]), True)


@pytest.mark.parametrize("backend", ["eager", "aot_eager", "inductor"])
def test_compiled_tiles_recompute(backend):
    # The partitioner honours the checkpoint tags: a compiled tiled MLP saves ~ its input, not [rows, I] activations.
    torch.manual_seed(0)
    m = lora_mlp(inter = 512)
    x = torch.randn(1, 8192, H)

    def saved_bytes(f):
        tot = [0]
        def pack(t):
            tot[0] += t.numel() * t.element_size()
            return t
        g = f if backend == "eager" else torch.compile(f, backend = backend, fullgraph = True)
        g(x)
        tot[0] = 0
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            y = g(x)
        y.pow(2).mean().backward()
        return tot[0]

    torch._dynamo.reset()
    untiled = saved_bytes(lambda x: m(x))
    tiled = saved_bytes(lambda x: tiled_call(m, x, n_tiles = 8, needs_rng = False))
    print(f"{backend}: saved untiled={untiled / 2**20:.2f} MB tiled={tiled / 2**20:.2f} MB")
    assert tiled < 2 * x.numel() * x.element_size() < untiled
