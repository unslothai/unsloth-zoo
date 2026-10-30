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

"""Compile gate for sequence_tiling: fullgraph=True, 0 graph breaks, recompiles <= tile buckets over 1k..32k rows,
every tile a checkpoint HOP in the graph, numerics equal to eager (forward and backward); CPU only."""
import copy
import functools
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


class _CompiledExpertMoE(torch.nn.Module):
    """Eager MoE whose expert math is a separately compiled function, as Unsloth's MoE kernels are: the per-expert
    token count changes from tile to tile, so Dynamo compiles a static graph first and a dynamic one later."""

    def __init__(self, expert_fn, h = 8, e = 4):
        super().__init__()
        self.gate = torch.nn.Linear(h, e, bias = False)
        self.w = torch.nn.Parameter(torch.randn(e, h, h) / 3)
        self.expert_fn = expert_fn

    def forward(self, x):
        h = x.reshape(-1, x.shape[-1])
        sel = self.gate(h).argmax(-1)
        out = torch.zeros_like(h)
        for e in range(self.w.shape[0]):
            idx = (sel == e).nonzero().squeeze(-1)
            if idx.numel() == 0:
                continue
            out = out.index_add(0, idx, self.expert_fn(h[idx], self.w[e]))
        return out.reshape(x.shape)


def _expert(x, w):
    # Token-local values, but the graph reduces over the token axis (as grouped-GEMM bookkeeping does), so the
    # static and the dynamic graph save different tensors.
    h = x @ w.t()
    g = torch.nn.functional.silu(h) + 0.0 * h.mean(0, keepdim = True)
    return torch.nn.functional.layer_norm(g, (g.shape[-1],)) * h.softmax(-1)


@pytest.mark.parametrize("n_tiles", [4, 16])
def test_inner_compiled_function_recompute_uses_the_same_graph(n_tiles):
    # Before: "Recomputed values ... have different metadata" / "A different number of tensors was saved" (OLMoE).
    torch._dynamo.reset()
    torch.manual_seed(0)
    m = _CompiledExpertMoE(torch.compile(_expert, backend = "inductor"))
    ref = copy.deepcopy(m)
    ref.expert_fn = _expert
    x = torch.randn(1, 64, 8, requires_grad = True)
    xr = x.detach().clone().requires_grad_()
    y = tiled_call(m, x, n_tiles = n_tiles, needs_rng = False)
    y.pow(2).sum().backward()
    yr = ref(xr)
    yr.pow(2).sum().backward()
    torch.testing.assert_close(y, yr, rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(x.grad, xr.grad, rtol = 1e-4, atol = 1e-5)
    torch.testing.assert_close(m.w.grad, ref.w.grad, rtol = 1e-4, atol = 1e-5)


def _block_with_tiled_mlp(n_tiles):
    m = GatedMLP()

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = m

        def forward(self, h):
            return h + tiled_call(self.mlp, h * 1.5, n_tiles = n_tiles, needs_rng = False)
    return Block()


@pytest.mark.parametrize("mode", ["unsloth_block", "env_break"])
def test_unsloth_block_compile_does_not_unroll_tiles(mode, monkeypatch):
    # Inside Unsloth's compiled blocks (fullgraph=False) the tile loop leaves the graph: compile work must not grow
    # with the tile count. Before, every tile added a checkpoint HOP to the graph (G4: ~1.1 s cold compile per tile).
    import types
    fake = types.ModuleType("unsloth.models._block_compile")
    fake._IN_COMPILED_REGION = [mode == "unsloth_block"]
    monkeypatch.setitem(sys.modules, "unsloth.models._block_compile", fake)
    if mode == "env_break":
        monkeypatch.setenv("UNSLOTH_TILED_COMPILE", "break")
    seen = {}
    for n in (4, 16, 64):
        torch._dynamo.reset()
        hops, graphs = [0], [0]

        def backend(gm, example_inputs):
            graphs[0] += 1
            hops[0] += sum(1 for node in gm.graph.nodes if "checkpoint" in str(node.target))
            return gm.forward
        torch.manual_seed(0)
        block = _block_with_tiled_mlp(n)
        ref = copy.deepcopy(block)
        x = torch.randn(1, 4 * n, H, requires_grad = True)
        xr = x.detach().clone().requires_grad_()
        y = torch.compile(block, backend = backend)(x)
        y.pow(2).sum().backward()
        yr = ref.mlp(xr * 1.5) + xr
        yr.pow(2).sum().backward()
        torch.testing.assert_close(y, yr)
        torch.testing.assert_close(x.grad, xr.grad)
        assert hops[0] == 0, hops
        seen[n] = graphs[0]
    assert len(set(seen.values())) == 1, seen  # same compile work at 4, 16 and 64 tiles


def test_user_compile_keeps_the_unrolled_fullgraph(monkeypatch):
    # Outside Unsloth's blocks a user's compile may be fullgraph=True, where a graph break is an error.
    monkeypatch.delenv("UNSLOTH_TILED_COMPILE", raising = False)
    monkeypatch.delitem(sys.modules, "unsloth.models._block_compile", raising = False)
    torch._dynamo.reset()
    block = _block_with_tiled_mlp(4)
    x = torch.randn(1, 16, H, requires_grad = True)
    torch.compile(block, backend = "aot_eager", fullgraph = True)(x).sum().backward()


def test_compiled_tile_body_is_one_graph_for_every_tile_and_layer(monkeypatch):
    # compile_tile: the tile loop leaves the caller's graph and every tile of every layer runs one compiled body.
    from torch._dynamo.utils import counters
    from unsloth_zoo import sequence_tiling as ST

    class Layers(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlps = torch.nn.ModuleList([GatedMLP() for _ in range(3)])

        def forward(self, h):
            for mlp in self.mlps:
                h = h + tiled_call(functools.partial(GatedMLP.forward, mlp), h, n_tiles = 8, needs_rng = False)
            return h

    graphs = {}
    for mode in ("break", "compile_tile"):
        monkeypatch.setenv("UNSLOTH_TILED_COMPILE", mode)
        torch._dynamo.reset()
        ST._COMPILED_TILE[0] = None
        counters.clear()
        torch.manual_seed(0)
        m = Layers()
        ref = copy.deepcopy(m)
        x = torch.randn(1, 37, H, requires_grad = True)  # uneven tiles
        y = torch.compile(m, backend = "aot_eager")(x)
        y.pow(2).sum().backward()
        h = x.detach().clone().requires_grad_()
        yr = h
        for mlp in ref.mlps:
            yr = yr + mlp(yr)
        yr.pow(2).sum().backward()
        torch.testing.assert_close(y, yr)
        torch.testing.assert_close(x.grad, h.grad)
        graphs[mode] = counters["stats"]["unique_graphs"]
    assert graphs["compile_tile"] - graphs["break"] == 1, graphs
