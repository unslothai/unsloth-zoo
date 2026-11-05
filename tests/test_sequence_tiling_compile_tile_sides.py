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

"""compile_tile: a tile body is fully compiled or eager, and its recompute takes its forward's side (qwen3_moe CheckpointError)."""

import copy
import os
import sys

import pytest
import torch
import torch.utils.checkpoint as C

sys.path.insert(0, os.path.dirname(__file__))
from _sequence_tiling_models import GatedMLP, H  # noqa: E402
from unsloth_zoo import sequence_tiling as ST  # noqa: E402


class _BreakingMLP(GatedMLP):
    """Graph break mid-body, like a PEFT ParamWrapper."""

    def forward(self, x):
        torch._dynamo.graph_break()
        return super().forward(x)


def _trace_saves(monkeypatch):
    """Per checkpoint frame: (frame, forward packs, recompute packs), each True if packed by an AOT graph."""
    import traceback
    frames = []

    def compiled():
        return any("runtime_wrapper" in f.name for f in traceback.extract_stack())

    oc, orr = C._checkpoint_hook.__init__, C._recomputation_hook.__init__

    def cinit(self, frame):
        oc(self, frame)
        pack = self.pack_hook

        def traced(x):
            if not any(f is frame for f, _, _ in frames):
                frames.append((frame, [], []))
            next(r for f, r, _ in frames if f is frame).append(compiled())
            return pack(x)
        self.pack_hook = traced

    def rinit(self, ref, gid):
        orr(self, ref, gid)
        pack = self.pack_hook

        def traced(x):
            frame = ref()
            for f, _, r in frames:
                if f is frame:
                    r.append(compiled())
            return pack(x)
        self.pack_hook = traced

    monkeypatch.setattr(C._checkpoint_hook, "__init__", cinit)
    monkeypatch.setattr(C._recomputation_hook, "__init__", rinit)
    return frames


def _layers(cls, n = 2):
    class Layers(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlps = torch.nn.ModuleList([cls() for _ in range(n)])

        def forward(self, h):
            for mlp in self.mlps:
                h = h + ST.tiled_call(mlp, h, n_tiles = 4, needs_rng = False)
            return h
    return Layers


def _run(monkeypatch, cls):
    monkeypatch.setenv("UNSLOTH_TILED_COMPILE", "compile_tile")
    torch._dynamo.reset()
    ST._COMPILED_TILES.clear()
    getattr(ST, "_EAGER_TILES", set()).clear()
    frames = _trace_saves(monkeypatch)
    torch.manual_seed(0)
    m = _layers(cls)()
    ref = copy.deepcopy(m)
    x = torch.randn(1, 37, H, requires_grad = True)
    y = torch.compile(m, backend = "aot_eager")(x)
    y.pow(2).sum().backward()
    h = x.detach().clone().requires_grad_()
    yr = h
    for mlp in ref.mlps:
        yr = yr + mlp(yr)
    yr.pow(2).sum().backward()
    torch.testing.assert_close(y, yr)
    torch.testing.assert_close(x.grad, h.grad)
    for a, b in zip(m.parameters(), ref.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    return frames


def test_a_body_that_does_not_trace_whole_runs_eager_in_forward_and_recompute(monkeypatch):
    frames = _run(monkeypatch, _BreakingMLP)
    assert len(frames) == 8, len(frames)
    for _, fwd, rec in frames:
        assert fwd == rec, (fwd, rec)
        assert not any(fwd), fwd


def test_a_body_that_traces_whole_stays_one_compiled_graph(monkeypatch):
    frames = _run(monkeypatch, GatedMLP)
    assert len(frames) == 8
    for _, fwd, rec in frames:
        assert fwd == rec and fwd and all(fwd), (fwd, rec)
    assert not getattr(ST, "_EAGER_TILES", set())


def test_a_checkpointed_tile_keeps_its_side_when_the_body_falls_back_later(monkeypatch):
    # A later compile failure turns the body eager; already-compiled forwards still recompute compiled.
    monkeypatch.setenv("UNSLOTH_TILED_COMPILE", "compile_tile")
    torch._dynamo.reset()
    ST._COMPILED_TILES.clear()
    ST._EAGER_TILES.clear()
    frames = _trace_saves(monkeypatch)
    torch.manual_seed(0)
    mlp = GatedMLP()
    ref = copy.deepcopy(mlp)
    x = torch.randn(1, 37, H, requires_grad = True)
    y = ST._tiled_call_eager(mlp, x, n_tiles = 4, needs_rng = False, output_contract = None, compile_tile = True)
    ST._EAGER_TILES.update(ST._COMPILED_TILES)
    y.pow(2).sum().backward()
    xr = x.detach().clone().requires_grad_()
    ref(xr).pow(2).sum().backward()
    torch.testing.assert_close(x.grad, xr.grad)
    for _, fwd, rec in frames:
        assert fwd == rec and all(fwd), (fwd, rec)
    frames.clear()
    y = ST._tiled_call_eager(mlp, x, n_tiles = 4, needs_rng = False, output_contract = None, compile_tile = True)
    y.sum().backward()
    for _, fwd, rec in frames:
        assert fwd == rec and not any(fwd), (fwd, rec)


class _InlineFlagMLP(GatedMLP):
    """Branches on UNSLOTH_DECODE_COMPILE like the bnb 4-bit expert forwards."""

    def forward(self, x):
        from unsloth_zoo.temporary_patches.utils import UNSLOTH_DECODE_COMPILE
        if not UNSLOTH_DECODE_COMPILE[0]:
            x = x.exp().log()
        return super().forward(x)


def test_a_tile_checkpointed_inside_a_block_recomputes_with_the_blocks_inline_flag():
    # Recompute runs after the block reset the flag; unpinned it took the other branch (OLMoE-1B-7B 4-bit CheckpointError).
    from unsloth_zoo.temporary_patches.utils import UNSLOTH_DECODE_COMPILE
    torch.manual_seed(0)
    mlp = _InlineFlagMLP()
    ref = copy.deepcopy(mlp)
    x = torch.randn(1, 37, H, requires_grad = True)
    UNSLOTH_DECODE_COMPILE[0] = True
    try:
        y = ST._tiled_call_eager(mlp, x, n_tiles = 4, needs_rng = False, output_contract = None, compile_tile = False)
        xr = x.detach().clone().requires_grad_()
        yr = ref(xr)
    finally:
        UNSLOTH_DECODE_COMPILE[0] = False
    y.pow(2).sum().backward()
    yr.pow(2).sum().backward()
    assert UNSLOTH_DECODE_COMPILE[0] is False
    torch.testing.assert_close(x.grad, xr.grad)
    for a, b in zip(mlp.parameters(), ref.parameters()):
        torch.testing.assert_close(a.grad, b.grad)


_CHURN = [0]


class _GuardChurnMLP(GatedMLP):
    """Reads a global Dynamo guards on: every new value recompiles the tile frame."""

    def forward(self, x):
        return super().forward(x) * (1.0 + 0.0 * _CHURN[0])


@pytest.mark.parametrize("train", [True, False])
def test_a_tile_frame_past_the_recompile_limit_runs_eager(monkeypatch, train):
    # fullgraph=True turns the recompile limit into FailOnRecompileLimitHit, which is not a TorchDynamoException;
    # without fullgraph Dynamo ran the frame eagerly. The tile must fall back to eager, not raise.
    monkeypatch.setenv("UNSLOTH_TILED_COMPILE", "compile_tile")
    limit = "recompile_limit" if "recompile_limit" in torch._dynamo.config._config else "cache_size_limit"  # torch 2.6
    with torch._dynamo.config.patch(**{limit: 2}):
        _past_the_recompile_limit(train)


def _past_the_recompile_limit(train):
    torch._dynamo.reset()
    ST._COMPILED_TILES.clear()
    ST._EAGER_TILES.clear()
    torch.manual_seed(0)
    mlp = _GuardChurnMLP()
    ref = copy.deepcopy(mlp)
    for i in range(4):
        _CHURN[0] = i
        x = torch.randn(1, 37, H, requires_grad = train)
        if train:
            y = ST._tiled_call_eager(mlp, x, n_tiles = 4, needs_rng = False, output_contract = None, compile_tile = True)
            y.pow(2).sum().backward()
            xr = x.detach().clone().requires_grad_()
            ref(xr).pow(2).sum().backward()
            torch.testing.assert_close(x.grad, xr.grad)
        else:
            with torch.no_grad():
                y = ST._tiled_call_eager(mlp, x, n_tiles = 4, needs_rng = False, output_contract = None, compile_tile = True)
                torch.testing.assert_close(y, ref(x))
    assert ST._EAGER_TILES, "the frame past the limit should have turned eager"
    if train:
        for a, b in zip(mlp.parameters(), ref.parameters()):
            torch.testing.assert_close(a.grad, b.grad)
