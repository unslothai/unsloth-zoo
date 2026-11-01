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

"""Tiled dense MLP inside Unsloth's compiled block: the tile loop is one opaque op, so the block keeps one graph."""

import copy
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from unsloth_zoo import tiling_registry as R

H, I = 32, 96


class _LoraMLP(torch.nn.Module):
    """Frozen base projections plus trainable low-rank adapters, like a PEFT LoRA MLP."""

    def __init__(self, dropout = 0.0):
        super().__init__()
        self.g = torch.nn.Linear(H, I, bias = False).requires_grad_(False)
        self.u = torch.nn.Linear(H, I, bias = False).requires_grad_(False)
        self.d = torch.nn.Linear(I, H, bias = False).requires_grad_(False)
        self.A = torch.nn.Parameter(torch.randn(H, 4) * 0.1)
        self.B = torch.nn.Parameter(torch.randn(4, I) * 0.1)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.d(F.silu(self.g(x) + (self.drop(x) @ self.A) @ self.B) * self.u(x))


# Library classes may skip the RNG replay (see needs_rng_replay); a test class defined here would always replay.
_LoraMLP.__module__ = "transformers.models.foo.modeling_foo"
_LoraMLP.__name__ = _LoraMLP.__qualname__ = "FooMLP"


class _Block(torch.nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.o = torch.nn.Linear(H, H, bias = False)
        self.mlp = mlp

    def forward(self, x):
        x = x + self.o(F.rms_norm(x, (H,)))
        return x + self.mlp(F.rms_norm(x, (H,)))


@pytest.fixture
def in_unsloth_block(monkeypatch):
    """Unsloth's compiled-region flag, as unsloth.models._block_compile sets it while its block runs."""
    fake = types.ModuleType("unsloth.models._block_compile")
    fake._IN_COMPILED_REGION = [True]
    monkeypatch.setitem(sys.modules, "unsloth.models._block_compile", fake)
    monkeypatch.delenv("UNSLOTH_TILED_COMPILE", raising = False)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _run(block, x, backend = "aot_eager", fullgraph = False):
    from torch._dynamo.utils import counters
    counters.clear()
    y = torch.compile(block, backend = backend, fullgraph = fullgraph, dynamic = True)(x)
    y.float().pow(2).sum().backward()
    return y, sum(counters["graph_break"].values())


def _ref(block, x):
    y = block(x)
    y.float().pow(2).sum().backward()
    return y


def _grads(block, x):
    return [x.grad] + [p.grad for p in block.parameters() if p.requires_grad]


def _spy_op(monkeypatch):
    seen = []
    real = R._op_forward_impl

    def spy(x, params, n_tiles, amp = None):
        seen.append(n_tiles)
        return real(x, params, n_tiles, amp)
    monkeypatch.setattr(R, "_op_forward_impl", spy)
    return seen


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_tiled_mlp_keeps_the_block_one_graph(in_unsloth_block, monkeypatch, backend):
    torch.manual_seed(0)
    block = _Block(_LoraMLP())
    ref = copy.deepcopy(block)
    R.install_module(block.mlp, "rows:8")
    seen = _spy_op(monkeypatch) if hasattr(R, "_op_forward_impl") else []
    for rows in (64, 61):
        x = torch.randn(1, rows, H, requires_grad = True)
        xr = x.detach().clone().requires_grad_()
        y, breaks = _run(block, x, backend = backend)
        yr = _ref(ref, xr)
        assert breaks == 0
        torch.testing.assert_close(y, yr)
        for a, b in zip(_grads(block, x), _grads(ref, xr)):
            torch.testing.assert_close(a, b)
        block.zero_grad(set_to_none = True)
        ref.zero_grad(set_to_none = True)
    assert seen and set(seen) == {8}


def test_fullgraph_compile_in_an_unsloth_block_does_not_break(in_unsloth_block):
    torch.manual_seed(0)
    block = _Block(_LoraMLP())
    R.install_module(block.mlp, "rows:8")
    x = torch.randn(1, 64, H, requires_grad = True)
    _run(block, x, fullgraph = True)


def test_the_op_saves_only_its_input(in_unsloth_block):
    torch.manual_seed(0)
    mlp = _LoraMLP()
    R.install_module(mlp, "rows:8")
    state = mlp.__dict__["_unsloth_tiling"]
    x = torch.randn(1, 64, H, requires_grad = True)
    params = R._op_params(mlp, state)
    own = {p.data_ptr() for p in mlp.parameters()}
    saved = []

    def pack(t):
        if t.data_ptr() not in own:
            saved.append(t.numel())
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = R._run_tiled_op(state, x, params, 8)
    assert saved == [x.numel()]
    y.sum().backward()
    xr = x.detach().clone().requires_grad_()
    mlp.zero_grad(set_to_none = True)
    torch.testing.assert_close(x.grad, torch.autograd.grad(R._inner_forward(mlp)(xr).sum(), xr)[0])


def test_autocast_matches_eager_tiles(in_unsloth_block, monkeypatch):
    # Tiling itself changes bf16 rounding (per-tile weight-grad partials), so the reference is the same tiles run eagerly.
    torch.manual_seed(0)
    block = _Block(_LoraMLP())
    ref = copy.deepcopy(block)
    R.install_module(block.mlp, "rows:8")
    R.install_module(ref.mlp, "rows:8")
    x = torch.randn(1, 64, H, requires_grad = True)
    xr = x.detach().clone().requires_grad_()
    seen = []
    real = R._tile_fn

    def spy(*args):
        seen.append(torch.is_autocast_enabled("cpu"))
        return real(*args)
    monkeypatch.setattr(R, "_tile_fn", spy)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        y, breaks = _run(block, x)
        yr = _ref(ref, xr)
    # Compiled graphs run with autocast off; the tile body must still see the caller's autocast.
    assert seen and all(seen)
    assert breaks == 0 and y.dtype == yr.dtype
    # Inductor fuses the tile body (one bf16 rounding where eager rounds per op): compare at bf16 resolution.
    for a, b in zip([y] + _grads(block, x), [yr] + _grads(ref, xr)):
        assert (a - b).abs().max() <= 1e-2 * b.abs().max()


def test_rng_replay_falls_back_to_the_tile_frames(in_unsloth_block, monkeypatch):
    torch.manual_seed(0)
    block = _Block(_LoraMLP(dropout = 0.1))
    R.install_module(block.mlp, "rows:8")
    seen = _spy_op(monkeypatch)
    x = torch.randn(1, 64, H, requires_grad = True)
    _, breaks = _run(block, x)
    assert seen == [] and breaks > 0


def test_env_selects_the_previous_route(in_unsloth_block, monkeypatch):
    monkeypatch.setenv("UNSLOTH_TILED_COMPILE", "compile_tile")
    torch.manual_seed(0)
    block = _Block(_LoraMLP())
    R.install_module(block.mlp, "rows:8")
    seen = _spy_op(monkeypatch)
    x = torch.randn(1, 64, H, requires_grad = True)
    _, breaks = _run(block, x)
    assert seen == [] and breaks > 0


def test_auto_inside_the_block_tiles_only_when_it_pays(in_unsloth_block, monkeypatch):
    torch.manual_seed(0)
    block = _Block(_LoraMLP())
    R.install_module(block.mlp, "auto")
    state = block.mlp.__dict__["_unsloth_tiling"]
    state.tile_rows = 8
    state.model_stats = (0, 1)
    seen = _spy_op(monkeypatch)
    x = torch.randn(1, 64, H, requires_grad = True)
    _, breaks = _run(block, x)
    assert seen == [] and breaks == 0
    # Same rows once the estimated saving clears the floor: tiled, and still one graph.
    monkeypatch.setattr(R, "AUTO_MIN_SAVING_MIB", 0)
    monkeypatch.setattr(R, "AUTO_MIN_SAVING_FRACTION", 0.0)
    torch._dynamo.reset()
    _, breaks = _run(block, x)
    assert seen == [8] and breaks == 0


def test_modules_sharing_parameters_resolve_by_class(in_unsloth_block):
    torch.manual_seed(0)
    a = _LoraMLP()
    b = _LoraMLP()
    b.A, b.B = a.A, a.B
    R.install_module(a, "rows:8")
    R.install_module(b, "rows:8")
    x = torch.randn(64, H)
    params = R._op_params(a, a.__dict__["_unsloth_tiling"])
    R._op_forward_impl(x, params, 8)
    other = type("BarMLP", (_LoraMLP,), {})()
    other.A, other.B = a.A, a.B
    R.install_module(other, "rows:8")
    with pytest.raises(RuntimeError, match = "could not find the module"):
        R._op_forward_impl(x, params, 8)


def test_a_width_changing_mlp_keeps_the_tile_frame_route(in_unsloth_block, monkeypatch):
    class AlignerMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(H, I)
            self.fc2 = torch.nn.Linear(I, 2 * H)

        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(x)))
    AlignerMLP.__module__ = "transformers.models.foo.modeling_foo"
    torch.manual_seed(0)
    mlp = AlignerMLP()
    ref = copy.deepcopy(mlp)
    R.install_module(mlp, "rows:8")
    assert not mlp.__dict__["_unsloth_tiling"].op_ok
    seen = _spy_op(monkeypatch)
    x = torch.randn(1, 64, H, requires_grad = True)
    y, breaks = _run(mlp, x)
    assert seen == [] and breaks > 0
    torch.testing.assert_close(y, ref(x.detach()))
