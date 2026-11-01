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
    real = R._op_forward_with_token

    def spy(x, params, n_tiles, amp = None):
        seen.append(n_tiles)
        return real(x, params, n_tiles, amp)
    monkeypatch.setattr(R, "_op_forward_with_token", spy)
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
    assert saved == [x.numel(), 1]  # the input and the one-element module token
    y.sum().backward()
    xr = x.detach().clone().requires_grad_()
    mlp.zero_grad(set_to_none = True)
    torch.testing.assert_close(x.grad, torch.autograd.grad(R._inner_forward(mlp)(xr).sum(), xr)[0])


def test_autocast_matches_eager_tiles(in_unsloth_block, monkeypatch):
    # Tiling itself changes bf16 rounding (per-tile weight-grad partials), so the reference is the same tiles run eagerly.
    # bf16 activations under bf16 autocast (Unsloth's 16-bit LoRA): the op keeps the activation dtype.
    torch.manual_seed(0)
    block = _Block(_LoraMLP()).to(torch.bfloat16)
    ref = copy.deepcopy(block)
    R.install_module(block.mlp, "rows:8")
    R.install_module(ref.mlp, "rows:8")
    x = torch.randn(1, 64, H, dtype = torch.bfloat16, requires_grad = True)
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
    # Inductor fuses the tile body (one bf16 rounding where eager rounds per op) and the weights are bf16 too:
    # compare at a few bf16 ulps (eps 7.8e-3).
    for a, b in zip([y] + _grads(block, x), [yr] + _grads(ref, xr)):
        assert (a - b).abs().max() <= 2e-2 * b.abs().max()


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


def test_shared_adapters_over_different_weights_keep_the_tile_frame_route(in_unsloth_block):
    # Failing before: two MLPs sharing their adapters but not their frozen weights resolved to one module, so one
    # ran the other's forward (wrong outputs and gradients). Now neither takes the op; a fully tied copy still does.
    torch.manual_seed(0)
    a, b = _LoraMLP(), _LoraMLP()
    b.A, b.B = a.A, a.B
    R.install_module(a, "rows:8")
    R.install_module(b, "rows:8")
    assert not a.__dict__["_unsloth_tiling"].op_ok and not b.__dict__["_unsloth_tiling"].op_ok
    x = torch.randn(64, H)
    for mlp in (a, b):
        got = torch.compile(mlp, backend = "aot_eager", dynamic = True)(x)
        torch.testing.assert_close(got, type(mlp).forward(mlp, x))
    tied = _LoraMLP()
    tied.g, tied.u, tied.d, tied.A, tied.B = a.g, a.u, a.d, a.A, a.B
    c = _LoraMLP()
    c.A, c.B = tied.A, tied.B
    c.g, c.u, c.d = tied.g, tied.u, tied.d
    R.install_module(tied, "rows:8")
    R.install_module(c, "rows:8")
    assert tied.__dict__["_unsloth_tiling"].op_ok and c.__dict__["_unsloth_tiling"].op_ok


def test_parameter_hooks_run_once(in_unsloth_block):
    # Failing before: the recompute differentiated the real parameters, so a register_hook ran per tile and again on
    # the op's returned gradient (8 tiles: 9 calls, doubled gradient).
    torch.manual_seed(0)
    mlp = _LoraMLP()
    ref = copy.deepcopy(mlp)
    R.install_module(mlp, "rows:8")
    calls = []

    def hook(g):
        calls.append(1)
        return g * 2
    mlp.A.register_hook(hook)
    ref.A.register_hook(lambda g: g * 2)
    block, ref_block = _Block(mlp), _Block(ref)
    ref_block.o.load_state_dict(block.o.state_dict())
    x = torch.randn(2, 32, H)
    _run(block, x.clone().requires_grad_(True))
    _ref(ref_block, x.clone().requires_grad_(True))
    assert len(calls) == 1, len(calls)
    torch.testing.assert_close(mlp.A.grad, ref.A.grad)


def test_autocast_to_another_dtype_keeps_the_mlp_output_dtype(in_unsloth_block, monkeypatch):
    # Failing before: the op allocated its output like the float32 input, so a bf16 autocast MLP returned float32.
    torch.manual_seed(0)
    mlp = _LoraMLP()
    R.install_module(mlp, "rows:8")
    seen = _spy_op(monkeypatch)
    x = torch.randn(1, 32, H)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        want = type(mlp).forward(mlp, x)
        got = torch.compile(mlp, backend = "aot_eager", dynamic = True)(x)
    assert got.dtype == want.dtype == torch.bfloat16, (got.dtype, want.dtype)
    assert not seen  # the per-tile route, not the op


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


class _TransposedGradMatmul(torch.autograd.Function):
    """x @ w whose weight gradient comes back transposed in memory, as some mm backwards (fused LoRA) return it."""

    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return x @ w

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        return g @ w.t(), (g.reshape(-1, g.shape[-1]).t() @ x.reshape(-1, x.shape[-1])).t()


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_a_transposed_parameter_gradient_is_laid_out_like_the_parameter(in_unsloth_block, backend):
    class FooMLP(_LoraMLP):
        def forward(self, x):
            return self.d(F.silu(self.g(x) + _TransposedGradMatmul.apply(x, self.A) @ self.B) * self.u(x))
    FooMLP.__module__ = "transformers.models.foo.modeling_foo"
    torch.manual_seed(0)
    block = _Block(FooMLP())
    ref = copy.deepcopy(block)
    R.install_module(block.mlp, "rows:8")
    x = torch.randn(1, 64, H, requires_grad = True)
    xr = x.detach().clone().requires_grad_()
    y, breaks = _run(block, x, backend = backend)
    _ref(ref, xr)
    assert breaks == 0
    for a, b in zip(_grads(block, x), _grads(ref, xr)):
        torch.testing.assert_close(a, b)


def test_a_deepcopy_of_an_installed_mlp_runs_compiled(in_unsloth_block):
    # Failing before: the copy carried op_ok but was never registered, so the op could not find its module.
    torch.manual_seed(0)
    mlp = _LoraMLP()
    R.install_module(mlp, "rows:8")
    clone = copy.deepcopy(mlp)
    x = torch.randn(64, H)
    got = torch.compile(clone, backend = "aot_eager", dynamic = True)(x)
    torch.testing.assert_close(got, type(clone).forward(clone, x))


class _PartlyUsedMLP(_LoraMLP):
    def __init__(self):
        super().__init__()
        self.unused = torch.nn.Parameter(torch.zeros(4))


_PartlyUsedMLP.__module__ = "transformers.models.foo.modeling_foo"
_PartlyUsedMLP.__name__ = _PartlyUsedMLP.__qualname__ = "FooPartlyUsedMLP"


def test_an_unused_parameter_keeps_the_tile_frame_route(in_unsloth_block):
    # Failing before: the op returned zeros for a parameter the forward never uses, so Adam's moments and weight
    # decay moved it. The install probe now keeps such a module on the per-tile route (None gradient).
    torch.manual_seed(0)
    mlp = _PartlyUsedMLP()
    R.install_module(mlp, "rows:8")
    assert not mlp.__dict__["_unsloth_tiling"].op_ok
    _run(_Block(mlp), torch.randn(2, 32, H, requires_grad = True))
    assert mlp.unused.grad is None and mlp.A.grad is not None


def test_saved_tensor_hooks_that_copy_keep_the_module(in_unsloth_block):
    # Failing before: backward found its module by the saved parameters' storage, which save_on_cpu replaces.
    torch.manual_seed(0)
    mlp = _LoraMLP()
    ref = copy.deepcopy(mlp)
    R.install_module(mlp, "rows:4")
    x = torch.randn(64, H)
    # As save_on_cpu on a GPU: the saved tensors come back as copies.
    with torch.autograd.graph.saved_tensors_hooks(lambda t: t.clone(), lambda t: t):
        torch.compile(mlp, backend = "aot_eager", dynamic = True)(x).pow(2).sum().backward()
    type(ref).forward(ref, x).pow(2).sum().backward()
    torch.testing.assert_close(mlp.A.grad, ref.A.grad)


def test_an_uninstalled_mlp_leaves_the_op_registry(in_unsloth_block):
    # Failing before: a detached module stayed registered, so a later install sharing its adapters was declined
    # against a module that no longer tiles.
    torch.manual_seed(0)
    a = _LoraMLP()
    R.install_module(a, "rows:8")
    R.uninstall_module(a) if hasattr(R, "uninstall_module") else R._detach(a)
    assert a not in R._OP_MODULES
