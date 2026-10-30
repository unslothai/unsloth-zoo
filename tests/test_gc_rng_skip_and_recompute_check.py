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

"""UNSLOTH_GC_SKIP_RNG_WITHOUT_DROPOUT skips the RNG snapshot only for layers that cannot draw random numbers, with
losses, gradients and the global RNG stream bitwise unchanged; UNSLOTH_GC_RECOMPUTE_CHECK compares each checkpoint's
forward outputs with its recompute. CPU, with MINIMUM_SIZE above every tensor so the offload branch stays off."""
import functools
import types
import warnings

import pytest
import torch


@pytest.fixture
def G(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ALLOW_CPU", "1")
    try:
        from unsloth_zoo import gradient_checkpointing as module
    except ImportError as e:
        pytest.skip(f"unsloth_zoo import unavailable: {e}")
    for name, value in (
        ("FIRST_PASS", False), ("LAST_GC_INDEX", 0), ("CURRENT_GC_INDEX", 0),
        ("MINIMUM_SIZE", 1 << 40), ("CPU_INDEX", 0),
        ("CPU_BUFFERS", [torch.empty(0)]), ("BACKWARD_PASS", False),
        ("USE_DOUBLE_BUFFER", False), ("GPU_BUFFERS", []), ("GPU_BUFFERS_B", None),
        ("GC_SKIP_RNG_WITHOUT_DROPOUT", False), ("GC_RECOMPUTE_CHECK", "0"),
        ("GC_RNG_STATS", {"skipped": 0, "kept": 0}), ("GC_RECOMPUTE_MISMATCHES", []),
    ):
        monkeypatch.setattr(module, name, value, raising = False)
    monkeypatch.setattr(module, "_DRAWS_RANDOM", __import__("weakref").WeakKeyDictionary(), raising = False)
    return module


class _Attn(torch.nn.Module):
    """Mimics HF attention: a float rate attribute applied through F.dropout only in training."""
    def __init__(self, d, rate):
        super().__init__()
        self.proj = torch.nn.Linear(d, d)
        self.attention_dropout = rate

    def forward(self, h):
        return torch.nn.functional.dropout(self.proj(h), p = self.attention_dropout if self.training else 0.0)


class _Layer(torch.nn.Module):
    def __init__(self, d = 16, mlp_dropout = 0.0, attention_dropout = 0.0, lora_dropout = 0.0):
        super().__init__()
        self.attn = _Attn(d, attention_dropout)
        self.up = torch.nn.Linear(d, 2 * d)
        self.down = torch.nn.Linear(2 * d, d)
        self.drop = torch.nn.Dropout(mlp_dropout)
        # PEFT style: Identity when the rate is 0, Dropout otherwise.
        self.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Dropout(lora_dropout) if lora_dropout else torch.nn.Identity()})

    def forward(self, h):
        h = h + self.attn(h)
        return h + self.drop(self.down(torch.nn.functional.gelu(self.up(self.lora_dropout["default"](h)))))


def _step(G, layers, x, preserve = True, wrap = "bound"):
    h = x
    for layer in layers:
        function = {"bound": layer.forward, "module": layer, "partial": functools.partial(layer.__call__)}[wrap]
        h = G.UnslothCheckpointFunction.apply(function, preserve, h)
    loss = h.square().mean()
    loss.backward()
    grads = [p.grad.clone() for l in layers for p in l.parameters()] + [x.grad.clone()]
    for l in layers: l.zero_grad(set_to_none = True)
    x.grad = None
    return loss.detach(), grads


def _layers(seed = 0, **kwargs):
    torch.manual_seed(seed)
    return torch.nn.ModuleList([_Layer(**kwargs) for _ in range(3)])


def test_checkpoint_module_unwraps_common_callables(G):
    layer = _Layer()
    assert G._checkpoint_module(layer) is layer
    assert G._checkpoint_module(layer.forward) is layer
    assert G._checkpoint_module(functools.partial(layer.__call__, foo = 1)) is layer
    for kwargs in ({"scale": 2}, {"scale": torch.ones(1, requires_grad = True)}):  # partial / tensor keyword binder
        bound, _ = G._bind_checkpoint_kwargs(layer.forward, kwargs)
        assert G._checkpoint_module(bound) is layer
    assert G._checkpoint_module(lambda h: h * 2) is None


@pytest.mark.parametrize("kwargs, training, draws", [
    ({}, True, False),
    ({"mlp_dropout": 0.1}, True, True),
    ({"mlp_dropout": 0.1}, False, False),
    ({"attention_dropout": 0.1}, True, True),
    ({"attention_dropout": 0.1}, False, False),
    ({"lora_dropout": 0.05}, True, True),
])
def test_random_draw_detection(G, kwargs, training, draws):
    layer = _Layer(**kwargs).train(training)
    assert G._scan_draws_random(layer) is draws


def test_detection_reads_config_rates_and_router_jitter(G):
    layer = _Layer()
    layer.attn.config = types.SimpleNamespace(attention_dropout = 0.1)
    assert G._scan_draws_random(layer)
    layer = _Layer()
    layer.attn.router_jitter_noise = 0.01
    assert G._scan_draws_random(layer)


def test_snapshot_is_kept_when_the_flag_is_off_or_the_callable_is_unknown(G, monkeypatch):
    layer = _Layer()
    assert G._rng_snapshot_needed(layer.forward)  # flag off
    monkeypatch.setattr(G, "GC_SKIP_RNG_WITHOUT_DROPOUT", True)
    assert not G._rng_snapshot_needed(layer.forward)
    assert G._rng_snapshot_needed(lambda h: h)


@pytest.mark.parametrize("wrap", ["bound", "module", "partial"])
def test_skipping_the_snapshot_without_dropout_is_bitwise_identical(G, monkeypatch, wrap):
    layers = _layers()
    torch.manual_seed(7)
    x = torch.randn(2, 5, 16, requires_grad = True)
    torch.manual_seed(123)
    ref_loss, ref_grads = _step(G, layers, x, wrap = wrap)
    ref_rng = torch.get_rng_state()
    monkeypatch.setattr(G, "GC_SKIP_RNG_WITHOUT_DROPOUT", True)
    torch.manual_seed(123)
    loss, grads = _step(G, layers, x, wrap = wrap)
    assert G.GC_RNG_STATS["skipped"] == 3 and G.GC_RNG_STATS["kept"] == 0
    assert torch.equal(loss, ref_loss)
    for a, b in zip(grads, ref_grads): assert torch.equal(a, b)
    assert torch.equal(torch.get_rng_state(), ref_rng)  # the global RNG stream is untouched


@pytest.mark.parametrize("kwargs", [{"mlp_dropout": 0.2}, {"attention_dropout": 0.2}, {"lora_dropout": 0.2}])
def test_live_dropout_keeps_the_snapshot_and_the_exact_masks(G, monkeypatch, kwargs):
    layers = _layers(**kwargs)
    torch.manual_seed(7)
    x = torch.randn(2, 5, 16, requires_grad = True)
    torch.manual_seed(123)
    ref_loss, ref_grads = _step(G, layers, x)
    monkeypatch.setattr(G, "GC_SKIP_RNG_WITHOUT_DROPOUT", True)
    torch.manual_seed(123)
    loss, grads = _step(G, layers, x)
    assert G.GC_RNG_STATS["kept"] == 3 and G.GC_RNG_STATS["skipped"] == 0
    assert torch.equal(loss, ref_loss)
    for a, b in zip(grads, ref_grads): assert torch.equal(a, b)


def test_tiled_mlp_skips_the_snapshot_only_without_dropout(G, monkeypatch):
    try:
        from unsloth_zoo.tiled_mlp import TiledMLP
    except Exception as e:
        pytest.skip(f"tiled_mlp unavailable: {e}")

    def run(mlp):
        torch.manual_seed(5)
        x = torch.randn(2, 8, 16, requires_grad = True)
        torch.manual_seed(9)
        out = TiledMLP.apply(mlp.forward, mlp, x, True, 4, None)
        out = out[0] if isinstance(out, tuple) else out
        out.square().mean().backward()
        return [p.grad.clone() for p in mlp.parameters()] + [x.grad.clone()], torch.get_rng_state()

    for rate, expect in ((0.0, "skipped"), (0.3, "kept")):
        torch.manual_seed(0)
        mlp = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Dropout(rate), torch.nn.Linear(32, 16))
        monkeypatch.setattr(G, "GC_SKIP_RNG_WITHOUT_DROPOUT", False)
        ref, ref_rng = run(mlp)
        mlp.zero_grad(set_to_none = True)
        monkeypatch.setattr(G, "GC_SKIP_RNG_WITHOUT_DROPOUT", True)
        before = dict(G.GC_RNG_STATS)
        got, rng = run(mlp)
        assert G.GC_RNG_STATS[expect] == before[expect] + 1
        for a, b in zip(got, ref): assert torch.equal(a, b)
        assert torch.equal(rng, ref_rng)


def test_recompute_check_is_off_by_default_and_quiet_when_deterministic(G, monkeypatch):
    layers = _layers()
    x = torch.randn(2, 5, 16, requires_grad = True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _step(G, layers, x)
        monkeypatch.setattr(G, "GC_RECOMPUTE_CHECK", "1")
        _step(G, layers, x)
        layers_drop = _layers(mlp_dropout = 0.3)
        _step(G, layers_drop, x)  # dropout with the RNG snapshot recomputes the same masks
    assert G.GC_RECOMPUTE_MISMATCHES == []


class _Drifting(torch.nn.Module):
    """Routes differently on its second call, like an MoE router whose choice changes between forward and recompute."""
    def __init__(self):
        super().__init__()
        self.a, self.b = torch.nn.Linear(16, 16), torch.nn.Linear(16, 16)
        self.calls = 0

    def forward(self, h):
        self.calls += 1
        return (self.a if self.calls % 2 else self.b)(h)


@pytest.mark.parametrize("mode", ["1", "raise"])
def test_recompute_check_catches_routing_drift(G, monkeypatch, mode):
    monkeypatch.setattr(G, "GC_RECOMPUTE_CHECK", mode)
    layer = _Drifting()
    x = torch.randn(2, 5, 16, requires_grad = True)
    y = G.UnslothCheckpointFunction.apply(layer.forward, True, x)
    if mode == "raise":
        with pytest.raises(RuntimeError, match = "recompute differs"):
            y.sum().backward()
    else:
        with pytest.warns(UserWarning, match = "recompute differs"):
            y.sum().backward()
    assert len(G.GC_RECOMPUTE_MISMATCHES) == 1


def test_recompute_check_catches_lost_rng_state(G, monkeypatch):
    # Dropout recomputed without the RNG snapshot draws different masks: exactly what skipping it wrongly would do.
    monkeypatch.setattr(G, "GC_RECOMPUTE_CHECK", "1")
    layers = _layers(mlp_dropout = 0.3)
    x = torch.randn(2, 5, 16, requires_grad = True)
    with pytest.warns(UserWarning, match = "recompute differs"):
        _step(G, layers, x, preserve = False)
    assert len(G.GC_RECOMPUTE_MISMATCHES) == 3
