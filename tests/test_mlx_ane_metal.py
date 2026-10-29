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

"""The Neural Engine / GPU split of dense SwiGLU MLPs, on real MLX and the real Neural Engine."""

from __future__ import annotations

import concurrent.futures

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_simulation import mlx_is_simulated  # noqa: E402

if mlx_is_simulated():
    pytest.skip("needs real MLX and the Neural Engine runtime", allow_module_level = True)

import mlx.nn as nn  # noqa: E402

from unsloth_zoo.mlx import ane  # noqa: E402

if not ane.available():
    pytest.skip("no Neural Engine runtime", allow_module_level = True)

HIDDEN, CHANNELS, TILE = 256, 1024, 64


class SwiGLU(nn.Module):
    def __init__(self, act = nn.silu, hidden = HIDDEN):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, CHANNELS, bias = False)
        self.up_proj = nn.Linear(hidden, CHANNELS, bias = False)
        self.down_proj = nn.Linear(CHANNELS, hidden, bias = False)
        self.act = act

    def __call__(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class Model(nn.Module):
    def __init__(self, act = nn.silu, tower = False):
        super().__init__()
        self.tower = [SwiGLU(act, hidden = 2 * HIDDEN)] if tower else []   # a wider vision MLP, visited first
        self.layers = [SwiGLU(act), SwiGLU(act)]


def _model(act = nn.silu, seed = 0, tower = False):
    mx.random.seed(seed)
    model = Model(act, tower)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size = 64, bits = 4)
    model.eval()
    return model


def _rows(n, seed = 1):
    return (mx.random.normal((1, n, HIDDEN), key = mx.random.key(seed)) * 0.5).astype(mx.bfloat16)


def _cosine(a, b):
    a, b = a.astype(mx.float32).reshape(-1), b.astype(mx.float32).reshape(-1)
    return ((a * b).sum() / mx.sqrt((a * a).sum() * (b * b).sum())).item()


@pytest.fixture
def prepared(monkeypatch):
    monkeypatch.setenv(ane.OPT_IN, "1")
    monkeypatch.setattr(ane, "_DOWN_SLAB", 256)   # several down slabs at this size
    model = _model()
    status = ane.prepare_ane_prefill(model, tile = TILE, share = 0.5)
    assert status.layers == 2 and status.reason == ""
    model.__dict__["_unsloth_ane_prepared"] = True
    yield model
    ane.release_ane_prefill(model)


def test_split_tiles_match_the_native_mlp_and_the_rest_stays_native(prepared):
    mlp = prepared.layers[1]
    x = _rows(2 * TILE + 5)
    native = mlp(x)
    with ane.ane_prefill_mlp(prepared):
        split = mlp(x)
        decode = mlp(x[:, :3])
    assert type(mlp) is SwiGLU
    assert split.shape == native.shape and split.dtype == native.dtype
    assert _cosine(split[:, :2 * TILE], native[:, :2 * TILE]) > 0.999
    error = mx.abs(split[:, :2 * TILE] - native[:, :2 * TILE]).max() / mx.abs(native[:, :2 * TILE]).max()
    assert error.item() < 0.05   # cosine alone would accept a scaled output
    assert not mx.array_equal(split[:, :2 * TILE], native[:, :2 * TILE]).item()   # the split did run
    assert mx.array_equal(split[:, 2 * TILE:], mlp(x[:, 2 * TILE:])).item()       # the short tail did not
    assert mx.array_equal(decode, mlp(x[:, :3])).item()


def test_a_partial_tile_worth_padding_takes_the_split(prepared):
    mlp = prepared.layers[1]
    mlp.__dict__[ane._STATE].pad_from = TILE // 2
    x, alone, short = _rows(TILE + TILE // 2 + 3), _rows(TILE // 2, seed = 2), _rows(TILE + TILE // 2 - 1, seed = 3)
    with ane.ane_prefill_mlp(prepared):
        split, split_alone, split_short = mlp(x), mlp(alone), mlp(short)
    for got, native in ((split[:, TILE:], mlp(x[:, TILE:])), (split_alone, mlp(alone))):
        assert got.shape == native.shape and _cosine(got, native) > 0.999
        assert (mx.abs(got - native).max() / mx.abs(native).max()).item() < 0.05
        assert not mx.array_equal(got, native).item()
    assert mx.array_equal(split_short[:, TILE:], mlp(short[:, TILE:])).item()   # a tail below the threshold


def test_each_split_reads_its_own_layer(prepared):
    x = _rows(TILE)
    natives = [layer(x) for layer in prepared.layers]
    with ane.ane_prefill_mlp(prepared):
        splits = [layer(x) for layer in reversed(prepared.layers)][::-1]
    for native, split, other in zip(natives, splits, natives[::-1]):
        assert _cosine(split, native) > 0.999 > _cosine(split, other)
        assert not mx.array_equal(split, native).item()   # each layer ran its own split, not the native fallback
    assert all(ane._STATE in layer.__dict__ for layer in prepared.layers)


def test_a_failed_evaluation_falls_back_to_the_gpu_for_good(prepared, monkeypatch):
    split = prepared.layers[0].__dict__[ane._STATE]

    def fail():
        raise ane.AneError("injected")

    calls = []
    monkeypatch.setattr(split.request, "run", lambda: (calls.append(1), fail()))
    x = _rows(TILE)
    with ane.ane_prefill_mlp(prepared):
        out, again = prepared.layers[0](x), prepared.layers[0](x)
    assert split.failed and len(calls) == 1
    assert mx.array_equal(out, prepared.layers[0](x)).item() and mx.array_equal(again, out).item()


def test_a_timed_out_program_is_never_unloaded_and_stops_every_split(prepared, monkeypatch):
    import time

    monkeypatch.setattr(ane, "_ABANDONED", [])
    split, other = (layer.__dict__[ane._STATE] for layer in prepared.layers)
    calls = []
    monkeypatch.setattr(ane, "_WATCHDOG", 0.05)
    monkeypatch.setattr(split.request, "run", lambda: time.sleep(0.5))
    monkeypatch.setattr(other.request, "run", lambda: calls.append(1))
    torn_down = []

    class Spy:   # records a finalizer that actually releases its native object
        def __init__(self, real):
            self.real = real

        def __call__(self):
            if self.real.alive:
                torn_down.append(self)
            return self.real()

        def detach(self):
            return self.real.detach()

    owners = (split.program, split.request, split.surfaces.input, split.surfaces.output)
    for owner in owners:
        monkeypatch.setattr(owner, "_finalizer", Spy(owner._finalizer))
    x = _rows(TILE)
    with ane.ane_prefill_mlp(prepared):
        prepared.layers[0](x)
        out = prepared.layers[1](x)
    assert not calls and not other.failed and mx.array_equal(out, prepared.layers[1](x)).item()
    ane.release_ane_prefill(prepared)
    assert split.failed and split in ane._ABANDONED
    assert not torn_down and not any(owner._finalizer.real.alive for owner in owners)   # nor at exit
    time.sleep(0.5)   # let the worker finish before the next test queues behind it


def test_only_the_common_mlp_shape_is_attached():
    model = _model(tower = True)
    assert ane.prepare_ane_prefill(model, tile = TILE, share = 0.5).layers == 2
    assert ane._STATE not in model.tower[0].__dict__
    ane.release_ane_prefill(model)


@pytest.mark.parametrize("error", [ane.AneError, concurrent.futures.TimeoutError])
def test_a_failed_calibration_leaves_the_model_native(error, monkeypatch, tmp_path):
    monkeypatch.setattr(ane, "_CACHE_PATH", str(tmp_path / "cache.json"))

    def fail(self):
        raise error("injected")

    monkeypatch.setattr(ane._Split, "evaluate", fail)
    model = _model()
    status = ane.prepare_ane_prefill(model, tile = TILE)
    assert status.layers == 0 and "injected" in status.reason
    assert all(ane._STATE not in layer.__dict__ for layer in model.layers)


@pytest.mark.parametrize("replaced", ["projection", "unquantized", "scales"])
def test_a_replaced_projection_detaches_the_split(prepared, replaced):
    mlp = prepared.layers[0]
    if replaced == "projection":
        mlp.gate_proj = nn.QuantizedLinear(HIDDEN, CHANNELS, bias = False, group_size = 64, bits = 4)
    elif replaced == "unquantized":   # like an adapter wrapper: no scales or biases
        mlp.gate_proj = nn.Linear(HIDDEN, CHANNELS, bias = False)
        mlp.gate_proj.set_dtype(mx.bfloat16)
    else:
        mlp.down_proj.scales = mlp.down_proj.scales * 2
    x = _rows(TILE)
    with ane.ane_prefill_mlp(prepared):
        out = mlp(x)
    assert ane._STATE not in mlp.__dict__ and mx.array_equal(out, mlp(x)).item()


def test_overlapping_scopes_keep_the_split_until_the_last_exit(prepared):
    first, second = ane.ane_prefill_mlp(prepared), ane.ane_prefill_mlp(prepared)
    first.__enter__()
    second.__enter__()
    first.__exit__(None, None, None)
    assert type(prepared.layers[0]) is not SwiGLU
    second.__exit__(None, None, None)
    assert type(prepared.layers[0]) is SwiGLU


def test_only_swiglu_mlps_are_attached(monkeypatch):
    model = _model(act = nn.gelu)
    assert ane.prepare_ane_prefill(model, tile = TILE, share = 0.5).layers == 0
    assert all(ane._STATE not in layer.__dict__ for layer in model.layers)


def _projection(bits, group_size = 64, inputs = CHANNELS, outputs = HIDDEN, seed = 5):
    mx.random.seed(seed)
    linear = nn.Linear(inputs, outputs, bias = False)
    linear.set_dtype(mx.bfloat16)
    return nn.QuantizedLinear.from_linear(linear, group_size = group_size, bits = bits)


def _down(bits, group_size = 64):
    return _projection(bits, group_size)


@pytest.mark.parametrize("change", ["training", "mixed_group_size"])
def test_ineligible_projections_are_not_attached(change):
    model = _model()
    if change == "training":
        model.layers[0].train()
    else:
        model.layers[0].down_proj = _down(4, group_size = 32)
    assert ane.prepare_ane_prefill(model, tile = TILE, share = 0.5).layers == 1
    assert ane._STATE not in model.layers[0].__dict__ and ane._STATE in model.layers[1].__dict__
    ane.release_ane_prefill(model)


def test_a_projection_with_more_bits_is_split_and_costed_at_its_own(monkeypatch):
    monkeypatch.setattr(ane, "_DOWN_SLAB", 256)
    model = _model()
    mixed, uniform = model.layers
    mixed.down_proj = _down(5)   # as mixed-precision checkpoints raise single down projections
    assert ane.prepare_ane_prefill(model, tile = TILE, share = 0.5).layers == 2
    model.__dict__["_unsloth_ane_prepared"] = True
    x = _rows(TILE)
    native = mixed(x)
    with ane.ane_prefill_mlp(model):
        split = mixed(x)
    assert _cosine(split, native) > 0.999 and not mx.array_equal(split, native).item()
    assert (mx.abs(split - native).max() / mx.abs(native).max()).item() < 0.05
    assert mixed.__dict__[ane._STATE].rest[2].bits == 5
    k = CHANNELS // 2
    assert ane._split_bytes(mixed, k) - ane._split_bytes(uniform, k) == HIDDEN * (CHANNELS - k) // 8
    ane.release_ane_prefill(model)


@pytest.mark.parametrize("budget_layers", [5, 1])
def test_calibration_runs_on_the_most_common_quantization_and_keys_on_every_projection(budget_layers, monkeypatch,
                                                                                         tmp_path):
    monkeypatch.setattr(ane, "_CACHE_PATH", str(tmp_path / "cache.json"))
    mx.random.seed(0)
    model = Model()
    model.layers += [SwiGLU() for _ in range(3)]
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size = 64, bits = 4)
    model.eval()
    for end in (0, -1):   # the first layer visited in either order is the minority quantization
        model.layers[end].down_proj = _down(5)
    calibrated, closed, close = [], [], ane._Split.close

    def calibrate(m, mlp, tile, surfaces):
        calibrated.append(ane._Split(mlp, CHANNELS // 2, tile, surfaces))
        return 0.5, 10.0, 6.0, calibrated[-1]

    monkeypatch.setattr(ane, "_calibrate", calibrate)
    monkeypatch.setattr(ane._Split, "close", lambda self: (closed.append(self), close(self)))
    cost = ane._split_bytes(model.layers[0], CHANNELS // 2)   # the dearest layer: every layer fits within it
    monkeypatch.setattr(ane, "_memory_budget", lambda: budget_layers * cost)
    assert ane.prepare_ane_prefill(model, tile = TILE).layers == budget_layers
    assert len(calibrated) == 1
    serving = [layer for layer in model.layers if layer.__dict__.get(ane._STATE) is calibrated[0]]
    if budget_layers == 5:   # the calibration split serves the uniform layer it was measured on, and only that one
        assert len(serving) == 1 and ane._bits(serving[0]) == (4, 4, 4) and not closed
    else:   # the probe was not admitted, so its split is closed
        assert not serving and closed == [calibrated[0]]
    ane.release_ane_prefill(model)
    keys = {ane._machine_key(model.layers[i], TILE) for i in (0, 1, 2, 3)}
    model.layers[2].up_proj = _projection(5, inputs = HIDDEN, outputs = CHANNELS)
    model.layers[3].gate_proj = _projection(5, inputs = HIDDEN, outputs = CHANNELS)
    assert len(keys) == 2 and len({ane._machine_key(model.layers[i], TILE) for i in (0, 1, 2, 3)}) == 4


@pytest.mark.parametrize("split_ms,share", [(9.6, 0.0), (6.0, 0.625)])
def test_calibration_takes_the_balanced_share_only_with_a_gain_and_caches_it(split_ms, share, monkeypatch, tmp_path):
    monkeypatch.setattr(ane, "_CACHE_PATH", str(tmp_path / "cache.json"))
    # GPU MLP 10 ms; the Neural Engine 2.5 ms at half the channels balances at 2/3, aligned down to 640 of 1024.
    timings = iter([10.0, 2.5, split_ms])
    monkeypatch.setattr(ane, "_median_ms", lambda function, **_: (function(), next(timings))[1])
    status = ane.prepare_ane_prefill(_model(), tile = TILE)
    assert (status.layers, status.share) == ((2, share) if share else (0, 0.0))
    monkeypatch.setattr(ane, "_median_ms", None)   # a cached decision measures nothing
    model = _model(seed = 3)
    assert ane.prepare_ane_prefill(model, tile = TILE).layers == (2 if share else 0)
    if share:   # a tile split in 6 of the GPU's 10 ms: padding pays from ceil(64 * 0.6) rows
        assert [layer.__dict__[ane._STATE].pad_from for layer in model.layers] == [39, 39]
    ane.release_ane_prefill(model)


def test_free_memory_admits_layers_by_their_own_cost(monkeypatch):
    mx.random.seed(0)
    model = Model(nn.silu, False)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size = 64, bits = 4, class_predicate = lambda path, m: hasattr(m, "to_quantized") and (
        {"group_size": 64, "bits": 8} if path.startswith("layers.0") else True))
    model.eval()
    # int8 gate/up/down rows for 512 channels, plus the GPU's copy of down's other 512 columns and its scales/biases
    four, eight = ane._split_bytes(model.layers[1], 512), ane._split_bytes(model.layers[0], 512)
    assert four == 3 * 512 * 256 + 256 * 512 // 2 + 2 * 256 * 8 * 2 and eight == four + 256 * 512 // 2
    for budget, layers in ((four + eight - 1, 1), (four + eight, 2), (min(four, eight) - 1, 0)):
        monkeypatch.setattr(ane, "_memory_budget", lambda: budget)
        status = ane.prepare_ane_prefill(model, tile = TILE, share = 0.5)
        assert status.layers == layers and (status.reason == "not enough free memory") == (not layers)
        assert sum(ane._STATE in layer.__dict__ for layer in model.layers) == layers
        ane.release_ane_prefill(model)


def test_the_budget_is_half_of_what_the_model_leaves(monkeypatch):
    monkeypatch.setattr(ane.mx, "device_info", lambda: {"memory_size": 1000})
    monkeypatch.setattr(ane.mx, "get_active_memory", lambda: 250)
    assert ane._memory_budget() == 300


def test_a_calibration_split_no_layer_has_memory_for_is_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(ane, "_CACHE_PATH", str(tmp_path / "cache.json"))
    timings = iter([10.0, 2.5, 6.0])
    monkeypatch.setattr(ane, "_median_ms", lambda function, **_: (function(), next(timings))[1])
    monkeypatch.setattr(ane, "_memory_budget", lambda: 0)
    closed, close = [], ane._Split.close
    monkeypatch.setattr(ane._Split, "close", lambda self: (closed.append(self.k), close(self)))
    assert ane.prepare_ane_prefill(_model(), tile = TILE).reason == "not enough free memory"
    assert closed == [640]


def test_scope_is_opt_in():
    from unsloth_zoo.mlx.inference import INFERENCE_SCOPES

    (scope,) = [s for s in INFERENCE_SCOPES if s.enter is ane.ane_prefill_mlp]
    assert scope.opt_in == ane.OPT_IN
