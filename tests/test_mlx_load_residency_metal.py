# SPDX-License-Identifier: AGPL-3.0-only
"""A finished load must own its weights, measured as the memory evaluating them allocates.
Several layers, so evaluating only the first cannot reach the total."""

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_simulation import mlx_is_simulated

if mlx_is_simulated():
    pytest.skip("Requires native MLX", allow_module_level = True)
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason = "Requires Metal")

import sys

import mlx.nn as nn
from mlx.utils import tree_flatten
from unsloth_zoo.mlx.loader import _MLXQuantizationSpec, _apply_mlx_quantization, _finish_load

_SIDES = (2048, 1024, 512)
_BYTES = sum(side * side * 4 for side in _SIDES)
_SMALLEST_LAYER = min(side * side * 4 for side in _SIDES)


@pytest.fixture(autouse = True)
def _real_mlx_still_installed():
    # A sibling file's torch shim would make these measure nothing.
    if sys.modules.get("mlx.core") is not mx:
        pytest.skip("another test replaced mlx.core with the simulation shim")


def _lazy_model(tmp_path):
    path = str(tmp_path / "weights.safetensors")
    mx.save_safetensors(path, {
        str(i): mx.random.normal((side, side)).astype(mx.float32)
        for i, side in enumerate(_SIDES)
    })
    mx.clear_cache()
    mapped = mx.load(path)
    model = nn.Sequential(*(nn.Linear(side, side, bias = False) for side in _SIDES))
    for i, layer in enumerate(model.layers):
        layer.update({"weight": mapped[str(i)]})
    return model


def test_runtime_quantization_reads_each_weight_before_its_kernel(tmp_path, monkeypatch):
    """A quantize kernel evaluated while its source is still being read keeps a GPU command
    buffer waiting on the disk: by then, reading each source must allocate nothing."""
    quantize, evaluate = mx.quantize, mx.eval
    pending, reads = {}, []

    def recording_quantize(weight, *args, **kwargs):
        quantized = quantize(weight, *args, **kwargs)
        pending[id(quantized[0])] = (quantized[0], weight)  # held, so the id is never reused
        return quantized

    def checking_eval(*arrays):
        for _, array in tree_flatten(list(arrays)):
            _, source = pending.pop(id(array), (None, None))
            if source is not None:
                before = mx.get_active_memory()
                evaluate(source)
                reads.append(mx.get_active_memory() - before)
        return evaluate(*arrays)

    model = _lazy_model(tmp_path)
    monkeypatch.setattr(mx, "quantize", recording_quantize)
    monkeypatch.setattr(mx, "eval", checking_eval)
    spec = _MLXQuantizationSpec(enabled = True, bits = 4, group_size = 64)
    model, _ = _apply_mlx_quantization(model, {}, spec, is_vlm = False)
    mx.eval(model.parameters())  # as the load branches do after quantizing
    assert len(reads) == len(_SIDES) and not any(reads), reads


def test_finish_load_returns_resident_weights(tmp_path):
    model = _lazy_model(tmp_path)
    before = mx.get_active_memory()
    returned, tokenizer = _finish_load(model, "tokenizer")
    assert (returned, tokenizer) == (model, "tokenizer")
    assert mx.get_active_memory() - before >= _BYTES


def test_lazy_weights_env_leaves_the_load_lazy(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_MLX_LAZY_WEIGHTS", "1")
    model = _lazy_model(tmp_path)
    before = mx.get_active_memory()
    _finish_load(model, None)
    # Under the smallest layer, not under the total: any one layer evaluated is a leak.
    assert mx.get_active_memory() - before < _SMALLEST_LAYER


def test_unwalkable_weights_report_but_do_not_fail_the_load(tmp_path, capsys):
    model = _lazy_model(tmp_path)

    def parameters():
        raise RuntimeError("cannot walk")

    model.parameters = parameters
    assert _finish_load(model, None) == (model, None)
    assert "cannot walk" in capsys.readouterr().out


def test_an_unreadable_weight_reports_but_does_not_fail_the_load(tmp_path, capsys, monkeypatch):
    model = _lazy_model(tmp_path)

    def refuse(*_args, **_kwargs):
        raise RuntimeError("cannot read shard")

    monkeypatch.setattr(mx, "eval", refuse)
    assert _finish_load(model, None) == (model, None)
    assert "cannot read shard" in capsys.readouterr().out
