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

"""MLX merged saves fuse LoRA / DoRA / QLoRA in float32 and round once (stochastic by default, as the torch merge),
and dtype=None says when it casts a float32 checkpoint down. Needs the real mlx runtime (the CPU build is enough)."""

import importlib
import sys

import pytest


def _real_mlx_runtime():
    try:
        importlib.import_module("mlx_lm.tuner.lora")
    except Exception:
        return False
    origin = getattr(sys.modules.get("mlx.core"), "__file__", "") or ""
    return "mlx_simulation" not in origin


if not _real_mlx_runtime():
    pytest.skip("needs the real mlx runtime", allow_module_level=True)

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import SwitchLinear
from mlx_lm.tuner.dora import DoRALinear
from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear

from unsloth_zoo.mlx.loader import _convert_mlx_dtype
from unsloth_zoo.mlx.utils import _fuse_mlx_module

IN, OUT, R = 256, 128, 8


def _factors(module, rel, seed=0):
    rng = np.random.default_rng(seed)
    module.lora_a = mx.array(rng.standard_normal(module.lora_a.shape).astype(np.float32) / np.sqrt(IN))
    module.lora_b = mx.array(rng.standard_normal(module.lora_b.shape).astype(np.float32) * rel)
    mx.eval(module.parameters())
    return module


def _linear(dtype, bias=False, seed=0):
    mx.random.seed(seed)
    base = nn.Linear(IN, OUT, bias=bias)
    base.weight = mx.random.normal(base.weight.shape) * 0.02
    base.set_dtype(dtype)
    return base


def _exact(module, weight=None):
    w = (module.linear.weight if weight is None else weight).astype(mx.float32)
    return w + (module.scale * module.lora_b.T) @ module.lora_a.T


def _kept(fused, base_w, delta):
    """Fraction of the float32 delta the stored weight keeps (1.0 = all of it on average)."""
    stored = fused.astype(mx.float32) - base_w.astype(mx.float32)
    return (mx.sum(stored * delta) / mx.sum(delta * delta)).item()


@pytest.fixture
def nearest(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MERGE_ROUNDING", "nearest")


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_dense_lora_is_the_float32_sum_rounded_once(nearest, dtype):
    lora = _factors(LoRALinear.from_base(_linear(dtype, bias=True), r=R, scale=2.0), rel=0.01)
    exact = _exact(lora)
    fused = _fuse_mlx_module(lora, True)
    twice = LoRALinear.fuse(lora, dequantize=True).weight
    assert type(fused) is nn.Linear and fused.weight.dtype == dtype
    assert mx.array_equal(fused.weight, exact.astype(dtype)).item()
    assert not mx.array_equal(twice, exact.astype(dtype)).item()  # mlx-lm rounds the delta, then the sum
    assert mx.array_equal(fused.bias, lora.linear.bias).item()


def test_stochastic_default_keeps_a_delta_below_half_a_bf16_step(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MERGE_ROUNDING", raising=False)
    lora = _factors(LoRALinear.from_base(_linear(mx.bfloat16), r=R, scale=2.0), rel=0.00003)
    # |W| kept away from 0, so every element of this delta is under half a bf16 step of its weight.
    sign = mx.where(mx.random.uniform(shape=(OUT, IN)) < 0.5, -1.0, 1.0)
    lora.linear.weight = (sign * mx.random.uniform(0.015, 0.03, (OUT, IN))).astype(mx.bfloat16)
    w, exact = lora.linear.weight, _exact(lora)
    delta = exact - w.astype(mx.float32)
    sr = _fuse_mlx_module(lora, True).weight
    assert sr.dtype == mx.bfloat16
    assert 0.9 < _kept(sr, w, delta) < 1.1
    assert _kept(LoRALinear.fuse(lora, dequantize=True).weight, w, delta) < 0.5
    monkeypatch.setenv("UNSLOTH_MERGE_ROUNDING", "rtn")
    assert _kept(_fuse_mlx_module(lora, True).weight, w, delta) < 0.5
    # Reproducible per key, and only ever one of the two neighbours of the float32 value.
    monkeypatch.delenv("UNSLOTH_MERGE_ROUNDING")
    again = _fuse_mlx_module(lora, True).weight
    assert mx.array_equal(sr, again).item()
    assert mx.all(mx.abs(sr.astype(mx.float32) - exact) <= mx.abs(exact) * 2.0 ** -7).item()
    assert not mx.array_equal(sr, _fuse_mlx_module(lora, True, key="other.q_proj").weight).item()


def test_stochastic_float16_is_unbiased_and_stays_between_neighbours(monkeypatch):
    from unsloth_zoo.mlx.utils import _mlx_stochastic_round
    x = mx.array(np.float32(1.0) + np.arange(4096, dtype=np.float32) * np.float32(2.0 ** -14))
    out = _mlx_stochastic_round(x, mx.float16, "k").astype(mx.float32)
    step = 2.0 ** -10
    assert mx.all(mx.abs(out - x) < step).item()
    assert abs(mx.mean(out - x).item()) < step / 20
    assert not mx.array_equal(out, x.astype(mx.float16).astype(mx.float32)).item()
    special = mx.array([0.0, -0.0, float("inf"), float("-inf"), 1e-9, -1e-9])
    assert mx.all(mx.isfinite(_mlx_stochastic_round(special, mx.float16, "s")[:2])).item()
    assert np.isnan(np.array(_mlx_stochastic_round(mx.array([float("nan")]), mx.float16, "n"))).all()


def test_qlora_merges_onto_float32_dequant_once_and_merged_4bit_is_untouched(nearest):
    q = nn.QuantizedLinear.from_linear(_linear(mx.bfloat16), group_size=64, bits=4)
    lora = _factors(LoRALinear.from_base(q, r=R, scale=2.0), rel=0.01)
    w32 = mx.dequantize(q.weight, q.scales, q.biases, group_size=64, bits=4, dtype=mx.float32)
    fused = _fuse_mlx_module(lora, True)
    assert type(fused) is nn.Linear and fused.weight.dtype == q.scales.dtype
    assert mx.array_equal(fused.weight, _exact(lora, w32).astype(q.scales.dtype)).item()
    assert isinstance(_fuse_mlx_module(lora, False), nn.QuantizedLinear)


def test_dora_fuses_in_float32_and_stays_16_bit(nearest):
    dora = _factors(DoRALinear.from_base(_linear(mx.bfloat16), r=R, scale=2.0), rel=0.01)
    adapted = _exact(dora)
    want = (dora.m / mx.linalg.norm(adapted, axis=1))[:, None] * adapted
    fused = _fuse_mlx_module(dora, True)
    assert fused.weight.dtype == mx.bfloat16  # mlx-lm's DoRA fuse promotes to float32 through the fp32 magnitude
    assert mx.array_equal(fused.weight, want.astype(mx.bfloat16)).item()


def test_switch_and_embedding_lora_fuse_in_float32(nearest):
    mx.random.seed(1)
    sw = SwitchLinear(IN, OUT, 4)
    sw.set_dtype(mx.bfloat16)
    lsw = LoRASwitchLinear.from_base(sw, r=R, scale=2.0)
    lsw.lora_b = mx.random.normal(lsw.lora_b.shape) * 0.01
    want = sw.weight.astype(mx.float32) + (lsw.scale * lsw.lora_b) @ lsw.lora_a.reshape(4, -1, IN)
    fused = _fuse_mlx_module(lsw, True)
    assert type(fused) is SwitchLinear and mx.array_equal(fused.weight, want.astype(mx.bfloat16)).item()

    emb = nn.Embedding(64, IN)
    emb.set_dtype(mx.bfloat16)
    lemb = LoRAEmbedding.from_base(emb, r=R, scale=2.0)
    lemb.lora_b = mx.random.normal(lemb.lora_b.shape) * 0.01
    want = emb.weight.astype(mx.float32) + (lemb.scale * lemb.lora_a) @ lemb.lora_b
    fused = _fuse_mlx_module(lemb, True)
    assert type(fused) is nn.Embedding and mx.array_equal(fused.weight, want.astype(mx.bfloat16)).item()


def test_float32_base_is_the_exact_float32_sum():
    lora = _factors(LoRALinear.from_base(_linear(mx.float32), r=R, scale=2.0), rel=0.01)
    fused = _fuse_mlx_module(lora, True).weight
    assert fused.dtype == mx.float32 and mx.allclose(fused, _exact(lora), rtol=0, atol=1e-7).item()


def test_unknown_rounding_value_warns_and_rounds_to_nearest(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MERGE_ROUNDING", "banana")
    lora = _factors(LoRALinear.from_base(_linear(mx.bfloat16), r=R, scale=2.0), rel=0.01)
    with pytest.warns(UserWarning, match="UNSLOTH_MERGE_ROUNDING"):
        fused = _fuse_mlx_module(lora, True).weight
    assert mx.array_equal(fused, _exact(lora).astype(mx.bfloat16)).item()


class _Block(nn.Module):
    def __init__(self, dtype, extra_fp32=False):
        super().__init__()
        self.proj = nn.Linear(IN, OUT)
        self.norm = nn.RMSNorm(OUT)
        self.set_dtype(dtype)
        if extra_fp32:
            self.gate = nn.Linear(4, 4)  # a small float32 tensor inside a 16-bit checkpoint


def _dtypes(model):
    return {str(v.dtype) for _, v in tree_flatten(model.parameters())}


def test_dtype_none_reports_a_float32_checkpoint_cast_down(capsys):
    model = _Block(mx.float32)
    _convert_mlx_dtype(model, mx.bfloat16, default_dtype=True)
    out = capsys.readouterr().out
    assert _dtypes(model) == {"mlx.core.bfloat16"}
    assert "dtype=None" in out and "3 float32 tensors" in out and "dtype='float32'" in out


def test_explicit_dtype_16bit_checkpoints_and_stray_float32_tensors_stay_quiet(capsys):
    _convert_mlx_dtype(_Block(mx.float32), mx.bfloat16)
    _convert_mlx_dtype(_Block(mx.float16), mx.bfloat16, default_dtype=True)
    _convert_mlx_dtype(_Block(mx.bfloat16, extra_fp32=True), mx.bfloat16, default_dtype=True)
    kept = _Block(mx.float32)
    _convert_mlx_dtype(kept, mx.float32, default_dtype=True)
    assert "float32 tensors" not in capsys.readouterr().out
    assert _dtypes(kept) == {"mlx.core.float32"}
