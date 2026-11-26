# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""dtype=None reports the float32 weights it casts down, and a 16-bit LoRA merge rounds the fused weight once
(unslothai/unsloth#19834). Runs on any real MLX build, CPU included."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
from mlx_simulation import mlx_is_simulated

if mlx_is_simulated():
    pytest.skip("Requires native MLX", allow_module_level = True)

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRALinear

from unsloth_zoo.mlx.loader import _convert_mlx_dtype
from unsloth_zoo.mlx.utils import _fuse_mlx_module


class _Block(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.proj = nn.Linear(64, 32)
        self.norm = nn.RMSNorm(32)
        self.set_dtype(dtype)


def _dtypes(model):
    return {str(v.dtype) for _, v in tree_flatten(model.parameters())}


def test_default_dtype_reports_float32_weights_cast_down(capsys):
    model = _Block(mx.float32)
    _convert_mlx_dtype(model, mx.bfloat16, default_dtype = True)
    out = capsys.readouterr().out
    assert _dtypes(model) == {"mlx.core.bfloat16"}
    assert "dtype=None" in out and "3 float32 tensors" in out and "dtype='float32'" in out


def test_explicit_dtype_and_16bit_checkpoints_stay_quiet(capsys):
    explicit = _Block(mx.float32)
    _convert_mlx_dtype(explicit, mx.bfloat16)
    half = _Block(mx.float16)
    _convert_mlx_dtype(half, mx.bfloat16, default_dtype = True)
    kept = _Block(mx.float32)
    _convert_mlx_dtype(kept, mx.float32, default_dtype = True)
    assert "dtype=None" not in capsys.readouterr().out
    assert _dtypes(explicit) == _dtypes(half) == {"mlx.core.bfloat16"}
    assert _dtypes(kept) == {"mlx.core.float32"}


def _trained_lora(dtype, bias = False, seed = 0):
    mx.random.seed(seed)
    base = nn.Linear(256, 128, bias = bias)
    base.set_dtype(dtype)
    lora = LoRALinear.from_base(base, r = 16, scale = 2.0)
    # Trained factors stay float32, as in the issue; small enough that the delta sits near the base weight's 16-bit step.
    lora.lora_a = mx.random.normal(lora.lora_a.shape) * 0.02
    lora.lora_b = mx.random.normal(lora.lora_b.shape) * 0.002
    mx.eval(lora.parameters())
    return lora


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_fused_16bit_weight_is_rounded_once(dtype):
    lora = _trained_lora(dtype)
    exact = lora.linear.weight.astype(mx.float32) + (lora.scale * lora.lora_b.T) @ lora.lora_a.T

    fused = _fuse_mlx_module(lora, dequantize = True)
    upstream = LoRALinear.fuse(lora, dequantize = True)

    assert type(fused) is nn.Linear and fused.weight.dtype == dtype
    assert mx.array_equal(fused.weight, exact.astype(dtype)).item()
    once = mx.abs(fused.weight.astype(mx.float32) - exact)
    twice = mx.abs(upstream.weight.astype(mx.float32) - exact)
    assert mx.all(once <= twice).item()
    assert mx.mean(once).item() < mx.mean(twice).item()


def test_fuse_keeps_bias_and_leaves_float32_and_quantized_bases_to_mlx_lm():
    with_bias = _trained_lora(mx.bfloat16, bias = True)
    fused = _fuse_mlx_module(with_bias, dequantize = True)
    assert mx.array_equal(fused.bias, with_bias.linear.bias).item()

    full = _trained_lora(mx.float32)
    assert mx.array_equal(_fuse_mlx_module(full, dequantize = True).weight,
                          LoRALinear.fuse(full, dequantize = True).weight).item()

    quantized = LoRALinear.from_base(nn.QuantizedLinear.from_linear(nn.Linear(256, 128)), r = 8)
    assert isinstance(_fuse_mlx_module(quantized, dequantize = False), nn.QuantizedLinear)
