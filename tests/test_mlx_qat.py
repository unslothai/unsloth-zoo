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

"""QAT for MLX LoRA: parity with fuse(), straight-through gradients, guards.

The load-bearing test is `test_qat_forward_matches_fused_module`: QAT is only
meaningful if the quantizer it simulates is the one `merged_4bit` actually
writes, so expectations are taken from calling the real `fuse()` rather than
from re-deriving the merge arithmetic here. A test that recomputed the merge
would share any bug with the implementation and pass while proving nothing.
"""

import importlib
import sys

import pytest


def _real_mlx_runtime():
    """True only on the genuine mlx stack.

    Sibling test files install the mlx_simulation stub process-wide with no
    teardown, so a bare `import mlx` can succeed against the shim.
    """
    try:
        lora = importlib.import_module("mlx_lm.tuner.lora")
    except Exception:
        return False
    if not isinstance(getattr(lora, "LoRALinear", None), type):
        return False
    origin = getattr(sys.modules.get("mlx.core"), "__file__", "") or ""
    return "mlx_simulation" not in origin


if not _real_mlx_runtime():
    pytest.skip("needs the real mlx runtime", allow_module_level=True)

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner.lora import LoRALinear

from unsloth_zoo.mlx.qat import (
    apply_mlx_qat,
    mlx_qat_module_count,
    remove_mlx_qat,
)

DIMS = 128
GROUP_SIZE = 64
BITS = 4


class _Holder(nn.Module):
    """Minimal container exposing named_modules() over one LoRA layer."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer


def _make_lora(bias=False, bits=BITS, dropout=0.0, quantized=True, seed=0):
    mx.random.seed(seed)
    base = nn.Linear(DIMS, DIMS, bias=bias)
    if quantized:
        base = nn.QuantizedLinear.from_linear(
            base, group_size=GROUP_SIZE, bits=bits, mode="affine",
        )
    layer = LoRALinear.from_base(base, r=8, scale=2.0, dropout=dropout)
    # lora_b initialises to zeros; a zero delta would make the merge trivial.
    layer.lora_b = mx.random.normal(layer.lora_b.shape) * 0.05
    mx.eval(layer.lora_b, layer.lora_a)
    return layer


# --------------------------------------------------------------------------
# Parity with the artifact that actually ships
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bias", [False, True])
def test_qat_forward_matches_fused_module(bias):
    """QAT's forward must equal the module `merged_4bit` writes to disk.

    Expectations come from the real `fuse(dequantize=False)`, not from
    re-deriving `quantize(dequantize(W) + BA)` in the test.
    """
    layer = _make_lora(bias=bias)
    fused = layer.fuse(dequantize=False)

    apply_mlx_qat(_Holder(layer), "auto")

    x = mx.random.normal((4, DIMS), dtype=mx.float32)
    qat_out, fused_out = layer(x), fused(x)
    mx.eval(qat_out, fused_out)

    # Residual is the dense-dequantised matmul vs quantized_matmul difference.
    max_delta = float(mx.abs(qat_out - fused_out).max())
    scale = float(mx.abs(fused_out).max())
    assert max_delta <= 1e-4 * max(scale, 1.0), (
        f"QAT forward diverges from fuse(): max|d|={max_delta}"
    )


def test_qat_forward_includes_base_bias():
    """A dropped bias term still 'runs' — it must be caught explicitly.

    Regression guard: Qwen2/2.5 carry biases on q/k/v projections, and omitting
    them shifts the loss without raising.
    """
    layer = _make_lora(bias=True)
    base_bias = mx.array(layer.linear.bias)
    assert float(mx.abs(base_bias).max()) > 0.0

    apply_mlx_qat(_Holder(layer), "auto")
    x = mx.zeros((1, DIMS), dtype=mx.float32)
    out = layer(x)
    mx.eval(out)
    # With a zero input the output is exactly the bias.
    assert float(mx.abs(out[0] - base_bias).max()) <= 1e-5


def test_qat_perturbs_the_forward():
    """Sanity: QAT must actually change the forward, else parity is vacuous."""
    layer = _make_lora()
    x = mx.random.normal((4, DIMS), dtype=mx.float32)
    before = layer(x)
    mx.eval(before)

    apply_mlx_qat(_Holder(layer), "auto")
    after = layer(x)
    mx.eval(after)
    assert float(mx.abs(before - after).max()) > 1e-4


# --------------------------------------------------------------------------
# Straight-through estimator
# --------------------------------------------------------------------------

def test_straight_through_jacobian_is_identity():
    """The fake-quant must be transparent to the backward pass.

    The guarantee is on the *Jacobian* of the STE expression: d/dw of
    `w + stop_gradient(quant(w) - w)` is 1. Note this is not the same as the
    downstream loss gradient matching an unquantized run — that one is
    evaluated at the fake-quantized value and legitimately differs by the
    quantization error, which is exactly the signal QAT trains against.
    """
    weight = mx.random.normal((DIMS, DIMS))

    def ste_sum(w):
        packed, scales, biases = mx.quantize(
            w, group_size=GROUP_SIZE, bits=BITS, mode="affine")
        fake = mx.dequantize(
            packed, scales, biases,
            group_size=GROUP_SIZE, bits=BITS, mode="affine")
        return (w + mx.stop_gradient(fake - w)).sum()

    grad = mx.grad(ste_sum)(weight)
    mx.eval(grad)
    assert float(mx.abs(grad - mx.ones_like(grad)).max()) <= 1e-6

    # And the forward really is the quantized value, not a pass-through.
    packed, scales, biases = mx.quantize(
        weight, group_size=GROUP_SIZE, bits=BITS, mode="affine")
    fake = mx.dequantize(
        packed, scales, biases,
        group_size=GROUP_SIZE, bits=BITS, mode="affine")
    mx.eval(fake)
    assert float(mx.abs(fake - weight).max()) > 0.0


def test_gradients_reach_lora_parameters_under_qat():
    layer = _make_lora()
    apply_mlx_qat(_Holder(layer), "auto")
    x = mx.random.normal((4, DIMS), dtype=mx.float32)

    def loss(a, b):
        layer.lora_a, layer.lora_b = a, b
        return (layer(x) ** 2).sum()

    ga, gb = mx.grad(loss, argnums=(0, 1))(layer.lora_a, layer.lora_b)
    mx.eval(ga, gb)
    assert float(mx.abs(ga).max()) > 0.0
    assert float(mx.abs(gb).max()) > 0.0


# --------------------------------------------------------------------------
# Apply / remove lifecycle
# --------------------------------------------------------------------------

def test_apply_is_idempotent_and_removable():
    layer = _make_lora()
    holder = _Holder(layer)
    x = mx.random.normal((4, DIMS), dtype=mx.float32)
    stock = layer(x)
    mx.eval(stock)

    assert apply_mlx_qat(holder, "auto") == 1
    assert apply_mlx_qat(holder, "auto") == 0      # already patched
    assert mlx_qat_module_count(holder) == 1

    assert remove_mlx_qat(holder) == 1
    assert mlx_qat_module_count(holder) == 0
    restored = layer(x)
    mx.eval(restored)
    assert float(mx.abs(stock - restored).max()) == 0.0


def test_qat_preserves_class_name():
    """The save path keys on type(module).__name__; the stand-in must match."""
    layer = _make_lora()
    original_name = type(layer).__name__
    apply_mlx_qat(_Holder(layer), "auto")
    assert type(layer).__name__ == original_name
    assert isinstance(layer, LoRALinear)


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "scheme", ["int8-int4", "fp8-int4", "fp8-fp8", "cactus"])
def test_torchao_only_schemes_are_rejected(scheme):
    layer = _make_lora()
    with pytest.raises(NotImplementedError, match="torchao"):
        apply_mlx_qat(_Holder(layer), scheme)


def test_unknown_scheme_is_rejected():
    layer = _make_lora()
    with pytest.raises(ValueError, match="unsupported qat_scheme"):
        apply_mlx_qat(_Holder(layer), "int3")


def test_non_string_scheme_is_rejected():
    layer = _make_lora()
    with pytest.raises(TypeError):
        apply_mlx_qat(_Holder(layer), 4)


def test_scheme_bits_must_match_the_base_quantization():
    """int8 QAT on a 4-bit base would simulate a grid fuse() never writes."""
    layer = _make_lora(bits=4)
    with pytest.raises(ValueError, match="8-bit"):
        apply_mlx_qat(_Holder(layer), "int8")


def test_matching_scheme_bits_are_accepted():
    layer = _make_lora(bits=4)
    assert apply_mlx_qat(_Holder(layer), "int4") == 1


def test_unquantized_base_is_rejected():
    layer = _make_lora(quantized=False)
    with pytest.raises(ValueError, match="requires a quantized base"):
        apply_mlx_qat(_Holder(layer), "auto")


def test_lora_dropout_is_rejected():
    layer = _make_lora(dropout=0.1)
    with pytest.raises(NotImplementedError, match="lora_dropout=0"):
        apply_mlx_qat(_Holder(layer), "auto")


def test_model_without_lora_is_rejected():
    holder = _Holder(nn.Linear(DIMS, DIMS))
    with pytest.raises(ValueError, match="no LoRA layers"):
        apply_mlx_qat(holder, "auto")


def test_mixed_quantization_grids_are_rejected():
    class _Two(nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.a, self.b = a, b

    holder = _Two(_make_lora(bits=4, seed=0), _make_lora(bits=8, seed=1))
    with pytest.raises(ValueError, match="single quantization grid"):
        apply_mlx_qat(holder, "auto")


def test_full_finetuning_is_rejected():
    holder = _Holder(_make_lora())
    holder._unsloth_full_finetuning = True
    with pytest.raises(NotImplementedError, match="full_finetuning"):
        apply_mlx_qat(holder, "auto")


def test_dora_is_rejected():
    dora = pytest.importorskip("mlx_lm.tuner.dora")
    base = nn.QuantizedLinear.from_linear(
        nn.Linear(DIMS, DIMS, bias=False),
        group_size=GROUP_SIZE, bits=BITS, mode="affine")
    layer = dora.DoRALinear.from_base(base, r=8, scale=2.0)
    with pytest.raises(NotImplementedError, match="DoRA"):
        apply_mlx_qat(_Holder(layer), "auto")


def test_switch_linear_moe_is_rejected():
    switch_layers = pytest.importorskip("mlx_lm.models.switch_layers")
    from mlx_lm.tuner.lora import LoRASwitchLinear

    base = switch_layers.SwitchLinear(DIMS, DIMS, num_experts=2, bias=False)
    base = base.to_quantized(group_size=GROUP_SIZE, bits=BITS, mode="affine")
    layer = LoRASwitchLinear.from_base(base, r=8, scale=2.0)
    with pytest.raises(NotImplementedError, match="MoE|SwitchLinear"):
        apply_mlx_qat(_Holder(layer), "auto")
