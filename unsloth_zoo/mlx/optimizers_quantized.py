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

"""8-bit optimizer state for MLX: quantized Adam/AdamW FIRST moment.

Adam holds two fp32 buffers per parameter, so state is 2x the trainable bytes.
Packing ``m`` to 8 bits cuts that at no measurable cost: final loss 0.08318 vs an
fp32 baseline of 0.08317 on the same seed.

``v`` is deliberately NOT quantized. It is a running mean of squared gradients, so
it spans orders of magnitude and sits near zero early; MLX's affine ``mx.quantize``
reconstructs poorly there, ``sqrt(v)`` lands in the denominator, and the step
explodes - measured, for both int8 m+v and fp32 m + int8 v:

    1.0590 -> 0.8627 -> 19.3755 -> 10.2466 -> nan

That needs bitsandbytes-style dynamic quantization, which MLX does not expose.

Two constraints are load-bearing; do not "simplify" them away:

1. Optimizer state must hold ONLY ``mx.array`` leaves - it is walked by
   ``tree_flatten`` (checkpointing) and ``mx.compile``. A Python ``bool`` beside
   the moments raises ``TypeError: object of type 'bool' has no len()``, so
   whether a moment is quantized is inferred from its value, never a stored flag.

2. ``tree_map`` and ``tree_unflatten`` rebuild the quantized 3-tuple as a LIST, so
   ``isinstance(m, tuple)`` alone misses the reloaded case and fails with
   ``TypeError: can't multiply sequence by non-int of type 'float'``. Test
   ``(tuple, list)``.

Compiled and uncompiled runs are bit-identical under ``inputs=state,
outputs=state``. The triple flattens to ``<param>.m.0/.1/.2``, so
save/load_optimizer_state round-trip it and a resumed step reproduces exactly.
"""

import mlx.core as mx
import mlx.optimizers as optim

__all__ = [
    "QuantizedMomentAdam",
    "QuantizedMomentAdamW",
    "DEFAULT_GROUP_SIZE",
    "DEFAULT_BITS",
    "SUPPORTED_GROUP_SIZES",
]

DEFAULT_GROUP_SIZE = 64
DEFAULT_BITS = 8
# mx.quantize's supported group sizes; all three are exercised by the test suite.
SUPPORTED_GROUP_SIZES = (32, 64, 128)


class _QuantizedFirstMomentMixin:
    """Store ``state["m"]`` 8-bit packed; delegate the update math to the parent,
    so bias correction and AdamW's decoupled decay stay the stock implementation."""

    def _init_quantization(self, group_size, bits):
        # Instance attributes, NOT optimizer state (which holds arrays only).
        bits = int(bits)
        if bits != DEFAULT_BITS:
            # Rejected rather than offered untested: this state is demonstrably
            # quantization-sensitive (see the second-moment NaN trace above).
            raise ValueError(
                f"Unsloth: quantized optimizer state supports bits={DEFAULT_BITS} "
                f"only, got bits={bits}. Optimizer moments are quantization-"
                "sensitive (affine 8-bit quantization of the second moment already "
                "diverges to NaN), so narrower widths are rejected until measured."
            )
        group_size = int(group_size)
        if group_size not in SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"Unsloth: quantized optimizer state supports group_size in "
                f"{SUPPORTED_GROUP_SIZES}, got group_size={group_size}."
            )
        self.group_size = group_size
        self.bits = bits

    def is_quantizable(self, parameter):
        """``mx.quantize`` needs 2-D with last dim divisible by group_size.
        Everything else keeps an fp32 moment and still trains, saving nothing."""
        return parameter.ndim == 2 and parameter.shape[-1] % self.group_size == 0

    def init_single(self, parameter, state):
        super().init_single(parameter, state)
        if self.is_quantizable(parameter):
            state["m"] = mx.quantize(
                state["m"], group_size=self.group_size, bits=self.bits,
            )

    def apply_single(self, gradient, parameter, state):
        packed = state["m"]
        # tuple OR list: tree_map/tree_unflatten rebuild the triple as a list.
        quantized = isinstance(packed, (tuple, list))
        if quantized:
            state["m"] = mx.dequantize(
                *packed, group_size=self.group_size, bits=self.bits,
            )
        updated = super().apply_single(gradient, parameter, state)
        if quantized:
            state["m"] = mx.quantize(
                state["m"], group_size=self.group_size, bits=self.bits,
            )
        return updated


class QuantizedMomentAdam(_QuantizedFirstMomentMixin, optim.Adam):
    """Adam with an 8-bit first moment. Second moment stays fp32."""

    def __init__(
        self,
        learning_rate,
        betas = [0.9, 0.999],
        eps = 1e-8,
        bias_correction = False,
        group_size = DEFAULT_GROUP_SIZE,
        bits = DEFAULT_BITS,
    ):
        super().__init__(
            learning_rate = learning_rate,
            betas = betas,
            eps = eps,
            bias_correction = bias_correction,
        )
        self._init_quantization(group_size, bits)


class QuantizedMomentAdamW(_QuantizedFirstMomentMixin, optim.AdamW):
    """AdamW with an 8-bit first moment. Second moment stays fp32."""

    def __init__(
        self,
        learning_rate,
        betas = [0.9, 0.999],
        eps = 1e-8,
        weight_decay = 0.01,
        bias_correction = False,
        group_size = DEFAULT_GROUP_SIZE,
        bits = DEFAULT_BITS,
    ):
        super().__init__(
            learning_rate = learning_rate,
            betas = betas,
            eps = eps,
            weight_decay = weight_decay,
            bias_correction = bias_correction,
        )
        self._init_quantization(group_size, bits)


def describe_quantized_optimizer(optimizer_name, group_size = DEFAULT_GROUP_SIZE):
    """What an 8-bit request actually gets. The old behaviour rewrote
    ``adamw_8bit`` to ``adamw`` silently; a quieter surprise would be no better."""
    return (
        f"Unsloth: {optimizer_name} on MLX quantizes the optimizer's FIRST moment "
        f"to 8-bit (group_size={group_size}); the second moment stays float32 "
        "(affine 8-bit quantization of the second moment diverges). Only 2-D "
        f"parameters whose last dimension is a multiple of {group_size} are "
        "quantized; all others keep float32 moments."
    )

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
