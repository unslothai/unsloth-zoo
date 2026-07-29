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

Adam keeps two fp32 buffers per parameter (``m``, ``v``), so optimizer state is
2x the trainable-parameter bytes. Storing ``m`` as 8-bit packed arrays cuts that
materially at no measurable cost to the loss curve.

ONLY the first moment is quantized. The second moment stays fp32, deliberately:
``v`` is a running mean of SQUARED gradients, so its values span many orders of
magnitude and sit near zero early in training. MLX's ``mx.quantize`` is an affine
per-group scheme, and the reconstruction error near zero is large relative to the
value; ``sqrt(v)`` then lands in the update denominator and the step explodes.
Measured on a 2-layer MLP, quantizing ``v`` diverges within three steps:

    1.0590 -> 0.8627 -> 19.3755 -> 10.2466 -> nan

both for int8 m + int8 v and for fp32 m + int8 v. Quantizing ``m`` alone is
stable: final loss 0.08318 against an fp32 baseline of 0.08317 on the same seed.
Making ``v`` quantizable needs a dynamic/exponential mapping with stochastic
rounding (what bitsandbytes implements); MLX exposes no such primitive, so this
module does not pretend to offer it.

Two MLX-specific constraints are load-bearing here. Do not "simplify" them away:

1. Optimizer state must contain ONLY ``mx.array`` leaves. ``Optimizer.state`` is
   walked by ``tree_flatten`` (for checkpointing) and by ``mx.compile``'s
   inputs/outputs. A plain Python ``bool`` stored alongside the moments raises
   ``TypeError: object of type 'bool' has no len()``. Whether a moment is
   quantized is therefore inferred from its VALUE, never from a stored flag.

2. A quantized moment is the 3-tuple ``mx.quantize`` returns, but
   ``mlx.utils.tree_map`` (used by ``Optimizer.apply_gradients``) and
   ``tree_unflatten`` (used when reloading a checkpoint) rebuild containers, so
   the triple comes back as a LIST. Checking ``isinstance(m, tuple)`` alone
   silently misses the reloaded case and fails with
   ``TypeError: can't multiply sequence by non-int of type 'float'``. Always test
   ``(tuple, list)``.

Verified against ``mx.compile`` with ``inputs=state, outputs=state`` (how
MLXTrainer compiles its step): compiled and uncompiled runs are bit-identical.
The quantized triple flattens to ``<param>.m.0/.1/.2``, all arrays, so
``save_optimizer_state`` / ``load_optimizer_state`` round-trip it unchanged and a
resumed step reproduces the continuous one exactly.
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
    """Store ``state["m"]`` 8-bit packed; delegate the update math to the parent.

    ``apply_single`` dequantizes into the state, calls the parent (so bias
    correction and AdamW's decoupled decay stay byte-for-byte the stock
    implementation), then requantizes what the parent wrote back.
    """

    def _init_quantization(self, group_size, bits):
        # Plain instance attributes, NOT optimizer state: state holds arrays only
        # (see the module docstring). self.betas / self.eps are stored the same way.
        bits = int(bits)
        if bits != DEFAULT_BITS:
            # Rejected rather than offered untested. Optimizer state here is
            # demonstrably quantization-sensitive, not theoretically so: affine
            # 8-bit quantization of the SECOND moment already diverges to NaN
            # (1.0590 -> 0.8627 -> 19.3755 -> nan). A narrower width for the first
            # moment needs its own convergence evidence before it ships.
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
        """``mx.quantize`` needs a 2-D array whose last dim divides group_size.

        Everything else (biases, norms, embeddings with an odd width) keeps an
        fp32 moment, so a model with no eligible parameter still trains -- it
        just saves nothing.
        """
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
    """One-line description of what an 8-bit optimizer request actually gets.

    The historical behaviour was to rewrite ``adamw_8bit`` to plain ``adamw``
    with no message, so users asking for 8-bit silently got fp32 state. Replacing
    that with a quieter surprise would be no better, so state the shape of the
    saving explicitly.
    """
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
