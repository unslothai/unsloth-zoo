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

"""8-bit optimizer state for MLX: quantized Adam/AdamW FIRST moment.

Adam holds two moment buffers per parameter, so state is 2x the trainable bytes.
Packing ``m`` to 8 bits cuts that by ``1 - (f*(1/b + 2/G) + (1-f) + 1)/2`` for a
dtype of ``b`` bytes, group size ``G`` and an eligible byte fraction ``f``.
Measured end to end on SmolLM2-135M, group_size=64:

    full fine-tune, float32 state   35.9%     LoRA r=32/r=64   35.9%
    full fine-tune, bfloat16 state  23.4%     LoRA r=16        34.6%

Only the first two are the "8-bit vs 32-bit" figure. MLX creates moments with
``mx.zeros_like(parameter)``, so a bfloat16 model has bfloat16 moments and the
packed moment is being compared against 2 bytes, not 4.

Eligibility is measured in elements, not shape: moments are quantized over a
flattened view, so a rank-16 LoRA_B and a 3-D MoE expert stack pack as readily as
a square matrix. An earlier rule required 2-D with a divisible last axis, which
silently excluded LoRA_B below rank 64 and held r=16/r=32 to 18.3%. Anything under
``MIN_QUANTIZE_SIZE`` elements keeps a full-width moment: those are the norms and
biases, worth no bytes and the least tolerant of the error. bitsandbytes draws the
same line at the same threshold (``min_8bit_size``).

That floor is why r=16 lands at 34.6% rather than 35.9%: under grouped-query
attention the k/v projections are narrow, so their LoRA_B is ``(kv_dim, r)`` and at
r=16 that is 3072 elements on a 135M model. The figure is therefore model-shape
dependent below r=32; a square fixture will report the full 35.9% and overstate it.

This shrinks PERSISTENT state, not peak allocation: ``apply_single`` materialises
a full-width ``m`` per parameter per step, and peak rose 11.7% on Metal and 29.9%
on the CPU backend in a 135M full fine-tune. Loss impact over 200 steps is 0.44%
MAPE full fine-tune and 0.76% LoRA against the same seed and data order.

``v`` is deliberately NOT quantized. It is a running mean of squared gradients, so
it spans orders of magnitude and sits near zero early; MLX's affine ``mx.quantize``
reconstructs poorly there, ``sqrt(v)`` lands in the denominator, and the step
explodes - measured, for both int8 m+v and fp32 m + int8 v:

    1.0590 -> 0.8627 -> 19.3755 -> 10.2466 -> nan

That needs bitsandbytes-style dynamic quantization, which MLX does not expose.
``v`` keeps whatever dtype the parameter has; it is unquantized, not float32.

Three constraints are load-bearing; do not "simplify" them away:

1. Optimizer state must hold ONLY ``mx.array`` leaves - it is walked by
   ``tree_flatten`` (checkpointing) and ``mx.compile``. A Python ``bool`` beside
   the moments raises ``TypeError: object of type 'bool' has no len()``, so
   group_size/bits live on the instance and are checkpointed as safetensors
   metadata by ``save_optimizer_state``.

2. ``mx.quantize`` returns a LIST on mlx 0.32, and ``tree_unflatten`` rebuilds the
   triple as a list too, so ``isinstance(m, tuple)`` alone fails with
   ``TypeError: can't multiply sequence by non-int of type 'float'``. Test
   ``(tuple, list)``.

3. Whether to re-pack is decided from ``is_quantizable(parameter)``, never from
   the current type of ``state["m"]``. A checkpoint written by plain AdamW loads
   an unpacked array, and inferring from it would leave that moment full width
   for the rest of the run while the banner claims otherwise.

Compiled and uncompiled runs agree under ``inputs=state, outputs=state``. The
triple flattens to ``<param>.m.0/.1/.2``, so save/load_optimizer_state round-trip
it and a resumed step reproduces the next parameters and state exactly.
"""

import mlx.core as mx
import mlx.optimizers as optim

__all__ = [
    "QuantizedMomentAdam",
    "QuantizedMomentAdamW",
    "DEFAULT_GROUP_SIZE",
    "DEFAULT_BITS",
    "SUPPORTED_GROUP_SIZES",
    "MIN_QUANTIZE_SIZE",
    "unpack_quantized_moments",
    "state_has_quantized_moments",
    "announce_quantized_optimizer",
]

DEFAULT_GROUP_SIZE = 64
DEFAULT_BITS = 8
# mx.quantize's supported group sizes; all three are exercised by the test suite.
SUPPORTED_GROUP_SIZES = (32, 64, 128)
# Below this, packing a moment saves bytes that do not matter on parameters that
# do not tolerate the error well. Matches bitsandbytes' min_8bit_size default.
MIN_QUANTIZE_SIZE = 4096
# Names already announced in this process; see announce_quantized_optimizer.
_ANNOUNCED = set()


def _as_int(value, name):
    """``int()`` truncates, so ``bits=8.9`` would pass as 8. Reject instead."""
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"Unsloth: {name} must be an integer, got {value!r}.")
    return int(value)


class _QuantizedFirstMomentMixin:
    """Store ``state["m"]`` 8-bit packed; delegate the update math to the parent,
    so bias correction and AdamW's decoupled decay stay the stock implementation."""

    def _init_quantization(self, group_size, bits):
        # Instance attributes, NOT optimizer state (which holds arrays only).
        bits = _as_int(bits, "bits")
        if bits != DEFAULT_BITS:
            # Rejected rather than offered untested: this state is demonstrably
            # quantization-sensitive (see the second-moment NaN trace above).
            raise ValueError(
                f"Unsloth: quantized optimizer state supports bits={DEFAULT_BITS} "
                f"only, got bits={bits}. Optimizer moments are quantization-"
                "sensitive (affine 8-bit quantization of the second moment already "
                "diverges to NaN), so narrower widths are rejected until measured."
            )
        group_size = _as_int(group_size, "group_size")
        if group_size not in SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"Unsloth: quantized optimizer state supports group_size in "
                f"{SUPPORTED_GROUP_SIZES}, got group_size={group_size}."
            )
        self.group_size = group_size
        self.bits = bits

    def is_quantizable(self, parameter):
        """``mx.quantize`` constrains the LAST axis only, so a flattened view makes
        eligibility a question of element count rather than shape. That admits 3-D
        MoE expert stacks and LoRA_B at rank 16/32, which the previous 2-D rule
        rejected while their partner matrices were packed.

        Tensors under ``MIN_QUANTIZE_SIZE`` stay full width: they are a rounding
        error in the state budget but include the norm and bias parameters, which
        are the ones least worth adding quantization error to. bitsandbytes draws
        the same line at the same threshold (``min_8bit_size``)."""
        return (
            parameter.size >= MIN_QUANTIZE_SIZE
            and parameter.size % self.group_size == 0
        )

    def _pack(self, moment):
        """Quantize over a flat view; the parameter's shape is restored on unpack."""
        return mx.quantize(
            moment.reshape(1, -1), group_size=self.group_size, bits=self.bits,
        )

    def _unpack(self, packed, shape):
        """``mx.dequantize`` is self-describing, so this also reads checkpoints
        written by the earlier per-row layout, whose triple carries its own shape."""
        moment = mx.dequantize(
            *packed, group_size=self.group_size, bits=self.bits,
        )
        return moment.reshape(shape)

    def init_single(self, parameter, state):
        super().init_single(parameter, state)
        if self.is_quantizable(parameter):
            state["m"] = self._pack(state["m"])

    def apply_single(self, gradient, parameter, state):
        packed = state["m"]
        # tuple OR list: mx.quantize and tree_unflatten both give a list.
        if isinstance(packed, (tuple, list)):
            moment = self._unpack(packed, parameter.shape)
            # Affine dequantization does not land exactly on zero, and v == 0
            # means this coordinate has only seen zero gradients, so its true
            # moment is zero. Without this the residue is divided by eps alone.
            state["m"] = mx.where(state["v"] == 0, mx.zeros_like(moment), moment)
        updated = super().apply_single(gradient, parameter, state)
        # Re-pack from eligibility, not from what was loaded: a checkpoint
        # written by plain AdamW arrives unpacked and must not stay that way.
        if self.is_quantizable(parameter):
            state["m"] = self._pack(state["m"])
        return updated


class QuantizedMomentAdam(_QuantizedFirstMomentMixin, optim.Adam):
    """Adam with an 8-bit first moment. Second moment is left unquantized."""

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
    """AdamW with an 8-bit first moment. Second moment is left unquantized."""

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
    """One line. The only thing a user cannot infer from the name is that this
    quantizes ONE moment, where bitsandbytes' ``adamw_8bit`` quantizes both.
    Eligibility, dtype and memory behaviour live in this module's docstring; they
    are reference material and do not belong in stdout on every run."""
    return (
        f"Unsloth: {optimizer_name} on MLX quantizes Adam's first moment to 8-bit "
        f"(group_size={group_size}); the second moment is unquantized."
    )


def announce_quantized_optimizer(
    optimizer_name, group_size = DEFAULT_GROUP_SIZE, enabled = True,
):
    """Print the routing line at most once per process. Returns whether it printed.

    ``_build_optimizer`` runs per training run and per rank, so an unguarded print
    put five wrapped lines on screen for every rank and every re-run of a notebook
    cell, for what is the default optimizer in every Unsloth example.
    """
    if not enabled or optimizer_name in _ANNOUNCED:
        return False
    _ANNOUNCED.add(optimizer_name)
    print(describe_quantized_optimizer(optimizer_name, group_size))
    return True


def unpack_quantized_moments(state, group_size = DEFAULT_GROUP_SIZE, bits = DEFAULT_BITS):
    """Dequantize every packed first moment in a loaded state tree.

    Lets a checkpoint written by a quantized optimizer load into a plain
    ``optim.Adam``/``AdamW``, which would otherwise reach ``b1 * m`` on a list and
    raise ``TypeError: can't multiply sequence by non-int of type 'float'`` from
    inside mlx with no mention of quantization.

    Applies the same ``v == 0`` mask ``apply_single`` does. Carrying the residue out
    to a plain optimizer would reintroduce the blow-up there: measured 5.96e-08 of
    residue moving inactive coordinates by 5.4e-02 over 200 zero-gradient steps.
    """
    if isinstance(state, dict):
        out = {}
        for key, value in state.items():
            if key == "m" and isinstance(value, (tuple, list)) and len(value) == 3:
                moment = mx.dequantize(*value, group_size = group_size, bits = bits)
                second = state.get("v")
                # ``v`` is the only full-width leaf here, so it carries the shape a
                # flat-packed moment has to be restored to before the mask lines up.
                if isinstance(second, mx.array) and second.size == moment.size:
                    moment = moment.reshape(second.shape)
                    moment = mx.where(second == 0, mx.zeros_like(moment), moment)
                out[key] = moment
            else:
                out[key] = unpack_quantized_moments(value, group_size, bits)
        return out
    if isinstance(state, list):
        return [unpack_quantized_moments(v, group_size, bits) for v in state]
    return state


def state_has_quantized_moments(state):
    """True if any ``m`` leaf in the tree is a packed triple."""
    if isinstance(state, dict):
        return any(
            (key == "m" and isinstance(value, (tuple, list)) and len(value) == 3)
            or state_has_quantized_moments(value)
            for key, value in state.items()
        )
    if isinstance(state, list):
        return any(state_has_quantized_moments(v) for v in state)
    return False
