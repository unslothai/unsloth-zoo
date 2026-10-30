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

"""8-bit Adam/AdamW first moment; v stays unquantized (quantized v goes NaN); state holds only mx.array leaves
(tree_flatten, mx.compile); mx.quantize returns a list on mlx 0.32, so test (tuple, list); re-pack from
is_quantizable, never from the loaded type; saves persistent state, not peak memory."""

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
# Below this many elements, packing saves bytes that do not matter on parameters that tolerate error poorly; matches bitsandbytes min_8bit_size default.
MIN_QUANTIZE_SIZE = 4096
# Names already announced in this process; see announce_quantized_optimizer.
_ANNOUNCED = set()


def _as_int(value, name):
    """``int()`` truncates, so ``bits=8.9`` would pass as 8. Reject instead."""
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"Unsloth: {name} must be an integer, got {value!r}.")
    return int(value)


class _QuantizedFirstMomentMixin:
    """Stores state["m"] 8-bit packed and delegates the update math to the parent, keeping bias correction and AdamW decoupled decay stock."""

    def _init_quantization(self, group_size, bits):
        # Instance attributes, NOT optimizer state (which holds arrays only).
        bits = _as_int(bits, "bits")
        if bits != DEFAULT_BITS:
            # Other bits are rejected, not offered untested: this state is quantization-sensitive (second-moment NaN trace in the module docstring).
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
        """Eligibility is by element count, not shape: mx.quantize only constrains the last axis, so flattened moments
        admit 3-D expert stacks and LoRA_B at rank 16/32; tensors under MIN_QUANTIZE_SIZE (norms, biases) stay full
        width, as bitsandbytes min_8bit_size does."""
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
        """Also reads checkpoints from the earlier per-row layout, since mx.dequantize is self-describing and the triple carries its own shape."""
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
            # Affine dequantization does not land exactly on zero; v == 0 means only zero gradients were seen, so the
            # true moment is zero; otherwise the residue is divided by eps alone.
            state["m"] = mx.where(state["v"] == 0, mx.zeros_like(moment), moment)
        updated = super().apply_single(gradient, parameter, state)
        # Re-pack from is_quantizable, not the loaded type: a checkpoint written by plain AdamW arrives unpacked and must not stay that way.
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
    """Prints the routing line once per process, returning whether it printed; _build_optimizer runs per run and rank,
    so an unguarded print repeats per rank and notebook re-run."""
    if not enabled or optimizer_name in _ANNOUNCED:
        return False
    _ANNOUNCED.add(optimizer_name)
    print(describe_quantized_optimizer(optimizer_name, group_size))
    return True


def unpack_quantized_moments(state, group_size = DEFAULT_GROUP_SIZE, bits = DEFAULT_BITS):
    """Dequantizes packed first moments so a quantized checkpoint loads into plain optim.Adam/AdamW (else b1 * m on a
    list raises TypeError); applies the v == 0 mask as apply_single does, since carried residue would move inactive
    coordinates."""
    if isinstance(state, dict):
        out = {}
        for key, value in state.items():
            if key == "m" and isinstance(value, (tuple, list)) and len(value) == 3:
                moment = mx.dequantize(*value, group_size = group_size, bits = bits)
                second = state.get("v")
                # v is the only full-width leaf, so it carries the shape a flat-packed moment must be restored to before the v == 0 mask lines up.
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
