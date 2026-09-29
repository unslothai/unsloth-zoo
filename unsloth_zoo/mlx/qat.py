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

"""MLX LoRA QAT: fake-quantize the merged W + BA at the base grid with an STE,
matching what LoRALinear.fuse(dequantize=False) writes for merged_4bit.
Not the CUDA/torchao base-only target: fuse() quantizes W + BA, not W.
DoRA, MoE, VLMs, unquantized bases, dropout and torchao-only schemes are refused.
"""

from __future__ import annotations

__all__ = (
    "SUPPORTED_MLX_QAT_SCHEMES",
    "TORCHAO_ONLY_QAT_SCHEMES",
    "validate_mlx_qat_request",
    "validate_mlx_qat_target_modules",
    "apply_mlx_qat",
    "remove_mlx_qat",
    "mlx_qat_module_count",
)

# None = inherit the base layer's bits.
SUPPORTED_MLX_QAT_SCHEMES = {
    "auto": None,
    "int4": 4,
    "int8": 8,
}

# torchao-only (activation / fp8 grids): refused, never approximated.
TORCHAO_ONLY_QAT_SCHEMES = (
    "int8-int4",
    "fp8-int4",
    "fp8-fp8",
    "cactus",
)

_DORA_REFUSAL = (
    "Unsloth: qat_scheme is not supported for DoRA on MLX yet. DoRA's fuse() "
    "rescales the merged weight by m / ||W + BA|| before quantizing, so a "
    "LoRA-shaped fake-quant would not match the saved weights."
)

_QAT_FLAG = "_unsloth_mlx_qat_active"
_QAT_ORIGINAL_CLASS = "_unsloth_mlx_qat_original_class"


def _dropout_probability(module):
    """Recover ``p`` from an ``mlx.nn.Dropout`` (it stores ``_p_1 = 1 - p``)."""
    dropout = getattr(module, "dropout", None)
    if dropout is None:
        return 0.0
    p_1 = getattr(dropout, "_p_1", None)
    if p_1 is None:
        return 0.0
    return 1.0 - float(p_1)


def _model_has_quantized_module(model):
    import mlx.nn as nn

    quantized_types = [nn.QuantizedLinear, nn.QuantizedEmbedding]
    try:
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear
        quantized_types.append(QuantizedSwitchLinear)
    except Exception:
        pass
    quantized_types = tuple(t for t in quantized_types if isinstance(t, type))
    return any(
        isinstance(module, quantized_types)
        for _, module in model.named_modules()
    )


def _quantization_grid(module):
    # mode defaults to affine: mlx-community configs and older layers omit it.
    return (
        module.group_size,
        module.bits,
        getattr(module, "mode", None) or "affine",
    )


def _validate_qat_base_modules(named_bases, requested_bits):
    """Shared by preflight and post-LoRA pass so they cannot disagree."""
    import mlx.nn as nn

    if not named_bases:
        raise ValueError(
            "Unsloth: qat_scheme was requested but no LoRA targets were "
            "resolved, so there is nothing for QAT to fake-quantize. Check "
            "target_modules / finetune_language_layers."
        )

    unquantized = [name for name, module in named_bases
                   if not isinstance(module, nn.QuantizedLinear)]
    if unquantized:
        raise ValueError(
            "Unsloth: qat_scheme requires quantized LoRA targets — "
            f"{len(unquantized)} target(s) are unquantized "
            f"({_preview(unquantized)}), so save_method='merged_4bit' would "
            "not requantize them and there is no quantization for QAT to "
            "simulate. Load with load_in_4bit=True (or a pre-quantized "
            "-4bit/-8bit repo) to use QAT."
        )

    grids = {_quantization_grid(module) for _, module in named_bases}
    if len(grids) != 1:
        raise ValueError(
            "Unsloth: qat_scheme requires a single quantization grid across "
            f"all LoRA targets, found {sorted(grids)}."
        )
    group_size, bits, mode = next(iter(grids))

    # mx.quantize returns no biases for mxfp4/nvfp4/mxfp8; only affine fits _qat_call.
    if mode != "affine":
        raise NotImplementedError(
            f"Unsloth: qat_scheme is not supported for {mode!r}-quantized "
            "bases on MLX yet — only the affine grid is implemented. Load the "
            "model with load_in_4bit=True (affine) to use QAT."
        )

    if requested_bits is not None and requested_bits != bits:
        raise ValueError(
            f"Unsloth: qat_scheme requests {requested_bits}-bit weights but "
            f"the LoRA targets are quantized to {bits}-bit. QAT must simulate "
            "the grid that save_method='merged_4bit' will actually write. Use "
            "qat_scheme='auto' to inherit the base quantization."
        )
    return group_size, bits, mode


def validate_mlx_qat_target_modules(named_bases, qat_scheme="auto"):
    """named_bases must be exactly what linear_to_lora_layers will replace."""
    return _validate_qat_base_modules(
        list(named_bases), _resolve_qat_bits(qat_scheme),
    )


def _lora_layer_types():
    from mlx_lm.tuner.lora import LoRALinear, LoRASwitchLinear

    dora_types = []
    try:
        from mlx_lm.tuner.dora import DoRALinear
        dora_types.append(DoRALinear)
    except Exception:
        pass
    try:
        from mlx_lm.tuner.dora import DoRAEmbedding
        dora_types.append(DoRAEmbedding)
    except Exception:
        pass

    switch_types = [LoRASwitchLinear]
    import sys
    vlm_lora = sys.modules.get("mlx_vlm.trainer.lora_layers")
    if vlm_lora is not None:
        vlm_switch = getattr(vlm_lora, "LoRASwitchLinear", None)
        if isinstance(vlm_switch, type):
            switch_types.append(vlm_switch)

    return LoRALinear, tuple(dora_types), tuple(switch_types)


_STE_QMM = {}


def _ste_quantized_matmul(group_size, bits, mode):
    """quantized_matmul forward (bit-exact to the saved QuantizedLinear), STE backward.

    A dense GEMM over the fake-quantized weight rounds differently from the
    quantized_matmul merged_4bit ships (0.017 loss drift on Metal), and paying for
    both matmuls in the forward doubles the cost; the custom VJP needs neither.
    """
    key = (group_size, bits, mode)
    fn = _STE_QMM.get(key)
    if fn is not None:
        return fn
    import mlx.core as mx
    grid = {"group_size": group_size, "bits": bits, "mode": mode}

    @mx.custom_function
    def fn(x, merged, packed, scales, biases):
        return mx.quantized_matmul(x, packed, scales, biases, transpose=True, **grid)

    @fn.vjp
    def fn_vjp(primals, cotangent, output):
        x, merged, packed, scales, biases = primals
        fake = mx.dequantize(packed, scales, biases, **grid).astype(x.dtype)
        dx = cotangent @ fake
        # Straight-through: the quantizer is the identity for the merged weight.
        d_merged = (
            cotangent.reshape(-1, cotangent.shape[-1]).T
            @ x.reshape(-1, x.shape[-1])
        ).astype(merged.dtype)
        return (dx, d_merged, mx.zeros_like(packed), mx.zeros_like(scales),
                mx.zeros_like(biases))

    _STE_QMM[key] = fn
    return fn


def _qat_call(self, x):
    """Mirrors LoRALinear.fuse(dequantize=False); STE for the gradient."""
    import mlx.core as mx

    base = self.linear
    group_size, bits, mode = _quantization_grid(base)
    weight = mx.dequantize(
        base.weight, base.scales, base.biases,
        group_size=group_size, bits=bits, mode=mode,
    )
    delta = ((self.scale * self.lora_b.T) @ self.lora_a.T).astype(weight.dtype)
    merged = weight + delta

    packed, scales, biases = mx.quantize(
        merged, group_size=group_size, bits=bits, mode=mode,
    )
    y = _ste_quantized_matmul(group_size, bits, mode)(
        x, merged, packed, scales, biases,
    )
    # fuse() keeps the base bias; dropping it breaks Qwen2 q/k/v.
    if "bias" in base:
        y = y + base.bias
    return y


def _resolve_qat_bits(qat_scheme):
    if qat_scheme is True:
        return None
    if not isinstance(qat_scheme, str):
        raise TypeError(
            f"Unsloth: qat_scheme must be a string or True, got "
            f"{type(qat_scheme).__name__}."
        )
    scheme = qat_scheme.strip().lower()
    if scheme in TORCHAO_ONLY_QAT_SCHEMES:
        raise NotImplementedError(
            f"Unsloth: qat_scheme={qat_scheme!r} is a torchao scheme and is not "
            "expressible with MLX's quantizer, which is weight-only and affine "
            "(group_size/bits/mode). Use 'int4', 'int8', or 'auto' to inherit "
            "the base model's quantization."
        )
    if scheme not in SUPPORTED_MLX_QAT_SCHEMES:
        supported = ", ".join(sorted(SUPPORTED_MLX_QAT_SCHEMES))
        raise ValueError(
            f"Unsloth: unsupported qat_scheme={qat_scheme!r} for MLX. "
            f"Supported: {supported}."
        )
    return SUPPORTED_MLX_QAT_SCHEMES[scheme]


def _qat_targets(model):
    """lora_wrapped includes unquantized bases so the validator can name them."""
    import mlx.nn as nn

    lora_linear_type, dora_types, switch_types = _lora_layer_types()

    patchable, dora, switch, lora_wrapped = [], [], [], []
    for name, module in model.named_modules():
        if dora_types and isinstance(module, dora_types):
            dora.append(name)
            continue
        if switch_types and isinstance(module, switch_types):
            switch.append(name)
            continue
        if not isinstance(module, lora_linear_type):
            continue
        lora_wrapped.append((name, module))
        if isinstance(getattr(module, "linear", None), nn.QuantizedLinear):
            patchable.append((name, module))
    return patchable, dora, switch, lora_wrapped


def _preview(names, limit=3):
    shown = ", ".join(names[:limit])
    if len(names) > limit:
        shown += f", +{len(names) - limit} more"
    return shown


def validate_mlx_qat_request(model, qat_scheme="auto", *, lora_dropout=None,
                             use_dora=False):
    """Pre-mutation checks; full_finetuning returns before adapters, so check here."""
    requested_bits = _resolve_qat_bits(qat_scheme)

    if getattr(model, "_unsloth_full_finetuning", False):
        raise NotImplementedError(
            "Unsloth: qat_scheme is not supported with full_finetuning=True on "
            "MLX yet — QAT currently simulates the LoRA merge performed by "
            "save_method='merged_4bit', which only requantizes when the base "
            "is quantized."
        )

    from .utils import _is_vlm_model
    if _is_vlm_model(model):
        # Coverage gate, not a known incompatibility: VLM merge path unvalidated.
        raise NotImplementedError(
            "Unsloth: qat_scheme is not supported for VLMs on MLX yet — the "
            "vision-tower and projector merge paths have not been validated "
            "against save_method='merged_4bit'. Use a text-only model, or "
            "train the VLM without qat_scheme."
        )

    if use_dora:
        raise NotImplementedError(_DORA_REFUSAL)

    if lora_dropout is not None and float(lora_dropout) > 0.0:
        raise NotImplementedError(
            "Unsloth: qat_scheme requires lora_dropout=0 on MLX. QAT folds the "
            "adapter into the weight to match fuse(), which leaves no separate "
            f"LoRA activation path for dropout to act on (got {lora_dropout})."
        )

    # Backstop only; validate_mlx_qat_target_modules is authoritative.
    if not _model_has_quantized_module(model):
        raise ValueError(
            "Unsloth: qat_scheme requires a quantized base model — nothing in "
            "this model is quantized, so save_method='merged_4bit' would not "
            "requantize and there is no quantization for QAT to simulate. "
            "Load with load_in_4bit=True (or a pre-quantized -4bit/-8bit repo) "
            "to use QAT."
        )

    return requested_bits


def apply_mlx_qat(model, qat_scheme="auto"):
    """Patch LoRA layers (already attached) for QAT; returns count patched."""
    requested_bits = validate_mlx_qat_request(model, qat_scheme)

    patchable, dora, switch, lora_wrapped = _qat_targets(model)

    if dora:
        raise NotImplementedError(f"{_DORA_REFUSAL} Affected: {_preview(dora)}.")
    if switch:
        raise NotImplementedError(
            "Unsloth: qat_scheme is not supported for MoE / SwitchLinear "
            "experts on MLX yet — their fuse() merges a 3-D per-expert weight "
            f"with a different delta layout. Affected: {_preview(switch)}."
        )
    if not patchable and not lora_wrapped:
        raise ValueError(
            "Unsloth: qat_scheme was requested but the model has no LoRA "
            "layers. Call get_peft_model(...) with LoRA targets first."
        )

    group_size, bits, mode = _validate_qat_base_modules(
        [(name, module.linear) for name, module in lora_wrapped],
        requested_bits,
    )

    dropout_layers = [n for n, m in patchable if _dropout_probability(m) > 0.0]
    if dropout_layers:
        raise NotImplementedError(
            "Unsloth: qat_scheme requires lora_dropout=0 on MLX. QAT folds the "
            "adapter into the weight to match fuse(), which leaves no separate "
            "LoRA activation path for dropout to act on. Affected: "
            f"{_preview(dropout_layers)}."
        )

    patched = 0
    for _, module in patchable:
        if getattr(module, _QAT_FLAG, False):
            continue
        original = type(module)
        # Keep name + MRO: save path checks type(module).__name__ and isinstance.
        qat_class = type(
            original.__name__,
            (original,),
            {
                "__call__": _qat_call,
                "__module__": original.__module__,
                "__qualname__": getattr(original, "__qualname__", original.__name__),
                _QAT_FLAG: True,
                _QAT_ORIGINAL_CLASS: original,
            },
        )
        module.__class__ = qat_class
        patched += 1

    if patched:
        print(
            f"Unsloth: QAT enabled on {patched} LoRA layer(s) — simulating "
            f"{bits}-bit / group_size={group_size} / mode={mode!r} quantization.\n"
            "Training loss will read higher than a non-QAT run; the comparable "
            "number is the loss after save_method='merged_4bit'."
        )
    return patched


def remove_mlx_qat(model):
    restored = 0
    for _, module in model.named_modules():
        original = getattr(type(module), _QAT_ORIGINAL_CLASS, None)
        if original is None or not getattr(module, _QAT_FLAG, False):
            continue
        module.__class__ = original
        restored += 1
    return restored


def mlx_qat_module_count(model):
    return sum(
        1 for _, module in model.named_modules()
        if getattr(module, _QAT_FLAG, False)
    )
