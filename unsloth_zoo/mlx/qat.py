# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Quantization-aware training (QAT) for LoRA on Apple Silicon.

MLX's ``merged_4bit`` save path does not keep the base quantization untouched:
``mlx_lm.tuner.lora.LoRALinear.fuse(dequantize=False)`` dequantizes the base,
adds the LoRA delta, and then **requantizes** the merged result. So the weight
that actually ships is ``quantize(dequantize(W_q) + B @ A)`` -- a grid the model
never sees while training, because during training the LoRA delta is applied in
full precision on top of the quantized base.

This module closes that gap: it fake-quantizes the *merged* weight in the
forward pass, at the base layer's own ``group_size`` / ``bits`` / ``mode``, so
gradients reach the LoRA through the same quantizer that ``fuse()`` will apply
at save time. A straight-through estimator carries the gradient.

Note this differs deliberately from the CUDA/torchao ``qat_scheme`` path, which
fake-quantizes the *frozen base* and leaves the adapter in high precision. That
is correct for bitsandbytes, where the base stays quantized and the adapter is
merged differently. It is the wrong target for MLX: simulating ``quantize(W)``
while ``fuse()`` computes ``quantize(W + BA)`` would train against a quantizer
that never runs.

Expect the training loss to look *worse* with QAT enabled. That is the feature
working, not failing -- the pre-fuse loss of a non-QAT run describes a model
configuration that only exists in memory and is destroyed by the save. Measured
on Qwen2.5-0.5B-Instruct-4bit / wikitext-2 over 300 steps: the non-QAT run
improved 0.306 nats while training but kept only 0.111 of it after
``merged_4bit``; the QAT run improved 0.241 and kept 0.241.

Scope: LoRA over ``nn.QuantizedLinear`` on text models. DoRA, MoE/SwitchLinear
experts, VLMs, unquantized bases, ``lora_dropout > 0`` and the torchao-only
schemes are refused rather than approximated -- each merges differently, and a
fake-quant that does not match ``fuse()`` is worse than none.
"""

from __future__ import annotations

__all__ = (
    "SUPPORTED_MLX_QAT_SCHEMES",
    "TORCHAO_ONLY_QAT_SCHEMES",
    "validate_mlx_qat_request",
    "apply_mlx_qat",
    "remove_mlx_qat",
    "mlx_qat_module_count",
)

# Schemes MLX's affine quantizer can actually express, mapped to weight bits.
# ``None`` means "inherit whatever the base layer was quantized to".
SUPPORTED_MLX_QAT_SCHEMES = {
    "auto": None,
    "int4": 4,
    "int8": 8,
}

# Accepted by the CUDA path via torchao, but not expressible with mx.quantize:
# they quantize activations and/or use fp8 grids. Rejected explicitly rather
# than silently approximated with the nearest affine weight-only scheme.
TORCHAO_ONLY_QAT_SCHEMES = (
    "int8-int4",
    "fp8-int4",
    "fp8-fp8",
    "cactus",
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


def _lora_layer_types():
    """(LoRALinear, DoRA types, switch/MoE LoRA types) for the installed stack."""
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


def _qat_call(self, x):
    """Forward using a fake-quantized *merged* weight.

    Mirrors ``LoRALinear.fuse(dequantize=False)`` exactly: dequantize the base,
    add the LoRA delta, requantize. The straight-through estimator keeps the
    gradient flowing to ``lora_a`` / ``lora_b`` as if the quantizer were the
    identity.
    """
    import mlx.core as mx

    base = self.linear
    weight = mx.dequantize(
        base.weight,
        base.scales,
        base.biases,
        group_size=base.group_size,
        bits=base.bits,
        mode=base.mode,
    )
    delta = ((self.scale * self.lora_b.T) @ self.lora_a.T).astype(weight.dtype)
    merged = weight + delta

    packed, scales, biases = mx.quantize(
        merged, group_size=base.group_size, bits=base.bits, mode=base.mode,
    )
    fake = mx.dequantize(
        packed, scales, biases,
        group_size=base.group_size, bits=base.bits, mode=base.mode,
    )
    # Straight-through: forward uses `fake`, backward behaves like identity.
    straight_through = merged + mx.stop_gradient(fake - merged)

    y = x @ straight_through.T
    # `fuse()` carries the base bias through untouched; dropping it silently
    # corrupts every arch with biased projections (e.g. Qwen2/2.5 q/k/v).
    if "bias" in base:
        y = y + base.bias
    return y


def _resolve_qat_bits(qat_scheme):
    """Validate ``qat_scheme`` and return the requested bit width (or None)."""
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
    """Split LoRA modules into (patchable, dora, switch, unquantized)."""
    import mlx.nn as nn

    lora_linear_type, dora_types, switch_types = _lora_layer_types()

    patchable, dora, switch, unquantized = [], [], [], []
    for name, module in model.named_modules():
        if dora_types and isinstance(module, dora_types):
            dora.append(name)
            continue
        if switch_types and isinstance(module, switch_types):
            switch.append(name)
            continue
        if not isinstance(module, lora_linear_type):
            continue
        base = getattr(module, "linear", None)
        if isinstance(base, nn.QuantizedLinear):
            patchable.append((name, module))
        else:
            unquantized.append(name)
    return patchable, dora, switch, unquantized


def _preview(names, limit=3):
    shown = ", ".join(names[:limit])
    if len(names) > limit:
        shown += f", +{len(names) - limit} more"
    return shown


def validate_mlx_qat_request(model, qat_scheme="auto", *, lora_dropout=None):
    """Reject an unsupportable QAT request *before* anything is mutated.

    Everything checkable without adapters lives here so ``get_peft_model`` can
    call it up front: otherwise a rejected request either raises after LoRA has
    already been installed and trainability changed, or -- for
    ``full_finetuning=True``, which returns before adapters are ever created --
    is silently ignored.

    Returns the requested bit width (``None`` means "inherit the base").
    """
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
        # The LoRA layers are the same LoRALinear-over-QuantizedLinear, and all
        # of them (language model and vision tower) do get discovered here, so
        # this is a coverage gate rather than a known incompatibility: the VLM
        # train -> merged_4bit -> reload path has not been validated end to end
        # for QAT. Refuse rather than ship an unverified numeric claim.
        raise NotImplementedError(
            "Unsloth: qat_scheme is not supported for VLMs on MLX yet — the "
            "vision-tower and projector merge paths have not been validated "
            "against save_method='merged_4bit'. Use a text-only model, or "
            "train the VLM without qat_scheme."
        )

    if lora_dropout is not None and float(lora_dropout) > 0.0:
        raise NotImplementedError(
            "Unsloth: qat_scheme requires lora_dropout=0 on MLX. QAT folds the "
            "adapter into the weight to match fuse(), which leaves no separate "
            f"LoRA activation path for dropout to act on (got {lora_dropout})."
        )

    return requested_bits


def apply_mlx_qat(model, qat_scheme="auto"):
    """Fake-quantize the merged LoRA weight during training.

    Args:
        model: an MLX model that already has LoRA layers attached.
        qat_scheme: ``"auto"``/``True`` to inherit the base model's own
            quantization, or ``"int4"`` / ``"int8"`` to additionally assert the
            base was quantized to that width.

    Returns:
        The number of LoRA modules placed under QAT.
    """
    # Re-run the pre-mutation checks: apply_mlx_qat is public and may be called
    # directly, not only through get_peft_model.
    requested_bits = validate_mlx_qat_request(model, qat_scheme)

    patchable, dora, switch, unquantized = _qat_targets(model)

    if dora:
        raise NotImplementedError(
            "Unsloth: qat_scheme is not supported for DoRA on MLX yet. DoRA's "
            "fuse() rescales the merged weight by m / ||W + BA|| before "
            "quantizing, so a LoRA-shaped fake-quant would not match the saved "
            f"weights. Affected: {_preview(dora)}."
        )
    if switch:
        raise NotImplementedError(
            "Unsloth: qat_scheme is not supported for MoE / SwitchLinear "
            "experts on MLX yet — their fuse() merges a 3-D per-expert weight "
            f"with a different delta layout. Affected: {_preview(switch)}."
        )
    if not patchable:
        if unquantized:
            raise ValueError(
                "Unsloth: qat_scheme requires a quantized base model. The LoRA "
                f"layers wrap unquantized modules ({_preview(unquantized)}), so "
                "save_method='merged_4bit' would not requantize and there is no "
                "quantization for QAT to simulate. Load with q_bits=4 (or a "
                "pre-quantized -4bit/-8bit repo) to use QAT."
            )
        raise ValueError(
            "Unsloth: qat_scheme was requested but the model has no LoRA "
            "layers. Call get_peft_model(...) with LoRA targets first."
        )

    # Every layer must share one grid: the saved config records a single
    # global quantization, and a mixed set cannot be expressed there.
    grids = {
        (m.linear.group_size, m.linear.bits, m.linear.mode) for _, m in patchable
    }
    if len(grids) != 1:
        raise ValueError(
            "Unsloth: qat_scheme requires a single quantization grid across all "
            f"LoRA layers, found {sorted(grids)}."
        )
    group_size, bits, mode = next(iter(grids))

    # Only the affine grid round-trips through the three-value
    # quantize/dequantize used below: mx.quantize returns just (packed, scales)
    # for mxfp4/nvfp4/mxfp8, and those grids need different fake-quant maths
    # anyway. Reject explicitly rather than unpack-crash inside the forward.
    if mode not in (None, "affine"):
        raise NotImplementedError(
            f"Unsloth: qat_scheme is not supported for {mode!r}-quantized "
            "bases on MLX yet — only the affine grid is implemented. Load the "
            "model with load_in_4bit=True (affine) to use QAT."
        )

    if requested_bits is not None and requested_bits != bits:
        raise ValueError(
            f"Unsloth: qat_scheme={qat_scheme!r} requests {requested_bits}-bit "
            f"weights but the base model is quantized to {bits}-bit. QAT must "
            "simulate the grid that save_method='merged_4bit' will actually "
            "write. Use qat_scheme='auto' to inherit the base quantization."
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
        # Subclass rather than swap the module: the save path's DoRA detection
        # keys on type(module).__name__ and the quantization-map scan keys on
        # isinstance, so the stand-in has to keep both the name and the MRO.
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
    """Restore the stock LoRA forward. Returns the number of layers restored."""
    restored = 0
    for _, module in model.named_modules():
        original = getattr(type(module), _QAT_ORIGINAL_CLASS, None)
        if original is None or not getattr(module, _QAT_FLAG, False):
            continue
        module.__class__ = original
        restored += 1
    return restored


def mlx_qat_module_count(model):
    """Number of LoRA modules currently under QAT."""
    return sum(
        1 for _, module in model.named_modules()
        if getattr(module, _QAT_FLAG, False)
    )
