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

"""MLX LoRA QAT: train through the exact weights ``save_method="merged_4bit"`` ships.

``LoRALinear.fuse(dequantize=False)`` saves ``quantize(dequantize(W) + BA)``, so the QAT
forward runs that requantized merge through ``quantized_matmul`` (bit-exact to the saved
``QuantizedLinear``) with a straight-through gradient. Unlike the CUDA/torchao path, the
target is W + BA, not W. Only LoRA over affine ``QuantizedLinear`` text layers is supported.
"""

_SCHEME_BITS = {"auto": None, "int4": 4, "int8": 8}
_QAT_FLAG = "_unsloth_mlx_qat_active"
_QAT_CLASSES = {}
_STE_QMM = {}


def _quantization_grid(module):
    # mlx-community configs and older layers omit `mode`.
    return module.group_size, module.bits, getattr(module, "mode", None) or "affine"


def _requested_bits(qat_scheme):
    if qat_scheme is True:
        return None
    if not isinstance(qat_scheme, str):
        raise TypeError(f"Unsloth: qat_scheme must be a string or True, got {type(qat_scheme).__name__}.")
    scheme = qat_scheme.strip().lower()
    if scheme not in _SCHEME_BITS:
        # torchao schemes (int8-int4, fp8-*, ...) quantize activations or use fp8; not expressible here.
        raise NotImplementedError(
            f"Unsloth: qat_scheme={qat_scheme!r} is not supported on MLX, whose quantizer is "
            "weight-only affine. Use 'auto' (inherit the base grid), 'int4' or 'int8'."
        )
    return _SCHEME_BITS[scheme]


def validate_mlx_qat_request(model, qat_scheme, *, lora_dropout=0, use_dora=False):
    """Checks that need no LoRA targets; run before get_peft_model mutates anything."""
    _requested_bits(qat_scheme)
    from .utils import _is_vlm_model
    refusal = (
        "full_finetuning=True (QAT simulates the LoRA merge)" if getattr(model, "_unsloth_full_finetuning", False)
        else "VLMs (merge path not validated yet)" if _is_vlm_model(model)
        else "DoRA (fuse() rescales by m / ||W + BA|| before quantizing)" if use_dora
        else "lora_dropout > 0 (the adapter is folded into the weight)" if lora_dropout and float(lora_dropout) > 0
        else None
    )
    if refusal:
        raise NotImplementedError(f"Unsloth: qat_scheme is not supported on MLX with {refusal}.")


def validate_mlx_qat_targets(named_modules, qat_scheme):
    """The exact modules LoRA will wrap: all affine QuantizedLinear on one grid."""
    import mlx.nn as nn

    named_modules = list(named_modules)
    if not named_modules:
        raise ValueError("Unsloth: qat_scheme needs LoRA targets on the language layers; none were selected.")
    bad = [n for n, m in named_modules if not isinstance(m, nn.QuantizedLinear)]
    if bad:
        raise ValueError(
            f"Unsloth: qat_scheme needs quantized LoRA targets; {len(bad)} are not ({', '.join(bad[:3])}). "
            "Load with load_in_4bit=True or a pre-quantized repo."
        )
    grids = {_quantization_grid(m) for _, m in named_modules}
    if len(grids) != 1 or next(iter(grids))[2] != "affine":
        raise NotImplementedError(f"Unsloth: MLX QAT needs one affine grid across LoRA targets, found {sorted(grids)}.")
    bits = next(iter(grids))[1]
    requested = _requested_bits(qat_scheme)
    if requested is not None and requested != bits:
        raise ValueError(
            f"Unsloth: qat_scheme requests {requested}-bit but the LoRA targets are {bits}-bit; "
            "QAT must simulate the grid merged_4bit writes. Use qat_scheme='auto'."
        )


def _ste_quantized_matmul(group_size, bits, mode):
    key = (group_size, bits, mode)
    if key in _STE_QMM:
        return _STE_QMM[key]
    import mlx.core as mx

    grid = {"group_size": group_size, "bits": bits, "mode": mode}

    # A dense GEMM over the dequantized weight rounds differently from quantized_matmul
    # (0.017 loss drift across save on Metal); the custom VJP avoids a second matmul.
    @mx.custom_function
    def fn(x, merged, packed, scales, biases):
        return mx.quantized_matmul(x, packed, scales, biases, transpose=True, **grid)

    @fn.vjp
    def fn_vjp(primals, cotangent, output):
        x, merged, packed, scales, biases = primals
        dx = cotangent @ mx.dequantize(packed, scales, biases, **grid).astype(x.dtype)
        d_merged = cotangent.reshape(-1, cotangent.shape[-1]).T @ x.reshape(-1, x.shape[-1])
        return dx, d_merged.astype(merged.dtype), mx.zeros_like(packed), mx.zeros_like(scales), mx.zeros_like(biases)

    _STE_QMM[key] = fn
    return fn


def _qat_call(self, x):
    """LoRALinear.fuse(dequantize=False) applied on the fly, straight-through gradient."""
    import mlx.core as mx

    base = self.linear
    group_size, bits, mode = _quantization_grid(base)
    grid = {"group_size": group_size, "bits": bits, "mode": mode}
    weight = mx.dequantize(base.weight, base.scales, base.biases, **grid)
    merged = weight + ((self.scale * self.lora_b.T) @ self.lora_a.T).astype(weight.dtype)
    y = _ste_quantized_matmul(group_size, bits, mode)(x, merged, *mx.quantize(merged, **grid))
    return y + base.bias if "bias" in base else y


def apply_mlx_qat(model):
    """Switch every LoRALinear over a QuantizedLinear to the QAT forward; returns the count."""
    import mlx.nn as nn
    from mlx_lm.tuner.lora import LoRALinear

    patched = 0
    for _, module in model.named_modules():
        if (not isinstance(module, LoRALinear) or getattr(module, _QAT_FLAG, False)
                or not isinstance(module.linear, nn.QuantizedLinear)):
            continue
        original = type(module)
        if original not in _QAT_CLASSES:
            # Same name and MRO: the save path keys on type(module).__name__ and isinstance.
            _QAT_CLASSES[original] = type(original.__name__, (original,), {
                "__call__": lambda self, x: _qat_call(self, x),
                "__module__": original.__module__, "__qualname__": original.__qualname__, _QAT_FLAG: True,
            })
        module.__class__ = _QAT_CLASSES[original]
        patched += 1
    if not patched:
        raise ValueError("Unsloth: qat_scheme found no LoRA layer over a quantized base to train.")
    print(f"Unsloth: QAT on {patched} LoRA layers; the training loss is what merged_4bit will ship.")
    return patched
