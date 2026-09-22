# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""FP8 checkpoints whose quantized tensors have no FP8 module to land in.

transformers gives a pre-quantized fine-grained FP8 checkpoint two homes for a
`weight` plus `weight_scale_inv` pair: `FP8Linear` (an `nn.Linear` it swapped
in) and `FP8Experts` (a transformers experts class it knows). A remote-code
module that keeps its expert stack as a bare 3-D parameter (stepfun-ai/
Step-3.7-Flash-FP8: `MoELinear.weight`, e4m3 `(288, 1280, 4096)` with a
`(288, 10, 32)` fp32 block-scale grid) gets neither. The loader then casts the
e4m3 bytes to the parameter's bf16 unscaled and reports the scale as
UNEXPECTED, and the model trains from wrong weights: 15.1 loss on the first
step where the bf16 checkpoint gives 1.7.

With `dequantize = True` transformers already appends a generic
`weight$ + weight_scale_inv -> weight` converter that folds the scale in. This
patch appends the same converter when `dequantize` is off, with one twist: a
target that owns its own scale (`FP8Linear`, `FP8Experts`) passes through
untouched so the FP8 kernels keep their packed weights, and only a target
without a container is dequantized into its own dtype. A container whose
checkpoint tensor arrives without a scale (the checkpoint's
`modules_to_not_convert` forgot it, so the bf16 weight is cast to e4m3) gets a
scale of ones instead of the uninitialised memory it would otherwise read.
"""
import torch

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import logger

__all__ = [
    "patch_fp8_dequantize_weights_without_container",
]


def _module_and_attr(model, full_layer_name):
    if model is None or not full_layer_name:
        return None, None
    try:
        from transformers.quantizers.quantizers_utils import get_module_from_name
        return get_module_from_name(model, full_layer_name)
    except Exception:
        return None, None


def _scale_attr_for(attr):
    return "weight_scale_inv" if attr == "weight" else f"{attr}_scale_inv"


def _target_owns_scale(model, full_layer_name):
    """True when the parameter's module carries the matching `*_scale_inv`,
    which is how `FP8Linear` and `FP8Experts` store a packed weight."""
    module, attr = _module_and_attr(model, full_layer_name)
    if module is None or attr is None:
        return False
    scale = getattr(module, _scale_attr_for(attr), None)
    return isinstance(scale, torch.Tensor)


def _first(value):
    return value[0] if isinstance(value, (list, tuple)) else value


def _make_op(Fp8Dequantize):
    class Fp8DequantizeWithoutContainer(Fp8Dequantize):
        """`Fp8Dequantize` for targets with no FP8 container; pass-through otherwise."""

        def convert(self, input_dict, full_layer_name = None, model = None, **kwargs):
            if _target_owns_scale(model, full_layer_name):
                return self._pass_through(input_dict, full_layer_name, model)
            return super().convert(input_dict, full_layer_name = full_layer_name, model = model, **kwargs)

        def _pass_through(self, input_dict, full_layer_name, model):
            # Full names on purpose: the loader derives prefix and suffix from the
            # first output key found inside the layer name, and a full name gives it
            # an empty prefix and suffix, so every key here lands as written.
            module, attr = _module_and_attr(model, full_layer_name)
            base = full_layer_name[: -len(attr)] if attr and full_layer_name.endswith(attr) else full_layer_name + "."
            out = {}
            weight = None
            scale = None
            for key, value in input_dict.items():
                value = _first(value)
                pattern = key[:-1] if key.endswith("$") else key
                if pattern == "weight" or pattern == attr:
                    weight = value
                    out[full_layer_name] = value
                elif "scale_inv" in pattern:
                    scale = value
                    out[base + _scale_attr_for(attr)] = value
                elif "activation_scale" in pattern:
                    out[base + "activation_scale"] = value
                else:
                    out[base + pattern] = value
            if scale is None and weight is not None and module is not None:
                # The container will quantize the bf16 tensor it got into e4m3 with
                # whatever scale it finds; an uninitialised one is wrong by an
                # arbitrary factor, ones is exact up to e4m3 rounding.
                container_scale = getattr(module, _scale_attr_for(attr), None)
                if isinstance(container_scale, torch.Tensor):
                    out[base + _scale_attr_for(attr)] = torch.ones(
                        container_scale.shape, dtype = torch.float32, device = weight.device,
                    )
                    if UNSLOTH_ENABLE_LOGGING:
                        logger.info(
                            f"Unsloth: {full_layer_name} is an FP8 module but its checkpoint tensor "
                            f"ships no scale; loading it with a scale of ones."
                        )
            return out

        @property
        def reverse_op(self):
            return _make_reverse_op(Fp8Dequantize)(self.hf_quantizer)
    return Fp8DequantizeWithoutContainer


_FP8_DTYPES = tuple(
    getattr(torch, name) for name in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, name)
)


def _make_reverse_op(Fp8Dequantize):
    """What `save_pretrained` runs through this converter in reverse.

    transformers reverses every converter on save and `Fp8Dequantize.reverse_op`
    is `Fp8Quantize`, which would re-quantize a container's already packed e4m3
    weight with a fresh scale and, because the reversed source pattern `weight`
    also matches `weight_scale_inv`, quantize the scale grid itself. A packed
    weight, a scale and an activation scale are written as they are; only a
    weight this converter dequantized on load (bf16 now) is quantized back, so
    the saved checkpoint keeps the FP8 layout its config describes."""
    from transformers.integrations.finegrained_fp8 import Fp8Quantize

    class Fp8RequantizeWithoutContainer(Fp8Quantize):
        def convert(self, input_dict, full_layer_name = None, **kwargs):
            # The reversed source pattern `weight` also matches `weight_scale_inv`, so
            # every tensor arrives under the key `weight`; `full_layer_name` says what it is.
            name = full_layer_name or ""
            out = {}
            for key, value in input_dict.items():
                tensor = _first(value)
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.dtype in _FP8_DTYPES
                    or name.endswith("_scale_inv")
                    or name.endswith("activation_scale")
                ):
                    # Full name on purpose, so the saved key is the parameter's own.
                    out[name or key] = tensor
                else:
                    out.update(self._quantize_one(key, tensor))
            return out

    return Fp8RequantizeWithoutContainer


def patch_fp8_dequantize_weights_without_container():
    try:
        from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize
        from transformers.core_model_loading import WeightConverter
    except ImportError:
        return
    original = getattr(FineGrainedFP8HfQuantizer, "update_weight_conversions", None)
    if original is None or getattr(original, "_unsloth_uncontained_patched", False):
        return

    Op = _make_op(Fp8Dequantize)

    def _wants_fallback(self):
        if not getattr(self, "pre_quantized", False):
            return False
        if getattr(self.quantization_config, "dequantize", False):
            return False  # transformers appends its own converter in this mode
        return True

    def patched_update_weight_conversions(self, weight_conversions):
        conversions = original(self, weight_conversions)
        if not _wants_fallback(self):
            return conversions
        return list(conversions) + [
            WeightConverter(
                source_patterns = ["weight$", "weight_scale_inv", "activation_scale"],
                target_patterns = "weight",
                operations = [Op(self)],
            )
        ]

    patched_update_weight_conversions._unsloth_uncontained_patched = True
    patched_update_weight_conversions._unsloth_original = original
    FineGrainedFP8HfQuantizer.update_weight_conversions = patched_update_weight_conversions
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: FP8 weights whose module has no FP8 container are dequantized at load")
pass


def _register():
    try:
        from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
    except ImportError:
        return
    if getattr(FineGrainedFP8HfQuantizer, "update_weight_conversions", None) is None:
        return
    TEMPORARY_PATCHES.append(patch_fp8_dequantize_weights_without_container)
pass
_register()
