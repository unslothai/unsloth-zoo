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
"""Dequantize FP8 weights with no FP8 container (Step-3.7-Flash-FP8 `MoELinear`); the loader drops their scale."""
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
    module, attr = _module_and_attr(model, full_layer_name)
    if module is None or attr is None:
        return False
    scale = getattr(module, _scale_attr_for(attr), None)
    return isinstance(scale, torch.Tensor)


def _first(value):
    return value[0] if isinstance(value, (list, tuple)) else value


def _dequantized_targets(model):
    # On the model, not the op: the loader deepcopies the op per target key, but save reverses the original.
    if model is None:
        return set()
    names = getattr(model, "_unsloth_fp8_dequantized_targets", None)
    if isinstance(names, set):
        return names
    names = set()
    try:
        model._unsloth_fp8_dequantized_targets = names
    except Exception:
        return set()
    return names


def _make_op(Fp8Dequantize):
    class Fp8DequantizeWithoutContainer(Fp8Dequantize):
        def convert(self, input_dict, full_layer_name = None, model = None, **kwargs):
            if _target_owns_scale(model, full_layer_name):
                return self._pass_through(input_dict, full_layer_name, model)
            out = super().convert(input_dict, full_layer_name = full_layer_name, model = model, **kwargs)
            # Only weights that arrived with a scale are re-quantized on save.
            has_scale = any("scale_inv" in (k[:-1] if k.endswith("$") else k) for k in input_dict)
            # Reverse op only writes e4m3, so skip packed FP4; skip scales (MXFP8 E8M0 is uint8).
            arrived_packed_fp4 = any(
                isinstance(_first(v), torch.Tensor) and _first(v).dtype in _PACKED_FP4_DTYPES
                for k, v in input_dict.items()
                if "scale" not in (k[:-1] if k.endswith("$") else k)
            )
            if has_scale and not arrived_packed_fp4 and full_layer_name:
                _dequantized_targets(model).add(full_layer_name)
            return out

        def _pass_through(self, input_dict, full_layer_name, model):
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
                # Else the container quantizes against uninitialised scale memory.
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


_PACKED_FP4_DTYPES = tuple(
    d for d in (torch.int8, torch.uint8, getattr(torch, "float4_e2m1fn_x2", None)) if d is not None
)
_FP8_DTYPES = tuple(
    getattr(torch, name) for name in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, name)
)


def _make_reverse_op(Fp8Dequantize):
    """Stock `Fp8Quantize` would re-quantize packed weights and the scale grid; only re-quantize what we dequantized."""
    from transformers.integrations.finegrained_fp8 import Fp8Quantize

    class Fp8RequantizeWithoutContainer(Fp8Quantize):
        def convert(self, input_dict, full_layer_name = None, model = None, **kwargs):
            # Every tensor arrives keyed `weight`; only `full_layer_name` identifies it.
            name = full_layer_name or ""
            dequantized = _dequantized_targets(model)
            out = {}
            for key, value in input_dict.items():
                tensor = _first(value)
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.dtype in _FP8_DTYPES
                    or name.endswith("_scale_inv")
                    or name.endswith("activation_scale")
                    or name not in dequantized
                ):
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
