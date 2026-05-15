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
"""
MoE Expert Parameter 4-bit Quantization Patch for Transformers

Patches transformers' bitsandbytes quantization to handle MoE expert parameters
(gate_up_proj, down_proj) that are nn.Parameter instead of nn.Linear.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union
import os
from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING, logger
from .utils import patch_function, raise_error

# Check bitsandbytes availability
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Params4bit
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    Params4bit = None


__all__ = [
    "patch_bnb4bit_quantize_convert",
    "patch_bnb4bit_quantizer_param_needs_quantization",
    "patch_bnb4bit_quantizer_process_model",
    "patch_transformers_grouped_linear_4bit",
    "patch_transformers_weight_converter_kwargs",
    "replace_expert_params_with_bnb_params",
]


def _is_expert_module(module: nn.Module) -> bool:
    """
    Check if a module is an MoE experts module.
    Specifically, check if the module has gate_up_proj & down_proj attributes that are nn.Parameter.
    """
    return (
        hasattr(module, "gate_up_proj")
        and hasattr(module, "down_proj")
        and isinstance(module.gate_up_proj, nn.Parameter)
        and isinstance(module.down_proj, nn.Parameter)
    )


def replace_expert_params_with_bnb_params(
    model: nn.Module,
    modules_to_not_convert: Optional[List[str]] = None,
    quantization_config = None,
    pre_quantized: bool = False,
) -> nn.Module:
    """
    Replace MoE parameters (gate_up_proj, down_proj) of nn.Parameter type with Params4bit.
    """

    try:
        from transformers.quantizers.quantizers_utils import should_convert_module
    except Exception as e:
        return raise_error("transformers.quantizers.quantizers_utils.should_convert_module", e)
    
    has_been_replaced = False
    
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue

        if not _is_expert_module(module):
            continue
        
        gate_up_proj = module.gate_up_proj
        down_proj = module.down_proj

        if isinstance(gate_up_proj, Params4bit) or isinstance(down_proj, Params4bit):
            continue
        with torch.device("meta"):
            placeholder_gate_up = Params4bit(
                torch.zeros(gate_up_proj.shape),
                requires_grad=False,
                compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                quant_type=quantization_config.bnb_4bit_quant_type,
                quant_storage=quantization_config.bnb_4bit_quant_storage,
            )
            
            placeholder_down = Params4bit(
                torch.zeros(down_proj.shape),
                requires_grad=False,
                compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                quant_type=quantization_config.bnb_4bit_quant_type,
                quant_storage=quantization_config.bnb_4bit_quant_storage,
            )
        
        if pre_quantized:
            placeholder_gate_up.data = placeholder_gate_up.data.to(dtype=quantization_config.bnb_4bit_quant_storage)
            placeholder_down.data = placeholder_down.data.to(dtype=quantization_config.bnb_4bit_quant_storage)
        module.gate_up_proj = placeholder_gate_up
        module.down_proj = placeholder_down
        has_been_replaced = True
        
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(f"Unsloth: Prepared {module_name}'s gate_up_proj & down_proj for BNB 4-bit quantization (shapes: {gate_up_proj.shape}, {down_proj.shape})")
    
    if not has_been_replaced and UNSLOTH_ENABLE_LOGGING:
        # Demoted from warning to info+gated: dense (non-MoE) 4-bit loads hit this
        # path too (Phi3, GLM4 dense, Llama, Mistral, Gemma, Qwen2.5 dense, ...).
        # Surface only when verbose to avoid spamming every bnb 4-bit user.
        logger.info(
            f"Unsloth: No MoE expert parameters were found to be replaced for "
            f"{getattr(model, 'name_or_path', type(model).__name__)} (expected for non-MoE)"
        )
    
    return model


def patch_bnb4bit_quantize_convert():
    """
    Expert modules of nn.Parameter type are converted to Params4bit placeholders during weight loading.
    Also preserves the original shape of the expert parameters for PEFT LoRA compatibility.
    """
    
    try:
        from transformers.integrations.bitsandbytes import Bnb4bitQuantize
    except Exception as e:
        return raise_error("transformers.integrations.bitsandbytes.Bnb4bitQuantize", e)

    if hasattr(Bnb4bitQuantize.convert, "_unsloth_moe_patched"):
        return
    
    original_convert = Bnb4bitQuantize.convert
    
    def patched_convert(
        self,
        input_dict: dict[str, Union[list[torch.Tensor], torch.Tensor]],
        full_layer_name: str | None = None,
        model: torch.nn.Module | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Patched Bnb4bitQuantize.convert that quantizes MoE expert nn.Parameter weights.

        input_dict: per-tensor mapping from transformers' WeightConverter.materialize_tensors.
            For dense Linear weights this is dict[str, list[Tensor]] (matching upstream).
            For MoE expert nn.Parameter weights some dispatch paths pass a bare Tensor
            instead of a list, so the unwrap below tolerates both forms.
        full_layer_name: state-dict-style dotted name of the parameter being converted.
        """
        value = list(input_dict.values())[0]
        # transformers v5 passes input_dict[key] as list[Tensor] for most weights
        # (matches upstream Bnb4bitQuantize.convert which does `value = value[0]`).
        # For some MoE expert dispatch paths it arrives as a bare Tensor; tolerate both.
        if isinstance(value, (list, tuple)):
            value = value[0]

        try:
            from transformers.quantizers.quantizers_utils import get_module_from_name
            module, _ = get_module_from_name(model, full_layer_name)

            if _is_expert_module(module):
                old_value = model.get_parameter_or_buffer(full_layer_name)
                
                old_dict = {k: v for k, v in old_value.__dict__.items()}
                new_value = Params4bit(value, requires_grad=False, **old_dict).to(value.device)
                
                # Preserve _original_shape for expert params (critical for PEFT LoRA compatibility)
                original_shape = value.shape
                if original_shape is not None:
                    setattr(new_value, "_original_shape", original_shape)
                
                module._is_hf_initialized = True
                return {full_layer_name: new_value}
        
        except (KeyError, AttributeError) as e:
            # Expected non-fatal: get_module_from_name didn't resolve, or module
            # didn't have the expected expert-shaped attributes. Fall through to
            # original convert. Logged at info-level under UNSLOTH_ENABLE_LOGGING.
            if UNSLOTH_ENABLE_LOGGING:
                logger.info(f"Unsloth: expert convert fall-through for {full_layer_name}: {e}")
        except Exception:
            # Unexpected: use logger.exception so the traceback is preserved for
            # post-mortem. Still fall through to original_convert (don't break the
            # whole load) but make sure the bug is visible — broad swallowing was
            # what masked B1 (the list[Tensor] unwrap regression).
            logger.exception(f"Unsloth: unexpected error in patched_convert for {full_layer_name}")

        # Fall back to original convert for non-expert params or in case of any failure
        return original_convert(self, input_dict, full_layer_name=full_layer_name, model=model, **kwargs)
    
    patched_convert._unsloth_moe_patched = True
    patch_function(Bnb4bitQuantize, "convert", patched_convert, match_level = "relaxed")

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched Bnb4bitQuantize.convert for MoE expert parameter support")
pass
TEMPORARY_PATCHES.append(patch_bnb4bit_quantize_convert)


def patch_bnb4bit_quantizer_param_needs_quantization():
    """Recognize MoE expert modules of Params4bit type as needing quantization."""
    
    try:
        from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
        from transformers.quantizers.quantizers_utils import get_module_from_name
    except Exception as e:
        return raise_error("transformers.quantizers.quantizer_bnb_4bit.Bnb4BitHfQuantizer", e)
    
    if hasattr(Bnb4BitHfQuantizer.param_needs_quantization, "_unsloth_moe_patched"):
        return
    
    original_param_needs_quantization = Bnb4BitHfQuantizer.param_needs_quantization
    
    def patched_param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        if original_param_needs_quantization(self, model, param_name, **kwargs):
            return True
        
        try:
            module, name = get_module_from_name(model, param_name)
            if name in ("gate_up_proj", "down_proj"):
                param = getattr(module, name, None)
                # Only treat as MoE expert needing quantization if it's a
                # Params4bit that has NOT already been quantized (bnb_quantized=False).
                # This protects against a hypothetical re-invocation after first quantize.
                if (
                    isinstance(param, Params4bit)
                    and not getattr(param, "bnb_quantized", False)
                ):
                    return True
        except (KeyError, AttributeError) as e:
            # Expected when get_module_from_name can't resolve the name. Fall through.
            if UNSLOTH_ENABLE_LOGGING:
                logger.info(
                    f"Unsloth: param_needs_quantization fall-through for {param_name}: {e}"
                )
        
        return False
    
    patched_param_needs_quantization._unsloth_moe_patched = True
    patch_function(Bnb4BitHfQuantizer, "param_needs_quantization", patched_param_needs_quantization)

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched Bnb4BitHfQuantizer.param_needs_quantization for MoE expert parameters")
pass
TEMPORARY_PATCHES.append(patch_bnb4bit_quantizer_param_needs_quantization)


def patch_bnb4bit_quantizer_process_model():
    try:
        from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
    except Exception as e:
        return raise_error("transformers.quantizers.quantizer_bnb_4bit.Bnb4BitHfQuantizer", e)
    
    # Fast return if already patched
    if hasattr(Bnb4BitHfQuantizer._process_model_before_weight_loading, "_unsloth_moe_patched"):
        return
    
    original_process_model_before_weight_loading = Bnb4BitHfQuantizer._process_model_before_weight_loading
    
    def patched_process_model_before_weight_loading(self, model, device_map, **kwargs):
        original_process_model_before_weight_loading(self, model, device_map, **kwargs)
        
        # Use the patched replace_expert_params_with_bnb_params function
        model = replace_expert_params_with_bnb_params(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )
        return model
    
    patched_process_model_before_weight_loading._unsloth_moe_patched = True
    patch_function(Bnb4BitHfQuantizer, "_process_model_before_weight_loading", patched_process_model_before_weight_loading, match_level = "relaxed")
    pass
pass
TEMPORARY_PATCHES.append(patch_bnb4bit_quantizer_process_model)


def _maybe_dequant_params4bit_weight(weight, input_dtype):
    """If `weight` is a packed Params4bit, dequantize to logical shape and cast
    to `input_dtype`. Otherwise return `weight` unchanged.
    """
    if isinstance(weight, Params4bit) and getattr(weight, "quant_state", None) is not None:
        original_shape = getattr(weight, "_original_shape", None)
        dequant = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
        if original_shape is not None and tuple(dequant.shape) != tuple(original_shape):
            dequant = dequant.reshape(original_shape)
        return dequant.to(input_dtype)
    return weight


def patch_transformers_grouped_linear_4bit():
    """
    transformers v5's `grouped_mm_experts_forward` and `batched_mm_experts_forward`
    (in `transformers.integrations.moe`) read `self.gate_up_proj` /
    `self.down_proj` raw and pass them through `_grouped_linear` /
    `_batched_linear` -> `weight.transpose(-2, -1)` -> `torch._grouped_mm` /
    `torch.bmm` -> `mat_a.to(weight.dtype)` ...

    For MoE arches whose experts class is NOT replaced by an unsloth-zoo
    per-arch patch (e.g. some Glm4Moe variants, Gemma4MoE), the experts forward
    therefore sees the raw Params4bit (uint8 packed storage). The matmul ops raise:
        - grouped_mm path:  `RuntimeError: Expected mat_a to be Float32, BFloat16 or
                            Float16 matrix, got Byte` (during training)
        - batched_mm path:  `RuntimeError: batch1 must be a 3D tensor` (during
                            autoregressive decoding where the batched_mm dispatcher
                            is used instead of grouped_mm)

    Fix: wrap `_grouped_linear` and `_batched_linear` to detect Params4bit
    weights, dequantize via `bnb.functional.dequantize_4bit` (using
    `_original_shape` to recover the logical 3D `(E, in, out)` shape from
    packed `(N, 1)` storage), cast to the input dtype, then delegate to the
    original function which handles `is_transposed` orientation correctly.

    Forward-only -- base weights are frozen; gradient flow stays on LoRA paths
    that the per-arch wrappers (when they exist) inject separately. For arches
    without a per-arch wrapper this gives base-only forward; PEFT's
    `_activate_lora` parametrization adds the delta on top.
    """
    try:
        from transformers.integrations import moe as tf_moe
    except ImportError:
        return
    if not HAS_BNB:
        return

    if hasattr(tf_moe, "_grouped_linear") and not getattr(tf_moe._grouped_linear, "_unsloth_4bit_moe_patched", False):
        _original_grouped_linear = tf_moe._grouped_linear

        def _patched_grouped_linear(input, weight, offs, bias=None, is_transposed=False):
            weight = _maybe_dequant_params4bit_weight(weight, input.dtype)
            return _original_grouped_linear(input, weight, offs, bias=bias, is_transposed=is_transposed)

        _patched_grouped_linear._unsloth_4bit_moe_patched = True
        patch_function(tf_moe, "_grouped_linear", _patched_grouped_linear, match_level="relaxed")

    if hasattr(tf_moe, "_batched_linear") and not getattr(tf_moe._batched_linear, "_unsloth_4bit_moe_patched", False):
        _original_batched_linear = tf_moe._batched_linear

        def _patched_batched_linear(input, weight, bias=None, is_transposed=False):
            weight = _maybe_dequant_params4bit_weight(weight, input.dtype)
            return _original_batched_linear(input, weight, bias=bias, is_transposed=is_transposed)

        _patched_batched_linear._unsloth_4bit_moe_patched = True
        patch_function(tf_moe, "_batched_linear", _patched_batched_linear, match_level="relaxed")

    # batched_mm_experts_forward indexes `self.gate_up_proj[expert_ids]` BEFORE
    # passing to `_batched_linear`. For a packed Params4bit (storage shape
    # (N, 1) uint8), the indexing returns garbage (a (S, 1) uint8 slice) and
    # the dtype information is lost — `_patched_batched_linear` no longer sees
    # a Params4bit. Wrap the experts-forward dispatcher itself so the dequant
    # happens BEFORE indexing.
    if hasattr(tf_moe, "batched_mm_experts_forward") and not getattr(tf_moe.batched_mm_experts_forward, "_unsloth_4bit_moe_patched", False):
        _original_batched_mm_experts_forward = tf_moe.batched_mm_experts_forward

        def _patched_batched_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights):
            # Temporarily swap any packed Params4bit experts to dequantized 3D tensors
            # for the duration of this forward call. setattr-restore pattern keeps the
            # base layer Params4bit intact for subsequent calls / save / merge.
            swapped = []
            for attr in ("gate_up_proj", "down_proj", "up_proj"):
                w = getattr(self, attr, None)
                if isinstance(w, Params4bit) and getattr(w, "quant_state", None) is not None:
                    dequant = _maybe_dequant_params4bit_weight(w, hidden_states.dtype)
                    object.__setattr__(self, attr, dequant)
                    swapped.append((attr, w))
            try:
                return _original_batched_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights)
            finally:
                for attr, original in swapped:
                    object.__setattr__(self, attr, original)

        _patched_batched_mm_experts_forward._unsloth_4bit_moe_patched = True
        patch_function(tf_moe, "batched_mm_experts_forward", _patched_batched_mm_experts_forward, match_level="relaxed")
        # Also re-register in the dispatcher mapping so the forward decorator picks
        # up the patched version (the decorator stores a reference at class-decoration
        # time via ALL_EXPERTS_FUNCTIONS.get(...)).
        try:
            if hasattr(tf_moe, "ALL_EXPERTS_FUNCTIONS"):
                tf_moe.ALL_EXPERTS_FUNCTIONS["batched_mm"] = _patched_batched_mm_experts_forward
        except Exception:
            pass

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched transformers.integrations.moe._grouped_linear / _batched_linear / batched_mm_experts_forward for 4-bit MoE expert support")
pass
TEMPORARY_PATCHES.append(patch_transformers_grouped_linear_4bit)


def patch_transformers_weight_converter_kwargs():
    """
    PEFT 0.19's `convert_peft_adapter_state_dict_for_transformers` (in
    `peft/utils/transformers_weight_conversion.py:268`) constructs new
    `WeightConverter` instances via:
        new_conversion = orig_conversion.__class__(
            source_patterns=...,
            target_patterns=...,
            distributed_operation=...,
            quantization_operation=...,
            operations=...,
        )
    This is forward-compatibility code for a later transformers release. With
    transformers 5.6.2 the `WeightConverter.__init__` signature is
    `(self, source_patterns, target_patterns, operations)` and the extra
    kwargs raise `TypeError: WeightConverter.__init__() got an unexpected
    keyword argument 'distributed_operation'`. This blocks every
    `PeftModel.from_pretrained` for any MoE-fused 4-bit model on peft 0.19.

    Fix: patch `WeightConverter.__init__` to accept and ignore unknown kwargs
    (`distributed_operation`, `quantization_operation`, etc.). The original
    init already stores everything it needs from `source_patterns`,
    `target_patterns`, and `operations`. The ignored kwargs only affect
    distributed / quantization codepaths that aren't exercised at adapter-load
    time.
    """
    try:
        from transformers.core_model_loading import WeightConverter
    except ImportError:
        return

    if getattr(WeightConverter.__init__, "_unsloth_kwargs_patched", False):
        return

    _original_init = WeightConverter.__init__

    import inspect
    try:
        original_params = set(inspect.signature(_original_init).parameters)
    except (TypeError, ValueError):
        original_params = {"self", "source_patterns", "target_patterns", "operations"}

    def _patched_init(self, *args, **kwargs):
        # Drop kwargs that the installed transformers version does not accept.
        # Forwarding only the kwargs the original signature recognises makes us
        # forward-compatible with peft versions written against newer
        # transformers releases.
        accepted = {k: v for k, v in kwargs.items() if k in original_params}
        return _original_init(self, *args, **accepted)

    _patched_init._unsloth_kwargs_patched = True
    WeightConverter.__init__ = _patched_init

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched transformers WeightConverter.__init__ to ignore unknown kwargs (peft 0.19 forward-compat)")
pass
TEMPORARY_PATCHES.append(patch_transformers_weight_converter_kwargs)
