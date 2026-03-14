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
from typing import Optional, List, Tuple, Union
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


def _check_bnb_available():
    if not HAS_BNB:
        return False
    return True


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
        
        # TODO: Can remove this?
        logger.info(f"Unsloth: Prepared {module_name}'s gate_up_proj & down_proj for BNB 4-bit quantization (shapes: {gate_up_proj.shape}, {down_proj.shape})")
    
    if not has_been_replaced:
        logger.warning(f"Unsloth: No expert parameters were found to be replaced for {model.name_or_path}")
    
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
        input_dict:
            - Dictionary containing raw tensors for the parameter to be quantized.
            - For MoE module of nn.Parameter type, value is a tensor. TODO: Fix the comment
        full_layer_name: Name of the module to be quantized.
        """
        value = list(input_dict.values())[0]
        
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
        
        except Exception as e:
            logger.warning(f"Unsloth: Error handling expert param quantization for {full_layer_name}: {e}")
            pass

        # Fall back to original convert for non-expert params or in case of any failure
        return original_convert(self, input_dict, full_layer_name=full_layer_name, model=model, **kwargs)
    
    patched_convert._unsloth_moe_patched = True
    patch_function(Bnb4bitQuantize, "convert", patched_convert, match_level = "relaxed")
    
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
                if isinstance(param, Params4bit):
                    return True
        except Exception as e:
            # TODO: Can we raise an error here?
            logger.warning(
                f"Unsloth: Error checking MoE expert param_needs_quantization for {param_name}: {e}"
            )
            pass
        
        return False
    
    patched_param_needs_quantization._unsloth_moe_patched = True
    patch_function(Bnb4BitHfQuantizer, "param_needs_quantization", patched_param_needs_quantization)
    
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
