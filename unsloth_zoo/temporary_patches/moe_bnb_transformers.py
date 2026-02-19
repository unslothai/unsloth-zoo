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

Key features:
- Preserves Qwen3MoeExperts module structure (compatible with grouped GEMM)
- Preserves _original_shape for PEFT LoRA compatibility
- Minimal code changes - hooks into existing transformers quantization pipeline
- Robust to different transformers versions and model architectures
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
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
    """Check if bitsandbytes is available."""
    if not HAS_BNB:
        return False
    return True


def _is_expert_param_name(full_layer_name: str) -> bool:
    """Check if a parameter name indicates an MoE expert parameter."""
    if not full_layer_name:
        return False
    return "gate_up_proj" in full_layer_name or "down_proj" in full_layer_name


def _is_expert_module(module: nn.Module) -> bool:
    """Check if a module is an MoE experts module."""
    return (
        hasattr(module, "gate_up_proj")
        and hasattr(module, "down_proj")
        and isinstance(module.gate_up_proj, nn.Parameter)
        and isinstance(module.down_proj, nn.Parameter)
    )


def replace_expert_params_with_bnb_params(
    model: nn.Module,
    modules_to_not_convert: Optional[List[str]] = None,
    quantization_config=None,
    pre_quantized: bool = False,
) -> nn.Module:
    """
    Replace MoE expert parameters (gate_up_proj, down_proj) with Params4bit placeholders.
    
    This follows the same pattern as replace_with_bnb_linear - creates placeholders
    on meta device that will be quantized during weight loading.
    
    Args:
        model: The model to convert
        modules_to_not_convert: List of module names to skip
        quantization_config: BitsAndBytesConfig with quantization parameters
        pre_quantized: Whether the model is pre-quantized
    
    Returns:
        The model with expert params replaced
    """
    if not _check_bnb_available():
        return model
    
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    
    has_been_replaced = False
    
    # Default quantization settings
    compute_dtype = torch.bfloat16
    compress_statistics = True
    quant_type = "nf4"
    quant_storage = torch.uint8
    
    if quantization_config is not None:
        compute_dtype = getattr(quantization_config, 'bnb_4bit_compute_dtype', torch.bfloat16)
        compress_statistics = getattr(quantization_config, 'bnb_4bit_use_double_quant', True)
        quant_type = getattr(quantization_config, 'bnb_4bit_quant_type', 'nf4')
        quant_storage = getattr(quantization_config, 'bnb_4bit_quant_storage', torch.uint8)
    
    # Find all MoE expert modules
    for module_name, module in model.named_modules():
        # Skip if in modules_to_not_convert
        if any(skip_name in module_name for skip_name in modules_to_not_convert):
            continue
        
        if not _is_expert_module(module):
            continue
        
        gate_up_proj = module.gate_up_proj
        down_proj = module.down_proj
        
        # Skip if already quantized
        if isinstance(gate_up_proj, Params4bit) or isinstance(down_proj, Params4bit):
            continue
        
        # Get original shapes for PEFT LoRA compatibility
        gate_up_shape = tuple(gate_up_proj.shape)
        down_proj_shape = tuple(down_proj.shape)
        
        # Create Params4bit placeholders on meta device
        with torch.device("meta"):
            placeholder_gate_up = Params4bit(
                torch.zeros(gate_up_shape, device="meta"),
                requires_grad=False,
                compress_statistics=compress_statistics,
                quant_type=quant_type,
                quant_storage=quant_storage,
            )
            placeholder_down = Params4bit(
                torch.zeros(down_proj_shape, device="meta"),
                requires_grad=False,
                compress_statistics=compress_statistics,
                quant_type=quant_type,
                quant_storage=quant_storage,
            )
        
        # Preserve original shape for PEFT LoRA (critical!)
        setattr(placeholder_gate_up, "_original_shape", gate_up_shape)
        setattr(placeholder_down, "_original_shape", down_proj_shape)
        
        if pre_quantized:
            # Set the right dtype for pre-quantized models
            placeholder_gate_up.data = placeholder_gate_up.data.to(dtype=quant_storage)
            placeholder_down.data = placeholder_down.data.to(dtype=quant_storage)
        
        # Replace the parameters
        module.gate_up_proj = placeholder_gate_up
        module.down_proj = placeholder_down
        has_been_replaced = True
        
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(
                f"Unsloth: Prepared {module_name}.gate_up_proj and .down_proj for BNB 4-bit quantization "
                f"(shape: {gate_up_shape}, {down_proj_shape})"
            )
    
    return model


def patch_bnb4bit_quantize_convert():
    """
    Patch Bnb4bitQuantize.convert to handle expert parameters correctly.
    
    This ensures:
    1. Expert parameters preserve _original_shape for PEFT LoRA
    2. Handles both single tensor and multi-tensor cases robustly
    """
    if not _check_bnb_available():
        return
    
    try:
        from transformers.integrations.bitsandbytes import Bnb4bitQuantize
    except Exception as e:
        return raise_error("transformers.integrations.bitsandbytes.Bnb4bitQuantize", e)
    
    # Check if already patched
    if hasattr(Bnb4bitQuantize.convert, "_unsloth_moe_patched"):
        return
    
    original_convert = Bnb4bitQuantize.convert
    
    def patched_convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        full_layer_name: str | None = None,
        model: torch.nn.Module | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Patched convert that handles expert parameters correctly.
        """
        value = list(input_dict.values())[0]
        
        # Handle expert parameters - they may come as lists or single tensors
        is_expert_param = _is_expert_param_name(full_layer_name)
        
        if isinstance(value, torch.Tensor) and is_expert_param:
            # expert param with multiple tensors
            pass
        else:
            # for non-expert params, take the first tensor
            value = value[0]
        
        # Handle expert parameters specially to preserve _original_shape
        if is_expert_param and model is not None and full_layer_name is not None:
            try:
                from transformers.quantizers.quantizers_utils import get_module_from_name
                module, _ = get_module_from_name(model, full_layer_name)
                
                # Get old parameter to preserve attributes
                old_value = model.get_parameter_or_buffer(full_layer_name)
                
                # Preserve _original_shape if it exists, otherwise use current shape
                original_shape = getattr(old_value, "_original_shape", None)
                if original_shape is None and hasattr(value, 'shape'):
                    original_shape = tuple(value.shape)
                
                # Filter out _original_shape from old_dict (Params4bit doesn't accept it in __init__)
                old_dict = {k: v for k, v in old_value.__dict__.items() if k != "_original_shape"}

                # Support Conv1D transpose if needed (from original transformers code)
                if hasattr(module, "source_cls"):
                    try:
                        from transformers.pytorch_utils import Conv1D
                        if issubclass(module.source_cls, Conv1D):
                            value = value.T
                    except:
                        pass
                
                # Create Params4bit
                new_value = Params4bit(value, requires_grad=False, **old_dict).to(value.device)
                
                # Preserve _original_shape for expert params (critical for PEFT LoRA!)
                if original_shape is not None:
                    setattr(new_value, "_original_shape", original_shape)
                    if UNSLOTH_ENABLE_LOGGING:
                        logger.debug(
                            f"Unsloth: Preserved _original_shape={original_shape} for {full_layer_name}"
                        )
                
                # Set module initialization flag if it exists
                if hasattr(module, "_is_hf_initialized"):
                    module._is_hf_initialized = True
                
                return {full_layer_name: new_value}
            except Exception as e:
                if UNSLOTH_ENABLE_LOGGING:
                    logger.warning(f"Unsloth: Error handling expert param {full_layer_name}: {e}")
                # Fall back to original behavior
                pass
        
        # Fall back to original convert for non-expert params or if module lookup fails
        return original_convert(self, input_dict, full_layer_name=full_layer_name, model=model, **kwargs)
    
    patched_convert._unsloth_moe_patched = True
    patch_function(Bnb4bitQuantize, "convert", patched_convert)
    
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched Bnb4bitQuantize.convert for MoE expert parameter support")


def patch_bnb4bit_quantizer_param_needs_quantization():
    """
    Patch Bnb4BitHfQuantizer.param_needs_quantization to recognize expert parameters
    (gate_up_proj, down_proj) that are Params4bit placeholders as needing quantization.
    
    This ensures quantization ops are assigned during weight loading for expert params.
    """
    if not _check_bnb_available():
        return
    
    try:
        from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
        from transformers.quantizers.quantizers_utils import get_module_from_name
    except Exception as e:
        return raise_error("transformers.quantizers.quantizer_bnb_4bit.Bnb4BitHfQuantizer", e)
    
    # Check if already patched
    if hasattr(Bnb4BitHfQuantizer.param_needs_quantization, "_unsloth_moe_patched"):
        return
    
    original_param_needs_quantization = Bnb4BitHfQuantizer.param_needs_quantization
    
    def patched_param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        """Patched to recognize expert Params4bit placeholders as needing quantization."""
        # First check the original logic (Linear4bit weights)
        if original_param_needs_quantization(self, model, param_name, **kwargs):
            return True
        
        # Then check if it's an expert parameter (gate_up_proj or down_proj) that is a Params4bit placeholder
        try:
            module, name = get_module_from_name(model, param_name)
            if name in ("gate_up_proj", "down_proj"):
                param = getattr(module, name, None)
                if isinstance(param, Params4bit):
                    return True
        except Exception:
            # If we can't check, fall back to original behavior
            pass
        
        return False
    
    patched_param_needs_quantization._unsloth_moe_patched = True
    patch_function(Bnb4BitHfQuantizer, "param_needs_quantization", patched_param_needs_quantization)
    
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched Bnb4BitHfQuantizer.param_needs_quantization for MoE expert parameters")


def patch_bnb4bit_quantizer_process_model():
    """
    Patch Bnb4BitHfQuantizer._process_model_before_weight_loading to ensure
    replace_expert_params_with_bnb_params is called.
    
    We patch the quantizer directly (rather than replace_with_bnb_linear) because
    the quantizer may have cached its import of replace_with_bnb_linear from
    _LazyModule before our patch runs, so our patch would never be used.
    """
    if not _check_bnb_available():
        return
    
    try:
        from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
    except Exception as e:
        return raise_error("transformers.quantizers.quantizer_bnb_4bit.Bnb4BitHfQuantizer", e)
    
    # Check if already patched
    if hasattr(Bnb4BitHfQuantizer._process_model_before_weight_loading, "_unsloth_moe_patched"):
        return
    
    original_process = Bnb4BitHfQuantizer._process_model_before_weight_loading
    
    def patched_process_model_before_weight_loading(self, model, device_map, **kwargs):
        """Patched to use our replace_expert_params_with_bnb_params (works regardless of import order)."""
        from transformers.integrations.bitsandbytes import replace_with_bnb_linear
        
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.llm_int8_skip_modules, model._keep_in_fp32_modules
        )
        if self.quantization_config.llm_int8_enable_fp32_cpu_offload:
            if isinstance(device_map, dict):
                keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]
                self.modules_to_not_convert.extend(keys_on_cpu)
        
        model = replace_with_bnb_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )
        
        # Use our replace_expert_params_with_bnb_params (guaranteed to be called)
        model = replace_expert_params_with_bnb_params(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )
        return model
    
    patched_process_model_before_weight_loading._unsloth_moe_patched = True
    patch_function(Bnb4BitHfQuantizer, "_process_model_before_weight_loading", patched_process_model_before_weight_loading, match_level = "relaxed")
    
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched Bnb4BitHfQuantizer._process_model_before_weight_loading for MoE expert parameters")


def patch_moe_bnb_transformers():
    """
    Main patch function that applies all MoE BNB quantization patches.
    
    This patches transformers' quantization pipeline at three strategic points:
    
    1. _process_model_before_weight_loading: Creates Params4bit placeholders for expert params
       (called BEFORE weight loading, prepares the model structure)
    
    2. param_needs_quantization: Recognizes expert Params4bit placeholders as needing quantization
       (called DURING weight loading planning, ensures quantization ops are assigned)
    
    3. convert: Handles expert param quantization during actual weight loading
       (called DURING weight loading, performs the quantization)
    
    This should be called early (via TEMPORARY_PATCHES) to ensure
    patches are applied before model loading.
    """
    if not _check_bnb_available():
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning("Unsloth: bitsandbytes not available, skipping MoE BNB patches")
        return
    
    # Stage 1: Create Params4bit placeholders BEFORE weight loading
    patch_bnb4bit_quantizer_process_model()
    
    # Stage 2: Recognize expert params DURING loading planning
    patch_bnb4bit_quantizer_param_needs_quantization()
    
    # Stage 3: Quantize expert params DURING actual weight loading
    patch_bnb4bit_quantize_convert()
    
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Applied MoE expert parameter 4-bit quantization patches")


TEMPORARY_PATCHES.append(patch_moe_bnb_transformers)
