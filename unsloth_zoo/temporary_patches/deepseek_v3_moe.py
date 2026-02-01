# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import sys
from .common import (
    TEMPORARY_PATCHES,
    torch_compile,
    _torch_compile,
    get_torch_compile_options,
    UNSLOTH_ENABLE_LOGGING,
)
from .utils import (
    patch_function,
    patch_function_past_key_values,
    dedent,
    KWARGS_TYPE,
    raise_error,
    logger,
    Cache,
    process_return,
)
from ..hf_utils import dtype_from_config
from .moe_utils import (
    forward_moe_backend,
    patch_param_wrapper_for_moe,
)


# Module-level flag to track if we've attempted patching
_patch_attempted = False
_patch_successful = False


def _do_patch_deepseek_v3():
    """
    Internal function that actually applies the patches.
    Returns True if successful, False otherwise.
    """
    # All Unsloth Zoo code licensed under AGPL3

    global _patch_attempted, _patch_successful

    if _patch_successful:
        return True

    # Try to import the DeepSeekV3 MoE classes
    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
            DeepseekV3NaiveMoe,
            DeepseekV3MoE,
            DeepseekV3TopkRouter,
            DeepseekV3Config,
        )
    except Exception as e:
        # DeepSeekV3 not available yet
        return False

    # Check if already patched
    if hasattr(DeepseekV3NaiveMoe, "_unsloth_already_patched"):
        _patch_successful = True
        return True

    # Patch PEFT ParamWrapper for separated LoRA weights
    patch_param_wrapper_for_moe()

    # ====================================================================
    # Define LoRA extraction function for DeepSeekV3 (Standard Format)
    # ====================================================================
    def _deepseek_v3_lora_extractor(wrapper, weight_A, weight_B, scaling, num_experts):
        """
        Custom LoRA extractor for DeepSeekV3.

        DeepSeekV3 expert weights are stored as (E, out_dim, in_dim) and PEFT's ParamWrapper
        treats dim1 as in_features and dim2 as out_features. For correct separated LoRA
        (X @ first @ second), we need to pick the weight that connects to the actual input dim.
        """
        total_rank = weight_A.shape[0]
        rank_per_expert = total_rank // num_experts
        dim_A = weight_A.shape[1]
        dim_B = weight_B.shape[0]

        input_dim = None
        if hasattr(wrapper, "parameter_name"):
            if wrapper.parameter_name == "gate_up_proj":
                base = wrapper.get_base_layer() if hasattr(wrapper, "get_base_layer") else None
                input_dim = getattr(base, "hidden_dim", None)
            elif wrapper.parameter_name == "down_proj":
                base = wrapper.get_base_layer() if hasattr(wrapper, "get_base_layer") else None
                input_dim = getattr(base, "intermediate_dim", None)

        if input_dim is None:
            base = wrapper.get_base_layer() if hasattr(wrapper, "get_base_layer") else None
            input_dim = getattr(base, "hidden_dim", None)

        # If lora_A connects to input_dim: standard (A then B)
        if input_dim is not None and dim_A == input_dim:
            first_weight = weight_A.view(num_experts, rank_per_expert, dim_A)
            first_weight = first_weight.permute(0, 2, 1).contiguous()  # (E, input_dim, R)
            second_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
            second_weight = second_weight.permute(1, 2, 0).contiguous()  # (E, R, out_dim)
            return first_weight, second_weight, scaling, num_experts

        # If lora_B connects to input_dim: swapped (B then A)
        if input_dim is not None and dim_B == input_dim:
            first_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
            first_weight = first_weight.permute(1, 0, 2).contiguous()  # (E, input_dim, R)
            second_weight = weight_A.view(num_experts, rank_per_expert, dim_A).contiguous()  # (E, R, out_dim)
            return first_weight, second_weight, scaling, num_experts

        # Fallback: standard (A then B)
        first_weight = weight_A.view(num_experts, rank_per_expert, dim_A)
        first_weight = first_weight.permute(0, 2, 1).contiguous()
        second_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
        second_weight = second_weight.permute(1, 2, 0).contiguous()
        return first_weight, second_weight, scaling, num_experts

    # Register the extractor on the NaiveMoe class (avoid binding as instance method)
    DeepseekV3NaiveMoe._unsloth_lora_extractor_fn = staticmethod(_deepseek_v3_lora_extractor)
    # Also mark the model type for weight preprocessing
    DeepseekV3NaiveMoe._unsloth_model_type = "deepseek_v3"
    DeepseekV3NaiveMoe._unsloth_already_patched = True

    # ====================================================================
    # Patch DeepseekV3NaiveMoe.forward to use backend dispatch in moe_utils
    # ====================================================================

    def naive_moe_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Patched forward for Expert layer.
        Dispatches to moe_utils backend selection.
        """
        return forward_moe_backend(self, hidden_states, top_k_index, top_k_weights)

    # Apply patch to DeepseekV3NaiveMoe
    DeepseekV3NaiveMoe.forward = naive_moe_forward

    # ====================================================================
    # Patch DeepseekV3MoE.forward to mark model type
    # ====================================================================

    def patched_moe_forward(self, hidden_states):
        """
        Patched forward that adds model type marker for proper LoRA extraction.
        """
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # Mark the experts module for proper LoRA extraction
        self.experts._unsloth_model_type = "deepseek_v3"

        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(
            *orig_shape
        )
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states

    # Apply patch to DeepseekV3MoE
    DeepseekV3MoE.forward = patched_moe_forward

    _patch_successful = True

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched DeepSeekV3 MoE for Split LoRA support.")

    return True


def patch_deepseek_v3():
    """
    Patches DeepSeekV3 MoE to support Split LoRA using grouped GEMM.

    This function attempts to patch immediately if DeepSeekV3 is available.
    If not, it sets up sys.meta_path hooks to patch when DeepSeekV3 is imported.
    """
    # All Unsloth Zoo code licensed under AGPL3

    global _patch_attempted

    if _patch_attempted:
        return
    _patch_attempted = True

    # Try immediate patching first
    if _do_patch_deepseek_v3():
        return

    # If that didn't work, set up sys.meta_path import finder
    _setup_meta_path_hook()


class DeepSeekV3Finder:
    """
    Import finder that patches DeepSeekV3 when it's first imported.
    """
    # All Unsloth Zoo code licensed under AGPL3

    def find_module(self, fullname, path=None):
        # We only care about transformers.models.deepseek_v3
        if fullname == "transformers.models.deepseek_v3" or fullname.startswith(
            "transformers.models.deepseek_v3."
        ):
            return self
        return None

    def find_spec(self, fullname, path, target=None):
        # Python 3.4+ uses find_spec instead of find_module
        if fullname == "transformers.models.deepseek_v3" or fullname.startswith(
            "transformers.models.deepseek_v3."
        ):
            return None  # Let normal import proceed, we'll patch after
        return None

    def load_module(self, fullname):
        # This shouldn't be called in Python 3.4+, but just in case
        return None


def _setup_meta_path_hook():
    """
    Set up sys.meta_path import finder to catch DeepSeekV3 imports.
    """
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Setting up DeepSeekV3 import hook for delayed patching.")

    # Insert our finder at the beginning of sys.meta_path
    finder = DeepSeekV3Finder()
    if finder not in sys.meta_path:
        sys.meta_path.insert(0, finder)

    # Also monkey-patch __import__ as a backup
    _setup_import_hook()


def _setup_import_hook():
    """
    Set up __import__ hook as a backup to sys.meta_path.
    """
    import builtins

    original_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Call original import
        module = original_import(name, globals, locals, fromlist, level)

        # Check if this is DeepSeekV3 being imported
        if (
            name == "transformers.models.deepseek_v3"
            or name == "transformers.models.deepseek_v3.modeling_deepseek_v3"
            or (fromlist and any("deepseek_v3" in str(f) for f in fromlist))
        ):
            # Try to apply patches
            _do_patch_deepseek_v3()

        return module

    builtins.__import__ = patched_import


# Register the patch - it will be called when unsloth is imported
TEMPORARY_PATCHES.append(patch_deepseek_v3)

# Also try to patch right now in case DeepSeekV3 is already imported
try:
    _do_patch_deepseek_v3()
except:
    pass
