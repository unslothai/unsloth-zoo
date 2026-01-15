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

"""
MoE 4-bit Quantization Module

Provides bitsandbytes-style 4-bit quantization for Mixture of Experts layers.
This is necessary because transformers' Qwen3MoeExperts uses nn.Parameter
tensors instead of nn.Linear modules, so bitsandbytes' standard quantization
skips them.

Design follows bitsandbytes patterns:
- Uses Params4bit for per-expert weight quantization
- Quantization happens on .to(cuda), just like Linear4bit
- Integrates with model loading via replace_with_bnb_moe_experts()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Any, Tuple
import os
import warnings
from ..log import logger

UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"

# Check bitsandbytes availability
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Params4bit
    from bitsandbytes.functional import dequantize_4bit
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    Params4bit = None


def _check_bnb_available():
    """Check if bitsandbytes is available and raise helpful error if not."""
    if not HAS_BNB:
        raise ImportError(
            "bitsandbytes is required for MoE 4-bit quantization. "
            "Install via: pip install bitsandbytes"
        )


class MoeExperts4bit(nn.Module):
    """
    Base class for 4-bit quantized MoE experts.

    Follows bitsandbytes Linear4bit patterns for compatibility and maintainability:
    - Stores per-expert weights as Params4bit
    - Quantization happens when moving to CUDA (like Linear4bit)
    - Self-contained forward using bnb.matmul_4bit for each expert

    The module replaces Qwen3MoeExperts or similar, providing the same interface
    but with quantized weights.

    NOTE: BNB 4-bit MoE uses loop-based forward (each expert processed individually).
    For maximum throughput, consider using grouped_mm backend with native bf16 weights.
    Set UNSLOTH_MOE_BACKEND=grouped_mm environment variable.
    """
    _warned_loop_based = False

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        compute_dtype: Optional[torch.dtype] = None,
        compress_statistics: bool = True,
        quant_type: str = "nf4",
        quant_storage: torch.dtype = torch.uint8,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize MoeExperts4bit.

        Args:
            num_experts: Number of experts in the MoE layer
            hidden_dim: Hidden dimension of the model
            intermediate_dim: Intermediate dimension of FFN
            compute_dtype: Dtype for computation (typically bf16 or fp16)
            compress_statistics: Whether to compress quantization statistics
            quant_type: Quantization type ("nf4" or "fp4")
            quant_storage: Storage dtype for quantized weights
            device: Device to place weights on
        """
        super().__init__()
        _check_bnb_available()

        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.compute_dtype = compute_dtype
        self.compute_type_is_set = compute_dtype is not None
        self.compress_statistics = compress_statistics
        self.quant_type = quant_type
        self.quant_storage = quant_storage

        # Flag for detection
        self._is_bnb_4bit = True

        # Per-expert quantized weights stored as ParameterLists for proper registration
        # Each expert's gate_up_proj: Params4bit of shape [2*intermediate_dim, hidden_dim]
        # Each expert's down_proj: Params4bit of shape [hidden_dim, intermediate_dim]
        self._bnb_gate_up_weights = nn.ParameterList()
        self._bnb_down_weights = nn.ParameterList()

        # These will hold the original Parameters for weight loading
        # They get converted to Params4bit when .to(cuda) is called
        self._gate_up_proj_pending = None
        self._down_proj_pending = None

        # Activation function
        self.act_fn = F.silu

    def set_compute_type(self, x: torch.Tensor):
        """Set compute dtype based on input, following Linear4bit pattern."""
        if x.dtype in [torch.float32, torch.bfloat16]:
            self.compute_dtype = x.dtype
        elif x.dtype == torch.float16:
            if self.compute_dtype in [None, torch.float32]:
                warnings.warn(
                    "Input type into MoeExperts4bit is torch.float16, but compute_dtype=torch.float32. "
                    "This may lead to slow inference.",
                )
                warnings.filterwarnings("ignore", message=".*slow inference.*")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        Override to handle loading 3D stacked weights and convert to Params4bit.

        This is called during model.load_state_dict() - the key place where on-the-fly
        quantization happens.
        """
        gate_up_key = prefix + "gate_up_proj"
        down_key = prefix + "down_proj"

        # Check if loading original stacked format
        if gate_up_key in state_dict:
            # Store the weights temporarily - will be quantized on .to(cuda)
            gate_up_proj = state_dict.pop(gate_up_key)
            down_proj = state_dict.pop(down_key)

            # If on CUDA, quantize immediately
            if gate_up_proj.device.type == 'cuda':
                self._quantize_and_store(gate_up_proj, down_proj)
            else:
                # Store for later quantization on .to(cuda)
                self.register_buffer("_gate_up_proj_pending", gate_up_proj)
                self.register_buffer("_down_proj_pending", down_proj)
        else:
            # Try loading per-expert quantized format
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _quantize_and_store(self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor):
        """Quantize stacked weights to per-expert Params4bit."""
        num_experts = gate_up_proj.shape[0]
        self.num_experts = num_experts

        # Clear existing
        self._bnb_gate_up_weights = nn.ParameterList()
        self._bnb_down_weights = nn.ParameterList()

        device = gate_up_proj.device

        for expert_idx in range(num_experts):
            # Extract 2D weight for this expert
            gate_up_weight = gate_up_proj[expert_idx].data.clone()  # [2*I, H]
            down_weight = down_proj[expert_idx].data.clone()  # [H, I]

            # Create Params4bit - quantization happens here since on CUDA
            gate_up_4bit = Params4bit(
                gate_up_weight,
                requires_grad=False,
                compress_statistics=self.compress_statistics,
                quant_type=self.quant_type,
                quant_storage=self.quant_storage,
                module=None,
            )

            down_4bit = Params4bit(
                down_weight,
                requires_grad=False,
                compress_statistics=self.compress_statistics,
                quant_type=self.quant_type,
                quant_storage=self.quant_storage,
                module=None,
            )

            # Move to device (triggers quantization)
            gate_up_4bit = gate_up_4bit.to(device)
            down_4bit = down_4bit.to(device)

            self._bnb_gate_up_weights.append(gate_up_4bit)
            self._bnb_down_weights.append(down_4bit)

        # Clear pending buffers
        if hasattr(self, "_gate_up_proj_pending") and self._gate_up_proj_pending is not None:
            del self._gate_up_proj_pending
        if hasattr(self, "_down_proj_pending") and self._down_proj_pending is not None:
            del self._down_proj_pending

    def _apply(self, fn, recurse=True):
        """
        Override _apply to handle quantization when .to(cuda) is called.

        This is the key hook for on-the-fly quantization - when the model is moved
        to CUDA, pending weights are quantized.
        """
        # Check if moving to CUDA and have pending weights
        if hasattr(self, "_gate_up_proj_pending") and self._gate_up_proj_pending is not None:
            # Apply the function to get the target device
            test_tensor = fn(torch.zeros(1))
            if test_tensor.device.type == 'cuda':
                # Quantize now
                pending_gate_up = fn(self._gate_up_proj_pending)
                pending_down = fn(self._down_proj_pending)
                self._quantize_and_store(pending_gate_up, pending_down)
                # Skip applying to pending weights since we've consumed them
                return self

        return super()._apply(fn, recurse)

    def _matmul_4bit(self, x: torch.Tensor, weight: Params4bit) -> torch.Tensor:
        """
        Perform 4-bit matmul using bitsandbytes, following Linear4bit.forward pattern.
        """
        quant_state = weight.quant_state
        w = weight.t()
        return bnb.matmul_4bit(x, w, bias=None, quant_state=quant_state)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optimized forward pass using bnb.matmul_4bit for each expert.
        Only iterates over experts that actually have tokens routed to them.
        """
        # Set compute type on first forward (like Linear4bit)
        if not self.compute_type_is_set:
            self.set_compute_type(hidden_states)
            self.compute_type_is_set = True

            # Warn about loop-based forward (only once)
            if UNSLOTH_ENABLE_LOGGING and not MoeExperts4bit._warned_loop_based:
                MoeExperts4bit._warned_loop_based = True
                logger.warning(
                    "Unsloth: Using BNB 4-bit MoE with loop-based forward. "
                    "For higher throughput, consider using grouped_mm backend with bf16 model."
                )

        inp_dtype = hidden_states.dtype
        if self.compute_dtype is not None:
            hidden_states = hidden_states.to(self.compute_dtype)

        final_hidden_states = torch.zeros_like(hidden_states)

        # Optimized routing: only process experts that have tokens
        # This is much faster than iterating over all 128 experts
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts, top_k, num_tokens]
            # Find which experts actually have tokens (sum across top_k and tokens dims)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)

        # Only loop over experts that have tokens routed to them
        for expert_idx_t in expert_hit:
            expert_idx = expert_idx_t.item()
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[token_idx]

            # Get quantized weights for this expert
            gate_up_weight = self._bnb_gate_up_weights[expert_idx]
            down_weight = self._bnb_down_weights[expert_idx]

            # Compute gate_up projection using bnb.matmul_4bit
            gate_up_out = self._matmul_4bit(current_state, gate_up_weight)
            gate, up = gate_up_out.chunk(2, dim=-1)

            # Activation
            current_hidden_states = self.act_fn(gate) * up

            # Compute down projection using bnb.matmul_4bit
            current_hidden_states = self._matmul_4bit(current_hidden_states, down_weight)

            # Apply routing weights
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

            # Scatter back
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states
            )

        return final_hidden_states.to(inp_dtype)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Save quantized weights to state dict, following Linear4bit pattern."""
        # Handle case where weights haven't been loaded yet (empty ParameterLists)
        if len(self._bnb_gate_up_weights) == 0:
            # Fall back to default state dict saving
            super()._save_to_state_dict(destination, prefix, keep_vars)
            return

        for expert_idx in range(self.num_experts):
            gate_up_4bit = self._bnb_gate_up_weights[expert_idx]
            down_4bit = self._bnb_down_weights[expert_idx]

            # Save weight data
            gate_up_prefix = f"{prefix}gate_up_projs.{expert_idx}."
            down_prefix = f"{prefix}down_projs.{expert_idx}."

            destination[f"{gate_up_prefix}weight"] = (
                gate_up_4bit.data if keep_vars else gate_up_4bit.data.detach()
            )
            destination[f"{down_prefix}weight"] = (
                down_4bit.data if keep_vars else down_4bit.data.detach()
            )

            # Save quant state components
            if hasattr(gate_up_4bit, 'quant_state') and gate_up_4bit.quant_state is not None:
                for k, v in gate_up_4bit.quant_state.as_dict(packed=True).items():
                    destination[f"{gate_up_prefix}weight.{k}"] = v if keep_vars else v.detach()

            if hasattr(down_4bit, 'quant_state') and down_4bit.quant_state is not None:
                for k, v in down_4bit.quant_state.as_dict(packed=True).items():
                    destination[f"{down_prefix}weight.{k}"] = v if keep_vars else v.detach()


class Qwen3MoeExperts4bit(MoeExperts4bit):
    """4-bit quantized version of Qwen3MoeExperts."""
    pass


class Qwen3VLMoeExperts4bit(MoeExperts4bit):
    """4-bit quantized version of Qwen3VLMoeTextExperts."""
    pass


def replace_with_bnb_moe_experts(
    model: nn.Module,
    quantization_config=None,
) -> Tuple[nn.Module, bool]:
    """
    Replace MoE expert modules with 4-bit quantized versions BEFORE weights are loaded.

    This follows the same pattern as transformers' replace_with_bnb_linear - we replace
    the modules on meta device, then weights are loaded and quantized on-the-fly.

    Args:
        model: The model to convert
        quantization_config: BitsAndBytesConfig with quantization parameters

    Returns:
        Tuple of (model, has_been_replaced)
    """
    _check_bnb_available()

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
    for module_name, module in list(model.named_modules()):
        if hasattr(module, 'experts') and hasattr(module.experts, 'gate_up_proj'):
            experts = module.experts

            # Skip if already quantized
            if getattr(experts, '_is_bnb_4bit', False):
                continue

            # Get dimensions from the original module
            gate_up_proj = experts.gate_up_proj
            num_experts = gate_up_proj.shape[0]
            intermediate_dim = gate_up_proj.shape[1] // 2
            hidden_dim = gate_up_proj.shape[2]

            # Determine quantized class based on module type
            module_class_name = type(module).__name__
            if 'Qwen3VL' in module_class_name or 'qwen3_vl' in type(module).__module__:
                quantized_cls = Qwen3VLMoeExperts4bit
            else:
                quantized_cls = Qwen3MoeExperts4bit

            # Create quantized version on meta device (like replace_with_bnb_linear does)
            with torch.device("meta"):
                new_experts = quantized_cls(
                    num_experts=num_experts,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    compute_dtype=compute_dtype,
                    compress_statistics=compress_statistics,
                    quant_type=quant_type,
                    quant_storage=quant_storage,
                )

            # Copy activation function if present
            if hasattr(experts, 'act_fn'):
                new_experts.act_fn = experts.act_fn

            # Replace the module
            module.experts = new_experts
            has_been_replaced = True

            if UNSLOTH_ENABLE_LOGGING:
                logger.info(
                    f"Unsloth: Prepared {module_name}.experts for BNB 4-bit quantization "
                    f"({num_experts} experts, hidden={hidden_dim}, intermediate={intermediate_dim})"
                )

    return model, has_been_replaced


def is_moe_quantized(module: nn.Module) -> bool:
    """Check if an MoE experts module is 4-bit quantized."""
    return getattr(module, '_is_bnb_4bit', False)


# Compatibility functions for post-loading quantization (kept for manual use)
def quantize_moe_experts_inplace(model: nn.Module, verbose: bool = True) -> int:
    """
    Post-loading quantization (uses more memory but works with any model).

    NOTE: Prefer replace_with_bnb_moe_experts() for on-the-fly quantization
    which uses less peak memory.
    """
    _check_bnb_available()

    count = 0

    for name, module in list(model.named_modules()):
        if hasattr(module, 'experts') and hasattr(module.experts, 'gate_up_proj'):
            experts = module.experts

            if getattr(experts, '_is_bnb_4bit', False):
                continue

            module_class_name = type(module).__name__
            if 'Qwen3VL' in module_class_name or 'qwen3_vl' in type(module).__module__:
                quantized_cls = Qwen3VLMoeExperts4bit
            else:
                quantized_cls = Qwen3MoeExperts4bit

            compute_dtype = experts.gate_up_proj.dtype

            if verbose:
                print(f"Unsloth: Quantizing {name}.experts to 4-bit...")

            # Create new quantized module and copy weights
            new_experts = quantized_cls(
                num_experts=experts.gate_up_proj.shape[0],
                hidden_dim=experts.gate_up_proj.shape[2],
                intermediate_dim=experts.gate_up_proj.shape[1] // 2,
                compute_dtype=compute_dtype,
            )
            new_experts._quantize_and_store(experts.gate_up_proj, experts.down_proj)

            if hasattr(experts, 'act_fn'):
                new_experts.act_fn = experts.act_fn

            module.experts = new_experts
            del experts
            torch.cuda.empty_cache()

            count += 1

    if verbose and count > 0:
        print(f"Unsloth: Quantized {count} MoE expert modules to 4-bit.")

    return count


def post_quantize_moe_experts(model: nn.Module, load_in_4bit: bool = False, verbose: bool = True) -> nn.Module:
    """
    Post-loading hook to quantize MoE experts to 4-bit.

    NOTE: This uses post-quantization which requires more peak memory.
    For lower memory usage, use replace_with_bnb_moe_experts() before weight loading.
    """
    if not load_in_4bit or not HAS_BNB:
        return model

    has_moe = False
    for name, module in model.named_modules():
        if hasattr(module, 'experts') and hasattr(module.experts, 'gate_up_proj'):
            has_moe = True
            break

    if not has_moe:
        return model

    if verbose:
        print("Unsloth: Quantizing MoE experts to 4-bit...")

    count = quantize_moe_experts_inplace(model, verbose=verbose)

    return model
