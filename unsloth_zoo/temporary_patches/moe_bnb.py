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
"""bitsandbytes-style 4-bit quantization for Mixture of Experts layers.

Needed because transformers' Qwen3MoeExperts uses nn.Parameter tensors
instead of nn.Linear modules, which bitsandbytes' standard quantization
skips. Follows bnb patterns: Params4bit per-expert weights, quantization
on .to(cuda) like Linear4bit, integrated via replace_with_bnb_moe_experts().
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
    """Raise a helpful error if bitsandbytes is unavailable."""
    if not HAS_BNB:
        raise ImportError(
            "bitsandbytes is required for MoE 4-bit quantization. "
            "Install via: pip install bitsandbytes"
        )


class MoeExperts4bit(nn.Module):
    """Base class for 4-bit quantized MoE experts.

    Follows bitsandbytes Linear4bit patterns: per-expert Params4bit weights,
    quantized on .to(cuda), self-contained forward via bnb.matmul_4bit per
    expert. Replaces Qwen3MoeExperts (or similar) with the same interface.

    NOTE: forward is loop-based (each expert processed individually). For max
    throughput use the grouped_mm backend with native bf16 weights:
    set UNSLOTH_MOE_BACKEND=grouped_mm.
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
        """quant_type is "nf4" or "fp4"; compute_dtype is typically bf16/fp16."""
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

        self._is_bnb_4bit = True

        # Per-expert quantized weights as ParameterLists (proper registration).
        # gate_up_proj: [2*intermediate_dim, hidden_dim]; down_proj: [hidden_dim, intermediate_dim]
        self._bnb_gate_up_weights = nn.ParameterList()
        self._bnb_down_weights = nn.ParameterList()

        # Original Parameters held for loading; converted to Params4bit on .to(cuda)
        self._gate_up_proj_pending = None
        self._down_proj_pending = None

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
        """Load 3D stacked weights and convert to Params4bit.

        Called during model.load_state_dict() - where on-the-fly quantization happens.
        """
        gate_up_key = prefix + "gate_up_proj"
        down_key = prefix + "down_proj"

        # Loading original stacked format
        if gate_up_key in state_dict:
            gate_up_proj = state_dict.pop(gate_up_key)
            down_proj = state_dict.pop(down_key)

            if gate_up_proj.device.type == 'cuda':
                self._quantize_and_store(gate_up_proj, down_proj)
            else:
                # Defer quantization until .to(cuda)
                self.register_buffer("_gate_up_proj_pending", gate_up_proj)
                self.register_buffer("_down_proj_pending", down_proj)
        else:
            # Per-expert quantized format
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _quantize_and_store(self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor):
        """Quantize stacked weights to per-expert Params4bit."""
        num_experts = gate_up_proj.shape[0]
        self.num_experts = num_experts

        self._bnb_gate_up_weights = nn.ParameterList()
        self._bnb_down_weights = nn.ParameterList()

        device = gate_up_proj.device

        for expert_idx in range(num_experts):
            gate_up_weight = gate_up_proj[expert_idx].detach().clone()  # [2*I, H]
            down_weight = down_proj[expert_idx].detach().clone()  # [H, I]

            # Params4bit quantizes here (on CUDA)
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

        if hasattr(self, "_gate_up_proj_pending") and self._gate_up_proj_pending is not None:
            del self._gate_up_proj_pending
        if hasattr(self, "_down_proj_pending") and self._down_proj_pending is not None:
            del self._down_proj_pending

    def _apply(self, fn, recurse=True):
        """Quantize pending weights when the model is moved to CUDA."""
        if hasattr(self, "_gate_up_proj_pending") and self._gate_up_proj_pending is not None:
            test_tensor = fn(torch.zeros(1))  # probe the target device
            if test_tensor.device.type == 'cuda':
                pending_gate_up = fn(self._gate_up_proj_pending)
                pending_down = fn(self._down_proj_pending)
                self._quantize_and_store(pending_gate_up, pending_down)
                return self  # pending weights consumed

        return super()._apply(fn, recurse)

    def _matmul_4bit(self, x: torch.Tensor, weight: Params4bit) -> torch.Tensor:
        """4-bit matmul via bitsandbytes, following Linear4bit.forward."""
        quant_state = weight.quant_state
        w = weight.t()
        return bnb.matmul_4bit(x, w, bias=None, quant_state=quant_state)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward via bnb.matmul_4bit, only over experts that have routed tokens."""
        # Set compute type on first forward (like Linear4bit)
        if not self.compute_type_is_set:
            self.set_compute_type(hidden_states)
            self.compute_type_is_set = True

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

        # Only process experts that have tokens (much faster than all 128)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts, top_k, num_tokens]
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)

        for expert_idx_t in expert_hit:
            expert_idx = expert_idx_t.item()
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[token_idx]

            gate_up_weight = self._bnb_gate_up_weights[expert_idx]
            down_weight = self._bnb_down_weights[expert_idx]

            gate_up_out = self._matmul_4bit(current_state, gate_up_weight)
            gate, up = gate_up_out.chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = self._matmul_4bit(current_hidden_states, down_weight)

            # Apply routing weights and scatter back
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states
            )

        return final_hidden_states.to(inp_dtype)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Save quantized weights to state dict, following Linear4bit pattern."""
        # Weights not loaded yet (empty ParameterLists): default save
        if len(self._bnb_gate_up_weights) == 0:
            super()._save_to_state_dict(destination, prefix, keep_vars)
            return

        for expert_idx in range(self.num_experts):
            gate_up_4bit = self._bnb_gate_up_weights[expert_idx]
            down_4bit = self._bnb_down_weights[expert_idx]

            gate_up_prefix = f"{prefix}_bnb_gate_up_weights.{expert_idx}."
            down_prefix = f"{prefix}_bnb_down_weights.{expert_idx}."

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
    """Replace MoE expert modules with 4-bit quantized versions before weights load.

    Like transformers' replace_with_bnb_linear: swap modules on meta device,
    then weights are loaded and quantized on-the-fly.

    Returns (model, has_been_replaced).
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

            if getattr(experts, '_is_bnb_4bit', False):
                continue  # already quantized

            gate_up_proj = experts.gate_up_proj
            num_experts = gate_up_proj.shape[0]

            module_class_name = type(module).__name__
            if 'Qwen3VL' in module_class_name or 'qwen3_vl' in type(module).__module__:
                quantized_cls = Qwen3VLMoeExperts4bit
                # Qwen3-VL: transposed grouped_mm format gate_up_proj (E, H, 2*I)
                intermediate_dim = gate_up_proj.shape[2] // 2
                hidden_dim = gate_up_proj.shape[1]
            else:
                quantized_cls = Qwen3MoeExperts4bit
                # Qwen3-MoE: F.linear format (E, 2*I, H)
                intermediate_dim = gate_up_proj.shape[1] // 2
                hidden_dim = gate_up_proj.shape[2]

            # Create quantized version on meta device (like replace_with_bnb_linear)
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

            if hasattr(experts, 'act_fn'):
                new_experts.act_fn = experts.act_fn

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
    """Post-loading quantization (more memory, but works with any model).

    Prefer replace_with_bnb_moe_experts() for lower-peak on-the-fly quantization.
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
    """Post-loading hook to quantize MoE experts to 4-bit.

    Uses more peak memory; prefer replace_with_bnb_moe_experts() before loading.
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
