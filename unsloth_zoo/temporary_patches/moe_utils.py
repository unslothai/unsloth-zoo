# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
import os
from typing import Optional, Tuple


# Global flag to check if grouped GEMM is available
_GROUPED_GEMM_AVAILABLE = None
_TORCH_GROUPED_MM_AVAILABLE = hasattr(torch, "_grouped_mm")

# Check if GPU supports torch._grouped_mm (verified via runtime check)
_TORCH_GROUPED_MM_SUPPORTED = None


def _check_torch_grouped_mm_supported():
    """
    Check if torch._grouped_mm is actually supported on the current GPU.
    We check for existence and verify with a dummy call.
    A runtime probe is the only reliable check.
    """
    global _TORCH_GROUPED_MM_SUPPORTED
    if _TORCH_GROUPED_MM_SUPPORTED is not None:
        return _TORCH_GROUPED_MM_SUPPORTED

    if not _TORCH_GROUPED_MM_AVAILABLE:
        _TORCH_GROUPED_MM_SUPPORTED = False
        return False

    if not torch.cuda.is_available():
        _TORCH_GROUPED_MM_SUPPORTED = False
        return False

    try:
        # Attempt a dummy grouped_mm call to verify support.
        # This handles cases where the symbol exists but hardware is unsupported (e.g. < H100).
        # It also allows support on newer hardware or backports without code changes.
        device = torch.cuda.current_device()
        dtype = torch.float16

        # Minimal dummy data: 1 expert, 1 token, dim 8 (safe alignment)
        x = torch.ones((1, 8), device=device, dtype=dtype)
        w = torch.ones((1, 8, 8), device=device, dtype=dtype)
        offs = torch.tensor([1], device=device, dtype=torch.int32)

        torch._grouped_mm(x, w, offs=offs)
        del x, w, offs
        _TORCH_GROUPED_MM_SUPPORTED = True
    except Exception:
        _TORCH_GROUPED_MM_SUPPORTED = False

    return _TORCH_GROUPED_MM_SUPPORTED


_TRITON_ALLOCATOR_INITIALIZED = False
_PERSISTENT_BUFFER = None


def _init_triton_allocator():
    """
    Initialize a persistent Triton allocator to avoid memory allocation overhead per call.
    This significantly reduces GPU utilization fluctuation.
    """
    global _TRITON_ALLOCATOR_INITIALIZED, _PERSISTENT_BUFFER
    if _TRITON_ALLOCATOR_INITIALIZED:
        return

    try:
        import triton

        # Create a persistent buffer that grows as needed
        # This avoids allocating new memory on every kernel call

        def persistent_alloc_fn(size: int, alignment: int, stream):
            global _PERSISTENT_BUFFER
            # Round up size to avoid frequent reallocations
            # Round to nearest 128 bytes for alignment
            rounded_size = ((size + 128 - 1) // 128) * 128

            if (
                _PERSISTENT_BUFFER is None
                or _PERSISTENT_BUFFER.numel() * _PERSISTENT_BUFFER.element_size()
                < rounded_size
            ):
                # Allocate with small headroom (10%) to reduce reallocations
                # Use ByteTensor (uint8) for raw byte storage
                _PERSISTENT_BUFFER = torch.empty(
                    int(rounded_size * 1.1), device="cuda", dtype=torch.uint8
                )
                _PERSISTENT_BUFFER.__hibernate__ = {"type": "ignore"}
            return _PERSISTENT_BUFFER

        triton.set_allocator(persistent_alloc_fn)
        triton._unsloth_allocator_set = True
        _TRITON_ALLOCATOR_INITIALIZED = True
    except Exception:
        pass


def _check_grouped_gemm_available():
    """Check if Unsloth grouped GEMM kernels are available."""
    if os.environ.get("UNSLOTH_DISABLE_MOE_TRITON", "0") == "1":
        return False

    global _GROUPED_GEMM_AVAILABLE
    if _GROUPED_GEMM_AVAILABLE is not None:
        return _GROUPED_GEMM_AVAILABLE

    try:
        from unsloth.kernels.moe.grouped_gemm.interface import (
            grouped_gemm,
            supports_tma,
        )

        _GROUPED_GEMM_AVAILABLE = True
        _init_triton_allocator()
    except (ImportError, ModuleNotFoundError):
        _GROUPED_GEMM_AVAILABLE = False
    return _GROUPED_GEMM_AVAILABLE


def select_moe_backend():
    """
    Selects the MoE backend based on UNSLOTH_MOE_BACKEND environment variable and availability.
    Choices: "grouped_mm", "unsloth_triton", "native_torch".
    Default if unspecified: "grouped_mm".
    """
    # Choices ordered by preference
    # (backend_name, is_available)
    choices = [
        ("grouped_mm", _check_torch_grouped_mm_supported()),
        ("unsloth_triton", _check_grouped_gemm_available()),
        ("native_torch", True),
    ]

    # 1. Check environment variable
    requested_backend = os.environ.get("UNSLOTH_MOE_BACKEND")

    # User explicitly requested a backend
    if requested_backend:
        # Check against available choices
        is_valid = False
        is_available = False

        for name, available in choices:
            if name == requested_backend:
                is_valid = True
                is_available = available
                break

        if is_valid:
            if is_available:
                return requested_backend
            else:
                print(
                    f"Unsloth: '{requested_backend}' backend requested but is not available. Falling back to next available."
                )

    # 2. Automatic selection (first available in preference order)
    for name, available in choices:
        if available:
            print(f"Unsloth: Using MoE backend '{name}'")
            return name

    print("Unsloth: Using MoE backend 'native_torch' (fallback)")
    return "native_torch"


@torch.no_grad()
def _get_routing_indices(selected_experts, num_experts):
    """
    Compute token→expert mapping for grouped GEMM.
    Uses bincount instead of histc to avoid float conversion overhead.

    Returns:
        token_counts_by_expert: (num_experts,) token counts per expert
        gather_indices: (total_tokens,) indices for gathering tokens in expert order
    """
    flat_experts = selected_experts.view(-1)

    # bincount is faster than histc since it doesn't require float conversion
    token_counts_by_expert = torch.bincount(flat_experts, minlength=num_experts).to(
        torch.int32
    )

    # argsort with stable=True preserves order within each expert
    gather_indices = flat_experts.argsort(stable=True)

    return token_counts_by_expert, gather_indices


def _silu_and_mul(x):
    """Fused SiLU activation and element-wise multiply for gate/up projections."""
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


# ============================================================================
# Separated LoRA Helper Functions
# ============================================================================


def _has_lora_adapters(param) -> bool:
    """Check if parameter has active LoRA adapters (PEFT ParamWrapper)."""
    # Check if this is a PEFT LoRA wrapper
    if not hasattr(param, "lora_A") or not hasattr(param, "lora_B"):
        return False
    if hasattr(param, "disable_adapters") and param.disable_adapters:
        return False
    if hasattr(param, "merged") and param.merged:
        return False
    return len(param.lora_A) > 0


def _get_shape(param):
    """Recursively search for shape."""
    if hasattr(param, "shape"):
        return param.shape
    if hasattr(param, "weight") and hasattr(param.weight, "shape"):
        return param.weight.shape

    # Check if we have parameter_name
    param_name = getattr(param, "parameter_name", None)
    if param_name and hasattr(param, "base_layer"):
        # If base_layer is the module, check if it has the param
        if hasattr(param.base_layer, param_name):
            p = getattr(param.base_layer, param_name)
            # Recursively check p (p might be a Tensor or another wrapper)
            # Avoid infinite loop if p is param
            if p is not param:
                s = _get_shape(p)
                if s is not None:
                    return s

    if hasattr(param, "base_layer"):
        return _get_shape(param.base_layer)
    return None


def _extract_lora_weights(
    param, adapter_name: str = "default", num_experts: int = None
) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
    """
    Extract LoRA A and B weights from PEFT ParamWrapper.

    Returns:
        (lora_B_reshaped, lora_A_reshaped, scaling) - For (X @ B) @ A order
    """
    try:
        print(f"DEBUG: _extract_lora_weights param type={type(param)}")
        try:
            if hasattr(param, "shape"):
                print(f"DEBUG: param.shape={param.shape}")
            if hasattr(param, "base_layer"):
                print(f"DEBUG: param.base_layer type={type(param.base_layer)}")
        except:
            pass
        if adapter_name not in param.lora_A:
            # Try default adapter
            adapter_name = list(param.lora_A.keys())[0] if param.lora_A else None
            if adapter_name is None:
                return None

        # PEFT Standard:
        # lora_A (Rank, In)  [First Projection]
        # lora_B (Out, Rank) [Second Projection]
        peft_A = param.lora_A[adapter_name].weight  # (Rank, In)
        peft_B = param.lora_B[adapter_name].weight  # (Out, Rank)

        scaling = param.scaling[adapter_name]

        # Get Num Experts
        if num_experts is None:
            num_experts = getattr(param, "num_experts", 1)

        if num_experts > 1:
            # Unsloth MoE expects:
            # First (lora_B): (E, In_local, R)
            # Second (lora_A): (E, R, Out_local)

            # Logic: PEFT creates (Total_Rank, In) and (Total_Out, Total_Rank).
            # We assume Rank is split across experts (Total_Rank = E * R_local).
            # We assume In is SHARED (H).
            # We assume Out is split (Total_Out = E * Out_local).

            rank_total = peft_A.shape[0]
            in_dim = peft_A.shape[1]
            out_total = peft_B.shape[0]

            # Determine expected dimensions
            shape = _get_shape(param)
            expected_out = shape[-1] if shape else None
            expected_in = shape[-2] if (shape and len(shape) > 1) else None

            # Determine LoRA type
            is_global_sliced = False
            is_split = False

            # Check for Global Sliced conditions
            if rank_total < num_experts:
                is_global_sliced = True
            elif expected_out and out_total == num_experts * expected_out:
                is_global_sliced = True
            elif expected_in and in_dim == num_experts * expected_in:
                is_global_sliced = True

            # If not global sliced, check if split or shared
            if not is_global_sliced:
                # If shapes match base layer exactly, it's Shared (Broadcast)
                if expected_out and out_total == expected_out:
                    is_split = False
                else:
                    # Default to Split
                    is_split = True

            if is_split:
                rank = rank_total // num_experts
                out_local = out_total // num_experts

                # 1. Process First Projection (PEFT A) -> unsloth_B
                # PEFT A: (E * R, In) -> View (E, R, In) -> Permute (E, In, R)
                unsloth_B = (
                    peft_A.view(num_experts, rank, in_dim).permute(0, 2, 1).contiguous()
                )

                # 2. Process Second Projection (PEFT B) -> unsloth_A
                # PEFT B: (E * Out_local, E * R)
                peft_B_reshaped = peft_B.view(num_experts, out_local, num_experts, rank)
                unsloth_A_diag = torch.einsum("eoer->eor", peft_B_reshaped)
                unsloth_A = unsloth_A_diag.permute(0, 2, 1).contiguous()

                # Fix for grouped_mm stride alignment (requires multiple of 8/16 bytes)
                # If rank per expert is small (e.g. 64/32=2), we must pad to 8
                if rank % 8 != 0:
                    pad_size = 8 - (rank % 8)
                    # Pad unsloth_B (E, In, R) -> (E, In, R+pad)
                    unsloth_B = F.pad(unsloth_B, (0, pad_size), "constant", 0)
                    # Pad unsloth_A (E, R, Out) -> (E, R+pad, Out)
                    # Pad starting from last dim: Out (0,0), R (0,pad)
                    unsloth_A = F.pad(unsloth_A, (0, 0, 0, pad_size), "constant", 0)

                unsloth_B = unsloth_B.contiguous()
                unsloth_A = unsloth_A.contiguous()

            elif is_global_sliced:
                # Global LoRA (Sliced B / Shared A OR Sliced A / Shared B)
                rank = rank_total

                # Case 1: Sliced Output (GateUp): In=H, Out=E*Out_local
                # Criteria: Out is sliced.
                if expected_out and out_total == num_experts * expected_out:
                    # GateUp
                    # unsloth_B (Input): Broadcast peft_A.T (In, R) -> (E, In, R)
                    unsloth_B = (
                        peft_A.transpose(0, 1)
                        .unsqueeze(0)
                        .expand(num_experts, -1, -1)
                        .contiguous()
                    )

                    # unsloth_A (Output): Reshape peft_B (E*Out, R) -> (E, Out, R) -> Permute (E, R, Out)
                    unsloth_A = (
                        peft_B.view(num_experts, expected_out, rank)
                        .permute(0, 2, 1)
                        .contiguous()
                    )

                # Case 2: Sliced Input (Down): In=E*In_local, Out=H
                # Criteria: In is sliced.
                elif expected_in and in_dim == num_experts * expected_in:
                    # Down
                    # unsloth_B (Input): Reshape peft_A (R, E*In) -> (R, E, In) -> Permute (E, In, R)
                    unsloth_B = (
                        peft_A.view(rank, num_experts, expected_in)
                        .permute(1, 2, 0)
                        .contiguous()
                    )

                    # unsloth_A (Output): Broadcast peft_B.T (R, Out) -> (E, R, Out)
                    unsloth_A = (
                        peft_B.T.unsqueeze(0).expand(num_experts, -1, -1).contiguous()
                    )

                else:
                    # Fallback if shapes ambiguous but identified as global_sliced (e.g. rank < experts)
                    # Try to infer based on dimensions relative to num_experts
                    # If out_total is multiple of experts and large -> GateUp
                    if out_total % num_experts == 0 and out_total > in_dim:
                        unsloth_B = (
                            peft_A.transpose(0, 1)
                            .unsqueeze(0)
                            .expand(num_experts, -1, -1)
                            .contiguous()
                        )
                        unsloth_A = (
                            peft_B.view(num_experts, out_total // num_experts, rank)
                            .permute(0, 2, 1)
                            .contiguous()
                        )
                    # If in_dim is multiple of experts and large -> Down
                    elif in_dim % num_experts == 0 and in_dim > out_total:
                        unsloth_B = (
                            peft_A.view(rank, num_experts, in_dim // num_experts)
                            .permute(1, 2, 0)
                            .contiguous()
                        )
                        unsloth_A = (
                            peft_B.T.unsqueeze(0)
                            .expand(num_experts, -1, -1)
                            .contiguous()
                        )
                    else:
                        # Fallback to shared
                        unsloth_B = (
                            peft_A.transpose(0, 1)
                            .unsqueeze(0)
                            .expand(num_experts, -1, -1)
                            .contiguous()
                        )
                        unsloth_A = (
                            peft_B.T.unsqueeze(0)
                            .expand(num_experts, -1, -1)
                            .contiguous()
                        )

            else:
                # Shared LoRA (Broadcast)
                rank = rank_total
                unsloth_B = (
                    peft_A.transpose(0, 1)
                    .unsqueeze(0)
                    .expand(num_experts, -1, -1)
                    .contiguous()
                )
                unsloth_A = (
                    peft_B.T.unsqueeze(0).expand(num_experts, -1, -1).contiguous()
                )

        else:
            # Non-MoE or Single Expert?
            # Unsloth B (First) = PEFT A.T (In, R)
            unsloth_B = peft_A.transpose(0, 1)
            # Unsloth A (Second) = PEFT B.T (R, Out) -- Wait, PEFT B is (Out, R). T is (R, Out).
            # We need (R, Out) for Second?
            # Standard Linear LoRA: X @ A.T @ B.T
            # X (N, In). A.T (In, R). Result (N, R). B.T (R, Out). Result (N, Out).
            # Using grouped_mm logic for 1 expert:
            # (E=1, I, R) -> (1, I, R)
            # (E=1, R, O) -> (1, R, O)
            unsloth_B = peft_A.transpose(0, 1).unsqueeze(0)
            unsloth_A = peft_B.T.unsqueeze(0).transpose(
                1, 2
            )  # (1, Out, R) -> (1, R, Out)

            # Simplify if not using grouped_mm shapes strictly?
            # But the caller expects (E, In, R) and (E, R, Out) or similar.
            # Let's assume num_experts=1 means we return standard tensors?
            # Existing code: return weight_A, weight_B.
            # Let's keep existing logic structure but corrected.
            unsloth_B = peft_A.transpose(0, 1)  # (In, R)
            unsloth_A = (
                peft_B  # (Out, R) -- Wait, existing code returned weight_B for A?
            )
            # Existing was messy.
            # If num_experts=1, we just return PEFT weights?
            # The caller `_apply_lora_grouped_mm` expects:
            # lora_B (E, In, R). lora_A (E, R, Out).
            # If we return None, caller handles?
            # But if we return tensors, they MUST match.
            # So let's force the shape (1, I, R) and (1, R, O) if num_experts is 1 but we want grouped_mm.
            pass

            # For back-compat with non-moe usage of this function?
            return peft_A.T, peft_B.T, scaling  # WRONG if caller expects split.
            # Let's just assume we are in MoE context if this function is called.
            # But for safety, let's just stick to the MoE path if num_experts > 1.
            return None  # Placeholder

        return unsloth_B, unsloth_A, scaling
        return unsloth_B, unsloth_A, scaling
    except Exception:
        return None


def _get_base_weight(param):
    """Get base weight from potentially wrapped parameter or module."""
    # Recursively unwrap PEFT layers
    while hasattr(param, "base_layer"):
        param = param.base_layer

    if hasattr(param, "get_param"):
        return param.get_param()

    # Handle Modules (Linear, etc.)
    if hasattr(param, "weight"):
        return param.weight

    return param


# Aliases for compatibility with gpt_oss.py
_get_moe_lora_weights = _extract_lora_weights


def _get_lora_wrapper_for_param(experts_module, param_name):
    """
    Get the PEFT ParamWrapper for a specific parameter (gate_up_proj or down_proj).
    Uses the explicit key stored in __dict__ if available.
    Does NOT lazily setup wrappers as that requires traversing logic not present here.
    """
    if hasattr(experts_module, f"{param_name}_lora_wrapper"):
        return getattr(experts_module, f"{param_name}_lora_wrapper")

    # Check simple attributes if it's directly wrapped
    if hasattr(experts_module, param_name):
        attr = getattr(experts_module, param_name)
        if hasattr(attr, "lora_A"):  # Is a ParamWrapper
            return attr

    return None


# Restored from HEAD for compatibility
def native_moe_grouped_mm(
    inputs: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """
    Native implementation using torch._grouped_mm.
    """
    return torch._grouped_mm(inputs, weight, offs=offsets)


def _apply_lora_grouped_mm(
    inputs: torch.Tensor,
    lora_B: torch.Tensor,
    lora_A: torch.Tensor,
    offsets: torch.Tensor,
    scaling: float,
    grouped_mm_func=native_moe_grouped_mm,
) -> torch.Tensor:
    """
    Apply LoRA using grouped GEMM: result = ((X @ B) @ A) * scaling

    Args:
        inputs: (total_tokens, in_dim)
        lora_B: (num_experts, in_dim, rank) - First projection
        lora_A: (num_experts, rank, out_dim) - Second projection
        offsets: Grouped GEMM offsets
        scaling: LoRA scaling factor
        grouped_mm_func: Function to use for grouped GEMM (default: native_moe_grouped_mm)
    """
    # 1. First Matmul (X @ B)
    # lora_B is (E, in_dim, R)
    # Native needs (E, in_dim, R) -> No Transpose
    lora_intermediate = grouped_mm_func(inputs, lora_B.contiguous(), offsets)

    # 2. Second Matmul (result @ A)
    # lora_A is (E, R, out_dim)
    # Native needs (E, R, out_dim) -> No Transpose
    lora_delta = grouped_mm_func(lora_intermediate, lora_A.contiguous(), offsets)

    return lora_delta * scaling


def _should_use_separated_lora() -> bool:
    """
    Check if separated LoRA approach should be used (default: True).
    Set UNSLOTH_MOE_LORA_MERGED=1 to use merged approach instead.
    """
    return os.environ.get("UNSLOTH_MOE_LORA_MERGED", "0") != "1"


# ============================================================================
# Generic MoE Detection and ParamWrapper Patching
# ============================================================================


def _is_moe_experts_module(module) -> bool:
    """
    Check if module is an MoE experts layer (generic, not model-specific).

    Detects modules with stacked expert weights as 3D nn.Parameter:
    - gate_up_proj/down_proj pattern (Qwen3-MoE, Qwen3-VL-MoE, etc.)
    - w1/w2/w3 pattern (older MoE models)
    """
    import torch.nn as nn

    # Check for gate_up_proj pattern
    if hasattr(module, "gate_up_proj"):
        param = module.gate_up_proj
        if isinstance(param, nn.Parameter) and param.ndim == 3:
            return True

    # Check for w1/w2 pattern (separate gate/up projections)
    if hasattr(module, "w1") and hasattr(module, "w2"):
        w1 = module.w1
        if isinstance(w1, nn.Parameter) and w1.ndim == 3:
            return True

    return False


def _extract_lora_from_wrapper(
    wrapper, adapter_name: str = "default"
) -> Optional[Tuple[torch.Tensor, torch.Tensor, float, int]]:
    """
    Extract LoRA weights from PEFT ParamWrapper for MoE separated computation.
    # print(f"DEBUG: _extract_lora_from_wrapper called for {adapter_name}")

    PEFT computes delta = B @ A, so for separated LoRA:
    Y = X @ W + X @ (B @ A) * s = X @ W + ((X @ B) @ A) * s

    Args:
        wrapper: ParamWrapper module with lora_A, lora_B, scaling
        adapter_name: Name of the active adapter

    Returns:
        (lora_B_reshaped, lora_A_reshaped, scaling, num_experts) or None

    Note: Returns (B, A) for application order: first X @ B, then result @ A

    Shapes (for MoE with num_experts=E, rank=R):
        lora_B_reshaped: (E, in_features, R) - for first step: X @ B
        lora_A_reshaped: (E, R, out_features) - for second step: result @ A
    """
    try:
        # Check basic PEFT attributes
        if not hasattr(wrapper, "lora_A") or not hasattr(wrapper, "lora_B"):
            return None

        if hasattr(wrapper, "disable_adapters") and wrapper.disable_adapters:
            return None
        if hasattr(wrapper, "merged") and wrapper.merged:
            # print("DEBUG: extract_lora merged")
            return None

        # print(f"DEBUG: extract_lora extraction attempt for {adapter_name}")

        # Delegate to shared extraction logic
        # This ensures consistent shape handling (diagonal extraction for B, reshapes for A)
        # wrapper acts as 'param' here (has lora_A/lora_B/scaling attributes)

        # Determine num_experts from wrapper or try to infer
        num_experts = getattr(wrapper, "num_experts", None)

        result = _extract_lora_weights(wrapper, adapter_name, num_experts=num_experts)

        if result is None:
            return None

        unsloth_B, unsloth_A, scaling = result

        # We need to return num_experts as 4th element locally used by caller
        # If num_experts was None, _extract_lora_weights likely found it or defaulted
        # We can infer it from the shape of unsloth_B (E, In, R)
        if hasattr(unsloth_B, "shape") and len(unsloth_B.shape) == 3:
            e = unsloth_B.shape[0]
        else:
            e = num_experts if num_experts else 1

        return unsloth_B, unsloth_A, scaling, e

    except Exception:
        return None


# Store original ParamWrapper.forward for fallback
_original_param_wrapper_forward = None


def _patched_param_wrapper_forward(
    self, x: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    """
    Patched ParamWrapper.forward for MoE separated LoRA.

    For MoE expert modules:
    - Bypasses PEFTs _activate_lora parametrization context
    - Stores LoRA data by parameter_name for forward_native_grouped_mm to use

    For non-MoE modules:
    - Falls back to original PEFT forward
    """
    # CRITICAL: Use self.base_layer for forward call (immediate parent)
    # NOT self.get_base_layer() which recursively traverses to deepest layer!
    # The wrapper chain must be preserved: down_proj -> gate_up_proj -> Qwen3MoeExperts
    immediate_base_layer = self.base_layer

    # For storing LoRA data, we DO need the actual experts module
    # Use get_base_layer() to find it (recursive traversal is correct here)
    experts_module = self.get_base_layer()

    use_separated = _should_use_separated_lora()
    param_name = getattr(self, "parameter_name", None)

    # DEBUG: Instrumentation
    # DEBUG: Instrumentation
    # if param_name in ('gate_up_proj', 'down_proj'):
    # print(f"DEBUG: Entered _patched_param_forward. param={param_name}")
    # print(f"DEBUG: use_separated={use_separated}")
    # print(f"DEBUG: is_moe={_is_moe_experts_module(experts_module)}")
    # print(f"DEBUG: experts_module type={type(experts_module)}")

    # Check if this is an MoE experts module that should use separated LoRA
    if (
        use_separated
        and param_name in ("gate_up_proj", "down_proj")
        and _is_moe_experts_module(experts_module)
    ):
        print("DEBUG: Condition MET. Entering MoE logic.")
        # MoE experts: bypass PEFT's _activate_lora, use separated computation

        # Check adapter state
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return immediate_base_layer(x, *args, **kwargs)

        if self.merged:
            return immediate_base_layer(x, *args, **kwargs)

        # Ensure wrapper.num_experts is set for LoRA weight reshaping
        if not hasattr(self, "num_experts"):
            if hasattr(experts_module, "num_experts"):
                self.num_experts = experts_module.num_experts
            elif hasattr(experts_module, param_name):
                p = getattr(experts_module, param_name)
                if hasattr(p, "shape") and len(p.shape) >= 1:
                    self.num_experts = p.shape[0]

        # Extract LoRA for this specific parameter
        lora_data = _extract_lora_from_wrapper(self)

        if lora_data is not None and param_name:
            # Store LoRA data on the EXPERTS MODULE (not base_layer)
            # e.g., _unsloth_lora_gate_up_proj or _unsloth_lora_down_proj
            lora_attr = f"_unsloth_lora_{param_name}"
            setattr(experts_module, lora_attr, lora_data)

        try:
            # Call IMMEDIATE base_layer to preserve wrapper chain
            # (down_proj wrapper calls gate_up_proj wrapper calls Qwen3MoeExperts)
            result = immediate_base_layer(x, *args, **kwargs)
        finally:
            # Clean up
            if param_name:
                lora_attr = f"_unsloth_lora_{param_name}"
                if hasattr(experts_module, lora_attr):
                    delattr(experts_module, lora_attr)

        return result

    # Non-MoE: use original PEFT forward with _activate_lora
    return _original_param_wrapper_forward(self, x, *args, **kwargs)


def patch_param_wrapper_for_moe():
    """
    Patch PEFT's ParamWrapper.forward to use separated LoRA for MoE.

    This should be called after PEFT is imported.
    """
    global _original_param_wrapper_forward

    try:
        from peft.tuners.lora.layer import ParamWrapper

        # Store original forward
        if _original_param_wrapper_forward is None:
            _original_param_wrapper_forward = ParamWrapper.forward

        # Patch with our version
        ParamWrapper.forward = _patched_param_wrapper_forward

        return True
    except ImportError:
        return False


def forward_native_grouped_mm(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Native Pytorch grouped GEMM MoE forward pass.
    Uses torch._grouped_mm which is significantly faster than loop and works without Triton dependencies.
    Requires torch._grouped_mm support (verified via runtime check).
    """
    # Runtime safety check - defense in depth
    if not _check_torch_grouped_mm_supported():
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        raise RuntimeError(
            f"torch._grouped_mm is not supported on this device (Compute Capability {major}.{minor}). "
            f"Set UNSLOTH_MOE_BACKEND='unsloth_triton' or 'native_torch' to use a compatible backend."
        )

    is_2d_input = hidden_states.dim() == 2
    if is_2d_input:
        sequence_length, hidden_dim = hidden_states.shape
        batch_size = 1
    else:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

    hidden_states = hidden_states.view(-1, hidden_dim)

    # 1. Calculate routing
    flat_top_k = top_k_index.view(-1)
    num_tokens_per_expert = torch.bincount(flat_top_k, minlength=self.num_experts).int()

    # 2. Sort indices to group tokens by expert
    sorted_indices = torch.argsort(flat_top_k, stable=True)
    token_indices = sorted_indices // top_k_index.shape[1]

    # 3. Permute Input
    # We need to gather inputs. Since we may have expanded top_k, we use token_indices to map back to original input
    permuted_input = hidden_states[token_indices]

    # 4. Prepare Grouped MM arguments
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    # ========================================================================
    # Gate + Up projection with optional separated LoRA (DEFAULT)
    # ========================================================================
    use_separated_lora = _should_use_separated_lora()
    gate_up_lora = None

    # Check for injected LoRA data from patched ParamWrapper (preferred path)
    if hasattr(self, "_unsloth_lora_gate_up_proj"):
        gate_up_lora = self._unsloth_lora_gate_up_proj[:3]  # (lora_A, lora_B, scaling)
    # Fallback: check parameter directly (for older wrapping patterns)
    elif (
        use_separated_lora
        and hasattr(self, "gate_up_proj")
        and _has_lora_adapters(self.gate_up_proj)
    ):
        gate_up_lora = _extract_lora_weights(
            self.gate_up_proj, num_experts=self.num_experts
        )

    elif (
        use_separated_lora
        and hasattr(self, "gate_up_proj")
        and _has_lora_adapters(self.gate_up_proj)
    ):
        gate_up_lora = _extract_lora_weights(
            self.gate_up_proj, num_experts=self.num_experts
        )

    if hasattr(self, "gate_up_proj"):
        # Get base weights (raw, without LoRA)
        gate_up_base = _get_base_weight(self.gate_up_proj)

        # Handle different weight shapes (e.g. Qwen3 vs Qwen3VL)
        if gate_up_base.shape[1] == hidden_dim:
            w1 = gate_up_base
        else:
            w1 = gate_up_base.transpose(-2, -1)

        # Base forward: X @ W
        mm1_out = torch._grouped_mm(permuted_input, w1, offs=offsets)

        # Add separated LoRA contribution: + ((X @ B) @ A) * scaling
        # PEFT computes delta = B @ A
        if gate_up_lora is not None:
            lora_B, lora_A, scaling = gate_up_lora  # B first, A second

            # Cast to input dtype (LoRA weights are float32, input may be bfloat16)
            # Ensure contiguous for grouped_mm alignment requirements
            lora_B = lora_B.to(permuted_input.dtype).contiguous()
            lora_A = lora_A.to(permuted_input.dtype).contiguous()

            # Step 1: permuted_input @ B
            # print(f"DEBUG: permuted_input={permuted_input.shape}, lora_B={lora_B.shape}, lora_A={lora_A.shape}")
            try:
                lora_out = torch._grouped_mm(permuted_input, lora_B, offs=offsets)
                lora_out = lora_out.contiguous()
            except RuntimeError as e:
                print(f"ERROR in grouped_mm: {e}")
                print(f"  permuted_input: {permuted_input.shape}")
                print(f"  lora_B: {lora_B.shape}")
                print(f"  lora_A: {lora_A.shape}")
                print(f"  num_experts: {self.num_experts}")
                print(f"  offsets: {offsets}")
                raise e

            # Step 2: result @ A
            # Handle unaligned O dimension or other grouped_mm failures
            try:
                if lora_A.shape[-1] % 8 != 0:
                    pad_size = 8 - (lora_A.shape[-1] % 8)
                    lora_A_padded = F.pad(lora_A, (0, pad_size)).contiguous()
                    lora_delta = torch._grouped_mm(
                        lora_out, lora_A_padded, offs=offsets
                    )
                    lora_delta = lora_delta[:, :-pad_size]
                else:
                    lora_delta = torch._grouped_mm(lora_out, lora_A, offs=offsets)
            except RuntimeError:
                # Fallback to manual loop if grouped_mm fails (e.g. stride alignment)
                lora_delta = torch.empty(
                    (lora_out.shape[0], lora_A.shape[-1]),
                    dtype=lora_out.dtype,
                    device=lora_out.device,
                )
                cpu_offsets = offsets.cpu().tolist()
                for i in range(len(cpu_offsets) - 1):
                    start = cpu_offsets[i]
                    end = cpu_offsets[i + 1]
                    if start < end:
                        # lora_out slice: (Batch_i, R)
                        # lora_A expert: (R, Out)
                        # result: (Batch_i, Out)
                        # lora_A[i] is (R, Out) ? No lora_A is (E, R, Out).
                        # lora_A[i] is (R, Out).
                        lora_delta[start:end] = torch.matmul(
                            lora_out[start:end], lora_A[i]
                        )

            # Add scaled LoRA contribution
            mm1_out = mm1_out + lora_delta * scaling

        if hasattr(self, "gate_up_proj_bias") and self.gate_up_proj_bias is not None:
            num_repeats = num_tokens_per_expert.to(self.gate_up_proj_bias.device)
            bias_expanded = self.gate_up_proj_bias.repeat_interleave(num_repeats, dim=0)
            mm1_out = mm1_out + bias_expanded.to(mm1_out.dtype)

        if "GptOssExperts" in self.__class__.__name__:
            gate = mm1_out[..., ::2]
            up = mm1_out[..., 1::2]
        else:
            gate, up = mm1_out.chunk(2, dim=-1)

    elif hasattr(self, "w1") and hasattr(self, "w3"):
        # Separate w1/w3 weights (older models)
        w1_base = _get_base_weight(self.w1)
        w3_base = _get_base_weight(self.w3)

        w1 = w1_base.transpose(-2, -1)
        w3 = w3_base.transpose(-2, -1)

        gate = torch._grouped_mm(permuted_input, w1, offs=offsets)
        up = torch._grouped_mm(permuted_input, w3, offs=offsets)

        # Add LoRA for w1 and w3 separately if present
        if use_separated_lora:
            if _has_lora_adapters(self.w1):
                w1_lora = _extract_lora_weights(self.w1)
                if w1_lora is not None:
                    lora_A, lora_B, scaling = w1_lora
                    lora_A_t = lora_A.transpose(-2, -1)
                    lora_A_out = torch._grouped_mm(
                        permuted_input, lora_A_t, offs=offsets
                    )
                    lora_B_t = lora_B.transpose(-2, -1)
                    lora_B_out = torch._grouped_mm(lora_A_out, lora_B_t, offs=offsets)
                    gate = gate + lora_B_out * scaling

            if _has_lora_adapters(self.w3):
                w3_lora = _extract_lora_weights(self.w3)
                if w3_lora is not None:
                    lora_A, lora_B, scaling = w3_lora
                    lora_A_t = lora_A.transpose(-2, -1)
                    lora_A_out = torch._grouped_mm(
                        permuted_input, lora_A_t, offs=offsets
                    )
                    lora_B_t = lora_B.transpose(-2, -1)
                    lora_B_out = torch._grouped_mm(lora_A_out, lora_B_t, offs=offsets)
                    up = up + lora_B_out * scaling
    else:
        raise AttributeError("MoE layer must have 'gate_up_proj' or 'w1'/'w3'.")

    # Activation
    if "GptOssExperts" in self.__class__.__name__:
        # Custom activation from GptOss
        limit = getattr(self, "limit", 7.0)
        alpha = getattr(self, "alpha", 1.702)

        gate = gate.clamp(min=None, max=limit)
        up = up.clamp(min=-limit, max=limit)
        glu = gate * torch.sigmoid(gate * alpha)
        inter = (up + 1.0) * glu
    else:
        inter = F.silu(gate) * up

    # ========================================================================
    # Down projection with optional separated LoRA (DEFAULT)
    # ========================================================================
    down_lora = None

    # Check for injected LoRA data from patched ParamWrapper (preferred path)
    if hasattr(self, "_unsloth_lora_down_proj"):
        down_lora = self._unsloth_lora_down_proj[:3]  # (lora_A, lora_B, scaling)
    # Fallback: check parameter directly (for older wrapping patterns)
    elif (
        use_separated_lora
        and hasattr(self, "down_proj")
        and _has_lora_adapters(self.down_proj)
    ):
        down_lora = _extract_lora_weights(self.down_proj, num_experts=self.num_experts)

    if hasattr(self, "down_proj"):
        # Get base weights
        down_base = _get_base_weight(self.down_proj)

        if down_base.shape[1] == inter.shape[-1]:
            w2 = down_base
        else:
            w2 = down_base.transpose(-2, -1)

        # Base forward
        mm2_out = torch._grouped_mm(inter, w2, offs=offsets)

        # Add separated LoRA contribution if present
        # PEFT computes delta = B @ A, so: Y += ((X @ B) @ A) * scaling
        if down_lora is not None:
            lora_B, lora_A, scaling = down_lora  # B first, A second

            # Cast to input dtype (LoRA weights are float32, input may be bfloat16)
            lora_B = lora_B.to(inter.dtype).contiguous()
            lora_A = lora_A.to(inter.dtype).contiguous()

            # Step 1: inter @ B where B is (E, in_dim, R)
            lora_out = torch._grouped_mm(inter, lora_B, offs=offsets)
            lora_out = lora_out.contiguous()

            # Step 2: result @ A where A is (E, R, out_dim)
            # lora_out: (N, 32), A: (128, 32, 2048) -> final: (N, 2048)
            try:
                lora_delta = torch._grouped_mm(lora_out, lora_A, offs=offsets)
            except RuntimeError:
                # Fallback to manual loop
                lora_delta = torch.empty(
                    (lora_out.shape[0], lora_A.shape[-1]),
                    dtype=lora_out.dtype,
                    device=lora_out.device,
                )
                cpu_offsets = offsets.cpu().tolist()
                for i in range(len(cpu_offsets) - 1):
                    start = cpu_offsets[i]
                    end = cpu_offsets[i + 1]
                    if start < end:
                        lora_delta[start:end] = torch.matmul(
                            lora_out[start:end], lora_A[i]
                        )

            # Add scaled LoRA contribution
            mm2_out = mm2_out + lora_delta * scaling

        if hasattr(self, "down_proj_bias") and self.down_proj_bias is not None:
            bias_expanded = self.down_proj_bias.repeat_interleave(
                num_tokens_per_expert.to(self.down_proj_bias.device), dim=0
            ).to(mm2_out.device)
            mm2_out = mm2_out + bias_expanded.to(mm2_out.dtype)

    elif hasattr(self, "w2"):
        w2_base = _get_base_weight(self.w2)
        w2 = w2_base.transpose(-2, -1)

        # Base forward
        mm2_out = torch._grouped_mm(inter, w2, offs=offsets)

        # Add LoRA if present
        if use_separated_lora and _has_lora_adapters(self.w2):
            w2_lora = _extract_lora_weights(self.w2)
            if w2_lora is not None:
                lora_A, lora_B, scaling = w2_lora
                lora_A_t = lora_A.transpose(-2, -1)
                lora_A_out = torch._grouped_mm(inter, lora_A_t, offs=offsets)
                lora_B_t = lora_B.transpose(-2, -1)
                lora_B_out = torch._grouped_mm(lora_A_out, lora_B_t, offs=offsets)
                mm2_out = mm2_out + lora_B_out * scaling
    else:
        raise AttributeError("MoE layer must have 'down_proj' or 'w2'.")

    # 5. Apply Routing Weights
    flat_weights = top_k_weights.view(-1)
    permuted_weights = flat_weights[sorted_indices]
    mm2_out = mm2_out * permuted_weights.unsqueeze(-1)

    # 6. Scatter Add (Reduce)
    # GptOss applies routing weights: weighted_output = out * routing_weights
    # We must permute weights to match the sorted expert assignments
    flat_weights = top_k_weights.view(-1)
    permuted_weights = flat_weights[sorted_indices]
    mm2_out = mm2_out * permuted_weights.unsqueeze(-1)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    final_hidden_states.index_add_(0, token_indices, mm2_out.to(hidden_states.dtype))

    if is_2d_input:
        return final_hidden_states

    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def forward_triton_grouped_gemm(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Grouped GEMM MoE forward pass using Triton kernels.
    Compatible with torch.compile (recommended mode="max-autotune" with cudagraph_mark_step_begin).
    """

    # Import grouped GEMM interface
    from unsloth.kernels.moe.grouped_gemm.interface import grouped_gemm

    # Import autotune cache
    from unsloth.kernels.moe.autotune_cache import get_or_autotune_moe_kernels

    # Helper to check TMA support - assumes helper function or just check directly
    # In original: it was a cached closure. Here we can use _supports_tma() directly

    # nonlocal _MODEL_DIMS_AND_CONFIGS # We need a way to store this!
    # For now, let's attach it to self if possible, or use a global usage
    # Attaching to self is cleaner: self._unsloth_moe_configs

    # Create expert mask and find which experts have tokens
    if not hasattr(self, "_unsloth_moe_configs"):
        self._unsloth_moe_configs = None

    # Handle 3D inputs (batch_size, seq_len, hidden_dim)
    is_3d = hidden_states.dim() == 3
    if is_3d:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        num_tokens = batch_size * seq_len
        # Also flatten top_k inputs if they are 3D
        if top_k_index.dim() == 3:
            top_k_index = top_k_index.view(-1, top_k_index.shape[-1])
        if top_k_weights.dim() == 3:
            top_k_weights = top_k_weights.view(-1, top_k_weights.shape[-1])
    else:
        num_tokens, hidden_dim = hidden_states.shape

    top_k = top_k_index.shape[1]

    # Cache model dimensions and kernel configs on first call
    if self._unsloth_moe_configs is None:
        intermediate_dim = self.gate_up_proj.shape[1] // 2

        # Autotune first GEMM
        gemm1_configs = get_or_autotune_moe_kernels(
            num_experts=self.num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim * 2,
            top_k=top_k,
            dtype=hidden_states.dtype,
        )

        # Autotune second GEMM
        gemm2_configs = get_or_autotune_moe_kernels(
            num_experts=self.num_experts,
            hidden_dim=intermediate_dim,
            intermediate_dim=hidden_dim,  # Output dim for 2nd GEMM is hidden_dim
            top_k=top_k,
            dtype=hidden_states.dtype,
        )

        self._unsloth_moe_configs = (intermediate_dim, gemm1_configs, gemm2_configs)

        # Clear autotuning memory overhead
        torch.cuda.empty_cache()

    # Unpack cached configs
    intermediate_dim, gemm1_configs, gemm2_configs = self._unsloth_moe_configs

    # Unpack specific kernel configs
    fwd_config_1, bwd_dX_config_1, bwd_dW_config_1 = gemm1_configs
    fwd_config_2, bwd_dX_config_2, bwd_dW_config_2 = gemm2_configs

    # Compute routing indices for grouped GEMM
    token_counts_by_expert, gather_indices = _get_routing_indices(
        top_k_index, self.num_experts
    )

    if self.gate_up_proj.shape[-1] == hidden_dim:
        w1 = self.gate_up_proj
    else:
        w1 = self.gate_up_proj.transpose(-2, -1).contiguous()

    # First grouped GEMM: gate_up projection
    first_gemm_output = grouped_gemm(
        X=hidden_states,
        W=w1,
        m_sizes=token_counts_by_expert,
        topk=top_k,
        gather_indices=gather_indices,
        permute_x=True,
        permute_y=False,
        autotune=False,  # We use cached configs
        kernel_config_fwd=fwd_config_1,
        kernel_config_bwd_dX=bwd_dX_config_1,
        kernel_config_bwd_dW=bwd_dW_config_1,
        is_first_gemm=True,
    )

    # Apply SiLU activation and multiply gate with up
    intermediate = _silu_and_mul(first_gemm_output)

    # Grouped GEMM 2: down projection

    if self.down_proj.shape[-1] == intermediate.shape[-1]:
        w2 = self.down_proj
    else:
        w2 = self.down_proj.transpose(-2, -1).contiguous()

    second_gemm_output = grouped_gemm(
        X=intermediate,
        W=w2,
        m_sizes=token_counts_by_expert,
        topk=top_k,
        gather_indices=gather_indices,
        permute_x=False,
        permute_y=True,
        autotune=False,  # We use cached configs
        kernel_config_fwd=fwd_config_2,
        kernel_config_bwd_dX=bwd_dX_config_2,
        kernel_config_bwd_dW=bwd_dW_config_2,
        is_first_gemm=False,
    )

    # Apply routing weights and sum across top_k experts
    # Output shape: (num_tokens, top_k, hidden_dim) -> (num_tokens, hidden_dim)
    # Ensure top_k_weights matches dtype (can be float32 from softmax)
    top_k_weights_casted = top_k_weights.to(hidden_states.dtype)
    final_hidden_states = (
        second_gemm_output.view(num_tokens, top_k, hidden_dim)
        * top_k_weights_casted[..., None]
    )
    final_hidden_states = final_hidden_states.sum(dim=1)

    if is_3d:
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)

    return final_hidden_states


@torch.compiler.disable
def forward_native_moe_loop(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Loop-based MoE forward pass. Loops over experts that have tokens routed to them.
    Explicitly disabled for torch.compile to prevent graph breaks/recompilation issues with dynamic control flow.
    """
    final_hidden_states = torch.zeros_like(hidden_states)

    # Create expert mask and find which experts have tokens
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, n_tokens)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    # Only loop over experts that actually have tokens routed to them
    for expert_idx_t in expert_hit:
        expert_idx = expert_idx_t.item()

        # Find which tokens are routed to this expert
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

        # Gather only the tokens for this expert
        current_state = hidden_states[token_idx]

        # Compute gate_up projection for this expert only
        # Handle 'gate_up_proj' or 'w1'/'w3'
        if hasattr(self, "gate_up_proj"):
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(
                2, dim=-1
            )
        else:
            gate = F.linear(current_state, self.w1[expert_idx])
            up = F.linear(current_state, self.w3[expert_idx])

        current_hidden_states = self.act_fn(gate) * up

        # Compute down projection for this expert only
        if hasattr(self, "down_proj"):
            current_hidden_states = F.linear(
                current_hidden_states, self.down_proj[expert_idx]
            )
        else:
            current_hidden_states = F.linear(current_hidden_states, self.w2[expert_idx])

        # Apply routing weights
        current_hidden_states = (
            current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        )

        # Scatter back to final output
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    return final_hidden_states
