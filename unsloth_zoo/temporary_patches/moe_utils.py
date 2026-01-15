
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
from typing import Optional, Tuple, Callable
import math


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
    if _TRITON_ALLOCATOR_INITIALIZED: return

    try:
        import triton

        # Create a persistent buffer that grows as needed
        # This avoids allocating new memory on every kernel call

        def persistent_alloc_fn(size: int, alignment: int, stream):
            global _PERSISTENT_BUFFER
            # Round up size to avoid frequent reallocations
            # Round to nearest 128 bytes for alignment
            rounded_size = ((size + 128 - 1) // 128) * 128

            if _PERSISTENT_BUFFER is None or _PERSISTENT_BUFFER.numel() * _PERSISTENT_BUFFER.element_size() < rounded_size:
                # Allocate with small headroom (10%) to reduce reallocations
                # Use ByteTensor (uint8) for raw byte storage
                _PERSISTENT_BUFFER = torch.empty(
                    int(rounded_size * 1.1), device = "cuda", dtype = torch.uint8
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
        from unsloth.kernels.moe.grouped_gemm.interface import grouped_gemm, supports_tma
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
        ("grouped_mm",     _check_torch_grouped_mm_supported()),
        ("unsloth_triton", _check_grouped_gemm_available()),
        ("native_torch",   True),
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
                print(f"Unsloth: '{requested_backend}' backend requested but is not available. Falling back to next available.")

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
    token_counts_by_expert = torch.bincount(
        flat_experts,
        minlength=num_experts
    ).to(torch.int32)

    # argsort with stable=True preserves order within each expert
    gather_indices = flat_experts.argsort(stable=True)

    return token_counts_by_expert, gather_indices


@torch.compile(dynamic=True)
def _compiled_moe_routing(
    top_k_index: torch.Tensor,
    num_experts: int,
    num_tokens: int,
    hidden_dim: int,
    top_k: int
):
    """
    Compiled routing logic to eliminate Python overhead.
    """
    flat_top_k = top_k_index.view(-1)
    num_tokens_per_expert = torch.bincount(flat_top_k, minlength=num_experts).int()
    sorted_indices = torch.argsort(flat_top_k, stable=True)
    token_indices = sorted_indices // top_k
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    return num_tokens_per_expert, sorted_indices, token_indices, offsets


@torch.compile(fullgraph=True, dynamic=True)
def _compiled_moe_scatter(
    final_hidden_states: torch.Tensor,
    token_indices: torch.Tensor,
    mm2_out: torch.Tensor,
    permuted_weights: torch.Tensor
):
    """
    Compiled scatter logic.
    """
    mm2_out = mm2_out * permuted_weights.unsqueeze(-1)
    final_hidden_states.index_add_(0, token_indices, mm2_out.to(final_hidden_states.dtype))
    return final_hidden_states


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
    if not hasattr(param, 'lora_A') or not hasattr(param, 'lora_B'):
        return False
    if hasattr(param, 'disable_adapters') and param.disable_adapters:
        return False
    if hasattr(param, 'merged') and param.merged:
        return False
    return len(param.lora_A) > 0





def _get_base_weight(param):
    """Get base weight from potentially wrapped parameter."""
    if hasattr(param, 'get_param'):
        return param.get_param()
    return param


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
    if hasattr(module, 'gate_up_proj'):
        param = module.gate_up_proj
        if isinstance(param, nn.Parameter) and param.ndim == 3:
            return True

    # Check for w1/w2 pattern (separate gate/up projections)
    if hasattr(module, 'w1') and hasattr(module, 'w2'):
        w1 = module.w1
        if isinstance(w1, nn.Parameter) and w1.ndim == 3:
            return True

    return False


_MOE_LORA_DEBUG = os.environ.get("UNSLOTH_MOE_LORA_DEBUG", "0") == "1"

def _get_moe_lora_weights(
    wrapper,
    adapter_name: str = 'default'
) -> Optional[Tuple[torch.Tensor, torch.Tensor, float, int]]:
    """
    Extract LoRA weights from PEFT ParamWrapper for MoE separated computation.

    PEFT stores LoRA weights for 3D MoE parameters as:
        lora_A.weight: (E*R, in_features)  - first projection
        lora_B.weight: (out_features, E*R) - second projection

    We reshape them to:
        lora_A: (E, R, in_features)  - for grouping by expert
        lora_B: (E, out_features, R) - for grouping by expert

    The actual computation order depends on which dimension matches the input,
    which is handled dynamically in _apply_lora_grouped_mm.

    Args:
        wrapper: ParamWrapper module with lora_A, lora_B, scaling
        adapter_name: Name of the active adapter

    Returns:
        (lora_A_reshaped, lora_B_reshaped, scaling, num_experts) or None
        - lora_A: (E, R, in_features)
        - lora_B: (E, out_features, R)
    """
    if not hasattr(wrapper, 'lora_A') or not hasattr(wrapper, 'lora_B'):
        if _MOE_LORA_DEBUG:
            print(f"[MoE LoRA] Wrapper missing lora_A or lora_B attributes")
        return None

    if hasattr(wrapper, 'disable_adapters') and wrapper.disable_adapters:
        if _MOE_LORA_DEBUG:
            print(f"[MoE LoRA] Adapters disabled on wrapper")
        return None
    if hasattr(wrapper, 'merged') and wrapper.merged:
        if _MOE_LORA_DEBUG:
            print(f"[MoE LoRA] Adapters already merged")
        return None

    if not wrapper.lora_A:
        if _MOE_LORA_DEBUG:
            print(f"[MoE LoRA] lora_A is empty")
        return None

    if adapter_name not in wrapper.lora_A:
        adapter_name = list(wrapper.lora_A.keys())[0]

    lora_A_module = wrapper.lora_A[adapter_name]
    lora_B_module = wrapper.lora_B[adapter_name]

    # PEFT stores:
    #   lora_A.weight: (E*R, in_features) - projects from in_features to rank
    #   lora_B.weight: (out_features, E*R) - projects from rank to out_features
    #   delta = lora_B @ lora_A has shape (out_features, in_features)

    weight_A = lora_A_module.weight  # (E*R, in_features)
    weight_B = lora_B_module.weight  # (out_features, E*R)

    scaling = wrapper.scaling[adapter_name]
    num_experts = getattr(wrapper, 'num_experts', 1)

    if num_experts <= 1:
        return None

    # Derive rank per expert
    rank_per_expert = weight_A.shape[0] // num_experts
    in_features = weight_A.shape[1]
    out_features = weight_B.shape[0]

    # Verify dimensions are compatible
    if weight_A.shape[0] != num_experts * rank_per_expert:
        if _MOE_LORA_DEBUG:
            print(f"[MoE LoRA] weight_A shape mismatch: {weight_A.shape[0]} != {num_experts * rank_per_expert}")
        return None

    if weight_B.shape[1] != num_experts * rank_per_expert:
        if _MOE_LORA_DEBUG:
            print(f"[MoE LoRA] weight_B shape mismatch: {weight_B.shape[1]} != {num_experts * rank_per_expert}")
        return None

    # Reshape lora_A: (E*R, in_features) -> (E, R, in_features)
    # NOTE: Do NOT use .contiguous() here - it breaks autograd by creating a copy!
    lora_A = weight_A.view(num_experts, rank_per_expert, in_features)

    # Reshape lora_B: (out_features, E*R) -> (out_features, R, E) -> (E, out_features, R)
    # Must match reference: weight_B.view(out, r, E).permute(2, 0, 1)
    # NOTE: Do NOT use .contiguous() here - it breaks autograd by creating a copy!
    lora_B = weight_B.view(out_features, rank_per_expert, num_experts).permute(2, 0, 1)

    if _MOE_LORA_DEBUG:
        print(f"[MoE LoRA] lora_A: {weight_A.shape} -> {lora_A.shape} (E, R, in)")
        print(f"[MoE LoRA] lora_B: {weight_B.shape} -> {lora_B.shape} (E, out, R)")

    # Return (lora_A, lora_B, scaling, num_experts)
    return lora_A, lora_B, scaling, num_experts


# Helper to bridge explicit calling convention to Unsloth's Triton kernel

# Removed obsolete loop and Triton fallbacks

def native_moe_grouped_mm(inputs: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """
    Native implementation using torch._grouped_mm with a critical fix for backward pass.
    The native backward kernel crashes on 0-stride (broadcasted) gradients (e.g. from sum().backward()).
    We register a hook to ensure dY is contiguous before it reaches the kernel.
    """
    out = torch._grouped_mm(inputs, weight, offs=offsets)
    if out.requires_grad:
        out.register_hook(lambda x: x.contiguous())
    return out



def _apply_grouped_mm_with_lora(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    lora_data: Optional[Tuple[torch.Tensor, torch.Tensor, float, int]] = None,
) -> torch.Tensor:
    """
    Apply grouped GEMM: X @ W + (X @ B @ A) * scaling

    Args:
        inputs: (total_tokens, in_dim)
        weight: (num_experts, in_dim, out_dim) - already transposed if needed
        offsets: Grouped GEMM offsets
        lora_data: Optional (lora_B, lora_A, scaling, num_experts)
            lora_B: (num_experts, in_dim, r) - Blocked layout
            lora_A: (num_experts, r, out_dim) - Blocked layout
    """
    # 1. Base forward: X @ W
    # Base weights are typically frozen under LoRA training, but we still need dX to flow back.
    # Triton path is OK for base, but for LoRA weights we must use an autograd-safe path.
    out = native_moe_grouped_mm(inputs, weight, offsets)

    # 2. Add Separated LoRA: + ((X @ B) @ A) * scaling
    if lora_data is not None:
        lora_B, lora_A, scaling = lora_data[:3]  # lora_B is (E, in, R), lora_A is (E, R, out)

        # CRITICAL: Use torch._grouped_mm (or safe loop fallback) for LoRA so grads
        # reliably flow to LoRA A/B weights. Triton grouped GEMM is not guaranteed
        # to have a correct backward for these small dynamic shapes.
        out = out + _apply_lora_grouped_mm(inputs, lora_B, lora_A, offsets, float(scaling))

    return out


# Store original ParamWrapper.forward for fallback
_original_param_wrapper_forward = None


def _patched_param_wrapper_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Patched ParamWrapper.forward for MoE separated LoRA.

    For MoE modules with stored weakref (gate_up_proj, down_proj):
    - Bypasses PEFT's _activate_lora entirely by calling base_layer directly
    - forward_native_grouped_mm extracts LoRA weights from the stored weakref

    For non-MoE modules:
    - Falls back to original PEFT forward
    """
    # Get IMMEDIATE base_layer for forward call
    # CRITICAL: Use self.base_layer NOT self.get_base_layer()!
    # get_base_layer() recursively traverses to deepest layer (Qwen3MoeExperts),
    # which would skip the gate_up_proj wrapper's forward entirely.
    immediate_base_layer = self.base_layer

    # For storing wrapper reference, we DO need the actual experts module
    # Use get_base_layer() to find it (recursive traversal is correct here)
    experts_module = self.get_base_layer()

    param_name = getattr(self, 'parameter_name', None)

    # Check if this module has a stored weakref (set by patch_param_wrapper_for_moe)
    if param_name in ('gate_up_proj', 'down_proj'):
        # MoE experts: bypass PEFT's _activate_lora completely.
        #
        # IMPORTANT:
        # The fast MoE kernels need access to this ParamWrapper to extract LoRA A/B.
        # In many flows we don't call patch_param_wrapper_for_moe(model=...), so also
        # attach lazily on first forward.
        try:
            # key = f"{param_name}_lora_wrapper"
            # Explicitly use the correct key based on param_name
            if param_name == 'gate_up_proj':
                 key = 'gate_up_proj_lora_wrapper'
            elif param_name == 'down_proj':
                 key = 'down_proj_lora_wrapper'
            else:
                 key = f"{param_name}_lora_wrapper"

            # Store on the actual experts module (Qwen3MoeExperts), not on intermediate wrappers
            if key not in experts_module.__dict__:
                experts_module.__dict__[key] = self

            # Ensure wrapper.num_experts is set for _get_moe_lora_weights reshaping logic.
            if not hasattr(self, "num_experts"):
                if hasattr(experts_module, "num_experts"):
                    self.num_experts = experts_module.num_experts
                else:
                    p = getattr(experts_module, param_name, None)
                    if hasattr(p, "shape") and len(p.shape) >= 1:
                        self.num_experts = p.shape[0]
        except Exception:
            pass

        # Call the IMMEDIATE base_layer to preserve the wrapper chain
        # (e.g., down_proj calls gate_up_proj which calls Qwen3MoeExperts)
        return immediate_base_layer(x, *args, **kwargs)


    # Non-MoE: use original PEFT forward with _activate_lora
    return _original_param_wrapper_forward(self, x, *args, **kwargs)


def patch_param_wrapper_for_moe(model=None):
    """
    Patch PEFT's ParamWrapper.forward to use separated LoRA for MoE.

    If model is provided, also stores weakrefs to ParamWrappers on base layers
    for efficient LoRA weight extraction during forward.

    This should be called after PEFT is imported and get_peft_model is called.
    """
    global _original_param_wrapper_forward

    try:
        from peft.tuners.lora.layer import ParamWrapper

        # Store original forward (only once)
        if _original_param_wrapper_forward is None:
            _original_param_wrapper_forward = ParamWrapper.forward

        # Patch with our simplified version
        ParamWrapper.forward = _patched_param_wrapper_forward

        # If model provided, store weakrefs for lazy LoRA extraction
        if model is not None:
            for module in model.modules():
                if not isinstance(module, ParamWrapper):
                    continue
                param_name = getattr(module, 'parameter_name', None)
                if param_name not in ('gate_up_proj', 'down_proj'):
                    continue
                # Navigate through nested ParamWrappers to find the actual experts module
                base_layer = module.get_base_layer()
                while hasattr(base_layer, 'get_base_layer') and callable(base_layer.get_base_layer):
                    try:
                        next_layer = base_layer.get_base_layer()
                        if next_layer is None:
                            break
                        base_layer = next_layer
                    except:
                        break
                if base_layer is None:
                    continue

                # CRITICAL: Set num_experts on wrapper so _get_moe_lora_weights can reshape correctly
                # Get num_experts from base_layer (the MoE experts module)
                if hasattr(base_layer, 'num_experts'):
                    module.num_experts = base_layer.num_experts
                elif hasattr(base_layer, param_name):
                    # Infer from parameter shape: (num_experts, ..., ...)
                    param = getattr(base_layer, param_name)
                    if hasattr(param, 'shape') and len(param.shape) >= 1:
                        module.num_experts = param.shape[0]

                # Store reference to ParamWrapper for lazy LoRA extraction
                # forward_native_grouped_mm will use this to get LoRA weights
                base_layer.__dict__[f'{param_name}_lora_wrapper'] = module

        return True
    except ImportError:
        return False


def _apply_lora_grouped_mm(
    inputs: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    offsets: torch.Tensor,
    scaling: float,
    grouped_mm_func: Optional[Callable] = None,
    token_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply LoRA using native torch._grouped_mm for proper gradient computation.

    Uses torch._grouped_mm instead of Triton kernels to ensure gradients flow
    correctly to LoRA weights during training.

    The computation order is determined dynamically based on which LoRA weight's
    dimensions match the input. This handles both gate_up_proj and down_proj
    correctly regardless of whether the base weight is transposed.

    Args:
        inputs: (N, in_dim) - already permuted by expert
        lora_A: (E, R, features_A) - PEFT lora_A reshaped
        lora_B: (E, features_B, R) - PEFT lora_B reshaped
        offsets: (E,) cumulative token counts per expert
        scaling: LoRA scaling factor
        grouped_mm_func: Optional function(inputs, weights, offsets, token_counts) -> output
        token_counts: (E,) token counts per expert (required for some backends like Triton)

    Returns:
        lora_delta: (N, out_dim) scaled LoRA contribution
    """
    if grouped_mm_func is None:
        if not (_TORCH_GROUPED_MM_AVAILABLE and _check_torch_grouped_mm_supported()):
            raise RuntimeError("torch._grouped_mm is required for MoE LoRA training but is not available.")
        grouped_mm_func = native_moe_grouped_mm

    # ---------------------------------------------------------------------
    # Rank padding for torch._grouped_mm alignment
    #
    # Some torch._grouped_mm builds require 16-byte stride alignment.
    # For bf16/fp16 this effectively means the "rank" dimension should be a
    # multiple of 8 elements; for fp32 it's a multiple of 4 elements.
    #
    # We only pad when needed, and we do it at call time (NOT at extraction
    # time) so autograd still flows back to the original PEFT weights.
    # ---------------------------------------------------------------------
    def _pad_rank_for_grouped_mm(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lora_A: (E, R, in_features or out_features)
        # lora_B: (E, out_features or in_features, R)
        # Rank dims are a.dim=1 and b.dim=2.
        if a.dim() != 3 or b.dim() != 3:
            return a, b
        # 16-byte alignment expressed in elements for this dtype
        align_elems = max(1, 16 // a.element_size())
        r = int(a.shape[1])
        rem = r % align_elems
        if rem == 0:
            return a, b
        pad = align_elems - rem
        # Pad along rank dim: a dim=1, b dim=2
        # F.pad expects pad spec for last dim first: (last_left, last_right, prev_left, prev_right, ...)
        # a: pad dim=1 => second-to-last dim => pad list: (0,0) for dim=2, (0,pad) for dim=1
        a_padded = F.pad(a, (0, 0, 0, pad))
        # b: pad dim=2 => last dim => pad list: (0,pad)
        b_padded = F.pad(b, (0, pad))
        return a_padded, b_padded


    # Cast to input dtype while preserving gradients
    lora_A_cast = lora_A.to(inputs.dtype)
    lora_B_cast = lora_B.to(inputs.dtype)

    # Apply rank padding only for the native torch._grouped_mm backend.
    # Triton kernels do not have the same strict stride requirements here.
    if grouped_mm_func == native_moe_grouped_mm:
        lora_A_cast, lora_B_cast = _pad_rank_for_grouped_mm(lora_A_cast, lora_B_cast)

    input_dim = inputs.shape[-1]

    # Dynamic dimension matching (like moe_lora_split.py)
    # lora_A is (E, R, features_A)
    # lora_B is (E, features_B, R)
    #
    # Case 1: lora_A's last dim matches input -> X @ lora_A.T, then result @ lora_B.T
    # Case 2: lora_B's second dim matches input -> X @ lora_B, then result @ lora_A

    if lora_A.shape[-1] == input_dim:
        # lora_A matches: X @ lora_A.T -> (N, R), then result @ lora_B.T -> (N, out)

        # Native torch._grouped_mm expects (E, In, Out)
        # Triton grouped_gemm expects (E, Out, In)

        # 1. First Matmul (lora_A)
        # lora_A is (E, R, In).
        if grouped_mm_func == native_moe_grouped_mm:
            # Native needs (E, In, R) -> Transpose
            lora_intermediate = grouped_mm_func(inputs, lora_A_cast.transpose(-2, -1).contiguous(), offsets)
        else:
            # Triton needs (E, R, In) -> No Transpose
            if token_counts is None:
                lora_intermediate = grouped_mm_func(inputs, lora_A_cast.contiguous(), offsets)
            else:
                lora_intermediate = grouped_mm_func(inputs, lora_A_cast.contiguous(), offsets, token_counts)

        # 2. Second Matmul (lora_B)
        # lora_B is (E, Out, R).
        if grouped_mm_func == native_moe_grouped_mm:
            # Native needs (E, R, Out) -> Transpose
            lora_delta = grouped_mm_func(lora_intermediate, lora_B_cast.transpose(-2, -1).contiguous(), offsets)
        else:
            # Triton needs (E, Out, R) -> No Transpose
            if token_counts is None:
                lora_delta = grouped_mm_func(lora_intermediate, lora_B_cast.contiguous(), offsets)
            else:
                lora_delta = grouped_mm_func(lora_intermediate, lora_B_cast.contiguous(), offsets, token_counts)

    elif lora_B.shape[1] == input_dim:
        # lora_B matches: X @ lora_B -> (N, R), then result @ lora_A -> (N, out)

        # 1. First Matmul (lora_B)
        # lora_B is (E, In, R) (In this branch input_dim matches dim 1)
        # Wait, lora_B definition above says (E, Out, R).
        # If lora_B matches input, then lora_B is (E, In, R).

        if grouped_mm_func == native_moe_grouped_mm:
             # Native needs (E, In, R) -> No Transpose
             lora_intermediate = grouped_mm_func(inputs, lora_B_cast.contiguous(), offsets)
        else:
             # Triton needs (E, R, In) -> Transpose
             if token_counts is None:
                 lora_intermediate = grouped_mm_func(inputs, lora_B_cast.transpose(-2, -1).contiguous(), offsets)
             else:
                 lora_intermediate = grouped_mm_func(inputs, lora_B_cast.transpose(-2, -1).contiguous(), offsets, token_counts)

        # 2. Second Matmul (lora_A)
        # lora_A is (E, R, Out).
        if grouped_mm_func == native_moe_grouped_mm:
             # Native needs (E, R, Out) -> No Transpose
             lora_delta = grouped_mm_func(lora_intermediate, lora_A_cast.contiguous(), offsets)
        else:
             # Triton needs (E, Out, R) -> Transpose
             if token_counts is None:
                 lora_delta = grouped_mm_func(lora_intermediate, lora_A_cast.transpose(-2, -1).contiguous(), offsets)
             else:
                 lora_delta = grouped_mm_func(lora_intermediate, lora_A_cast.transpose(-2, -1).contiguous(), offsets, token_counts)
    else:
        raise ValueError(
            f"LoRA shapes do not match input. input_dim={input_dim}, "
            f"lora_A={tuple(lora_A.shape)}, lora_B={tuple(lora_B.shape)}"
        )

    return lora_delta * scaling


def _setup_moe_lora_wrappers_lazy(experts_module):
    """
    Lazily detect and setup LoRA wrappers for MoE experts module.
    Called on first forward pass if wrappers haven't been setup yet.

    This searches the module hierarchy for ParamWrapper modules
    that wrap gate_up_proj/down_proj and sets up the necessary references.
    """
    if getattr(experts_module, '_moe_lora_setup_done', False):
        return

    try:
        from peft.tuners.lora.layer import ParamWrapper
    except ImportError:
        experts_module._moe_lora_setup_done = True
        return

    # The experts module should have a _parent reference or we can find wrappers
    # by checking if gate_up_proj/down_proj are ParamWrappers themselves
    for param_name in ('gate_up_proj', 'down_proj'):
        # Check if the attribute is a ParamWrapper
        attr = getattr(experts_module, param_name, None)
        if attr is None:
            continue

        # If it's a ParamWrapper directly attached
        if isinstance(attr, ParamWrapper):
            wrapper = attr
            # Set num_experts on wrapper if not set
            if not hasattr(wrapper, 'num_experts') or getattr(wrapper, 'num_experts', 1) <= 1:
                wrapper.num_experts = experts_module.num_experts
            # Store reference
            experts_module.__dict__[f'{param_name}_lora_wrapper'] = wrapper
            if _MOE_LORA_DEBUG:
                print(f"[MoE LoRA] Found direct ParamWrapper for {param_name}, num_experts={wrapper.num_experts}")
            continue

        # Check if there's a wrapper stored in module's _modules that wraps this param
        # This handles the case where PEFT replaces the parameter with a wrapped version
        for name, module in experts_module._modules.items() if hasattr(experts_module, '_modules') else []:
            if isinstance(module, ParamWrapper):
                wrapper_param_name = getattr(module, 'parameter_name', None)
                if wrapper_param_name == param_name:
                    if not hasattr(module, 'num_experts') or getattr(module, 'num_experts', 1) <= 1:
                        module.num_experts = experts_module.num_experts
                    experts_module.__dict__[f'{param_name}_lora_wrapper'] = module
                    if _MOE_LORA_DEBUG:
                        print(f"[MoE LoRA] Found ParamWrapper in _modules for {param_name}, num_experts={module.num_experts}")

    experts_module._moe_lora_setup_done = True


def _get_lora_wrapper_for_param(experts_module, param_name: str):
    """
    Get LoRA wrapper for a parameter, with lazy detection fallback.
    """
    # First check if already set up
    wrapper = experts_module.__dict__.get(f'{param_name}_lora_wrapper')
    if wrapper is not None:
        return wrapper

    # Try lazy setup if not done yet
    if not getattr(experts_module, '_moe_lora_setup_done', False):
        _setup_moe_lora_wrappers_lazy(experts_module)
        wrapper = experts_module.__dict__.get(f'{param_name}_lora_wrapper')

    return wrapper


# Helper to get wrapper
def _get_lora_wrapper_for_param(experts_module, param_name):
    """
    Get the PEFT ParamWrapper for a specific parameter (gate_up_proj or down_proj).
    Uses the explicit key stored in __dict__ if available, otherwise searches hierarchy.
    """
    # 1. Fast path: check if explicitly stored in __dict__
    key = f"{param_name}_lora_wrapper"
    if key in experts_module.__dict__:
       return experts_module.__dict__[key]

    # 2. Lazy Setup: try to find it in modules if not already done
    # This handles cases where patch_param_wrapper_for_moe wasn't called on the whole model
    if not getattr(experts_module, '_moe_lora_lazy_search_done', False):
         _setup_moe_lora_wrappers_lazy(experts_module)
         experts_module._moe_lora_lazy_search_done = True
         if key in experts_module.__dict__:
             return experts_module.__dict__[key]

    return None


@torch.compiler.disable
def forward_native_grouped_mm(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Optimized Qwen3 MoE forward pass.
    Removes generic checks and function call overhead.

    For LoRA: Uses native torch._grouped_mm to ensure proper gradient flow
    to LoRA weights during training. The base model GEMM can use faster
    Triton kernels since base weights are frozen.
    """
    # Fast path for Qwen3 dimensionality
    if hidden_states.dim() == 2:
        sequence_length, hidden_dim = hidden_states.shape
        batch_size = 1
    else:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

    # 1. Routing
    flat_top_k = top_k_index.view(-1)
    num_tokens_per_expert = torch.bincount(flat_top_k, minlength=self.num_experts).int()
    sorted_indices = torch.argsort(flat_top_k, stable=True)
    token_indices = sorted_indices // top_k_index.shape[1]
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    # 2. Permute Input
    permuted_input = hidden_states[token_indices]

    # 3. Gate Up Proj (Fused) - INLINE
    w1 = self.gate_up_proj.transpose(-2, -1)

    # Base GEMM
    mm1_out = native_moe_grouped_mm(permuted_input, w1, offsets)

    # LoRA
    gate_up_wrapper = _get_lora_wrapper_for_param(self, 'gate_up_proj')
    if gate_up_wrapper is not None:
        lora_data = _get_moe_lora_weights(gate_up_wrapper)
        if lora_data is not None:
            lora_A, lora_B, scaling = lora_data[:3]
            lora_delta = _apply_lora_grouped_mm(permuted_input, lora_A, lora_B, offsets, scaling,
                                              grouped_mm_func = native_moe_grouped_mm)
            mm1_out = mm1_out + lora_delta

    # Activation
    gate, up = mm1_out.chunk(2, dim=-1)
    inter = F.silu(gate) * up

    # 4. Down Proj
    w2 = self.down_proj.transpose(-2, -1)

    # Base GEMM
    mm2_out = native_moe_grouped_mm(inter, w2, offsets)

    # LoRA
    down_wrapper = _get_lora_wrapper_for_param(self, 'down_proj')
    if down_wrapper is not None:
        lora_data = _get_moe_lora_weights(down_wrapper)
        if lora_data is not None:
            lora_A, lora_B, scaling = lora_data[:3]
            lora_delta = _apply_lora_grouped_mm(inter, lora_A, lora_B, offsets, scaling,
                                              grouped_mm_func = native_moe_grouped_mm)
            mm2_out = mm2_out + lora_delta

    # 5. Scatter
    flat_weights = top_k_weights.view(-1)
    permuted_weights = flat_weights[sorted_indices]
    mm2_out = mm2_out * permuted_weights.unsqueeze(-1)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device
    )
    final_hidden_states.index_add_(0, token_indices, mm2_out.to(hidden_states.dtype))

    if sequence_length == hidden_states.shape[0]: # 2D case
            return final_hidden_states


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
            intermediate_dim=hidden_dim, # Output dim for 2nd GEMM is hidden_dim
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

    # Import kernel config classes for LoRA
    try:
        from unsloth.kernels.moe.grouped_gemm.kernels.tuning import (
            KernelConfigForward, KernelConfigBackward_dX, KernelConfigBackward_dW
        )
    except ImportError:
        # Fallback if internal structure changes
        from unsloth.kernels.moe.autotune_cache import (
            KernelConfigForward, KernelConfigBackward_dX, KernelConfigBackward_dW
        )

    # First grouped GEMM: gate_up projection
    first_gemm_output = grouped_gemm(
        X=hidden_states,
        W=w1,
        m_sizes=token_counts_by_expert,
        topk=top_k,
        gather_indices=gather_indices,
        permute_x=True,
        permute_y=False,
        autotune=False, # We use cached configs
        kernel_config_fwd=fwd_config_1,
        kernel_config_bwd_dX=bwd_dX_config_1,
        kernel_config_bwd_dW=bwd_dW_config_1,
        is_first_gemm=True,
    )

    # LoRA for gate_up_proj using Triton grouped_gemm
    gate_up_wrapper = _get_lora_wrapper_for_param(self, 'gate_up_proj')
    if gate_up_wrapper is not None:
        lora_data = _get_moe_lora_weights(gate_up_wrapper)
        if lora_data is not None:
            lora_A, lora_B, scaling, _ = lora_data
            # lora_A: (E, R, in_features), lora_B: (E, out_features, R)
            # Dimension matching: check which weight's dim matches input
            # Case 1: lora_A[-1] == input_dim -> X @ lora_A.T @ lora_B.T
            # Case 2: lora_B[1] == input_dim -> X @ lora_B @ lora_A

            lora_rank = lora_A.shape[1]  # R
            lora_blk = 16 if lora_rank % 16 == 0 else (32 if lora_rank % 32 == 0 else 64)

            if lora_A.shape[-1] == hidden_dim:
                # Case 1: X @ lora_A.T -> (M, R), then @ lora_B.T -> (M, out)
                first_W = lora_A  # (E, R, in_features)
                second_W = lora_B  # (E, out_features, R)
                second_out = lora_B.shape[1]
            elif lora_B.shape[1] == hidden_dim:
                # Case 2: X @ lora_B -> (M, R), then @ lora_A -> (M, out)
                # lora_B is (E, out_features, R) but we want (E, R, out_features) for first matmul
                # Actually grouped_gemm expects (E, N, K) where output is (M, N)
                # So we need to transpose: lora_B.transpose(-2, -1) = (E, R, out_features)
                first_W = lora_B.transpose(-2, -1).contiguous()  # (E, R, hidden_dim)
                # For second matmul, lora_A is (E, R, in_features)
                # We want intermediate (M, R) @ lora_A to give (M, in_features)
                # lora_A transposed: (E, in_features, R)
                second_W = lora_A.transpose(-2, -1).contiguous()  # (E, in_features, R)
                second_out = lora_A.shape[-1]
            else:
                raise ValueError(f"LoRA shapes don't match input: hidden_dim={hidden_dim}, "
                               f"lora_A={lora_A.shape}, lora_B={lora_B.shape}")

            # First matmul configs
            lora_fwd_config = KernelConfigForward(
                BLOCK_SIZE_M=64, BLOCK_SIZE_N=lora_blk, BLOCK_SIZE_K=32,
                num_warps=4, num_stages=2,
                use_tma_load_x=False, use_tma_load_w=False, use_tma_store=False
            )
            lora_bwd_dX_config = KernelConfigBackward_dX(
                BLOCK_SIZE_M=64, BLOCK_SIZE_N=lora_blk, BLOCK_SIZE_K=32,
                num_warps=4, num_stages=2,
                use_tma_load_dy=False, use_tma_load_w=False, use_tma_store=False
            )
            lora_bwd_dW_config = KernelConfigBackward_dW(
                BLOCK_SIZE_M=64, BLOCK_SIZE_N=lora_blk, BLOCK_SIZE_K=32,
                num_warps=4, num_stages=2,
                use_tma_load_dy=False, use_tma_load_x=False
            )

            # First matmul: X -> (M, R)
            lora_intermediate = grouped_gemm(
                X=hidden_states,
                W=first_W.to(hidden_states.dtype),
                m_sizes=token_counts_by_expert,
                topk=top_k,
                gather_indices=gather_indices,
                permute_x=True,
                permute_y=False,
                autotune=False,
                kernel_config_fwd=lora_fwd_config,
                kernel_config_bwd_dX=lora_bwd_dX_config,
                kernel_config_bwd_dW=lora_bwd_dW_config,
                is_first_gemm=True,
            )

            # Second matmul configs
            out_blk = 64 if second_out % 64 == 0 else (32 if second_out % 32 == 0 else 16)

            lora_fwd_config2 = KernelConfigForward(
                BLOCK_SIZE_M=64, BLOCK_SIZE_N=out_blk, BLOCK_SIZE_K=lora_blk,
                num_warps=4, num_stages=2,
                use_tma_load_x=False, use_tma_load_w=False, use_tma_store=False
            )
            lora_bwd_dX_config2 = KernelConfigBackward_dX(
                BLOCK_SIZE_M=64, BLOCK_SIZE_N=out_blk, BLOCK_SIZE_K=lora_blk,
                num_warps=4, num_stages=2,
                use_tma_load_dy=False, use_tma_load_w=False, use_tma_store=False
            )
            lora_bwd_dW_config2 = KernelConfigBackward_dW(
                BLOCK_SIZE_M=64, BLOCK_SIZE_N=out_blk, BLOCK_SIZE_K=lora_blk,
                num_warps=4, num_stages=2,
                use_tma_load_dy=False, use_tma_load_x=False
            )

            # Second matmul: intermediate -> (M, out)
            lora_delta = grouped_gemm(
                X=lora_intermediate,
                W=second_W.to(hidden_states.dtype),
                m_sizes=token_counts_by_expert,
                topk=top_k,
                gather_indices=gather_indices,
                permute_x=False,
                permute_y=False,
                autotune=False,
                kernel_config_fwd=lora_fwd_config2,
                kernel_config_bwd_dX=lora_bwd_dX_config2,
                kernel_config_bwd_dW=lora_bwd_dW_config2,
                is_first_gemm=False,
            )

            first_gemm_output = first_gemm_output + lora_delta * scaling

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
        autotune=False, # We use cached configs
        kernel_config_fwd=fwd_config_2,
        kernel_config_bwd_dX=bwd_dX_config_2,
        kernel_config_bwd_dW=bwd_dW_config_2,
        is_first_gemm=False,
    )

    # LoRA for down_proj using Triton grouped_gemm
    down_wrapper = _get_lora_wrapper_for_param(self, 'down_proj')
    if down_wrapper is not None:
        lora_data = _get_moe_lora_weights(down_wrapper)
        if lora_data is not None:
            lora_A, lora_B, scaling, _ = lora_data
            # lora_A: (E, R, in_features), lora_B: (E, out_features, R)
            # For down_proj: input is intermediate (after SiLU)
            input_dim_down = intermediate.shape[-1]

            lora_rank = lora_A.shape[1]
            lora_blk = 16 if lora_rank % 16 == 0 else (32 if lora_rank % 32 == 0 else 64)

            if lora_A.shape[-1] == input_dim_down:
                # Case 1: X @ lora_A.T -> (M, R), then @ lora_B.T -> (M, out)
                first_W = lora_A
                second_W = lora_B
                second_out = lora_B.shape[1]
            elif lora_B.shape[1] == input_dim_down:
                # Case 2: X @ lora_B -> (M, R), then @ lora_A -> (M, out)
                first_W = lora_B.transpose(-2, -1).contiguous()
                second_W = lora_A.transpose(-2, -1).contiguous()
                second_out = lora_A.shape[-1]
            else:
                raise ValueError(f"down_proj LoRA shapes don't match input: input_dim={input_dim_down}, "
                               f"lora_A={lora_A.shape}, lora_B={lora_B.shape}")

            lora_fwd_config = KernelConfigForward(
                BLOCK_SIZE_M=64, BLOCK_SIZE_N=lora_blk, BLOCK_SIZE_K=32,
                num_warps=4, num_stages=2,
                use_tma_load_x=False, use_tma_load_w=False, use_tma_store=False
            )
            lora_bwd_dX_config = KernelConfigBackward_dX(
                BLOCK_SIZE_M=64, BLOCK_SIZE_N=lora_blk, BLOCK_SIZE_K=32,
                num_warps=4, num_stages=2,
                use_tma_load_dy=False, use_tma_load_w=False, use_tma_store=False
            )
            lora_bwd_dW_config = KernelConfigBackward_dW(
                BLOCK_SIZE_M=64, BLOCK_SIZE_N=lora_blk, BLOCK_SIZE_K=32,
                num_warps=4, num_stages=2,
                use_tma_load_dy=False, use_tma_load_x=False
            )

            # First matmul: intermediate -> (M, R)
            lora_intermediate_down = grouped_gemm(
                X=intermediate,
                W=first_W.to(intermediate.dtype),
                m_sizes=token_counts_by_expert,
                topk=top_k,
                gather_indices=gather_indices,
                permute_x=False,
                permute_y=False,
                autotune=False,
                kernel_config_fwd=lora_fwd_config,
                kernel_config_bwd_dX=lora_bwd_dX_config,
                kernel_config_bwd_dW=lora_bwd_dW_config,
                is_first_gemm=True,
            )

            out_blk = 64 if second_out % 64 == 0 else (32 if second_out % 32 == 0 else 16)

            lora_fwd_config2 = KernelConfigForward(
                BLOCK_SIZE_M=64, BLOCK_SIZE_N=out_blk, BLOCK_SIZE_K=lora_blk,
                num_warps=4, num_stages=2,
                use_tma_load_x=False, use_tma_load_w=False, use_tma_store=False
            )
            lora_bwd_dX_config2 = KernelConfigBackward_dX(
                BLOCK_SIZE_M=64, BLOCK_SIZE_N=out_blk, BLOCK_SIZE_K=lora_blk,
                num_warps=4, num_stages=2,
                use_tma_load_dy=False, use_tma_load_w=False, use_tma_store=False
            )
            lora_bwd_dW_config2 = KernelConfigBackward_dW(
                BLOCK_SIZE_M=64, BLOCK_SIZE_N=out_blk, BLOCK_SIZE_K=lora_blk,
                num_warps=4, num_stages=2,
                use_tma_load_dy=False, use_tma_load_x=False
            )

            # Second matmul: lora_intermediate_down -> (M, out)
            lora_delta = grouped_gemm(
                X=lora_intermediate_down,
                W=second_W.to(intermediate.dtype),
                m_sizes=token_counts_by_expert,
                topk=top_k,
                gather_indices=gather_indices,
                permute_x=False,
                permute_y=False,
                autotune=False,
                kernel_config_fwd=lora_fwd_config2,
                kernel_config_bwd_dX=lora_bwd_dX_config2,
                kernel_config_bwd_dW=lora_bwd_dW_config2,
                is_first_gemm=False,
            )

            second_gemm_output = second_gemm_output + lora_delta * scaling

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
    Includes LoRA support via PEFT wrapper logic if present.
    """
    final_hidden_states = torch.zeros_like(hidden_states)

    # Create expert mask and find which experts have tokens
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, n_tokens)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    # Get wrappers once
    gate_up_wrapper = _get_lora_wrapper_for_param(self, 'gate_up_proj')
    down_wrapper = _get_lora_wrapper_for_param(self, 'down_proj')

    gate_up_lora = _get_moe_lora_weights(gate_up_wrapper) if gate_up_wrapper else None
    down_lora = _get_moe_lora_weights(down_wrapper) if down_wrapper else None

    # Only loop over experts that actually have tokens routed to them
    for expert_idx_t in expert_hit:
        expert_idx = expert_idx_t.item()

        # Find which tokens are routed to this expert
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

        # Gather only the tokens for this expert
        current_state = hidden_states[token_idx]

        # Compute gate_up projection for this expert only
        if hasattr(self, "gate_up_proj"):
            # Base weight: x @ w.T
            w = self.gate_up_proj[expert_idx]
            gate_up = current_state @ w.T

            # Add LoRA
            if gate_up_lora:
                lora_A, lora_B, scaling, _ = gate_up_lora
                l_A = lora_A[expert_idx]
                l_B = lora_B[expert_idx]
                gate_up = gate_up + _apply_lora_slice(current_state, l_A, l_B, scaling)

            gate, up = gate_up.chunk(2, dim=-1)

        else:
            # Fallback for w1/w3
            gate = current_state @ self.w1[expert_idx].T
            up   = current_state @ self.w3[expert_idx].T

        current_hidden_states = F.silu(gate) * up

        # Compute down projection for this expert only
        if hasattr(self, "down_proj"):
            w2 = self.down_proj[expert_idx]
            down_out = current_hidden_states @ w2.T

            # Add LoRA
            if down_lora:
                lora_A, lora_B, scaling, _ = down_lora
                l_A = lora_A[expert_idx]
                l_B = lora_B[expert_idx]
                delta = _apply_lora_slice(current_hidden_states, l_A, l_B, scaling)
                down_out = down_out + delta

            current_hidden_states = down_out

        else:
            current_hidden_states = current_hidden_states @ self.w2[expert_idx].T

        # Apply routing weights
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

        # Scatter back to final output
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


def _apply_lora_slice(
    inputs: torch.Tensor,
    l_A: torch.Tensor,
    l_B: torch.Tensor,
    scaling: float
) -> torch.Tensor:
    """
    Apply LoRA slice for a single expert.
    Handles dynamic shape matching (normal vs transposed).
    """
    in_dim = inputs.shape[-1]

    # Cast to input dtype
    l_A = l_A.to(inputs.dtype)
    l_B = l_B.to(inputs.dtype)

    if l_A.shape[1] == in_dim:
        # Case 1: Standard LoRA (A is first)
        # l_A: (R, in)
        # l_B: (out, R)
        # inputs @ l_A.T -> (N, R)
        # result @ l_B.T -> (N, out)
        return (inputs @ l_A.T @ l_B.T) * scaling

    elif l_B.shape[0] == in_dim:
        # Case 2: Transposed LoRA (B is first)
        # l_B: (in, R) -- derived from (out, R) where out=in
        # l_A: (R, out) -- derived from (R, in) where in=out
        # inputs @ l_B -> (N, R)
        # result @ l_A -> (N, out)
        return (inputs @ l_B @ l_A) * scaling

    else:
        raise RuntimeError(
            f"LoRA shape mismatch for single expert. "
            f"inputs={inputs.shape}, l_A={l_A.shape}, l_B={l_B.shape}"
        )
