
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
    #   lora_A.weight: (rank, in_features)   -> Input Projection
    #   lora_B.weight: (out_features, rank)  -> Output Projection
    #
    # Unsloth execution order:
    #   Step 1: X @ Input_Matrix  -> (N, R)
    #   Step 2: Res @ Output_Matrix -> (N, Out)
    #
    # So we map:
    #   weight_B (Input Matrix)  = lora_A.weight
    #   weight_A (Output Matrix) = lora_B.weight

    # NOTE: We keep variable names weight_A/weight_B consistent with their usage below
    # where weight_B is reshaped for "First Step" and weight_A for "Second Step".

    weight_B = lora_A_module.weight  # (rank, in_dim)
    weight_A = lora_B_module.weight  # (out_dim, rank)

    scaling = wrapper.scaling[adapter_name]
    num_experts = getattr(wrapper, 'num_experts', 1)

    if num_experts <= 1:
        return None

    # Logic explanation:
    # weight_B (Input Proj): (E*R, in_dim) or (R_total, in_dim)
    #   rank_per_expert is derived from this matrix primarily?
    #   Actually weight_A (Output Proj) is (out_dim, E*R).
    #   PEFT usually stores (Rank, In) and (Out, Rank).

    # Let's adjust derivation to be robust.
    # weight_B (from lora_A): (E*R, in_dim)
    # weight_A (from lora_B): (out_dim, E*R)

    rank_per_expert = weight_B.shape[0] // num_experts
    in_dim = weight_B.shape[1]   # input dimension

    # Output dim from weight_A
    out_dim = weight_A.shape[0]

    # Verify dimensions are compatible
    if weight_B.shape[0] != num_experts * rank_per_expert:
         return None

    if weight_A.shape[1] != num_experts * rank_per_expert:
        return None

    # Reshape B (Input Proj) for first step: X @ B
    # weight_B is (E*R, in_dim).
    # We want B_reshaped to be (E, in_dim, R) typically for grouped_mm?
    # Actually native_moe_grouped_mm expects weight as (E, in, out).
    #
    # For Step 1: X(N, in) @ B -> (N, R).
    # So B must be (E, in, R).
    # weight_B is (E*R, in).
    # reshape(num_experts, rank_per_expert, in_dim) -> (E, R, in).
    # permute(0, 2, 1) -> (E, in, R).

    B_reshaped = weight_B.reshape(num_experts, rank_per_expert, in_dim)
    B_reshaped = B_reshaped.permute(0, 2, 1).contiguous() # (E, in, R)

    # Reshape A (Output Proj) for second step: Res @ A
    # weight_A is (out_dim, E*R).
    # We want A to be (E, R, out).
    # reshape(out_dim, num_experts, rank_per_expert) -> (out, E, R).
    # permute(1, 2, 0) -> (E, R, out).

    A_reshaped = weight_A.reshape(out_dim, num_experts, rank_per_expert)
    A_reshaped = A_reshaped.permute(1, 2, 0).contiguous() # (E, R, out)

    # Return (B, A) for application order: (X @ B) @ A
    return B_reshaped, A_reshaped, scaling, num_experts


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
    base_layer = self.get_base_layer()
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

            if key not in base_layer.__dict__:
                base_layer.__dict__[key] = self

            # Ensure wrapper.num_experts is set for _get_moe_lora_weights reshaping logic.
            if not hasattr(self, "num_experts"):
                if hasattr(base_layer, "num_experts"):
                    self.num_experts = base_layer.num_experts
                else:
                    p = getattr(base_layer, param_name, None)
                    if hasattr(p, "shape") and len(p.shape) >= 1:
                        self.num_experts = p.shape[0]
        except Exception:
            pass

        # The LoRA is extracted lazily in forward_native_grouped_mm via base_layer.__dict__.
        return base_layer(x, *args, **kwargs)

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
                base_layer = module.get_base_layer()
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
    lora_B: torch.Tensor,
    lora_A: torch.Tensor,
    offsets: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """
    Apply LoRA using native torch._grouped_mm for proper gradient computation.

    Uses torch._grouped_mm instead of Triton kernels to ensure gradients flow
    correctly to LoRA weights during training.

    Args:
        inputs: (N, in_dim) - already permuted by expert
        lora_B: (E, in_dim, R) - first projection
        lora_A: (E, R, out_dim) - second projection
        offsets: (E,) cumulative token counts per expert
        scaling: LoRA scaling factor

    Returns:
        lora_delta: (N, out_dim) scaled LoRA contribution
    """
    # Cast to input dtype while preserving gradients
    lora_B_cast = lora_B.to(inputs.dtype)
    lora_A_cast = lora_A.to(inputs.dtype)

    # Use torch._grouped_mm when available/supported; otherwise fall back to a safe loop.
    if _TORCH_GROUPED_MM_AVAILABLE and _check_torch_grouped_mm_supported():
        # Step 1: X @ B -> (N, R)
        lora_intermediate = native_moe_grouped_mm(inputs, lora_B_cast, offsets)

        # Step 2: result @ A -> (N, out_dim)
        lora_delta = native_moe_grouped_mm(lora_intermediate, lora_A_cast, offsets)

        return lora_delta * scaling

    # Fallback to pure pytorch matmul if grouped_mm not available
    # We avoid iterations on experts as requested.
    raise RuntimeError("torch._grouped_mm is required for MoE LoRA training but is not available.")


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

    # Base GEMM (frozen weights - can use Triton for speed)
    mm1_out = native_moe_grouped_mm(permuted_input, w1, offsets)

    # LoRA (extract fresh each forward - weights change during training)
    # CRITICAL: Use torch._grouped_mm for LoRA to ensure gradient flow
    gate_up_wrapper = _get_lora_wrapper_for_param(self, 'gate_up_proj')
    if gate_up_wrapper is not None:
        lora_data = _get_moe_lora_weights(gate_up_wrapper)
        if lora_data is not None:
            lora_B, lora_A, scaling = lora_data[:3]
            lora_delta = _apply_lora_grouped_mm(permuted_input, lora_B, lora_A, offsets, scaling)
            mm1_out = mm1_out + lora_delta  # Use + instead of add_ for cleaner gradient flow

    # Activation
    gate, up = mm1_out.chunk(2, dim=-1)
    inter = F.silu(gate) * up

    # 4. Down Proj
    w2 = self.down_proj.transpose(-2, -1)

    # Base GEMM (frozen weights - can use Triton for speed)
    mm2_out = native_moe_grouped_mm(inter, w2, offsets)

    # LoRA (extract fresh each forward - weights change during training)
    # CRITICAL: Use torch._grouped_mm for LoRA to ensure gradient flow
    down_wrapper = _get_lora_wrapper_for_param(self, 'down_proj')
    if down_wrapper is not None:
        lora_data = _get_moe_lora_weights(down_wrapper)
        if lora_data is not None:
            lora_B, lora_A, scaling = lora_data[:3]
            lora_delta = _apply_lora_grouped_mm(inter, lora_B, lora_A, offsets, scaling)
            mm2_out = mm2_out + lora_delta  # Use + instead of add_ for cleaner gradient flow

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

    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


# Removed unused Triton and Loop implementations to prevent accidental usage.

# -----------------------------------------------------------------------------
# Compatibility shims (kept for imports from temporary_patches/qwen3_moe.py)
# -----------------------------------------------------------------------------

def forward_triton_grouped_gemm(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Backwards-compatible entrypoint expected by qwen3_moe patch code.
    We intentionally route to the native grouped_mm path to keep MoE-LoRA training correct.
    """
    return forward_native_grouped_mm(self, hidden_states, top_k_index, top_k_weights)


@torch.compiler.disable
def forward_native_moe_loop(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Backwards-compatible loop entrypoint expected by qwen3_moe patch code.
    We route to the native grouped_mm path (no Python expert loop).
    """
    return forward_native_grouped_mm(self, hidden_states, top_k_index, top_k_weights)
