
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
    try:
        if not hasattr(wrapper, 'lora_A') or not hasattr(wrapper, 'lora_B'):
            return None

        if hasattr(wrapper, 'disable_adapters') and wrapper.disable_adapters:
            return None
        if hasattr(wrapper, 'merged') and wrapper.merged:
            return None

        if not wrapper.lora_A:
            return None

        if adapter_name not in wrapper.lora_A:
            adapter_name = list(wrapper.lora_A.keys())[0]

        lora_A_module = wrapper.lora_A[adapter_name]
        lora_B_module = wrapper.lora_B[adapter_name]
        # PEFT stores: A.weight = (E*R, out_dim), B.weight = (in_dim, E*R)
        # where delta = B @ A = (in_dim, E*R) @ (E*R, out_dim) = (in_dim, out_dim)
        weight_A = lora_A_module.weight  # (E*R, out_dim)
        weight_B = lora_B_module.weight  # (in_dim, E*R)
        scaling = wrapper.scaling[adapter_name]
        num_experts = getattr(wrapper, 'num_experts', 1)

        if num_experts > 1:
            rank_per_expert = weight_A.shape[0] // num_experts
            in_dim = weight_B.shape[0]   # input dimension for first step
            out_dim = weight_A.shape[1]  # output dimension for second step

            # Reshape B for first step: X @ B where X is (N, in_dim)
            # B.weight: (in_dim, E*R) -> reshape to (in_dim, E, R) -> permute to (E, in_dim, R)
            B_reshaped = weight_B.reshape(in_dim, num_experts, rank_per_expert)
            B_reshaped = B_reshaped.permute(1, 0, 2).contiguous()  # (E, in_dim, R)

            # Reshape A for second step: result @ A where result is (N, R)
            # A.weight: (E*R, out_dim) -> reshape to (E, R, out_dim)
            A_reshaped = weight_A.reshape(num_experts, rank_per_expert, out_dim)  # (E, R, out_dim)
        else:
            B_reshaped = weight_B.T  # (E*R, in_dim)
            A_reshaped = weight_A    # (E*R, out_dim)

        # Return (B, A) for application order: (X @ B) @ A
        return B_reshaped, A_reshaped, scaling, num_experts
    except Exception:
        return None


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
    out = torch._grouped_mm(inputs, weight, offs=offsets)

    # 2. Add Separated LoRA: + ((X @ B) @ A) * scaling
    if lora_data is not None:
        lora_B, lora_A, scaling = lora_data[:3]  # lora_B is (E, in, R), lora_A is (E, R, out)

        # Cast to input dtype (LoRA weights are float32, input may be bfloat16)
        lora_B = lora_B.to(inputs.dtype)
        lora_A = lora_A.to(inputs.dtype)

        # Step 1: inputs @ B
        # inputs: (N, in), lora_B: (E, in, R) -> lora_out: (N, R)
        lora_out = torch._grouped_mm(inputs, lora_B, offs=offsets)

        # Step 2: result @ A
        # lora_out: (N, R), lora_A: (E, R, out) -> lora_delta: (N, out)
        lora_delta = torch._grouped_mm(lora_out, lora_A, offs=offsets)

        # Add scaled LoRA contribution
        out = out + lora_delta * scaling

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
        # MoE experts: bypass PEFT's _activate_lora completely
        # The LoRA is extracted lazily in forward_native_grouped_mm via the weakref
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
                # Store reference to ParamWrapper for lazy LoRA extraction
                # forward_native_grouped_mm will use this to get LoRA weights
                base_layer.__dict__[f'{param_name}_lora_wrapper'] = module

        return True
    except ImportError:
        return False


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
    mm1_out = torch._grouped_mm(permuted_input, w1, offs=offsets)

    # LoRA (extract fresh each forward - weights change during training)
    gate_up_wrapper = self.__dict__.get('gate_up_proj_lora_wrapper')
    if gate_up_wrapper is not None:
        lora_data = _get_moe_lora_weights(gate_up_wrapper)
        if lora_data is not None:
            lora_B, lora_A, scaling = lora_data[:3]
            lora_out = torch._grouped_mm(permuted_input, lora_B.to(permuted_input.dtype), offs=offsets)
            lora_delta = torch._grouped_mm(lora_out, lora_A.to(permuted_input.dtype), offs=offsets)
            mm1_out.add_(lora_delta, alpha=scaling)

    # Activation
    gate, up = mm1_out.chunk(2, dim=-1)
    inter = F.silu(gate) * up

    # 4. Down Proj
    w2 = self.down_proj.transpose(-2, -1)

    mm2_out = torch._grouped_mm(inter, w2, offs=offsets)

    # LoRA (extract fresh each forward - weights change during training)
    down_wrapper = self.__dict__.get('down_proj_lora_wrapper')
    if down_wrapper is not None:
        lora_data = _get_moe_lora_weights(down_wrapper)
        if lora_data is not None:
            lora_B, lora_A, scaling = lora_data[:3]
            lora_out = torch._grouped_mm(inter, lora_B.to(inter.dtype), offs=offsets)
            lora_delta = torch._grouped_mm(lora_out, lora_A.to(inter.dtype), offs=offsets)
            mm2_out.add_(lora_delta, alpha=scaling)

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
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        else:
            gate = F.linear(current_state, self.w1[expert_idx])
            up   = F.linear(current_state, self.w3[expert_idx])

        current_hidden_states = F.silu(gate) * up

        # Compute down projection for this expert only
        if hasattr(self, "down_proj"):
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
        else:
            current_hidden_states = F.linear(current_hidden_states, self.w2[expert_idx])

        # Apply routing weights
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

        # Scatter back to final output
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states
