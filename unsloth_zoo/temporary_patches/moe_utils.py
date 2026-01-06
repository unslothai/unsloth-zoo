
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


# Global flag to check if grouped GEMM is available
_GROUPED_GEMM_AVAILABLE = None
_TORCH_GROUPED_MM_AVAILABLE = hasattr(torch, "_grouped_mm")

# Check if GPU supports torch._grouped_mm (requires compute capability >= 9.0)
_TORCH_GROUPED_MM_SUPPORTED = None

def _check_torch_grouped_mm_supported():
    """
    Check if torch._grouped_mm is actually supported on the current GPU.
    It requires CUDA compute capability >= 9.0 (H100 and newer).
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
        # Get compute capability of current device
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major + minor / 10.0
        # torch._grouped_mm requires compute capability >= 9.0 (H100)
        _TORCH_GROUPED_MM_SUPPORTED = compute_capability >= 9.0
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


def _silu_and_mul(x):
    """Fused SiLU activation and element-wise multiply for gate/up projections."""
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


def forward_native_grouped_mm(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Native Pytorch grouped GEMM MoE forward pass.
    Uses torch._grouped_mm which is significantly faster than loop and works without Triton dependencies.
    Requires CUDA compute capability >= 9.0 (H100 and newer).
    """
    # Runtime safety check - defense in depth
    if not _check_torch_grouped_mm_supported():
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        raise RuntimeError(
            f"torch._grouped_mm requires CUDA compute capability >= 9.0 (H100), "
            f"but current device has {major}.{minor}. "
            f"Set UNSLOTH_MOE_BACKEND='unsloth_triton' or 'native_torch' to use a compatible backend."
        )


    if hidden_states.dim() == 2:
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

    # gate_up_proj is [E, 2*H, H]. We transpose to [E, H, 2*H] for x @ w
    if hasattr(self, "gate_up_proj"):
        # Handle different weight shapes (e.g. Qwen3 vs Qwen3VL)
        # If dim 1 matches hidden state dim, no transpose needed.
        if self.gate_up_proj.shape[1] == hidden_dim:
            w1 = self.gate_up_proj
        else:
            w1 = self.gate_up_proj.transpose(-2, -1)
        mm1_out = torch._grouped_mm(permuted_input, w1, offs=offsets)
        gate, up = mm1_out.chunk(2, dim=-1)
    elif hasattr(self, "w1") and hasattr(self, "w3"):
        w1 = self.w1.transpose(-2, -1)
        w3 = self.w3.transpose(-2, -1)
        gate = torch._grouped_mm(permuted_input, w1, offs=offsets)
        up = torch._grouped_mm(permuted_input, w3, offs=offsets)
    else:
        raise AttributeError("MoE layer must have 'gate_up_proj' or 'w1'/'w3'.")

    # Activation
    inter = F.silu(gate) * up

    # Grouped GEMM 2
    if hasattr(self, "down_proj"):
        if self.down_proj.shape[1] == inter.shape[-1]:
            w2 = self.down_proj
        else:
            w2 = self.down_proj.transpose(-2, -1)
    elif hasattr(self, "w2"):
        w2 = self.w2.transpose(-2, -1)
    else:
        raise AttributeError("MoE layer must have 'down_proj' or 'w2'.")

    mm2_out = torch._grouped_mm(inter, w2, offs=offsets)

    # 5. Apply Routing Weights
    flat_weights = top_k_weights.view(-1)
    permuted_weights = flat_weights[sorted_indices]
    mm2_out = mm2_out * permuted_weights.unsqueeze(-1)

    # 6. Scatter Add (Reduce)
    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device
    )

    final_hidden_states.index_add_(0, token_indices, mm2_out.to(hidden_states.dtype))

    if hidden_states.dim() == 2:
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
    final_hidden_states = (
        second_gemm_output.view(num_tokens, top_k, hidden_dim)
        * top_k_weights[..., None]
    )
    final_hidden_states = final_hidden_states.sum(dim=1)

    if is_3d:
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)

    return final_hidden_states
