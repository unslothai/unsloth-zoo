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
import torch
import torch.nn.functional as F
import os
import shutil
import sys
import importlib
import importlib.util
from typing import Optional, Tuple
from torch.autograd import Function
from unsloth_zoo.mlx import is_mlx_available

UNSLOTH_COMPILE_LOCATION = os.environ.get(
    "UNSLOTH_COMPILE_LOCATION", "unsloth_compiled_cache"
)

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Params4bit
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    Params4bit = None


def _get_compile_location() -> str:
    return os.path.abspath(
        os.environ.get("UNSLOTH_COMPILE_LOCATION", UNSLOTH_COMPILE_LOCATION)
    )


def _log_info(message: str):
    if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
        print(message)


def install_to_cache(source_path, destination_filename=None):
    """Copy a file into unsloth_compiled_cache so compiled modules can use it."""
    compile_location = _get_compile_location()
    if not os.path.exists(compile_location):
        try:
            os.makedirs(compile_location)
        except:
            pass

    current_file = os.path.abspath(source_path)
    if destination_filename is None:
        destination_filename = os.path.basename(current_file)

    destination = os.path.abspath(os.path.join(compile_location, destination_filename))

    if current_file != destination:
        try:
            shutil.copy(current_file, destination)
        except Exception:
            pass


install_to_cache(__file__, "moe_utils.py")

_CACHED_FORWARD_MOE_BACKEND = None
_CACHED_MOE_UTILS_MODULE = None


def _load_cached_moe_utils_module():
    global _CACHED_MOE_UTILS_MODULE

    cache_file = os.path.abspath(os.path.join(_get_compile_location(), "moe_utils.py"))
    current_file = os.path.abspath(__file__)
    if not os.path.isfile(cache_file) or cache_file == current_file:
        return None

    try:
        module_name = "unsloth_cached_moe_utils"
        module = sys.modules.get(module_name, None)
        if module is not None and os.path.abspath(getattr(module, "__file__", "")) == cache_file:
            _CACHED_MOE_UTILS_MODULE = module
            return module

        spec = importlib.util.spec_from_file_location(module_name, cache_file)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        _CACHED_MOE_UTILS_MODULE = module
        return module
    except Exception:
        return None


def get_forward_moe_backend():
    """Resolve forward_moe_backend from the compiled cache copy, else the local def."""
    global _CACHED_FORWARD_MOE_BACKEND
    module = _load_cached_moe_utils_module()
    if module is not None and hasattr(module, "forward_moe_backend"):
        _CACHED_FORWARD_MOE_BACKEND = module.forward_moe_backend
        return _CACHED_FORWARD_MOE_BACKEND

    _CACHED_FORWARD_MOE_BACKEND = forward_moe_backend
    return _CACHED_FORWARD_MOE_BACKEND

# Grouped MM wrapper around torch._grouped_mm; native backward works correctly.


def _grouped_mm_with_backward_fix(
    inputs: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Grouped matmul via torch._grouped_mm with contiguous inputs.

    Some low-rank LoRA weights are contiguous but still have row strides below
    the kernel alignment requirement, so keep a narrow fallback for those cases.
    """
    inputs = inputs.contiguous()
    weight = weight.contiguous()
    try:
        return torch._grouped_mm(inputs, weight, offs=offsets)
    except RuntimeError as exc:
        message = str(exc)
        if "strides should be multiple of 16 bytes" not in message:
            raise
        return _manual_grouped_mm(inputs, weight, offsets)


def _manual_grouped_mm(
    inputs: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Differentiable grouped matmul fallback for torch._grouped_mm alignment gaps."""
    outputs = []
    start = 0
    for expert_idx, end in enumerate(offsets.detach().cpu().tolist()):
        if start < end:
            outputs.append(torch.matmul(inputs[start:end], weight[expert_idx]))
        start = end
    if outputs:
        return torch.cat(outputs, dim=0)
    return inputs.new_empty((0, weight.shape[-1]))


_GROUPED_GEMM_AVAILABLE = None
_TORCH_GROUPED_MM_AVAILABLE = hasattr(torch, "_grouped_mm")

# GPU support for torch._grouped_mm, verified via runtime probe.
_TORCH_GROUPED_MM_SUPPORTED = None


def _check_torch_grouped_mm_supported():
    """Check torch._grouped_mm support on the current GPU; a runtime probe is the only reliable check."""
    global _TORCH_GROUPED_MM_SUPPORTED
    if _TORCH_GROUPED_MM_SUPPORTED is not None: return _TORCH_GROUPED_MM_SUPPORTED

    if not _TORCH_GROUPED_MM_AVAILABLE:
        _TORCH_GROUPED_MM_SUPPORTED = False
        return False

    if not torch.cuda.is_available():
        _TORCH_GROUPED_MM_SUPPORTED = False
        return False

    try:
        # Dummy call verifies real support (symbol may exist but hardware unsupported, e.g. < H100).
        device = torch.cuda.current_device()
        dtype = torch.float16

        # 1 expert, 1 token, dim 8 (safe alignment).
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
_original_peft_get_peft_model = None


def _init_triton_allocator():
    """Initialize a persistent Triton allocator to avoid per-call allocation overhead."""
    global _TRITON_ALLOCATOR_INITIALIZED, _PERSISTENT_BUFFER
    if _TRITON_ALLOCATOR_INITIALIZED: return

    try:
        import triton

        # Persistent buffer that grows as needed, avoiding per-kernel allocations.
        def persistent_alloc_fn(size: int, alignment: int, stream):
            global _PERSISTENT_BUFFER
            # Round up to nearest 128 bytes for alignment / fewer reallocations.
            rounded_size = ((size + 128 - 1) // 128) * 128

            if (
                _PERSISTENT_BUFFER is None
                or _PERSISTENT_BUFFER.numel() * _PERSISTENT_BUFFER.element_size()
                < rounded_size
            ):
                # 10% headroom; uint8 for raw byte storage.
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
    if os.environ.get("UNSLOTH_DISABLE_MOE_TRITON", "0") == "1": return False
    if is_mlx_available(): return False

    global _GROUPED_GEMM_AVAILABLE
    if _GROUPED_GEMM_AVAILABLE is not None: return _GROUPED_GEMM_AVAILABLE

    try:
        from unsloth.kernels.moe.grouped_gemm.interface import grouped_gemm, supports_tma
        _GROUPED_GEMM_AVAILABLE = True
        _init_triton_allocator()
    except (ImportError, ModuleNotFoundError):
        _GROUPED_GEMM_AVAILABLE = False
    return _GROUPED_GEMM_AVAILABLE


from functools import lru_cache, wraps


@lru_cache(maxsize=1)
def select_moe_backend():
    """Select MoE backend from UNSLOTH_MOE_BACKEND + availability.

    Choices: "grouped_mm", "unsloth_triton", "native_torch" (default "grouped_mm").
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    requested = os.environ.get("UNSLOTH_MOE_BACKEND")
    if requested:
        if requested == "grouped_mm" and _check_torch_grouped_mm_supported():
            return "grouped_mm"
        if requested == "unsloth_triton" and _check_grouped_gemm_available():
            return "unsloth_triton"
        if requested == "native_torch":
            return "native_torch"
        _log_info(f"Unsloth: '{requested}' backend requested but is not available. Falling back to next available.")

    if _check_torch_grouped_mm_supported():
        _log_info("Unsloth: Using MoE backend 'grouped_mm'")
        return "grouped_mm"
    if _check_grouped_gemm_available():
        _log_info("Unsloth: Using MoE backend 'unsloth_triton'")
        return "unsloth_triton"
    return "native_torch"


def swap_moe_weights_for_call(experts_module, gate_up_proj, down_proj, forward_fn, *args):
    """Temporarily install dequantized weights for one forward call, then restore.

    Uses object.__setattr__ to bypass nn.Module Parameter (de)registration
    (re-registers hooks, unnecessary for read-only temp tensors). Used by the
    FP8 and bnb4bit MoE dispatchers.
    """
    original_gate_up = experts_module.gate_up_proj
    original_down = experts_module.down_proj
    object.__setattr__(experts_module, "gate_up_proj", gate_up_proj)
    object.__setattr__(experts_module, "down_proj", down_proj)
    try:
        return forward_fn(experts_module, *args)
    finally:
        object.__setattr__(experts_module, "gate_up_proj", original_gate_up)
        object.__setattr__(experts_module, "down_proj", original_down)


def forward_moe_backend(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Dispatch MoE forward to the selected backend (keeps model-specific patches minimal)."""
    # This Unsloth Zoo code section is licensed under AGPL3

    # Absolute imports: this function is also copied into
    # unsloth_compiled_cache/moe_utils.py where relative imports of sibling
    # helpers don't resolve (only the dispatcher is copied).
    # Keep `except ImportError` around ONLY the import; runtime errors in the
    # bnb4bit/fp8 path must propagate, not fall through to a crashing backend.
    _moe_uses_bnb4bit_expert_weights = forward_moe_backend_bnb4bit = None
    try:
        from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import (
            _moe_uses_bnb4bit_expert_weights,
            forward_moe_backend_bnb4bit,
        )
    except ImportError:
        pass
    if _moe_uses_bnb4bit_expert_weights is not None and _moe_uses_bnb4bit_expert_weights(self):
        result = forward_moe_backend_bnb4bit(self, hidden_states, top_k_index, top_k_weights)
        if result is not None:
            return result

    _moe_uses_fp8_expert_weights = forward_moe_backend_fp8 = None
    try:
        from unsloth_zoo.temporary_patches.moe_utils_fp8 import (
            _moe_uses_fp8_expert_weights,
            forward_moe_backend_fp8,
        )
    except ImportError:
        pass
    if _moe_uses_fp8_expert_weights is not None and _moe_uses_fp8_expert_weights(self):
        return forward_moe_backend_fp8(self, hidden_states, top_k_index, top_k_weights)

    backend = select_moe_backend()
    if backend == "grouped_mm":
        return forward_native_grouped_mm(self, hidden_states, top_k_index, top_k_weights)
    if backend == "unsloth_triton":
        return forward_triton_grouped_gemm(self, hidden_states, top_k_index, top_k_weights)
    return forward_native_moe_loop(self, hidden_states, top_k_index, top_k_weights)


@torch.no_grad()
def _get_routing_indices(selected_experts, num_experts):
    """Compute token->expert mapping for grouped GEMM.

    Returns (token_counts_by_expert (num_experts,), gather_indices (total_tokens,)).
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    flat_experts = selected_experts.view(-1)

    # bincount avoids histc's float conversion overhead.
    token_counts_by_expert = torch.bincount(flat_experts, minlength=num_experts).to(torch.int32)

    # stable=True preserves order within each expert.
    gather_indices = flat_experts.argsort(stable=True)

    return token_counts_by_expert, gather_indices


def _silu_and_mul(x):
    """Fused SiLU + element-wise multiply for gate/up projections."""
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


# Separated LoRA helpers.


def _has_lora_adapters(param) -> bool:
    """Check for active LoRA adapters (PEFT ParamWrapper)."""
    if not hasattr(param, "lora_A") or not hasattr(param, "lora_B"):
        return False
    if hasattr(param, "disable_adapters") and param.disable_adapters:
        return False
    if hasattr(param, "merged") and param.merged:
        return False
    return len(param.lora_A) > 0


def _canonical_lora_weights_for_grouped_mm(
    weight_A: torch.Tensor,
    weight_B: torch.Tensor,
    num_experts: int,
    rank_per_expert: int,
    dim_A: int,
    dim_B: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    first_weight = weight_A.view(num_experts, rank_per_expert, dim_A)
    first_weight = first_weight.permute(0, 2, 1).contiguous()
    second_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
    second_weight = second_weight.permute(1, 2, 0).contiguous()
    return first_weight, second_weight


def _reversed_lora_weights_for_grouped_mm(
    weight_A: torch.Tensor,
    weight_B: torch.Tensor,
    num_experts: int,
    rank_per_expert: int,
    dim_A: int,
    dim_B: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    first_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
    first_weight = first_weight.permute(1, 0, 2).contiguous()
    second_weight = weight_A.view(num_experts, rank_per_expert, dim_A).contiguous()
    return first_weight, second_weight


def _get_param_shape_from_module(module, parameter_name):
    if module is None or parameter_name is None or not hasattr(module, parameter_name):
        return None
    param = getattr(module, parameter_name)
    if hasattr(param, "get_param"):
        param = param.get_param()
    elif hasattr(param, "weight"):
        param = param.weight
    return tuple(param.shape)


def _get_moe_lora_io_dims(wrapper, experts_module=None):
    base = None
    if wrapper is not None and hasattr(wrapper, "get_base_layer"):
        base = wrapper.get_base_layer()
    if experts_module is None:
        experts_module = base
    if experts_module is None:
        experts_module = getattr(wrapper, "base_layer", None)

    parameter_name = getattr(wrapper, "parameter_name", None)
    source = experts_module if experts_module is not None else base
    if source is None:
        return None, None
    _set_gpt_oss_grouped_mm_format_on_experts(source)

    shape = _get_param_shape_from_module(source, parameter_name)
    if shape is not None and len(shape) >= 3:
        grouped_mm_format = bool(getattr(source, "_unsloth_grouped_mm_format", False))
        if grouped_mm_format:
            return shape[-2], shape[-1]
        return shape[-1], shape[-2]

    hidden_dim = getattr(source, "hidden_dim", None)
    intermediate_dim = getattr(source, "intermediate_dim", None)
    if hidden_dim is None or intermediate_dim is None:
        return None, None
    if parameter_name == "gate_up_proj":
        return hidden_dim, 2 * intermediate_dim
    if parameter_name == "down_proj":
        return intermediate_dim, hidden_dim
    return None, None


def extract_moe_lora_weights_for_grouped_mm(
    wrapper,
    weight_A: torch.Tensor,
    weight_B: torch.Tensor,
    scaling,
    num_experts: int,
    *,
    experts_module=None,
    input_dim=None,
    output_dim=None,
    model_name: str = "MoE",
    enable_logging: bool = None,
    logger_obj=None,
) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
    total_rank = weight_A.shape[0]
    rank_per_expert = total_rank // num_experts
    dim_A = weight_A.shape[1]
    dim_B = weight_B.shape[0]

    if num_experts <= 1:
        return weight_A.T, weight_B.T, scaling, num_experts

    if input_dim is None or output_dim is None:
        inferred_input_dim, inferred_output_dim = _get_moe_lora_io_dims(
            wrapper, experts_module=experts_module,
        )
        if input_dim is None:
            input_dim = inferred_input_dim
        if output_dim is None:
            output_dim = inferred_output_dim

    canonical_match = (
        input_dim is not None
        and output_dim is not None
        and dim_A == input_dim
        and dim_B == output_dim
    )
    reversed_match = (
        input_dim is not None
        and output_dim is not None
        and dim_A == output_dim
        and dim_B == input_dim
    )

    if canonical_match and reversed_match:
        if bool(getattr(wrapper, "_did_swap_in_out_features", False)):
            first_weight, second_weight = _reversed_lora_weights_for_grouped_mm(
                weight_A, weight_B, num_experts, rank_per_expert, dim_A, dim_B,
            )
        else:
            first_weight, second_weight = _canonical_lora_weights_for_grouped_mm(
                weight_A, weight_B, num_experts, rank_per_expert, dim_A, dim_B,
            )
        return first_weight, second_weight, scaling, num_experts

    if canonical_match:
        first_weight, second_weight = _canonical_lora_weights_for_grouped_mm(
            weight_A, weight_B, num_experts, rank_per_expert, dim_A, dim_B,
        )
        return first_weight, second_weight, scaling, num_experts

    if reversed_match:
        first_weight, second_weight = _reversed_lora_weights_for_grouped_mm(
            weight_A, weight_B, num_experts, rank_per_expert, dim_A, dim_B,
        )
        return first_weight, second_weight, scaling, num_experts

    if logger_obj is not None:
        if enable_logging is None:
            enable_logging = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
        if enable_logging and (input_dim is not None or output_dim is not None):
            logger_obj.warning(
                f"Unsloth: {model_name} LoRA extractor could not match either layout "
                f"(weight_A={tuple(weight_A.shape)}, weight_B={tuple(weight_B.shape)}, "
                f"expected input_dim={input_dim}, output_dim={output_dim}, "
                f"num_experts={num_experts}). Falling back to canonical layout. "
                "If this is a new PEFT version, the LoRA delta may be wrong."
        )

    first_weight, second_weight = _canonical_lora_weights_for_grouped_mm(
        weight_A, weight_B, num_experts, rank_per_expert, dim_A, dim_B,
    )
    return first_weight, second_weight, scaling, num_experts


def _extract_lora_from_wrapper(
    wrapper, adapter_name: str = "default", experts_module=None
) -> Optional[Tuple[torch.Tensor, torch.Tensor, float, int]]:
    """Extract LoRA weights from a PEFT ParamWrapper for MoE separated grouped_mm.

    PEFT 3D ParamWrapper gives lora_A: (E*R, in_dim), lora_B: (out_dim, E*R);
    reshaped to first_weight (E, in_dim, R), second_weight (E, R, out_dim) so
    delta = X @ first @ second. Handles both standard (E, out, in) Qwen3-MoE and
    transposed (E, in, out) Qwen3-VL-MoE base weight layouts.

    Returns (first_weight, second_weight, scaling, num_experts) or None.
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    try:
        if not hasattr(wrapper, "lora_A") or not hasattr(wrapper, "lora_B"):
            return None

        if hasattr(wrapper, "disable_adapters") and wrapper.disable_adapters:
            return None
        if hasattr(wrapper, "merged") and wrapper.merged:
            return None

        if not wrapper.lora_A:
            return None

        if adapter_name not in wrapper.lora_A:
            adapter_name = list(wrapper.lora_A.keys())[0]

        lora_A_module = wrapper.lora_A[adapter_name]
        lora_B_module = wrapper.lora_B[adapter_name]

        weight_A = lora_A_module.weight  # (E*R, dim1)
        weight_B = lora_B_module.weight  # (dim2, E*R)
        scaling = wrapper.scaling[adapter_name]
        num_experts = getattr(wrapper, "num_experts", 1)

        if experts_module is None:
            experts_module = wrapper.get_base_layer() if hasattr(wrapper, "get_base_layer") else None

        # Model-specific LoRA extractor attached to the experts module, if any.
        extractor_fn = getattr(experts_module, "_unsloth_lora_extractor_fn", None)

        if extractor_fn is not None:
            return extractor_fn(wrapper, weight_A, weight_B, scaling, num_experts)

        return extract_moe_lora_weights_for_grouped_mm(
            wrapper,
            weight_A,
            weight_B,
            scaling,
            num_experts,
            experts_module=experts_module,
            model_name="MoE",
        )
    except Exception:
        return None


def _extract_lora_weights(
    param, adapter_name: str = "default", num_experts: int = None, experts_module=None
) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
    """Compat wrapper around _extract_lora_from_wrapper; returns (first, second, scaling)."""
    # This Unsloth Zoo code section is licensed under AGPL3

    # Pass num_experts through so _extract_lora_from_wrapper can use it.
    if num_experts is not None and not hasattr(param, "num_experts"):
        param.num_experts = num_experts

    result = _extract_lora_from_wrapper(param, adapter_name, experts_module=experts_module)
    if result is None:
        return None
    return result[0], result[1], result[2]


def _get_base_weight(param):
    """Get base weight from a potentially wrapped parameter or module."""
    # This Unsloth Zoo code section is licensed under AGPL3

    while hasattr(param, "base_layer"):
        param = param.base_layer

    if HAS_BNB and isinstance(param, Params4bit):
        if getattr(param, "quant_state", None) is None:
            raise RuntimeError(
                "unsloth: _get_base_weight saw a Params4bit with quant_state=None. "
                "This usually means the model was used in forward before loading "
                "completed quantization (meta placeholder still in place), or the "
                "MoE quantizer patch did not fire for this expert. "
                f"data.shape={tuple(param.data.shape)}, device={param.device}."
            )
        return bnb.functional.dequantize_4bit(param.data, param.quant_state)

    if hasattr(param, "get_param"):
        return param.get_param()

    if hasattr(param, "weight"):
        return param.weight

    return param


def _get_lora_wrapper_for_param(experts_module, param_name):
    """Get the PEFT ParamWrapper for gate_up_proj or down_proj; does not lazily set up wrappers."""
    # This Unsloth Zoo code section is licensed under AGPL3

    if hasattr(experts_module, f"{param_name}_lora_wrapper"):
        return getattr(experts_module, f"{param_name}_lora_wrapper")

    if hasattr(experts_module, param_name):
        attr = getattr(experts_module, param_name)
        if hasattr(attr, "lora_A"):  # ParamWrapper
            return attr

    return None


def native_moe_grouped_mm(
    inputs: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Grouped_mm with backward fix for PyTorch's grouped_mm backward stride bug."""
    return _grouped_mm_with_backward_fix(inputs, weight, offsets)


def _apply_lora_grouped_mm(
    inputs: torch.Tensor,
    lora_B: torch.Tensor,
    lora_A: torch.Tensor,
    offsets: torch.Tensor,
    scaling: float,
    grouped_mm_func=native_moe_grouped_mm,
) -> torch.Tensor:
    """Apply LoRA via grouped GEMM: result = ((X @ B) @ A) * scaling.

    inputs (total_tokens, in_dim); lora_B (E, in_dim, R); lora_A (E, R, out_dim).
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    # X @ B then result @ A; both already in native (E, ...) layout, no transpose.
    lora_intermediate = grouped_mm_func(inputs, lora_B.contiguous(), offsets)
    lora_delta = grouped_mm_func(lora_intermediate, lora_A.contiguous(), offsets)

    return lora_delta * scaling


def _should_use_separated_lora() -> bool:
    """Use separated LoRA (default True); UNSLOTH_MOE_LORA_MERGED=1 forces the merged path."""
    return os.environ.get("UNSLOTH_MOE_LORA_MERGED", "0") != "1"


# Model-specific weight preprocessing hooks: each model registers a transposition
# function so the generic backend works across weight layouts.

_WEIGHT_PREPROCESSORS = {}


def register_weight_preprocessor(model_type: str, preprocessor_fn):
    """Register a weight preprocessor (weight, proj_type, hidden_dim) -> weight for a model type."""
    _WEIGHT_PREPROCESSORS[model_type] = preprocessor_fn


def get_weight_preprocessor(model_type: str):
    """Get registered weight preprocessor for model type."""
    return _WEIGHT_PREPROCESSORS.get(model_type)


def preprocess_weight(
    weight: torch.Tensor, proj_type: str, hidden_dim: int, model_type=None
):
    """Preprocess a weight into (E, in_dim, out_dim) for grouped_mm.

    Uses a registered model-specific preprocessor if present, else transposes
    by shape. proj_type is "gate_up" or "down".
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    if model_type and model_type in _WEIGHT_PREPROCESSORS:
        return _WEIGHT_PREPROCESSORS[model_type](weight, proj_type, hidden_dim)

    if proj_type == "gate_up":
        # Want (E, hidden_dim, 2*intermediate).
        if weight.shape[1] == hidden_dim:
            return weight
        else:
            return weight.transpose(-2, -1)
    else:  # down
        # Want (E, intermediate, hidden_dim).
        if weight.shape[2] == hidden_dim:
            return weight
        else:
            return weight.transpose(-2, -1)


# Generic MoE detection and ParamWrapper patching.


def _normalize_model_type(value) -> str:
    if value is None:
        return ""
    return str(value).lower().replace("-", "_")


def _iter_model_configs(model):
    seen = set()
    queue = [model]
    while queue and len(seen) < 8:
        current = queue.pop(0)
        if current is None:
            continue
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        config = getattr(current, "config", None)
        if config is not None:
            yield config

        for attr in ("base_model", "model"):
            nested = getattr(current, attr, None)
            if nested is not None and nested is not current:
                queue.append(nested)


def _is_gpt_oss_model(model) -> bool:
    for config in _iter_model_configs(model):
        model_type = _normalize_model_type(getattr(config, "model_type", None))
        if model_type == "gpt_oss":
            return True

        for attr in ("_name_or_path", "name_or_path"):
            name = getattr(config, attr, None)
            if name is None:
                continue
            # Match only the final path component so parent directories like
            # /data/gpt-oss-tests/qwen3-7b do not count as gpt-oss.
            base = str(name).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            if "gpt_oss" in _normalize_model_type(base):
                return True

    return False


def _set_gpt_oss_grouped_mm_format_on_experts(module) -> bool:
    if module is None:
        return False
    if module.__class__.__name__ != "GptOssExperts":
        return False
    if bool(getattr(module, "_unsloth_grouped_mm_format", False)):
        return False
    # Require the gpt-oss (E, in, out) weight signature: gate_up's out dim is
    # twice down's in dim. Same-named classes with other layouts stay unflagged.
    gate_shape = _get_param_shape_from_module(module, "gate_up_proj")
    down_shape = _get_param_shape_from_module(module, "down_proj")
    if gate_shape is None or down_shape is None:
        return False
    if len(gate_shape) < 3 or len(down_shape) < 3:
        return False
    if gate_shape[0] != down_shape[0]:
        return False
    if gate_shape[-2] != down_shape[-1] or gate_shape[-1] != 2 * down_shape[-2]:
        return False
    module._unsloth_grouped_mm_format = True
    return True


def patch_gpt_oss_grouped_mm_format(model) -> int:
    """
    Mark GPT-OSS experts as storing weights in grouped_mm format.

    Stock transformers GPT-OSS experts use (E, in_dim, out_dim) tensors but do
    not carry Unsloth's `_unsloth_grouped_mm_format` instance flag. Set it on
    live expert modules so the shared MoE LoRA extractor chooses GPT-OSS
    ordering instead of the Qwen-style fallback.
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    if model is None or not _is_gpt_oss_model(model):
        return 0

    modules = getattr(model, "modules", None)
    if not callable(modules):
        return 0

    updated = 0
    for module in modules():
        if _set_gpt_oss_grouped_mm_format_on_experts(module):
            updated += 1
    return updated


def _patch_peft_get_peft_model_for_moe():
    # This Unsloth Zoo code section is licensed under AGPL3

    global _original_peft_get_peft_model
    if _original_peft_get_peft_model is not None:
        return

    try:
        import peft
    except Exception:
        return

    original_get_peft_model = getattr(peft, "get_peft_model", None)
    if original_get_peft_model is None:
        return
    if getattr(original_get_peft_model, "_unsloth_moe_patched", False):
        return

    _original_peft_get_peft_model = original_get_peft_model

    @wraps(original_get_peft_model)
    def patched_get_peft_model(model, *args, **kwargs):
        peft_model = original_get_peft_model(model, *args, **kwargs)
        try:
            patch_gpt_oss_grouped_mm_format(model)
            if peft_model is not model:
                patch_gpt_oss_grouped_mm_format(peft_model)
        except Exception:
            pass
        return peft_model

    patched_get_peft_model._unsloth_moe_patched = True
    peft.get_peft_model = patched_get_peft_model

    for module_name in ("peft.mapping_func", "peft.mapping"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if getattr(module, "get_peft_model", None) is original_get_peft_model:
            module.get_peft_model = patched_get_peft_model


def _is_moe_experts_module(module) -> bool:
    """Generic check for an MoE experts layer with stacked 3D expert weights.

    Matches gate_up_proj/down_proj (Qwen3-MoE etc.) or w1/w2/w3 (older models).
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    import torch.nn as nn

    # After PEFT's parametrize wrapping, gate_up_proj is a Tensor (not Parameter),
    # so accept both.
    if hasattr(module, "gate_up_proj"):
        param = module.gate_up_proj
        # 4-bit params are packed into 2D tensors.
        if HAS_BNB and isinstance(param, Params4bit) and param.ndim == 2:
            return True
        # Standard MoE weights are 3D (num_experts, in, out).
        if isinstance(param, (nn.Parameter, torch.Tensor)) and param.ndim in (2, 3):
            return True

    # w1/w2 pattern (separate gate/up projections).
    if hasattr(module, "w1") and hasattr(module, "w2"):
        w1 = module.w1
        if isinstance(w1, (nn.Parameter, torch.Tensor)) and w1.ndim in (2, 3):
            return True

    return False


# Aliases for compatibility with gpt_oss.py
_get_moe_lora_weights = _extract_lora_from_wrapper


# Store original ParamWrapper.forward for fallback
_original_param_wrapper_forward = None


def _patched_param_wrapper_forward(
    self, x: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    """Patched ParamWrapper.forward for MoE separated LoRA.

    For MoE experts: bypass PEFT's _activate_lora and stash LoRA data by
    parameter_name for forward_native_grouped_mm. For non-MoE: original forward.
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    # Use self.base_layer (immediate parent), NOT get_base_layer() which recurses
    # to the deepest layer; the wrapper chain down_proj -> gate_up_proj ->
    # Qwen3MoeExperts must be preserved.
    immediate_base_layer = self.base_layer

    # For stashing LoRA data we need the actual experts module (recursive lookup).
    experts_module = self.get_base_layer()

    use_separated = _should_use_separated_lora()
    param_name = getattr(self, "parameter_name", None)

    if (
        use_separated
        and param_name in ("gate_up_proj", "down_proj")
        and _is_moe_experts_module(experts_module)
    ):
        # MoE experts: bypass PEFT's _activate_lora, use separated computation.
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return immediate_base_layer(x, *args, **kwargs)

        if self.merged:
            return immediate_base_layer(x, *args, **kwargs)

        # Ensure wrapper.num_experts is set for LoRA weight reshaping.
        if not hasattr(self, "num_experts"):
            if hasattr(experts_module, "num_experts"):
                self.num_experts = experts_module.num_experts
            elif hasattr(experts_module, param_name):
                p = getattr(experts_module, param_name)
                if hasattr(p, "shape") and len(p.shape) >= 1:
                    self.num_experts = p.shape[0]

        # Extract LoRA for this parameter and stash on the experts module
        # (not base_layer): _unsloth_lora_gate_up_proj / _unsloth_lora_down_proj.
        lora_data = _extract_lora_from_wrapper(self)

        if lora_data is not None and param_name:
            lora_attr = f"_unsloth_lora_{param_name}"
            setattr(experts_module, lora_attr, lora_data)

        try:
            # Immediate base_layer preserves the wrapper chain.
            result = immediate_base_layer(x, *args, **kwargs)
        finally:
            if param_name:
                lora_attr = f"_unsloth_lora_{param_name}"
                if hasattr(experts_module, lora_attr):
                    delattr(experts_module, lora_attr)

        return result

    # Non-MoE: original PEFT forward with _activate_lora.
    return _original_param_wrapper_forward(self, x, *args, **kwargs)


def patch_param_wrapper_for_moe():
    """Patch PEFT's ParamWrapper.forward for MoE separated LoRA (call after PEFT import)."""
    # This Unsloth Zoo code section is licensed under AGPL3

    global _original_param_wrapper_forward

    module = _load_cached_moe_utils_module()
    if module is not None and hasattr(module, "patch_param_wrapper_for_moe"):
        try:
            return module.patch_param_wrapper_for_moe()
        except Exception:
            pass

    try:
        from peft.tuners.lora.layer import ParamWrapper

        if _original_param_wrapper_forward is None:
            _original_param_wrapper_forward = ParamWrapper.forward

        ParamWrapper.forward = _patched_param_wrapper_forward
        _patch_peft_get_peft_model_for_moe()

        return True
    except ImportError:
        return False


def forward_native_grouped_mm(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Native PyTorch grouped-GEMM MoE forward via torch._grouped_mm (no Triton; needs runtime support)."""
    # This Unsloth Zoo code section is licensed under AGPL3

    # Runtime safety check (defense in depth).
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

    # Routing: count tokens per expert, sort to group by expert, gather inputs.
    flat_top_k = top_k_index.view(-1)
    num_tokens_per_expert = torch.bincount(flat_top_k, minlength=self.num_experts).int()
    sorted_indices = torch.argsort(flat_top_k, stable=True)
    token_indices = sorted_indices // top_k_index.shape[-1]
    permuted_input = hidden_states[token_indices]
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    # Gate + Up projection with optional separated LoRA (default).
    use_separated_lora = _should_use_separated_lora()
    gate_up_lora = None

    # Prefer LoRA injected by the patched ParamWrapper; fall back to the parameter.
    if getattr(self, "_unsloth_lora_gate_up_proj", None) is not None:
        gate_up_lora = self._unsloth_lora_gate_up_proj[:3]  # (first, second, scaling)
    elif (
        use_separated_lora
        and hasattr(self, "gate_up_proj")
        and _has_lora_adapters(self.gate_up_proj)
    ):
        gate_up_lora = _extract_lora_weights(
            self.gate_up_proj, num_experts=self.num_experts, experts_module=self
        )

    if hasattr(self, "gate_up_proj"):
        gate_up_base = _get_base_weight(self.gate_up_proj)
        model_type = getattr(self, "_unsloth_model_type", None)

        # grouped_mm backward needs contiguous weights; preprocess may return a transposed view.
        w1 = preprocess_weight(gate_up_base, "gate_up", hidden_dim, model_type)
        mm1_out = _grouped_mm_with_backward_fix(permuted_input, w1, offsets)

        # Separated LoRA: + ((X @ first) @ second) * scaling.
        if gate_up_lora is not None:
            first_weight, second_weight, scaling = gate_up_lora

            # Cast to input dtype (LoRA is float32) and make contiguous for grouped_mm.
            first_weight = first_weight.to(permuted_input.dtype).contiguous()
            second_weight = second_weight.to(permuted_input.dtype).contiguous()

            try:
                lora_out = _grouped_mm_with_backward_fix(permuted_input, first_weight, offsets)
                lora_out = lora_out.contiguous()
            except RuntimeError as e:
                raise e

            # Second matmul; pad an unaligned output dim or fall back on failure.
            try:
                if second_weight.shape[-1] % 8 != 0:
                    pad_size = 8 - (second_weight.shape[-1] % 8)
                    second_weight_padded = F.pad(
                        second_weight, (0, pad_size)
                    ).contiguous()
                    lora_delta = _grouped_mm_with_backward_fix(
                        lora_out, second_weight_padded, offsets
                    )
                    lora_delta = lora_delta[:, :-pad_size]
                else:
                    lora_delta = _grouped_mm_with_backward_fix(
                        lora_out, second_weight, offsets
                    )
            except RuntimeError:
                # Manual loop fallback on grouped_mm failure (e.g. stride alignment).
                lora_delta = torch.empty(
                    (lora_out.shape[0], second_weight.shape[-1]),
                    dtype=lora_out.dtype,
                    device=lora_out.device,
                )
                cpu_offsets = offsets.cpu().tolist()
                prev_offset = 0
                for i, end in enumerate(cpu_offsets):
                    if prev_offset < end:
                        lora_delta[prev_offset:end] = torch.matmul(
                            lora_out[prev_offset:end], second_weight[i]
                        )
                    prev_offset = end

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
        # Separate w1/w3 weights (older models).
        w1_base = _get_base_weight(self.w1)
        w3_base = _get_base_weight(self.w3)

        w1 = w1_base.transpose(-2, -1)
        w3 = w3_base.transpose(-2, -1)

        gate = _grouped_mm_with_backward_fix(permuted_input, w1, offsets)
        up = _grouped_mm_with_backward_fix(permuted_input, w3, offsets)

        # Add LoRA for w1 and w3 separately if present.
        if use_separated_lora:
            if _has_lora_adapters(self.w1):
                w1_lora = _extract_lora_weights(self.w1, experts_module=self)
                if w1_lora is not None:
                    lora_A, lora_B, scaling = w1_lora
                    lora_A_t = lora_A.transpose(-2, -1)
                    lora_A_out = _grouped_mm_with_backward_fix(
                        permuted_input, lora_A_t, offsets
                    )
                    lora_B_t = lora_B.transpose(-2, -1)
                    lora_B_out = _grouped_mm_with_backward_fix(lora_A_out, lora_B_t, offsets)
                    gate = gate + lora_B_out * scaling

            if _has_lora_adapters(self.w3):
                w3_lora = _extract_lora_weights(self.w3, experts_module=self)
                if w3_lora is not None:
                    lora_A, lora_B, scaling = w3_lora
                    lora_A_t = lora_A.transpose(-2, -1)
                    lora_A_out = _grouped_mm_with_backward_fix(
                        permuted_input, lora_A_t, offsets
                    )
                    lora_B_t = lora_B.transpose(-2, -1)
                    lora_B_out = _grouped_mm_with_backward_fix(lora_A_out, lora_B_t, offsets)
                    up = up + lora_B_out * scaling
    else:
        raise AttributeError("MoE layer must have 'gate_up_proj' or 'w1'/'w3'.")

    # Activation
    if "GptOssExperts" in self.__class__.__name__:
        # Custom GptOss activation.
        limit = getattr(self, "limit", 7.0)
        alpha = getattr(self, "alpha", 1.702)

        gate = gate.clamp(min=None, max=limit)
        up = up.clamp(min=-limit, max=limit)
        glu = gate * torch.sigmoid(gate * alpha)
        inter = (up + 1.0) * glu
    elif hasattr(self, 'act_fn') and callable(self.act_fn):
        inter = self.act_fn(gate) * up
    else:
        inter = F.silu(gate) * up

    # Down projection with optional separated LoRA (default).
    down_lora = None

    # Prefer LoRA injected by the patched ParamWrapper; fall back to the parameter.
    if getattr(self, "_unsloth_lora_down_proj", None) is not None:
        down_lora = self._unsloth_lora_down_proj[:3]  # (first, second, scaling)
    elif (
        use_separated_lora
        and hasattr(self, "down_proj")
        and _has_lora_adapters(self.down_proj)
    ):
        down_lora = _extract_lora_weights(self.down_proj, num_experts=self.num_experts, experts_module=self)

    if hasattr(self, "down_proj"):
        down_base = _get_base_weight(self.down_proj)
        model_type = getattr(self, "_unsloth_model_type", None)
        w2 = preprocess_weight(down_base, "down", hidden_dim, model_type)
        mm2_out = _grouped_mm_with_backward_fix(inter, w2, offsets)

        if down_lora is not None:
            first_weight, second_weight, scaling = down_lora

            # Cast to input dtype (LoRA is float32) and make contiguous for grouped_mm.
            first_weight = first_weight.to(inter.dtype).contiguous()
            second_weight = second_weight.to(inter.dtype).contiguous()

            lora_out = _grouped_mm_with_backward_fix(inter, first_weight, offsets)
            lora_out = lora_out.contiguous()

            try:
                lora_delta = _grouped_mm_with_backward_fix(lora_out, second_weight, offsets)
            except RuntimeError:
                # Manual loop fallback.
                lora_delta = torch.empty(
                    (lora_out.shape[0], second_weight.shape[-1]),
                    dtype=lora_out.dtype,
                    device=lora_out.device,
                )
                cpu_offsets = offsets.cpu().tolist()
                prev_offset = 0
                for i, end in enumerate(cpu_offsets):
                    if prev_offset < end:
                        lora_delta[prev_offset:end] = torch.matmul(
                            lora_out[prev_offset:end], second_weight[i]
                        )
                    prev_offset = end

            mm2_out = mm2_out + lora_delta * scaling

        if hasattr(self, "down_proj_bias") and self.down_proj_bias is not None:
            bias_expanded = self.down_proj_bias.repeat_interleave(
                num_tokens_per_expert.to(self.down_proj_bias.device), dim=0
            ).to(mm2_out.device)
            mm2_out = mm2_out + bias_expanded.to(mm2_out.dtype)

    elif hasattr(self, "w2"):
        w2_base = _get_base_weight(self.w2)
        w2 = w2_base.transpose(-2, -1)
        mm2_out = _grouped_mm_with_backward_fix(inter, w2, offsets)

        if use_separated_lora and _has_lora_adapters(self.w2):
            w2_lora = _extract_lora_weights(self.w2, experts_module=self)
            if w2_lora is not None:
                lora_A, lora_B, scaling = w2_lora
                lora_A_t = lora_A.transpose(-2, -1).contiguous()
                lora_A_out = _grouped_mm_with_backward_fix(inter, lora_A_t, offsets)
                lora_B_t = lora_B.transpose(-2, -1).contiguous()
                lora_B_out = _grouped_mm_with_backward_fix(lora_A_out, lora_B_t, offsets)
                mm2_out = mm2_out + lora_B_out * scaling
    else:
        raise AttributeError("MoE layer must have 'down_proj' or 'w2'.")

    # Apply routing weights and scatter-add (reduce).
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
    """Grouped-GEMM MoE forward via Triton kernels (torch.compile-compatible, mode="max-autotune")."""
    # This Unsloth Zoo code section is licensed under AGPL3

    from unsloth.kernels.moe.grouped_gemm.interface import grouped_gemm
    from unsloth.kernels.moe.autotune_cache import get_or_autotune_moe_kernels

    if not hasattr(self, "_unsloth_moe_configs"):
        self._unsloth_moe_configs = None

    use_separated_lora = _should_use_separated_lora()

    # gate_up LoRA from the patched ParamWrapper (mirrors the down block below).
    gate_up_lora = None
    if getattr(self, "_unsloth_lora_gate_up_proj", None) is not None:
        gate_up_lora = self._unsloth_lora_gate_up_proj[:3]
    elif (
        use_separated_lora
        and hasattr(self, "gate_up_proj")
        and _has_lora_adapters(self.gate_up_proj)
    ):
        gate_up_lora = _extract_lora_weights(
            self.gate_up_proj, num_experts=self.num_experts
        )

    # Flatten 3D inputs (batch_size, seq_len, hidden_dim).
    is_3d = hidden_states.dim() == 3
    if is_3d:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        num_tokens = batch_size * seq_len
        if top_k_index.dim() == 3:
            top_k_index = top_k_index.view(-1, top_k_index.shape[-1])
        if top_k_weights.dim() == 3:
            top_k_weights = top_k_weights.view(-1, top_k_weights.shape[-1])
    else:
        num_tokens, hidden_dim = hidden_states.shape

    top_k = top_k_index.shape[1]

    # Cache model dims and kernel configs on first call.
    if self._unsloth_moe_configs is None:
        intermediate_dim = self.gate_up_proj.shape[1] // 2

        # Autotune first GEMM.
        gemm1_configs = get_or_autotune_moe_kernels(
            num_experts=self.num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim * 2,
            top_k=top_k,
            dtype=hidden_states.dtype,
        )

        # Autotune second GEMM (output dim is hidden_dim).
        gemm2_configs = get_or_autotune_moe_kernels(
            num_experts=self.num_experts,
            hidden_dim=intermediate_dim,
            intermediate_dim=hidden_dim,
            top_k=top_k,
            dtype=hidden_states.dtype,
        )

        self._unsloth_moe_configs = (intermediate_dim, gemm1_configs, gemm2_configs)
        torch.cuda.empty_cache()

    intermediate_dim, gemm1_configs, gemm2_configs = self._unsloth_moe_configs
    fwd_config_1, bwd_dX_config_1, bwd_dW_config_1 = gemm1_configs
    fwd_config_2, bwd_dX_config_2, bwd_dW_config_2 = gemm2_configs

    token_counts_by_expert, gather_indices = _get_routing_indices(
        top_k_index, self.num_experts
    )
    offsets = torch.cumsum(token_counts_by_expert, dim=0, dtype=torch.int32)

    if self.gate_up_proj.shape[-1] == hidden_dim:
        w1 = self.gate_up_proj
    else:
        w1 = self.gate_up_proj.transpose(-2, -1).contiguous()

    # First grouped GEMM: gate_up projection.
    first_gemm_output = grouped_gemm(
        X=hidden_states,
        W=w1,
        m_sizes=token_counts_by_expert,
        topk=top_k,
        gather_indices=gather_indices,
        permute_x=True,
        permute_y=False,
        autotune=False,  # cached configs
        kernel_config_fwd=fwd_config_1,
        kernel_config_bwd_dX=bwd_dX_config_1,
        kernel_config_bwd_dW=bwd_dW_config_1,
        is_first_gemm=True,
    )

    # Separated LoRA for gate_up. grouped_gemm ran permute_x=True so first_gemm_output
    # is expert-sorted; _apply_lora_grouped_mm wants pre-permuted input, so gather via
    # gather_indices // top_k (expert-sorted row -> originating token row).
    if gate_up_lora is not None:
        first_weight, second_weight, scaling = gate_up_lora
        first_weight = first_weight.to(hidden_states.dtype)
        second_weight = second_weight.to(hidden_states.dtype)
        permuted_hidden = hidden_states[gather_indices // top_k]
        gate_up_lora_delta = _apply_lora_grouped_mm(
            permuted_hidden,
            first_weight,
            second_weight,
            offsets,
            scaling,
            grouped_mm_func=native_moe_grouped_mm,
        )
        first_gemm_output = first_gemm_output + gate_up_lora_delta

    # Activation + gate*up.
    if hasattr(self, 'act_fn') and callable(self.act_fn):
        gate, up = first_gemm_output.chunk(2, dim=-1)
        intermediate = self.act_fn(gate) * up
    else:
        intermediate = _silu_and_mul(first_gemm_output)

    # Grouped GEMM 2: down projection.
    down_lora = None
    if getattr(self, "_unsloth_lora_down_proj", None) is not None:
        down_lora = self._unsloth_lora_down_proj[:3]
    elif (
        use_separated_lora
        and hasattr(self, "down_proj")
        and _has_lora_adapters(self.down_proj)
    ):
        down_lora = _extract_lora_weights(self.down_proj, num_experts=self.num_experts)

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
        autotune=False,  # cached configs
        kernel_config_fwd=fwd_config_2,
        kernel_config_bwd_dX=bwd_dX_config_2,
        kernel_config_bwd_dW=bwd_dW_config_2,
        is_first_gemm=False,
    )

    # Separated LoRA for down (intermediate already permuted from step 1, same offsets).
    if down_lora is not None:
        first_weight, second_weight, scaling = down_lora

        first_weight = first_weight.to(intermediate.dtype)
        second_weight = second_weight.to(intermediate.dtype)

        lora_delta = _apply_lora_grouped_mm(
            intermediate,
            first_weight,
            second_weight,
            offsets,
            scaling,
            grouped_mm_func=native_moe_grouped_mm
        )

        second_gemm_output = second_gemm_output + lora_delta

    # Apply routing weights and sum across top_k: (num_tokens, top_k, hidden) -> (num_tokens, hidden).
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
    """Loop over experts with routed tokens; torch.compile-disabled to avoid graph breaks on dynamic control flow."""
    # This Unsloth Zoo code section is licensed under AGPL3
    final_hidden_states = torch.zeros_like(hidden_states)
    use_separated_lora = _should_use_separated_lora()

    gate_up_lora = getattr(self, "_unsloth_lora_gate_up_proj", None)
    if gate_up_lora is not None:
        gate_up_lora = gate_up_lora[:3]
    elif (
        use_separated_lora
        and hasattr(self, "gate_up_proj")
        and _has_lora_adapters(self.gate_up_proj)
    ):
        gate_up_lora = _extract_lora_weights(
            self.gate_up_proj, num_experts=self.num_experts, experts_module=self
        )
    # Pre-cast LoRA factors to the activation dtype once (avoid per-expert .to()).
    # `scaling` is left alone: a Python float is a no-op, a tensor broadcasts.
    if gate_up_lora is not None:
        _gate_up_first, _gate_up_second, _gate_up_scaling = gate_up_lora
        gate_up_lora = (
            _gate_up_first.to(hidden_states.dtype),
            _gate_up_second.to(hidden_states.dtype),
            _gate_up_scaling,
        )

    down_lora = getattr(self, "_unsloth_lora_down_proj", None)
    if down_lora is not None:
        down_lora = down_lora[:3]
    elif (
        use_separated_lora
        and hasattr(self, "down_proj")
        and _has_lora_adapters(self.down_proj)
    ):
        down_lora = _extract_lora_weights(
            self.down_proj, num_experts=self.num_experts, experts_module=self
        )
    if down_lora is not None:
        _down_first, _down_second, _down_scaling = down_lora
        down_lora = (
            _down_first.to(hidden_states.dtype),
            _down_second.to(hidden_states.dtype),
            _down_scaling,
        )

    # Expert mask -> which experts have tokens.
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, n_tokens)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    # Some patches (Qwen3-VL-MoE) store experts in grouped_mm layout (E, in, out)
    # rather than F.linear's (E, out, in) and set _unsloth_grouped_mm_format=True.
    # Prefer it over the shape check, which is unsafe when intermediate_dim == hidden_dim.
    grouped_mm_format = bool(getattr(self, "_unsloth_grouped_mm_format", False))

    # GPT-OSS uses interleaved gate/up, clamped swiglu, and per-expert biases.
    is_gpt_oss = "GptOssExperts" in self.__class__.__name__

    for expert_idx_t in expert_hit:
        expert_idx = expert_idx_t.item()

        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]

        # gate_up projection for this expert ('gate_up_proj' or 'w1'/'w3').
        if hasattr(self, "gate_up_proj"):
            gate_up_weight = self.gate_up_proj[expert_idx]
            if grouped_mm_format or gate_up_weight.shape[-1] != current_state.shape[-1]:
                gate_up_weight = gate_up_weight.T
            gate_up = F.linear(current_state, gate_up_weight)
            if gate_up_lora is not None:
                first_weight, second_weight, scaling = gate_up_lora
                lora_delta = current_state @ first_weight[expert_idx]
                lora_delta = lora_delta @ second_weight[expert_idx]
                gate_up = gate_up + lora_delta * scaling
            if is_gpt_oss:
                gate_up_bias = getattr(self, "gate_up_proj_bias", None)
                if gate_up_bias is not None:
                    gate_up = gate_up + gate_up_bias[expert_idx].to(gate_up.dtype)
                gate = gate_up[..., ::2]
                up = gate_up[..., 1::2]
            else:
                gate, up = gate_up.chunk(2, dim=-1)
        else:
            gate = F.linear(current_state, self.w1[expert_idx])
            up = F.linear(current_state, self.w3[expert_idx])

        if is_gpt_oss:
            limit = getattr(self, "limit", 7.0)
            alpha = getattr(self, "alpha", 1.702)
            gate = gate.clamp(min=None, max=limit)
            up = up.clamp(min=-limit, max=limit)
            current_hidden_states = (up + 1.0) * (gate * torch.sigmoid(gate * alpha))
        elif hasattr(self, "act_fn") and callable(self.act_fn):
            current_hidden_states = self.act_fn(gate) * up
        else:
            current_hidden_states = F.silu(gate) * up

        # down projection for this expert.
        if hasattr(self, "down_proj"):
            down_weight = self.down_proj[expert_idx]
            # Mirror gate_up: prefer the flag over the shape heuristic (unsafe at square dims).
            if grouped_mm_format or down_weight.shape[-1] != current_hidden_states.shape[-1]:
                down_weight = down_weight.T
            down = F.linear(current_hidden_states, down_weight)
            if down_lora is not None:
                first_weight, second_weight, scaling = down_lora
                lora_delta = current_hidden_states @ first_weight[expert_idx]
                lora_delta = lora_delta @ second_weight[expert_idx]
                down = down + lora_delta * scaling
            if is_gpt_oss:
                down_bias = getattr(self, "down_proj_bias", None)
                if down_bias is not None:
                    down = down + down_bias[expert_idx].to(down.dtype)
            current_hidden_states = down
        else:
            current_hidden_states = F.linear(current_hidden_states, self.w2[expert_idx])

        current_hidden_states = (
            current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        )

        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    return final_hidden_states
