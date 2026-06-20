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

from .common import (
    TEMPORARY_PATCHES,
    UNSLOTH_ENABLE_LOGGING,
)
from .utils import (
    patch_function,
    logger,
)

# Grouped GEMM kernel integration for MoE training acceleration.
from .moe_utils import (
    patch_param_wrapper_for_moe,
    get_forward_moe_backend,
    extract_moe_lora_weights_for_grouped_mm,
)


def _make_ernie4_5_moe_lora_extractor():
    """LoRA extractor for the fused ERNIE 4.5 MoE experts modules.

    Ernie4_5_MoeExperts / Ernie4_5_VLMoeMoeExperts store their experts as 3D
    parameters identical in layout to Qwen3MoeExperts:
        gate_up_proj : (num_experts, 2 * intermediate_dim, hidden_dim)
        down_proj    : (num_experts, hidden_dim,           intermediate_dim)
    and expose ``hidden_dim`` / ``intermediate_dim`` attributes, so the io-dims
    of each separated LoRA can be read straight off the module.
    """
    def _get_ernie4_5_moe_lora_dims(wrapper):
        if wrapper is None or not hasattr(wrapper, "get_base_layer"):
            return None, None

        base = wrapper.get_base_layer()
        param_name = getattr(wrapper, "parameter_name", None)
        if param_name == "gate_up_proj":
            input_dim = getattr(base, "hidden_dim", None)
            output_dim = getattr(base, "intermediate_dim", None)
            return input_dim, None if output_dim is None else 2 * output_dim
        if param_name == "down_proj":
            return getattr(base, "intermediate_dim", None), getattr(base, "hidden_dim", None)

        return None, None

    def _ernie4_5_moe_lora_extractor(wrapper, weight_A, weight_B, scaling, num_experts):
        input_dim, output_dim = _get_ernie4_5_moe_lora_dims(wrapper)
        return extract_moe_lora_weights_for_grouped_mm(
            wrapper,
            weight_A,
            weight_B,
            scaling,
            num_experts,
            input_dim=input_dim,
            output_dim=output_dim,
            model_name="ERNIE 4.5 MoE",
            enable_logging=UNSLOTH_ENABLE_LOGGING,
            logger_obj=logger,
        )

    return _ernie4_5_moe_lora_extractor


# (modeling module, fused experts class name) for every ERNIE 4.5 MoE variant.
# Ernie4_5_VLMoe's experts class is named Ernie4_5_VLMoeMoeExperts (double Moe).
_ERNIE4_5_MOE_EXPERTS = (
    ("transformers.models.ernie4_5_moe.modeling_ernie4_5_moe", "Ernie4_5_MoeExperts"),
    ("transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe", "Ernie4_5_VLMoeMoeExperts"),
)


def patch_ernie4_5_moe():
    """Patch ERNIE 4.5 MoE (ernie4_5_moe / ernie4_5_vl_moe) experts for Split
    LoRA via grouped GEMM.

    Transformers >= 5 loads the fused Ernie4_5_MoeExperts (and, for the VL model,
    Ernie4_5_VLMoeMoeExperts). Their native grouped-mm experts ``forward`` cannot
    consume a packed 2D bitsandbytes Params4bit ``gate_up_proj`` under 4bit QLoRA
    and raises ``IndexError: Dimension out of range`` on the first backward.
    Replacing the experts ``forward`` with Unsloth's MoE backend dispatcher routes
    bnb-4bit experts through dequantize + grouped_mm, exactly as for
    qwen3_moe / glm4_moe / deepseek_v3_moe / gemma4_moe.

    The sparse MoE block (including ERNIE's shared experts and custom top-k
    router) keeps its native forward and still calls
    ``self.experts(hidden_states, top_k_index, top_k_weights)``, whose signature
    matches the backend dispatcher, so only the experts forward is replaced.
    """
    import importlib

    # Install the ParamWrapper MoE forward patch (separated LoRA on the fused
    # experts parameters). Idempotent; qwen3_moe already installs it at import.
    patch_param_wrapper_for_moe()

    _ernie_lora_extractor = _make_ernie4_5_moe_lora_extractor()
    patched_any = False
    for module_path, experts_name in _ERNIE4_5_MOE_EXPERTS:
        # Transformers < 5 has no ERNIE 4.5 MoE experts classes (they only ship
        # on 5.x), so the import/getattr fails there and this is a strict no-op.
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue
        experts_cls = getattr(module, experts_name, None)
        if experts_cls is None:
            continue
        if getattr(experts_cls, "_unsloth_already_patched", False):
            patched_any = True
            continue

        experts_cls._unsloth_lora_extractor_fn = staticmethod(_ernie_lora_extractor)
        # Pass the shared backend dispatcher FUNCTION OBJECT directly (no closure)
        # so patch_function can serialize its SOURCE into the compiled cache.
        patch_function(experts_cls, "forward", get_forward_moe_backend())
        experts_cls._unsloth_already_patched = True
        patched_any = True

    if patched_any and UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched ERNIE 4.5 MoE experts for Split LoRA support.")


TEMPORARY_PATCHES.append(patch_ernie4_5_moe)
