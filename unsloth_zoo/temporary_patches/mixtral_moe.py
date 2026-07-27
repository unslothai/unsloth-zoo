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


def _make_mixtral_moe_lora_extractor():
    """LoRA extractor for the fused MixtralExperts. Same 3D layout as Qwen3MoeExperts /
    Lfm2MoeExperts (gate_up_proj (E, 2*I, H), down_proj (E, H, I)); io-dims read off
    hidden_dim / intermediate_dim."""
    def _get_mixtral_moe_lora_dims(wrapper):
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

    def _mixtral_moe_lora_extractor(wrapper, weight_A, weight_B, scaling, num_experts):
        input_dim, output_dim = _get_mixtral_moe_lora_dims(wrapper)
        return extract_moe_lora_weights_for_grouped_mm(
            wrapper,
            weight_A,
            weight_B,
            scaling,
            num_experts,
            input_dim=input_dim,
            output_dim=output_dim,
            model_name="Mixtral MoE",
            enable_logging=UNSLOTH_ENABLE_LOGGING,
            logger_obj=logger,
        )

    return _mixtral_moe_lora_extractor


def patch_mixtral_moe():
    """Patch Mixtral MoE (mixtral) fused experts for Split LoRA via grouped GEMM. On
    transformers 5.x MixtralExperts stores its experts as fused 3D nn.Parameters
    (gate_up_proj (E, 2*I, H), down_proj (E, H, I)). Under 4bit QLoRA those become packed
    bnb Params4bit blobs, and the native experts.forward's nn.functional.linear over the
    packed weight raises `mat1 and mat2 shapes cannot be multiplied`. Routing it through
    Unsloth's MoE backend (dequantize + grouped_mm) fixes it, as for qwen3_moe / lfm2_moe /
    glm4_moe / etc. The MixtralSparseMoeBlock keeps its native routing, calling
    self.experts(...) whose signature matches the dispatcher, so only the experts forward
    is replaced.
    """
    # Separated LoRA on the fused experts params. Idempotent (qwen3_moe installs it too).
    patch_param_wrapper_for_moe()

    # Transformers < 5 has no MixtralExperts -> strict no-op there.
    try:
        from transformers.models.mixtral.modeling_mixtral import MixtralExperts
    except Exception:
        return

    if getattr(MixtralExperts, "_unsloth_already_patched", False):
        return

    _mixtral_lora_extractor = _make_mixtral_moe_lora_extractor()
    MixtralExperts._unsloth_lora_extractor_fn = staticmethod(_mixtral_lora_extractor)

    # Pass the function object directly (no closure): patch_function serializes the source into
    # the compiled cache, so a closure var would be a NameError there. Mirrors lfm2_moe.py.
    patch_function(MixtralExperts, "forward", get_forward_moe_backend())
    MixtralExperts._unsloth_already_patched = True

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched Mixtral MoE experts for Split LoRA support.")


TEMPORARY_PATCHES.append(patch_mixtral_moe)
