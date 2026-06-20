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


def _make_lfm2_moe_lora_extractor():
    """LoRA extractor for the fused Lfm2MoeExperts module.

    Lfm2MoeExperts stores its experts as 3D parameters identical in layout to
    Qwen3MoeExperts:
        gate_up_proj : (num_experts, 2 * intermediate_dim, hidden_dim)
        down_proj    : (num_experts, hidden_dim,           intermediate_dim)
    and exposes ``hidden_dim`` / ``intermediate_dim`` attributes, so the io-dims
    of each separated LoRA can be read straight off the module.
    """
    def _get_lfm2_moe_lora_dims(wrapper):
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

    def _lfm2_moe_lora_extractor(wrapper, weight_A, weight_B, scaling, num_experts):
        input_dim, output_dim = _get_lfm2_moe_lora_dims(wrapper)
        return extract_moe_lora_weights_for_grouped_mm(
            wrapper,
            weight_A,
            weight_B,
            scaling,
            num_experts,
            input_dim=input_dim,
            output_dim=output_dim,
            model_name="LFM2 MoE",
            enable_logging=UNSLOTH_ENABLE_LOGGING,
            logger_obj=logger,
        )

    return _lfm2_moe_lora_extractor


def patch_lfm2_moe():
    """Patch LFM2 MoE (lfm2_moe) experts for Split LoRA via grouped GEMM.

    Transformers >= 5 loads Lfm2MoeExperts, whose ``forward`` runs the native
    grouped-mm experts path. Under 4bit QLoRA the fused ``gate_up_proj`` becomes
    a packed 2D bitsandbytes Params4bit, which the native grouped-mm kernel
    cannot consume (it expects a 3D ``(E, in, out)`` weight) and raises
    ``IndexError: Dimension out of range`` on the first backward. Replacing the
    experts ``forward`` with Unsloth's MoE backend dispatcher routes the bnb-4bit
    experts through dequantize + grouped_mm, exactly as is already done for
    qwen3_moe / glm4_moe / deepseek_v3_moe / gemma4_moe.

    The Lfm2MoeSparseMoeBlock keeps its native routing (sigmoid + optional
    expert bias + norm_topk_prob + routed scaling) and still calls
    ``self.experts(hidden_states, top_k_index, top_k_weights)``, whose signature
    matches the backend dispatcher, so only the experts forward is replaced.
    """
    # Install the ParamWrapper MoE forward patch (separated LoRA on the fused
    # experts parameters). qwen3_moe already installs it at import; calling it
    # again is idempotent and keeps this file self-contained.
    patch_param_wrapper_for_moe()

    # Transformers < 5 has no Lfm2MoeExperts (lfm2_moe only ships on 5.x), so the
    # import fails there and this patch is a strict no-op.
    try:
        from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeExperts
    except Exception:
        return

    if getattr(Lfm2MoeExperts, "_unsloth_already_patched", False):
        return

    _lfm2_lora_extractor = _make_lfm2_moe_lora_extractor()
    Lfm2MoeExperts._unsloth_lora_extractor_fn = staticmethod(_lfm2_lora_extractor)

    # IMPORTANT: pass the shared backend dispatcher FUNCTION OBJECT directly (no
    # closure). patch_function serializes the function SOURCE into the compiled
    # cache module, so a closure variable would be unresolved (NameError) there.
    # This mirrors qwen3_moe.py exactly.
    patch_function(Lfm2MoeExperts, "forward", get_forward_moe_backend())
    Lfm2MoeExperts._unsloth_already_patched = True

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched LFM2 MoE experts for Split LoRA support.")


TEMPORARY_PATCHES.append(patch_lfm2_moe)
