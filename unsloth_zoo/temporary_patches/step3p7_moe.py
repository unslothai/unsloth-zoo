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


def _make_step3p7_moe_lora_extractor():
    """LoRA extractor for Step3p7Experts. Same 3D layout as Qwen3MoeExperts
    (gate_up_proj (E, 2*I, H), down_proj (E, H, I)); io-dims read off hidden_dim / intermediate_dim."""
    def _get_step3p7_moe_lora_dims(wrapper):
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

    def _step3p7_moe_lora_extractor(wrapper, weight_A, weight_B, scaling, num_experts):
        input_dim, output_dim = _get_step3p7_moe_lora_dims(wrapper)
        return extract_moe_lora_weights_for_grouped_mm(
            wrapper,
            weight_A,
            weight_B,
            scaling,
            num_experts,
            input_dim=input_dim,
            output_dim=output_dim,
            model_name="Step-3.7 MoE",
            enable_logging=UNSLOTH_ENABLE_LOGGING,
            logger_obj=logger,
        )

    return _step3p7_moe_lora_extractor


def patch_step3p7_moe():
    """Patch Step-3.7-Flash (transformers step3p7) routed experts for Split LoRA via grouped GEMM.

    Step3p7Experts keeps its own per-expert Python loop instead of transformers' generic experts
    dispatcher: 16-bit training runs 288 expert matmul pairs per layer one by one, and under 4-bit
    QLoRA the loop matmuls the packed bnb Params4bit directly ("size mismatch, got input (N),
    mat (N x H), vec (1)"). Routing it through Unsloth's MoE backend (dequantize + grouped_mm with
    the expert LoRA folded in) fixes both. The class's own `_apply_gate` is kept: Step-3.7 clamps
    the routed experts' swiglu on its last two layers (`swiglu_limits`), which a plain
    act_fn(gate) * up would drop. Step3p7SparseMoeBlock keeps its native routing and calls
    self.experts(hidden_states, top_k_index, top_k_weights), the backend's signature.
    """
    # Separated LoRA on the fused experts params. Idempotent (qwen3_moe installs it too).
    patch_param_wrapper_for_moe()

    # Transformers without the native step3p7 (4.x, early 5.x) -> strict no-op.
    try:
        from transformers.models.step3p7.modeling_step3p7 import Step3p7Experts
    except Exception:
        return

    if getattr(Step3p7Experts, "_unsloth_already_patched", False):
        return

    _step3p7_lora_extractor = _make_step3p7_moe_lora_extractor()
    Step3p7Experts._unsloth_lora_extractor_fn = staticmethod(_step3p7_lora_extractor)
    # Read by the MoE backends: apply Step3p7Experts._apply_gate (clamped swiglu) on [gate; up].
    Step3p7Experts._unsloth_own_apply_gate = True

    # Pass the function object directly (no closure): patch_function serializes the source into
    # the compiled cache, so a closure var would be a NameError there. Mirrors qwen3_moe.py.
    patch_function(Step3p7Experts, "forward", get_forward_moe_backend())
    Step3p7Experts._unsloth_already_patched = True

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched Step-3.7 MoE experts for Split LoRA support.")


TEMPORARY_PATCHES.append(patch_step3p7_moe)
