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

import math
import types

from .common import (
    TEMPORARY_PATCHES,
    UNSLOTH_ENABLE_LOGGING,
)
from .utils import (
    patch_function,
    logger,
)

from .moe_utils import (
    patch_param_wrapper_for_moe,
    get_forward_moe_backend,
    extract_moe_lora_weights_for_grouped_mm,
)


def _make_step3p7_moe_lora_extractor():
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


def _step3p7_apply_gate(self, gate_up):
    """Step3p7Experts' swiglu: clamp AFTER the activation (FP8Experts' own gate clamps before it)."""
    gate, up = gate_up.chunk(2, dim=-1)
    gate = self.act_fn(gate).clamp(max=self.limit)
    up = up.clamp(min=-self.limit, max=self.limit)
    return gate * up


def _adopt_step3p7_fp8_experts(model, limits):
    """Restore per-layer `swiglu_limits` clamps FP8Experts drops, and leave "eager", whose loop skips the LoRA stash."""
    try:
        from transformers.integrations.finegrained_fp8 import ALL_FP8_EXPERTS_FUNCTIONS
    except Exception:
        return
    routed = getattr(ALL_FP8_EXPERTS_FUNCTIONS, "_unsloth_fp8_dispatcher", False)
    for block, limit in limits:
        experts = getattr(block, "experts", None)
        if experts is None or type(experts).__name__ != "FP8Experts":
            continue
        if limit is not None and math.isfinite(limit):
            experts.limit = limit
            experts._apply_gate = types.MethodType(_step3p7_apply_gate, experts)
            experts._unsloth_own_apply_gate = True
        config = getattr(experts, "config", None)
        if routed and config is not None and getattr(config, "_experts_implementation", None) in (None, "eager"):
            config._experts_implementation = "grouped_mm"


def patch_step3p7_fp8_experts():
    try:
        import transformers.integrations.finegrained_fp8 as finegrained_fp8
        import transformers.models.step3p7.modeling_step3p7  # noqa: F401
    except Exception:
        return
    original = finegrained_fp8.replace_with_fp8_linear
    if getattr(original, "_unsloth_step3p7", False):
        return

    def replace_with_fp8_linear(model, *args, **kwargs):
        # By name: Unsloth's compiler re-creates the step3p7 classes.
        limits = [
            (block, getattr(block.experts, "limit", None))
            for block in model.modules()
            if type(getattr(block, "experts", None)).__name__ == "Step3p7Experts"
        ]
        model = original(model, *args, **kwargs)
        if limits:
            _adopt_step3p7_fp8_experts(model, limits)
        return model

    replace_with_fp8_linear._unsloth_step3p7 = True
    replace_with_fp8_linear.__wrapped__ = original
    finegrained_fp8.replace_with_fp8_linear = replace_with_fp8_linear


def patch_step3p7_moe():
    patch_param_wrapper_for_moe()

    try:
        from transformers.models.step3p7.modeling_step3p7 import Step3p7Experts
    except Exception:
        return

    patch_step3p7_fp8_experts()

    if getattr(Step3p7Experts, "_unsloth_already_patched", False):
        return

    _step3p7_lora_extractor = _make_step3p7_moe_lora_extractor()
    Step3p7Experts._unsloth_lora_extractor_fn = staticmethod(_step3p7_lora_extractor)
    Step3p7Experts._unsloth_own_apply_gate = True

    # No closure: patch_function serializes the source, so a closure var is a NameError there.
    if not patch_function(Step3p7Experts, "forward", get_forward_moe_backend()):
        for name in ("_unsloth_lora_extractor_fn", "_unsloth_own_apply_gate"):
            if name in vars(Step3p7Experts):
                delattr(Step3p7Experts, name)
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning("Unsloth: Could not patch Step-3.7 MoE experts; keeping the native forward.")
        return
    Step3p7Experts._unsloth_already_patched = True

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched Step-3.7 MoE experts for Split LoRA support.")


TEMPORARY_PATCHES.append(patch_step3p7_moe)
