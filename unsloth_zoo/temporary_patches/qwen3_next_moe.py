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
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from .common import (
    TEMPORARY_PATCHES,
    UNSLOTH_ENABLE_LOGGING,
)
from .utils import (
    patch_function,
    raise_error,
    logger,
)
from .moe_utils import patch_param_wrapper_for_moe
from .qwen3_moe import (
    _make_qwen_moe_lora_extractor,
    _make_qwen_moe_experts_forward,
    _make_qwen_moe_sparse_moe_block_forward,
    _patch_causal_lm_forward_for_hidden_states,
)


def patch_qwen3_next_moe():
    try:
        import transformers.models.qwen3_next.modeling_qwen3_next
        transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextSparseMoeBlock
    except Exception as e:
        return

    patch_param_wrapper_for_moe()
    _qwen3_next_lora_extractor = _make_qwen_moe_lora_extractor()

    try:
        import transformers.models.qwen3_next.modeling_qwen3_next
        transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextExperts._unsloth_lora_extractor_fn = _qwen3_next_lora_extractor
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: Could not register Qwen3NextExperts LoRA extractor: {e}")


    try:
        forward = _make_qwen_moe_experts_forward(
            module_name="unsloth_zoo.temporary_patches.qwen3_next_moe"
        )
        patch_function(
            transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextExperts,
            "forward",
            forward,
        )

        sparse_moe_block_forward = _make_qwen_moe_sparse_moe_block_forward(
            use_shared_expert=True,
            module_name="unsloth_zoo.temporary_patches.qwen3_next_moe",
        )
        patch_function(
            transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextSparseMoeBlock,
            "forward",
            sparse_moe_block_forward,
        )
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: Could not patch Qwen3NextExperts or Qwen3NextSparseMoeBlock: {e}")

    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM
        from transformers.modeling_outputs import CausalLMOutputWithPast
        _patch_causal_lm_forward_for_hidden_states(
            Qwen3NextForCausalLM,
            CausalLMOutputWithPast,
            "Qwen3NextForCausalLM",
        )
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: Could not patch Qwen3NextForCausalLM.forward: {e}")


pass
TEMPORARY_PATCHES.append(patch_qwen3_next_moe)
