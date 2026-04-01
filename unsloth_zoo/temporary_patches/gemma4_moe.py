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

import os
import torch
from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import patch_function, raise_error, logger
from .moe_utils import (
    patch_param_wrapper_for_moe,
    get_forward_moe_backend,
)


def patch_gemma4_moe():
    """
    Patches Gemma4 MoE to support Split LoRA using grouped GEMM.
    Gemma4 uses 128 experts with top_k=8 and a unique per_expert_scale parameter.
    """
    # Patch PEFT ParamWrapper for separated LoRA weights
    patch_param_wrapper_for_moe()

    # Try to import Gemma4 MoE classes
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMoEBlock
    except Exception:
        return  # Gemma4 not available in this transformers version

    # ====================================================================
    # Define LoRA extraction function for Gemma4 (Standard F.linear Format)
    # Weights are (E, out_dim, in_dim):
    #   gate_up_proj: (E, 2*I, H)
    #   down_proj:    (E, H, I)
    # ====================================================================
    def _gemma4_lora_extractor(wrapper, weight_A, weight_B, scaling, num_experts):
        """
        Custom LoRA extractor for Gemma4 MoE.

        Expectation for grouped_mm:
        - first_weight:  (E, in_dim, R)   [Input projection to rank]
        - second_weight: (E, R, out_dim)  [Rank projection to output]

        Gemma4 standard F.linear format:
        - weight_A: (E*R, in_dim)  -> reshape to (E, R, in_dim) -> permute to (E, in_dim, R)
        - weight_B: (out_dim, E*R) -> reshape to (out_dim, E, R) -> permute to (E, R, out_dim)
        """
        total_rank = weight_A.shape[0]
        rank_per_expert = total_rank // num_experts
        dim_A = weight_A.shape[1]
        dim_B = weight_B.shape[0]

        # Gemma4 uses standard F.linear format (E, out_dim, in_dim).
        # PEFT LoRA: A=(E*R, in_dim), B=(out_dim, E*R)
        # grouped_mm needs: first=(E, in_dim, R), second=(E, R, out_dim)
        # This mapping is the same for both gate_up_proj and down_proj.
        first_weight = weight_A.view(num_experts, rank_per_expert, dim_A)
        first_weight = first_weight.permute(0, 2, 1).contiguous()  # (E, in_dim, R)

        second_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
        second_weight = second_weight.permute(1, 2, 0).contiguous()  # (E, R, out_dim)

        return first_weight, second_weight, scaling, num_experts

    Gemma4TextMoEBlock._unsloth_lora_extractor_fn = staticmethod(_gemma4_lora_extractor)

    # ====================================================================
    # Patch Gemma4TextMoEBlock.forward with optimized grouped GEMM backend
    # The generic forward in moe_utils now handles act_fn and per_expert_scale
    # ====================================================================
    patch_function(Gemma4TextMoEBlock, "forward", get_forward_moe_backend(), force=True)

    # ====================================================================
    # Patch Gemma4ForConditionalGeneration.forward for GRPO training
    # When UNSLOTH_RETURN_HIDDEN_STATES=1, return hidden_states instead of logits
    # ====================================================================
    try:
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForConditionalGeneration,
            Gemma4CausalLMOutputWithPast,
        )

        _original_causal_lm_forward = Gemma4ForConditionalGeneration.forward

        def _patched_causal_lm_forward(
            self,
            input_ids=None,
            pixel_values=None,
            pixel_values_videos=None,
            input_features=None,
            attention_mask=None,
            input_features_mask=None,
            position_ids=None,
            image_position_ids=None,
            video_position_ids=None,
            past_key_values=None,
            mm_token_type_ids=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            logits_to_keep=0,
            **kwargs,
        ):
            RETURN_HIDDEN_STATES = (
                os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
            )

            if not RETURN_HIDDEN_STATES:
                return _original_causal_lm_forward(
                    self,
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    input_features=input_features,
                    attention_mask=attention_mask,
                    input_features_mask=input_features_mask,
                    position_ids=position_ids,
                    image_position_ids=image_position_ids,
                    video_position_ids=video_position_ids,
                    past_key_values=past_key_values,
                    mm_token_type_ids=mm_token_type_ids,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    logits_to_keep=logits_to_keep,
                    **kwargs,
                )

            # RETURN_HIDDEN_STATES mode - return hidden_states instead of logits
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                input_features=input_features,
                attention_mask=attention_mask,
                input_features_mask=input_features_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                image_position_ids=image_position_ids,
                video_position_ids=video_position_ids,
                **kwargs,
            )

            hidden_states = outputs[0]
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            logits = hidden_states[:, slice_indices, :]

            return Gemma4CausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                image_hidden_states=getattr(outputs, "image_hidden_states", None),
                audio_hidden_states=getattr(outputs, "audio_hidden_states", None),
            )

        _patched_causal_lm_forward.__qualname__ = _original_causal_lm_forward.__qualname__
        Gemma4ForConditionalGeneration.forward = _patched_causal_lm_forward
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(
                "Unsloth: Patched Gemma4ForConditionalGeneration.forward for GRPO hidden states."
            )
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(
                f"Unsloth: Could not patch Gemma4ForConditionalGeneration.forward: {e}"
            )

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched Gemma4 MoE for Split LoRA support.")


TEMPORARY_PATCHES.append(patch_gemma4_moe)
