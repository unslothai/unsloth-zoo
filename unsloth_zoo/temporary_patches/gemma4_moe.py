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
import torch.nn as nn
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
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextMoEBlock,
            Gemma4TextDecoderLayer,
        )
    except Exception:
        return  # Gemma4 not available in this transformers version

    # Check if already patched
    if hasattr(Gemma4TextMoEBlock, "_unsloth_already_patched"):
        return

    # ====================================================================
    # Remap decoder layer module names to match checkpoint key layout:
    #   moe.{gate_up_proj,down_proj} -> experts.{...}
    #   moe.per_expert_scale         -> router.per_expert_scale
    # ====================================================================
    _original_decoder_init = Gemma4TextDecoderLayer.__init__

    def _patched_decoder_init(self, config, layer_idx):
        _original_decoder_init(self, config, layer_idx)
        if getattr(self, "enable_moe_block", False) and "moe" in self._modules:
            moe_block = self._modules.pop("moe")
            self._modules["experts"] = moe_block
            object.__setattr__(self, "moe", moe_block)

            per_expert_scale_data = moe_block.per_expert_scale.data
            del moe_block._parameters["per_expert_scale"]
            self.router.per_expert_scale = nn.Parameter(per_expert_scale_data)
            # Non-persistent buffer keeps _init_weights happy without appearing in state_dict
            moe_block.register_buffer("per_expert_scale", torch.ones(config.num_experts), persistent=False)
            object.__setattr__(moe_block, "_router_ref", self.router)

    Gemma4TextDecoderLayer.__init__ = _patched_decoder_init

    # Gemma4 uses standard F.linear format (E, out_dim, in_dim), same as Qwen3MoE.
    # The default _extract_lora_from_wrapper handles this correctly — no custom extractor needed.

    # ====================================================================
    # Patch Gemma4TextMoEBlock.forward with optimized grouped GEMM backend.
    # Pre-multiply per_expert_scale into routing weights so the generic
    # grouped_mm forward doesn't need per_expert_scale special-casing.
    # ====================================================================
    _moe_backend = get_forward_moe_backend()

    def _gemma4_moe_forward(self, hidden_states, top_k_index, top_k_weights):
        # Fold per_expert_scale into routing weights before grouped_mm
        router_ref = getattr(self, "_router_ref", None)
        if router_ref is not None:
            pes = router_ref.per_expert_scale
            top_k_weights = top_k_weights * pes[top_k_index].to(top_k_weights.dtype)
        return _moe_backend(self, hidden_states, top_k_index, top_k_weights)

    patch_function(Gemma4TextMoEBlock, "forward", _gemma4_moe_forward, force=True)

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
            # Inject mm_token_type_ids=0 for text-only SFT
            if mm_token_type_ids is None and self.training:
                _ids = input_ids if input_ids is not None else inputs_embeds
                if _ids is not None:
                    mm_token_type_ids = torch.zeros(
                        _ids.shape[:2], dtype=torch.long, device=_ids.device
                    )

            # Drop stale mm_token_type_ids during KV cache generation
            _seq_ref = input_ids if input_ids is not None else inputs_embeds
            if mm_token_type_ids is not None and _seq_ref is not None:
                if mm_token_type_ids.shape[1] != _seq_ref.shape[1]:
                    mm_token_type_ids = None

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

            # RETURN_HIDDEN_STATES mode - return hidden_states instead of logits.
            # Drop return_dict from kwargs if TRL/caller pre-set it to avoid
            # "got multiple values for keyword argument 'return_dict'".
            kwargs.pop("return_dict", None)
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
                return_dict=True,
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

    Gemma4TextMoEBlock._unsloth_already_patched = True

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched Gemma4 MoE for Split LoRA support.")


TEMPORARY_PATCHES.append(patch_gemma4_moe)
