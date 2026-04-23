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
from .utils import patch_function, logger
from .moe_utils import (
    patch_param_wrapper_for_moe,
    get_forward_moe_backend,
)


def patch_gemma4_moe():
    """
    Patches Gemma4 MoE (26B-A4B and any other MoE-enabled Gemma 4 variant)
    for Split LoRA + grouped GEMM backend + text-only GRPO training.

    Compatible with transformers >= 5.5.0 (first PyPI release with Gemma 4
    support). The stacked expert tensor layout
    ``Gemma4TextExperts.gate_up_proj (E, 2*I, H)`` / ``down_proj (E, H, I)``
    matches Qwen3-MoE, so the default ``_extract_lora_from_wrapper`` + the
    generic ``get_forward_moe_backend`` cover LoRA extraction and MoE
    forward without arch-specific branches.

    The router's ``per_expert_scale`` is already folded into
    ``top_k_weights`` natively by ``Gemma4TextRouter.forward`` since
    transformers 5.5.0, so this patch does NOT need to fold it.
    """
    # Patch PEFT ParamWrapper for separated LoRA weights (idempotent).
    patch_param_wrapper_for_moe()

    # Gemma 4 available only on transformers >= 5.5.0. Bail cleanly on
    # older installs (training code elsewhere errors out with a clearer
    # version-pin message).
    try:
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextExperts,
        )
    except Exception:
        return  # Gemma4 not available in this transformers version

    if getattr(Gemma4TextExperts, "_unsloth_already_patched", False):
        return

    # ====================================================================
    # Replace Gemma4TextExperts.forward with the grouped-GEMM backend.
    # The stock forward (modeling_gemma4.py) is a Python loop over active
    # experts — not CUDA-graph capturable and slow at inference / decode.
    # The generic forward_native_grouped_mm handles:
    #   - act_fn (Gemma 4 sets self.act_fn to gelu_pytorch_tanh natively)
    #   - the (E, 2*I, H) / (E, H, I) expert layout via `chunk(2, -1)`
    # so no arch-specific branches are needed in moe_utils.py.
    # ====================================================================
    _moe_backend = get_forward_moe_backend()

    def _gemma4_moe_experts_forward(self, hidden_states, top_k_index, top_k_weights):
        # per_expert_scale is already applied on `top_k_weights` by
        # Gemma4TextRouter.forward since transformers 5.5.0 — do not
        # re-apply here.
        return _moe_backend(self, hidden_states, top_k_index, top_k_weights)

    patch_function(
        Gemma4TextExperts, "forward", _gemma4_moe_experts_forward, force=True
    )

    # ====================================================================
    # Patch Gemma4ForConditionalGeneration.forward for text-only GRPO /
    # SFT: inject a zero-filled ``mm_token_type_ids`` when the caller
    # (TRL GRPOTrainer, standard Trainer, etc.) doesn't supply one in
    # training mode. Without this, ``create_causal_mask_mapping``
    # (modeling_gemma4.py) raises
    # ``ValueError: mm_token_type_ids is required as a model input when
    # training`` on every text-only train step.
    #
    # Also supports UNSLOTH_RETURN_HIDDEN_STATES=1 (GRPO rollout path).
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
            # Inject mm_token_type_ids=0 for text-only SFT / GRPO.
            if mm_token_type_ids is None and self.training:
                _ids = input_ids if input_ids is not None else inputs_embeds
                if _ids is not None:
                    mm_token_type_ids = torch.zeros(
                        _ids.shape[:2], dtype=torch.long, device=_ids.device
                    )

            # Drop stale mm_token_type_ids during KV cache generation when
            # the sequence length no longer matches.
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

            # RETURN_HIDDEN_STATES mode — return hidden states instead of
            # logits. Used by GRPO rollout to stream forward features.
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
                "Unsloth: Patched Gemma4ForConditionalGeneration.forward for "
                "text-only training + GRPO hidden states."
            )
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(
                f"Unsloth: Could not patch Gemma4ForConditionalGeneration.forward: {e}"
            )

    Gemma4TextExperts._unsloth_already_patched = True

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched Gemma4 MoE for Split LoRA + grouped GEMM.")


TEMPORARY_PATCHES.append(patch_gemma4_moe)
