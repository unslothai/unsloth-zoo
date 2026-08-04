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
#
# ============================================================================
# Qwen3.5 targeted dtype-consistency patches for fp16 (UNSLOTH_FORCE_FLOAT32)
# training.
#
# Qwen3.5 is in unsloth_zoo.model_lists.FORCE_FLOAT32 because its GatedDeltaNet
# layers NaN/Inf in float16 training. When a user requests fp16 on a GPU that
# does not support bf16, unsloth loads in bf16 and patch_model_and_tokenizer
# down-casts the weights to fp16. The SFTTrainer may still autocast activations
# to bf16 (or keep the pre-quant fp32 residual stream), so hidden_states can end
# up with a different dtype than the down-casted linear weights. The first
# affected linear call then raises:
#
#   RuntimeError: expected mat1 and mat2 to have the same dtype,
#   but got: c10::BFloat16 != c10::Half
#
# These boundary patches are a targeted analog of gemma4_float32.py's
# _unsloth_gemma4_ple_cast_input helper: they align each projection's input
# (and Rotary position embeddings, for attention) with the actual weight dtype
# of the submodule, then cast the output back to the caller's dtype. This keeps
# the residual highway's dtype unchanged while guaranteeing that the fp16 matmuls
# never see a mismatched activation.
#
# All patches gate on UNSLOTH_FORCE_FLOAT32 == "1", so bf16 / fp32 training and
# fp16 training on architectures that do not need the FORCE_FLOAT32 fallback
# are untouched.
# ============================================================================

import os
import torch
from .common import TEMPORARY_PATCHES
from .utils import patch_function, raise_error


def _unsloth_get_linear_weight_dtype(module):
    """Return a representative fp Linear weight dtype, or None if absent."""
    for attr in ("q_proj", "in_proj_qkv", "gate_proj", "up_proj", "fc1", "lm_head"):
        linear = getattr(module, attr, None)
        if linear is None:
            continue
        weight = getattr(linear, "weight", None)
        if weight is None:
            continue
        quant_state = getattr(weight, "quant_state", None)
        if quant_state is not None:
            continue
        dtype = getattr(weight, "dtype", None)
        if dtype is None or not getattr(dtype, "is_floating_point", False):
            continue
        # Skip sub-2-byte floats (fp8) where casting would destroy values.
        if getattr(dtype, "itemsize", 2) < 2:
            continue
        return dtype
    return None


def _unsloth_cast_position_embeddings(position_embeddings, dtype):
    """Cast a (cos, sin) tuple to dtype when necessary."""
    if position_embeddings is None:
        return None
    cos, sin = position_embeddings
    if cos.dtype != dtype:
        cos = cos.to(dtype)
    if sin.dtype != dtype:
        sin = sin.to(dtype)
    return (cos, sin)


def patch_Qwen3_5GatedDeltaNet_dtype():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") != "1":
        return
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5
        cls = transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet
        cls.forward  # ensure attribute exists
    except Exception as e:
        return raise_error("Qwen3_5GatedDeltaNet.forward", e)

    original_forward = cls.forward

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        input_dtype = hidden_states.dtype
        target_dtype = _unsloth_get_linear_weight_dtype(self)
        if target_dtype is not None and hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(target_dtype)

        output = original_forward(
            self, hidden_states,
            cache_params=cache_params,
            attention_mask=attention_mask,
            **kwargs,
        )

        if isinstance(output, torch.Tensor) and output.dtype != input_dtype:
            output = output.to(input_dtype)
        return output
    pass
    patch_function(cls, "forward", forward, force=True, match_level="relaxed")
pass
TEMPORARY_PATCHES.append(patch_Qwen3_5GatedDeltaNet_dtype)


def patch_Qwen3_5Attention_dtype():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") != "1":
        return
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5
        cls = transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Attention
        cls.forward  # ensure attribute exists
    except Exception as e:
        return raise_error("Qwen3_5Attention.forward", e)

    original_forward = cls.forward

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        input_dtype = hidden_states.dtype
        target_dtype = _unsloth_get_linear_weight_dtype(self)
        if target_dtype is not None and hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(target_dtype)
        position_embeddings = _unsloth_cast_position_embeddings(position_embeddings, target_dtype or input_dtype)

        output = original_forward(
            self,
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

        if isinstance(output, tuple):
            first = output[0]
            if first.dtype != input_dtype:
                first = first.to(input_dtype)
            return (first,) + output[1:]
        elif isinstance(output, torch.Tensor) and output.dtype != input_dtype:
            output = output.to(input_dtype)
        return output
    pass
    patch_function(cls, "forward", forward, force=True, match_level="relaxed")
pass
TEMPORARY_PATCHES.append(patch_Qwen3_5Attention_dtype)


def patch_Qwen3_5MLP_dtype():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") != "1":
        return
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5
        cls = transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5MLP
        cls.forward  # ensure attribute exists
    except Exception as e:
        return raise_error("Qwen3_5MLP.forward", e)

    original_forward = cls.forward

    def forward(self, x):
        input_dtype = x.dtype
        target_dtype = _unsloth_get_linear_weight_dtype(self)
        if target_dtype is not None and x.dtype != target_dtype:
            x = x.to(target_dtype)

        output = original_forward(self, x)

        if isinstance(output, torch.Tensor) and output.dtype != input_dtype:
            output = output.to(input_dtype)
        return output
    pass
    patch_function(cls, "forward", forward, force=True, match_level="relaxed")
pass
TEMPORARY_PATCHES.append(patch_Qwen3_5MLP_dtype)


def patch_Qwen3_5ForCausalLM_dtype():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") != "1":
        return
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5
        cls = transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM
        CausalLMOutputWithPast = transformers.models.qwen3_5.modeling_qwen3_5.CausalLMOutputWithPast
    except Exception as e:
        return raise_error("Qwen3_5ForCausalLM.forward", e)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        lm_input = hidden_states[:, slice_indices, :]

        target_dtype = getattr(getattr(self.lm_head, "weight", None), "dtype", lm_input.dtype)
        if getattr(target_dtype, "is_floating_point", False) and lm_input.dtype != target_dtype:
            lm_input = lm_input.to(target_dtype)

        logits = self.lm_head(lm_input)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    pass
    patch_function(cls, "forward", forward, force=True, match_level="relaxed")
pass
TEMPORARY_PATCHES.append(patch_Qwen3_5ForCausalLM_dtype)
