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

import torch
import torch.nn as nn
import inspect
import importlib
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from .common import TEMPORARY_PATCHES, torch_compile, _torch_compile
from .utils import (
    patch_function,
    process_output_options,
    process_return,
    KWARGS_TYPE,
    raise_error,
    ImageInput,
    PreTokenizedInput,
    TextInput,
    Cache,
    StaticCache,
    HybridCache,
    Unpack,
    _get_unique_storage_name,
)
from textwrap import dedent
import re
import os

def patch_ministral3_config_mapping():
    # Fix for Ministral-3 VL models which have text_config.model_type = "ministral3"
    # but transformers CONFIG_MAPPING doesn't have "ministral3" as a key
    # The correct text config is MinistralConfig (model_type = "ministral")
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers import MinistralConfig
    except Exception as e:
        return raise_error("CONFIG_MAPPING or MinistralConfig", e)

    if "ministral3" not in CONFIG_MAPPING:
        CONFIG_MAPPING.register("ministral3", MinistralConfig)
pass
TEMPORARY_PATCHES.append(patch_ministral3_config_mapping)


def patch_tokenizer_convert_added_tokens():
    # Fix for tokenizer_config.json files that have additional_special_tokens as dicts
    # without the "__type": "AddedToken" field. These dicts have a "content" key instead.
    # transformers expects either strings or AddedToken objects, so we need to convert them.
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase, AddedToken
    except Exception as e:
        return raise_error("PreTrainedTokenizerBase", e)

    original_convert_added_tokens = PreTrainedTokenizerBase.convert_added_tokens
    if hasattr(original_convert_added_tokens, "_unsloth_patched"):
        return

    @classmethod
    def patched_convert_added_tokens(cls, obj, save=False, add_type_field=True):
        # Handle dicts with "content" key that don't have "__type" field
        if isinstance(obj, dict) and "content" in obj and "__type" not in obj:
            return AddedToken(**obj)
        return original_convert_added_tokens.__func__(cls, obj, save=save, add_type_field=add_type_field)

    patched_convert_added_tokens._unsloth_patched = True
    PreTrainedTokenizerBase.convert_added_tokens = patched_convert_added_tokens
pass
TEMPORARY_PATCHES.append(patch_tokenizer_convert_added_tokens)


def patch_tokenizer_extra_special_tokens():
    # Fix for tokenizer_config.json files that have extra_special_tokens as a list
    # instead of a dict. transformers expects extra_special_tokens to be a dict.
    # This is a bug in some Mistral model tokenizer configs.
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception as e:
        return raise_error("PreTrainedTokenizerBase", e)

    original_init = PreTrainedTokenizerBase.__init__
    if hasattr(original_init, "_unsloth_extra_special_tokens_patched"):
        return

    def patched_init(self, **kwargs):
        # Convert extra_special_tokens from list to empty dict if needed
        extra_special_tokens = kwargs.get("extra_special_tokens", {})
        if isinstance(extra_special_tokens, list):
            # extra_special_tokens should be a dict, but some models have it as a list
            # Convert to empty dict to avoid errors, the tokens are still in added_tokens_decoder
            kwargs["extra_special_tokens"] = {}
        return original_init(self, **kwargs)

    patched_init._unsloth_extra_special_tokens_patched = True
    PreTrainedTokenizerBase.__init__ = patched_init
pass
TEMPORARY_PATCHES.append(patch_tokenizer_extra_special_tokens)


def patch_merge_quantization_configs():
    # Fixes some issues with merging quantization configs
    try:
        import transformers.quantizers.auto
    except Exception as e:
        return raise_error("transformers.quantizers.auto", e)
    try:
        f = transformers.quantizers.auto.AutoHfQuantizer.merge_quantization_configs
    except Exception as e:
        return raise_error("transformers.quantizers.auto.AutoHfQuantizer.merge_quantization_configs", e)

    # Fast return if already patched
    unique_name = _get_unique_storage_name(transformers.quantizers.auto.AutoHfQuantizer, "merge_quantization_configs")
    if hasattr(transformers.quantizers.auto.AutoHfQuantizer, unique_name): return

    source = inspect.getsource(f)
    items = dir(transformers.quantizers.auto)

    # Fix as at 7th August 2025
    # ValueError: The model is quantized with Mxfp4Config but you are passing a NoneType config.
    # Please make sure to pass the same quantization config class to `from_pretrained` with different loading attributes.
    source = source.replace(
        "if quantization_config.__class__.__name__ != quantization_config_from_args.__class__.__name__:",
        "if quantization_config_from_args is not None and quantization_config.__class__.__name__ != quantization_config_from_args.__class__.__name__:",
    )

    exec("from transformers.quantizers.auto import (" + ",".join(x for x in items if x in source) + ")", globals())
    source = dedent(source)
    # Remove cls if classmethod
    is_classmethod = source.startswith("@classmethod")
    source = source[source.find("def"):]
    if is_classmethod:
        matches = re.match(r"(def[\s]{1,}[^(]{1,}\()[\s]{0,}cls[\s]{0,}\,[\s]{0,}", source)
        if matches is not None:
            found, replace = matches.group(0), matches.group(1)
            source = replace + source[len(found):]
    try:
        exec(source, globals())
    except Exception as e:
        return raise_error("", e)

    patch_function(transformers.quantizers.auto.AutoHfQuantizer, "merge_quantization_configs", merge_quantization_configs)
pass
TEMPORARY_PATCHES.append(patch_merge_quantization_configs)


def patch_CsmDepthDecoderForCausalLM_forward():
    try:
        import transformers.models.csm.modeling_csm
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from transformers.loss.loss_utils import ForCausalLMLoss
    except Exception as e:
        return raise_error("CsmDepthDecoderForCausalLM.forward", e)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        backbone_last_hidden_state: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: KWARGS_TYPE,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        kwargs = process_output_options(self, locals(), kwargs)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids = input_ids,
            backbone_last_hidden_state = backbone_last_hidden_state,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
            # Moved outputs to kwargs since transformers 4.54.0 deletes them
            # output_attentions = output_attentions,
            # output_hidden_states = output_hidden_states,
            cache_position = cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        if isinstance(logits_to_keep, int):
            if logits_to_keep == 0:
                # skip idx 0 logits since it's for the concatenated backbone last hidden state
                slice_indices = slice(1, None)
            else:
                slice_indices = slice(-logits_to_keep, None)
        else:
            slice_indices = logits_to_keep

        logits = self.codebooks_head(
            hidden_states[:, slice_indices, :], cache_position[slice_indices] if cache_position is not None else None
        )
        logits = logits.contiguous()

        loss = None
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            loss = ForCausalLMLoss(
                logits=logits, labels=None, vocab_size=self.config.vocab_size, shift_labels=shift_labels, **kwargs
            )

        return process_return(CausalLMOutputWithPast, {
            "loss" : loss,
            "logits" : logits,
            "past_key_values" : outputs.past_key_values,
            "hidden_states" : outputs.hidden_states,
            "attentions" : outputs.attentions,
        })
    pass
    success = patch_function(transformers.models.csm.modeling_csm.CsmDepthDecoderForCausalLM, "forward", forward)
    if success: return

    # New transformers removes output_attentions and output_hidden_states
    old_forward = forward
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        backbone_last_hidden_state: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: KWARGS_TYPE,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        new_kwargs = locals().copy()
        new_kwargs.pop('old_forward', None)
        kwargs = new_kwargs.pop('kwargs', dict())
        new_kwargs.update(kwargs)
        return old_forward(**new_kwargs)
    patch_function(transformers.models.csm.modeling_csm.CsmDepthDecoderForCausalLM, "forward", forward)
pass
TEMPORARY_PATCHES.append(patch_CsmDepthDecoderForCausalLM_forward)


def patch_CsmForConditionalGeneration_forward():
    try:
        import transformers.models.csm.modeling_csm
        from transformers.models.csm.modeling_csm import CsmOutputWithPast
        from transformers.loss.loss_utils import ForCausalLMLoss
    except Exception as e:
        return raise_error("CsmForConditionalGeneration.forward", e)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_values_cutoffs: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: KWARGS_TYPE,
    ) -> Union[Tuple, CsmOutputWithPast]:
        kwargs = process_output_options(self, locals(), kwargs)

        if input_ids is not None and input_ids.ndim == 2:
            merged_inputs = self._merge_input_ids_with_input_values(
                input_ids, input_values, input_values_cutoffs, labels
            )
            inputs_embeds = merged_inputs["inputs_embeds"]
            labels = merged_inputs["labels"]
            input_ids = None

        backbone_outputs = self.backbone_model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
            # Moved outputs to kwargs since transformers 4.54.0 deletes them
            # output_attentions = output_attentions,
            # output_hidden_states = output_hidden_states,
            cache_position = cache_position,
            **kwargs,
        )

        backbone_hidden_states = backbone_outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        backbone_logits = self.lm_head(backbone_hidden_states[:, slice_indices, :])

        loss = None
        backbone_loss = None
        depth_decoder_loss = None
        depth_decoder_outputs = None
        if labels is not None:
            # select first codebook as labels for the backbone model
            backbone_labels = labels[:, :, 0]
            backbone_loss = self.loss_function(
                logits=backbone_logits, labels=backbone_labels, vocab_size=self.config.vocab_size, **kwargs
            )

            # for the depth decoder, we need to select the frames to train on
            # those are frames where the label is not uniformly `ignore_index` along the codebook dimension
            train_mask = ~(labels[:, :, 1:] == -100).all(dim=-1)
            depth_decoder_input_ids = labels[train_mask][..., : self.config.num_codebooks - 1]
            # add place holder in position 0 that will be replaced by the backbone_last_hidden_state
            depth_decoder_input_ids = torch.nn.functional.pad(depth_decoder_input_ids, (1, 0), value=0)

            train_idxs = train_mask.nonzero(as_tuple=True)
            backbone_last_hidden_states = backbone_hidden_states[train_idxs[0], train_idxs[1] - 1, :]
            depth_decoder_labels = labels[train_mask]

            # Fix: explicitly pass kwargs to depth decoder to get access to num_items_in_batch
            depth_decoder_kwargs = kwargs.copy()
            # backbone loss num_items is based on the 0th codebooks index
            # while depth loss num_items is based on the the remaining 31 codebooks
            # therefore num_items_in_batch should be multiplied by 31
            if 'num_items_in_batch' in depth_decoder_kwargs:
                depth_decoder_kwargs['num_items_in_batch'] = depth_decoder_kwargs['num_items_in_batch'] * 31

            # make sure return_dict is set to True
            depth_decoder_kwargs.pop('return_dict', None)
            # Move output_attentions and output_hidden_states since transformers 4.54 deletes them
            depth_decoder_kwargs["output_attentions"   ] = output_attentions
            depth_decoder_kwargs["output_hidden_states"] = output_hidden_states

            depth_decoder_outputs = self.depth_decoder(
                input_ids = depth_decoder_input_ids,
                backbone_last_hidden_state = backbone_last_hidden_states,
                use_cache = use_cache,
                # output_attentions=output_attentions,
                # output_hidden_states=output_hidden_states,
                return_dict = True,
                labels = depth_decoder_labels,
                # Fix: explicitly pass kwargs to depth decoder to get access to num_items_in_batch
                **depth_decoder_kwargs,
            )

            depth_decoder_loss = depth_decoder_outputs.loss
            loss = backbone_loss + depth_decoder_loss

        return process_return(CsmOutputWithPast, {
            "loss" : loss,
            "backbone_loss" : backbone_loss,
            "depth_decoder_loss" : depth_decoder_loss,
            "logits" : backbone_logits,
            "past_key_values" : backbone_outputs.past_key_values,
            "hidden_states" : backbone_outputs.hidden_states,
            "attentions" : backbone_outputs.attentions,
            "depth_decoder_logits" : depth_decoder_outputs.logits if depth_decoder_outputs is not None else None,
            "depth_decoder_past_key_values" : depth_decoder_outputs.past_key_values
            if depth_decoder_outputs is not None
            else None,
            "depth_decoder_hidden_states" : depth_decoder_outputs.hidden_states
            if depth_decoder_outputs is not None
            else None,
            "depth_decoder_attentions" : depth_decoder_outputs.attentions if depth_decoder_outputs is not None else None,
        })
    pass
    success = patch_function(transformers.models.csm.modeling_csm.CsmForConditionalGeneration, "forward", forward)
    if success: return

    # New transformers removes output_attentions and output_hidden_states
    old_forward = forward
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_values_cutoffs: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: KWARGS_TYPE,
    ) -> Union[Tuple, CsmOutputWithPast]:
        new_kwargs = locals().copy()
        new_kwargs.pop('old_forward', None)
        kwargs = new_kwargs.pop('kwargs', dict())
        new_kwargs.update(kwargs)
        return old_forward(**new_kwargs)
    patch_function(transformers.models.csm.modeling_csm.CsmForConditionalGeneration, "forward", forward)
pass
TEMPORARY_PATCHES.append(patch_CsmForConditionalGeneration_forward)


def patch_transformers_masks():
    if os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") == "1":
        return
    try:
        import transformers.masking_utils as masking_utils
        import transformers.generation.utils as generation_utils
    except Exception as e:
        return raise_error("transformers.masking_utils", e)

    if hasattr(masking_utils, "__unsloth_mask_patch__"):
        return

    try:
        from torch.nn.attention.flex_attention import BlockMask
    except Exception:
        BlockMask = ()

    try:
        from torch.nn.attention.flex_attention import create_block_mask as torch_create_block_mask
    except Exception:
        torch_create_block_mask = None

    if torch_create_block_mask is not None:
        try:
            supports_compile = "_compile" in inspect.signature(torch_create_block_mask).parameters
        except Exception:
            supports_compile = True
        if not supports_compile:
            def create_block_mask_wrapper(*args, **kwargs):
                kwargs.pop("_compile", None)
                return torch_create_block_mask(*args, **kwargs)
            masking_utils.create_block_mask = create_block_mask_wrapper

    original_create_causal_mask = getattr(
        masking_utils,
        "_unsloth_original_create_causal_mask",
        masking_utils.create_causal_mask,
    )
    original_create_sliding_window_causal_mask = getattr(
        masking_utils,
        "_unsloth_original_create_sliding_window_causal_mask",
        masking_utils.create_sliding_window_causal_mask,
    )

    compiled_create_causal_mask = _torch_compile(
        original_create_causal_mask, fullgraph = False, dynamic = True
    )
    compiled_create_sliding_window_causal_mask = _torch_compile(
        original_create_sliding_window_causal_mask, fullgraph = False, dynamic = True
    )

    def wrap(f):
        def return_attention_mask(*args, **kwargs):
            input_embeds = kwargs.get("input_embeds", None)
            if input_embeds is not None and getattr(input_embeds, "requires_grad", False):
                attention_mask = kwargs.get("attention_mask", None)
                if isinstance(attention_mask, BlockMask) or (
                    isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4
                ):
                    return attention_mask
            return f(*args, **kwargs)
        return return_attention_mask
    pass

    masking_utils._unsloth_original_create_causal_mask = original_create_causal_mask
    masking_utils._unsloth_original_create_sliding_window_causal_mask = original_create_sliding_window_causal_mask
    masking_utils.create_causal_mask = wrap(compiled_create_causal_mask)
    masking_utils.create_sliding_window_causal_mask = wrap(compiled_create_sliding_window_causal_mask)
    masking_utils.create_masks_for_generate = wrap(masking_utils.create_masks_for_generate)
    generation_utils.create_masks_for_generate = masking_utils.create_masks_for_generate
    masking_utils.__unsloth_mask_patch__ = True
pass
TEMPORARY_PATCHES.append(patch_transformers_masks)


def patch_modernbert_attention_mask():
    """Fix ModernBERT attn_bias stride alignment for SDPA backward pass.

    The attention mask created by _prepare_4d_attention_mask uses .expand()
    which creates non-contiguous strides. The SDPA compiled backward kernel
    requires strides to be multiples of 4. Fix: patch _update_attention_mask
    on ModernBertModel to return contiguous masks BEFORE they enter
    torch.compile regions, so the inductor backward graph uses aligned strides.
    """
    try:
        import transformers.models.modernbert.modeling_modernbert as modernbert_module
    except Exception:
        return  # ModernBERT not available, skip

    ModernBertModel = getattr(modernbert_module, "ModernBertModel", None)
    if ModernBertModel is None:
        return

    original_update = getattr(ModernBertModel, "_update_attention_mask", None)
    if original_update is None:
        return

    def _update_attention_mask_contiguous(self, attention_mask, output_attentions=False):
        global_attention_mask, sliding_window_mask = original_update(self, attention_mask, output_attentions=output_attentions)
        # Make masks contiguous so SDPA backward (including compiled graphs)
        # gets strides that are multiples of 4
        if global_attention_mask is not None and not global_attention_mask.is_contiguous():
            global_attention_mask = global_attention_mask.contiguous()
        if sliding_window_mask is not None and not sliding_window_mask.is_contiguous():
            sliding_window_mask = sliding_window_mask.contiguous()
        return global_attention_mask, sliding_window_mask

    ModernBertModel._update_attention_mask = _update_attention_mask_contiguous
pass
TEMPORARY_PATCHES.append(patch_modernbert_attention_mask)


def patch_CsmForConditionalGeneration_merge():
    try:
        import transformers.models.csm.modeling_csm
    except Exception as e:
        return raise_error("CsmForConditionalGeneration._merge_input_ids_with_input_values", e)

    def _merge_input_ids_with_input_values(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        input_values_cutoffs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Merges the input_ids and input_values to produce a single inputs_embeds tensor:
        1 - Infers the codec model on the input_values to retreive codebook token.
        2 - Embeds codebook tokens and places them at the correct positions in the inputs_embeds tensor.
        3 - If labels are provided, expands them to match codebook dimensions and position the target codebook tokens in the inputs_embeds tensor.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The input ids to embed.
            input_values (`torch.Tensor` of shape `(batch_size, channels, audio_sequence_length)`):
                The audio input values to embed.
            input_values_cutoffs (`torch.Tensor` of shape `(batch_size, max_num_audio)`):
                The cutoffs of the audio input values relative to its batch index, padded with -1 when no audio.
        """
        inputs_embeds = self.embed_text_tokens(input_ids)

        if input_values is not None:
            # infer input_values_mask
            input_values_cutoffs = torch.nn.functional.pad(input_values_cutoffs, (1, 0))
            audio_lengths = input_values_cutoffs[input_values_cutoffs >= 0].diff()
            audio_lengths = audio_lengths[audio_lengths > 0]
            input_values_mask = torch.arange(input_values_cutoffs.max(), device=input_values.device).expand(
                len(audio_lengths), -1
            )
            input_values_mask = input_values_mask < audio_lengths.unsqueeze(1)

            # =======================================
            # TODO: @eustlb, this should be batched !!!
            # but requires making sure batched inference of the codec model works as intended
            with torch.no_grad():
                audio_tokens_list = []
                for batch_input_values, batch_input_values_cutoffs in zip(input_values, input_values_cutoffs):
                    batch_input_values_cutoffs = batch_input_values_cutoffs[batch_input_values_cutoffs >= 0]
                    for i in range(batch_input_values_cutoffs.shape[0] - 1):
                        start_idx = batch_input_values_cutoffs[i]
                        end_idx = batch_input_values_cutoffs[i + 1]
                        audio_batch = batch_input_values[..., start_idx:end_idx]
                        codec_outputs = self.codec_model.encode(audio_batch.unsqueeze(0))
                        codebook_ids = codec_outputs.audio_codes.transpose(1, -1)
                        audio_tokens_list.append(codebook_ids[0])

                max_audio_frames = max(el.shape[0] for el in audio_tokens_list)
                batched_audio_token_ids = torch.stack(
                    [torch.nn.functional.pad(el, (0, 0, 0, max_audio_frames - el.shape[0])) for el in audio_tokens_list]
                )
                audio_codes_mask = self.codec_model.get_audio_codes_mask(input_values_mask)
            # =======================================
            audio_token_id = self.config.audio_token_id
            audio_token_mask = input_ids == audio_token_id

            audio_embeds = self.backbone_model.embed_tokens(batched_audio_token_ids)
            inputs_embeds[audio_token_mask] = audio_embeds[audio_codes_mask]

            # same for the audio eos token
            audio_eos_frame_ids = (
                torch.ones((1, 1, self.config.num_codebooks), device=input_ids.device, dtype=torch.long)
                * self.config.codebook_eos_token_id
            )
            audio_eos_embeds = self.backbone_model.embed_tokens(audio_eos_frame_ids).squeeze(1)

            audio_eos_token_mask = input_ids == self.config.audio_eos_token_id
            inputs_embeds[audio_eos_token_mask] = audio_eos_embeds.repeat(audio_eos_token_mask.sum(), 1)

            # if the labels are provided, we need to expand the labels to (batch_size, seq_length, num_codebooks)
            if labels is not None:
                labels_expanded = labels.unsqueeze(-1).repeat(1, 1, self.config.num_codebooks)
                labels_expanded[audio_token_mask] = batched_audio_token_ids[audio_codes_mask]
                # fix make sure to set eos_token_id as a valid label to predict
                labels_expanded[audio_eos_token_mask] = audio_eos_frame_ids
                # mask depth decoder
                depth_decoder_ignore_frames_idxs = (labels == -101).nonzero(as_tuple=True)
                labels_expanded[depth_decoder_ignore_frames_idxs[0], depth_decoder_ignore_frames_idxs[1], 1:] = -100
                labels = labels_expanded

        return {"inputs_embeds": inputs_embeds, "labels": labels}
    pass
    patch_function(transformers.models.csm.modeling_csm.CsmForConditionalGeneration, "_merge_input_ids_with_input_values", _merge_input_ids_with_input_values)
pass
TEMPORARY_PATCHES.append(patch_CsmForConditionalGeneration_merge)


def patch_causal_conv1d_cuda_probe():
    """Probe causal_conv1d CUDA kernels and force slow path if broken.

    On GPUs whose compute capability is not supported by pre-built causal_conv1d
    CUDA kernels (e.g. sm_100 on B200), `import causal_conv1d` succeeds but calling
    `causal_conv1d_fn(...)` fails at runtime with "no kernel image is available".
    This probe runs a tiny forward pass at startup to detect the failure, then
    nullifies causal_conv1d_fn/causal_conv1d_update everywhere so all Mamba-family
    models fall back to their pure-PyTorch slow paths.
    """
    try:
        import causal_conv1d
        from causal_conv1d import causal_conv1d_fn
        from causal_conv1d import causal_conv1d_update
    except ImportError:
        return  # Package not installed, transformers already handles this
    pass

    if causal_conv1d_fn is None:
        return  # Already nullified
    pass

    if not torch.cuda.is_available():
        return
    pass

    # Probe: try a tiny CUDA forward pass
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        x = torch.randn(1, 4, 8, device=device, dtype=torch.float16)
        w = torch.randn(4, 4, device=device, dtype=torch.float16)
        b = torch.zeros(4, device=device, dtype=torch.float16)
        _ = causal_conv1d_fn(x, w, b, activation="silu")
        del x, w, b
        return  # CUDA kernels work fine
    except Exception:
        pass  # Fall through to disable
    pass

    print(
        "Unsloth: causal_conv1d CUDA kernels not compatible with this GPU. "
        "Using PyTorch slow path for Mamba models."
    )

    import sys

    # 1. Nullify the package exports themselves
    for mod_name in ("causal_conv1d", "causal_conv1d.causal_conv1d_interface"):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            if hasattr(mod, "causal_conv1d_fn"):
                mod.causal_conv1d_fn = None
            if hasattr(mod, "causal_conv1d_update"):
                mod.causal_conv1d_update = None
        pass
    pass

    # 2. Patch is_causal_conv1d_available to return False
    try:
        import transformers.utils.import_utils
        transformers.utils.import_utils.is_causal_conv1d_available = lambda: False
    except Exception:
        pass
    pass

    # 3. Dynamically scan all loaded modules and nullify broken causal_conv1d
    #    references. Uses identity checks (is) against the original function objects
    #    to avoid clobbering vllm's independent Triton-based causal_conv1d_fn/update.
    _original_fn = causal_conv1d_fn
    _original_update = causal_conv1d_update

    def _disabled_lazy_load():
        return (None, None)
    pass

    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        # Only nullify references that point to the causal_conv1d package's functions
        touched = False
        if getattr(mod, "causal_conv1d_fn", None) is _original_fn:
            mod.causal_conv1d_fn = None
            touched = True
        if getattr(mod, "causal_conv1d_update", None) is _original_update:
            mod.causal_conv1d_update = None
            touched = True
        # is_fast_path_available = all((causal_conv1d_fn, ...)) -- must be False
        # Only touch it on modules where we just nullified causal_conv1d refs
        if touched and getattr(mod, "is_fast_path_available", False):
            mod.is_fast_path_available = False
        # Replace lazy load stubs (Pattern B: mamba, falcon_mamba)
        if hasattr(mod, "_lazy_load_causal_conv1d"):
            mod._lazy_load_causal_conv1d = _disabled_lazy_load
        if hasattr(mod, "_causal_conv1d_cache"):
            mod._causal_conv1d_cache = (None, None)
    pass
pass
TEMPORARY_PATCHES.append(patch_causal_conv1d_cuda_probe)


def patch_GraniteMoeHybridMambaLayer_cuda_kernels_forward():
    try:
        import transformers.models.granitemoehybrid.modeling_granitemoehybrid
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
            GraniteMoeHybridMambaLayer,
            HybridMambaAttentionDynamicCache,
            apply_mask_to_padding_states,
            mamba_split_conv1d_scan_combined,
            mamba_chunk_scan_combined,
            selective_state_update,
            causal_conv1d_fn,
            causal_conv1d_update,
        )
    except:
        return

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.IntTensor] = None,
    ):
        # 1. Gated MLP's linear projection
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        projected_states = self.in_proj(hidden_states)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_params.conv_states[self.layer_idx].shape[0]
            == cache_params.ssm_states[self.layer_idx].shape[0]
            == batch_size
            and cache_position is not None
            and cache_position[0] > 0
        )

        # getting projected states from cache if it exists
        if use_precomputed_states:
            gate, hidden_states_B_C, dt = projected_states.squeeze(1).split(
                [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
            )

            # 2. Convolution sequence transformation
            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                cache_params.conv_states[self.layer_idx],
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )

            hidden_states, B, C = torch.split(
                hidden_states_B_C,
                [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                dim=-1,
            )

            # 3. SSM transformation
            A = -torch.exp(self.A_log.float())  # (nheads,)
            A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups)
            C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups)
            hidden_states_reshaped = hidden_states.view(batch_size, self.num_heads, self.head_dim)
            hidden_states = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            hidden_states = hidden_states.view(batch_size, self.num_heads * self.head_dim)
            hidden_states = self.norm(hidden_states, gate)

            # 4. Final linear projection
            out = self.out_proj(hidden_states)[:, None, ...]
        # Fused calculations or step by step if no initialized cache is found
        else:
            A = -torch.exp(self.A_log.float())  # (num_heads) or (intermediate_size, state_size)
            dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}

            # 2-4. Fused kernel for conv1d, SSM, and the final projection
            if self.training and cache_params is None:
                out = mamba_split_conv1d_scan_combined(
                    projected_states,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.variance_epsilon,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.head_dim,
                    ngroups=self.n_groups,
                    norm_before_gate=False,
                    return_final_states=False,
                    **dt_limit_kwargs,
                )

            else:
                gate, hidden_states_B_C, dt = projected_states.split(
                    [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
                )

                # 2. Convolution sequence transformation
                # Init cache
                if cache_params is not None:
                    # storing the states
                    # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                    conv_states = nn.functional.pad(
                        hidden_states_B_C_transposed,
                        (self.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0),
                    )
                    cache_params.conv_states[self.layer_idx].copy_(conv_states)

                if self.activation not in ["silu", "swish"]:
                    hidden_states_B_C = self.act(
                        self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2)
                    )
                else:
                    hidden_states_B_C = causal_conv1d_fn(
                        x=hidden_states_B_C.transpose(1, 2),
                        weight=self.conv1d.weight.squeeze(1),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                        seq_idx=seq_idx,
                    ).transpose(1, 2)

                hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
                hidden_states, B, C = torch.split(
                    hidden_states_B_C,
                    [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                    dim=-1,
                )

                # 3. SSM transformation
                scan_output, ssm_state = mamba_chunk_scan_combined(
                    hidden_states.view(batch_size, seq_len, -1, self.head_dim),
                    dt,
                    A,
                    B.view(batch_size, seq_len, self.n_groups, -1),
                    C.view(batch_size, seq_len, self.n_groups, -1),
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=None,
                    seq_idx=seq_idx,
                    return_final_states=True,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    **dt_limit_kwargs,
                )

                # Init cache
                if ssm_state is not None and cache_params is not None:
                    cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
                    cache_params.has_previous_state = True

                scan_output = scan_output.view(batch_size, seq_len, -1)
                # Multiply "gate" branch and apply extra normalization layer
                scan_output = self.norm(scan_output, gate)

                # 4. Final linear projection
                out = self.out_proj(scan_output)
        return out
    pass
    patch_function(transformers.models.granitemoehybrid.modeling_granitemoehybrid.GraniteMoeHybridMambaLayer, "cuda_kernels_forward", cuda_kernels_forward)
pass
TEMPORARY_PATCHES.append(patch_GraniteMoeHybridMambaLayer_cuda_kernels_forward)


def fix_mamba_ssm_float32():
    try:
        import mamba_ssm.ops.triton.ssd_chunk_scan
    except ImportError:
        return
    except Exception as e:
        return raise_error("mamba_ssm.ops.triton.ssd_chunk_scan", e)

    # Try getting file for mamba_ssm
    try:
        ssd_chunk_scan_file = inspect.getfile(mamba_ssm.ops.triton.ssd_chunk_scan)
        with open(ssd_chunk_scan_file, "r", encoding = "utf-8") as file: file = file.read()
    except Exception as e:
        return raise_error("mamba_ssm.ops.triton.ssd_chunk_scan", e)

    # Find dst +=|= tl.dot(a, b)
    matches = list(re.finditer(
        r" ([a-zA-Z0-9\_]{1,}) (\=|\+\=) tl\.dot\(([a-zA-Z0-9\_]{1,})\, ([a-zA-Z0-9\_]{1,})\)",
        file)
    )
    for match in matches:
        old = match.group(0)
        dst, adder, a, b = match.groups()
        accumulator = '' if adder == "=" else f', acc = {dst}'
        # Change to float32 if float16 seen otherwise leave as original precision
        new = f" {dst} = tl.dot("\
            f"{a}.to(tl.float32), "\
            f"{b}.to(tl.float32)"\
            f"{accumulator})"
        file = file.replace(old, new)
    pass

    try:
        # Reload module since we editted it
        with open(ssd_chunk_scan_file, "w", encoding = "utf-8") as f: f.write(file)
        importlib.reload(mamba_ssm.ops.triton.ssd_chunk_scan)
    except Exception as e:
        return raise_error("mamba_ssm.ops.triton.ssd_chunk_scan", e)
pass
TEMPORARY_PATCHES.append(fix_mamba_ssm_float32)


# Mllama Patches

def patch_MllamaVisionEncoderLayer():
    try:
        import math
        import inspect
        import transformers.models.mllama.modeling_mllama
        from transformers.models.mllama.modeling_mllama import (
            MllamaVisionConfig,
            MllamaVisionAttention,
            MllamaVisionMLP,
            MllamaVisionEncoder,
        )
        from transformers.modeling_layers import GradientCheckpointingLayer
    except Exception as e:
        return raise_error("transformers.models.mllama.modeling_mllama", e)


    # ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/modeling_mllama.py#L275C1-L315C28
    class MllamaVisionEncoderLayer(GradientCheckpointingLayer):
        def __init__(self, config: MllamaVisionConfig, is_gated: bool = False):
            super().__init__()

            self.hidden_size = config.hidden_size
            self.num_attention_heads = config.attention_heads
            self.is_gated = is_gated
            self.intermediate_size = config.intermediate_size

            self.self_attn = MllamaVisionAttention(config)
            self.mlp = MllamaVisionMLP(config)

            self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=config.norm_eps)

            if is_gated:
                self.gate_attn = nn.Parameter(torch.ones(1) * math.pi / 4)
                self.gate_ffn = nn.Parameter(torch.ones(1) * math.pi / 4)

        def forward(
            self,
            hidden_state: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
        ):
            # Self Attention
            residual = hidden_state
            hidden_state = self.input_layernorm(hidden_state)
            hidden_state, attn_weights = self.self_attn(hidden_state, attention_mask=attention_mask)
            if self.is_gated:
                hidden_state = self.gate_attn.tanh() * hidden_state
            hidden_state = residual + hidden_state

            # Feed forward
            residual = hidden_state
            hidden_state = self.post_attention_layernorm(hidden_state)
            hidden_state = self.mlp(hidden_state)
            if self.is_gated:
                hidden_state = self.gate_ffn.tanh() * hidden_state
            hidden_state = residual + hidden_state

            return hidden_state

    try:
        vision_encoder_forward_source = inspect.getsource(MllamaVisionEncoder.forward)
        if "gradient_checkpointing" not in vision_encoder_forward_source:
            transformers.models.mllama.modeling_mllama.MllamaVisionEncoderLayer = MllamaVisionEncoderLayer
    except Exception as e:
        return raise_error("transformers.models.mllama.modeling_mllama.MllamaVisionEncoderLayer", e)

pass
TEMPORARY_PATCHES.append(patch_MllamaVisionEncoderLayer)


# Patch Siglip for forced float32 / float16 only
def patch_SiglipEncoderLayer():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.siglip.modeling_siglip
    except Exception as e:
        return raise_error("transformers.models.siglip.modeling_siglip", e)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        hidden_states = hidden_states.to(torch.float32)
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states.to(torch.float16),
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states.to(torch.float16))
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
    pass
    patch_function(transformers.models.siglip.modeling_siglip.SiglipEncoderLayer, "forward", forward)
pass
TEMPORARY_PATCHES.append(patch_SiglipEncoderLayer)


def patch_Lfm2VlMultiModalProjector():
    """Fix Lfm2VlMultiModalProjector unconditionally creating LayerNorm.

    transformers 4.57.6 ignores config.projector_use_layernorm and always
    creates nn.LayerNorm + applies it in forward. The model checkpoint for
    LFM2.5-VL-1.6B has projector_use_layernorm=False and ships no layer_norm
    weights, so the LayerNorm gets randomly initialized and corrupts features.
    Fixed in transformers 5.0.0. This patch backports the fix.
    """
    try:
        import transformers.models.lfm2_vl.modeling_lfm2_vl as lfm2_vl_module
    except Exception:
        return

    Projector = getattr(lfm2_vl_module, "Lfm2VlMultiModalProjector", None)
    if Projector is None:
        return

    # Already patched or already has conditional logic (transformers >= 5.0.0)
    if hasattr(Projector, "_unsloth_patched") or "use_layer_norm" in (getattr(Projector.__init__, "__code__", None) and Projector.__init__.__code__.co_varnames or ()):
        return

    import torch.nn as nn
    original_init = Projector.__init__
    original_forward = Projector.forward

    def patched_init(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        self.use_layer_norm = getattr(config, "projector_use_layernorm", True)
        if not self.use_layer_norm:
            self.layer_norm = None

    def patched_forward(self, image_features):
        image_features = self.pixel_unshuffle(image_features)
        if getattr(self, "use_layer_norm", True) and self.layer_norm is not None:
            image_features = self.layer_norm(image_features)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    Projector.__init__ = patched_init
    Projector.forward = patched_forward
    Projector._unsloth_patched = True
pass
TEMPORARY_PATCHES.append(patch_Lfm2VlMultiModalProjector)


def patch_peft_dispatch_bnb_4bit():
    """Fix PEFT dispatch_bnb_4bit accessing compress_statistics on non-Params4bit weights.

    In transformers 5.0+, BNB quantization loading order changed so weights may still be
    nn.Parameter (not Params4bit) when PEFT tries to access .compress_statistics and .quant_type.
    This wraps the original dispatch to catch AttributeError and provide defaults.
    """
    try:
        import peft.tuners.lora.bnb as peft_bnb
        original_dispatch = peft_bnb.dispatch_bnb_4bit
    except (ImportError, AttributeError):
        return

    if hasattr(original_dispatch, "_unsloth_patched"):
        return

    def safe_dispatch_bnb_4bit(target, adapter_name, **kwargs):
        try:
            return original_dispatch(target, adapter_name, **kwargs)
        except AttributeError as e:
            if "compress_statistics" in str(e) or "quant_type" in str(e):
                # Transformers 5.0+: weight not yet quantized as Params4bit
                # Retry after ensuring weight has needed attributes
                w = target.weight
                if not hasattr(w, "compress_statistics"):
                    w.compress_statistics = getattr(
                        target, "_bnb_compress_statistics", True
                    )
                if not hasattr(w, "quant_type"):
                    w.quant_type = getattr(target, "_bnb_quant_type", "nf4")
                return original_dispatch(target, adapter_name, **kwargs)
            raise

    safe_dispatch_bnb_4bit._unsloth_patched = True
    peft_bnb.dispatch_bnb_4bit = safe_dispatch_bnb_4bit
pass
TEMPORARY_PATCHES.append(patch_peft_dispatch_bnb_4bit)


def patch_trl_push_to_hub_token():
    """Ensure to_dict() always includes push_to_hub_token for TRL compat.

    TRL 0.22.x through 0.27.1 do bare dict_args.pop("push_to_hub_token") in
    SFTTrainer.__init__ and IterativeSFTTrainer.__init__. On transformers 5.0+,
    TrainingArguments.to_dict() no longer includes push_to_hub_token, so the
    bare pop raises KeyError. Fix: monkey-patch to_dict() to always include it.
    """
    try:
        from unsloth_zoo.utils import Version
        import transformers
        if Version(transformers.__version__) < Version("5.0.0"):
            return  # Not needed pre-5.0, to_dict() already includes it
        from transformers import TrainingArguments
        _original_to_dict = TrainingArguments.to_dict
        if getattr(_original_to_dict, "_unsloth_patched", False):
            return
        def _patched_to_dict(self):
            d = _original_to_dict(self)
            if "push_to_hub_token" not in d:
                d["push_to_hub_token"] = None
            return d
        _patched_to_dict._unsloth_patched = True
        TrainingArguments.to_dict = _patched_to_dict
    except Exception:
        pass
pass
TEMPORARY_PATCHES.append(patch_trl_push_to_hub_token)


def patch_trl_vision_model_mapping():
    """Fix DPO vision model detection for TRL 0.22.x + transformers 5.0+.

    TRL 0.22.x does a bare import of MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES from
    transformers.models.auto.modeling_auto. This name was removed in transformers
    5.0.0, replaced by MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES. The import
    failure prevents DPO trainer from loading at all.

    Fix: inject the old name as an alias of the new name into the transformers
    auto modeling module BEFORE TRL imports it, so the bare import succeeds.
    Also patch already-loaded DPO module if it fell back to empty dict.
    """
    try:
        import transformers.models.auto.modeling_auto as auto_mod
    except ImportError:
        return
    # If the old name already exists and is populated, nothing to do
    existing = getattr(auto_mod, "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES", None)
    if existing is not None and len(existing) > 0:
        return
    # Inject the old name as alias of the new name
    new_mapping = getattr(auto_mod, "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES", None)
    if new_mapping is not None:
        auto_mod.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = new_mapping
    # Also patch already-loaded DPO module if present
    try:
        import trl.trainer.dpo_trainer as dpo_mod
        dpo_current = getattr(dpo_mod, "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES", None)
        if (dpo_current is None or len(dpo_current) == 0) and new_mapping is not None:
            dpo_mod.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = new_mapping
    except ImportError:
        pass
pass
TEMPORARY_PATCHES.append(patch_trl_vision_model_mapping)


def patch_vllm_safe_apply_chat_template():
    """Fix vLLM safe_apply_chat_template for transformers 5.0+.

    transformers 5.0.0 changed apply_chat_template(tokenize=True) to default
    return_dict=True, returning BatchEncoding instead of list[int]. vLLM's
    safe_apply_chat_template doesn't pass return_dict=False, causing TypeError
    in _validate_model_input when max(BatchEncoding) returns a string key.

    Fix: wrap the original function to inject return_dict=False when tokenize=True.
    """
    try:
        from unsloth_zoo.utils import Version
        import transformers
        if Version(transformers.__version__) < Version("5.0.0"):
            return

        import vllm.renderers.hf as hf_mod
        _original_safe_apply = getattr(hf_mod, "safe_apply_chat_template", None)
        if _original_safe_apply is None:
            return
        if getattr(_original_safe_apply, "_unsloth_patched", False):
            return

        def _patched_safe_apply(model_config, tokenizer, conversation, *,
                                tools=None, chat_template=None, tokenize=True, **kwargs):
            if tokenize:
                kwargs["return_dict"] = False
            return _original_safe_apply(
                model_config, tokenizer, conversation,
                tools=tools, chat_template=chat_template, tokenize=tokenize,
                **kwargs,
            )
        _patched_safe_apply._unsloth_patched = True
        hf_mod.safe_apply_chat_template = _patched_safe_apply
    except Exception:
        pass
pass
TEMPORARY_PATCHES.append(patch_vllm_safe_apply_chat_template)
