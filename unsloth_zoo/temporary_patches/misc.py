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
        # Only convert if "content" is a string (AddedToken expects str), not a nested dict
        if isinstance(obj, dict) and "content" in obj and "__type" not in obj and isinstance(obj["content"], str):
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

    target_cls = transformers.models.csm.modeling_csm.CsmDepthDecoderForCausalLM

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
            device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
            codebook_indices = torch.arange(seq_len, device=device) + past_seen_tokens
        else:
            codebook_indices = cache_position

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
            hidden_states[:, slice_indices, :], codebook_indices[slice_indices]
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

    # Wrap with (self, *args, **kwargs) so check_args_kwargs accepts any
    # removed params (output_attentions, output_hidden_states, cache_position).
    # Copy the original class signature onto the wrapper so
    # transformers._validate_model_kwargs (used by generate) still sees
    # the real named parameters like backbone_last_hidden_state.
    _original_forward_signature = inspect.signature(target_cls.forward)
    _full_forward = forward
    def forward(self, *args, **kwargs):
        return _full_forward(self, *args, **kwargs)
    forward.__signature__ = _original_forward_signature
    patch_function(target_cls, "forward", forward, match_level="relaxed")
pass
TEMPORARY_PATCHES.append(patch_CsmDepthDecoderForCausalLM_forward)


def patch_CsmForConditionalGeneration_forward():
    try:
        import transformers.models.csm.modeling_csm
        from transformers.models.csm.modeling_csm import CsmOutputWithPast
        from transformers.loss.loss_utils import ForCausalLMLoss
    except Exception as e:
        return raise_error("CsmForConditionalGeneration.forward", e)

    target_cls = transformers.models.csm.modeling_csm.CsmForConditionalGeneration

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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

    # Preserve the original signature on the wrapper so inspect.signature
    # (used by transformers._validate_model_kwargs among others) still sees
    # the real named parameters.
    _original_forward_signature = inspect.signature(target_cls.forward)
    _full_forward = forward
    def forward(self, *args, **kwargs):
        return _full_forward(self, *args, **kwargs)
    forward.__signature__ = _original_forward_signature
    patch_function(target_cls, "forward", forward, match_level="relaxed")
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
    # Multi-GPU device_map flex_attention fix: cache_position[0] returns a
    # 0-dim tensor on one device, but inner_mask may run on another device.
    # Move offset tensors to the executing device inside the closure instead of
    # using .item(), which would cause a graph break under torch.compile tracing.
    if hasattr(masking_utils, "add_offsets_to_mask_function"):
        _original_add_offsets = getattr(
            masking_utils,
            "_unsloth_original_add_offsets_to_mask_function",
            masking_utils.add_offsets_to_mask_function,
        )
        masking_utils._unsloth_original_add_offsets_to_mask_function = _original_add_offsets
        def add_offsets_wrapper(mask_function, q_offset, kv_offset):
            _q_is_tensor = isinstance(q_offset, torch.Tensor)
            _kv_is_tensor = isinstance(kv_offset, torch.Tensor)
            def inner_mask(batch_idx, head_idx, q_idx, kv_idx):
                _q_off = q_offset.to(getattr(q_idx, "device", q_offset.device), non_blocking=True) if _q_is_tensor else q_offset
                _kv_off = kv_offset.to(getattr(kv_idx, "device", kv_offset.device), non_blocking=True) if _kv_is_tensor else kv_offset
                return mask_function(batch_idx, head_idx, q_idx + _q_off, kv_idx + _kv_off)
            return inner_mask
        masking_utils.add_offsets_to_mask_function = add_offsets_wrapper

    # Fix padding/packed mask functions for multi-GPU: captured tensors may be
    # on a different device than the indices passed during flex_attention vmap.
    # Cache per-device copies to avoid repeated cross-device transfers.
    if hasattr(masking_utils, "padding_mask_function"):
        masking_utils._unsloth_original_padding_mask_function = getattr(
            masking_utils,
            "_unsloth_original_padding_mask_function",
            masking_utils.padding_mask_function,
        )
        def padding_mask_wrapper(padding_mask):
            _cache = {padding_mask.device: padding_mask}
            def inner_mask(batch_idx, head_idx, q_idx, kv_idx):
                device = getattr(kv_idx, "device", padding_mask.device)
                pm = _cache.get(device)
                if pm is None:
                    pm = padding_mask.to(device, non_blocking=True)
                    _cache[device] = pm
                return pm[batch_idx, kv_idx]
            return inner_mask
        masking_utils.padding_mask_function = padding_mask_wrapper

    if hasattr(masking_utils, "packed_sequence_mask_function"):
        masking_utils._unsloth_original_packed_sequence_mask_function = getattr(
            masking_utils,
            "_unsloth_original_packed_sequence_mask_function",
            masking_utils.packed_sequence_mask_function,
        )
        def packed_sequence_mask_wrapper(packed_sequence_mask):
            _cache = {packed_sequence_mask.device: packed_sequence_mask}
            def inner_mask(batch_idx, head_idx, q_idx, kv_idx):
                device = getattr(q_idx, "device", packed_sequence_mask.device)
                pm = _cache.get(device)
                if pm is None:
                    pm = packed_sequence_mask.to(device, non_blocking=True)
                    _cache[device] = pm
                return pm[batch_idx, q_idx] == pm[batch_idx, kv_idx]
            return inner_mask
        masking_utils.packed_sequence_mask_function = packed_sequence_mask_wrapper

    masking_utils.__unsloth_mask_patch__ = True
pass
TEMPORARY_PATCHES.append(patch_transformers_masks)


def patch_sdpa_bool_causal_mask():
    """Fix unslothai/unsloth#4906: inf grad_norm on Qwen3.5 at seq_len > 65536.

    Upstream bug: pytorch/pytorch#162588. Cutlass SDPA returns garbage
    gradients on bool causal masks at seq_len >= 2**16 (bf16, head_dim=256,
    no flash-attn). Drop pure causal bool masks and call with is_causal=True;
    convert non-pure bool masks to float additive bias. Below 2**16 we skip
    the wrapper since the bug cannot fire.
    """
    if os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") == "1":
        return
    try:
        import transformers.integrations.sdpa_attention as sdpa_mod
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception as e:
        return raise_error("transformers.integrations.sdpa_attention", e)

    current = getattr(sdpa_mod, "sdpa_attention_forward", None)
    if current is None:
        return
    if getattr(current, "__unsloth_bool_causal_mask_fix__", False):
        return  # already installed

    _orig = current

    def sdpa_attention_forward_unsloth(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout = 0.0,
        scaling = None,
        is_causal = None,
        **kwargs,
    ):
        m = attention_mask

        # Below 2**16 the Cutlass bool-mask overflow cannot fire
        # (pytorch/pytorch#162588), so skip the wrapper. The pure-causal
        # rewrite picks a heavier SDPA backend and costs ~2.5 GB on
        # Gemma4-31B LoRA SFT (8192 seq_len).
        _q_len = query.shape[2] if query.dim() >= 3 else 0
        _mask_key_len = m.shape[-1] if isinstance(m, torch.Tensor) and m.dim() >= 1 else 0
        if _q_len < 65536 and _mask_key_len < 65536:
            return _orig(
                module, query, key, value, attention_mask,
                dropout = dropout, scaling = scaling, is_causal = is_causal,
                **kwargs,
            )

        # Non-causal modules (BERT, SigLIP) keep their masks; explicit param wins.
        resolved_is_causal = (
            is_causal if is_causal is not None
            else getattr(module, "is_causal", True)
        )
        if not resolved_is_causal:
            return _orig(
                module, query, key, value, attention_mask,
                dropout = dropout, scaling = scaling, is_causal = is_causal,
                **kwargs,
            )

        # Sliding-window layers (Gemma2, Mistral, Qwen2/3) keep windowed masks.
        if kwargs.get("sliding_window", None) is not None:
            return _orig(
                module, query, key, value, attention_mask,
                dropout = dropout, scaling = scaling, is_causal = is_causal,
                **kwargs,
            )

        # Hybrid models (Qwen3.5) may pass a dict keyed by layer type.
        if isinstance(m, dict):
            layer_type = getattr(module, "layer_type", None)
            if layer_type not in m:
                return _orig(
                    module, query, key, value, attention_mask,
                    dropout = dropout, scaling = scaling, is_causal = is_causal,
                    **kwargs,
                )
            m = m[layer_type]

        # Only intercept 4D bool self-attn masks (not cross-attn or kv-cache decode).
        if not (
            isinstance(m, torch.Tensor)
            and m.dtype == torch.bool
            and m.dim() == 4
            and m.shape[-1] == m.shape[-2]
            and m.shape[-1] == query.shape[2]
        ):
            return _orig(
                module, query, key, value, attention_mask,
                dropout = dropout, scaling = scaling, is_causal = is_causal,
                **kwargs,
            )

        # Pure lower-triangular check via two O(1) probes: upper-tri is False
        # and last row sees first col. Packed/padded masks fail the second.
        S = m.shape[-1]
        is_pure_causal = (
            (S < 2)
            or (
                not m[0, 0, 0, 1].item()
                and m[0, 0, S - 1, 0].item()
            )
        )

        if is_pure_causal:
            # Drop mask, use native is_causal path (avoids Cutlass >65536 bug, faster).
            return _orig(
                module, query, key, value, None,
                dropout = dropout, scaling = scaling, is_causal = True,
                **kwargs,
            )

        # Non-pure bool mask (packed sequences, custom patterns): convert to float
        # additive bias so SDPA dispatches to the working (non-bool) kernel.
        m_float = torch.where(m, 0.0, torch.finfo(query.dtype).min).to(query.dtype)
        return _orig(
            module, query, key, value, m_float,
            dropout = dropout, scaling = scaling, is_causal = False,
            **kwargs,
        )

    sdpa_attention_forward_unsloth.__unsloth_bool_causal_mask_fix__ = True
    sdpa_mod.sdpa_attention_forward = sdpa_attention_forward_unsloth
    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_attention_forward_unsloth
pass
TEMPORARY_PATCHES.append(patch_sdpa_bool_causal_mask)


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


def patch_apply_chat_template_return_dict():
    """Restore pre-5.0 return type for apply_chat_template(tokenize=True).

    transformers 5.0+ changed the default of return_dict from False to True.
    """
    try:
        from unsloth_zoo.utils import Version
        import transformers
        if Version(transformers.__version__) < Version("5.0.0"):
            return

        import inspect
        from transformers import PreTrainedTokenizerBase

        _original_apply = PreTrainedTokenizerBase.apply_chat_template
        if getattr(_original_apply, "_unsloth_patched", False):
            return

        try:
            _orig_sig = inspect.signature(_original_apply)
            _has_return_dict = "return_dict" in _orig_sig.parameters
        except Exception:
            _has_return_dict = True

        if not _has_return_dict:
            return

        def _patched_apply_chat_template(self, conversation, *args, **kwargs):
            tokenize = kwargs.get("tokenize", True)
            if tokenize and "return_dict" not in kwargs:
                kwargs["return_dict"] = False
            return _original_apply(self, conversation, *args, **kwargs)

        _patched_apply_chat_template._unsloth_patched = True
        PreTrainedTokenizerBase.apply_chat_template = _patched_apply_chat_template
    except Exception:
        pass
pass
TEMPORARY_PATCHES.append(patch_apply_chat_template_return_dict)


def patch_qwen2vl_image_processor_pixel_attrs():
    """Add max_pixels/min_pixels property shims to Qwen2VLImageProcessor.

    transformers 5.x removed these as direct instance attributes (they
    are now stored inside self.size["longest_edge"/"shortest_edge"]).
    vLLM 0.15.x accesses image_processor.max_pixels directly.
    Only patch on transformers >= 5.0.0 to avoid breaking 4.x where
    __init__ sets self.max_pixels as an instance attribute.
    """
    try:
        from unsloth_zoo.utils import Version
        import transformers
        if Version(transformers.__version__) < Version("5.0.0"):
            return  # 4.x already has max_pixels/min_pixels as instance attrs
    except Exception:
        return

    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    except ImportError:
        return

    # Only add shims if not already present as a class-level descriptor
    if not isinstance(Qwen2VLImageProcessor.__dict__.get("max_pixels"), property):
        @property
        def _max_pixels(self):
            return self.size.get("longest_edge", self.size.get("max_pixels", None))
        @property
        def _min_pixels(self):
            return self.size.get("shortest_edge", self.size.get("min_pixels", None))
        Qwen2VLImageProcessor.max_pixels = _max_pixels
        Qwen2VLImageProcessor.min_pixels = _min_pixels

    try:
        from transformers.models.qwen2_5_vl.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        if not isinstance(Qwen2_5_VLImageProcessor.__dict__.get("max_pixels"), property):
            Qwen2_5_VLImageProcessor.max_pixels = _max_pixels
            Qwen2_5_VLImageProcessor.min_pixels = _min_pixels
    except (ImportError, NameError):
        pass
pass
TEMPORARY_PATCHES.append(patch_qwen2vl_image_processor_pixel_attrs)


def patch_deepseek_v2_moe_alias():
    """Alias DeepseekV2MoE -> DeepseekV2Moe in transformers' modeling_deepseek_v2.

    Transformers 5.x renamed `DeepseekV2MoE` to `DeepseekV2Moe`, but several
    HF trust_remote_code modeling files (notably deepseek-ai/deepseek-ocr's
    `modeling_deepseekocr.py`) still import the old name and crash with
    `ImportError: cannot import name 'DeepseekV2MoE' ... Did you mean:
    'DeepseekV2Moe'?` before the model can be loaded. Restoring the old
    spelling as an alias unblocks those notebooks without touching upstream.
    """
    try:
        from transformers.models.deepseek_v2 import modeling_deepseek_v2 as ddv2
    except Exception:
        return  # transformers without deepseek_v2 -- nothing to patch.
    if hasattr(ddv2, "DeepseekV2MoE"):
        return
    new_class = getattr(ddv2, "DeepseekV2Moe", None)
    if new_class is None:
        return
    ddv2.DeepseekV2MoE = new_class
pass
TEMPORARY_PATCHES.append(patch_deepseek_v2_moe_alias)


def patch_transformers_legacy_import_utils():
    """Restore ``is_torch_fx_available`` on ``transformers.utils.import_utils``.

    Transformers 5.x dropped this helper. ``deepseek-ai/DeepSeek-OCR``'s
    trust_remote_code modeling file imports it at module load time and
    crashes with ``ImportError: cannot import name 'is_torch_fx_available'``
    before any config can be parsed. The helper used to be a thin
    ``importlib.util.find_spec("torch.fx") is not None`` check. Re-add it
    with the same semantics so legacy modeling files import cleanly.

    No-op when ``transformers`` is absent or when ``is_torch_fx_available``
    already exists (older transformers).
    """
    try:
        from transformers.utils import import_utils as iu
    except Exception:
        return
    if hasattr(iu, "is_torch_fx_available"):
        return

    import importlib.util as _ilu

    def is_torch_fx_available():
        return _ilu.find_spec("torch.fx") is not None

    iu.is_torch_fx_available = is_torch_fx_available
    # transformers re-exports a bunch of helpers via the parent ``utils``
    # package; mirror the symbol there too if it's already missing so
    # ``from transformers.utils import is_torch_fx_available`` works.
    try:
        from transformers import utils as tu
        if not hasattr(tu, "is_torch_fx_available"):
            tu.is_torch_fx_available = is_torch_fx_available
    except Exception:
        pass
pass
TEMPORARY_PATCHES.append(patch_transformers_legacy_import_utils)


def patch_llama_attention_back_compat_aliases():
    """Alias legacy ``Llama{Flash,Sdpa}Attention[2]`` symbols on
    ``transformers.models.llama.modeling_llama``.

    Transformers 5.x collapsed the per-backend attention subclasses into a
    single ``LlamaAttention`` that selects flash/sdpa/eager via
    ``config._attn_implementation``. Several HF trust_remote_code modeling
    files (notably ``deepseek-ai/DeepSeek-OCR``'s ``modeling_deepseekv2.py``,
    which lists ``LlamaFlashAttention2`` in a top-level ``from ... import``
    plus an ``ATTENTION_CLASSES`` dict but only ever uses MLA in practice)
    pre-date that refactor and crash with ``ImportError: cannot import name
    'LlamaFlashAttention2'`` before the model can load.

    Re-export the unified class under the old names so legacy import lines
    succeed. The behaviour is identical: at construction time
    ``LlamaAttention`` dispatches to the configured backend internally, so
    callers that referenced the old subclasses still pick up the right
    kernel path. Pre-5.0 transformers releases that still ship the
    subclasses (4.57.6 etc.) keep their originals unchanged via the
    ``hasattr`` guard.

    Idempotent. No-op when ``transformers`` is absent (MLX-only path) or
    when ``LlamaAttention`` itself is missing.
    """
    try:
        from transformers.models.llama import modeling_llama as mllama
    except Exception:
        return
    base = getattr(mllama, "LlamaAttention", None)
    if base is None:
        return
    for legacy_name in (
        "LlamaFlashAttention2",
        "LlamaSdpaAttention",
        "LlamaFlexAttention",
    ):
        if not hasattr(mllama, legacy_name):
            setattr(mllama, legacy_name, base)
pass
TEMPORARY_PATCHES.append(patch_llama_attention_back_compat_aliases)


def patch_deepseek_v2_config_default_ints():
    """Coerce ``None`` values for required-int fields on Deepseek-V2 configs.

    The official ``deepseek-ai/DeepSeek-OCR`` config.json has
    ``kv_lora_rank: null`` and ``q_lora_rank: null`` because the OCR
    head doesn't use MLA, but transformers 5.x ``StrictDataclass`` (and
    legacy ``PretrainedConfig`` users that read these as ints later)
    reject ``None`` for fields annotated ``int``. Walk the dict that
    ``DeepseekV2Config.from_dict`` consumes and replace ``None`` with the
    init default for known-required-int fields.

    No-op on transformers builds without ``deepseek_v2`` or when the
    fields are already non-None. Idempotent via the ``_unsloth_patched``
    sentinel on the wrapped ``from_dict``.
    """
    try:
        from transformers.models.deepseek_v2 import configuration_deepseek_v2 as cdv2
    except Exception:
        return
    Config = getattr(cdv2, "DeepseekV2Config", None)
    if Config is None:
        return
    orig_from_dict = getattr(Config, "from_dict", None)
    if orig_from_dict is None or getattr(orig_from_dict, "_unsloth_patched", False):
        return

    import inspect
    int_defaults = {}
    try:
        sig = inspect.signature(Config.__init__)
        for name, p in sig.parameters.items():
            if isinstance(p.default, int) and not isinstance(p.default, bool):
                int_defaults[name] = p.default
    except (TypeError, ValueError):
        return

    if not int_defaults:
        return

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        if isinstance(config_dict, dict):
            config_dict = dict(config_dict)
            for key, default in int_defaults.items():
                if key in config_dict and config_dict[key] is None:
                    config_dict[key] = default
        return orig_from_dict.__func__(cls, config_dict, **kwargs)

    from_dict._unsloth_patched = True
    Config.from_dict = from_dict
pass
TEMPORARY_PATCHES.append(patch_deepseek_v2_config_default_ints)


def patch_deepseek_ocr_prepare_inputs_for_generation():
    """Post-load patch for ``DeepseekOCRForCausalLM.prepare_inputs_for_generation``.

    The trust_remote_code ``modeling_deepseekocr.py`` shipped with
    ``deepseek-ai/DeepSeek-OCR`` predates the transformers Cache API
    change: when ``past_key_values`` is a ``Cache`` instance,
    ``past_length`` is taken from ``cache.seen_tokens`` which is a
    ``torch.Tensor`` in transformers 5.x but used to be a Python int.
    The next line then calls::

        cache_position = torch.arange(
            past_length, past_length + position_ids.shape[-1], device=...
        )

    and torch 2.10+ rejects Tensor positional args to ``torch.arange``:
    ``TypeError: arange() received an invalid combination of arguments``.

    Fix: wrap ``transformers.dynamic_module_utils.get_class_in_module`` so
    that after the dynamic module loads, we look for the
    ``DeepseekOCRForCausalLM`` (or the parent ``DeepseekV2ForCausalLM``)
    on the module and replace its ``prepare_inputs_for_generation`` with
    a thin wrapper that coerces tensor scalars to ``int`` before the
    arange call. The wrapper is identical in behaviour for the
    int-past_length case and the legacy 2D ``past_key_values`` tuple
    case, so existing code paths are preserved.

    Idempotent; sentinel both on the inner method and on the
    ``get_class_in_module`` wrapper so re-runs don't stack.
    """
    try:
        from transformers import dynamic_module_utils as dmu
    except Exception:
        return
    gcim = getattr(dmu, "get_class_in_module", None)
    if gcim is None or getattr(gcim, "_unsloth_arange_patched", False):
        return
    _orig = gcim

    def _patch_cls(cls):
        if cls is None or not isinstance(cls, type):
            return
        name = getattr(cls, "__name__", "")
        if "Deepseek" not in name and "DeepSeek" not in name:
            return
        method = getattr(cls, "prepare_inputs_for_generation", None)
        if method is None or getattr(method, "_unsloth_arange_patched", False):
            return
        import functools, torch as _torch

        @functools.wraps(method)
        def prepare_inputs_for_generation(self, *args, **kwargs):
            # Replicate the original method but coerce a Tensor-shaped
            # `past_length` to a Python int before torch.arange runs.
            # Simplest: monkey-patch torch.arange transiently in the
            # bound method's globals so we don't have to maintain a
            # forked copy of the upstream method body.
            mod_globals = getattr(method, "__globals__", None)
            _orig_arange = _torch.arange

            def _safe_arange(*aa, **kk):
                aa2 = tuple(
                    int(x.item()) if isinstance(x, _torch.Tensor) and x.numel() == 1 else x
                    for x in aa
                )
                return _orig_arange(*aa2, **kk)

            if mod_globals is not None and mod_globals.get("torch") is _torch:
                mod_globals["torch"] = _ShimTorch(_torch, _safe_arange)
                try:
                    return method(self, *args, **kwargs)
                finally:
                    mod_globals["torch"] = _torch
            else:
                # Fallback: temporarily monkey-patch torch.arange itself.
                _torch.arange = _safe_arange
                try:
                    return method(self, *args, **kwargs)
                finally:
                    _torch.arange = _orig_arange

        prepare_inputs_for_generation._unsloth_arange_patched = True
        cls.prepare_inputs_for_generation = prepare_inputs_for_generation

    class _ShimTorch:
        """Forward every attribute to ``torch`` except ``arange``."""
        __slots__ = ("_torch", "_arange")
        def __init__(self, t, arange):
            object.__setattr__(self, "_torch", t)
            object.__setattr__(self, "_arange", arange)
        def __getattr__(self, name):
            if name == "arange":
                return object.__getattribute__(self, "_arange")
            return getattr(object.__getattribute__(self, "_torch"), name)

    import sys as _sys

    def get_class_in_module(*args, **kwargs):
        cls = _orig(*args, **kwargs)
        try:
            _patch_cls(cls)
            # Also walk the SAME module the class lives in: trust_remote_code
            # only returns the one class actually requested (AutoConfig vs
            # AutoModel), but the module file declares siblings we also
            # need to patch (DeepseekV2 / DeepseekV3 / DeepseekOCR
            # ForCausalLM all live in the same dynamic module pair).
            mod_name = getattr(cls, "__module__", None)
            mod = _sys.modules.get(mod_name) if mod_name else None
            if mod is not None:
                for attr in dir(mod):
                    sibling = getattr(mod, attr, None)
                    if isinstance(sibling, type) and sibling is not cls:
                        _patch_cls(sibling)
        except Exception:
            pass
        return cls

    get_class_in_module._unsloth_arange_patched = True
    # Preserve any sentinels from the prior notebook_deps wrap.
    for attr in ("_unsloth_patched",):
        if getattr(_orig, attr, False):
            setattr(get_class_in_module, attr, True)
    dmu.get_class_in_module = get_class_in_module
    if hasattr(dmu, "get_class_from_dynamic_module"):
        try:
            src_globals = getattr(dmu.get_class_from_dynamic_module, "__globals__", None)
            if isinstance(src_globals, dict) and "get_class_in_module" in src_globals:
                src_globals["get_class_in_module"] = get_class_in_module
        except Exception:
            pass
pass
TEMPORARY_PATCHES.append(patch_deepseek_ocr_prepare_inputs_for_generation)
