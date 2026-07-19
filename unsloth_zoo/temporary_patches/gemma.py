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

from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import torch
import torch.nn as nn
import os
from .common import TEMPORARY_PATCHES, torch_compile
from .utils import (
    patch_function,
    process_output_options,
    KWARGS_TYPE,
    raise_error,
    ImageInput,
    PreTokenizedInput,
    TextInput,
    Cache,
    StaticCache,
    HybridCache,
    HAS_HYBRID_CACHE,
    Unpack,
    patch_function_past_key_values,
    dedent,
)
import inspect


def _prepare_gemma3_sdpa_attention_mask(attention_mask, query_states, key_states, sliding_window=None):
    if attention_mask is None or attention_mask.dim() != 2:
        return attention_mask

    q_len = query_states.shape[2]
    kv_len = key_states.shape[2]
    mask_len = attention_mask.shape[-1]
    if mask_len < kv_len:
        pad = torch.ones(
            (attention_mask.shape[0], kv_len - mask_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat((attention_mask, pad), dim=-1)
    elif mask_len > kv_len:
        attention_mask = attention_mask[:, -kv_len:]

    padding_mask = attention_mask[:, None, None, :].to(query_states.device) != 0
    if q_len == 1:
        return padding_mask

    q_positions = torch.arange(q_len, device=query_states.device)[:, None]
    k_positions = torch.arange(kv_len, device=query_states.device)[None, :]
    cache_offset = kv_len - q_len
    causal_mask = k_positions <= (q_positions + cache_offset)
    if sliding_window is not None:
        causal_mask = causal_mask & (k_positions > (q_positions + cache_offset - sliding_window))
    return padding_mask & causal_mask[None, None, :, :]


def _gemma3_rms_norm(x, weight, eps, out_dtype):
    # Inline Gemma3RMSNorm so compiled prepare() doesn't nest another compiled
    # forward, which broke Dynamo on older torch (unsloth#3535).
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    hidden_states_fp32 = x_fp32 * torch.rsqrt(variance + eps)
    output_fp32 = hidden_states_fp32 * (1.0 + weight.to(torch.float32))
    return output_fp32.to(out_dtype)


def _make_gemma3_attn_forwards(forward_function, has_cache_position):
    """Build the past_key_value / past_key_values forward variants."""
    functions = []
    if has_cache_position:
        def forward_past_key_value(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_value=None, cache_position=None, **kwargs):
            return forward_function(self, hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)
        def forward_past_key_values(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, cache_position=None, **kwargs):
            return forward_function(self, hidden_states, position_embeddings, attention_mask, past_key_values, cache_position, **kwargs)
    else:
        def forward_past_key_value(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_value=None, **kwargs):
            return forward_function(self, hidden_states, position_embeddings, attention_mask, past_key_value, kwargs.pop("cache_position", None), **kwargs)
        def forward_past_key_values(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
            return forward_function(self, hidden_states, position_embeddings, attention_mask, past_key_values, kwargs.pop("cache_position", None), **kwargs)
    functions.append(forward_past_key_value)
    functions.append(forward_past_key_values)
    return functions


def _resolve_truncation(padding, truncation, max_length):
    # HF activates "longest_first" truncation when max_length is set with padding=False and no explicit
    # truncation. We drop padding to pad manually, so pin the strategy the tokenizer would have derived
    # (from the caller's original padding) and pass it explicitly, keeping truncation behaviour identical.
    if truncation is not None:
        return truncation
    return "longest_first" if (max_length is not None and padding is False) else False
pass


def _fix_double_bos_and_pad(
    text_inputs, bos_token_id, pad_token_id, image_token_id,
    return_mm_token_type_ids, padding, padding_side, return_tensors,
    max_length = None, pad_to_multiple_of = None, model_max_length = None,
):
    # Gemma3 doubles the BOS (chat template + tokenizer). Strip the duplicate on every row and on
    # every per-token field returned (attention_mask, token_type_ids, special_tokens_mask,
    # offset_mapping, ...), rebuild mm token type ids, then pad each field so ragged rows stack.
    # Honours "do_not_pad"/max_length/model max/pad_to_multiple_of and the return_tensors=None list contract.
    n_rows = len(text_inputs["input_ids"])
    input_lens = [len(x) for x in text_inputs["input_ids"]]
    double_bos = [bos_token_id, bos_token_id]
    strip = [x[:2] == double_bos for x in text_inputs["input_ids"]]
    # only fields whose rows match the matching input_ids row length are token aligned; this keeps
    # non-aligned tokenizer outputs out of the per-row strip/pad. overflowing_tokens is a per-example
    # list of tails, so exclude it by name too in case a tail length coincidentally matches its row.
    non_aligned = {"overflowing_tokens", "overflow_to_sample_mapping", "num_truncated_tokens", "length"}
    per_token_keys = [
        k for k, v in text_inputs.items()
        if k not in non_aligned
        and isinstance(v, (list, tuple)) and len(v) == n_rows
        and all(isinstance(r, (list, tuple)) and len(r) == input_lens[i] for i, r in enumerate(v))
    ]
    for k in per_token_keys:
        text_inputs[k] = [r[1:] if strip[i] else r for i, r in enumerate(text_inputs[k])]
    if return_mm_token_type_ids:
        text_inputs["token_type_ids"] = [[int(y == image_token_id) for y in x] for x in text_inputs["input_ids"]]
        if "token_type_ids" not in per_token_keys: per_token_keys.append("token_type_ids")
    if padding not in (False, None, "do_not_pad"):
        if padding == "max_length" and max_length is not None:
            max_len = max_length
        elif padding == "max_length" and model_max_length is not None:
            max_len = model_max_length
        else:
            max_len = max((len(x) for x in text_inputs["input_ids"]), default = 0)
        if pad_to_multiple_of:
            max_len = -(-max_len // pad_to_multiple_of) * pad_to_multiple_of
        def fill_for(key):
            if key == "input_ids": return pad_token_id or 0
            if key == "special_tokens_mask": return 1
            sample = next((r for r in text_inputs[key] if r), None)
            return (0, 0) if (sample is not None and isinstance(sample[0], (tuple, list))) else 0
        def pad_seq(seq, fill):
            delta = max_len - len(seq)
            if delta <= 0: return list(seq)
            return ([fill]*delta + list(seq)) if padding_side == "left" else (list(seq) + [fill]*delta)
        for key in per_token_keys:
            fill = fill_for(key)
            text_inputs[key] = [pad_seq(x, fill) for x in text_inputs[key]]
    if "length" in text_inputs:   # return_length: report the post-strip/pad token counts
        text_inputs["length"] = [len(x) for x in text_inputs["input_ids"]]
    return text_inputs
pass


def patch_Gemma3Processor():
    import re
    try:
        import transformers.models.gemma3.processing_gemma3
        from transformers.models.gemma3.processing_gemma3 import Gemma3ProcessorKwargs
        from transformers.image_utils import make_nested_list_of_images
        from transformers.feature_extraction_utils import BatchFeature
        from transformers.utils import to_py_obj
    except Exception as e:
        return raise_error("Gemma3Processor.__call__", e)

    # Check if the target __call__ has `videos` or `audio` arguments
    target_call = transformers.models.gemma3.processing_gemma3.Gemma3Processor.__call__
    target_params = inspect.signature(target_call).parameters
    has_videos = "videos" in target_params
    has_audio = "audio" in target_params

    def _gemma3_call_impl(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: ImageInput = None,
        audio: Any = None,
        **kwargs: Unpack[Gemma3ProcessorKwargs],
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("Provide at least one of `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            Gemma3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        batched_images = None
        if images is not None:
            try:
                batched_images = make_nested_list_of_images(images)
            except ValueError as e:
                # Maybe it's texts and not images? Gemma3 defaults to images
                if text is None:
                    text = images
                    images = None
                else:
                    raise ValueError(e)
        pass
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        image_inputs = {}
        if images is not None:
            # batched_images = make_nested_list_of_images(images)
            image_inputs = self.image_processor(batched_images, **output_kwargs["images_kwargs"])

            # Create empty text to be replaced with placeholders
            if not text:
                text = [" ".join([self.boi_token] * len(images)) for images in batched_images]

            if len(batched_images) != len(text):
                raise ValueError(
                    f"Received inconsistently sized batches of images ({len(batched_images)}) and text ({len(text)})."
                )

            # Replace image tokens by the full expanded sequence
            batch_num_crops = to_py_obj(image_inputs.pop("num_crops"))
            text_with_crops = text
            for batch_idx, (prompt, images_for_item, num_crops_for_item) in enumerate(zip(text, batched_images, batch_num_crops)):
                image_indexes = [m.start() for m in re.finditer(self.boi_token, prompt)]

                if len(images_for_item) != len(image_indexes):
                    raise ValueError(
                        f"Prompt contained {len(image_indexes)} image tokens but received {len(images_for_item)} images."
                    )

                iterable_num_crops = num_crops_for_item

                if isinstance(num_crops_for_item, int):
                    if len(image_indexes) > 0:
                        iterable_num_crops = [num_crops_for_item] + [0] * (len(image_indexes) - 1)
                    else:
                        iterable_num_crops = []

                # Insert additional image tokens for Pan-and-Scan crops
                for num, idx in reversed(list(zip(iterable_num_crops, image_indexes))):
                    if isinstance(num, int) and num > 0:
                        formatted_image_text = (
                            f"Here is the original image {self.boi_token} and here are some crops to help you see better "
                            + " ".join([self.boi_token] * num)
                        )
                        prompt = prompt[:idx] + formatted_image_text + prompt[idx + len(self.boi_token) :]
                        text_with_crops[batch_idx] = prompt

            # Expand placeholder image tokens to the full image token sequence
            text = [prompt.replace(self.boi_token, self.full_image_sequence) for prompt in text]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        # text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"], return_tensors="np")
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", True)

        # Tokenize unpadded so the double-BOS strip cannot desync row lengths, then pad afterwards.
        padding = output_kwargs["text_kwargs"].pop("padding", False)
        padding_side = output_kwargs["text_kwargs"].pop("padding_side", None) or \
            getattr(self.tokenizer, "padding_side", "left")
        # HF derives truncation from padding + max_length (max_length with padding=False and no explicit
        # truncation truncates). We drop padding to pad manually, so pin the truncation the tokenizer would
        # have used from the original padding, keeping truncation behaviour identical.
        max_length = output_kwargs["text_kwargs"].get("max_length", None)
        output_kwargs["text_kwargs"]["truncation"] = _resolve_truncation(
            padding, output_kwargs["text_kwargs"].get("truncation", None), max_length)
        text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
        # ignore the tokenizer's uninitialised model_max_length sentinel (~1e30) for "max_length" padding
        _mml = getattr(self.tokenizer, "model_max_length", None)
        if not (isinstance(_mml, int) and 0 < _mml < int(1e15)): _mml = None
        text_inputs = _fix_double_bos_and_pad(
            text_inputs, self.tokenizer.bos_token_id, self.tokenizer.pad_token_id,
            self.image_token_id, return_mm_token_type_ids, padding, padding_side, return_tensors,
            max_length = max_length,
            pad_to_multiple_of = output_kwargs["text_kwargs"].get("pad_to_multiple_of", None),
            model_max_length = _mml,
        )
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)
    pass

    if has_videos or has_audio:
        __call__ = _gemma3_call_impl
    else:
        def __call__(
            self,
            images: ImageInput = None,
            text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            **kwargs: Unpack[Gemma3ProcessorKwargs],
        ) -> BatchFeature:
            videos = kwargs.pop("videos", None)
            audio = kwargs.pop("audio", None)
            return _gemma3_call_impl(self, images=images, text=text, videos=videos, audio=audio, **kwargs)

    patch_function(transformers.models.gemma3.processing_gemma3.Gemma3Processor, "__call__", __call__, match_level="relaxed")
pass
TEMPORARY_PATCHES.append(patch_Gemma3Processor)


def patch_Gemma3ForConditionalGeneration_causal_mask():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.gemma3.modeling_gemma3
        transformers.models.gemma3.modeling_gemma3.Gemma3Model
        transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration
    except Exception as e:
        return raise_error("Gemma3ForConditionalGeneration._update_causal_mask", e)

    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids,
        past_key_values,
        cache_position,
        input_tensor,
        is_training: bool = False,
    ):
        if self.config.text_config._attn_implementation == "flash_attention_2":
            return attention_mask

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted
            # form and requires no inversion or slicing.
            return attention_mask

        using_static_cache = isinstance(past_key_values, StaticCache)
        min_dtype = torch.finfo(torch.float16).min
        inputs_lead_dim, sequence_length = input_tensor.shape[:2]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        elif HAS_HYBRID_CACHE and isinstance(past_key_values, HybridCache):
            # Gated on HAS_HYBRID_CACHE: transformers 5.x removed HybridCache,
            # and the typing.Any fallback from utils.py would raise here.
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            return attention_mask

        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=torch.float16, device=cache_position.device
        )

        # Causal diagonal mask only if training, otherwise attend to the whole prefix. Training-specific attn for prefix is handled below
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)

        # Apply bidirectional mask on images if token type ids are provided
        if token_type_ids is not None and sequence_length != 1:
            token_type_mask = token_type_ids.unsqueeze(1) == token_type_ids.unsqueeze(2)
            token_type_mask[token_type_ids == 0] = False  # if text token do not change anything
            token_type_mask = token_type_mask.unsqueeze(1).to(causal_mask.device, dtype=torch.bool)
            causal_mask = causal_mask.clone()
            causal_mask[:, :, :, :sequence_length] = causal_mask[:, :, :, :sequence_length].masked_fill(
                token_type_mask, 0.0
            )

        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]

            # Then apply padding mask (will mask pad tokens)
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

        return causal_mask
    pass
    if hasattr(transformers.models.gemma3.modeling_gemma3, "Gemma3Model"):
        patch_function(transformers.models.gemma3.modeling_gemma3.Gemma3Model, "_update_causal_mask", _update_causal_mask)
    else:
        patch_function(transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration, "_update_causal_mask", _update_causal_mask)
pass
TEMPORARY_PATCHES.append(patch_Gemma3ForConditionalGeneration_causal_mask)


def patch_Gemma3TextScaledWordEmbedding():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.gemma3.modeling_gemma3
        transformers.models.gemma3.modeling_gemma3.Gemma3TextScaledWordEmbedding
    except Exception as e:
        return raise_error("Gemma3ForConditionalGeneration._update_causal_mask", e)

    def forward(self, input_ids: torch.Tensor):
        input_embeds = torch.nn.functional.embedding(
            input_ids,
            weight = self.weight,
            padding_idx = self.padding_idx,
        )
        return input_embeds.to(torch.float32) * self.embed_scale
    pass
    patch_function(transformers.models.gemma3.modeling_gemma3.Gemma3TextScaledWordEmbedding, "forward", forward, fullgraph = True)
pass
TEMPORARY_PATCHES.append(patch_Gemma3TextScaledWordEmbedding)


def patch_Gemma3RMSNorm():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.gemma3.modeling_gemma3
        transformers.models.gemma3.modeling_gemma3.Gemma3RMSNorm
    except Exception as e:
        return raise_error("Gemma3RMSNorm.forward", e)

    def forward(self, x): # x can be fp32 (from embeddings) or fp16 (from MLP/Attn)
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        hidden_states_fp32 = x_fp32 * torch.rsqrt(variance + self.eps)

        # self.weight may be bf16; cast to fp32 for the (1.0 + weight) op.
        output_fp32 = hidden_states_fp32 * (1.0 + self.weight.to(torch.float32))

        # Clamp to fp16 range before casting back to fp16
        fp16_max = torch.finfo(torch.float16).max
        fp16_min = torch.finfo(torch.float16).min
        clamped_output_fp32 = torch.clamp(output_fp32, min=fp16_min, max=fp16_max)

        return clamped_output_fp32.to(torch.float16) # Output fp16
    pass
    patch_function(transformers.models.gemma3.modeling_gemma3.Gemma3RMSNorm, "forward", forward, fullgraph = True)
pass
TEMPORARY_PATCHES.append(patch_Gemma3RMSNorm)


def patch_Gemma3MLP():
    try:
        import transformers.models.gemma3.modeling_gemma3
        transformers.models.gemma3.modeling_gemma3.Gemma3MLP
    except Exception as e:
        return raise_error("Gemma3MLP.forward", e)

    def forward(self, x):
        # If forcing float32, keep the original float32 path.
        if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
            gate_proj_out = self.gate_proj(x)
            up_proj_out = self.up_proj(x)

            # Upcast to fp32
            gate_proj_fp32 = gate_proj_out.to(torch.float32)
            up_proj_fp32 = up_proj_out.to(torch.float32)
            activated_fp32 = self.act_fn(gate_proj_fp32) # Activation in fp32
            intermediate_fp32 = activated_fp32 * up_proj_fp32 # Product in fp32

            # Downcast and down_proj
            intermediate_fp16 = intermediate_fp32.to(torch.float16)
            down_proj_out = self.down_proj(intermediate_fp16)
            return down_proj_out

        # Otherwise, keep inputs in their native dtype and only cast
        # the intermediate to the down_proj compute dtype if needed.
        intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        # Prefer compute_dtype (bnb Linear4bit) if present; fallback to weight dtype
        target_dtype = getattr(self.down_proj, "compute_dtype", None)
        if target_dtype is None:
            weight = getattr(self.down_proj, "weight", None)
            target_dtype = getattr(weight, "dtype", None)

        if target_dtype is not None:
            try:
                is_float = torch.is_floating_point(torch.empty((), dtype=target_dtype))
            except Exception:
                is_float = False
            if is_float and intermediate.dtype != target_dtype:
                intermediate = intermediate.to(target_dtype)

        return self.down_proj(intermediate)
    pass
    patch_function(transformers.models.gemma3.modeling_gemma3.Gemma3MLP, "forward", forward, fullgraph = False)
pass
TEMPORARY_PATCHES.append(patch_Gemma3MLP)


def patch_Gemma3Attention():
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.gemma3.modeling_gemma3
        transformers.models.gemma3.modeling_gemma3.Gemma3Attention
        from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward
    except Exception as e:
        return raise_error("Gemma3Attention.forward", e)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
    scaled_dot_product_attention = torch.compiler.disable(scaled_dot_product_attention, recursive = True)
    torch_jit_is_tracing = torch.jit.is_tracing

    def prepare(
        hidden_states,
        query_states_fp16,
        key_states_fp16,
        value_states_fp16,
        query_hidden_shape,
        kv_hidden_shape,
        position_embeddings,
        attention_mask,
        q_norm,
        k_norm,
    ):
        # 2. Upcast Q, K, V for norm and RoPE, and then transpose for attention
        # (bsz, num_specific_heads, q_len, head_dim)
        query_states_fp32 = query_states_fp16.view(query_hidden_shape).to(torch.float32).transpose(1, 2)
        key_states_fp32   = key_states_fp16.view(kv_hidden_shape).to(torch.float32).transpose(1, 2)
        value_states_fp32 = value_states_fp16.view(kv_hidden_shape).to(torch.float32).transpose(1, 2) # V for attention also fp32

        # 3. Normalization: inline RMSNorm, then clamp+emit fp16 to match patch_Gemma3RMSNorm.
        fp16_max = torch.finfo(torch.float16).max
        query_norm_out_fp16 = _gemma3_rms_norm(query_states_fp32, q_norm.weight, q_norm.eps, torch.float32)
        key_norm_out_fp16   = _gemma3_rms_norm(key_states_fp32,   k_norm.weight, k_norm.eps, torch.float32)
        query_norm_out_fp16 = torch.clamp(query_norm_out_fp16, min=-fp16_max, max=fp16_max).to(torch.float16)
        key_norm_out_fp16   = torch.clamp(key_norm_out_fp16,   min=-fp16_max, max=fp16_max).to(torch.float16)

        query_states_fp32 = query_norm_out_fp16.to(torch.float32)
        key_states_fp32   = key_norm_out_fp16.to(torch.float32)

        # 4. Rotary Positional Embeddings in fp32
        if not (isinstance(position_embeddings, tuple) and len(position_embeddings) == 2):
            raise ValueError("Position embeddings not provided as (cos, sin) tuple to Gemma3Attention")

        cos, sin = position_embeddings
        cos_fp32 = cos.to(torch.float32)
        sin_fp32 = sin.to(torch.float32)
        query_states_fp32, key_states_fp32 = apply_rotary_pos_emb(query_states_fp32, key_states_fp32, cos = cos_fp32, sin = sin_fp32)

        # 6. Core Attention mechanism (SDPA) in fp32
        attn_mask_for_sdpa = attention_mask
        if isinstance(attn_mask_for_sdpa, torch.Tensor) and attn_mask_for_sdpa.dtype != torch.bool:
            attn_mask_for_sdpa = attn_mask_for_sdpa.to(torch.float32)
        return (
            query_states_fp32.contiguous(),
            key_states_fp32.contiguous(),
            value_states_fp32.contiguous(),
            cos_fp32,
            sin_fp32,
            attn_mask_for_sdpa,
        )
    pass
    prepare = torch_compile(prepare, fullgraph = True, dynamic = True)

    def forward_function(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: KWARGS_TYPE,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]

        # Head shapes (fall back to config if attrs are absent)
        num_heads = getattr(self, "num_heads", self.config.num_attention_heads)
        num_key_value_heads = getattr(self, "num_key_value_heads", self.config.num_key_value_heads)
        head_dim = self.head_dim

        # Projection view shape: (bsz, q_len, num_specific_heads, head_dim)
        query_hidden_shape = (bsz, q_len, num_heads, head_dim)
        kv_hidden_shape    = (bsz, q_len, num_key_value_heads, head_dim)

        # 1. Projections (q, k, v) in fp16
        query_states_fp16 = self.q_proj(hidden_states) # output fp16
        key_states_fp16   = self.k_proj(hidden_states) # output fp16
        value_states_fp16 = self.v_proj(hidden_states) # output fp16

        # 2. Upcast Q, K, V for norm and RoPE, and then transpose for attention
        # (bsz, num_specific_heads, q_len, head_dim)
        """ ####### REPLACED WITH TORCH_COMPILED_MODULE
        query_states_fp32 = query_states_fp16.view(query_hidden_shape).to(torch.float32).transpose(1, 2)
        key_states_fp32   = key_states_fp16.view(kv_hidden_shape).to(torch.float32).transpose(1, 2)
        value_states_fp32 = value_states_fp16.view(kv_hidden_shape).to(torch.float32).transpose(1, 2) # V for attention also fp32

        # 3. Normalization (q_norm, k_norm are RMSNorms)
        query_norm_out_fp16 = self.q_norm(query_states_fp32)
        key_norm_out_fp16   = self.k_norm(key_states_fp32)

        query_states_fp32 = query_norm_out_fp16.to(torch.float32)
        key_states_fp32   = key_norm_out_fp16.to(torch.float32)

        # 4. Rotary Positional Embeddings in fp32
        if not (isinstance(position_embeddings, tuple) and len(position_embeddings) == 2):
            raise ValueError("Position embeddings not provided as (cos, sin) tuple to Gemma3Attention")

        cos, sin = position_embeddings
        cos_fp32 = cos.to(torch.float32)
        sin_fp32 = sin.to(torch.float32)
        query_states_fp32, key_states_fp32 = apply_rotary_pos_emb(query_states_fp32, key_states_fp32, cos = cos_fp32, sin = sin_fp32)
        """
        (
            query_states_fp32,
            key_states_fp32,
            value_states_fp32,
            cos_fp32,
            sin_fp32,
            attn_mask_for_sdpa,
        ) = prepare(
            hidden_states,
            query_states_fp16,
            key_states_fp16,
            value_states_fp16,
            query_hidden_shape,
            kv_hidden_shape,
            position_embeddings,
            attention_mask,
            self.q_norm,
            self.k_norm,
        )

        # 5. KV Cache update (using fp32 K, V)
        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin_fp32, "cos": cos_fp32, "cache_position": cache_position
            }
            # Add sliding_window if the attribute exists (common in newer models)
            if hasattr(self, "sliding_window") and self.sliding_window is not None:
                 cache_kwargs["sliding_window"] = self.sliding_window
            key_states_fp32, value_states_fp32 = past_key_value.update(
                key_states_fp32, value_states_fp32, self.layer_idx, cache_kwargs
            )

        # 6. Core Attention mechanism (SDPA) in fp32
        """ ####### REPLACED WITH TORCH_COMPILED_MODULE
        attn_mask_for_sdpa = attention_mask
        if attn_mask_for_sdpa is not None and attn_mask_for_sdpa.dtype != torch.bool:
            attn_mask_for_sdpa = attn_mask_for_sdpa.to(torch.float32)
        """
        # output_attentions = kwargs.get("output_attentions", False)
        attn_impl = getattr(self.config, "_attn_implementation", "sdpa")
        if attn_impl == "flex_attention":
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]
            attn_output_fp32, attn_weights = attention_interface(
                self,
                query_states_fp32,
                key_states_fp32,
                value_states_fp32,
                attn_mask_for_sdpa,
                dropout = self.attention_dropout if self.training else 0.0,
                scaling = getattr(self, "scaling", None),
                sliding_window = getattr(self, "sliding_window", None),
                **kwargs,
            )
        else:
            attn_mask_for_sdpa = _prepare_gemma3_sdpa_attention_mask(
                attn_mask_for_sdpa,
                query_states_fp32,
                key_states_fp32,
                getattr(self, "sliding_window", None),
            )
            is_causal = query_states_fp32.shape[2] > 1 and attn_mask_for_sdpa is None and getattr(self, "is_causal", True)
            # During jit tracing shapes are tensors, so is_causal may be a tensor; SDPA needs a bool.
            if torch_jit_is_tracing() and isinstance(is_causal, torch.Tensor): is_causal = is_causal.item()
            attn_output_fp32 = scaled_dot_product_attention(
                query_states_fp32.contiguous(),
                key_states_fp32.contiguous(),
                value_states_fp32.contiguous(),
                attn_mask = attn_mask_for_sdpa,
                dropout_p = self.attention_dropout if self.training else 0.0,
                is_causal = is_causal,
                scale = getattr(self, "scaling", None), # Use self.scaling if defined, else SDPA default
                enable_gqa = getattr(self, "num_key_value_groups", 1) != 1,
            )
            attn_weights = None # Defaulting to None

        # 7. Reshape and downcast for output projection.
        # SDPA returns (bsz, heads, q_len, head_dim) needing transpose;
        # flex_attention returns (bsz, q_len, heads, head_dim) already transposed.
        if attn_impl != "flex_attention":
            attn_output_fp32 = attn_output_fp32.transpose(1, 2).contiguous()

        attn_output_fp32 = attn_output_fp32.reshape(bsz, q_len, -1)

        attn_output_fp16 = attn_output_fp32.to(torch.float16)

        # 8. Output Projection (o_proj) in fp16
        attn_output_projected = self.o_proj(attn_output_fp16) # fp16 output

        return attn_output_projected, attn_weights # 3-tuple return
    pass

    has_cache_position = "cache_position" in inspect.signature(
        transformers.models.gemma3.modeling_gemma3.Gemma3Attention.forward
    ).parameters
    functions = _make_gemma3_attn_forwards(forward_function, has_cache_position)
    patch_function_past_key_values(transformers.models.gemma3.modeling_gemma3.Gemma3Attention, "forward", functions, match_level="relaxed")
pass
TEMPORARY_PATCHES.append(patch_Gemma3Attention)


def patch_Gemma3RMSNorm_generic():
    # Must do this since torch.compile cannot trace through def prepare for q_norm, k_norm
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1": return
    try:
        import transformers.models.gemma3.modeling_gemma3
        transformers.models.gemma3.modeling_gemma3.Gemma3RMSNorm
    except Exception as e:
        return raise_error("Gemma3RMSNorm.forward", e)

    def forward(self, x):
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        hidden_states_fp32 = x_fp32 * torch.rsqrt(variance + self.eps)
        output_fp32 = hidden_states_fp32 * (1.0 + self.weight.to(torch.float32))
        return output_fp32.to(x.dtype)
    pass
    patch_function(transformers.models.gemma3.modeling_gemma3.Gemma3RMSNorm, "forward", forward, fullgraph = True)
pass
TEMPORARY_PATCHES.append(patch_Gemma3RMSNorm_generic)


def patch_Gemma3Attention_generic():
    # Non float16 forced also has some benefits
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1": return
    try:
        import transformers.models.gemma3.modeling_gemma3
        transformers.models.gemma3.modeling_gemma3.Gemma3Attention
        from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward
    except Exception as e:
        return raise_error("Gemma3Attention.forward", e)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
    scaled_dot_product_attention = torch.compiler.disable(scaled_dot_product_attention, recursive = True)
    torch_jit_is_tracing = torch.jit.is_tracing

    def prepare(
        hidden_states,
        query_states_fp16,
        key_states_fp16,
        value_states_fp16,
        query_hidden_shape,
        kv_hidden_shape,
        position_embeddings,
        attention_mask,
        q_norm,
        k_norm,
    ):
        # 2. Upcast Q, K, V for norm and RoPE, and then transpose for attention
        # (bsz, num_specific_heads, q_len, head_dim)
        query_states_fp32 = query_states_fp16.view(query_hidden_shape).transpose(1, 2)
        key_states_fp32   = key_states_fp16.view(kv_hidden_shape).transpose(1, 2)
        value_states_fp32 = value_states_fp16.view(kv_hidden_shape).transpose(1, 2) # V for attention also fp32

        # 3. Normalization: inline RMSNorm, output dtype mirrors input to match patch_Gemma3RMSNorm_generic.
        query_norm_out_fp16 = _gemma3_rms_norm(query_states_fp32, q_norm.weight, q_norm.eps, query_states_fp32.dtype)
        key_norm_out_fp16   = _gemma3_rms_norm(key_states_fp32,   k_norm.weight, k_norm.eps, key_states_fp32.dtype)

        query_states_fp32 = query_norm_out_fp16#.to(torch.float32)
        key_states_fp32   = key_norm_out_fp16#.to(torch.float32)

        # 4. Rotary Positional Embeddings in fp32
        if not (isinstance(position_embeddings, tuple) and len(position_embeddings) == 2):
            raise ValueError("Position embeddings not provided as (cos, sin) tuple to Gemma3Attention")

        cos, sin = position_embeddings
        cos_fp32 = cos#.to(torch.float32)
        sin_fp32 = sin#.to(torch.float32)
        query_states_fp32, key_states_fp32 = apply_rotary_pos_emb(query_states_fp32, key_states_fp32, cos = cos_fp32, sin = sin_fp32)

        # 6. Core Attention mechanism (SDPA) in fp32
        attn_mask_for_sdpa = attention_mask
        if isinstance(attn_mask_for_sdpa, torch.Tensor) and attn_mask_for_sdpa.dtype != torch.bool:
            attn_mask_for_sdpa = attn_mask_for_sdpa#.to(torch.float32)
            attn_mask_for_sdpa = attn_mask_for_sdpa.to(query_states_fp32.dtype)
        return (
            query_states_fp32.contiguous(),
            key_states_fp32.contiguous(),
            value_states_fp32.contiguous(),
            cos_fp32,
            sin_fp32,
            attn_mask_for_sdpa,
        )
    pass
    # We must patch RMSNorm as well since q_norm, k_norm can't be traced correctly!
    prepare = torch_compile(prepare, fullgraph = True, dynamic = True)

    def forward_function(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: KWARGS_TYPE,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]

        # Head shapes (fall back to config if attrs are absent)
        num_heads = getattr(self, "num_heads", self.config.num_attention_heads)
        num_key_value_heads = getattr(self, "num_key_value_heads", self.config.num_key_value_heads)
        head_dim = self.head_dim

        # Projection view shape: (bsz, q_len, num_specific_heads, head_dim)
        query_hidden_shape = (bsz, q_len, num_heads, head_dim)
        kv_hidden_shape    = (bsz, q_len, num_key_value_heads, head_dim)

        # 1. Projections (q, k, v) in fp16
        query_states_fp16 = self.q_proj(hidden_states) # output fp16
        key_states_fp16   = self.k_proj(hidden_states) # output fp16
        value_states_fp16 = self.v_proj(hidden_states) # output fp16

        # 2. Upcast Q, K, V for norm and RoPE, and then transpose for attention
        # (bsz, num_specific_heads, q_len, head_dim)
        """ ####### REPLACED WITH TORCH_COMPILED_MODULE
        query_states_fp32 = query_states_fp16.view(query_hidden_shape).to(torch.float32).transpose(1, 2)
        key_states_fp32   = key_states_fp16.view(kv_hidden_shape).to(torch.float32).transpose(1, 2)
        value_states_fp32 = value_states_fp16.view(kv_hidden_shape).to(torch.float32).transpose(1, 2) # V for attention also fp32

        # 3. Normalization (q_norm, k_norm are RMSNorms)
        query_norm_out_fp16 = self.q_norm(query_states_fp32)
        key_norm_out_fp16   = self.k_norm(key_states_fp32)

        query_states_fp32 = query_norm_out_fp16.to(torch.float32)
        key_states_fp32   = key_norm_out_fp16.to(torch.float32)

        # 4. Rotary Positional Embeddings in fp32
        if not (isinstance(position_embeddings, tuple) and len(position_embeddings) == 2):
            raise ValueError("Position embeddings not provided as (cos, sin) tuple to Gemma3Attention")

        cos, sin = position_embeddings
        cos_fp32 = cos.to(torch.float32)
        sin_fp32 = sin.to(torch.float32)
        query_states_fp32, key_states_fp32 = apply_rotary_pos_emb(query_states_fp32, key_states_fp32, cos = cos_fp32, sin = sin_fp32)
        """
        (
            query_states_fp32,
            key_states_fp32,
            value_states_fp32,
            cos_fp32,
            sin_fp32,
            attn_mask_for_sdpa,
        ) = prepare(
            hidden_states,
            query_states_fp16,
            key_states_fp16,
            value_states_fp16,
            query_hidden_shape,
            kv_hidden_shape,
            position_embeddings,
            attention_mask,
            self.q_norm,
            self.k_norm,
        )

        # 5. KV Cache update (using fp32 K, V)
        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin_fp32, "cos": cos_fp32, "cache_position": cache_position
            }
            # Add sliding_window if the attribute exists (common in newer models)
            if hasattr(self, "sliding_window") and self.sliding_window is not None:
                 cache_kwargs["sliding_window"] = self.sliding_window
            key_states_fp32, value_states_fp32 = past_key_value.update(
                key_states_fp32, value_states_fp32, self.layer_idx, cache_kwargs
            )

        # 6. Core Attention mechanism (SDPA) in fp32
        """ ####### REPLACED WITH TORCH_COMPILED_MODULE
        attn_mask_for_sdpa = attention_mask
        if attn_mask_for_sdpa is not None and attn_mask_for_sdpa.dtype != torch.bool:
            attn_mask_for_sdpa = attn_mask_for_sdpa.to(torch.float32)
        """
        # output_attentions = kwargs.get("output_attentions", False)
        attn_impl = getattr(self.config, "_attn_implementation", "sdpa")
        if attn_impl == "flex_attention":
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]
            attn_output_fp32, attn_weights = attention_interface(
                self,
                query_states_fp32,
                key_states_fp32,
                value_states_fp32,
                attn_mask_for_sdpa,
                dropout = self.attention_dropout if self.training else 0.0,
                scaling = getattr(self, "scaling", None),
                sliding_window = getattr(self, "sliding_window", None),
                **kwargs,
            )
        else:
            attn_mask_for_sdpa = _prepare_gemma3_sdpa_attention_mask(
                attn_mask_for_sdpa,
                query_states_fp32,
                key_states_fp32,
                getattr(self, "sliding_window", None),
            )
            is_causal = query_states_fp32.shape[2] > 1 and attn_mask_for_sdpa is None and getattr(self, "is_causal", True)
            # During jit tracing shapes are tensors, so is_causal may be a tensor; SDPA needs a bool.
            if torch_jit_is_tracing() and isinstance(is_causal, torch.Tensor): is_causal = is_causal.item()
            attn_output_fp32 = scaled_dot_product_attention(
                query_states_fp32.contiguous(),
                key_states_fp32.contiguous(),
                value_states_fp32.contiguous(),
                attn_mask = attn_mask_for_sdpa,
                dropout_p = self.attention_dropout if self.training else 0.0,
                is_causal = is_causal,
                scale = getattr(self, "scaling", None), # Use self.scaling if defined, else SDPA default
                enable_gqa = getattr(self, "num_key_value_groups", 1) != 1,
            )
            attn_weights = None # Defaulting to None

        # 7. Reshape and downcast for output projection.
        # SDPA returns (bsz, heads, q_len, head_dim) needing transpose;
        # flex_attention returns (bsz, q_len, heads, head_dim) already transposed.
        if attn_impl != "flex_attention":
            attn_output_fp32 = attn_output_fp32.transpose(1, 2).contiguous()

        attn_output_fp32 = attn_output_fp32.reshape(bsz, q_len, -1)

        attn_output_fp16 = attn_output_fp32#.to(torch.float16)

        # 8. Output Projection (o_proj) in fp16
        attn_output_projected = self.o_proj(attn_output_fp16) # fp16 output

        return attn_output_projected, attn_weights # 3-tuple return
    pass

    has_cache_position = "cache_position" in inspect.signature(
        transformers.models.gemma3.modeling_gemma3.Gemma3Attention.forward
    ).parameters
    functions = _make_gemma3_attn_forwards(forward_function, has_cache_position)
    patch_function_past_key_values(transformers.models.gemma3.modeling_gemma3.Gemma3Attention, "forward", functions, match_level="relaxed")
pass
TEMPORARY_PATCHES.append(patch_Gemma3Attention_generic)
