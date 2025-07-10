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
from .common import TEMPORARY_PATCHES, torch_compile_options
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
    Unpack,
)


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

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos = None,
        audio = None,
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

        text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
        # Fix double BOS tokens
        double_bos_token_id = [self.tokenizer.bos_token_id]*2
        input_ids = text_inputs["input_ids"]
        text_inputs["input_ids"] = [x[1:] if x[:2] == double_bos_token_id else x for x in input_ids]

        # Add token type ids manually, as tokenizer can't do arbitrary position token types
        # [TODO] FAILS for batched tokens since text_inputs["input_ids"] is a list of lists, so np.array creates an object!
        if return_mm_token_type_ids:
            input_ids = text_inputs["input_ids"]
            image_token_id = self.image_token_id
            mm_token_type_ids = [[1 if y == image_token_id else 0 for y in x] for x in input_ids]
            # array_ids = np.array(text_inputs["input_ids"])
            # mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            # mm_token_type_ids[array_ids == self.image_token_id] = 1
            # text_inputs = {k: v.tolist() for k, v in text_inputs.items()}  # in case user requested list inputs
            text_inputs["token_type_ids"] = mm_token_type_ids#.tolist()
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)
    pass
    patch_function(transformers.models.gemma3.processing_gemma3.Gemma3Processor, "__call__", __call__)
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
        elif isinstance(past_key_values, HybridCache):
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
        # Internals in fp32
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        hidden_states_fp32 = x_fp32 * torch.rsqrt(variance + self.eps)

        # self.weight is bf16 (from vision.py loading if UNSLOTH_FORCE_FLOAT32="1")
        # So, cast self.weight to fp32 for the (1.0 + weight) operation
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
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "0": return
    try:
        import transformers.models.gemma3.modeling_gemma3
        transformers.models.gemma3.modeling_gemma3.Gemma3MLP
    except Exception as e:
        return raise_error("Gemma3MLP.forward", e)

    def forward(self, x): # x is fp16 from RMSNorm
        gate_proj_out = self.gate_proj(x)
        up_proj_out = self.up_proj(x)

        #  Upcast to fp32
        gate_proj_fp32 = gate_proj_out.to(torch.float32)
        up_proj_fp32 = up_proj_out.to(torch.float32)
        activated_fp32 = self.act_fn(gate_proj_fp32) # Activation in fp32
        intermediate_fp32 = activated_fp32 * up_proj_fp32 # Product in fp32

        # Downcast and down_proj
        intermediate_fp16 = intermediate_fp32.to(torch.float16)
        down_proj_out = self.down_proj(intermediate_fp16)
        return down_proj_out
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
        self,
        hidden_states,
        query_states_fp16,
        key_states_fp16,
        value_states_fp16,
        query_hidden_shape,
        kv_hidden_shape,
        position_embeddings,
        attention_mask,
    ):
        # 2. Upcast Q, K, V for norm and RoPE, and then transpose for attention
        # (bsz, num_specific_heads, q_len, head_dim)
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

        # 6. Core Attention mechanism (SDPA) in fp32
        attn_mask_for_sdpa = attention_mask
        if attn_mask_for_sdpa is not None and attn_mask_for_sdpa.dtype != torch.bool:
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
    prepare = torch.compile(prepare, fullgraph = True, dynamic = True, options = torch_compile_options)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: KWARGS_TYPE,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1] # For reshaping o_proj output later

        # Determine head shapes
        # Assuming these attributes are standard for Gemma3Attention
        # If not, they might come from self.config
        num_heads = getattr(self, "num_heads", self.config.num_attention_heads)
        num_key_value_heads = getattr(self, "num_key_value_heads", self.config.num_key_value_heads)
        head_dim = self.head_dim

        # For projections view: (bsz, q_len, num_specific_heads, head_dim)
        query_hidden_shape = (bsz, q_len, num_heads, head_dim)
        kv_hidden_shape    = (bsz, q_len, num_key_value_heads, head_dim)

        # 1. Projections (q, k, v) in fp16
        # hidden_states is already fp16. Weights of q_proj, k_proj, v_proj are fp16.
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
            self,
            hidden_states,
            query_states_fp16,
            key_states_fp16,
            value_states_fp16,
            query_hidden_shape,
            kv_hidden_shape,
            position_embeddings,
            attention_mask,
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
        is_causal = query_states_fp32.shape[2] > 1 and attn_mask_for_sdpa is None and getattr(self, "is_causal", True)
        # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
        # We convert it to a bool for the SDPA kernel that only accepts bools.
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

        # 7. Reshape and Downcast for Output Projection
        # attn_output_fp32 from SDPA is (bsz, num_heads, q_len, head_dim)
        attn_output_fp32 = attn_output_fp32.transpose(1, 2).contiguous()

        # Reshape to (bsz, q_len, num_query_heads * head_dim) which is (bsz, q_len, model_hidden_size)
        # Using -1 for the last dimension is robust and aligns with your original example.
        attn_output_fp32 = attn_output_fp32.reshape(bsz, q_len, -1) # REVISED FIX

        attn_output_fp16 = attn_output_fp32.to(torch.float16)

        # 8. Output Projection (o_proj) in fp16
        attn_output_projected = self.o_proj(attn_output_fp16) # fp16 output

        return attn_output_projected, attn_weights # 3-tuple return
    pass
    patch_function(transformers.models.gemma3.modeling_gemma3.Gemma3Attention, "forward", forward)
pass
TEMPORARY_PATCHES.append(patch_Gemma3Attention)


def patch_Gemma3Attention_normal():
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
        self,
        hidden_states,
        query_states_fp16,
        key_states_fp16,
        value_states_fp16,
        query_hidden_shape,
        kv_hidden_shape,
        position_embeddings,
        attention_mask,
    ):
        # 2. Upcast Q, K, V for norm and RoPE, and then transpose for attention
        # (bsz, num_specific_heads, q_len, head_dim)
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

        # 6. Core Attention mechanism (SDPA) in fp32
        attn_mask_for_sdpa = attention_mask
        if attn_mask_for_sdpa is not None and attn_mask_for_sdpa.dtype != torch.bool:
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
    prepare = torch.compile(prepare, fullgraph = True, dynamic = True, options = torch_compile_options)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: KWARGS_TYPE,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1] # For reshaping o_proj output later

        # Determine head shapes
        # Assuming these attributes are standard for Gemma3Attention
        # If not, they might come from self.config
        num_heads = getattr(self, "num_heads", self.config.num_attention_heads)
        num_key_value_heads = getattr(self, "num_key_value_heads", self.config.num_key_value_heads)
        head_dim = self.head_dim

        # For projections view: (bsz, q_len, num_specific_heads, head_dim)
        query_hidden_shape = (bsz, q_len, num_heads, head_dim)
        kv_hidden_shape    = (bsz, q_len, num_key_value_heads, head_dim)

        # 1. Projections (q, k, v) in fp16
        # hidden_states is already fp16. Weights of q_proj, k_proj, v_proj are fp16.
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
            self,
            hidden_states,
            query_states_fp16,
            key_states_fp16,
            value_states_fp16,
            query_hidden_shape,
            kv_hidden_shape,
            position_embeddings,
            attention_mask,
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
        is_causal = query_states_fp32.shape[2] > 1 and attn_mask_for_sdpa is None and getattr(self, "is_causal", True)
        # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
        # We convert it to a bool for the SDPA kernel that only accepts bools.
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

        # 7. Reshape and Downcast for Output Projection
        # attn_output_fp32 from SDPA is (bsz, num_heads, q_len, head_dim)
        attn_output_fp32 = attn_output_fp32.transpose(1, 2).contiguous()

        # Reshape to (bsz, q_len, num_query_heads * head_dim) which is (bsz, q_len, model_hidden_size)
        # Using -1 for the last dimension is robust and aligns with your original example.
        attn_output_fp32 = attn_output_fp32.reshape(bsz, q_len, -1) # REVISED FIX

        attn_output_fp16 = attn_output_fp32.to(torch.float16)

        # 8. Output Projection (o_proj) in fp16
        attn_output_projected = self.o_proj(attn_output_fp16) # fp16 output

        return attn_output_projected, attn_weights # 3-tuple return
    pass
    patch_function(transformers.models.gemma3.modeling_gemma3.Gemma3Attention, "forward", forward)
pass
TEMPORARY_PATCHES.append(patch_Gemma3Attention)
