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
from .common import TEMPORARY_PATCHES, torch_compile
from .utils import (
    patch_function,
    KWARGS_TYPE,
    raise_error,
)

def patch_Gemma3nConvNormAct_forward():
    try:
        import timm.layers.conv_bn_act
        timm.layers.conv_bn_act.ConvNormAct
    except Exception as e:
        return raise_error("timm.layers.conv_bn_act.ConvNormAct", e)

    # Counteract high weights in Conv layers for Gemma 3N by forcing to float32
    def forward(self, x):
        old_dtype = x.dtype
        x = x.to(torch.float32)
        with torch.autocast(device_type = "cuda", dtype = torch.float32, enabled = True):
            x = self.conv(x)
        x = self.bn(x)
        aa = getattr(self, 'aa', None)
        if aa is not None:
            x = self.aa(x)
        return x.to(old_dtype)
    pass
    patch_function(timm.layers.conv_bn_act.ConvNormAct, "forward", forward, fullgraph = True)
pass
# We only execute this for float16 so it's not always executed
# TEMPORARY_PATCHES.append(patch_Gemma3nConvNormAct_forward)

@torch_compile
def Gemma3nRMSNorm_forward(self, x: torch.Tensor) -> torch.Tensor:
    # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
    # See https://github.com/huggingface/transformers/pull/29402
    output = self._norm(x.float()) * self.weight.float()
    return output.type_as(x)

def patch_Gemma3nMultimodalEmbedder_forward():
    try:
        import transformers.models.gemma3n.modeling_gemma3n
        from transformers.models.gemma3n.modeling_gemma3n import Gemma3nMultimodalEmbedder
    except Exception as e:
        return raise_error("timm.layers.multimodal_embedder.MultimodalEmbedder", e)

    def Gemma3nMultimodalEmbedder_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is not None:
            emb_norm = Gemma3nRMSNorm_forward(self.soft_embedding_norm, inputs_embeds)
        else:
            hard_emb = self.embedding(input_ids - self.vocab_offset)
            emb_norm = Gemma3nRMSNorm_forward(self.hard_embedding_norm, hard_emb)

        old_dtype = emb_norm.dtype
        emb_norm = emb_norm.to(torch.float32)
        with torch.autocast(device_type = "cuda", dtype = torch.float32, enabled = True):
            emb_norm_proj = self.embedding_projection(emb_norm)
        emb_norm_proj = emb_norm_proj.to(old_dtype)

        return Gemma3nRMSNorm_forward(self.embedding_post_projection_norm, emb_norm_proj)
    pass

    patch_function(transformers.models.gemma3n.modeling_gemma3n.Gemma3nMultimodalEmbedder, "forward", Gemma3nMultimodalEmbedder_forward, fullgraph = True)
pass

def patch_Gemma3nTextAltUp_predict():
    try:
        import transformers.models.gemma3n.modeling_gemma3n
        from transformers.models.gemma3n.modeling_gemma3n import Gemma3nTextAltUp
    except Exception as e:
        return raise_error("Gemma3nTextAltUp.predict", e)
    
    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        new_dtype = self.router_input_scale.dtype
        hidden_states = hidden_states.to(new_dtype)
        x = hidden_states[self.config.altup_active_idx]
        router_inputs = Gemma3nRMSNorm_forward(self.router_norm, x) * self.router_input_scale
        routed = self.modality_router(router_inputs)
        modalities = torch.tanh(routed.float()).type_as(x)

        if self.training and self.config.altup_coef_clip is not None:
            self.prediction_coefs.weight.data.clamp_(-self.config.altup_coef_clip, self.config.altup_coef_clip)

        # Project and then transpose all 2D matrices contained so that mulmat gives the correct result
        all_coefs: torch.Tensor = (
            self.prediction_coefs(modalities)
            .reshape(*modalities.shape[:-1], self.config.altup_num_inputs, self.config.altup_num_inputs)
            .permute(0, 1, 3, 2)
        )

        # permute hidden_states to [batch_size, num_tokens, hidden_size, altup_num_inputs]
        predictions = torch.matmul(hidden_states.permute(1, 2, 3, 0), all_coefs)
        predictions = predictions.permute(3, 0, 1, 2)  # undo the permute
        predictions += hidden_states  # add the original input
        return predictions.contiguous().type_as(hidden_states)
    pass
    patch_function(transformers.models.gemma3n.modeling_gemma3n.Gemma3nTextAltUp, "predict", predict, fullgraph = True)
pass

def patch_Gemma3nConv_Embed_forwards():
    # helper function to patch both forwards
    try:
        patch_Gemma3nConvNormAct_forward()
        patch_Gemma3nMultimodalEmbedder_forward()
        patch_Gemma3nTextAltUp_predict()
    except Exception as e:
        return raise_error("patch_Gemma3nConv_Embed_forwards", e)
pass


def patch_Gemma3nTextAltUp_functions():
    try:
        import transformers.models.gemma3n.modeling_gemma3n
        from transformers.models.gemma3n.modeling_gemma3n import Gemma3nTextAltUp
    except Exception as e:
        return raise_error("Gemma3nTextAltUp", e)

    if hasattr(Gemma3nTextAltUp, "predict"):
        patch_function(transformers.models.gemma3n.modeling_gemma3n.Gemma3nTextAltUp, "predict", Gemma3nTextAltUp.predict, fullgraph = True)
    if hasattr(Gemma3nTextAltUp, "correct"):
        patch_function(transformers.models.gemma3n.modeling_gemma3n.Gemma3nTextAltUp, "correct", Gemma3nTextAltUp.correct, fullgraph = True)
    if hasattr(Gemma3nTextAltUp, "scale_corrected_output"):
        patch_function(transformers.models.gemma3n.modeling_gemma3n.Gemma3nTextAltUp, "scale_corrected_output", Gemma3nTextAltUp.scale_corrected_output, fullgraph = True)
pass
TEMPORARY_PATCHES.append(patch_Gemma3nTextAltUp_functions)

def patch_Gemma3nModel_get_placeholder_mask():
    try:
        import transformers.models.gemma3n.modeling_gemma3n
        from transformers.models.gemma3n.modeling_gemma3n import Gemma3nModel
    except Exception as e:
        return raise_error("Gemma3nModel.get_place_holder_mask", e)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholdr mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_audio_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            ).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_audio_mask = input_ids == self.config.audio_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0] * image_features.shape[1]}"
            )

        n_audio_tokens = special_audio_mask.sum()
        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if audio_features is not None and inputs_embeds[special_audio_mask].numel() != audio_features.numel():
            raise ValueError(
                f"Audio features and image tokens do not match: tokens: {n_audio_tokens}, features {audio_features.shape[0] * audio_features.shape[1]}"
            )

        return special_image_mask, special_audio_mask
    pass
    patch_function(transformers.models.gemma3n.modeling_gemma3n.Gemma3nModel, "get_placeholder_mask", get_placeholder_mask, match_level="relaxed")
pass
TEMPORARY_PATCHES.append(patch_Gemma3nModel_get_placeholder_mask)
