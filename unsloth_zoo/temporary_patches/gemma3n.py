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

import re
from typing import Union, List, Optional, Tuple
import inspect
import torch
import torch.nn as nn
import os
import logging

from .common import TEMPORARY_PATCHES, torch_compile_options, UNSLOTH_ENABLE_LOGGING

logger = logging.getLogger(__name__)


def patch_Gemma3nConvNormAct_forward():
    try:
        import timm.layers.conv_bn_act
        timm.layers.conv_bn_act.ConvNormAct
    except:
        return
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
    old_keys = inspect.signature(timm.layers.conv_bn_act.ConvNormAct.forward).parameters
    new_keys = inspect.signature(forward).parameters
    if old_keys != new_keys:
        if UNSLOTH_ENABLE_LOGGING:
            print("Unsloth: Failed to patch patch_Gemma3nConvNormAct_forward.")
    else:
        forward = torch.compile(forward, fullgraph = False, dynamic = True, options = torch_compile_options)
        timm.layers.conv_bn_act.ConvNormAct.forward = forward
    return
pass
# TEMPORARY_PATCHES.append(patch_Gemma3nConvNormAct_forward)


def patch_Gemma3nTextAltUp_functions():
    try:
        import transformers.models.gemma3n.modeling_gemma3n
        transformers.models.gemma3n.modeling_gemma3n.Gemma3nTextAltUp
    except:
        return

    if hasattr(transformers.models.gemma3n.modeling_gemma3n.Gemma3nTextAltUp, "predict"):
        predict = transformers.models.gemma3n.modeling_gemma3n.Gemma3nTextAltUp.predict
        if not hasattr(predict, "get_compiler_config"):
            transformers.models.gemma3n.modeling_gemma3n.Gemma3nTextAltUp.predict = \
                torch.compile(predict, fullgraph = False, dynamic = True, options = torch_compile_options)
            if UNSLOTH_ENABLE_LOGGING:
                print("Unsloth: Patched Gemma3nTextAltUp.predict")
    pass
    if hasattr(transformers.models.gemma3n.modeling_gemma3n.Gemma3nTextAltUp, "correct"):
        correct = transformers.models.gemma3n.modeling_gemma3n.Gemma3nTextAltUp.correct
        if not hasattr(correct, "get_compiler_config"):
            transformers.models.gemma3n.modeling_gemma3n.Gemma3nTextAltUp.correct = \
                torch.compile(correct, fullgraph = False, dynamic = True, options = torch_compile_options)
            if UNSLOTH_ENABLE_LOGGING:
                print("Unsloth: Patched Gemma3nTextAltUp.correct")
    pass
    if hasattr(transformers.models.gemma3n.modeling_gemma3n.Gemma3nTextAltUp, "scale_corrected_output"):
        scale_corrected_output = transformers.models.gemma3n.modeling_gemma3n.Gemma3nTextAltUp.scale_corrected_output
        if not hasattr(scale_corrected_output, "get_compiler_config"):
            transformers.models.gemma3n.modeling_gemma3n.Gemma3nTextAltUp.scale_corrected_output = \
                torch.compile(scale_corrected_output, fullgraph = False, dynamic = True, options = torch_compile_options)
            if UNSLOTH_ENABLE_LOGGING:
                print("Unsloth: Patched Gemma3nTextAltUp.scale_corrected_output")
    pass
    return
pass
TEMPORARY_PATCHES.append(patch_Gemma3nTextAltUp_functions)

def patch_Gemma3nModel_get_image_features():
    try:
        import transformers.models.gemma3n.modeling_gemma3n
        transformers.models.gemma3n.modeling_gemma3n.Gemma3nModel.get_image_features
    except:
        return
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Projects the last hidden state from the vision model into language model space.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.

        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values, do_pooling=False, return_dict=True
        ).last_hidden_state
        # Convert from (batch, channels, height, width) to (batch, height * width, channels) where:
        # height == width and height * width == Gemma3nConfig.vision_soft_tokens_per_image.
        vision_outputs = vision_outputs.reshape(
            vision_outputs.shape[0],
            self.config.vision_config.hidden_size,
            self.config.vision_soft_tokens_per_image,
        ).permute(0, 2, 1)
        # Normalize and embed the soft tokens into language model space.
        # vision_outputs *= self.config.vision_config.hidden_size**0.5
        return self.embed_vision(inputs_embeds=vision_outputs)
    pass
    old_keys = inspect.signature(transformers.models.gemma3n.modeling_gemma3n.Gemma3nModel.get_image_features).parameters
    new_keys = inspect.signature(get_image_features).parameters
    if old_keys != new_keys:
        if UNSLOTH_ENABLE_LOGGING:
            print("Unsloth: Failed to patch Gemma3nModel.get_image_features.")
    else:
        transformers.models.gemma3n.modeling_gemma3n.Gemma3nModel.get_image_features = get_image_features
    return
pass
TEMPORARY_PATCHES.append(patch_Gemma3nModel_get_image_features)
