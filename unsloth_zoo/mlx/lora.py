# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the GNU Affero General Public License, version 3 or later.

import copy
import math

import mlx.core as mx
import mlx.nn as nn


class LoRAPointwiseConv2d(nn.Module):
    """LoRA for plain, ungrouped 1x1 convolutions in channel-last layout."""

    @staticmethod
    def supports(module):
        return (type(module) is nn.Conv2d and module.groups == 1
                and module.weight.shape[1:3] == (1, 1))

    @staticmethod
    def from_base(base, r=8, scale=1.0, dropout=0.0):
        if not LoRAPointwiseConv2d.supports(base):
            raise ValueError("LoRA requires a plain, ungrouped 1x1 Conv2d.")
        module = LoRAPointwiseConv2d()
        module.conv = base
        module.scale = scale
        module.dropout = nn.Dropout(dropout)
        width = base.weight.shape[-1]
        bound = 1 / math.sqrt(width)
        module.lora_a = mx.random.uniform(low=-bound, high=bound, shape=(width, r))
        module.lora_b = mx.zeros((r, base.weight.shape[0]))
        return module

    def __call__(self, x):
        y = self.conv(x)
        weight = (self.lora_a @ self.lora_b).T[:, None, None, :].astype(x.dtype)
        delta = mx.conv2d(
            self.dropout(x), weight, self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups,
        )
        return y + (self.scale * delta).astype(y.dtype)

    def fuse(self):
        conv = copy.deepcopy(self.conv)
        delta = (self.lora_a @ self.lora_b).T[:, None, None, :]
        conv.weight = conv.weight + (self.scale * delta).astype(conv.weight.dtype)
        return conv
