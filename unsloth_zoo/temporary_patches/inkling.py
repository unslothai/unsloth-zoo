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
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Inkling (thinkingmachines/Inkling-Small) checkpoint config compatibility.

The published config.json carries `intermediate_size` (the MoE expert width,
2048 on Inkling-Small) and `dense_intermediate_size` (the width of the leading
dense layers, 16384), and no `moe_intermediate_size`. transformers'
`InklingTextConfig` overwrites `intermediate_size` with the dense width and
leaves `moe_intermediate_size` at its class default of 3072, so every routed
and shared expert is built at the wrong width and loading stops with
"You set `ignore_mismatched_sizes` to `False`" (gate_up_proj ckpt
(256, 4096, 4096) vs model (256, 6144, 4096)). Plain transformers fails the
same way.

When a config supplies `dense_intermediate_size` and `intermediate_size` but
not `moe_intermediate_size`, its `intermediate_size` is the MoE width: that is
what a separate dense width means. A config that spells out
`moe_intermediate_size` is left alone.
"""
import functools

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import logger

__all__ = ["patch_inkling_text_config"]


def patch_inkling_text_config():
    try:
        from transformers.models.inkling.configuration_inkling import InklingTextConfig
    except Exception:
        return
    original = InklingTextConfig.__init__
    if getattr(original, "_unsloth_patched", False):
        return

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        if (
            "moe_intermediate_size" not in kwargs
            and kwargs.get("dense_intermediate_size") is not None
            and kwargs.get("intermediate_size") is not None
        ):
            kwargs["moe_intermediate_size"] = kwargs["intermediate_size"]
        return original(self, *args, **kwargs)

    __init__._unsloth_patched = True
    InklingTextConfig.__init__ = __init__
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched InklingTextConfig to read the MoE width from a dense/MoE split config.")
pass

TEMPORARY_PATCHES.append(patch_inkling_text_config)
