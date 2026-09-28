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

"""Inkling fixes: Inkling-Small MoE width; embed_norm applied twice on transformers 5.17 (transformers#47827, fixed by #48786)."""
import functools
import inspect

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import logger

__all__ = ["patch_inkling_text_config", "patch_inkling_double_embed_norm"]


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


def _identity(hidden_states):
    return hidden_states


def _transformers_has_double_embed_norm_release(version = None):
    try:
        from packaging.version import Version
        if version is None:
            import transformers
            version = transformers.__version__
        return Version(version).release[:2] == (5, 17)
    except Exception:
        return False


def patch_inkling_double_embed_norm():
    try:
        from transformers.models.inkling import modeling_inkling
    except Exception:
        return
    if not _transformers_has_double_embed_norm_release():
        return
    mm_model = getattr(modeling_inkling, "InklingModel", None)
    text_model = getattr(modeling_inkling, "InklingTextModel", None)
    if mm_model is None or text_model is None:
        return
    if getattr(mm_model.forward, "_unsloth_patched", False):
        return
    try:
        mm_source = inspect.getsource(mm_model.forward)
        text_source = inspect.getsource(text_model.forward)
    except Exception:
        return
    # #48786 moves the norm into embed_tokens, so a backport fails these checks and stays unpatched
    if "self.language_model.embed_norm(inputs_embeds)" not in mm_source:
        return
    if "self.embed_norm(inputs_embeds)" not in text_source:
        return

    original_mm_forward = mm_model.forward
    original_text_forward = text_model.forward

    @functools.wraps(original_text_forward)
    def text_forward(self, *args, **kwargs):
        if not getattr(self, "_unsloth_inputs_embeds_normed", False):
            return original_text_forward(self, *args, **kwargs)
        self._unsloth_inputs_embeds_normed = False
        norm = self.embed_norm
        # Restore any instance-level forward (accelerate hooks)
        had_forward = "forward" in norm.__dict__
        previous_forward = norm.__dict__.get("forward")
        norm.forward = _identity
        try:
            return original_text_forward(self, *args, **kwargs)
        finally:
            if had_forward:
                norm.forward = previous_forward
            else:
                del norm.__dict__["forward"]

    @functools.wraps(original_mm_forward)
    def mm_forward(self, *args, **kwargs):
        language_model = self.language_model
        language_model._unsloth_inputs_embeds_normed = True
        try:
            return original_mm_forward(self, *args, **kwargs)
        finally:
            language_model._unsloth_inputs_embeds_normed = False

    text_forward._unsloth_patched = True
    mm_forward._unsloth_patched = True
    text_model.forward = text_forward
    mm_model.forward = mm_forward
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched Inkling so embed_norm is applied once on the multimodal path.")
pass

TEMPORARY_PATCHES.append(patch_inkling_double_embed_norm)
