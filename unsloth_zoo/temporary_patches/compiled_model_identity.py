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

"""Keep compiled-cache model classes counted as transformers' own code (is_custom_code)."""
from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import logger

__all__ = ["patch_compiled_model_is_custom_code", "UNSLOTH_COMPILED_MODULE_PREFIX"]

UNSLOTH_COMPILED_MODULE_PREFIX = "unsloth_compiled_module_"


def patch_compiled_model_is_custom_code():
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return
    original = PreTrainedModel.__dict__.get("is_custom_code", None)
    if original is None or getattr(original, "_unsloth_patched", False):
        return
    original_function = original.__func__ if isinstance(original, classmethod) else original

    def is_custom_code(cls) -> bool:
        if cls.is_remote_code():
            return True
        module = getattr(cls, "__module__", "") or ""
        if module.startswith("transformers."):
            return False
        if module.startswith(UNSLOTH_COMPILED_MODULE_PREFIX):
            return False
        return bool(original_function(cls))

    method = classmethod(is_custom_code)
    method._unsloth_patched = True
    PreTrainedModel.is_custom_code = method
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched PreTrainedModel.is_custom_code so compiled classes keep their checkpoint conversions.")
pass

TEMPORARY_PATCHES.append(patch_compiled_model_is_custom_code)
