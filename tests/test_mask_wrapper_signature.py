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


"""The patched mask builders keep the originals' signatures, so callers filtering by them still pass arguments."""
import inspect

import transformers.masking_utils as masking_utils

from unsloth_zoo.temporary_patches.misc import patch_transformers_masks


def test_patched_mask_builders_expose_original_signature():
    patch_transformers_masks()
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        original = getattr(masking_utils, f"_unsloth_original_{name}")
        patched = getattr(masking_utils, name)
        assert patched is not original
        assert inspect.signature(patched) == inspect.signature(original)
        accepted = set(inspect.signature(patched).parameters)
        assert "config" in accepted and "attention_mask" in accepted
