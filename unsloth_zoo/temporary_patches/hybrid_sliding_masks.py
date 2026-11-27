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

"""hybrid_sliding StaticCache sliding mask; self-contained (only .common) for CPU backend."""

import logging
import os

from .common import TEMPORARY_PATCHES

__all__ = [
    "patch_hybrid_sliding_generate_masks",
]

logger = logging.getLogger(__name__)
_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"


def _unwrap_chain(fn):
    """fn, then every __wrapped__ / torch.compile original below it."""
    seen = []
    while fn is not None and not any(fn is s for s in seen) and len(seen) < 16:
        seen.append(fn)
        fn = getattr(fn, "__wrapped__", None) or getattr(fn, "_torchdynamo_orig_callable", None)
    return seen


def _is_builder(fn, name, known):
    """True when fn is, or wraps, the masking_utils builder `name` (by identity with a known copy, else by origin)."""
    for f in _unwrap_chain(fn):
        if any(f is k for k in known):
            return True
        if getattr(f, "__name__", None) == name and getattr(f, "__module__", None) == "transformers.masking_utils":
            return True
    return False


def patch_hybrid_sliding_generate_masks():
    """generate prebuilds the attention masks for a compileable cache (StaticCache) from the layer pattern table, which
    in transformers 5.24 gives the sliding side of "hybrid_sliding" layers (Inkling, Inkling 2, Inkling 2 Loop) a full
    causal mask sized to the whole static buffer. Those layers keep window bounded KV states, so once the static
    buffer is longer than the window the mask and the states disagree (a size mismatch, or a wrong mask). Point the
    entry at the same sliding window builder the plain "sliding_attention" entry uses. Unsloth wraps the public mask
    builders (misc.patch_transformers_masks, gpt_oss) while the table keeps the originals, so the entry is matched by
    unwrapping, not by identity with the public attribute. A no-op where transformers already maps it otherwise."""
    try:
        import transformers.masking_utils as masking_utils
    except Exception:
        return
    mapping = getattr(masking_utils, "LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING", None)
    entry = mapping.get("hybrid_sliding") if isinstance(mapping, dict) else None
    if not isinstance(entry, dict) or not callable(entry.get("sliding_attention")):
        return
    known_causal = [
        getattr(masking_utils, name, None)
        for name in ("create_causal_mask", "_unsloth_original_create_causal_mask", "_old_create_causal_mask")
    ] + [mapping.get("full_attention")]
    known_causal = [f for f in known_causal if callable(f)]
    current = entry["sliding_attention"]
    if not _is_builder(current, "create_causal_mask", known_causal):
        return
    # The unwrapped original first: the plain sliding layers use it, and a wrapper may be compiled.
    candidates = (
        mapping.get("sliding_attention"),
        getattr(masking_utils, "_unsloth_original_create_sliding_window_causal_mask", None),
        getattr(masking_utils, "create_sliding_window_causal_mask", None),
    )
    for sliding in candidates:
        if callable(sliding) and _is_builder(sliding, "create_sliding_window_causal_mask", ()):
            entry["sliding_attention"] = sliding
            if _ENABLE_LOGGING:
                logger.info("Unsloth: hybrid_sliding layers get a sliding window mask in generate.")
            return


TEMPORARY_PATCHES.append(patch_hybrid_sliding_generate_masks)
