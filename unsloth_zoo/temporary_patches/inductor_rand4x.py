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

"""Backport of pytorch/pytorch#198680 (commit 983d11f9) to torch 2.13 / 2.14.

pytorch#184377 (torch 2.13.0 through 2.14.x) emits `triton_helpers.rand4x` / `randn4x` for
every 1-D CUDA (and ROCm) Triton kernel. That helper rebuilds the Philox offsets as
`min(offsets) // 4 + arange(BLOCK // 4)`, which is only right for contiguous offsets, and it
returns different numbers than `tl.rand` for the same seed and offset. Dropout fed by a
materialised permute then gets duplicated masks, and a recompute tiled differently from its
forward (activation checkpointing) draws a different mask (pytorch#198333). torch 2.15 turns
the path off unless TORCHINDUCTOR_USE_RAND4X=1; this does the same.

Gated on the function body (the 1-D check without `use_rand4x`) and on
`torch._inductor.config` lacking `use_rand4x`, so torch 2.12 and older (no 4x path) and
2.15+ (fixed) are untouched. A monkeypatch is not part of Inductor's FX graph or AOTAutograd
cache key, so a tag is added to `torch.compiler.config.cache_key_tag`; without it a warm
cache keeps serving kernels generated with rand4x.
"""

import inspect
import os

__all__ = ["patch_inductor_rand4x"]

CACHE_KEY_TAG = "unsloth/inductor_rand4x_off:pytorch198680"


def _needs_patch(fn, config):
    if hasattr(config, "use_rand4x"):
        return False
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return "triton_tensor_ndim() == 1" in source and "use_rand4x" not in source


def _add_cache_key_tag(tag):
    import torch

    config = getattr(torch.compiler, "config", None)
    if config is None or not hasattr(config, "cache_key_tag"):
        return False
    tags = [t for t in config.cache_key_tag.split(",") if t]
    if tag not in tags:
        config.cache_key_tag = ",".join(tags + [tag])
    return True


def patch_inductor_rand4x():
    """True when the 4x path is now off (patched here or earlier), else False."""
    try:
        from torch._inductor import config
        from torch._inductor.codegen.triton import TritonOverrides
    except Exception:
        return False

    original = TritonOverrides.__dict__.get("_can_use_4x_random")
    if original is None:
        return False
    original = getattr(original, "__func__", original)
    if getattr(original, "_unsloth_rand4x_off", False):
        return True
    if not _needs_patch(original, config):
        return False
    # Same opt-in as torch 2.15's `config.use_rand4x`.
    if os.environ.get("TORCHINDUCTOR_USE_RAND4X") == "1":
        return False

    def _can_use_4x_random(*args, **kwargs):
        return False

    _can_use_4x_random._unsloth_rand4x_off = True
    _can_use_4x_random._unsloth_original = original
    _can_use_4x_random.__doc__ = original.__doc__
    TritonOverrides._can_use_4x_random = staticmethod(_can_use_4x_random)
    _add_cache_key_tag(CACHE_KEY_TAG)
    return True


try:
    patch_inductor_rand4x()
except Exception:
    pass
