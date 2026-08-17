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
#
# ============================================================================
# Fast sliding-window attention router for Muse Glimmer.
#
# Muse Glimmer interleaves 39 sliding_attention layers (window 2048) with 13
# full_attention layers. It runs `sdpa`, and every sliding layer is handed a
# dense [B, 1, S, S] boolean mask that is exactly the causal + sliding band, so
# each of those layers pays O(S^2) in compute and mask traffic even though only
# a 2048 key window is ever read.
#
# This routes just those layers through the block-local O(S*w) kernel that
# already ships for gemma-4 (`_banded_sdpa_core`), reusing its band probe
# (`_mask_is_plain_band`) so the engage / defer decision is provably the same
# one gemma-4 makes. The full-attention layers, the vision tower, decode steps
# (Sq != Sk), padded or packed batches (the probe rejects them) and every
# non-Muse-Glimmer module defer untouched to the original SDPA.
#
# Why a separate router rather than reusing gemma4_flash_sliding: Muse Glimmer
# exposes the sliding flag as `is_local_attention`, and gemma-4's router gates on
# `is_sliding`, which Muse Glimmer never defines. That router is therefore a
# strict no-op here. Only the routing differs; the kernel and the band probe are
# shared.
#
# Scope, stated plainly: this only helps long context. Blocking into (w, 2w)
# tiles is a loss below 2x the window and roughly break-even at 2x, so the gate
# below turns it on only from 3x the window up (S >= 6144 at a 2048 window).
# A 1024, 2048 or 4096 token finetune is deliberately untouched.
#
# On by default. UNSLOTH_MUSE_GLIMMER_BANDED_SDPA=0 reverts to plain SDPA.
# FlashAttention-2's window kernel is preferred when flash-attn happens to be
# importable, but it is not a dependency: the pure-SDPA banded path is what runs
# on a stock install.
# ============================================================================

import os
import functools
import torch
from .common import TEMPORARY_PATCHES, logger
from .utils import raise_error
from .gemma4_banded_attention import _banded_sdpa_core, _mask_is_plain_band

__all__ = ["patch_muse_glimmer_banded_sliding_attention"]

try:
    from flash_attn import flash_attn_func as _flash_attn_func
    _HAS_FA2 = True
except Exception:
    _flash_attn_func = None
    _HAS_FA2 = False

_ORIG_SDPA = [None]      # boxed reference to the wrapped sdpa function
_ENGAGED = [0]           # count of FA2-path invocations (debug)
_BANDED_ENGAGED = [0]    # count of banded-path invocations (debug)
_DEFERRED = [0]          # count of sliding calls that fell through (debug)

# Engage only from this multiple of the window upward. Measured on B200 at Muse
# Glimmer's shapes (32 / 2 heads, head_dim 128, w = 2048, bf16), the banded
# kernel against the real sdpa_attention_forward, forward plus backward:
#     S = 1w  0.60x    S = 1.5w 0.43x    S = 2w  0.72x    S = 3w  1.12x
#     S = 4w  1.43x    S = 6w   2.10x    S = 8w  2.85x    S = 16w 5.74x
# Below 2w the band already covers nearly the whole causal triangle, so blocking
# into (w, 2w) tiles costs about 2x the FLOPs of the plain causal kernel and the
# patch would be a slowdown. 3w is the first length with a clear win, so that is
# where it turns on, and nothing below it is touched.
_MIN_SEQ_MULTIPLE_OF_WINDOW = 3


def muse_glimmer_banded_stats():
    """Engagement counters, for tests and benchmarks."""
    return {
        "fa2"      : _ENGAGED[0],
        "banded"   : _BANDED_ENGAGED[0],
        "deferred" : _DEFERRED[0],
        "has_fa2"  : _HAS_FA2,
    }


@functools.lru_cache(maxsize=1)
def _enabled():
    """Whether the Muse Glimmer sliding-window fast router is active.

    On by default. The pure-SDPA banded kernel works on every dtype and GPU, so
    this must NOT be gated on flash-attn being importable: that would wrongly
    disable the fallback that most installs actually run. Set
    UNSLOTH_MUSE_GLIMMER_BANDED_SDPA=0 to revert to plain SDPA.

    Cached with maxsize=1 since the env var is read once per process. Any code or
    test that toggles UNSLOTH_MUSE_GLIMMER_BANDED_SDPA at runtime must call
    _enabled.cache_clear() afterwards for the change to take effect.
    """
    return os.environ.get("UNSLOTH_MUSE_GLIMMER_BANDED_SDPA", "1") != "0"


@functools.lru_cache(maxsize=1)
def _force_banded():
    """Optional override: force the pure-SDPA banded kernel even when flash-attn
    is importable. The router already prefers FA2 automatically, so this is only
    for benchmarking the two kernels against each other; off by default.

    Cached with maxsize=1 since the env var is read once per process. Any code or
    test that toggles UNSLOTH_MUSE_GLIMMER_BANDED_FORCE at runtime must call
    _force_banded.cache_clear() afterwards for the change to take effect.
    """
    return os.environ.get("UNSLOTH_MUSE_GLIMMER_BANDED_FORCE", "0") == "1"


def _is_muse_glimmer_sliding_module(module):
    """Strict gate: a Muse Glimmer text attention layer that is actually sliding.

    Name-prefixed rather than isinstance-checked because Unsloth's compiler
    regenerates MuseGlimmerTextAttention into unsloth_compiled_cache, so the
    class the model really runs is not the one importable from transformers.
    The vision tower is a separate MuseGlimmerVisionAttention, so the
    MuseGlimmerText prefix excludes it by construction.
    """
    if not getattr(module, "is_local_attention", False): return False
    cls = type(module).__name__
    return cls.startswith("MuseGlimmerText") and cls.endswith("Attention")


def _sdpa_maybe_muse_glimmer_banded(module, query, key, value, attention_mask,
                                    dropout=0.0, scaling=None, is_causal=None, **kwargs):
    # Shared layer + band gate for both fast kernels (no _HAS_FA2 / dtype clause
    # here, so the banded fallback is reachable without flash-attn).
    if (_enabled()
            and _is_muse_glimmer_sliding_module(module)
            and query.dim() == 4):
        w = kwargs.get("sliding_window", None) or getattr(module, "sliding_window", None)
        Sq = query.shape[2]
        Sk = key.shape[2]
        # Mirror SDPA's derivation. With no explicit mask, only a causal module
        # may take the causal window; a bidirectional call (is_causal False) must
        # stay bidirectional instead of being forced causal.
        causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
        # Dropout is deliberately excluded rather than forwarded: the block-local
        # kernel draws its mask over (w, 2w) tiles, so its RNG stream cannot match
        # full SDPA's. Muse Glimmer's attention_dropout is 0.0, so this only ever
        # defers a configuration that was never comparable anyway.
        active_dropout = dropout if getattr(module, "training", False) else 0.0
        if (w and Sq == Sk and Sq >= _MIN_SEQ_MULTIPLE_OF_WINDOW * w
                and active_dropout == 0.0
                and (attention_mask is not None or causal)
                and _mask_is_plain_band(attention_mask, Sq, w)):
            # Prefer FlashAttention-2's window kernel when importable and the dtype
            # and head_dim are supported; otherwise fall to the pure-SDPA banded
            # kernel so the O(S*w) speedup is automatic with or without flash-attn.
            # A runtime FA2 failure (unsupported GPU / build, CPU tensors) falls to
            # the banded kernel too, so it is not skipped, and only then to the
            # original SDPA.
            use_fa2 = (_HAS_FA2 and not _force_banded()
                       and query.shape[-1] <= 256
                       and query.dtype in (torch.float16, torch.bfloat16))
            if use_fa2:
                try:
                    # flash_attn_func wants (B, S, H, D); a causal window of w keys
                    # is window_size=(w-1, 0). FA2 handles GQA (H q, Hkv kv heads).
                    out = _flash_attn_func(
                        query.transpose(1, 2),
                        key.transpose(1, 2),
                        value.transpose(1, 2),
                        dropout_p=0.0,
                        softmax_scale=scaling,
                        causal=True,
                        window_size=(w - 1, 0),
                    )
                    _ENGAGED[0] += 1
                    if _ENGAGED[0] == 1:
                        logger.info_once(
                            f"Unsloth: FlashAttention-2 sliding window engaged for "
                            f"Muse Glimmer (window={w})."
                        )
                    return out, None                      # (B, S, H, D), no weights
                except Exception as e:
                    logger.warning_once(
                        f"Unsloth: Muse Glimmer FA2 sliding fell back to banded SDPA ({e})")
            # Reached when FA2 is not used or its attempt failed: try the pure-SDPA
            # block-local kernel before deferring to the original SDPA.
            try:
                # ng folds the kv heads up to the q heads exactly as SDPA's GQA
                # expansion does (Muse Glimmer is 32 / 2, so ng = 16).
                ng = getattr(module, "num_key_value_groups", query.shape[1] // key.shape[1])
                out = _banded_sdpa_core(query, key, value, w, scaling, 0.0, ng)
                _BANDED_ENGAGED[0] += 1
                if _BANDED_ENGAGED[0] == 1:
                    logger.info_once(
                        f"Unsloth: banded sliding SDPA engaged for Muse Glimmer "
                        f"(window={w})."
                    )
                return out, None                      # (B, S, H, D), no weights
            except Exception as e:
                logger.warning_once(
                    f"Unsloth: Muse Glimmer banded sliding fell back to SDPA ({e})")
        _DEFERRED[0] += 1
    return _ORIG_SDPA[0](module, query, key, value, attention_mask,
                         dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)


def patch_muse_glimmer_banded_sliding_attention():
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception as e:
        raise_error("transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS", e)
        return
    try:
        current = ALL_ATTENTION_FUNCTIONS["sdpa"]
    except Exception as e:
        raise_error("ALL_ATTENTION_FUNCTIONS['sdpa']", e)
        return
    if getattr(current, "_unsloth_muse_glimmer_banded", False):
        return
    # Wraps whatever is currently installed, so chaining behind gemma-4's router
    # is safe: a non-Muse-Glimmer module falls straight through to it.
    _ORIG_SDPA[0] = current
    # unsloth runs TEMPORARY_PATCHES three times (init, pre_compile, post_compile),
    # and every sdpa router here guards re-entry by reading one sentinel attribute
    # off the *installed* entry only. Once this wrapper sits on top, the wrapped
    # router would no longer see its own sentinel and would wrap a second time on
    # the next pass, so forward the sentinels of the function we wrap. Idempotency
    # is then preserved for the whole chain, not just for this router.
    try:
        for name, value in vars(current).items():
            if name.startswith("_unsloth_"):
                setattr(_sdpa_maybe_muse_glimmer_banded, name, value)
    except Exception:
        pass
    _sdpa_maybe_muse_glimmer_banded._unsloth_muse_glimmer_banded = True
    # Direct assignment: AttentionInterface.register() does not update the
    # global mapping that layers read via ALL_ATTENTION_FUNCTIONS["sdpa"].
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _sdpa_maybe_muse_glimmer_banded


def unpatch_muse_glimmer_banded_sliding_attention():
    """Restore the sdpa entry this router wrapped. For A/B benchmarking and tests."""
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return
    current = ALL_ATTENTION_FUNCTIONS.get("sdpa", None)
    if getattr(current, "_unsloth_muse_glimmer_banded", False) and _ORIG_SDPA[0] is not None:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = _ORIG_SDPA[0]


TEMPORARY_PATCHES.append(patch_muse_glimmer_banded_sliding_attention)
