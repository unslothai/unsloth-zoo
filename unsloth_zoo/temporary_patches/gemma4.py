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

import torch
import os
from .common import TEMPORARY_PATCHES
from .utils import raise_error


# ============================================================================
# Gemma4 num_kv_shared_layers == 0 cache-init fix.
#
# Root cause: transformers/cache_utils.py DynamicCache.__init__ (and
# StaticCache.__init__) contains:
#
#     if hasattr(decoder_config, "num_kv_shared_layers"):
#         layer_types = layer_types[: -decoder_config.num_kv_shared_layers]
#
# When num_kv_shared_layers == 0 (Gemma-4 26B-A4B and 31B both ship this way),
# Python evaluates `layer_types[:-0] == layer_types[:0] == []`, so the cache
# is constructed with zero layer slots and the very first attention forward
# raises `IndexError: list index out of range` from `Cache.update`.
#
# Bug confirmed present in transformers 4.57.6, 5.5.0, and main.
#
# Primary fix (proxy approach): patch `get_text_config` on Gemma4Config and
# Gemma4TextConfig so that, when the resolved text config has
# num_kv_shared_layers == 0, a thin proxy is returned that raises
# AttributeError for that attribute. The upstream `hasattr(decoder_config,
# "num_kv_shared_layers")` check becomes False, the buggy slice is skipped,
# and the cache is built with the full layer list.
#
# Why the proxy approach is preferred over wrapping Cache.__init__:
#   - Only Gemma-4 classes are touched. DynamicCache and StaticCache are
#     byte-identical to upstream for every other model.
#   - No attribute mutation: the real Gemma4TextConfig is never modified, so
#     there are no thread-safety, finally-block, or __slots__ concerns.
#   - Survives Unsloth's compiler cache (`unsloth_compiled_cache/
#     unsloth_compiled_module_gemma4.py`). The compiler copies attention and
#     MLP forward methods but never cache code, and the config classes are
#     not touched by the compiler at all.
#   - Downstream reads of `config.num_kv_shared_layers` (e.g.
#     Gemma4TextMLP.__init__, Gemma4TextAttention.__init__) still see the
#     original 0 because they read from `self.config` directly, not from
#     `self.config.get_text_config(...)`.
#
# Fallback (wrapper approach): a hardened Cache.__init__ wrapper is also
# installed as defense-in-depth. It only fires when the proxy path did not
# apply (for example, a hypothetical future transformers refactor that
# bypasses get_text_config). The wrapper uses a try/finally to hide and
# restore the attribute and bails out to the original init on any mutation
# failure, rather than converting IndexError to TypeError via a None
# sentinel.
# ============================================================================


class _Gemma4KVSharedSafeProxy:
    """Thin read-only proxy around Gemma4TextConfig.

    When `num_kv_shared_layers == 0`, `hasattr(proxy, "num_kv_shared_layers")`
    must return False so that upstream's `layer_types[:-0]` slice is skipped.
    All other attribute lookups (`sliding_window`, `layer_types`,
    `num_hidden_layers`, etc.) forward to the real config object.

    Important: Python looks up dunder methods on the TYPE, not the instance,
    so `__getattr__` alone is NOT sufficient for iteration, len, etc. We
    explicitly forward every dunder method that PreTrainedConfig is known
    to rely on so the proxy survives callers like
    `PreTrainedConfig.validate_token_ids`, which does
    `for value in self.get_text_config(decoder=True): ...`.

    `__slots__` avoids creating a `__dict__`, which keeps the proxy cheap
    and lets default `copy`/`deepcopy` traverse only the `_real` slot.
    """

    __slots__ = ("_real",)

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name):
        # Only invoked when normal attribute lookup fails (i.e. not a slot
        # and not a method of this proxy class).
        if name == "num_kv_shared_layers":
            raise AttributeError(
                "num_kv_shared_layers is 0 (no KV sharing), hidden from "
                "the cache constructor to avoid layer_types[:-0] == [] bug"
            )
        return getattr(object.__getattribute__(self, "_real"), name)

    def get_text_config(self, decoder=None, encoder=None):
        # If upstream recursively calls get_text_config on the proxy, return
        # self so the proxy is not unwrapped back into a raw config.
        return self

    # ---- dict-like dunder forwarding ----
    # PreTrainedConfig.__iter__ yields attribute names from self.__dict__.
    # Validators such as validate_token_ids rely on this iteration.
    def __iter__(self):
        return iter(object.__getattribute__(self, "_real"))

    def __len__(self):
        real = object.__getattribute__(self, "_real")
        try:
            return len(real)
        except TypeError:
            return len(getattr(real, "__dict__", {}))

    def __contains__(self, item):
        if item == "num_kv_shared_layers":
            return False
        real = object.__getattribute__(self, "_real")
        try:
            return item in real
        except TypeError:
            return item in getattr(real, "__dict__", {})

    def __getitem__(self, key):
        if key == "num_kv_shared_layers":
            raise KeyError(key)
        real = object.__getattribute__(self, "_real")
        try:
            return real[key]
        except TypeError:
            return getattr(real, key)

    def __eq__(self, other):
        if isinstance(other, _Gemma4KVSharedSafeProxy):
            other = object.__getattribute__(other, "_real")
        return object.__getattribute__(self, "_real") == other

    def __hash__(self):
        try:
            return hash(object.__getattribute__(self, "_real"))
        except TypeError:
            return id(self)

    def __bool__(self):
        return bool(object.__getattribute__(self, "_real"))

    def __repr__(self):
        return f"_Gemma4KVSharedSafeProxy({object.__getattribute__(self, '_real')!r})"


def _wrap_get_text_config_for_kv_zero(cls):
    """Replace `cls.get_text_config` with a version that wraps the result
    in `_Gemma4KVSharedSafeProxy` whenever the resolved config has
    `num_kv_shared_layers == 0`. Idempotent.
    """
    _sentinel = "_unsloth_kv_shared_get_text_config_patched"
    if getattr(cls.get_text_config, _sentinel, False):
        return
    _orig = cls.get_text_config

    def get_text_config(self, decoder=None, encoder=None):
        result = _orig(self, decoder=decoder, encoder=encoder)
        # -1 sentinel distinguishes "absent" from "present and 0".
        if result is not None and getattr(result, "num_kv_shared_layers", -1) == 0:
            return _Gemma4KVSharedSafeProxy(result)
        return result

    setattr(get_text_config, _sentinel, True)
    get_text_config.__qualname__ = _orig.__qualname__
    get_text_config.__doc__ = _orig.__doc__
    cls.get_text_config = get_text_config


def patch_Gemma4Config_kv_shared_zero():
    """Patch Gemma4Config.get_text_config. This is the path exercised by
    StaticCache (called from transformers.generation.utils) and any caller
    that passes the top-level `model.config` to a cache constructor."""
    try:
        from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
    except ImportError:
        # transformers without Gemma-4 (e.g. 4.57.6): nothing to patch.
        return
    except Exception as e:
        return raise_error("Gemma4Config.get_text_config kv_shared_zero fix", e)
    try:
        _wrap_get_text_config_for_kv_zero(Gemma4Config)
    except Exception as e:
        return raise_error("Gemma4Config.get_text_config kv_shared_zero fix", e)
pass
TEMPORARY_PATCHES.append(patch_Gemma4Config_kv_shared_zero)


def patch_Gemma4TextConfig_kv_shared_zero():
    """Patch Gemma4TextConfig.get_text_config. This is the path exercised by
    DynamicCache when `Gemma4TextModel.forward` auto-constructs the cache
    with `DynamicCache(config=self.config)` where `self.config` is the
    already-nested Gemma4TextConfig."""
    try:
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    except ImportError:
        return
    except Exception as e:
        return raise_error("Gemma4TextConfig.get_text_config kv_shared_zero fix", e)
    try:
        _wrap_get_text_config_for_kv_zero(Gemma4TextConfig)
    except Exception as e:
        return raise_error("Gemma4TextConfig.get_text_config kv_shared_zero fix", e)
pass
TEMPORARY_PATCHES.append(patch_Gemma4TextConfig_kv_shared_zero)


# ----------------------------------------------------------------------------
# Defense-in-depth: hardened Cache.__init__ wrapper.
#
# This only takes effect if the proxy patches above did NOT apply (for
# example, a hypothetical future transformers that bypasses get_text_config,
# or a third-party composite model that wraps Gemma4TextConfig without
# exposing it via get_text_config). In the steady state this wrapper is a
# pure pass-through: the proxy catches the bug first.
# ----------------------------------------------------------------------------

def _make_kv_shared_zero_safe_init(_orig_init, _resolve_decoder_config):
    """Return a hardened `__init__` wrapper.

    Behavior:
      1. Resolve the decoder config the cache class will see.
      2. If num_kv_shared_layers is present AND equals 0 AND we have not
         already been bypassed by the proxy (the proxy raises AttributeError
         from hasattr, so the check short-circuits to pass-through), then
         transiently delete the attribute and call the original init inside
         a try/finally that restores the value.
      3. If the attribute cannot be deleted (e.g. dataclass __slots__, frozen
         config, C extension descriptor), BAIL OUT to the original init,
         preserving the original IndexError rather than converting it to a
         TypeError via a None sentinel.
    """

    def __init__(self, *args, **kwargs):
        decoder_config = None
        try:
            decoder_config = _resolve_decoder_config(args, kwargs)
        except Exception:
            decoder_config = None

        # Proxy path already bypassed the buggy branch via AttributeError,
        # so hasattr returns False for the Gemma-4 zero case and this whole
        # block is skipped. This check only fires if the proxy patch did not
        # apply for some reason.
        if (
            decoder_config is not None
            and hasattr(decoder_config, "num_kv_shared_layers")
            and getattr(decoder_config, "num_kv_shared_layers", -1) == 0
        ):
            try:
                del decoder_config.num_kv_shared_layers
            except (AttributeError, TypeError):
                # Attribute cannot be mutated: fall through to original
                # init and let the upstream error surface (no worse than
                # without the patch).
                return _orig_init(self, *args, **kwargs)
            try:
                _orig_init(self, *args, **kwargs)
            finally:
                # Always restore, even if __init__ raised.
                try:
                    object.__setattr__(decoder_config, "num_kv_shared_layers", 0)
                except Exception:
                    try:
                        setattr(decoder_config, "num_kv_shared_layers", 0)
                    except Exception:
                        pass
            return

        return _orig_init(self, *args, **kwargs)

    __init__._unsloth_kv_shared_zero_patched = True
    __init__.__qualname__ = getattr(_orig_init, "__qualname__", "__init__")
    __init__.__doc__ = getattr(_orig_init, "__doc__", None)
    return __init__


def patch_DynamicCache_kv_shared_zero():
    try:
        from transformers.cache_utils import DynamicCache
    except Exception as e:
        return raise_error("DynamicCache num_kv_shared_layers fix", e)

    if getattr(DynamicCache.__init__, "_unsloth_kv_shared_zero_patched", False):
        return

    def _resolve(args, kwargs):
        # DynamicCache.__init__(self, ddp_cache_data=None, config=None, ...)
        config = kwargs.get("config", None)
        if config is None and len(args) >= 2:
            config = args[1]
        if config is None:
            return None
        try:
            return config.get_text_config(decoder=True)
        except Exception:
            return None

    DynamicCache.__init__ = _make_kv_shared_zero_safe_init(DynamicCache.__init__, _resolve)
pass
TEMPORARY_PATCHES.append(patch_DynamicCache_kv_shared_zero)


def patch_StaticCache_kv_shared_zero():
    try:
        from transformers.cache_utils import StaticCache
    except Exception as e:
        return raise_error("StaticCache num_kv_shared_layers fix", e)

    if getattr(StaticCache.__init__, "_unsloth_kv_shared_zero_patched", False):
        return

    def _resolve(args, kwargs):
        # StaticCache.__init__(self, config, max_cache_len, ...)
        config = kwargs.get("config", None)
        if config is None and len(args) >= 1:
            config = args[0]
        if config is None:
            return None
        try:
            return config.get_text_config(decoder=True)
        except Exception:
            return None

    StaticCache.__init__ = _make_kv_shared_zero_safe_init(StaticCache.__init__, _resolve)
pass
TEMPORARY_PATCHES.append(patch_StaticCache_kv_shared_zero)


# ============================================================================
# Gemma4 AudioAttention patch - fix attention_invalid_logits_value overflow in fp16
# The config value is -1e9 which overflows fp16 max (65504).
# On Tesla T4, autocast can downcast attn_weights to fp16, causing masked_fill to fail.
# ============================================================================

def patch_Gemma4AudioAttention():
    try:
        import transformers.models.gemma4.modeling_gemma4
        Gemma4AudioAttention = getattr(
            transformers.models.gemma4.modeling_gemma4, "Gemma4AudioAttention", None
        )
        if Gemma4AudioAttention is None:
            return
    except Exception as e:
        return raise_error("Gemma4AudioAttention.forward", e)

    _original_audio_attn_forward = Gemma4AudioAttention.forward

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        # Clamp attention_invalid_logits_value to dtype-safe range before attention
        # Only needed for fp16 (Tesla T4) where -1e9 overflows. bf16 supports up to ~3.4e38.
        original_value = getattr(self.config, "attention_invalid_logits_value", -1e9)
        needs_clamp = hidden_states.dtype == torch.float16 and abs(original_value) > 65000.0
        if needs_clamp:
            self.config.attention_invalid_logits_value = -65000.0
        try:
            result = _original_audio_attn_forward(self, hidden_states, position_embeddings, attention_mask, **kwargs)
        finally:
            if needs_clamp:
                self.config.attention_invalid_logits_value = original_value
        return result
    pass
    Gemma4AudioAttention.forward = forward
pass
TEMPORARY_PATCHES.append(patch_Gemma4AudioAttention)
