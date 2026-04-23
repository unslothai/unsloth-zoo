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
from .utils import raise_error, patch_function


# ============================================================================
# Gemma-4 variant summary (affects which fix engages which model):
#
#   - Gemma-4 31B:      num_kv_shared_layers = 0  -> Fix 1 engages, Fix 2 is a no-op
#   - Gemma-4 26B-A4B:  num_kv_shared_layers = 0  -> Fix 1 engages, Fix 2 is a no-op
#   - Gemma-4 E4B:      num_kv_shared_layers = 18 -> Fix 1 is a no-op, Fix 2 engages
#   - Gemma-4 E2B:      num_kv_shared_layers = 20 -> Fix 1 is a no-op, Fix 2 engages
#   - Gemma3n:          num_kv_shared_layers = 15 -> both are no-ops (different model)
#   - Llama/Qwen/etc:   attribute absent          -> both are no-ops
#
# The two fixes are mutually exclusive at runtime: Fix 1 only acts when the
# value is exactly 0, Fix 2 only acts when the value is strictly greater
# than 0. Non-Gemma-4 models never hit either branch because the attribute
# is not present.
# ============================================================================


# ============================================================================
# Fix 1: num_kv_shared_layers == 0 cache-init bug (Gemma-4 31B and 26B-A4B).
#
# Root cause: transformers/cache_utils.py DynamicCache.__init__ (and StaticCache.
# __init__) contains:
#
#     if hasattr(decoder_config, "num_kv_shared_layers"):
#         layer_types = layer_types[: -decoder_config.num_kv_shared_layers]
#
# When num_kv_shared_layers == 0 (Gemma-4 26B-A4B and 31B both ship this way),
# Python evaluates `layer_types[:-0] == layer_types[:0] == []`, so the cache is
# constructed with zero layer slots and the very first attention forward
# raises `IndexError: list index out of range` from `Cache.update`.
#
# Bug confirmed present in transformers 4.57.6, 5.5.0, and main.
# Filed for upstream fix.
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
#     unsloth_compiled_module_gemma4.py`) trivially. The compiler copies
#     attention/MLP forward methods but never cache code, and the config
#     classes are not touched by the compiler at all.
#   - Downstream reads of `config.num_kv_shared_layers` (e.g.
#     Gemma4TextMLP.__init__, Gemma4TextAttention.__init__) still see the
#     original 0 because they read from `self.config` directly, not from
#     `self.config.get_text_config(...)`.
#
# Fallback (wrapper approach): a hardened Cache.__init__ wrapper is also
# installed as defense-in-depth. It only fires when the proxy path did not
# apply (e.g. a future transformers refactor that bypasses get_text_config).
# The wrapper uses a try/finally to hide-and-restore the attribute and bails
# out to the original init on any mutation failure (rather than converting
# IndexError -> TypeError by setting a None sentinel).
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

    def __setattr__(self, name, value):
        # Forward writes to the wrapped config so callers like vLLM's
        # ``VllmConfig.with_hf_config`` (which assigns
        # ``tie_word_embeddings`` onto ``get_text_config()``) do not hit
        # ``AttributeError: ... no __dict__ for setting new attributes``.
        # Allow writes to our own ``_real`` slot (used by deepcopy /
        # ``__init__``) to go through object.__setattr__.
        if name == "_real":
            object.__setattr__(self, name, value)
            return
        try:
            real = object.__getattribute__(self, "_real")
        except AttributeError:
            # Instance still being initialized (e.g. deepcopy sets slots
            # after __new__). Fall back to the slot.
            object.__setattr__(self, name, value)
            return
        setattr(real, name, value)

    def __delattr__(self, name):
        if name == "_real":
            object.__delattr__(self, name)
            return
        try:
            real = object.__getattribute__(self, "_real")
        except AttributeError:
            object.__delattr__(self, name)
            return
        delattr(real, name)

    def __getattr__(self, name):
        # Only invoked when normal attribute lookup fails (i.e. not a slot
        # and not a method of this proxy class).
        if name == "num_kv_shared_layers":
            raise AttributeError(
                "num_kv_shared_layers is 0 (no KV sharing) -- hidden from "
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
        # transformers < 5.x (e.g. 4.57.6) has no Gemma-4 -> nothing to patch.
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
         config, C extension descriptor), BAIL OUT to the original init --
         preserving the original IndexError rather than converting it to a
         TypeError via a None sentinel (this was a bug in the first draft).
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
                # Attribute cannot be mutated -- fall through to original
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
# Fix 2: use_cache=False with KV sharing produces garbage (Gemma-4 E2B and E4B).
#
# Root cause: Gemma4TextAttention.forward uses
#
#     if self.is_kv_shared_layer and past_key_values is not None:
#         key_states, value_states = past_key_values.shared_layers[...]
#     else:
#         # compute K/V locally from current hidden_states
#
# The cache is the ONLY place where the KV values produced by the
# "store_full_length_kv" layers are stashed for later reuse by the
# is_kv_shared_layer layers. When the caller passes `use_cache=False`,
# Gemma4TextModel.forward skips auto-construction of past_key_values:
#
#     if use_cache and past_key_values is None:
#         past_key_values = DynamicCache(config=self.config)
#
# so every is_kv_shared_layer falls through to the `else` branch and
# recomputes K/V from the current layer's hidden states, which is wrong.
# Symptoms: training loss explodes, logits become garbage. See
# huggingface/transformers#45242.
#
# Affects models with num_kv_shared_layers > 0 (Gemma-4 E2B=20, E4B=18).
# Does NOT affect 31B or 26B-A4B (num_kv_shared_layers=0 -> no layer is
# is_kv_shared_layer -> path is byte-identical with or without cache).
#
# Fix: wrap Gemma4TextModel.forward (and Gemma4Model.forward for the
# multimodal parent) so that when num_kv_shared_layers > 0 and the caller
# passes past_key_values=None with use_cache falsy, we transparently
# construct a DynamicCache, run the original forward, then drop
# past_key_values from the returned `BaseModelOutputWithPast` to preserve
# the caller's use_cache=False semantics.
#
# This wrapper is a pure pass-through for:
#   - Any model that doesn't expose num_kv_shared_layers (not Gemma-4)
#   - Gemma-4 31B / 26B-A4B (num_kv_shared_layers == 0)
#   - Any Gemma-4 call that already has past_key_values or use_cache=True
# ============================================================================

def _make_kv_shared_use_cache_false_safe_forward(_orig_forward):
    """Wrap a Gemma-4 text-model-style forward so it builds an internal
    DynamicCache when KV sharing is active but the caller passed use_cache=False.
    The internal cache is nulled out in the returned output so the caller's
    semantics are preserved.

    Argument resolution uses `inspect.signature.bind_partial` so that
    `past_key_values` and `use_cache` are correctly resolved whether the
    caller passed them positionally or by keyword. Calling the inner forward
    via `bound.args` / `bound.kwargs` (rather than mixing positional and
    keyword) avoids the `TypeError: got multiple values for argument`
    case that a naive `kwargs[...] = ...` injection would trigger when the
    caller already passed the same argument positionally.
    """
    import inspect

    try:
        _sig = inspect.signature(_orig_forward)
    except (TypeError, ValueError):
        _sig = None

    def forward(self, *args, **kwargs):
        num_kv_shared = getattr(
            getattr(self, "config", None), "num_kv_shared_layers", 0
        )
        # Nothing to do for variants without KV sharing (31B, 26B-A4B, non-Gemma-4).
        if not num_kv_shared or num_kv_shared <= 0:
            return _orig_forward(self, *args, **kwargs)

        # Resolve past_key_values and use_cache from EITHER args or kwargs.
        # bind_partial handles both positional and keyword forms uniformly.
        if _sig is None:
            # Fallback if signature introspection failed: kwargs-only path.
            past_key_values = kwargs.get("past_key_values", None)
            use_cache_kw = kwargs.get("use_cache", None)
            arguments = None
        else:
            try:
                bound = _sig.bind_partial(self, *args, **kwargs)
            except TypeError:
                # Caller passed something the original forward will reject;
                # let the original forward raise the canonical error.
                return _orig_forward(self, *args, **kwargs)
            arguments = bound.arguments
            past_key_values = arguments.get("past_key_values", None)
            use_cache_kw = arguments.get("use_cache", None)

        use_cache_resolved = (
            use_cache_kw
            if use_cache_kw is not None
            else getattr(self.config, "use_cache", True)
        )

        # Only intervene when the original forward would have left
        # past_key_values as None (i.e. use_cache is explicitly False or
        # config.use_cache is False and the caller did not pass one in).
        if past_key_values is not None or use_cache_resolved:
            return _orig_forward(self, *args, **kwargs)

        # Build a local cache JUST for cross-layer KV sharing; do not leak
        # it back to the caller since they asked for use_cache=False.
        try:
            from transformers.cache_utils import DynamicCache
        except Exception:
            return _orig_forward(self, *args, **kwargs)

        # Inject the cache and force use_cache=True on the inner call so the
        # original forward takes the same mask/position path it would have
        # with a real cache. The outer result is cleaned up below to preserve
        # the caller's use_cache=False contract.
        if arguments is not None:
            arguments["past_key_values"] = DynamicCache(config=self.config)
            arguments["use_cache"] = True
            # Re-bind through args/kwargs to avoid duplicate-keyword TypeErrors
            # if the caller had passed past_key_values or use_cache positionally.
            inner_args = bound.args
            inner_kwargs = bound.kwargs
            # bound.args / bound.kwargs are derived from `arguments`, so they
            # already contain the injected values in the correct slot.
            result = _orig_forward(*inner_args, **inner_kwargs)
        else:
            kwargs["past_key_values"] = DynamicCache(config=self.config)
            kwargs["use_cache"] = True
            result = _orig_forward(self, *args, **kwargs)

        # Preserve the caller's use_cache=False contract: null the cache
        # from EVERY shape the result might have:
        #   1. ModelOutput (BaseModelOutputWithPast) -- attribute + dict slot
        #   2. plain dict
        #   3. tuple (Gemma4Model.forward is decorated with @can_return_tuple,
        #      so return_dict=False produces a tuple where past_key_values is
        #      typically the second element of BaseModelOutputWithPast.to_tuple())
        try:
            if isinstance(result, tuple):
                # ModelOutput.to_tuple() drops None entries, so the cache
                # may not be at a fixed index. Reconstruct by scanning for
                # the injected DynamicCache instance and dropping it.
                injected = (
                    arguments["past_key_values"]
                    if arguments is not None
                    else kwargs["past_key_values"]
                )
                new_items = tuple(None if x is injected else x for x in result)
                result = new_items
            else:
                # ModelOutput subclasses behave like dicts internally; using
                # __setitem__ keeps both the attribute slot and the underlying
                # OrderedDict in sync. Fall back to attribute assignment if
                # the result does not support __setitem__.
                set_via_item = False
                try:
                    if hasattr(result, "__setitem__") and "past_key_values" in result:
                        result["past_key_values"] = None
                        set_via_item = True
                except (TypeError, KeyError):
                    set_via_item = False
                if not set_via_item and hasattr(result, "past_key_values"):
                    try:
                        result.past_key_values = None
                    except (AttributeError, TypeError):
                        pass
        except Exception:
            pass
        return result

    forward._unsloth_kv_shared_use_cache_false_patched = True
    forward.__qualname__ = getattr(_orig_forward, "__qualname__", "forward")
    forward.__doc__ = getattr(_orig_forward, "__doc__", None)
    return forward


def _patch_forward_for_kv_shared_no_cache(cls):
    """Install the wrapper on `cls.forward`. Idempotent."""
    if getattr(cls.forward, "_unsloth_kv_shared_use_cache_false_patched", False):
        return
    cls.forward = _make_kv_shared_use_cache_false_safe_forward(cls.forward)


def patch_Gemma4TextModel_forward_kv_shared_no_cache():
    """Patch Gemma4TextModel.forward (the language decoder). This is the class
    that auto-creates the cache and drives per-layer attention. Fires for
    text-only model loading (Gemma4ForCausalLM -> Gemma4TextModel)."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
    except ImportError:
        return
    except Exception as e:
        return raise_error("Gemma4TextModel.forward use_cache=False fix", e)
    try:
        _patch_forward_for_kv_shared_no_cache(Gemma4TextModel)
    except Exception as e:
        return raise_error("Gemma4TextModel.forward use_cache=False fix", e)
pass
TEMPORARY_PATCHES.append(patch_Gemma4TextModel_forward_kv_shared_no_cache)


def patch_Gemma4Model_forward_kv_shared_no_cache():
    """Patch Gemma4Model.forward (the multimodal parent). Its forward calls
    into the nested Gemma4TextModel, so patching Gemma4TextModel alone is
    enough to fix attention, but we also wrap Gemma4Model as a safety net
    in case a future refactor moves cache auto-creation up a level."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4Model
    except ImportError:
        return
    except Exception as e:
        return raise_error("Gemma4Model.forward use_cache=False fix", e)
    try:
        _patch_forward_for_kv_shared_no_cache(Gemma4Model)
    except Exception as e:
        return raise_error("Gemma4Model.forward use_cache=False fix", e)
pass
TEMPORARY_PATCHES.append(patch_Gemma4Model_forward_kv_shared_no_cache)


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


# Gemma-4 float16 MLP overflow fix.
#
# `down_proj(act_fn(gate_proj(x)) * up_proj(x))` overflows fp16 at layers.0
# (E2B) / layers.1 (E4B): the product + fp16 matmul accumulator saturate to
# +-inf, poison the residual stream, and generation samples NaN logits that
# trip the CUDA categorical sampler on GRPO step ~2.
#
# Fix: fp32 gate*up, clamp to a safe bound, fp16 cast, nan_to_num on the
# down_proj output. Gated on gate output dtype so bf16/fp32 users see no
# change and no env flag is required. RMSNorm / Attention / Embedding
# patches are unnecessary (verified by bisection - identical loss/kl/grad
# trajectories).


def patch_Gemma4TextMLP():
    """fp16 overflow clamp for Gemma4TextMLP.

    Does gate*up in fp32, clamps to a safe fp16 bound, then nan_to_nums
    the down_proj output. Self-gated on gate dtype - no-op on bf16/fp32.
    """
    try:
        import transformers.models.gemma4.modeling_gemma4 as mod
    except ImportError:
        return
    try:
        Gemma4TextMLP = mod.Gemma4TextMLP
    except AttributeError as e:
        return raise_error("Gemma4TextMLP.forward", e)

    # Largest value representable in both fp16 and bf16 (65536 rounds to
    # fp16 inf).
    _SAFE_FP16 = 65280.0

    def forward(self, x):
        gate = self.gate_proj(x)
        # Check matmul output dtype so autocast / PEFT fp16 casts are caught.
        if gate.dtype != torch.float16:
            return self.down_proj(self.act_fn(gate) * self.up_proj(x))
        product = self.act_fn(gate.float()) * self.up_proj(x).float()
        product = torch.clamp(product, min=-_SAFE_FP16, max=_SAFE_FP16)
        out = self.down_proj(product.to(gate.dtype))
        # Zero overflows so the residual identity path survives.
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        patch_function(
            Gemma4TextMLP, "forward", forward, fullgraph=False,
        )
    except Exception as e:
        return raise_error("Gemma4TextMLP.forward", e)
pass
TEMPORARY_PATCHES.append(patch_Gemma4TextMLP)
