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
# Gemma-4 variant summary (which fix engages which model):
#   - 31B / 26B-A4B:  num_kv_shared_layers = 0  -> Fix 1 engages, Fix 2 no-op
#   - E4B (18) / E2B (20): num_kv_shared_layers > 0 -> Fix 1 no-op, Fix 2 engages
#   - Gemma3n (15) / Llama / Qwen / etc.: both no-ops (different / absent attr)
# The fixes are mutually exclusive: Fix 1 acts only when the value is exactly
# 0, Fix 2 only when it is > 0.
# ============================================================================


# ============================================================================
# Fix 1: num_kv_shared_layers == 0 cache-init bug (Gemma-4 31B and 26B-A4B).
#
# Root cause: transformers/cache_utils.py DynamicCache/StaticCache __init__:
#     if hasattr(decoder_config, "num_kv_shared_layers"):
#         layer_types = layer_types[: -decoder_config.num_kv_shared_layers]
# When num_kv_shared_layers == 0, `layer_types[:-0] == []`, so the cache gets
# zero layer slots and the first attention forward raises IndexError from
# Cache.update. Present in transformers 4.57.6, 5.5.0, and main (filed upstream).
#
# Primary fix (proxy): patch get_text_config on Gemma4Config / Gemma4TextConfig
# to return a proxy that raises AttributeError for num_kv_shared_layers when it
# is 0, so the upstream hasattr check is False and the buggy slice is skipped.
# Preferred over wrapping Cache.__init__ because it touches only Gemma-4
# classes, mutates nothing, and survives the compiler cache. Downstream reads
# of config.num_kv_shared_layers still see 0 (they read self.config directly).
#
# Fallback (wrapper): a hardened Cache.__init__ wrapper is installed as
# defense-in-depth; it only fires if the proxy path did not apply.
# ============================================================================


class _Gemma4KVSharedSafeProxy:
    """Read-only proxy around Gemma4TextConfig hiding num_kv_shared_layers when 0.

    Makes `hasattr(proxy, "num_kv_shared_layers")` False so upstream's
    `layer_types[:-0]` slice is skipped; all other lookups forward to the real
    config. Dunder methods are forwarded explicitly because Python resolves them
    on the type, so callers like PreTrainedConfig.validate_token_ids (which
    iterates the config) still work. `__slots__` keeps the proxy cheap.
    """

    __slots__ = ("_real",)

    # __dict__/vars() still expose num_kv_shared_layers (no instance dict -> forwards
    # to _real); keep it that way -- to_dict() deepcopies __dict__, so sealing it
    # would drop the field from serialization. #6089

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name):
        # Only invoked when normal attribute lookup fails.
        if name == "num_kv_shared_layers":
            raise AttributeError(
                "num_kv_shared_layers is 0 (no KV sharing) -- hidden from "
                "the cache constructor to avoid layer_types[:-0] == [] bug"
            )
        return getattr(object.__getattribute__(self, "_real"), name)

    def get_text_config(self, decoder=None, encoder=None):
        # Return self so recursive get_text_config calls don't unwrap the proxy.
        return self

    # ---- dict-like dunder forwarding (used by config validators) ----
    def __iter__(self):
        # Hide num_kv_shared_layers, like __getattr__/__contains__/__getitem__:
        # validate_token_ids does `for n in cfg: getattr(cfg, n)`, so yielding it
        # would re-raise the AttributeError from __getattr__. See unslothai/unsloth#6089.
        for name in object.__getattribute__(self, "_real"):
            if name == "num_kv_shared_layers":
                continue
            yield name

    def __len__(self):
        # Consistent with __iter__ (hidden attr excluded).
        return sum(1 for _ in self)

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
    """Wrap cls.get_text_config to return a proxy when num_kv_shared_layers == 0. Idempotent."""
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
    """Patch Gemma4Config.get_text_config (StaticCache / top-level config path)."""
    try:
        from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
    except ImportError:
        # transformers < 5.x has no Gemma-4 -> nothing to patch.
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
    """Patch Gemma4TextConfig.get_text_config (DynamicCache auto-construct path)."""
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
# Defense-in-depth: hardened Cache.__init__ wrapper, only active if the proxy
# patches above did not apply. In steady state it is a pure pass-through.
# ----------------------------------------------------------------------------

def _make_kv_shared_zero_safe_init(_orig_init, _resolve_decoder_config):
    """Hardened cache __init__: transiently delete num_kv_shared_layers==0 around
    the original init (try/finally restores it), bailing to the original init if
    the attribute cannot be deleted so the upstream error surfaces unchanged.
    """

    def __init__(self, *args, **kwargs):
        decoder_config = None
        try:
            decoder_config = _resolve_decoder_config(args, kwargs)
        except Exception:
            decoder_config = None

        # Skipped when the proxy path applied (hasattr returns False there);
        # this only fires if the proxy patch did not apply.
        if (
            decoder_config is not None
            and hasattr(decoder_config, "num_kv_shared_layers")
            and getattr(decoder_config, "num_kv_shared_layers", -1) == 0
        ):
            try:
                del decoder_config.num_kv_shared_layers
            except (AttributeError, TypeError):
                # Cannot mutate; fall through to original init.
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
# Gemma4TextAttention.forward only reuses stored K/V from the cache for
# is_kv_shared_layer layers; with use_cache=False, Gemma4TextModel.forward
# never auto-constructs the cache, so those layers recompute K/V from the wrong
# hidden states. Symptoms: training loss explodes, logits become garbage.
# See huggingface/transformers#45242.
#
# Affects num_kv_shared_layers > 0 (E2B=20, E4B=18); 31B / 26B-A4B (=0) are
# unaffected.
#
# Fix: wrap Gemma4TextModel.forward (and Gemma4Model.forward) so that when KV
# sharing is active and the caller passes past_key_values=None with use_cache
# falsy, we build a DynamicCache, run the original forward, then drop it from
# the output to preserve use_cache=False semantics. Pass-through otherwise.
# ============================================================================

def _make_kv_shared_use_cache_false_safe_forward(_orig_forward):
    """Wrap a Gemma-4 forward to build an internal DynamicCache when KV sharing
    is active but the caller passed use_cache=False, nulling it from the output.

    Uses inspect.signature.bind_partial so past_key_values / use_cache resolve
    whether passed positionally or by keyword, and calls the inner forward via
    bound.args/bound.kwargs to avoid duplicate-argument TypeErrors.
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
        # No-op without KV sharing (31B, 26B-A4B, non-Gemma-4).
        if not num_kv_shared or num_kv_shared <= 0:
            return _orig_forward(self, *args, **kwargs)

        # Resolve past_key_values / use_cache from args or kwargs.
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

        # Only intervene when the original forward would leave past_key_values
        # as None (use_cache falsy and no cache passed in).
        if past_key_values is not None or use_cache_resolved:
            return _orig_forward(self, *args, **kwargs)

        # Local cache for cross-layer KV sharing only; not leaked to the caller.
        try:
            from transformers.cache_utils import DynamicCache
        except Exception:
            return _orig_forward(self, *args, **kwargs)

        # Inject the cache and force use_cache=True so the inner forward takes
        # the same mask/position path; the result is cleaned up below.
        if arguments is not None:
            arguments["past_key_values"] = DynamicCache(config=self.config)
            arguments["use_cache"] = True
            # bound.args/kwargs derive from `arguments`, so the injected values
            # land in the correct slot without duplicate-keyword TypeErrors.
            inner_args = bound.args
            inner_kwargs = bound.kwargs
            result = _orig_forward(*inner_args, **inner_kwargs)
        else:
            kwargs["past_key_values"] = DynamicCache(config=self.config)
            kwargs["use_cache"] = True
            result = _orig_forward(self, *args, **kwargs)

        # Null the cache from every result shape (ModelOutput, dict, tuple) to
        # preserve the caller's use_cache=False contract.
        try:
            if isinstance(result, tuple):
                # to_tuple() drops None entries, so the cache index isn't fixed;
                # scan for the injected DynamicCache and drop it.
                injected = (
                    arguments["past_key_values"]
                    if arguments is not None
                    else kwargs["past_key_values"]
                )
                new_items = tuple(None if x is injected else x for x in result)
                result = new_items
            else:
                # ModelOutput acts dict-like; __setitem__ keeps the attribute
                # slot and OrderedDict in sync. Fall back to attribute assignment.
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


# NOTE: On the Gemma-4 E-series (E2B / E4B, num_kv_shared_layers > 0) in
# transformers 5.5.0 the cache-free forward (use_cache=False) is the CORRECT
# path: teacher-forced loss goes to 0 and greedy decode reproduces the target,
# while the KV-cache path (use_cache=True) is the broken one
# (huggingface/transformers#45242). Wrapping the forward to build an internal
# cache for use_cache=False therefore INVERTS correctness - it forces the broken
# cached computation onto the path that was already right, so a memorised model
# generates garbage and teacher-forced loss explodes to ~12. We keep the wrapper
# defined above for reference, but do NOT register it. Generation safety is
# handled in unsloth by defaulting Gemma-4 shared-KV generate() to use_cache=False.
def patch_Gemma4TextModel_forward_kv_shared_no_cache():
    return


def patch_Gemma4Model_forward_kv_shared_no_cache():
    return


# Gemma4 AudioAttention: config attention_invalid_logits_value (-1e9) overflows
# fp16 max (65504); on Tesla T4 autocast downcasts attn_weights to fp16 and
# masked_fill fails.

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
        # Clamp attention_invalid_logits_value: only fp16 (Tesla T4) overflows
        # at -1e9; bf16 supports up to ~3.4e38.
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


# Gemma4 PEFT reload: make PeftModel.from_pretrained handle adapters saved
# against Gemma4ClippableLinear.inner .linear modules.

def patch_Gemma4ClippableLinear_peft_reload():
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
        from peft.tuners.lora.model import LoraModel
    except ImportError:
        return
    except Exception as e:
        return raise_error("Gemma4ClippableLinear PEFT reload", e)

    original_create_and_replace = LoraModel._create_and_replace
    if getattr(original_create_and_replace, "_unsloth_gemma4_clippable_linear_patched", False):
        return

    def create_and_replace(
        self,
        peft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key=None,
        **kwargs,
    ):
        # Match by identity OR name+structure: the compiler emits a duplicate
        # Gemma4ClippableLinear, so a compiled model's modules aren't isinstance of the
        # transformers class (breaks a 2nd adapter, e.g. GRPO "ref"). #6089
        if isinstance(target, Gemma4ClippableLinear) or (
            type(target).__name__ == "Gemma4ClippableLinear" and hasattr(target, "linear")
        ):
            return original_create_and_replace(
                self,
                peft_config,
                adapter_name,
                target.linear,
                "linear",
                target,
                current_key=current_key,
                **kwargs,
            )
        return original_create_and_replace(
            self,
            peft_config,
            adapter_name,
            target,
            target_name,
            parent,
            current_key=current_key,
            **kwargs,
        )

    create_and_replace._unsloth_gemma4_clippable_linear_patched = True
    create_and_replace._unsloth_original_create_and_replace = original_create_and_replace
    LoraModel._create_and_replace = create_and_replace
pass
TEMPORARY_PATCHES.append(patch_Gemma4ClippableLinear_peft_reload)


# Gemma-4 float16 MLP overflow fix.
#
# `down_proj(act_fn(gate_proj(x)) * up_proj(x))` overflows fp16 at layers.0
# (E2B) / layers.1 (E4B): product + matmul accumulator saturate to +-inf,
# poison the residual stream, and generation samples NaN logits that trip the
# CUDA categorical sampler on GRPO step ~2. Fix: fp32 gate*up, clamp to a safe
# bound, fp16 cast, nan_to_num. Gated on gate dtype so bf16/fp32 see no change.


def patch_Gemma4TextMLP():
    """fp16 overflow clamp for Gemma4TextMLP (no-op on bf16/fp32)."""
    try:
        import transformers.models.gemma4.modeling_gemma4 as mod
    except ImportError:
        return
    try:
        Gemma4TextMLP = mod.Gemma4TextMLP
    except AttributeError as e:
        return raise_error("Gemma4TextMLP.forward", e)

    # Largest value representable in both fp16 and bf16 (65536 rounds to fp16 inf).
    _SAFE_FP16 = 65280.0

    def forward(self, x):
        gate = self.gate_proj(x)
        # Check matmul output dtype to catch autocast / PEFT fp16 casts.
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
