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
# unaffected. Confirmed on transformers 5.5.0 + gemma-4-E2B-it: use_cache=False
# teacher-forced loss ~16 vs ~6.8 (use_cache=True); under gradient checkpointing
# all 20 shared layers fall back to recomputing their own KV.
#
# Version window: gemma-4 first ships in transformers 5.5.0, and the buggy
# cache-dependent path exists only in 5.5.0 and 5.5.1. From 5.5.2 onward (the
# native `shared_kv_states` arg was backported into the 5.5.x line, then 5.6.0+)
# the cache-free path is fixed upstream. The gate below detects this by signature,
# not version, so it stays correct across the mid-patch-release backport.
#
# Fix: decouple the within-forward shared KV from past_key_values. The model
# forward attaches a fresh tiny carrier (a shared_layers dict + no-op update) to
# every Gemma4TextAttention, and the attention uses it whenever the real
# past_key_values is None. The store/retrieve then works under use_cache=False
# AND gradient checkpointing (where GradientCheckpointingLayer nulls the
# past_key_values kwarg before the attention, defeating any injected cache).
# Mirrors transformers' own later `shared_kv_states` fix; capability-gated to a
# no-op on builds that already pass `shared_kv_states` (forwards-compatibility).
# ============================================================================

def _gemma4_kv_sharing_needs_cache():
    """True iff this transformers build plumbs Gemma-4 cross-layer KV sharing through
    ``past_key_values`` (the buggy, cache-dependent mechanism: transformers 5.5.0-5.5.1).
    Newer builds (5.5.2+, where the native ``shared_kv_states`` attention argument was
    backported, through 5.x latest) pass the shared KV cache-independently, fixing the
    cache-free path and making this patch unnecessary, so we no-op there. Detection is by
    signature, not version, so it tracks the backport correctly (forwards-compatible)."""
    try:
        import inspect
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention
        return "shared_kv_states" not in inspect.signature(Gemma4TextAttention.forward).parameters
    except Exception:
        # Default to applying: no-op for models without KV sharing, safe where needed.
        return True


class _Gemma4SharedKVCarrier:
    """Carries Gemma-4 cross-layer shared K/V WITHIN a single forward, decoupled from
    ``past_key_values`` so the store/retrieve survives both use_cache=False and
    gradient checkpointing (GradientCheckpointingLayer nulls the past_key_values kwarg
    before the attention runs, see transformers/modeling_layers.py). Only
    ``shared_layers`` is used; update() is a no-op so non-shared layers never build an
    autoregressive cache. Mirrors transformers' own later ``shared_kv_states`` fix."""
    __slots__ = ("shared_layers",)
    def __init__(self):
        self.shared_layers = {}
    def update(self, key_states, value_states, layer_idx, cache_kwargs = None):
        return key_states, value_states
    def get_seq_length(self, layer_idx = 0):
        return 0


def _make_gemma4_attention_carrier_forward(_orig_attn_forward):
    """Feed the per-forward carrier to Gemma4TextAttention whenever the real
    ``past_key_values`` is None (use_cache=False, or nulled by gradient
    checkpointing), so the cross-layer KV store/retrieve still works. The carrier is
    attached to the module by the model-forward patch below. Pass-through otherwise
    (e.g. generation with a real cache), so this is a strict no-op when not needed."""
    def forward(self, *args, **kwargs):
        carrier = getattr(self, "_unsloth_shared_kv_carrier", None)
        if carrier is not None:
            # past_key_values is the 4th positional arg (after self) or a kwarg.
            if "past_key_values" in kwargs:
                if kwargs["past_key_values"] is None:
                    kwargs["past_key_values"] = carrier
            elif len(args) >= 4:
                if args[3] is None:
                    args = args[:3] + (carrier,) + args[4:]
            else:
                kwargs["past_key_values"] = carrier
        return _orig_attn_forward(self, *args, **kwargs)
    forward._unsloth_gemma4_carrier_patched = True
    forward.__qualname__ = getattr(_orig_attn_forward, "__qualname__", "forward")
    forward.__doc__ = getattr(_orig_attn_forward, "__doc__", None)
    return forward


def _patch_gemma4_attention_carrier():
    """Patch Gemma4TextAttention.forward to accept the shared-KV carrier. Idempotent."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention
    except Exception:
        return
    if getattr(Gemma4TextAttention.forward, "_unsloth_gemma4_carrier_patched", False):
        return
    Gemma4TextAttention.forward = _make_gemma4_attention_carrier_forward(Gemma4TextAttention.forward)


def _gemma4_force_nonreentrant_checkpointing(model):
    """Force gemma-4 text decoder layers to use NON-reentrant gradient checkpointing
    while the carrier is active, so the cross-layer shared K/V stays grad-connected.

    The shared K/V is produced by an earlier ("producer") layer and reused by the ~20
    later ("consumer") layers via the carrier. Under REENTRANT checkpointing -- which
    includes Unsloth's offloaded checkpointer (it forces use_reentrant=True) and torch's
    use_reentrant=True -- each layer is recomputed independently during backward in
    REVERSE order, so a consumer is recomputed before its producer and reads the detached
    no-grad K/V from the first forward; gradients from the shared layers then never reach
    the producer's k_proj/v_proj (the logits/loss look correct, only the gradient is
    wrong). Un-checkpointing the producer instead triggers "backward through the graph a
    second time" because its activation feeds many reentrant-checkpointed consumers.
    Non-reentrant recomputation keeps the whole region's autograd graph connected, so the
    producer K/V gradient is exact (verified: producer k/v_proj grads match the
    use_cache=True reference to cosine 1.0). Scoped to gemma-4 E-series on the legacy
    transformers window (5.5.0-5.5.1) where the carrier runs; trades Unsloth's CPU
    activation offload (a memory optimisation) for correct gradients on these ~1-2B
    models. No-op when gradient checkpointing is off (e.g. plain use_cache=False), and
    moot from transformers 5.5.2 where the native shared_kv_states path makes the whole
    patch a no-op."""
    import functools
    try:
        import torch.utils.checkpoint as _ckpt
    except Exception:
        return
    func = getattr(model, "_unsloth_gemma4_nonreentrant_func", None)
    if func is None:
        # Resolve the PRISTINE torch checkpoint. Unsloth's smart GC globally swaps
        # torch.utils.checkpoint.checkpoint for `unsloth_checkpoint`, which hard-forces
        # use_reentrant=True (for its CPU-offload CheckpointFunction) and would ignore the
        # use_reentrant=False below. Unsloth saves the original at `_old_checkpoint`, so
        # use that when the current one is an Unsloth shim.
        base = getattr(_ckpt, "checkpoint", None)
        if getattr(base, "__name__", "") in (
            "unsloth_checkpoint", "unsloth_gradient_checkpoint",
            "unsloth_offloaded_gradient_checkpoint",
        ):
            base = getattr(_ckpt, "_old_checkpoint", base)
        if base is None:
            return
        func = functools.partial(base, use_reentrant = False)
        try:
            model._unsloth_gemma4_nonreentrant_func = func
        except Exception:
            pass
    layers = getattr(model, "_unsloth_gemma4_decoder_layers", None)
    if layers is None:
        layers = [
            m for m in model.modules()
            if hasattr(m, "gradient_checkpointing")
            and type(getattr(m, "self_attn", None)).__name__ == "Gemma4TextAttention"
        ]
        try:
            model._unsloth_gemma4_decoder_layers = layers
        except Exception:
            pass
    for layer in layers:
        try:
            # Only override when this layer is actually being checkpointed.
            if getattr(layer, "gradient_checkpointing", False):
                layer._gradient_checkpointing_func = func
        except Exception:
            pass


def _gemma4_attach_shared_kv_carrier(model):
    """Create a fresh carrier and attach it to every Gemma4TextAttention under ``model``
    so shared layers can reach it even after gradient checkpointing nulls their
    past_key_values kwarg, and force non-reentrant checkpointing so the shared K/V stays
    grad-connected. The attention list is cached on the model; called once per forward.
    No-op without KV sharing."""
    attns = getattr(model, "_unsloth_gemma4_attns", None)
    if attns is None:
        attns = [m for m in model.modules() if m.__class__.__name__ == "Gemma4TextAttention"]
        # Only relevant when some layer actually shares KV (E2B/E4B); cache an empty
        # list for 31B / 26B-A4B / non-shared models so later forwards are a no-op.
        if not any(getattr(m, "is_kv_shared_layer", False) for m in attns):
            attns = []
        try:
            model._unsloth_gemma4_attns = attns
        except Exception:
            pass
    if not attns:
        return
    carrier = _Gemma4SharedKVCarrier()
    for attn in attns:
        try:
            attn._unsloth_shared_kv_carrier = carrier
        except Exception:
            pass
    _gemma4_force_nonreentrant_checkpointing(model)


def _make_kv_shared_use_cache_false_safe_forward(_orig_forward):
    """Wrap a Gemma-4 model forward to attach a fresh per-forward shared-KV carrier to
    every Gemma4TextAttention, so cross-layer KV sharing works under use_cache=False
    AND gradient checkpointing. The carrier (consumed by the attention patch) replaces
    the older approach of injecting a DynamicCache + forcing use_cache=True, which the
    GradientCheckpointingLayer defeated by nulling past_key_values at the attention."""
    def forward(self, *args, **kwargs):
        try:
            _gemma4_attach_shared_kv_carrier(self)
        except Exception:
            pass
        return _orig_forward(self, *args, **kwargs)

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
    """Patch Gemma4TextModel.forward + Gemma4TextAttention.forward so cross-layer KV
    sharing works under use_cache=False and gradient checkpointing."""
    if not _gemma4_kv_sharing_needs_cache():
        # transformers passes the shared KV via `shared_kv_states` (cache-independent),
        # so the cache-free path already works; this patch would be needless.
        return
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
    except ImportError:
        return
    except Exception as e:
        return raise_error("Gemma4TextModel.forward use_cache=False fix", e)
    try:
        _patch_gemma4_attention_carrier()
        _patch_forward_for_kv_shared_no_cache(Gemma4TextModel)
    except Exception as e:
        return raise_error("Gemma4TextModel.forward use_cache=False fix", e)
pass
TEMPORARY_PATCHES.append(patch_Gemma4TextModel_forward_kv_shared_no_cache)


def patch_Gemma4Model_forward_kv_shared_no_cache():
    """Patch Gemma4Model.forward (multimodal parent) to attach the carrier too, in
    case a refactor moves cache auto-creation above Gemma4TextModel."""
    if not _gemma4_kv_sharing_needs_cache():
        return
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4Model
    except ImportError:
        return
    except Exception as e:
        return raise_error("Gemma4Model.forward use_cache=False fix", e)
    try:
        _patch_gemma4_attention_carrier()
        _patch_forward_for_kv_shared_no_cache(Gemma4Model)
    except Exception as e:
        return raise_error("Gemma4Model.forward use_cache=False fix", e)
pass
TEMPORARY_PATCHES.append(patch_Gemma4Model_forward_kv_shared_no_cache)


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
