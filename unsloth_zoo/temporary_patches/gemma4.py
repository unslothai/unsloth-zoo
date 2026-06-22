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
# Fix 2: Gemma-4 E2B/E4B cross-layer KV sharing (num_kv_shared_layers > 0).
#
# Gemma4TextAttention reuses stored K/V only via the cache, but with use_cache=False
# the model never builds one, so is_kv_shared_layer layers recompute K/V from the wrong
# hidden states (loss explodes). See huggingface/transformers#45242. Only the buggy
# cache-dependent path in transformers 5.5.0-5.5.1 needs this; 5.5.2+ passes native
# shared_kv_states, so the gate below no-ops there (detected by signature, not version).
#
# Fix: attach a per-forward carrier (a shared_layers dict) to every attention and use it
# when past_key_values is None, so store/retrieve survives use_cache=False and GC (which
# nulls the past_key_values kwarg). Plus force non-reentrant checkpointing to keep the
# shared K/V grad-connected (see _gemma4_force_nonreentrant_checkpointing).
# ============================================================================

def _gemma4_kv_sharing_needs_cache():
    """True iff this build routes Gemma-4 KV sharing through past_key_values (the buggy
    5.5.0-5.5.1 path). 5.5.2+ passes native shared_kv_states so we no-op there; detected
    by signature, not version, to track the mid-release backport."""
    try:
        import inspect
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention
        return "shared_kv_states" not in inspect.signature(Gemma4TextAttention.forward).parameters
    except Exception:
        return True  # apply by default: no-op without KV sharing, safe where needed


class _Gemma4SharedKVCarrier:
    """Carries cross-layer shared K/V within one forward, decoupled from past_key_values
    so store/retrieve survives use_cache=False and GC. update() is a no-op so non-shared
    layers never build a cache."""
    __slots__ = ("shared_layers",)
    def __init__(self):
        self.shared_layers = {}
    def update(self, key_states, value_states, layer_idx, cache_kwargs = None):
        return key_states, value_states
    def get_seq_length(self, layer_idx = 0):
        return 0


def _make_gemma4_attention_carrier_forward(_orig_attn_forward):
    """Feed the carrier to attention as past_key_values when the real one is None
    (use_cache=False or nulled by GC). Pass-through otherwise -> no-op when unused."""
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


# Unsloth's checkpoint shims (all force/route reentrant); unwrap them to find pristine torch.
_UNSLOTH_CKPT_SHIM_NAMES = frozenset({
    "unsloth_checkpoint", "unsloth_gradient_checkpoint",
    "unsloth_offloaded_gradient_checkpoint",
})
# Capture pristine torch.utils.checkpoint at import, before any Unsloth GC patch swaps it,
# so forcing non-reentrant survives even stacked shims (_old_checkpoint then being a shim).
try:
    import torch.utils.checkpoint as _ckpt_at_import
    _PRISTINE_TORCH_CHECKPOINT = (
        _ckpt_at_import.checkpoint
        if getattr(_ckpt_at_import.checkpoint, "__name__", "") not in _UNSLOTH_CKPT_SHIM_NAMES
        else None
    )
except Exception:
    _PRISTINE_TORCH_CHECKPOINT = None


def _resolve_pristine_checkpoint(_ckpt):
    """Genuine non-reentrant-capable torch checkpoint: prefer the import-captured ref, then the
    set-once ref the GC patcher stashes before any shim stacks, else unwrap _old_checkpoint past
    stacked Unsloth shims. Returns None if every candidate is a shim."""
    if _PRISTINE_TORCH_CHECKPOINT is not None:
        return _PRISTINE_TORCH_CHECKPOINT
    # patch_unsloth_*_gradient_checkpointing stash the real fn here on the first patch, so it
    # survives stacked shims and a late gemma4 import (when module-level _old_checkpoint is a shim).
    captured = getattr(_ckpt, "_unsloth_pristine_checkpoint", None)
    if captured is not None and getattr(captured, "__name__", "") not in _UNSLOTH_CKPT_SHIM_NAMES:
        return captured
    cand = getattr(_ckpt, "checkpoint", None)
    seen = set()
    while getattr(cand, "__name__", "") in _UNSLOTH_CKPT_SHIM_NAMES:
        # Prefer the shim's own _old_checkpoint (survives stacking, where a single module-level
        # attr is overwritten), then fall back to the module-level one.
        nxt = getattr(cand, "_old_checkpoint", None)
        if nxt is None:
            nxt = getattr(_ckpt, "_old_checkpoint", None)
        if nxt is None or id(nxt) in seen or nxt is cand:
            break
        seen.add(id(nxt))
        cand = nxt
    # Never hand back a shim: forcing use_reentrant=False onto it would wrap the very offloaded
    # checkpointer we are trying to bypass, silently dropping the gradient fix. Signal "none".
    if getattr(cand, "__name__", "") in _UNSLOTH_CKPT_SHIM_NAMES:
        return None
    return cand


def _gemma4_model_has_kv_sharing(model):
    """True iff model has Gemma-4 E-series KV sharing (E2B/E4B). Cached; no-op signal otherwise."""
    flag = getattr(model, "_unsloth_gemma4_has_kv_sharing", None)
    if flag is None:
        flag = any(
            getattr(m, "is_kv_shared_layer", False)
            for m in model.modules() if m.__class__.__name__ == "Gemma4TextAttention"
        )
        try:
            model._unsloth_gemma4_has_kv_sharing = flag
        except Exception:
            pass
    return flag


def _gemma4_force_nonreentrant_checkpointing(model):
    """Force gemma-4 E-series decoder layers onto NON-reentrant gradient checkpointing so the
    cross-layer shared K/V stays grad-connected, on EVERY version with KV sharing. The producer
    layer's K/V is reused by ~20 consumers via a dict on a module attr / kwarg (the carrier on
    5.5.0-5.5.1, native shared_kv_states on 5.5.2+), not a checkpoint input. Under reentrant GC
    (incl. Unsloth's offloaded checkpointer) consumers recompute before the producer in reverse
    order and read detached no-grad K/V, so the producer's k/v_proj grads are wrong (loss looks
    fine). Non-reentrant keeps the region's graph connected -> exact grads (cosine 1.0 vs the
    use_cache=True reference, verified on 5.5.0 and 5.12.1). Trades Unsloth's CPU activation
    offload for correctness on these ~1-2B models. No-op without GC or KV sharing."""
    if not _gemma4_model_has_kv_sharing(model):
        return
    import functools
    try:
        import torch.utils.checkpoint as _ckpt
    except Exception:
        return
    # Resolve past Unsloth's smart-GC shim (it hard-forces use_reentrant=True and would
    # ignore the use_reentrant=False below).
    base = _resolve_pristine_checkpoint(_ckpt)
    if base is None:
        return
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
            if not getattr(layer, "gradient_checkpointing", False):
                continue
            # Preserve the layer's existing checkpoint kwargs (preserve_rng_state, context_fn,
            # ...); only override use_reentrant.
            existing = getattr(layer, "_gradient_checkpointing_func", None)
            extra = {}
            if isinstance(existing, functools.partial):
                extra = {k: v for k, v in (existing.keywords or {}).items()
                         if k != "use_reentrant"}
            layer._gradient_checkpointing_func = functools.partial(
                base, use_reentrant = False, **extra
            )
        except Exception:
            pass


def _gemma4_attach_shared_kv_carrier(model):
    """Attach a fresh per-forward carrier to every Gemma4TextAttention so shared layers reach
    it after GC nulls their past_key_values kwarg. Attention list cached; no-op without sharing."""
    attns = getattr(model, "_unsloth_gemma4_attns", None)
    if attns is None:
        attns = [m for m in model.modules() if m.__class__.__name__ == "Gemma4TextAttention"]
        # Cache an empty list for non-shared models (31B/26B-A4B) so later forwards no-op.
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


def _gemma4_clear_shared_kv_carrier(model):
    """Drop the producer K/V the carrier pinned. Only safe when no checkpointed backward will
    read it (not torch.is_grad_enabled()); no-op without KV sharing / without a carrier."""
    attns = getattr(model, "_unsloth_gemma4_attns", None)
    if not attns:
        return
    for attn in attns:
        carrier = getattr(attn, "_unsloth_shared_kv_carrier", None)
        if carrier is not None:
            try:
                carrier.shared_layers.clear()
            except Exception:
                pass
            try:
                del attn._unsloth_shared_kv_carrier
            except Exception:
                pass


def _gemma4_carrier_overlap_unsafe(model):
    """Overlapping checkpointed forwards corrupt grads through the module-scoped carrier ONLY when
    the model truly shares K/V (so a carrier is attached) AND gradient checkpointing is active on a
    Gemma-4 layer (so backward recompute re-enters attention and re-reads the carrier). Without
    sharing no carrier exists (31B/26B-A4B cache an empty attn list); without checkpointing backward
    uses saved activations and never re-reads it, so two forwards before one backward are safe."""
    if not _gemma4_model_has_kv_sharing(model):
        return False
    layers = getattr(model, "_unsloth_gemma4_decoder_layers", None)
    if layers is None:
        layers = [
            m for m in model.modules()
            if hasattr(m, "gradient_checkpointing")
            and type(getattr(m, "self_attn", None)).__name__ == "Gemma4TextAttention"
        ]
    return any(getattr(layer, "gradient_checkpointing", False) for layer in layers)


def _arm_carrier_outstanding_marker(model, output):
    """Mark the carrier outstanding until this forward's backward starts. A grad hook on the
    output clears it at the start of this graph's backward, so a later forward can tell whether
    a prior checkpointed graph is still pending (DPO/contrastive) vs already consumed (grad
    accumulation). No-op if nothing in the output carries grad (no backward graph to guard)."""
    tensor = getattr(output, "last_hidden_state", None)
    if not isinstance(tensor, torch.Tensor):
        if isinstance(output, torch.Tensor):
            tensor = output
        elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            tensor = output[0]
        else:
            tensor = None
    if tensor is None or not tensor.requires_grad or tensor.grad_fn is None:
        return
    try:
        model._unsloth_gemma4_carrier_outstanding = True
    except Exception:
        return
    def _clear(_grad):
        try: model._unsloth_gemma4_carrier_outstanding = False
        except Exception: pass
        return None
    try:
        tensor.register_hook(_clear)
    except Exception:
        # could not arm the auto-clear -> do not leave a sticky marker that false-positives later
        try: model._unsloth_gemma4_carrier_outstanding = False
        except Exception: pass


def _make_kv_shared_use_cache_false_safe_forward(_orig_forward, attach_carrier):
    """Wrap a Gemma-4 model forward: always force non-reentrant GC (the gradient fix, all
    versions), and on the 5.5.0-5.5.1 carrier window (attach_carrier=True) also attach a fresh
    per-forward carrier so the cache-free forward works. 5.5.2+ uses native shared_kv_states.

    Carrier scope (5.5.0-5.5.1 only): the carrier lives on the module because that is the only
    channel surviving GC backward recompute (recompute re-runs the layers, not this wrapper).
    Two checkpointed forward graphs alive before one backward (DPO/contrastive, summed microbatch
    losses) would have the second overwrite a carrier the first graph still needs -> we detect
    that and raise instead of silently corrupting grads (upgrade to >= 5.5.2 for function-scoped
    shared_kv_states). Single-forward / grad-accumulation is fine (the prior graph's backward
    clears the marker before the next forward)."""
    def forward(self, *args, **kwargs):
        try:
            _gemma4_force_nonreentrant_checkpointing(self)
        except Exception:
            pass
        if attach_carrier:
            # No backward to read the carrier under eval/no_grad: release its pinned K/V now
            # instead of holding it until the next forward.
            if not torch.is_grad_enabled():
                try:
                    _gemma4_attach_shared_kv_carrier(self)
                except Exception:
                    pass
                try:
                    return _orig_forward(self, *args, **kwargs)
                finally:
                    try:
                        _gemma4_clear_shared_kv_carrier(self)
                    except Exception:
                        pass
            # Training: a second outstanding checkpointed forward would clobber the first graph's
            # carrier during recompute -- but only when KV sharing AND gradient checkpointing are
            # both active (otherwise no carrier is re-read in backward). Only then track/reject, so
            # 31B/26B-A4B and checkpointing-off runs keep working with DPO/contrastive/summed loss.
            overlap_unsafe = _gemma4_carrier_overlap_unsafe(self)
            if overlap_unsafe and getattr(self, "_unsloth_gemma4_carrier_outstanding", False):
                raise RuntimeError(
                    "Unsloth: Gemma-4 E-series KV sharing on transformers 5.5.0/5.5.1 cannot run "
                    "two forward passes before a single backward (e.g. DPO/contrastive or summed "
                    "microbatch losses) while gradient checkpointing is on: the shared-KV carrier "
                    "is module-scoped, so the second forward would corrupt the first graph's "
                    "gradients. Upgrade to transformers >= 5.5.2 (function-scoped shared K/V), or "
                    "disable gradient checkpointing for these objectives."
                )
            try:
                _gemma4_attach_shared_kv_carrier(self)
            except Exception:
                pass
            out = _orig_forward(self, *args, **kwargs)
            if overlap_unsafe:
                _arm_carrier_outstanding_marker(self, out)
            return out
        return _orig_forward(self, *args, **kwargs)

    forward._unsloth_kv_shared_use_cache_false_patched = True
    forward.__qualname__ = getattr(_orig_forward, "__qualname__", "forward")
    forward.__doc__ = getattr(_orig_forward, "__doc__", None)
    return forward


def _patch_forward_for_kv_shared_no_cache(cls, attach_carrier):
    """Install the wrapper on `cls.forward`. Idempotent."""
    if getattr(cls.forward, "_unsloth_kv_shared_use_cache_false_patched", False):
        return
    cls.forward = _make_kv_shared_use_cache_false_safe_forward(cls.forward, attach_carrier)


def patch_Gemma4TextModel_forward_kv_shared_no_cache():
    """Patch Gemma4TextModel.forward: gradient-correct KV sharing under reentrant/offloaded GC
    (all versions), plus the carrier on the 5.5.0-5.5.1 window where use_cache=False is broken."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
    except ImportError:
        return
    except Exception as e:
        return raise_error("Gemma4TextModel.forward kv-shared GC fix", e)
    try:
        needs_carrier = _gemma4_kv_sharing_needs_cache()
        if needs_carrier:
            _patch_gemma4_attention_carrier()
        _patch_forward_for_kv_shared_no_cache(Gemma4TextModel, attach_carrier = needs_carrier)
    except Exception as e:
        return raise_error("Gemma4TextModel.forward kv-shared GC fix", e)
pass
TEMPORARY_PATCHES.append(patch_Gemma4TextModel_forward_kv_shared_no_cache)


def patch_Gemma4Model_forward_kv_shared_no_cache():
    """Same as the TextModel patch, on the multimodal parent Gemma4Model (if the forward routes through it)."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4Model
    except ImportError:
        return
    except Exception as e:
        return raise_error("Gemma4Model.forward kv-shared GC fix", e)
    try:
        needs_carrier = _gemma4_kv_sharing_needs_cache()
        if needs_carrier:
            _patch_gemma4_attention_carrier()
        _patch_forward_for_kv_shared_no_cache(Gemma4Model, attach_carrier = needs_carrier)
    except Exception as e:
        return raise_error("Gemma4Model.forward kv-shared GC fix", e)
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
