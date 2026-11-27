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

"""CPU / MPS generate: full-attention KV written in place into a chunk-grown buffer, returned as `[:, :, :n]` views.

DynamicLayer's per-token torch.cat copies every layer's whole KV each step (O(L^2) per rollout). A full-length
StaticCache is not the fix: attention then reads the whole limit (~3x slower on CPU). Growth is linear in chunks, not
the GPU cache's power-of-two buckets, since spare slots here cost RAM only. UNSLOTH_GROWABLE_KV=0 turns it off."""

import functools
import inspect
import os
import types

import torch

try:
    from transformers.cache_utils import DynamicCache as _DynamicCache, DynamicLayer as _DynamicLayer
    # 4.56's layers have no is_initialized (keys is None until the first update): keep the stock cache there.
    if not hasattr(_DynamicLayer(), "is_initialized"):
        raise ImportError
except Exception:
    _DynamicCache = _DynamicLayer = None

__all__ = [
    "GrowableDynamicLayer",
    "enabled",
    "make_growable",
    "install",
    "ENGAGED",
]

ENGAGED = [0]

_DEVICES = ("cpu", "mps")
_MODES = ("greedy_search", "sample")
# past_key_values up to transformers 5.2x, cache after the rename.
_CACHE_KEYS = ("past_key_values", "cache")


def enabled():
    return _DynamicLayer is not None and os.environ.get("UNSLOTH_GROWABLE_KV", "1") != "0"


def _chunk():
    try:
        return max(1, int(os.environ.get("UNSLOTH_CPU_KV_CHUNK", "256")))
    except ValueError:
        return 256


if _DynamicLayer is not None:

    class GrowableDynamicLayer(_DynamicLayer):
        """keys / values that something else reassigned (crop, batch ops, reorder, reset, offload) are adopted with
        capacity == length, so the next write reallocates: a tensor this layer did not allocate is never written."""

        def _forget(self):
            self._grow_k = self._grow_v = self._grow_kview = self._grow_vview = None

        def _grow_buffers(self):
            keys = getattr(self, "keys", None)
            values = getattr(self, "values", None)
            if keys is None or values is None or keys.dim() != 4 or values.dim() != 4:
                return None, None, 0
            if keys is getattr(self, "_grow_kview", None) and values is getattr(self, "_grow_vview", None):
                return self._grow_k, self._grow_v, keys.shape[-2]
            return keys, values, keys.shape[-2]

        def update(self, key_states, value_states, *args, **kwargs):
            if (
                not self.is_initialized
                or torch.compiler.is_compiling()
                # Grad recording: a later in-place append would bump the version of K / V saved for backward.
                or torch.is_grad_enabled()
                or not enabled()
                or key_states.dim() != 4
                or value_states.dim() != 4
            ):
                # Prefill and traced code take the stock path; its result is adopted next.
                self._forget()
                return super().update(key_states, value_states, *args, **kwargs)
            k_buf, v_buf, n = self._grow_buffers()
            if k_buf is not None and (
                k_buf.shape[:2] != key_states.shape[:2]
                or k_buf.shape[-1] != key_states.shape[-1]
                or v_buf.shape[:2] != value_states.shape[:2]
                or v_buf.shape[-1] != value_states.shape[-1]
                or k_buf.dtype != key_states.dtype
                or v_buf.dtype != value_states.dtype
                or k_buf.device != key_states.device
                or v_buf.device != value_states.device
            ):
                # Mismatches torch.cat would promote or reject: keep its semantics.
                self._forget()
                return super().update(key_states, value_states, *args, **kwargs)
            t = key_states.shape[-2]
            need = n + t
            # A cache kept from an inference_mode generate and reused outside it is read-only.
            stale = k_buf is not None and k_buf.is_inference() and not torch.is_inference_mode_enabled()
            if k_buf is None or need > k_buf.shape[-2] or stale:
                chunk = _chunk()
                capacity = ((need + chunk - 1) // chunk + 1) * chunk
                # Zeroed, not empty: torch.save of a returned view writes its whole storage.
                new_k = key_states.new_zeros((*key_states.shape[:-2], capacity, key_states.shape[-1]))
                new_v = value_states.new_zeros((*value_states.shape[:-2], capacity, value_states.shape[-1]))
                if n:
                    new_k[:, :, :n].copy_(k_buf[:, :, :n])
                    new_v[:, :, :n].copy_(v_buf[:, :, :n])
                k_buf, v_buf = new_k, new_v
            if t:
                k_buf[:, :, n:need].copy_(key_states)
                v_buf[:, :, n:need].copy_(value_states)
            self._grow_k, self._grow_v = k_buf, v_buf
            self.keys = self._grow_kview = k_buf[:, :, :need]
            self.values = self._grow_vview = v_buf[:, :, :need]
            return self.keys, self.values

        def reset(self, *args, **kwargs):
            out = super().reset(*args, **kwargs)
            self._forget()
            return out

        def batch_select_indices(self, *args, **kwargs):
            out = super().batch_select_indices(*args, **kwargs)
            self._forget()
            return out

        def batch_repeat_interleave(self, *args, **kwargs):
            out = super().batch_repeat_interleave(*args, **kwargs)
            self._forget()
            return out

        def crop(self, *args, **kwargs):
            out = super().crop(*args, **kwargs)
            self._forget()
            return out

else:  # pragma: no cover
    GrowableDynamicLayer = None


def make_growable(cache):
    """In place, only a fresh stock DynamicCache of plain full-attention DynamicLayers (sliding, chunked, hybrid /
    linear-attention, subclassed and offloaded caches keep the stock path). Returns whether it converted."""
    if not enabled() or type(cache) is not _DynamicCache or getattr(cache, "offloading", False):
        return False
    layers = getattr(cache, "layers", None)
    if layers is None:
        return False
    if len(layers) == 0:
        if getattr(cache, "layer_class_to_replicate", None) is not _DynamicLayer:
            return False
        cache.layer_class_to_replicate = GrowableDynamicLayer
    else:
        if any(type(layer) is not _DynamicLayer or getattr(layer, "is_initialized", False) for layer in layers):
            return False
        for layer in layers:
            layer.__class__ = GrowableDynamicLayer
    ENGAGED[0] += 1
    return True


def _mode_name(mode):
    return getattr(mode, "value", mode)


def _growable_prepare(prepare):
    """Wraps a model's _prepare_cache_for_generation: after generate builds its default cache, convert it."""
    try:
        signature = inspect.signature(prepare)
    except (TypeError, ValueError):
        return None
    if "model_kwargs" not in signature.parameters or "generation_mode" not in signature.parameters:
        return None

    @functools.wraps(prepare)
    def wrapper(self, *args, **kwargs):
        if not enabled() or torch.compiler.is_compiling():
            return prepare(self, *args, **kwargs)
        try:
            arguments = signature.bind(self, *args, **kwargs).arguments
            model_kwargs = arguments["model_kwargs"]
            # A caller's cache (incl. an empty legacy tuple generate turns into a new one) is never converted.
            supplied = any(model_kwargs.get(key) is not None for key in _CACHE_KEYS)
        except Exception:
            return prepare(self, *args, **kwargs)
        result = prepare(self, *args, **kwargs)
        try:
            device = getattr(self, "device", None)
            config = arguments.get("generation_config", None)
            if (
                not supplied
                and getattr(device, "type", None) in _DEVICES
                and _mode_name(arguments["generation_mode"]) in _MODES
                and not getattr(config, "is_assistant", False)
                # Dory's last_loop reads the cache layers by exact type.
                and getattr(getattr(self, "config", None), "recurrent_kv_cache_mode", None) is None
            ):
                for key in _CACHE_KEYS:
                    cache = model_kwargs.get(key)
                    if cache is not None:
                        make_growable(cache)
        except Exception:
            pass
        return result

    wrapper._unsloth_growable_kv = True
    return wrapper


def install(model):
    """Binds the growable-cache prepare step on this model instance (idempotent). Returns whether it is bound."""
    if _DynamicLayer is None:
        return False
    prepare = getattr(model, "_prepare_cache_for_generation", None)
    func = getattr(prepare, "__func__", None)
    owner = getattr(prepare, "__self__", None)
    if func is None or owner is None:
        return False
    if getattr(func, "_unsloth_growable_kv", False):
        return True
    wrapped = _growable_prepare(func)
    if wrapped is None:
        return False
    # On the owner (a PeftModel forwards the attribute); MethodType so deepcopy rebinds it.
    owner._prepare_cache_for_generation = types.MethodType(wrapped, owner)
    return True
