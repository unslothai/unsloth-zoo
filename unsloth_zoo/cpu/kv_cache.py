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

"""Growable full-attention KV layers for CPU / MPS generate (GRPO / online DPO rollouts and plain model.generate).

transformers' DynamicLayer appends each token with torch.cat, which copies the whole K and V of every layer on every
decode step: O(L) bytes per step and O(L^2) per rollout, and on CPU each fresh buffer is also page-faulted in. This
layer keeps a buffer with a chunk of spare slots, writes new tokens in place and hands attention a `[:, :, :n]` view,
so attention still reads exactly n positions (unlike a full-length StaticCache, which attends over the whole limit and
was ~3x slower on CPU). Growth is linear in chunks (UNSLOTH_GROWABLE_KV_MIN, 256 tokens) rather than Unsloth's GPU
power-of-two buckets, because spare slots here cost RAM only: at most two chunks per layer above what is written.

Only generate's own default DynamicCache is converted, only when every layer is a plain full-attention DynamicLayer
(sliding-window, chunked and linear-attention / hybrid caches keep their classes), only for greedy and sampling
decode (beam search reorders the whole cache every step anyway), and UNSLOTH_GROWABLE_KV=0 turns it off."""

import functools
import inspect
import os
import types

import torch

try:
    from transformers.cache_utils import DynamicCache as _DynamicCache, DynamicLayer as _DynamicLayer
except Exception:  # transformers before layered caches (< 4.56): keep the stock cache
    _DynamicCache = _DynamicLayer = None

__all__ = [
    "GrowableDynamicLayer",
    "enabled",
    "make_growable",
    "install",
    "ENGAGED",
]

# Caches converted so far (tests and run logs read it to prove the path ran).
ENGAGED = [0]

_DEVICES = ("cpu", "mps")
_MODES = ("greedy_search", "sample")
# generate writes its cache under one of these names (past_key_values up to 5.2x, cache after the rename).
_CACHE_KEYS = ("past_key_values", "cache")


def enabled():
    # Same switch as Unsloth's GPU growable cache: UNSLOTH_GROWABLE_KV=0 keeps the stock DynamicCache.
    return _DynamicLayer is not None and os.environ.get("UNSLOTH_GROWABLE_KV", "1") != "0"


def _chunk():
    try:
        return max(1, int(os.environ.get("UNSLOTH_GROWABLE_KV_MIN", "256")))
    except ValueError:
        return 256


if _DynamicLayer is not None:

    class GrowableDynamicLayer(_DynamicLayer):
        """A DynamicLayer whose keys / values are views of a preallocated buffer that grows in chunks.

        Anything else that reassigns `keys` / `values` (crop, batch_select_indices, batch_repeat_interleave,
        reorder_cache, reset, offload) is adopted on the next update as a full buffer: the next write then
        reallocates, so a tensor this layer did not allocate is never written in place."""

        def _grow_buffers(self):
            keys = getattr(self, "keys", None)
            if keys is not None and keys is getattr(self, "_grow_kview", None):
                return self._grow_k, self._grow_v, keys.shape[-2]
            # Not our view: adopt what is there (capacity == length, so the next write grows).
            values = getattr(self, "values", None)
            if keys is None or values is None or keys.dim() != 4 or values.dim() != 4:
                return None, None, 0
            return keys, values, keys.shape[-2]

        def update(self, key_states, value_states, *args, **kwargs):
            if (
                not self.is_initialized
                or torch.compiler.is_compiling()
                or key_states.dim() != 4
                or value_states.dim() != 4
            ):
                # First call (lazy init + prefill) and traced code take the stock path; the result is adopted next.
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
            ):
                # Shapes or dtypes torch.cat would broadcast / promote: keep its exact semantics.
                self._grow_kview = None
                return super().update(key_states, value_states, *args, **kwargs)
            t = key_states.shape[-2]
            need = n + t
            # A cache kept from a generate under inference_mode and reused outside it cannot be written in place.
            stale = k_buf is not None and k_buf.is_inference() and not torch.is_inference_mode_enabled()
            if k_buf is None or need > k_buf.shape[-2] or stale:
                chunk = _chunk()
                capacity = ((need + chunk - 1) // chunk + 1) * chunk
                new_k = key_states.new_empty((*key_states.shape[:-2], capacity, key_states.shape[-1]))
                new_v = value_states.new_empty((*value_states.shape[:-2], capacity, value_states.shape[-1]))
                if n:  # copy_ out of an inference tensor is allowed; writes into one are not
                    new_k[:, :, :n].copy_(k_buf[:, :, :n])
                    new_v[:, :, :n].copy_(v_buf[:, :, :n])
                k_buf, v_buf = new_k, new_v
                self._grow_k, self._grow_v = k_buf, v_buf
            k_buf[:, :, n:need].copy_(key_states)
            v_buf[:, :, n:need].copy_(value_states)
            self.keys = self._grow_kview = k_buf[:, :, :need]
            self.values = v_buf[:, :, :need]
            return self.keys, self.values

else:  # pragma: no cover
    GrowableDynamicLayer = None


def make_growable(cache):
    """Converts a fresh stock DynamicCache whose layers are all plain full-attention DynamicLayers, in place.
    Returns whether it did. Subclasses, offloaded caches, sliding / chunked / linear-attention layers and caches
    that already hold tokens are left alone."""
    if not enabled() or type(cache) is not _DynamicCache or getattr(cache, "offloading", False):
        return False
    layers = getattr(cache, "layers", None)
    if layers is None:
        return False
    if len(layers) == 0:
        # No config: layers are appended lazily from layer_class_to_replicate.
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
        if not enabled():
            return prepare(self, *args, **kwargs)
        try:
            arguments = signature.bind(self, *args, **kwargs).arguments
            model_kwargs = arguments["model_kwargs"]
            before = {key: model_kwargs.get(key) for key in _CACHE_KEYS}
        except Exception:
            return prepare(self, *args, **kwargs)
        result = prepare(self, *args, **kwargs)
        try:
            device = getattr(self, "device", None)
            config = arguments.get("generation_config", None)
            if (
                getattr(device, "type", None) in _DEVICES
                and _mode_name(arguments["generation_mode"]) in _MODES
                and not getattr(config, "is_assistant", False)
            ):
                for key in _CACHE_KEYS:
                    cache = model_kwargs.get(key)
                    # Only a cache generate created just now, never a caller's.
                    if cache is not None and cache is not before[key]:
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
    if func is None:
        return False
    if getattr(func, "_unsloth_growable_kv", False):
        return True
    wrapped = _growable_prepare(func)
    if wrapped is None:
        return False
    # Bound per instance (MethodType), so deepcopy rebinds it to the copy, like the generate wrapper.
    model._prepare_cache_for_generation = types.MethodType(wrapped, model)
    return True
