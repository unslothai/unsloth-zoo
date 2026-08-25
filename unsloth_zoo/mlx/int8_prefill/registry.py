# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""The allow-list of weights the int8 path may touch.

`mx.quantized_matmul` is a shared op: the same call serves MLP projections, tied
lm_heads, MLA multi-linears, and -- with a quantized KV cache -- attention itself, where
the "weight" is a freshly built per-step cache tensor. Rather than trying to tell these
apart from inside the hot path with shape heuristics, `warmup()` walks the model once,
decides per module (where the module *path* is known, which the op never sees), and
records the weights it approves.

That makes the hot path a dict lookup plus an identity check, and gives four properties
the reference implementation's per-call heuristics do not have:

  * `mx.eval` happens only here, so the hot path emits pure graph ops and survives
    `mx.compile`, which raises on eval inside a trace.
  * A registered shape has already been proven to evaluate on this device, so lazy
    evaluation cannot surface a kernel failure far from any try/except.
  * Scope policy ("MLP only") is a registration decision, not a shape guess.
  * Entries hold a strong reference to the weight array, so `id()` cannot be recycled
    underneath us. The reference's `_ws_cache[id(m)]` keeps no reference and will hand
    back another layer's per-channel scales after a GC.
"""

import logging
import threading

import mlx.core as mx

from .eligibility import is_eligible
from .scales import channel_scale

logger = logging.getLogger(__name__)


class Entry:
    """An approved weight and everything the int8 path needs for it.

    `expert_dim` is None for a dense 2-D weight and the expert count for a stacked
    `[E, N, K]` MoE weight. Nothing populates it yet -- `mx.gather_qmm` is phase 2 -- but
    carrying it now keeps the requant kernel indexing over `E*N` rows from day one, so
    MoE support is a GEMM rather than a rework.
    """

    __slots__ = ("w", "scales", "biases", "ws", "bits", "group_size", "n", "k",
                 "name", "expert_dim", "fn")

    def __init__(self, w, scales, biases, ws, bits, group_size, n, k, name,
                 expert_dim=None):
        # Lazily filled by patch.py with the differentiable wrapper for this weight.
        self.fn = None
        self.w = w
        self.scales = scales
        self.biases = biases
        self.ws = ws
        self.bits = bits
        self.group_size = group_size
        self.n = n
        self.k = k
        self.name = name
        self.expert_dim = expert_dim


_entries = {}
_lock = threading.Lock()


def get(weight):
    """Entry for `weight`, or None. The identity re-check is the point: an `id()` that
    survives its array would otherwise silently select another layer's scales."""
    e = _entries.get(id(weight))
    if e is not None and e.w is weight:
        return e
    return None


def clear():
    with _lock:
        _entries.clear()


def size():
    return len(_entries)


def names():
    return sorted(e.name for e in _entries.values())


def _dims_of(module):
    """(n, k) in elements. The packed last dim is words, not values."""
    n, packed_k = module["weight"].shape[-2:]
    return n, packed_k * 32 // module.bits


def register_module(module, name, exact_scales=None):
    """Approve one `nn.QuantizedLinear`-like module. Returns (ok, reason)."""
    if "weight" not in module or "scales" not in module:
        return False, "not a quantized module"
    biases = module.get("biases") if hasattr(module, "get") else None
    bits = getattr(module, "bits", None)
    group_size = getattr(module, "group_size", None)
    mode = getattr(module, "mode", "affine")
    if bits is None or group_size is None:
        return False, "missing bits/group_size"

    n, k = _dims_of(module)
    ok, why = is_eligible(n, k, bits, group_size, mode, has_biases=biases is not None)
    if not ok:
        return False, why

    w = module["weight"]
    ws = channel_scale(w, module["scales"], biases, bits, group_size, exact=exact_scales)
    # The one eval in the whole module. Materializing here is what keeps the hot path
    # trace-safe, and it also means a broken scale computation fails during warmup
    # rather than mid-generation.
    mx.eval(ws)

    with _lock:
        _entries[id(w)] = Entry(
            w, module["scales"], biases, ws, bits, group_size, n, k, name
        )
    return True, None


def warmup(model, scope="all", exact_scales=None):
    """Walk `model` and register every eligible quantized projection.

    `scope="mlp"` restricts to the feed-forward projections by module name, the
    conservative choice if a quality eval implicates attention. This is a name-based
    decision, which is only possible here -- the op-level hot path has no idea which
    module it is serving.

    Returns (registered, skipped).
    """
    import mlx.nn as nn

    registered, skipped = 0, 0
    for name, module in model.named_modules():
        if not isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            continue
        if scope == "mlp" and not any(
            part in name for part in ("mlp", "feed_forward", "ffn")
        ):
            skipped += 1
            continue
        ok, why = register_module(module, name, exact_scales=exact_scales)
        if ok:
            registered += 1
        else:
            skipped += 1
            logger.debug("Unsloth: MLX int8 skipping %s (%s)", name, why)

    logger.info(
        "Unsloth: MLX int8 prefill registered %d module(s), skipped %d (scope=%s)",
        registered, skipped, scope,
    )
    return registered, skipped
