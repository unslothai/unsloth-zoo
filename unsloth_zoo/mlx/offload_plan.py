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

"""How much of a model's PLE tables and routed experts to read from disk, priced without MLX."""

import glob
import json
import os
import re
import struct

__all__ = [
    "OFFLOAD_CONTEXT_STEP",
    "OFFLOAD_PLE_OVERHEAD",
    "PREFILL_STEPS",
    "SAFETY_MARGIN",
    "context_bytes",
    "expert_offload_overhead",
    "offload_footprint",
    "offload_inventory",
    "plan_offload",
]

OFFLOAD_CONTEXT_STEP = 256
_PLE_TENSORS = (".embed_tokens_per_layer.", ".ngram_embedding.shards.")
_EXPERT_TENSOR = re.compile(r"\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases|bias|global_scale)$")
_FLOAT_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}
_NESTED_CONFIGS = ("text_config", "thinker_config", "language_config", "llm_config")
_TOP_K = ("num_experts_per_tok", "top_k", "n_activated_experts", "experts_per_token")
# MLX's buffer cache, capped while an offloaded model lives.
_CACHE_CAP = 1 << 30
# The seen-row bitmaps and read buffers of offloaded PLE tables, and the capped cache.
OFFLOAD_PLE_OVERHEAD = (256 << 20) + _CACHE_CAP
PREFILL_STEPS = (2048, 4096, 8192)
# Kept free on top of the estimate: the larger of 2 GiB and 3% of it. The estimate is fitted to
# measured peaks; being slightly off must not run the machine out of memory.
SAFETY_MARGIN = (2 << 30, 0.03)
_TRANSIENT_RESERVE = 2 << 30


def _text_config(model_dir):
    """The first config, breadth first through the nested text configs, that sizes the decoder."""
    with open(os.path.join(model_dir, "config.json")) as stream:
        config = json.load(stream)
    queue = [config]
    while queue:
        current = queue.pop(0)
        if "num_hidden_layers" in current:
            return current
        queue += [current[key] for key in _NESTED_CONFIGS if isinstance(current.get(key), dict)]
    return config.get("text_config") or config


def _tensors(model_dir, dtype_bytes):
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(path, "rb") as stream:
            (length,) = struct.unpack("<Q", stream.read(8))
            header = json.loads(stream.read(length))
        for name, spec in header.items():
            if name != "__metadata__":
                start, end = spec["data_offsets"]
                size = end - start
                width = _FLOAT_BYTES.get(spec["dtype"])
                if dtype_bytes and width:
                    size = size // width * dtype_bytes
                yield name, spec["shape"], size


def offload_inventory(model_dir, dtype_bytes = None):
    """``offload.offloadable``'s sizes plus ``total_bytes``, from the safetensors headers alone (floats
    at ``dtype_bytes`` an element), to price a load before it is made. It counts every quantized
    ``switch_mlp``, so it can count experts a loaded model keeps; the ``mtp.`` head is never counted."""
    total = ple = 0
    layers = {}  # prefix -> [bytes, experts, quantized projections, global scales]
    for name, shape, size in _tensors(model_dir, dtype_bytes):
        total += size
        if name.startswith("mtp."):
            continue
        if any(part in name for part in _PLE_TENSORS):
            ple += size
            continue
        match = _EXPERT_TENSOR.search(name)
        if match is not None:
            layer = layers.setdefault(name[:match.start()], [0, 0, set(), False])
            layer[0] += size
            if name.endswith(".up_proj.weight"):
                layer[1] = shape[0]
            if match.group(2) == "scales":
                layer[2].add(match.group(1))
            layer[3] |= match.group(2) == "global_scale"
    config = _text_config(model_dir)
    top_k = next((config[key] for key in _TOP_K if isinstance(config.get(key), int)), None)
    quantized = [layer[:2] for layer in layers.values() if len(layer[2]) == 3 and layer[1] and not layer[3]]
    experts = sum(size for size, _ in quantized)
    floor = sum(size * min(count, max(8, top_k or 0)) // count for size, count in quantized)
    if not layers:
        reason = "the model has no stacked routed experts"
    elif not quantized:
        reason = (
            "its experts have global scales" if any(layer[3] for layer in layers.values())
            else "its routed experts are not quantized"
        )
    elif top_k is None:
        reason = "its routing top-k is unknown"
    elif floor >= experts:
        reason = f"its {max(count for _, count in quantized)} experts per layer all stay resident"
    else:
        reason = None
    return {
        "total_bytes": total,
        "ple_bytes": ple,
        "expert_bytes": experts if reason is None else 0,
        "expert_floor_bytes": floor if reason is None else 0,
        "experts_per_layer": max((count for _, count in quantized), default = 0),
        "expert_layer_bytes": max((size for size, _ in quantized), default = 0) if reason is None else 0,
        "ple": None if ple else "the model has no per-layer embedding table",
        "experts": reason,
    }


def expert_offload_overhead(inventory):
    """One layer's reads, a slot array's copy a pending computation holds, and the capped cache."""
    return 2 * inventory["expert_layer_bytes"] + _CACHE_CAP


def offload_footprint(inventory, ple, experts, ple_overhead = OFFLOAD_PLE_OVERHEAD, expert_overhead = None):
    """What the offloadable PLE tables and experts hold in memory with ``ple`` / ``experts`` offloaded
    (``None`` for off), working memory included; the rest of the weights stay as they are."""
    ple_bytes = inventory["ple_bytes"] if inventory["ple"] is None else 0
    held = ple_bytes * (1 - (ple or 0)) + (ple_overhead if ple and ple_bytes else 0)
    if experts and inventory["expert_bytes"]:
        overhead = expert_offload_overhead(inventory) if expert_overhead is None else expert_overhead
        return held + max(inventory["expert_floor_bytes"], inventory["expert_bytes"] * (1 - experts)) + overhead
    return held + inventory["expert_bytes"]


def context_bytes(model_dir, tokens, prefill_step = PREFILL_STEPS[0], dtype_bytes = 2):
    """What ``tokens`` of context cost, from the config: the attention KV cache doubled (the
    transients that grow with it), a prefill step's logits and a reserve. Fitted above
    Qwen3.8-Flash-Next's measured peaks (13.7 GiB over the weights at 262K tokens)."""
    config = _text_config(model_dir)
    layers = config.get("num_hidden_layers", 0)
    types = config.get("layer_types")
    if not types:
        interval = config.get("full_attention_interval")
        types = ["full_attention" if not interval or (i + 1) % interval == 0 else "linear_attention" for i in range(layers)]
    heads = config.get("num_key_value_heads") or config.get("num_attention_heads", 0)
    head_dim = config.get("head_dim") or config.get("hidden_size", 0) // max(1, config.get("num_attention_heads", 1))
    if not (types and heads and head_dim):
        raise ValueError("Unsloth: this model's config does not size its attention, so its context cannot be priced.")
    window = config.get("sliding_window") or tokens
    held = sum(tokens if kind == "full_attention" else min(tokens, window) for kind in types if "attention" in kind and kind != "linear_attention")
    return 2 * 2 * heads * head_dim * dtype_bytes * held + 4 * prefill_step * config.get("vocab_size", 0) + _TRANSIENT_RESERVE


def _largest_context(cost, limit, low, high):
    """The largest context in ``[low, high]`` costing at most ``limit``, or ``None``."""
    if cost(low) > limit:
        return None
    if cost(high) <= limit:
        return high
    lo, hi = 0, (high - low) // OFFLOAD_CONTEXT_STEP
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cost(low + mid * OFFLOAD_CONTEXT_STEP) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return low + lo * OFFLOAD_CONTEXT_STEP


def plan_offload(
    *,
    budget,
    resident,
    inventory,
    context_cost,
    ple_overhead,
    expert_overhead,
    ceiling,
    minimum = 4096,
    context = None,
    prefill_step = None,
    ple_offload = "auto",
    expert_offload = "auto",
    margin = SAFETY_MARGIN,
):
    """The 50/50 rule: context takes at most half of what the resident weights leave of ``budget``
    less ``margin`` (more only to reach ``minimum``), and the least offload that fits the rest is
    taken -- none, PLE, then PLE and as many experts as needed. With experts offloaded, each prefill
    step re-reads them, so the step with the fewest offloaded bytes per prompt token is taken.
    ``context_cost(tokens, step)`` prices the context; ``*_overhead`` is each offload's working
    memory. A ``context``, ``prefill_step`` or knob not ``"auto"``/``None`` is kept as given.
    Returns ``{"ple", "experts", "context", "prefill_step"}``, all ``None`` when nothing fits."""
    ple_bytes = inventory["ple_bytes"] if inventory["ple"] is None else 0
    expert_bytes, floor = inventory["expert_bytes"], inventory["expert_floor_bytes"]
    per_layer = inventory["experts_per_layer"]
    room = min(budget - margin[0], budget / (1 + margin[1])) - resident
    nothing = {"ple": None, "experts": None, "context": None, "prefill_step": None}

    def footprint(ple, experts):
        return offload_footprint(inventory, ple, experts, ple_overhead, expert_overhead)

    ple_choices = [ple_offload] if ple_offload != "auto" else ([None, 1.0] if ple_bytes else [None])
    expert_auto = expert_offload == "auto" and expert_bytes > 0
    expert_fixed = None if expert_offload == "auto" else expert_offload

    def choose(weights_room):
        for ple in ple_choices:
            if footprint(ple, expert_fixed) <= weights_room:
                return ple, expert_fixed
        if not expert_auto:
            return None
        ple = ple_choices[-1]
        keep = weights_room - footprint(ple, None) + expert_bytes - expert_overhead
        kept = min(per_layer, int(keep * per_layer // expert_bytes)) if keep > 0 else 0
        if kept * expert_bytes // per_layer < floor:
            return None
        return ple, (per_layer - kept) / per_layer

    first = prefill_step or PREFILL_STEPS[0]
    if context is None:
        target = _largest_context(lambda t: context_cost(t, first), room / 2, minimum, ceiling) or minimum
        if choose(room - context_cost(target, first)) is None:
            smallest = footprint(ple_choices[-1], 1.0 if expert_auto else expert_fixed)
            target = _largest_context(lambda t: context_cost(t, first), room - smallest, minimum, ceiling)
            if target is None:
                return nothing
    else:
        target = context
    chosen = choose(room - context_cost(target, first))
    if chosen is None:
        return nothing
    step = first
    if prefill_step is None and chosen[1]:
        # Offloaded expert bytes re-read per prompt token; disk speed scales every step alike.
        def per_token(candidate):
            return expert_bytes * candidate[1][1] / candidate[0] if candidate[1] and candidate[1][1] else 0

        best = (first, chosen)
        for candidate_step in PREFILL_STEPS[1:]:
            if candidate_step > target:
                break
            candidate = choose(room - context_cost(target, candidate_step))
            if candidate is not None and per_token((candidate_step, candidate)) < per_token(best):
                best = (candidate_step, candidate)
        step, chosen = best
    ple, experts = chosen
    if context is not None:
        return {"ple": ple, "experts": experts, "context": context, "prefill_step": step}
    fitted = _largest_context(lambda t: context_cost(t, step), room - footprint(ple, experts), target, ceiling)
    return {"ple": ple, "experts": experts, "context": fitted or target, "prefill_step": step}
