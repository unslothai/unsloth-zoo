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

"""Rollout routing replay (R3) for MoE GRPO with a colocated vLLM engine.

Opt-in with UNSLOTH_GRPO_ROUTING_REPLAY=1. vLLM (>= 0.14, `enable_return_routed_experts`)
returns the experts it routed every prompt and generated token to. The trainer then forces
those expert ids in its own forwards (old / ref / new logprobs, the loss forward and the
gradient checkpointing recompute), while the routing weights are still recomputed from the
trainer's router logits over the forced ids, so the router keeps its gradient.

Rows are matched by content: the key is the row's attended token ids (the training attention
mask, else the token ids with pad ids dropped on both sides), so any reordering, chunking, left
packing or sequence packing of the batch still finds its rollout. Every `llm.generate` call
between two training forwards belongs to one rollout, so a rollout split into sub-batches is
kept whole. A row or token with no match keeps its natural routing.

Routing weights (UNSLOTH_GRPO_ROUTING_REPLAY_WEIGHTS, on by default with R3, vLLM >= 0.28): the
colocated engine is patched to return the router's fp32 top-k weights next to the ids, and the
trainer forward uses those values. The gradient still goes through the trainer's own weights over
the forced ids (straight-through), so a trainable router and the hidden states feeding a frozen
one keep their gradient. Older vLLM returns ids only and the weights are recomputed as before.

Old-policy replay (R2, UNSLOTH_GRPO_ROUTING_REPLAY_R2=1, no vLLM needed): after each rollout (a
vLLM `generate` call, or a forward with a KV cache, i.e. HF generation), the first trainer forward of
a row records the experts it was routed to. That forward is the one pi_old comes from (TRL's
no-grad old logprob forward, or the first policy forward when pi_old is taken from it), so the later
forwards of that row (num_iterations > 1, later minibatches) replay them. Rows R3 has rollout routes
for keep those.
"""

__all__ = [
    "ROUTING_REPLAY_ENV",
    "ROUTING_REPLAY_WEIGHTS_ENV",
    "ROUTING_REPLAY_R2_ENV",
    "routing_replay_requested",
    "routing_weights_requested",
    "old_policy_replay_requested",
    "vllm_supports_routed_experts",
    "routing_replay_engine_args",
    "record_routed_experts",
    "wrap_vllm_generate",
    "install_routing_replay",
    "routing_replay_stats",
    "reset_routing_replay",
]

import functools
import inspect
import math
import os
import re
import typing
import warnings
from collections import OrderedDict
from collections.abc import Sequence

import torch
import torch.nn.functional as F

ROUTING_REPLAY_ENV = "UNSLOTH_GRPO_ROUTING_REPLAY"
ROUTING_REPLAY_WEIGHTS_ENV = "UNSLOTH_GRPO_ROUTING_REPLAY_WEIGHTS"
ROUTING_REPLAY_R2_ENV = "UNSLOTH_GRPO_ROUTING_REPLAY_R2"
_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")
_MAX_ROLLOUTS = 2
_MAX_CALLS_PER_ROLLOUT = 64
# Leak guard only: evicting a forward awaiting its recompute makes that recompute route naturally.
_MAX_PENDING = 1024
# R2 rows kept (old-policy routes of the latest no-grad forwards); older ones route naturally.
_MAX_R2_ROWS = 8192


def routing_replay_requested():
    return os.environ.get(ROUTING_REPLAY_ENV, "0") == "1"


def routing_weights_requested():
    return routing_replay_requested() and os.environ.get(ROUTING_REPLAY_WEIGHTS_ENV, "1") == "1"


def old_policy_replay_requested():
    return os.environ.get(ROUTING_REPLAY_R2_ENV, "0") == "1"


# Set once the colocated vLLM is patched to return weights: (ids | fp32 low 16 bits | high 16 bits).
_VLLM_WEIGHTS_PACKED = False
# The top-k of the engine this process built with the patch; routes from any other engine (an
# out-of-process server) keep their plain k columns and are read as ids only.
_PACKED_TOP_K = None
# Rollout weights further than this from the trainer's own weights over the same experts are not
# this router's (a scaled or differently normalised vLLM router): those rows keep the trainer's.
_WEIGHT_TOLERANCE = 0.1


def _pack_routed_weights(topk_ids, topk_weights):
    # GPU ops only: runs inside vLLM's CUDA graph capture, replayed every decode step.
    bits = topk_weights.float().contiguous().view(torch.int32)
    return torch.cat([topk_ids.to(torch.int32), bits & 0xFFFF, (bits >> 16) & 0xFFFF], dim = -1)


def _unpack_routed_weights(routed, top_k):
    """vLLM's widened capture [..., 3k] -> (ids [..., k], fp32 weights [..., k])."""
    routed = torch.as_tensor(routed).to(torch.int64)
    ids = routed[..., :top_k]
    bits = (routed[..., top_k : 2 * top_k] & 0xFFFF) | ((routed[..., 2 * top_k :] & 0xFFFF) << 16)
    # int64 -> int32 keeps the low 32 bits, i.e. the fp32 pattern.
    weights = bits.to(torch.int32).view(torch.float32)
    return ids, weights


def _select_reads_capture_fn(select):
    """True when vLLM's BaseRouter._select_experts reads self.capture_fn itself, as 0.28 - 0.34 do to capture the
    logical ids before the EPLB map; the wrapper below swaps that attribute for the call. A probe of the attribute
    names the compiled function loads, not of its source text, so a reformat or a renamed local keeps it."""
    if select is None:
        return False
    code = getattr(inspect.unwrap(select), "__code__", None)
    return code is not None and "capture_fn" in code.co_names


def _patch_vllm_routed_weights():
    """Make the in-process vLLM return fp32 top-k weights beside the expert ids. Every transport
    (0.28 - 0.30 slot buffer, 0.31+ aux output connector) sizes itself from
    _get_routed_experts_shape, so widening its top-k to 3k columns (ids, low and high halves of the
    fp32 bits; 257 experts selects uint16) carries the weights end to end. True when patched."""
    global _VLLM_WEIGHTS_PACKED
    if _VLLM_WEIGHTS_PACKED:
        return True
    try:
        from vllm.model_executor.layers.fused_moe import routed_experts_capturer as capturer_module
        from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    except Exception:
        return False
    shape_fn = getattr(capturer_module, "_get_routed_experts_shape", None)
    capturer_cls = getattr(capturer_module, "RoutedExpertsCapturer", None)
    select = getattr(BaseRouter, "_select_experts", None)
    capture = getattr(capturer_cls, "capture", None)
    if shape_fn is None or capture is None or not _select_reads_capture_fn(select):
        return False
    if getattr(shape_fn, "_unsloth_routed_weights", False):
        _VLLM_WEIGHTS_PACKED = True
        return True

    @functools.wraps(shape_fn)
    def routed_experts_shape(*args, **kwargs):
        global _PACKED_TOP_K
        num_layers, num_experts, top_k = shape_fn(*args, **kwargs)
        _PACKED_TOP_K = top_k
        return num_layers, max(num_experts, 257), 3 * top_k

    @functools.wraps(capture)
    def capture_with_weights(self, layer_id, topk_ids, *args, **kwargs):
        width = self.device_buffer.shape[-1]
        if topk_ids.dim() == 2 and 3 * topk_ids.shape[-1] == width:
            # Ids only (a monolithic kernel, a model's own capture source): zero weights = none.
            topk_ids = torch.cat([topk_ids.to(torch.int32), topk_ids.new_zeros(
                (topk_ids.shape[0], 2 * topk_ids.shape[-1]), dtype = torch.int32,
            )], dim = -1)
        return capture(self, layer_id, topk_ids, *args, **kwargs)

    @functools.wraps(select)
    def select_experts_with_weights(self, *args, **kwargs):
        capture_fn = self.capture_fn
        if capture_fn is None:
            return select(self, *args, **kwargs)
        logical = []
        # The original captures the logical ids before EPLB mapping: take them from its call.
        self.capture_fn = logical.append
        try:
            topk_weights, topk_ids = select(self, *args, **kwargs)
        finally:
            self.capture_fn = capture_fn
        capture_fn(_pack_routed_weights(logical[0] if logical else topk_ids, topk_weights))
        return topk_weights, topk_ids

    routed_experts_shape._unsloth_routed_weights = True
    capturer_module._get_routed_experts_shape = routed_experts_shape
    capturer_cls.capture = capture_with_weights
    BaseRouter._select_experts = select_experts_with_weights
    _VLLM_WEIGHTS_PACKED = True
    return True


def _engine_arg_names():
    try:
        from vllm import EngineArgs
        return set(inspect.signature(EngineArgs).parameters)
    except Exception:
        return set()


def vllm_supports_routed_experts():
    return "enable_return_routed_experts" in _engine_arg_names()


def _config_is_moe(config):
    for sub in (config, getattr(config, "text_config", None)):
        if sub is None:
            continue
        for name in ("num_experts", "num_local_experts", "n_routed_experts"):
            value = getattr(sub, name, None)
            if isinstance(value, int) and value > 1:
                return True
    return False


def _config_is_hybrid_attention(config):
    # Mixed layer kinds (sliding + full, linear / mamba + full) give vLLM several KV cache groups.
    for sub in (config, getattr(config, "text_config", None)):
        if sub is None:
            continue
        for name in ("layer_types", "layers_block_type", "hybrid_override_pattern"):
            value = getattr(sub, name, None)
            if isinstance(value, (list, tuple, str)) and len(set(value)) > 1:
                return True
    return False


def _loop_passes(config):
    """Passes over the same decoder layers (Inkling-2 style looped models), 1 when not looped."""
    for sub in (config, getattr(config, "text_config", None)):
        value = getattr(sub, "loop_num_passes", None) if sub is not None else None
        if isinstance(value, int) and value > 1:
            return value
    return 1


def _warn_looped(passes):
    warnings.warn(
        f"Unsloth: MoE routing replay is off for this looped model (loop_num_passes = {passes}): it keys "
        "routes by layer, and every pass reuses the same layers and routers, so the passes' routes would "
        "overwrite each other.",
        stacklevel = 3,
    )


def _vllm_at_least(major, minor):
    try:
        from importlib.metadata import version
        release = version("vllm").split("+")[0].split(".")
        return (int(release[0]), int(re.match(r"\d+", release[1]).group(0))) >= (major, minor)
    except Exception:
        # Unknown version: trust the engine arg check above.
        return True


def routing_replay_engine_args(config):
    """Extra vLLM engine args, empty unless requested, supported and the model is MoE."""
    if not routing_replay_requested() or not _config_is_moe(config):
        return {}
    if _loop_passes(config) > 1:
        _warn_looped(_loop_passes(config))
        return {}
    if not vllm_supports_routed_experts():
        warnings.warn(
            f"Unsloth: {ROUTING_REPLAY_ENV}=1 needs a vLLM with enable_return_routed_experts "
            "(0.14 or newer, 0.21 or newer for hybrid attention MoE). "
            "Routing replay is off.",
            stacklevel = 2,
        )
        return {}
    if _config_is_hybrid_attention(config) and not _vllm_at_least(0, 21):
        # vLLM 0.14 to 0.20 size the shared capture buffer by KV cache tokens of one group but
        # index it by slot, so a hybrid model's first generate dies with an IndexError. 0.21
        # keeps per-request buffers indexed by position.
        warnings.warn(
            f"Unsloth: {ROUTING_REPLAY_ENV}=1 needs vLLM 0.21 or newer for hybrid attention MoE "
            "models. Routing replay is off.",
            stacklevel = 2,
        )
        return {}
    args = {"enable_return_routed_experts": True}
    if "async_scheduling" in _engine_arg_names() and (
        (_vllm_at_least(0, 21) and not _vllm_at_least(0, 22)) or _vllm_at_least(0, 31)
    ):
        # vLLM 0.21 refuses routed experts capture with (default-on) async scheduling. From 0.31 the capture
        # goes through the aux output connector, and with async scheduling a prefix cache reset between two
        # generate calls (merged rollouts, weight syncs) fails the next step with "auxiliary output
        # generation changed with output in flight" (Qwen3-30B-A3B GRPO, vLLM 0.34, rollout 4).
        args["async_scheduling"] = False
    # Fused FlashInfer MoE kernels (auto on Blackwell) pick experts internally and leave the
    # capture all zeros (vllm-project/vllm#44116); the Triton backend goes through the hook.
    if "moe_backend" in _engine_arg_names():
        args["moe_backend"] = "triton"
    if routing_weights_requested() and not _patch_vllm_routed_weights():
        warnings.warn(
            f"Unsloth: {ROUTING_REPLAY_WEIGHTS_ENV}=1 needs vLLM 0.28 or newer to return routing weights; "
            "routing replay forces the rollout's expert ids and recomputes their weights.",
            stacklevel = 2,
        )
    return args


class _ReplayState:
    def __init__(self):
        # rollout id -> [(store, unpadded)], one entry per llm.generate call of that rollout.
        self.generations = OrderedDict()
        self.generation_id = 0
        self.forwarded_since_record = True
        self.pad_token_id = None
        self.current = None
        # Gradient checkpointing: routes each decoder forward forced per layer, so the recompute
        # in backward replays its own forward even after later forwards (chunked GRPO losses).
        self.in_forward = False
        self.record = None
        self.pending = []
        # Read in compiled regions instead of the list, whose length would become a Dynamo guard.
        self.has_pending = False
        self.recomputed = False
        self.layer_inputs = {}
        self.stats = dict(
            recorded = 0,
            forwards = 0,
            replayed_tokens = 0,
            unmatched_tokens = 0,
            router_calls_replayed = 0,
            router_calls_natural = 0,
            router_calls_recomputed = 0,
            r2_recorded = 0,
            r2_replayed_tokens = 0,
        )
        self.changed = None
        self.warned_invalid = False
        self.warned_evicted = False
        # Rollout fp32 routing weights [tokens, layers, k] beside `current` (NaN: none for that row).
        self.current_weights = None
        # R2: old-policy routes recorded by forwards without grad, attended tokens -> int16 rows.
        self.r2_on = False
        self.r2_exact = OrderedDict()
        self.r2_unpadded = OrderedDict()
        self.r2_capture = None
        self.r2_meta = None
        # Rollouts seen: R2 records nothing before the first, and each one starts a fresh table.
        self.r2_rollout = 0
        self.r2_generating = False
        self.num_layers = None
        self.moe_layers = ()


_STATE = _ReplayState()


def reset_routing_replay():
    global _STATE
    _STATE = _ReplayState()


def routing_replay_stats():
    out = dict(_STATE.stats)
    changed = [0, 0] if _STATE.changed is None else _STATE.changed.reshape(-1).tolist()
    out["changed_selections"] = int(changed[0])
    out["replayed_weight_rows"] = int(changed[1]) if len(changed) > 1 else 0
    return out


def _make_key(token_ids, pad_token_id):
    if pad_token_id is None:
        return tuple(token_ids)
    return tuple(t for t in token_ids if t != pad_token_id)


# A row two captures disagreed on: natural like -1, but known, so prefix reuse must not fill it.
_DISPUTED = -2


def _agreed(a, b):
    """Rows of two captures of the same tokens: where their expert sets differ, natural; where both
    were routed (or either was already disputed), _DISPUTED rather than -1 (unknown)."""
    if a.shape != b.shape:
        return a
    same = (a.sort(dim = -1).values == b.sort(dim = -1).values).all(dim = -1, keepdim = True)
    routed = lambda x: ~(x < 0).all(dim = -1, keepdim = True) | (x == _DISPUTED).all(dim = -1, keepdim = True)
    natural = torch.where(routed(a) & routed(b), _DISPUTED, -1).to(a.dtype).expand_as(a)
    return torch.where(same, a, natural)


def _count_conflicts(a, b):
    if a.shape != b.shape:
        return
    merged = _agreed(a, b)
    natural = lambda x: (x < 0).all(dim = -1)
    conflicts = int((natural(merged) & ~natural(a) & ~natural(b)).sum())
    if conflicts:
        _STATE.stats["conflicting_routes"] = _STATE.stats.get("conflicting_routes", 0) + conflicts


def _fill_prefix(rows, full, start, store, weights = None):
    """Fill rows [0, start) from this rollout's captures sharing that prefix; disagreeing captures leave -1.
    `weights`: this capture's weight rows, filled from the same captures as the ids (so their order matches)."""
    prefix = tuple(full[:start])
    stores = [store]
    if not _STATE.forwarded_since_record and _STATE.generations:
        # Earlier rollouts ran on older weights: their routes are not this policy's.
        stores += [call[0] for call in reversed(next(reversed(_STATE.generations.values())))]
    merged = rows[:start].clone()
    merged_weights = None if weights is None else weights[:start].clone()
    seen = torch.zeros(rows.shape[:-1], dtype = torch.bool)[:start]
    conflict = seen.clone()
    for candidate in stores:
        for key, earlier in candidate.items():
            m = min(start, len(key))
            if m == 0 or earlier.shape[1:] != rows.shape[1:] or key[:m] != prefix[:m]:
                continue
            part = earlier[:m]
            routed = ~(part < 0).all(dim = -1)
            same = (part.sort(dim = -1).values == merged[:m].sort(dim = -1).values).all(dim = -1)
            conflict[:m] |= routed & seen[:m] & ~same
            conflict[:m] |= (part == _DISPUTED).all(dim = -1)
            new = routed & ~seen[:m]
            if bool(new.any()):
                merged[:m] = torch.where(new.unsqueeze(-1), part, merged[:m])
                if merged_weights is not None:
                    earlier_weights = getattr(candidate, "weights", {}).get(key, None)
                    part_weights = torch.full(part.shape, float("nan")) if earlier_weights is None \
                        or earlier_weights.shape[1:] != part.shape[1:] else earlier_weights[:m]
                    merged_weights[:m] = torch.where(new.unsqueeze(-1), part_weights, merged_weights[:m])
                seen[:m] |= new
    keep = seen & ~conflict
    rows[:start] = torch.where(keep.unsqueeze(-1), merged, rows[:start])
    if weights is not None:
        weights[:start] = torch.where(keep.unsqueeze(-1), merged_weights, weights[:start])


class _Store(dict):
    """token ids -> expert id rows of one generate call; `weights` holds their fp32 routing weights
    (NaN where vLLM returned none) for the keys it has them for."""

    def __init__(self):
        super().__init__()
        self.weights = {}


def _as_routes(routed):
    """(ids int16, weights fp32 or None) from one vLLM capture."""
    routed = torch.as_tensor(routed)
    if _PACKED_TOP_K and routed.dim() >= 1 and routed.shape[-1] == 3 * _PACKED_TOP_K:
        ids, weights = _unpack_routed_weights(routed, _PACKED_TOP_K)
        return ids.to(torch.int16), weights
    return routed.to(torch.int16), None


def _clean_weights(rows, weights):
    # Usable top-k weights are finite, in [0, 1] and sum to at most 1 (scaled or garbage ones are
    # not this router's); an all zero row is an ids-only capture.
    w = weights.float()
    eps = 1e-3
    ok = torch.isfinite(w).all(-1) & (w >= 0).all(-1) & (w <= 1 + eps).all(-1)
    ok &= (w.sum(-1) <= 1 + eps) & (w > 0).any(-1) & ~(rows < 0).any(-1)
    return torch.where(ok.unsqueeze(-1), w, torch.full_like(w, float("nan")))


def record_routed_experts(request_outputs, pad_token_id = None, prompt_starts = None):
    """Store vLLM's per-token expert ids (and routing weights, when the engine returns them) per
    finished completion. prompt_starts: per-request `SamplingParams.routed_experts_prompt_start`
    (prompt position of the first returned row)."""
    if pad_token_id is None:
        pad_token_id = _STATE.pad_token_id
    store = _Store()
    unpadded = {}
    for index, request_output in enumerate(request_outputs):
        prompt_ids = list(getattr(request_output, "prompt_token_ids", None) or [])
        start = int(prompt_starts[index] or 0) if prompt_starts is not None else 0
        # vLLM 0.21 returns prompt routes on the request, generated-token routes on the completion.
        split = hasattr(request_output, "prompt_routed_experts")
        prompt_routed = getattr(request_output, "prompt_routed_experts", None)
        prompt_weights = None
        if prompt_routed is not None:
            prompt_routed, prompt_weights = _as_routes(prompt_routed)
        for completion in getattr(request_output, "outputs", None) or []:
            routed = getattr(completion, "routed_experts", None)
            weights = None
            if routed is None and split and prompt_routed is not None:
                # vLLM 0.21 returns None for an empty generation part; the prompt part stays on the request.
                routed = prompt_routed[:0]
                weights = None if prompt_weights is None else prompt_weights[:0]
            elif routed is not None:
                routed, weights = _as_routes(routed)
            if routed is None:
                continue
            full = prompt_ids + list(completion.token_ids)
            # vLLM never forwards the last sampled token, so it has no routing.
            rows = torch.full((len(full),) + tuple(routed.shape[1:]), -1, dtype = torch.int16)
            wrows = None
            if weights is not None or prompt_weights is not None:
                wrows = torch.full(rows.shape, float("nan"), dtype = torch.float32)
            if split:
                parts = [(len(prompt_ids), routed, weights)]
                if prompt_routed is not None and prompt_routed.shape[1:] == routed.shape[1:]:
                    parts.append((0, prompt_routed[: len(prompt_ids)], prompt_weights))
            else:
                parts = [(start, routed, weights)]
            for offset, part, part_weights in parts:
                n = max(0, min(len(full) - offset, part.shape[0]))
                rows[offset : offset + n] = part[:n]
                if wrows is not None and part_weights is not None and part_weights.shape[1:] == wrows.shape[1:]:
                    wrows[offset : offset + n] = part_weights[:n]
            if start > 0 and not split:
                _fill_prefix(rows, full, start, store, wrows)
            if rows.shape[-1] > 1:
                # A real top-k picks distinct experts; repeats (an all-zero capture) are not routing.
                ordered = rows.sort(dim = -1).values
                bad = (ordered[..., 1:] == ordered[..., :-1]).any(dim = -1) & (ordered[..., :1] >= 0).squeeze(-1)
                if bool(bad.any()):
                    rows[bad] = -1
                    _STATE.stats["invalid_rows"] = _STATE.stats.get("invalid_rows", 0) + int(bad.sum())
                    if not _STATE.warned_invalid:
                        _STATE.warned_invalid = True
                        warnings.warn(
                            "Unsloth: vLLM returned routed experts with repeated ids; those tokens keep "
                            "the trainer's own routing."
                        )
            if wrows is not None:
                wrows = _clean_weights(rows, wrows)
            # Keyed by every token: a pad id inside the sequence (pad == eos) can be real.
            key = tuple(full)
            if key in store:
                _count_conflicts(store[key], rows)
                rows = _agreed(store[key], rows)
                # _agreed keeps the earlier capture's expert order: only its weights stay aligned.
                wrows = store.weights.get(key, None)
            store[key] = rows
            if wrows is not None:
                store.weights[key] = wrows
                _STATE.stats["recorded_weights"] = _STATE.stats.get("recorded_weights", 0) + 1
            if pad_token_id is not None:
                unpadded[_make_key(full, pad_token_id)] = key
            _STATE.stats["recorded"] += 1
    if store:
        # A training forward closes the rollout; generate calls before the next one join it.
        if _STATE.forwarded_since_record or not _STATE.generations:
            _STATE.generation_id += 1
            _STATE.generations[_STATE.generation_id] = []
            _STATE.forwarded_since_record = False
        calls = next(reversed(_STATE.generations.values()))
        for key, rows in store.items():
            earlier = None
            for previous, _, _ in calls:
                if key in previous:
                    earlier = previous[key] if earlier is None else _agreed(earlier, previous[key])
            if earlier is not None:
                _count_conflicts(earlier, rows)
        calls.append((store, unpadded, pad_token_id))
        # generate without training forwards in between (evaluation) would otherwise grow forever.
        del calls[:-_MAX_CALLS_PER_ROLLOUT]
        while len(_STATE.generations) > _MAX_ROLLOUTS:
            _STATE.generations.popitem(last = False)
    return len(store)


def _lookup(key, exact = False):
    """Rows for `key`: all attended tokens when `exact`, else the tokens with pad ids dropped."""
    return _lookup_routes(key, exact)[0]


def _lookup_routes(key, exact = False):
    """(expert id rows, fp32 weight rows or None) for `key`; (None, None) when unrecorded."""
    for rollout in reversed(_STATE.generations.values()):
        found = found_weights = None
        for store, unpadded, pad in reversed(rollout):
            weights = None
            if exact or pad is None:
                rows = store.get(key, None)
                if rows is not None:
                    weights = getattr(store, "weights", {}).get(key, None)
            else:
                full = unpadded.get(key, None)
                rows = None
                if full is not None:
                    keep = torch.tensor([t != pad for t in full], dtype = torch.bool)
                    rows = store[full][keep]
                    weights = getattr(store, "weights", {}).get(full, None)
                    weights = None if weights is None else weights[keep]
            if rows is not None:
                if found is None:
                    found, found_weights = rows, weights
                else:
                    # The first (latest) capture's expert order is kept: only its weights stay aligned.
                    found = _agreed(found, rows)
        if found is not None:
            return found.clamp_min(-1), found_weights
    return None, None


def _prompt_starts(args, kwargs, count):
    params = kwargs.get("sampling_params", args[1] if len(args) > 1 else None)
    # Same per-prompt test as vLLM's offline _params_to_seq; a SamplingParams is no Sequence.
    if isinstance(params, Sequence):
        starts = [getattr(p, "routed_experts_prompt_start", 0) or 0 for p in params]
        if len(starts) != count:
            if any(starts):
                raise ValueError("cannot align routed_experts_prompt_start with the outputs")
            return None
        return starts if any(starts) else None
    start = getattr(params, "routed_experts_prompt_start", 0) or 0
    return [start] * count if start else None


def wrap_vllm_generate(llm):
    """Record routed experts from every `llm.generate` call. Idempotent."""
    generate = getattr(llm, "generate", None)
    if generate is None or getattr(generate, "_unsloth_routing_replay", False):
        return llm

    @functools.wraps(generate)
    def generate_with_routing_record(*args, **kwargs):
        _r2_new_rollout()
        outputs = generate(*args, **kwargs)
        try:
            outputs_list = outputs if isinstance(outputs, (list, tuple)) else [outputs]
            record_routed_experts(outputs_list, prompt_starts = _prompt_starts(args, kwargs, len(outputs_list)))
        except Exception as error:
            warnings.warn(f"Unsloth: routing replay could not record routed experts: {error}")
        return outputs

    generate_with_routing_record._unsloth_routing_replay = True
    llm.generate = generate_with_routing_record
    return llm


def _segments(input_ids, position_ids):
    """(row, start, end) spans: rows, or position_ids resets inside a packed [1, T] row."""
    rows, T = input_ids.shape
    if rows == 1 and position_ids is not None and position_ids.dim() >= 2 and position_ids.shape[-1] == T:
        pos = position_ids.reshape(-1, T)[0].tolist()
        starts = [i for i, p in enumerate(pos) if p == 0] or [0]
        if starts[0] != 0:
            starts = [0] + starts
        return [(0, s, e) for s, e in zip(starts, starts[1:] + [T])]
    return [(r, 0, T) for r in range(rows)]


def _fit_layers(rows, width, fill):
    """Rows in another layer numbering: vLLM 0.21+ numbers MoE layers only, R2 every decoder layer."""
    if rows.shape[1] == width:
        return rows
    moe = list(_STATE.moe_layers)
    if not moe or _STATE.num_layers is None:
        return None
    if rows.shape[1] == len(moe) and width == _STATE.num_layers:
        out = torch.full((rows.shape[0], width) + tuple(rows.shape[2:]), fill, dtype = rows.dtype)
        out[:, moe] = rows
        return out
    if rows.shape[1] == _STATE.num_layers and width == len(moe):
        return rows[:, moe]
    return None


def _r2_new_rollout():
    if _STATE.r2_on:
        _STATE.r2_exact.clear()
        _STATE.r2_unpadded.clear()
        _STATE.r2_rollout += 1


def _r2_lookup(key, exact):
    if not _STATE.r2_on:
        return None
    rows = (_STATE.r2_exact if exact else _STATE.r2_unpadded).get(key, None)
    return None if rows is None else torch.from_numpy(rows)


def _build_replay(input_ids, position_ids, attention_mask = None, use_r2 = False, r2_new = None):
    """Replay rows for this forward. r2_new: a list that receives the (row, start, end) segments
    neither R3 nor R2 has routes for, for R2 to record."""
    _STATE.current_weights = None
    if not _STATE.generations and not use_r2:
        return None
    pad = _STATE.pad_token_id
    ids = input_ids.detach().to("cpu")
    rows, T = ids.shape
    out = out_weights = None
    replayed = unmatched = r2_replayed = 0
    ids_list = ids.tolist()
    mask_list = None
    if torch.is_tensor(attention_mask) and tuple(attention_mask.shape) == (rows, T):
        mask_list = attention_mask.detach().to("cpu").bool().tolist()
    for row, start, end in _segments(ids, position_ids):
        tokens = ids_list[row][start:end]
        routed = weights = None
        from_r2 = False
        if mask_list is not None:
            # Padding is what the mask leaves out, so real pad id tokens keep their routing.
            cols = [start + i for i in range(end - start) if mask_list[row][start + i]]
            if cols:
                attended = tuple(ids_list[row][c] for c in cols)
                routed, weights = _lookup_routes(attended, exact = True)
                if routed is None and use_r2:
                    routed = _r2_lookup(attended, exact = True)
                    from_r2 = routed is not None
                if routed is None and pad is not None and pad in attended:
                    # An attended pad id is a real token (pad == eos): dropping it would match
                    # another history, so a miss keeps the natural routing.
                    unmatched += len(cols)
                    if r2_new is not None:
                        r2_new.append((row, start, end))
                    continue
        elif pad is not None and pad in tokens:
            # Packed segments hold only attended tokens: try the whole history first.
            cols = list(range(start, end))
            routed, weights = _lookup_routes(tuple(tokens), exact = True)
            if routed is None and use_r2:
                routed = _r2_lookup(tuple(tokens), exact = True)
                from_r2 = routed is not None
        if routed is None:
            cols = [start + i for i, t in enumerate(tokens) if pad is None or t != pad]
            if not cols:
                continue
            routed, weights = _lookup_routes(_make_key(tokens, pad))
            if routed is None and use_r2:
                routed = _r2_lookup(_make_key(tokens, pad), exact = False)
                from_r2 = routed is not None
        if routed is None and r2_new is not None:
            r2_new.append((row, start, end))
        if routed is None or routed.shape[0] != len(cols):
            unmatched += len(cols)
            continue
        if out is None:
            out = torch.full((rows * T,) + tuple(routed.shape[1:]), -1, dtype = torch.int16)
            if _VLLM_WEIGHTS_PACKED:
                out_weights = torch.full(out.shape, float("nan"), dtype = torch.float32)
        routed = _fit_layers(routed, out.shape[1], -1) if routed.dim() == 3 and out.dim() == 3 else routed
        if routed is None or routed.shape[1:] != out.shape[1:]:
            unmatched += len(cols)
            continue
        index = torch.tensor(cols) + row * T
        out[index] = routed.to(torch.int16)
        if out_weights is not None and weights is not None:
            weights = _fit_layers(weights, out.shape[1], float("nan")) if weights.dim() == 3 else None
            if weights is not None and weights.shape[1:] == out_weights.shape[1:]:
                out_weights[index] = weights
        replayed += len(cols)
        r2_replayed += len(cols) if from_r2 else 0
    _count("replayed_tokens", replayed)
    _count("unmatched_tokens", unmatched)
    if r2_replayed:
        _count("r2_replayed_tokens", r2_replayed)
    if out is None:
        return None
    if out_weights is not None:
        _STATE.current_weights = out_weights.to(input_ids.device, non_blocking = True)
    return out.to(input_ids.device, non_blocking = True).long()


def _r2_store_forward():
    """Store the routes a forward without grad used, per row, as the old policy's (R2)."""
    capture, meta = _STATE.r2_capture, _STATE.r2_meta
    _STATE.r2_capture = _STATE.r2_meta = None
    if not capture or meta is None or _STATE.num_layers is None:
        return
    ids, position_ids, attention_mask, segments = meta
    rows, T = ids.shape
    k = next(iter(capture.values())).shape[-1]
    full = torch.full((rows * T, _STATE.num_layers, k), -1, dtype = torch.int16)
    for layer_index, routed in capture.items():
        if layer_index < _STATE.num_layers and routed.shape == (rows * T, k):
            full[:, layer_index] = routed.to("cpu", torch.int16)
    full = full.numpy().copy()  # numpy: forwards under inference_mode make inference tensors
    pad = _STATE.pad_token_id
    ids_list = ids.tolist()
    mask_list = None
    if torch.is_tensor(attention_mask) and tuple(attention_mask.shape) == (rows, T):
        mask_list = attention_mask.detach().to("cpu").bool().tolist()
    stored = 0
    for row, start, end in segments:
        if mask_list is not None:
            cols = [start + i for i in range(end - start) if mask_list[row][start + i]]
        else:
            cols = list(range(start, end))
        if not cols:
            continue
        tokens = [ids_list[row][c] for c in cols]
        exact = tuple(tokens)
        # The first forward of a row after the rollout is pi_old's: later ones never overwrite it.
        _STATE.r2_exact.setdefault(exact, full[[c + row * T for c in cols]])
        # Without a pad id the key keeps every token, as the lookup's does.
        kept = [c for c, t in zip(cols, tokens) if pad is None or t != pad]
        _STATE.r2_unpadded.setdefault(_make_key(tokens, pad), full[[c + row * T for c in kept]])
        stored += 1
    for table in (_STATE.r2_exact, _STATE.r2_unpadded):
        while len(table) > _MAX_R2_ROWS:
            table.popitem(last = False)
    _count("r2_recorded", stored)


_MEDIA_KWARGS = ("pixel_values", "pixel_values_videos", "input_features")


def _input_ids(args, kwargs):
    input_ids = kwargs.get("input_ids", None)
    if input_ids is None and len(args) > 0 and torch.is_tensor(args[0]) and not args[0].is_floating_point():
        input_ids = args[0]
    return input_ids


def _decoder_pre_hook(module, args, kwargs, parent_input_ids = (), moe_layers = ()):
    if moe_layers and not torch.compiler.is_compiling():
        _wrap_checkpoint_funcs(moe_layers)
    input_ids = _input_ids(args, kwargs)
    if input_ids is None and parent_input_ids:
        parent_ids, embeds = parent_input_ids[-1], kwargs.get("inputs_embeds")
        if torch.is_tensor(parent_ids) and torch.is_tensor(embeds) and embeds.shape[:2] == parent_ids.shape:
            input_ids = parent_ids
    _count("forwards")
    _STATE.forwarded_since_record = True
    _STATE.in_forward = True
    _STATE.record = None
    if torch.is_grad_enabled() and _uses_checkpointing(module):
        if _STATE.recomputed:
            _STATE.pending.clear()
            _STATE.recomputed = False
        _STATE.record = {}
        _STATE.pending.append(_STATE.record)
        _STATE.has_pending = True
        if len(_STATE.pending) > _MAX_PENDING:
            del _STATE.pending[:-_MAX_PENDING]
            if not _STATE.warned_evicted:
                _STATE.warned_evicted = True
                warnings.warn(
                    f"Unsloth: routing replay kept {_MAX_PENDING} checkpointed forwards without a backward; "
                    "the oldest ones recompute with the trainer's own routing, so their gradients no longer "
                    "match their forward. Run backward more often (fewer chunks per backward)."
                )
    _STATE.r2_capture = _STATE.r2_meta = None
    use_r2 = False
    if _STATE.r2_on:
        # Generation hands the decoder a KV cache; scoring forwards build none before the call.
        # transformers 5.24 renamed past_key_values to cache.
        generating = any(kwargs.get(name, None) is not None for name in _CACHE_KWARGS)
        if generating and not _STATE.r2_generating:
            _r2_new_rollout()
        _STATE.r2_generating = generating
        use_r2 = not generating and _STATE.r2_rollout > 0
    if input_ids is None or input_ids.dim() != 2:
        _STATE.current = None
        _STATE.current_weights = None
        return None
    r2_new = [] if use_r2 else None
    position_ids, attention_mask = kwargs.get("position_ids", None), kwargs.get("attention_mask", None)
    # Left set after the forward on purpose: the gradient checkpointing recompute in
    # backward must see the same routing as the forward it replays.
    _STATE.current = _build_replay(input_ids, position_ids, attention_mask, use_r2 = use_r2, r2_new = r2_new)
    if r2_new:
        # R2: rows seen for the first time since the rollout record the routes this forward uses.
        _STATE.r2_capture = {}
        _STATE.r2_meta = (input_ids.detach().to("cpu"), position_ids, attention_mask, r2_new)
    return None


def _decoder_post_hook(module, args, output):
    _STATE.in_forward = False
    _STATE.record = None
    if _STATE.r2_capture is not None:
        _r2_store_forward()
    return None


_CACHE_KWARGS = ("cache", "past_key_values", "past_key_value", "cache_params")


def _uses_checkpointing(decoder):
    if getattr(decoder, "gradient_checkpointing", False):
        return True
    return any(getattr(layer, "gradient_checkpointing", False) for layer in getattr(decoder, "layers", None) or ())


def _fingerprint(ids):
    # The natural top-k ids, order-free per token. Kept whole rather than hashed: two forwards
    # whose routes merely hash alike would otherwise swap their replays in the recompute.
    return ids.sort(-1).values.to(torch.int16)


_NATURAL = object()


_SIGNATURE_PROJECTIONS = {}
_SIGNATURE_CHUNK = 4096


def _signature_projection(width, device):
    key = (width, str(device))
    projection = _SIGNATURE_PROJECTIONS.get(key)
    if projection is None:
        generator = torch.Generator().manual_seed(0x5EED)
        projection = torch.randn(width, 2, generator = generator, dtype = torch.float32).to(device)
        _SIGNATURE_PROJECTIONS[key] = projection
    return projection


def _project(x):
    # Chunked to bound the fp32 upcast.
    projection = _signature_projection(x.shape[-1], x.device)
    return torch.cat([
        x[a : a + _SIGNATURE_CHUNK].float() @ projection
        for a in range(0, x.shape[0], _SIGNATURE_CHUNK)
    ])


def _signature_impl(x: torch.Tensor) -> torch.Tensor:
    # Opaque: the seeded Generator is untraceable and the chunk loop would specialise on token count.
    return _project(x)


def _signature_fake(x):
    return x.new_empty((x.shape[0], 2), dtype = torch.float32)


def _input_signature(args, tokens):
    # Per token fp32 random projections of the router input. The recompute feeds the router the
    # same bits as the forward did, so this tells apart forwards whose natural routes are equal.
    # Random directions, not a plain sum: zero-centred (LayerNorm) inputs all sum to about zero.
    x = args[0] if args and torch.is_tensor(args[0]) else None
    if x is None or not x.is_floating_point() or tokens == 0 or x.numel() % tokens != 0:
        return None
    # Autocast would run the projection in bf16 / fp16 and merge nearby inputs.
    with torch.no_grad(), torch.autocast(device_type = x.device.type, enabled = False):
        x = x.detach().reshape(tokens, -1)
        return _SIGNATURE_OP(x) if torch.compiler.is_compiling() else _project(x)


def _recomputed_routes(layer_index, ids2, signature = None, layer_input = None):
    # A router call outside a decoder forward is a checkpoint recompute: pick the recorded forward
    # whose natural routing it reproduces, without a host sync. None when no forward was recorded.
    if torch.compiler.is_compiling():
        return _compiled_recomputed_routes(layer_index, ids2, signature, layer_input)
    if not _STATE.pending:
        return None
    _STATE.recomputed = True
    return _pick_recorded(layer_index, ids2, signature, layer_input)


def _compiled_recomputed_routes(layer_index, ids2, signature, layer_input):
    # Opaque so Dynamo never guards on the pending list's length (recompile per forward count).
    if not _STATE.has_pending:
        return None
    _STATE.recomputed = True
    return _RECOMPUTE_OP(ids2, signature, layer_input, layer_index)


def _no_weights(ids2):
    return torch.full(ids2.shape, float("nan"), dtype = torch.float32, device = ids2.device)


def _recompute_impl(
    ids2: torch.Tensor, signature: typing.Optional[torch.Tensor],
    layer_input: typing.Optional[torch.Tensor], layer_index: int,
) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _STATE.pending:
        # Run time: a non-reentrant recompute stops early, so a traced assignment would never run.
        _STATE.recomputed = True
    picked = _pick_recorded(layer_index, ids2, signature, layer_input) if _STATE.pending else _NATURAL
    if picked is _NATURAL:
        return (*_natural_impl(ids2), _no_weights(ids2))
    forced, valid = picked[0], picked[1]
    weights = picked[2] if len(picked) > 2 else None
    weights = _no_weights(ids2) if weights is None else weights.to(ids2.device, torch.float32).clone()
    return forced.long().clone(), valid.reshape(ids2.shape[0], 1).to(ids2.device).clone(), weights


def _recompute_fake(ids2, signature, layer_input, layer_index):
    return (*_natural_fake(ids2), ids2.new_empty(ids2.shape, dtype = torch.float32))


def _natural_impl(ids2: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    return ids2.long().clone(), torch.zeros(ids2.shape[0], 1, dtype = torch.bool, device = ids2.device)


def _natural_fake(ids2):
    return (
        ids2.new_empty(ids2.shape, dtype = torch.int64),
        ids2.new_empty((ids2.shape[0], 1), dtype = torch.bool),
    )


def _pick_recorded(layer_index, ids2, signature, layer_input):
    candidates = [r[layer_index] for r in _STATE.pending if layer_index in r and r[layer_index][1].shape == ids2.shape]
    if not candidates:
        # No forward replayed this layer: recompute it exactly as it ran, with no extra ops
        # (a non-reentrant checkpoint rejects a recompute that saves other tensors).
        return _NATURAL
    fp = _fingerprint(ids2)
    match = (torch.stack([c[0] for c in candidates]) == fp).flatten(1).all(-1)
    if torch.is_tensor(layer_input):
        # Recompute reuses the forward's input tensors, so the forward sharing the most addresses
        # is the one recomputed. A mask, not a narrowed list: in a compiled layer the addresses are only known at run time.
        recorded = [c[4] if len(c) > 4 and torch.is_tensor(c[4]) else layer_input[:0] for c in candidates]
        mask = _IDENTITY_MASK_OP(layer_input, recorded) if torch.compiler.is_compiling() \
            else _identity_mask(layer_input, recorded)
        match = match & mask.to(match.device, non_blocking = True)
    if signature is not None:
        # Narrow the natural matches to the forwards whose router input was the same too. Only a
        # refinement: when no input matches exactly, the natural matches stand as before.
        has = [len(c) > 3 and c[3] is not None and c[3].shape == signature.shape for c in candidates]
        if all(has):
            same = (torch.stack([c[3] for c in candidates]) == signature).flatten(1).all(-1)
        else:
            same = torch.stack([
                (c[3] == signature).all() if h else torch.zeros((), dtype = torch.bool, device = match.device)
                for c, h in zip(candidates, has)
            ])
        exact = match & same.to(match.device)
        match = torch.where(exact.any(), exact, match)
    # index_select, not forced_all[pick]: a 0-dim CUDA index is read back to the host.
    pick = match.to(torch.int32).argmax().reshape(1)
    forced_all = torch.stack([c[1] for c in candidates])
    forced = forced_all.index_select(0, pick)[0]
    # Several forwards with the same natural routing but different replays cannot be told
    # apart, so keep the natural routing there rather than guess. Only the expert set counts:
    # the same experts in another order route and weight every token identically.
    agree = ((forced_all.sort(-1).values == forced.sort(-1).values).flatten(1).all(-1) | ~match).all()
    valid = torch.stack([c[2] for c in candidates]).index_select(0, pick)[0] & match.any() & agree
    if not any(len(c) > 5 and c[5] is not None for c in candidates):
        return forced, valid
    # Rollout weights the picked forward replayed, so its recompute uses the same values.
    weights = torch.stack([
        c[5].to(forced.device, torch.float32) if len(c) > 5 and c[5] is not None and c[5].shape == forced.shape
        else torch.full(forced.shape, float("nan"), dtype = torch.float32, device = forced.device)
        for c in candidates
    ]).index_select(0, pick.to(forced.device))[0]
    return forced, valid, weights


def _softmax_gather(logits, ids, renorm):
    if renorm:
        # Equal to softmax over all experts gathered then renormalised, but without the 0 / 0 NaN
        # when every forced expert underflows (forced logits far below the natural top-k).
        return F.softmax(logits.float().gather(-1, ids), dim = -1)
    return F.softmax(logits.float(), dim = -1).gather(-1, ids)


# Below this the forced experts' probability mass is gone: 1 / total would reach 1e20 in the
# backward, and the router could not have picked them anyway.
_MIN_RENORM_TOTAL = 1e-20


def _probs_gather(probs, ids, renorm):
    w = probs.float().gather(-1, ids)
    if not renorm:
        return w
    # Forced experts whose probabilities (nearly) all underflowed have no weights to renormalise:
    # w / total there is 0 / 0, and a subnormal total still overflows the backward (~1 / total).
    # Those rows return zeros (the router hook then keeps the natural routing); dividing by 1
    # there keeps the masked-out branch's gradient finite.
    total = w.sum(dim = -1, keepdim = True)
    ok = total > _MIN_RENORM_TOTAL
    out = w / torch.where(ok, total, torch.ones_like(total))
    return torch.where(ok, out, torch.zeros_like(out))


def _sigmoid_gather(logits, ids, renorm):
    # DeepSeek-V3 / GLM-4 MoE style: the score correction bias only picks the experts; the weights are the
    # unbiased sigmoid scores over the picked ones, renormalised with the same 1e-20 floor.
    w = logits.float().sigmoid().gather(-1, ids)
    if not renorm:
        return w
    return w / (w.sum(dim = -1, keepdim = True) + 1e-20)


def _replay_scale(module):
    """The factor the router multiplies its top-k weights by (routed_scaling_factor), 1.0 when it has none."""
    return float(getattr(module, "_unsloth_replay_scale", 1.0))


def _scale_candidates(module):
    # A router may apply its scaling factor itself (DeepSeek-V3, HY-V3, LFM2-MoE) or leave it to its MoE block:
    # try both.
    for name in ("routed_scaling_factor", "router_scaling_factor"):
        scale = getattr(module, name, None)
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            continue
        if math.isfinite(scale) and scale > 0 and scale != 1.0:
            return (scale, 1.0)
    return (1.0,)


_BIAS_ARGS = ("e_score_correction_bias", "expert_bias")


def _bias_routed(module):
    """True for a router that picks experts by biased scores: it holds the bias, or its forward takes it."""
    if any(hasattr(module, name) for name in _BIAS_ARGS):
        return True
    try:
        params = inspect.signature(type(module).forward).parameters
    except (TypeError, ValueError):
        return False
    return any(name in params for name in _BIAS_ARGS)


def _apply_rollout_weights(weights, rollout, valid, renorm, total = 1.0):
    """Straight-through: the rollout's weight values, the gradient of the trainer's `weights`.
    Rows whose rollout weights are missing, not normalised like this router's (to `total`, its
    routed_scaling_factor), or far from the trainer's keep the trainer's. The delta is built without grad, so autograd saves nothing new
    (a non-reentrant checkpoint compares what its recompute saves)."""
    with torch.no_grad():
        trainer = weights.detach().float()
        rollout = rollout.to(trainer.device, torch.float32)
        diff = rollout - trainer
        ok = valid & torch.isfinite(rollout).all(-1, keepdim = True)
        ok = ok & (diff.abs() <= _WEIGHT_TOLERANCE).all(-1, keepdim = True)
        if renorm:
            ok = ok & ((rollout.sum(-1, keepdim = True) - total).abs() <= 1e-2 * max(1.0, total))
        delta = torch.where(ok, diff, torch.zeros_like(diff)).to(weights.dtype)
    return weights + delta, ok.sum()


def _formulas():
    # softmax_renorm also covers gpt-oss style softmax over the top-k logits: they are equal.
    # probs_*: transformers 5.2 - 5.7 Qwen MoE routers return softmax probabilities as "logits".
    return OrderedDict(
        softmax_renorm = lambda logits, ids: _softmax_gather(logits, ids, True),
        softmax = lambda logits, ids: _softmax_gather(logits, ids, False),
        probs_renorm = lambda probs, ids: _probs_gather(probs, ids, True),
        probs = lambda probs, ids: _probs_gather(probs, ids, False),
        sigmoid_renorm = lambda logits, ids: _sigmoid_gather(logits, ids, True),
        sigmoid = lambda logits, ids: _sigmoid_gather(logits, ids, False),
    )


def _formula_order(module):
    """Formula names in the order they are tried: sigmoid first for a router with a score correction bias, since
    near-zero logits make a renormalised sigmoid and a renormalised softmax agree (both about 1/k)."""
    names = list(_formulas())
    if _bias_routed(module):
        names = [n for n in names if n.startswith("sigmoid")] + [n for n in names if not n.startswith("sigmoid")]
    return names


def _looks_like_probs(logits2):
    with torch.no_grad():
        return bool(((logits2 >= 0).all() & ((logits2.float().sum(-1) - 1).abs() < 1e-3).all()).item())


def _split_router_output(output):
    """(logits position, weights position, ids position, dense) in a (logits, weights, ids) style
    tuple. dense: the weights are [tokens, experts] scores, zero off the top-k (Unsloth's GPT-OSS)."""
    if not isinstance(output, tuple) or len(output) != 3:
        return None
    ids_pos = [i for i, x in enumerate(output) if torch.is_tensor(x) and not x.is_floating_point()]
    float_pos = [i for i, x in enumerate(output) if torch.is_tensor(x) and x.is_floating_point()]
    if len(ids_pos) != 1 or len(float_pos) != 2:
        return None
    ids = output[ids_pos[0]]
    k = ids.shape[-1]
    weights_pos = [i for i in float_pos if output[i].shape[-1] == k and output[i].shape[:-1] == ids.shape[:-1]]
    logits_pos = [i for i in float_pos if output[i].shape[-1] > k]
    if len(weights_pos) == 1 and len(logits_pos) == 1:
        return logits_pos[0], weights_pos[0], ids_pos[0], False
    a, b = (output[i] for i in float_pos)
    if not weights_pos and a.shape == b.shape and a.shape[:-1] == ids.shape[:-1]:
        # (router_logits, router_scores, router_indices), both [tokens, experts].
        return float_pos[0], float_pos[1], ids_pos[0], True
    return None


def _count(name, amount = 1):
    # Python counters read inside a compiled region become Dynamo guards whose value changes every
    # call, recompiling the router (or a compiled decoder) each step. They are diagnostics only,
    # so skip them there.
    if not torch.compiler.is_compiling():
        _STATE.stats[name] += amount


def _router_hook(layer_index, compact = None):
    # compact: (position among the MoE layers, number of MoE layers) when some decoder layers are
    # dense. vLLM 0.21 numbers routed experts by MoE layer only, so its rows skip dense layers.
    def hook(module, args, output):
        replay = _STATE.current
        row = layer_index
        if compact is not None and replay is not None and replay.dim() == 3 and replay.shape[1] == compact[1]:
            row = compact[0]
        positions = _split_router_output(output)
        if positions is None:
            if not getattr(module, "_unsloth_replay_warned", False):
                module._unsloth_replay_warned = True
                warnings.warn(
                    f"Unsloth: routing replay does not recognise the outputs of "
                    f"{type(module).__name__} (it routes inside the MoE block); this layer keeps "
                    f"its natural routing."
                )
            _count("router_calls_natural")
            return None
        li, wi, ii, dense = positions
        logits, weights, ids = output[li], output[wi], output[ii]
        num_experts = logits.shape[-1]
        logits2 = logits.reshape(-1, num_experts)
        ids2 = ids.reshape(-1, ids.shape[-1])
        if dense:
            scores2 = weights.reshape(logits2.shape)
            weights2 = scores2.gather(-1, ids2.long())
        else:
            weights2 = weights.reshape(ids2.shape)
        signature = None
        if torch.compiler.is_compiling():
            # The flag, not the list: its length would become a guard. An open record is pending.
            wants_signature = _STATE.record is not None or (not _STATE.in_forward and _STATE.has_pending)
        else:
            wants_signature = _STATE.pending and (_STATE.record is not None or not _STATE.in_forward)
        if wants_signature:
            signature = _input_signature(args, ids2.shape[0])
        # Read once and clear; traced with the pre hook, this is the op's runtime output.
        layer_input = _STATE.layer_inputs.get(layer_index)
        _STATE.layer_inputs[layer_index] = None
        if _STATE.in_forward:
            recomputed = None
        elif torch.compiler.is_compiling() and _STATE.has_pending \
                and getattr(module, "_unsloth_replay_formula", None) is None:
            # No forward replayed this layer: recompute as it ran, without tracing the formula probe.
            _STATE.recomputed = True
            recomputed = _NATURAL
        else:
            recomputed = _recomputed_routes(layer_index, ids2, signature, layer_input)
        if recomputed is not None:
            _count("router_calls_recomputed")
        if recomputed is _NATURAL:
            _count("router_calls_natural")
            return None
        natural = recomputed is None and (
            replay is None
            or replay.shape[0] != ids2.shape[0]
            or row >= replay.shape[1]
            or replay.shape[2] != ids2.shape[-1]
        )
        # The recompute reweights whenever another pending forward replayed this layer (compiled:
        # always, via the opaque pick); reweight here too (all rows invalid) so the non-reentrant
        # checkpoint sees identical saved tensors. Eager too: a replayed and a natural forward of the
        # same length before one backward (rows without rollout routes) hit it there as well.
        # Eager, a router without a formula yet picks it here (below), so the first checkpointed forward
        # mirrors too: with R2 that forward is natural and the later ones of the same rows are replayed.
        known = getattr(module, "_unsloth_replay_formula", None)
        mirror = natural and _STATE.record is not None and (
            isinstance(known, str) or (known is None and not torch.compiler.is_compiling())
        )
        if natural and not mirror:
            _count("router_calls_natural")
            return None

        formula = getattr(module, "_unsloth_replay_formula", None)
        if formula is None:
            # Pick the weight formula that reproduces this router's own natural weights.
            # Probabilities must never be softmaxed again: near ties would pass the check below.
            probs = _looks_like_probs(logits2)
            # A router that states whether it renormalises the top-k decides it: top-k mass near
            # one would otherwise pass a normalised formula for an unnormalised router.
            renorm = getattr(module, "norm_topk_prob", None)
            formulas = _formulas()
            for name in _formula_order(module):
                fn = formulas[name]
                if name.startswith("probs") != probs:
                    continue
                if isinstance(renorm, bool) and name.endswith("_renorm") != renorm:
                    continue
                with torch.no_grad():
                    candidate = fn(logits2, ids2).float()
                for scale in _scale_candidates(module):
                    scaled = (candidate * scale).to(weights2.dtype).float()
                    if torch.allclose(scaled, weights2.float(), rtol = 2e-2, atol = 2e-3):
                        formula = name
                        module._unsloth_replay_scale = scale
                        break
                if formula is not None:
                    break
            else:
                formula = False
                warnings.warn(
                    f"Unsloth: routing replay does not recognise the routing weights of "
                    f"{type(module).__name__}; this layer keeps its natural routing."
                )
            module._unsloth_replay_formula = formula
        if formula is False:
            _count("router_calls_natural")
            return None

        if mirror:
            # Opaque: in-graph constants are rematerialised in backward, so saved tensors would differ.
            recomputed = _NATURAL_OP(ids2)
        rollout_weights = None
        if recomputed is None:
            # Replay rows live on the input ids' device; a device map can put this layer elsewhere.
            forced = replay[:, row, :].to(ids2.device, non_blocking = True)
            valid = (forced >= 0).all(dim = -1, keepdim = True)
            forced = torch.where(valid, forced, ids2.long()).clamp_(0, num_experts - 1)
            current_weights = _STATE.current_weights
            if current_weights is not None and current_weights.shape == replay.shape:
                rollout_weights = current_weights[:, row, :].to(ids2.device, non_blocking = True)
            if _STATE.record is not None:
                _STATE.has_pending = True
                # A copy: the output ids may alias forced, and MoE blocks can reuse them in place.
                _STATE.record[layer_index] = (
                    _fingerprint(ids2), forced.clone(), valid, signature, layer_input, rollout_weights,
                )
        else:
            forced, valid = recomputed[0], recomputed[1]
            rollout_weights = recomputed[2] if len(recomputed) > 2 and _VLLM_WEIGHTS_PACKED else None
            forced = torch.where(valid, forced, ids2.long())
            if mirror:
                # Recorded too, else an identical replayed forward would donate its forced experts.
                _STATE.record[layer_index] = (
                    _fingerprint(ids2), forced.clone(), valid, signature, layer_input, None,
                )
        reweight = _formulas()[formula]
        new_weights = reweight(logits2, forced)
        scale = _replay_scale(module)
        if scale != 1.0:
            new_weights = new_weights * scale
        if formula in ("probs_renorm", "sigmoid_renorm"):
            # The forced probabilities (nearly) all underflowed: nothing to replay with, so keep the
            # natural routing (_probs_gather returns zeros there).
            valid = valid & (new_weights.sum(dim = -1, keepdim = True) > 0)
            forced = torch.where(valid, forced, ids2.long())
        weight_rows = None
        if rollout_weights is not None:
            new_weights, weight_rows = _apply_rollout_weights(
                new_weights, rollout_weights, valid, formula.endswith("_renorm"), scale,
            )
        new_weights = new_weights.to(weights2.dtype)
        new_weights = torch.where(valid, new_weights, weights2)
        with torch.no_grad():
            changed = (forced.sort(-1).values != ids2.long().sort(-1).values).any(-1).sum()
            # One counter tensor (changed selections, rows on rollout weights): a second one would add a
            # recompile when it first turns from None into a tensor.
            changed = torch.stack([changed, torch.zeros_like(changed) if weight_rows is None else weight_rows.to(changed)])
            _STATE.changed = changed if _STATE.changed is None else _STATE.changed + changed.to(_STATE.changed.device)
        # A mirrored call keeps every row natural.
        _count("router_calls_natural" if mirror else "router_calls_replayed")

        if dense:
            new_weights = torch.zeros_like(scores2).scatter(-1, forced, new_weights)
            new_weights = torch.where(valid, new_weights, scores2)

        output = list(output)
        output[ii] = forced.to(ids.dtype).reshape(ids.shape)
        output[wi] = new_weights.reshape(weights.shape)
        return tuple(output)

    def hook_with_r2(module, args, output):
        result = hook(module, args, output)
        if _STATE.r2_capture is not None and _STATE.in_forward:
            # R2: the experts this forward without grad routed to (forced or natural).
            routed = output if result is None else result
            positions = _split_router_output(routed)
            if positions is not None:
                ids = routed[positions[2]]
                _STATE.r2_capture[layer_index] = ids.detach().reshape(-1, ids.shape[-1])
        return result

    return hook_with_r2


def _input_addresses(tensors):
    # Zero padded to one slot per tensor (static shape); a tensor, not a set, to flow through a graph.
    out = sorted({x.data_ptr() for x in tensors if x.numel() > 0})
    # Host metadata read back with tolist(): pin to CPU so a default device cannot move it.
    return torch.tensor(out + [0] * (len(tensors) - len(out)), dtype = torch.int64, device = "cpu")


def _identity_mask(layer_input, recorded):
    # Forwards sharing the most addresses, or all when not exactly one. Zero is padding.
    now = set(layer_input.tolist()) - {0}
    shared = [len(now.intersection(r.tolist())) for r in recorded]
    best = max(shared, default = 0)
    unique = best > 0 and shared.count(best) == 1
    return torch.tensor([not unique or n == best for n in shared], dtype = torch.bool, device = "cpu")


_CUSTOM_OP_PARAMS = frozenset(inspect.signature(torch.library.custom_op).parameters)


def _register_op(name, impl, fake):
    for i in range(64):
        full = name + (f"_{i}" if i else "")
        # A reloaded module copy may own the name; custom_op silently re-registers (torch 2.10 segfaults).
        if hasattr(torch.ops.unsloth_zoo, full):
            continue
        # Opaque host ops: a CUDA graph replay would skip them and reuse the captured CPU output,
        # so tag them cudagraph unsafe and Inductor keeps the graph out of CUDA graphs.
        tag = getattr(torch.Tag, "cudagraph_unsafe", None)
        kwargs = dict(tags = (tag,)) if tag is not None and "tags" in _CUSTOM_OP_PARAMS else {}
        try:
            op = torch.library.custom_op(f"unsloth_zoo::{full}", mutates_args = (), **kwargs)(impl)
        except RuntimeError:
            continue
        op.register_fake(fake)
        return getattr(torch.ops.unsloth_zoo, full).default
    raise RuntimeError(f"Unsloth: could not register the routing replay op {name}.")


_LAYER_INPUTS_OP = None
_IDENTITY_MASK_OP = None
_SIGNATURE_OP = None
_RECOMPUTE_OP = None
_NATURAL_OP = None


def _layer_inputs_op():
    # data_ptr() is untraceable and fullgraph forbids breaks: opaque ops, registered eagerly. Returning
    # the addresses lets the router hook read them in-graph (not frozen at trace time) and keeps the op
    # alive. Matching is a second op: no CPU kernels for Inductor to compile inside a GPU graph.
    global _LAYER_INPUTS_OP, _IDENTITY_MASK_OP, _SIGNATURE_OP, _RECOMPUTE_OP, _NATURAL_OP
    if _LAYER_INPUTS_OP is None:
        _LAYER_INPUTS_OP = _register_op("routing_replay_input_addresses", _layer_inputs_impl, _layer_inputs_fake)
    if _IDENTITY_MASK_OP is None:
        _IDENTITY_MASK_OP = _register_op("routing_replay_identity_mask", _identity_mask_impl, _identity_mask_fake)
    if _SIGNATURE_OP is None:
        _SIGNATURE_OP = _register_op("routing_replay_input_signature", _signature_impl, _signature_fake)
    if _RECOMPUTE_OP is None:
        _RECOMPUTE_OP = _register_op("routing_replay_recomputed_routes", _recompute_impl, _recompute_fake)
    if _NATURAL_OP is None:
        _NATURAL_OP = _register_op("routing_replay_natural_routes", _natural_impl, _natural_fake)
    return _LAYER_INPUTS_OP


# typing.List, not list[...]: torch < 2.7 infer_schema rejects the builtin generic.
def _layer_inputs_impl(tensors: typing.List[torch.Tensor]) -> torch.Tensor:
    return _input_addresses(tensors)


def _layer_inputs_fake(tensors):
    return torch.empty(len(tensors), dtype = torch.int64, device = "cpu")


def _identity_mask_impl(layer_input: torch.Tensor, recorded: typing.List[torch.Tensor]) -> torch.Tensor:
    return _identity_mask(layer_input, recorded)


def _identity_mask_fake(layer_input, recorded):
    return torch.empty(len(recorded), dtype = torch.bool, device = "cpu")


def _record_layer_inputs(layer_index, args, kwargs):
    # Read in compiled layers too: dropping these leaves identical forwards ambiguous.
    tensors = [
        x for value in list(args) + list(kwargs.values())
        for x in (value if isinstance(value, (tuple, list)) else (value,))
        if torch.is_tensor(x)
    ]
    if not tensors:
        addresses = None
    elif torch.compiler.is_compiling():
        addresses = _LAYER_INPUTS_OP(tensors)
    else:
        addresses = _input_addresses(tensors)
    _STATE.layer_inputs[layer_index] = addresses


def _layer_pre_hook(layer_index):
    def hook(module, args, kwargs):
        _record_layer_inputs(layer_index, args, kwargs)
        return None

    return hook


class _RecordingCall(functools.partial):
    # transformers >= 5.23 checkpoints forward, not __call__, so recompute must record inputs
    # itself; a partial over the original so checkpoint functions that unwrap it still find the
    # layer.
    def __call__(self, *args, **kwargs):
        # Python may flatten a wrapped partial into this one: its bound keywords are ours then.
        inner = getattr(self.func, "keywords", None)
        bound = {**(inner if isinstance(inner, dict) else {}), **self.keywords, **kwargs}
        _record_layer_inputs(self.layer_index, args, bound)
        return super().__call__(*args, **kwargs)


class _RecordingCheckpoint:
    # A class, not a closure: transformers stores it on the layer, which must still pickle.
    def __init__(self, checkpoint, layer_index):
        self._unsloth_replay_checkpoint = checkpoint
        self.layer_index = layer_index

    def __call__(self, function, *args, **kwargs):
        call = _RecordingCall(function)
        call.layer_index = self.layer_index
        return self._unsloth_replay_checkpoint(call, *args, **kwargs)


def _wrap_checkpoint_funcs(layers):
    # Per call, not once: gradient_checkpointing_enable (or Unsloth's offloaded checkpoint) may set
    # the function after routing replay is installed.
    for layer_index, layer in layers:
        checkpoint = layer.__dict__.get("_gradient_checkpointing_func")
        if callable(checkpoint) and not hasattr(checkpoint, "_unsloth_replay_checkpoint"):
            layer._gradient_checkpointing_func = _RecordingCheckpoint(checkpoint, layer_index)


def _is_router(module):
    name = type(module).__name__
    return "TopKRouter" in name or "TopkRouter" in name


def install_routing_replay(model, llm = None, tokenizer = None):
    """Hook every router of `model` and the decoder's forward. Returns the number of routers."""
    if not (routing_replay_requested() or old_policy_replay_requested()) \
            or getattr(model, "_unsloth_routing_replay_installed", False):
        return 0
    passes = max(_loop_passes(getattr(m, "config", None)) for m in (model, getattr(model, "get_base_model", lambda: None)()))
    if passes > 1:
        _warn_looped(passes)
        return 0
    if llm is not None:
        wrap_vllm_generate(llm)
    if tokenizer is not None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(getattr(tokenizer, "tokenizer", None), "pad_token_id", None)
        _STATE.pad_token_id = pad_token_id

    routers = []
    decoder = None
    for name, module in model.named_modules():
        if _is_router(module):
            match = _LAYER_INDEX_RE.search(f"{name}.")
            if match is not None:
                routers.append((int(match.group(1)), module))
        elif decoder is None and isinstance(getattr(module, "layers", None), torch.nn.ModuleList) \
                and hasattr(module, "embed_tokens"):
            decoder = module
            decoder_name = name
    if not routers or decoder is None:
        warnings.warn(
            "Unsloth: routing replay found no transformers >= 5 TopKRouter modules on this "
            "model (older transformers route inside the MoE block); routing replay is off."
        )
        return 0
    moe_layers = sorted({layer_index for layer_index, _ in routers})
    dense = len(moe_layers) < len(decoder.layers)
    _STATE.num_layers = len(decoder.layers)
    _STATE.moe_layers = tuple(moe_layers)
    _STATE.r2_on = old_policy_replay_requested()
    for layer_index, module in routers:
        compact = (moe_layers.index(layer_index), len(moe_layers)) if dense else None
        module.register_forward_hook(_router_hook(layer_index, compact))
    _layer_inputs_op()
    for layer_index in moe_layers:
        if layer_index < len(decoder.layers):
            decoder.layers[layer_index].register_forward_pre_hook(_layer_pre_hook(layer_index), with_kwargs = True)
    parent_input_ids = []
    if decoder_name:
        # Qwen3.5 embeds tokens in the parent. Keep its IDs only until that forward exits.
        parent = model.get_submodule(decoder_name.rpartition(".")[0])

        def parent_pre_hook(module, args, kwargs):
            # Routes are keyed by token ids alone, so two images behind the same tokens would swap
            # routes: media forwards keep their natural routing.
            media = any(kwargs.get(k) is not None for k in _MEDIA_KWARGS)
            parent_input_ids.append(None if media else _input_ids(args, kwargs))

        def parent_post_hook(module, args, output):
            if parent_input_ids:
                parent_input_ids.pop()

        parent.register_forward_pre_hook(parent_pre_hook, with_kwargs = True)
        parent.register_forward_hook(parent_post_hook, always_call = True)
    checkpointed = tuple(
        (layer_index, decoder.layers[layer_index]) for layer_index in moe_layers if layer_index < len(decoder.layers)
    )
    decoder.register_forward_pre_hook(
        functools.partial(_decoder_pre_hook, parent_input_ids = parent_input_ids, moe_layers = checkpointed),
        with_kwargs = True,
    )
    decoder.register_forward_hook(_decoder_post_hook, always_call = True)
    model._unsloth_routing_replay_installed = True
    kinds = []
    if routing_replay_requested():
        kinds.append("rollout ids and weights" if _VLLM_WEIGHTS_PACKED else "rollout ids")
    if _STATE.r2_on:
        kinds.append("old-policy routes (R2)")
    print(f"Unsloth: MoE routing replay is on for {len(routers)} routers ({', '.join(kinds)}).")
    return len(routers)
