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
"""

__all__ = [
    "ROUTING_REPLAY_ENV",
    "routing_replay_requested",
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
import os
import re
import warnings
from collections import OrderedDict

import torch
import torch.nn.functional as F

ROUTING_REPLAY_ENV = "UNSLOTH_GRPO_ROUTING_REPLAY"
_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")
_MAX_ROLLOUTS = 2


def routing_replay_requested():
    return os.environ.get(ROUTING_REPLAY_ENV, "0") == "1"


def vllm_supports_routed_experts():
    try:
        from vllm import EngineArgs
    except Exception:
        return False
    try:
        return "enable_return_routed_experts" in inspect.signature(EngineArgs).parameters
    except Exception:
        return False


def _config_is_moe(config):
    for sub in (config, getattr(config, "text_config", None)):
        if sub is None:
            continue
        for name in ("num_experts", "num_local_experts", "n_routed_experts"):
            value = getattr(sub, name, None)
            if isinstance(value, int) and value > 1:
                return True
    return False


def routing_replay_engine_args(config):
    """Extra vLLM engine args, empty unless requested, supported and the model is MoE."""
    if not routing_replay_requested() or not _config_is_moe(config):
        return {}
    if not vllm_supports_routed_experts():
        warnings.warn(
            f"Unsloth: {ROUTING_REPLAY_ENV}=1 needs a vLLM with enable_return_routed_experts "
            "(0.14 or newer, 0.22 or newer for hybrid attention MoE). "
            "Routing replay is off.",
            stacklevel = 2,
        )
        return {}
    return {"enable_return_routed_experts": True}


class _ReplayState:
    def __init__(self):
        # rollout id -> [(store, unpadded)], one entry per llm.generate call of that rollout.
        self.generations = OrderedDict()
        self.generation_id = 0
        self.forwarded_since_record = True
        self.pad_token_id = None
        self.current = None
        self.stats = dict(
            recorded = 0,
            forwards = 0,
            replayed_tokens = 0,
            unmatched_tokens = 0,
            router_calls_replayed = 0,
            router_calls_natural = 0,
        )
        self.changed = None


_STATE = _ReplayState()


def reset_routing_replay():
    global _STATE
    _STATE = _ReplayState()


def routing_replay_stats():
    out = dict(_STATE.stats)
    out["changed_selections"] = 0 if _STATE.changed is None else int(_STATE.changed.item())
    return out


def _make_key(token_ids, pad_token_id):
    if pad_token_id is None:
        return tuple(token_ids)
    return tuple(t for t in token_ids if t != pad_token_id)


def record_routed_experts(request_outputs, pad_token_id = None):
    """Store vLLM's per-token expert ids for each finished completion."""
    if pad_token_id is None:
        pad_token_id = _STATE.pad_token_id
    store = {}
    unpadded = {}
    for request_output in request_outputs:
        prompt_ids = list(getattr(request_output, "prompt_token_ids", None) or [])
        for completion in getattr(request_output, "outputs", None) or []:
            routed = getattr(completion, "routed_experts", None)
            if routed is None:
                continue
            full = prompt_ids + list(completion.token_ids)
            routed = torch.as_tensor(routed).to(torch.int16)
            # vLLM never forwards the last sampled token, so it has no routing.
            rows = torch.full((len(full),) + tuple(routed.shape[1:]), -1, dtype = torch.int16)
            n = min(len(full), routed.shape[0])
            rows[:n] = routed[:n]
            # Keyed by every token: a pad id inside the sequence (pad == eos) can be real.
            key = tuple(full)
            store[key] = rows
            if pad_token_id is not None:
                unpadded[_make_key(full, pad_token_id)] = key
            _STATE.stats["recorded"] += 1
    if store:
        # A training forward closes the rollout; generate calls before the next one join it.
        if _STATE.forwarded_since_record or not _STATE.generations:
            _STATE.generation_id += 1
            _STATE.generations[_STATE.generation_id] = []
            _STATE.forwarded_since_record = False
        next(reversed(_STATE.generations.values())).append((store, unpadded, pad_token_id))
        while len(_STATE.generations) > _MAX_ROLLOUTS:
            _STATE.generations.popitem(last = False)
    return len(store)


def _lookup(key, exact = False):
    """Rows for `key`: all attended tokens when `exact`, else the tokens with pad ids dropped."""
    for rollout in reversed(_STATE.generations.values()):
        for store, unpadded, pad in reversed(rollout):
            if exact or pad is None:
                rows = store.get(key, None)
            else:
                full = unpadded.get(key, None)
                rows = None
                if full is not None:
                    rows = store[full][torch.tensor([t != pad for t in full], dtype = torch.bool)]
            if rows is not None:
                return rows
    return None


def wrap_vllm_generate(llm):
    """Record routed experts from every `llm.generate` call. Idempotent."""
    generate = getattr(llm, "generate", None)
    if generate is None or getattr(generate, "_unsloth_routing_replay", False):
        return llm

    @functools.wraps(generate)
    def generate_with_routing_record(*args, **kwargs):
        outputs = generate(*args, **kwargs)
        try:
            record_routed_experts(outputs if isinstance(outputs, (list, tuple)) else [outputs])
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


def _build_replay(input_ids, position_ids, attention_mask = None):
    if not _STATE.generations:
        return None
    pad = _STATE.pad_token_id
    ids = input_ids.detach().to("cpu")
    rows, T = ids.shape
    out = None
    replayed = unmatched = 0
    ids_list = ids.tolist()
    mask_list = None
    if torch.is_tensor(attention_mask) and tuple(attention_mask.shape) == (rows, T):
        mask_list = attention_mask.detach().to("cpu").bool().tolist()
    for row, start, end in _segments(ids, position_ids):
        tokens = ids_list[row][start:end]
        routed = None
        if mask_list is not None:
            # Padding is what the mask leaves out, so real pad id tokens keep their routing.
            cols = [start + i for i in range(end - start) if mask_list[row][start + i]]
            if cols:
                routed = _lookup(tuple(ids_list[row][c] for c in cols), exact = True)
        if routed is None:
            cols = [start + i for i, t in enumerate(tokens) if pad is None or t != pad]
            if not cols:
                continue
            routed = _lookup(_make_key(tokens, pad))
        if routed is None or routed.shape[0] != len(cols):
            unmatched += len(cols)
            continue
        if out is None:
            out = torch.full((rows * T,) + tuple(routed.shape[1:]), -1, dtype = torch.int16)
        out[torch.tensor(cols) + row * T] = routed
        replayed += len(cols)
    _STATE.stats["replayed_tokens"] += replayed
    _STATE.stats["unmatched_tokens"] += unmatched
    if out is None:
        return None
    return out.to(input_ids.device, non_blocking = True).long()


def _decoder_pre_hook(module, args, kwargs):
    input_ids = kwargs.get("input_ids", None)
    if input_ids is None and len(args) > 0 and torch.is_tensor(args[0]) and not args[0].is_floating_point():
        input_ids = args[0]
    _STATE.stats["forwards"] += 1
    _STATE.forwarded_since_record = True
    if input_ids is None or input_ids.dim() != 2:
        _STATE.current = None
        return None
    # Left set after the forward on purpose: the gradient checkpointing recompute in
    # backward must see the same routing as the forward it replays.
    _STATE.current = _build_replay(
        input_ids, kwargs.get("position_ids", None), kwargs.get("attention_mask", None),
    )
    return None


def _softmax_gather(logits, ids, renorm):
    w = F.softmax(logits.float(), dim = -1).gather(-1, ids)
    return w / w.sum(dim = -1, keepdim = True) if renorm else w


def _formulas():
    # softmax_renorm also covers gpt-oss style softmax over the top-k logits: they are equal.
    return OrderedDict(
        softmax_renorm = lambda logits, ids: _softmax_gather(logits, ids, True),
        softmax = lambda logits, ids: _softmax_gather(logits, ids, False),
    )


def _split_router_output(output):
    """(logits position, weights position, ids position) in a (logits, weights, ids) style tuple."""
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
    if len(weights_pos) != 1 or len(logits_pos) != 1:
        return None
    return logits_pos[0], weights_pos[0], ids_pos[0]


def _router_hook(layer_index):
    def hook(module, args, output):
        replay = _STATE.current
        positions = _split_router_output(output)
        if replay is None or positions is None:
            _STATE.stats["router_calls_natural"] += 1
            return None
        li, wi, ii = positions
        logits, weights, ids = output[li], output[wi], output[ii]
        num_experts = logits.shape[-1]
        logits2 = logits.reshape(-1, num_experts)
        ids2 = ids.reshape(-1, ids.shape[-1])
        weights2 = weights.reshape(ids2.shape)
        if (
            replay.shape[0] != ids2.shape[0]
            or layer_index >= replay.shape[1]
            or replay.shape[2] != ids2.shape[-1]
        ):
            _STATE.stats["router_calls_natural"] += 1
            return None

        formula = getattr(module, "_unsloth_replay_formula", None)
        if formula is None:
            # Pick the weight formula that reproduces this router's own natural weights.
            for name, fn in _formulas().items():
                with torch.no_grad():
                    candidate = fn(logits2, ids2).to(weights2.dtype).float()
                if torch.allclose(candidate, weights2.float(), rtol = 2e-2, atol = 2e-3):
                    formula = name
                    break
            else:
                formula = False
                warnings.warn(
                    f"Unsloth: routing replay does not recognise the routing weights of "
                    f"{type(module).__name__}; this layer keeps its natural routing."
                )
            module._unsloth_replay_formula = formula
        if formula is False:
            _STATE.stats["router_calls_natural"] += 1
            return None

        forced = replay[:, layer_index, :]
        valid = (forced >= 0).all(dim = -1, keepdim = True)
        forced = torch.where(valid, forced, ids2.long()).clamp_(0, num_experts - 1)
        new_weights = _formulas()[formula](logits2, forced).to(weights2.dtype)
        new_weights = torch.where(valid, new_weights, weights2)
        with torch.no_grad():
            changed = (forced.sort(-1).values != ids2.long().sort(-1).values).any(-1).sum()
            _STATE.changed = changed if _STATE.changed is None else _STATE.changed + changed
        _STATE.stats["router_calls_replayed"] += 1

        output = list(output)
        output[ii] = forced.to(ids.dtype).reshape(ids.shape)
        output[wi] = new_weights.reshape(weights.shape)
        return tuple(output)

    return hook


def _is_router(module):
    name = type(module).__name__
    return name.endswith("TopKRouter") or name.endswith("TopkRouter")


def install_routing_replay(model, llm = None, tokenizer = None):
    """Hook every router of `model` and the decoder's forward. Returns the number of routers."""
    if not routing_replay_requested() or getattr(model, "_unsloth_routing_replay_installed", False):
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
    if not routers or decoder is None:
        warnings.warn(
            "Unsloth: routing replay found no transformers >= 5 TopKRouter modules on this "
            "model (older transformers route inside the MoE block); routing replay is off."
        )
        return 0
    for layer_index, module in routers:
        module.register_forward_hook(_router_hook(layer_index))
    decoder.register_forward_pre_hook(_decoder_pre_hook, with_kwargs = True)
    model._unsloth_routing_replay_installed = True
    print(f"Unsloth: MoE routing replay is on for {len(routers)} routers.")
    return len(routers)
