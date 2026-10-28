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
import typing
import warnings
from collections import OrderedDict
from collections.abc import Sequence

import torch
import torch.nn.functional as F

ROUTING_REPLAY_ENV = "UNSLOTH_GRPO_ROUTING_REPLAY"
_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")
_MAX_ROLLOUTS = 2
_MAX_CALLS_PER_ROLLOUT = 64
# Leak guard only: evicting a forward awaiting its recompute makes that recompute route naturally.
_MAX_PENDING = 1024


def routing_replay_requested():
    return os.environ.get(ROUTING_REPLAY_ENV, "0") == "1"


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
    if _vllm_at_least(0, 21) and not _vllm_at_least(0, 22) and "async_scheduling" in _engine_arg_names():
        # vLLM 0.21 refuses routed experts capture with (default-on) async scheduling.
        args["async_scheduling"] = False
    # Fused FlashInfer MoE kernels (auto on Blackwell) pick experts internally and leave the
    # capture all zeros (vllm-project/vllm#44116); the Triton backend goes through the hook.
    if "moe_backend" in _engine_arg_names():
        args["moe_backend"] = "triton"
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
        # bool(pending) for compiled regions: reading the list there guards on its length, so a
        # compiled layer would recompile once per number of pending forwards.
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
        )
        self.changed = None
        self.warned_invalid = False
        self.warned_evicted = False


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


def _agreed(a, b):
    """Rows of two captures of the same tokens: where their expert sets differ, -1 (natural)."""
    if a.shape != b.shape:
        return a
    same = (a.sort(dim = -1).values == b.sort(dim = -1).values).all(dim = -1, keepdim = True)
    return torch.where(same, a, torch.full_like(a, -1))


def _count_conflicts(a, b):
    if a.shape != b.shape:
        return
    merged = _agreed(a, b)
    natural = lambda x: (x < 0).all(dim = -1)
    conflicts = int((natural(merged) & ~natural(a) & ~natural(b)).sum())
    if conflicts:
        _STATE.stats["conflicting_routes"] = _STATE.stats.get("conflicting_routes", 0) + conflicts


def _fill_prefix(rows, full, start, store):
    """Fill rows [0, start) from this rollout's captures sharing that prefix; disagreeing captures leave -1."""
    prefix = tuple(full[:start])
    stores = [store]
    if not _STATE.forwarded_since_record and _STATE.generations:
        # Earlier rollouts ran on older weights: their routes are not this policy's.
        stores += [call[0] for call in reversed(next(reversed(_STATE.generations.values())))]
    merged = rows[:start].clone()
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
            new = routed & ~seen[:m]
            if bool(new.any()):
                merged[:m] = torch.where(new.unsqueeze(-1), part, merged[:m])
                seen[:m] |= new
    keep = seen & ~conflict
    rows[:start] = torch.where(keep.unsqueeze(-1), merged, rows[:start])


def record_routed_experts(request_outputs, pad_token_id = None, prompt_starts = None):
    """Store vLLM's per-token expert ids per finished completion. prompt_starts: per-request
    `SamplingParams.routed_experts_prompt_start` (prompt position of the first returned row)."""
    if pad_token_id is None:
        pad_token_id = _STATE.pad_token_id
    store = {}
    unpadded = {}
    for index, request_output in enumerate(request_outputs):
        prompt_ids = list(getattr(request_output, "prompt_token_ids", None) or [])
        start = int(prompt_starts[index] or 0) if prompt_starts is not None else 0
        # vLLM 0.21 returns prompt routes on the request, generated-token routes on the completion.
        split = hasattr(request_output, "prompt_routed_experts")
        prompt_routed = getattr(request_output, "prompt_routed_experts", None)
        if prompt_routed is not None:
            prompt_routed = torch.as_tensor(prompt_routed).to(torch.int16)
        for completion in getattr(request_output, "outputs", None) or []:
            routed = getattr(completion, "routed_experts", None)
            if routed is None and split and prompt_routed is not None:
                # vLLM 0.21 returns None for an empty generation part; the prompt part stays on the request.
                routed = prompt_routed[:0]
            if routed is None:
                continue
            full = prompt_ids + list(completion.token_ids)
            routed = torch.as_tensor(routed).to(torch.int16)
            # vLLM never forwards the last sampled token, so it has no routing.
            rows = torch.full((len(full),) + tuple(routed.shape[1:]), -1, dtype = torch.int16)
            if split:
                parts = [(len(prompt_ids), routed)]
                if prompt_routed is not None and prompt_routed.shape[1:] == routed.shape[1:]:
                    parts.append((0, prompt_routed[: len(prompt_ids)]))
            else:
                parts = [(start, routed)]
            for offset, part in parts:
                n = max(0, min(len(full) - offset, part.shape[0]))
                rows[offset : offset + n] = part[:n]
            if start > 0 and not split:
                _fill_prefix(rows, full, start, store)
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
            # Keyed by every token: a pad id inside the sequence (pad == eos) can be real.
            key = tuple(full)
            if key in store:
                _count_conflicts(store[key], rows)
                rows = _agreed(store[key], rows)
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
    for rollout in reversed(_STATE.generations.values()):
        found = None
        for store, unpadded, pad in reversed(rollout):
            if exact or pad is None:
                rows = store.get(key, None)
            else:
                full = unpadded.get(key, None)
                rows = None
                if full is not None:
                    rows = store[full][torch.tensor([t != pad for t in full], dtype = torch.bool)]
            if rows is not None:
                found = rows if found is None else _agreed(found, rows)
        if found is not None:
            return found
    return None


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
    _count("replayed_tokens", replayed)
    _count("unmatched_tokens", unmatched)
    if out is None:
        return None
    return out.to(input_ids.device, non_blocking = True).long()


_MEDIA_KWARGS = ("pixel_values", "pixel_values_videos", "input_features")


def _input_ids(args, kwargs):
    input_ids = kwargs.get("input_ids", None)
    if input_ids is None and len(args) > 0 and torch.is_tensor(args[0]) and not args[0].is_floating_point():
        input_ids = args[0]
    return input_ids


def _decoder_pre_hook(module, args, kwargs, parent_input_ids = ()):
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
    if input_ids is None or input_ids.dim() != 2:
        _STATE.current = None
        return None
    # Left set after the forward on purpose: the gradient checkpointing recompute in
    # backward must see the same routing as the forward it replays.
    _STATE.current = _build_replay(
        input_ids, kwargs.get("position_ids", None), kwargs.get("attention_mask", None),
    )
    return None


def _decoder_post_hook(module, args, output):
    _STATE.in_forward = False
    _STATE.record = None
    return None


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


def _projection_impl(x: torch.Tensor) -> torch.Tensor:
    # A copy: a custom op may not return a tensor it does not own (the cached projection).
    return _signature_projection(x.shape[-1], x.device).clone()


def _projection_fake(x):
    return x.new_empty((x.shape[-1], 2), dtype = torch.float32)


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
        # A fullgraph region cannot create the seeded Generator: the opaque op builds (and caches)
        # the same projection at run time, so a cold compiled layer traces without an eager warm-up.
        projection = _PROJECTION_OP(x) if torch.compiler.is_compiling() \
            else _signature_projection(x.shape[-1], x.device)
        # Chunked so the fp32 upcast stays bounded for long sequences.
        return torch.cat([
            x[a : a + _SIGNATURE_CHUNK].float() @ projection
            for a in range(0, tokens, _SIGNATURE_CHUNK)
        ])


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
    # Traced code must not read the pending list: Dynamo would guard on its length and the layer
    # would recompile once per number of pending forwards. The pick runs in an opaque op instead;
    # it returns the natural ids, all invalid, when no forward recorded this layer.
    if not _STATE.has_pending:
        return None
    _STATE.recomputed = True
    return _RECOMPUTE_OP(ids2, signature, layer_input, layer_index)


def _recompute_impl(
    ids2: torch.Tensor, signature: typing.Optional[torch.Tensor],
    layer_input: typing.Optional[torch.Tensor], layer_index: int,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    picked = _pick_recorded(layer_index, ids2, signature, layer_input) if _STATE.pending else _NATURAL
    if picked is _NATURAL:
        return _natural_impl(ids2)
    forced, valid = picked
    return forced.long().clone(), valid.reshape(ids2.shape[0], 1).to(ids2.device).clone()


def _recompute_fake(ids2, signature, layer_input, layer_index):
    return _natural_fake(ids2)


def _natural_impl(ids2: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    # The natural ids, every row invalid: reweighting with these keeps the natural routing.
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
    return forced, valid


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


def _formulas():
    # softmax_renorm also covers gpt-oss style softmax over the top-k logits: they are equal.
    # probs_*: transformers 5.2 - 5.7 Qwen MoE routers return softmax probabilities as "logits".
    return OrderedDict(
        softmax_renorm = lambda logits, ids: _softmax_gather(logits, ids, True),
        softmax = lambda logits, ids: _softmax_gather(logits, ids, False),
        probs_renorm = lambda probs, ids: _probs_gather(probs, ids, True),
        probs = lambda probs, ids: _probs_gather(probs, ids, False),
    )


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
            # Recording a forward picks the formula first, so none of them replayed this layer:
            # recompute it as it ran (as _recomputed_routes would), without tracing the formula probe.
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
        # A compiled checkpointed forward recomputes through the opaque pick, which always reweights
        # (all rows invalid when nothing was recorded). Reweight here too, every row invalid, so the
        # forward saves the same tensors as its recompute (non-reentrant checkpoint checks them).
        # The weights and ids keep their natural values: torch.where picks the natural ones.
        mirror = natural and _STATE.record is not None and torch.compiler.is_compiling() \
            and isinstance(getattr(module, "_unsloth_replay_formula", None), str)
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
            for name, fn in _formulas().items():
                if name.startswith("probs") != probs:
                    continue
                if isinstance(renorm, bool) and name.endswith("_renorm") != renorm:
                    continue
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
            _count("router_calls_natural")
            return None

        if mirror:
            # Through an opaque op and the recompute's ops below: in-graph constants would be
            # rematerialised in the backward instead of saved, and the saved tensors would differ.
            recomputed = _NATURAL_OP(ids2)
        if recomputed is None:
            # Replay rows live on the input ids' device; a device map can put this layer elsewhere.
            forced = replay[:, row, :].to(ids2.device, non_blocking = True)
            valid = (forced >= 0).all(dim = -1, keepdim = True)
            forced = torch.where(valid, forced, ids2.long()).clamp_(0, num_experts - 1)
            if _STATE.record is not None:
                _STATE.has_pending = True
                # A copy: the output ids may alias forced, and MoE blocks can reuse them in place.
                _STATE.record[layer_index] = (
                    _fingerprint(ids2), forced.clone(), valid, signature, layer_input,
                )
        else:
            forced, valid = recomputed
            forced = torch.where(valid, forced, ids2.long())
            if mirror:
                # Recorded too (natural ids, every row invalid): a replayed forward with the same
                # natural routes and router input would otherwise be the only candidate, and this
                # forward's recompute would take its forced experts. Input addresses tell them apart.
                _STATE.record[layer_index] = (
                    _fingerprint(ids2), forced.clone(), valid, signature, layer_input,
                )
        reweight = _formulas()[formula]
        new_weights = reweight(logits2, forced)
        if formula == "probs_renorm":
            # The forced probabilities (nearly) all underflowed: nothing to replay with, so keep the
            # natural routing (_probs_gather returns zeros there).
            valid = valid & (new_weights.sum(dim = -1, keepdim = True) > 0)
            forced = torch.where(valid, forced, ids2.long())
        new_weights = new_weights.to(weights2.dtype)
        new_weights = torch.where(valid, new_weights, weights2)
        with torch.no_grad():
            changed = (forced.sort(-1).values != ids2.long().sort(-1).values).any(-1).sum()
            _STATE.changed = changed if _STATE.changed is None else _STATE.changed + changed.to(_STATE.changed.device)
        _count("router_calls_replayed")

        if dense:
            new_weights = torch.zeros_like(scores2).scatter(-1, forced, new_weights)
            new_weights = torch.where(valid, new_weights, scores2)

        output = list(output)
        output[ii] = forced.to(ids.dtype).reshape(ids.shape)
        output[wi] = new_weights.reshape(weights.shape)
        return tuple(output)

    return hook


def _input_addresses(tensors):
    # Zero padded to one slot per tensor (static shape); a tensor, not a set, to flow through a graph.
    out = sorted({x.data_ptr() for x in tensors if x.numel() > 0})
    return torch.tensor(out + [0] * (len(tensors) - len(out)), dtype = torch.int64)


def _identity_mask(layer_input, recorded):
    # Forwards sharing the most addresses, or all when not exactly one. Zero is padding.
    now = set(layer_input.tolist()) - {0}
    shared = [len(now.intersection(r.tolist())) for r in recorded]
    best = max(shared, default = 0)
    unique = best > 0 and shared.count(best) == 1
    return torch.tensor([not unique or n == best for n in shared], dtype = torch.bool)


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
_PROJECTION_OP = None
_RECOMPUTE_OP = None
_NATURAL_OP = None


def _layer_inputs_op():
    # data_ptr() is untraceable and fullgraph forbids breaks: opaque ops, registered eagerly. Returning
    # the addresses lets the router hook read them in-graph (not frozen at trace time) and keeps the op
    # alive. Matching is a second op: no CPU kernels for Inductor to compile inside a GPU graph.
    global _LAYER_INPUTS_OP, _IDENTITY_MASK_OP, _PROJECTION_OP, _RECOMPUTE_OP, _NATURAL_OP
    if _LAYER_INPUTS_OP is None:
        _LAYER_INPUTS_OP = _register_op("routing_replay_input_addresses", _layer_inputs_impl, _layer_inputs_fake)
    if _IDENTITY_MASK_OP is None:
        _IDENTITY_MASK_OP = _register_op("routing_replay_identity_mask", _identity_mask_impl, _identity_mask_fake)
    if _PROJECTION_OP is None:
        _PROJECTION_OP = _register_op("routing_replay_signature_projection", _projection_impl, _projection_fake)
    if _RECOMPUTE_OP is None:
        _RECOMPUTE_OP = _register_op("routing_replay_recomputed_routes", _recompute_impl, _recompute_fake)
    if _NATURAL_OP is None:
        _NATURAL_OP = _register_op("routing_replay_natural_routes", _natural_impl, _natural_fake)
    return _LAYER_INPUTS_OP


# typing.List, not list[...]: torch < 2.7 infer_schema rejects the builtin generic.
def _layer_inputs_impl(tensors: typing.List[torch.Tensor]) -> torch.Tensor:
    return _input_addresses(tensors)


def _layer_inputs_fake(tensors):
    return torch.empty(len(tensors), dtype = torch.int64)


def _identity_mask_impl(layer_input: torch.Tensor, recorded: typing.List[torch.Tensor]) -> torch.Tensor:
    return _identity_mask(layer_input, recorded)


def _identity_mask_fake(layer_input, recorded):
    return torch.empty(len(recorded), dtype = torch.bool)


def _layer_pre_hook(layer_index):
    def hook(module, args, kwargs):
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
        return None

    return hook


def _is_router(module):
    name = type(module).__name__
    return "TopKRouter" in name or "TopkRouter" in name


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
            decoder_name = name
    if not routers or decoder is None:
        warnings.warn(
            "Unsloth: routing replay found no transformers >= 5 TopKRouter modules on this "
            "model (older transformers route inside the MoE block); routing replay is off."
        )
        return 0
    moe_layers = sorted({layer_index for layer_index, _ in routers})
    dense = len(moe_layers) < len(decoder.layers)
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
    decoder.register_forward_pre_hook(
        functools.partial(_decoder_pre_hook, parent_input_ids = parent_input_ids), with_kwargs = True,
    )
    decoder.register_forward_hook(_decoder_post_hook, always_call = True)
    model._unsloth_routing_replay_installed = True
    print(f"Unsloth: MoE routing replay is on for {len(routers)} routers.")
    return len(routers)
