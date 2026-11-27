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

"""Score centering for training-inference mismatch (arXiv 2609.20807) on GRPO loss_type="reinforce".

Opt-in with UNSLOTH_GRPO_SCORE_CENTERING=k (1 / on = 32). The colocated vLLM engine returns its
top-k log-probs per generated token (the head H). The per-token term added to the REINFORCE loss is
A * sum_H sg[q w(p / q) - alpha p] log p, with alpha = rho w(1 / rho) and
rho = (1 - q(H)) / (1 - p(H)): the sampler tail is modeled as the trainer tail rescaled to the
sampler's tail mass, so only the k head log-probs of the trainer are needed. Same estimator as verl
`score_centering_correction` and miles `score_centering_loss`.

Rows are matched to their rollout by content (prompt + completion token ids, pad ids dropped), as
routing replay does, so any batch reordering still finds its head. A row with no record gets no
correction (plain IS REINFORCE) and is counted in `score_centering_stats()`.
"""

__all__ = [
    "SCORE_CENTERING_ENV",
    "score_centering_top_k",
    "score_centering_weight_fn",
    "score_centering_correction",
    "grpo_topk_head_logprobs",
    "record_rollout_topk",
    "wrap_vllm_generate_topk",
    "lookup_rollout_topk",
    "align_topk_with_mask",
    "score_centering_stats",
    "reset_score_centering",
]

import functools
import math
import os
import warnings
from collections import OrderedDict

import torch

SCORE_CENTERING_ENV = "UNSLOTH_GRPO_SCORE_CENTERING"
_DEFAULT_TOP_K = 32
# Rollouts kept for lookup: the current one plus one older (a late consumer of a split rollout).
_MAX_ROLLOUTS = 2


def score_centering_top_k():
    """Head size k from UNSLOTH_GRPO_SCORE_CENTERING: 0 = off, 1 / true / on = 32, an integer >= 2 = k."""
    value = os.environ.get(SCORE_CENTERING_ENV, "0").strip().lower()
    if value in ("", "0", "false", "off", "no"):
        return 0
    if value in ("1", "true", "on", "yes"):
        return _DEFAULT_TOP_K
    try:
        k = int(value)
    except ValueError:
        raise ValueError(f"Unsloth: {SCORE_CENTERING_ENV} must be 0, 1 or a head size k >= 2, got {value!r}.")
    if k < 2:
        raise ValueError(f"Unsloth: {SCORE_CENTERING_ENV} head size must be >= 2, got {k}.")
    return k


def score_centering_weight_fn(mode, clip_min = None, clip_max = None):
    """Token-level IS rule as a function of a ratio, shared by the head and the sampled token.

    mode is vllm_importance_sampling_mode: "*_truncate" clamps (TIS), "*_mask" zeros ratios outside
    [clip_min, clip_max] (IcePop), None gives weight 1. verl `score_centering_weight_fn` applies the
    token rule to the head for sequence-level IS too; so does this.
    """
    if mode is None:
        return torch.ones_like
    mode = str(mode)
    if mode.endswith("_truncate"):
        if clip_min is None and clip_max is None:
            return torch.ones_like
        return lambda ratio: torch.clamp(ratio, min = clip_min, max = clip_max)
    if mode.endswith("_mask"):
        low = -math.inf if clip_min is None else float(clip_min)
        high = math.inf if clip_max is None else float(clip_max)
        return lambda ratio: torch.where((ratio >= low) & (ratio <= high), ratio, torch.zeros_like(ratio))
    raise ValueError(f"Unsloth: unknown vLLM importance sampling mode {mode!r} for score centering.")


def score_centering_correction(train_head_logps, sampler_head_logps, weight_fn, head_mask = None, eps = 1e-6):
    """Per-token centering term sum_H sg[q w - alpha p] log p and the head masses.

    train_head_logps: [..., k] trainer full-vocabulary log-probs at the head ids (with gradient).
    sampler_head_logps: [..., k] sampler log-probs at the same ids. head_mask: [..., k] bool, False for
    missing slots (no record, or fewer than k entries). Returns (correction [...], q_mass, p_mass), the
    masses detached. When the head covers the vocabulary only the gradient is meaningful (as in verl).
    """
    # float32 at least (verl casts to float32); float64 inputs stay float64 for oracles.
    dtype = torch.promote_types(train_head_logps.dtype, torch.float32)
    train_head_logps = train_head_logps.to(dtype)
    sampler_head_logps = sampler_head_logps.to(dtype)
    if head_mask is None:
        head_mask = torch.ones(train_head_logps.shape, dtype = torch.bool, device = train_head_logps.device)
    head_mask = head_mask & torch.isfinite(train_head_logps)
    with torch.no_grad():
        p = torch.where(head_mask, train_head_logps.exp(), 0.0)
        q_valid = head_mask & torch.isfinite(sampler_head_logps)
        q = torch.where(q_valid, sampler_head_logps.exp(), 0.0)
        p_mass, q_mass = p.sum(-1), q.sum(-1)
        rho = (1 - q_mass).clamp_min(eps) / (1 - p_mass).clamp_min(eps)
        alpha = rho * weight_fn(1.0 / rho)
        log_ratio = torch.where(q_valid, train_head_logps - sampler_head_logps, 0.0)
        head_weights = weight_fn(log_ratio.exp())
        # q = 0 (outside the sampler's support) contributes nothing, even against an unbounded weight.
        residual = torch.where(q_valid, q * head_weights, 0.0) - alpha.unsqueeze(-1) * p
        residual = torch.where(head_mask, residual, 0.0)
    correction = (residual * torch.where(head_mask, train_head_logps, 0.0)).sum(-1)
    return correction, q_mass, p_mass


def _head_projection(lm_head, lm_head_bias):
    """The GRPO scorer's projection precision (chunked_hidden_states_selective_log_softmax): float32-output
    logits under UNSLOTH_VLLM_FP32_LM_HEAD=1, or by default for a 16-bit CUDA head; otherwise the head's own
    dtype under the caller's autocast. Returns (mode, lm_head, lm_head_bias)."""
    from unsloth_zoo.fused_losses import chunked_logprob as _chunked_logprob
    from unsloth_zoo.rocm_build import is_rocm_torch
    flag = os.environ.get("UNSLOTH_VLLM_FP32_LM_HEAD", "").strip().lower()
    # is_rocm_torch, not torch.version.hip: ROCm SDK wheels leave hip unset, and the fp32 default would then copy
    # the whole head to float32 on every call (fp32_out_linear_ok is off on ROCm).
    fp32_cuda_default = (
        flag == "" and not is_rocm_torch(torch) and torch.cuda.is_available()
        and os.environ.get("UNSLOTH_LOGPROB_FP32_LOGITS", "1") == "1"
        and hasattr(torch.ops.aten.mm, "dtype")
    )
    fp32 = flag in ("1", "true", "on", "yes") or (fp32_cuda_default and _chunked_logprob.fp32_lm_head_auto_ok(lm_head))
    if not fp32:
        return "plain", lm_head, lm_head_bias
    if _chunked_logprob.fp32_out_linear_ok(lm_head):
        return "fp32_out", lm_head, lm_head_bias
    return "fp32", lm_head.to(torch.float32), None if lm_head_bias is None else lm_head_bias.to(torch.float32)


def _head_chunk_logps(x, lm_head, lm_head_bias, ids, logit_scale_multiply, logit_scale_divide,
                      logit_softcapping, temperature, from_hidden, mode = "plain"):
    out_device = x.device
    if from_hidden:
        x = x.to(device = lm_head.device, dtype = lm_head.dtype)
        if mode == "fp32_out":
            from unsloth_zoo.fused_losses import chunked_logprob as _chunked_logprob
            with torch.autocast(device_type = x.device.type, enabled = False):
                logits = _chunked_logprob.fp32_out_linear(x, lm_head, lm_head_bias)
        elif mode == "fp32":
            # The scorer runs under bf16 / fp16 autocast, which would downcast this matmul.
            with torch.autocast(device_type = x.device.type, enabled = False):
                logits = torch.nn.functional.linear(x, lm_head, lm_head_bias)
        else:
            logits = torch.nn.functional.linear(x, lm_head, lm_head_bias)
        if logit_scale_multiply != 0.0:
            logits = logits * logit_scale_multiply
        if logit_scale_divide != 0.0:
            logits = logits / logit_scale_divide
        if logit_softcapping != 0.0:
            logits = logit_softcapping * torch.tanh(logits / logit_softcapping)
    else:
        logits = x
    logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
    if temperature != 1.0:
        logits = logits / temperature
    lse = torch.logsumexp(logits, dim = -1, keepdim = True)
    out = logits.gather(-1, ids.to(logits.device).clamp_min(0)) - lse
    return out.to(out_device)


def grpo_topk_head_logprobs(hidden_states, lm_head, head_ids, chunks = 4, logit_scale_multiply = 0.0,
                            logit_scale_divide = 0.0, logit_softcapping = 0.0, temperature = 1.0,
                            lm_head_bias = None):
    """Trainer log-probs at the sampler head ids, normalized over the full vocabulary.

    hidden_states: [R, T, H] hidden states (projected by lm_head with the same scale / softcap /
    temperature as the GRPO scorer) or [R, T, V] logits already scaled by the forward.
    head_ids: [R, T, k] (negative = missing slot, gathered as id 0 and masked by the caller).
    Each row chunk is checkpointed: forward keeps no [rows, V] logits, backward recomputes them.
    Returns [R, T, k] float32 with gradient to hidden_states (and lm_head when it trains).
    """
    R, T, k = head_ids.shape
    from_hidden = hidden_states.shape[-1] == lm_head.shape[1] if lm_head is not None else False
    flat_x = hidden_states.reshape(R * T, hidden_states.shape[-1])
    flat_ids = head_ids.reshape(R * T, k).to(torch.long)
    n_rows = flat_x.shape[0]
    if n_rows == 0:
        return hidden_states.new_zeros((R, T, k), dtype = torch.float32)
    # Skip headless positions (all slots < 0, masked by the caller); keep rows per chunk so the peak does not grow.
    keep = (flat_ids >= 0).any(dim = -1)
    rows = None
    n_keep = int(keep.sum())
    if n_keep < n_rows:
        rows = keep.nonzero().squeeze(1)
        chunks = -(-int(chunks) * n_keep // n_rows)
        flat_x, flat_ids = flat_x.index_select(0, rows), flat_ids.index_select(0, rows)
        if n_keep == 0:
            return hidden_states.new_zeros((R, T, k), dtype = torch.float32) + 0 * flat_x.sum()
    n = flat_x.shape[0]
    chunks = max(1, min(int(chunks), n))
    bounds = [(i * n) // chunks for i in range(chunks + 1)]
    needs_grad = torch.is_grad_enabled() and (
        flat_x.requires_grad or (from_hidden and lm_head.requires_grad)
        or (lm_head_bias is not None and lm_head_bias.requires_grad)
    )
    mode = "plain"
    if from_hidden:
        mode, lm_head, lm_head_bias = _head_projection(lm_head, lm_head_bias)
    outs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        args = (flat_x[a:b], lm_head, lm_head_bias, flat_ids[a:b], logit_scale_multiply, logit_scale_divide,
                logit_softcapping, temperature, from_hidden, mode)
        if needs_grad:
            from .gradient_checkpointing import _torch_checkpoint
            outs.append(_torch_checkpoint()(_head_chunk_logps, *args, use_reentrant = False))
        else:
            outs.append(_head_chunk_logps(*args))
    out = torch.cat(outs, dim = 0)
    if rows is not None:
        out = out.new_zeros((n_rows, k)).index_copy(0, rows, out)
    return out.reshape(R, T, k)


class _TopKState:
    def __init__(self):
        self.rollouts = OrderedDict()
        self.rollout_id = 0
        self.consumed_since_record = True
        self.stats = dict(recorded = 0, rows_matched = 0, rows_unmatched = 0, tokens_short_head = 0)
        self.warned_unmatched = False


_STATE = _TopKState()


def reset_score_centering():
    global _STATE
    _STATE = _TopKState()


def score_centering_stats():
    return dict(_STATE.stats)


def _make_key(token_ids, pad_token_id):
    if pad_token_id is None:
        return tuple(token_ids)
    return tuple(t for t in token_ids if t != pad_token_id)


def _position_head(position, k):
    """(ids, logps) of the k best-ranked entries of one vLLM logprob position (dict token -> Logprob).
    vLLM puts the sampled token first with its true rank; it belongs to the head only if rank <= k."""
    entries = []
    for token_id, logprob in position.items():
        rank = getattr(logprob, "rank", None)
        if rank is not None and rank <= k:
            entries.append((rank, int(token_id), float(logprob.logprob)))
    entries.sort()
    return [e[1] for e in entries[:k]], [e[2] for e in entries[:k]]


def record_rollout_topk(request_outputs, k):
    """Store vLLM's top-k head per generated token of every finished completion. Returns rows stored."""
    store = {}
    for request_output in request_outputs:
        prompt_ids = list(getattr(request_output, "prompt_token_ids", None) or [])
        for completion in getattr(request_output, "outputs", None) or []:
            positions = getattr(completion, "logprobs", None)
            token_ids = list(getattr(completion, "token_ids", None) or [])
            if positions is None or len(token_ids) == 0:
                continue
            n = min(len(token_ids), len(positions))
            ids = torch.full((len(token_ids), k), -1, dtype = torch.int32)
            logps = torch.full((len(token_ids), k), -math.inf, dtype = torch.float32)
            short = 0
            for j in range(n):
                position = positions[j]
                if position is None:
                    short += 1
                    continue
                head_ids, head_logps = _position_head(position, k)
                m = len(head_ids)
                if m < k:
                    short += 1
                if m:
                    ids[j, :m] = torch.tensor(head_ids, dtype = torch.int32)
                    logps[j, :m] = torch.tensor(head_logps, dtype = torch.float32)
            _STATE.stats["tokens_short_head"] += short + (len(token_ids) - n)
            # Keyed by every token; lookups drop pad ids on both sides (a pad id can be a real EOS).
            store[tuple(prompt_ids + token_ids)] = (ids, logps)
            _STATE.stats["recorded"] += 1
    if store:
        # A training lookup closes the rollout; generate calls before the next one join it.
        if _STATE.consumed_since_record or not _STATE.rollouts:
            _STATE.rollout_id += 1
            _STATE.rollouts[_STATE.rollout_id] = {"raw": {}, "index": {}}
            _STATE.consumed_since_record = False
        rollout = next(reversed(_STATE.rollouts.values()))
        rollout["raw"].update(store)
        rollout["index"].clear()
        while len(_STATE.rollouts) > _MAX_ROLLOUTS:
            _STATE.rollouts.popitem(last = False)
    return len(store)


def _raise_logprobs(params, k):
    # TRL asks for logprobs=0 (the sampled token); a larger request is left alone.
    items = params if isinstance(params, (list, tuple)) else [params]
    for p in items:
        if p is None or not hasattr(p, "logprobs"):
            continue
        current = getattr(p, "logprobs", None)
        if current is None or current < k:
            p.logprobs = k


def _lift_engine_max_logprobs(llm, k):
    # Same lift as unsloth's async engine: max_logprobs is checked only at add_request.
    for path in (("llm_engine", "model_config"), ("llm_engine", "vllm_config", "model_config")):
        obj = llm
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "max_logprobs"):
            limit = getattr(obj, "max_logprobs", None)
            if limit is not None and 0 <= limit < k:
                obj.max_logprobs = k


def wrap_vllm_generate_topk(llm, k):
    """Request the top-k head from every `llm.generate` call and record it. Idempotent."""
    generate = getattr(llm, "generate", None)
    if generate is None or getattr(generate, "_unsloth_score_centering", False):
        return llm
    _lift_engine_max_logprobs(llm, k)

    @functools.wraps(generate)
    def generate_with_topk_record(*args, **kwargs):
        if "sampling_params" in kwargs:
            _raise_logprobs(kwargs["sampling_params"], k)
        elif len(args) > 1:
            _raise_logprobs(args[1], k)
        outputs = generate(*args, **kwargs)
        try:
            outputs_list = outputs if isinstance(outputs, (list, tuple)) else [outputs]
            record_rollout_topk(outputs_list, k)
        except Exception as error:
            warnings.warn(f"Unsloth: score centering could not record the sampler top-k head: {error}")
        return outputs

    generate_with_topk_record._unsloth_score_centering = True
    llm.generate = generate_with_topk_record
    return llm


def _lookup(key, pad_token_id):
    for rollout in reversed(_STATE.rollouts.values()):
        raw = rollout["raw"]
        if pad_token_id is None:
            found = raw.get(key, None)
        else:
            index = rollout["index"].get(pad_token_id, None)
            if index is None:
                index = {_make_key(full, pad_token_id): full for full in raw}
                rollout["index"][pad_token_id] = index
            full = index.get(key, None)
            found = None if full is None else raw[full]
        if found is not None:
            return found
    return None


def lookup_rollout_topk(input_ids, logits_to_keep, k, pad_token_id = None):
    """Sampler heads for a TRL batch laid out as [left-padded prompt | right-padded completion].

    Returns (ids [B, logits_to_keep, k] long, -1 = missing; logps [B, logits_to_keep, k] float32;
    matched [B] bool), column j holding completion token j's head, on input_ids' device.
    """
    B = input_ids.shape[0]
    ids = torch.full((B, logits_to_keep, k), -1, dtype = torch.int32)
    logps = torch.full((B, logits_to_keep, k), -math.inf, dtype = torch.float32)
    matched = torch.zeros(B, dtype = torch.bool)
    rows = input_ids.detach().cpu().tolist()
    for b, row in enumerate(rows):
        found = _lookup(_make_key(row, pad_token_id), pad_token_id)
        if found is None:
            continue
        head_ids, head_logps = found
        n = min(head_ids.shape[0], logits_to_keep)
        kk = min(head_ids.shape[1], k)
        ids[b, :n, :kk] = head_ids[:n, :kk]
        logps[b, :n, :kk] = head_logps[:n, :kk]
        matched[b] = True
    n_matched = int(matched.sum())
    _STATE.stats["rows_matched"] += n_matched
    _STATE.stats["rows_unmatched"] += B - n_matched
    _STATE.consumed_since_record = True
    if n_matched < B and not _STATE.warned_unmatched:
        _STATE.warned_unmatched = True
        warnings.warn(
            f"Unsloth: score centering found no sampler top-k head for {B - n_matched} of {B} rows; "
            "those rows train with plain IS REINFORCE. Score centering needs the colocated vLLM engine "
            "(fast_inference = True) generating through llm.generate."
        )
    device = input_ids.device
    return ids.to(device = device, dtype = torch.long), logps.to(device), matched.to(device)


def align_topk_with_mask(tensor, attention_mask, pad_value):
    """align_logprobs_with_mask for a [B, T, k] tensor: row b starts at its first nonzero mask column."""
    device = tensor.device
    T_src = tensor.shape[1]
    B, T = attention_mask.shape
    if T_src == 0:
        return torch.full((B, T, tensor.shape[2]), pad_value, dtype = tensor.dtype, device = device)
    left = torch.argmax(attention_mask.to(torch.int32), dim = 1).to(device)
    src = torch.arange(T, device = device).unsqueeze(0) - left.unsqueeze(1)
    valid = (src >= 0) & (src < T_src)
    index = src.clamp(0, T_src - 1).unsqueeze(-1).expand(B, T, tensor.shape[2])
    aligned = tensor.gather(1, index)
    return aligned.masked_fill(~valid.unsqueeze(-1), pad_value)
