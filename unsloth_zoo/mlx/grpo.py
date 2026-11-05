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

"""MLX GRPO completions, rewards, advantages and clipped policy objective; mirrors TRL 0.24's GRPOTrainer as Unsloth
patches it on CUDA (unsloth_zoo.rl_replacements.grpo_compute_loss)."""

import copy
import hashlib
import math
import os
import sys
import warnings
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import generate as generate_module
from .generate import GenerationDefaults, GenerationRequest, SamplingParams
from .preference import (
    _VLM_TOKEN_SPAN_KEYS, _chat_template, _make_preference_cce_scorer, _preference_ids,
    _preference_logits, _response_mask, _supervised_tokens,
)
from .utils import (
    _RAW_INPUT_IDS_FOR_LABELS,
    _VLM_POSITION_GENERATING_MODEL_TYPES,
    _VLM_RELOCATABLE_SIDECAR_KEYS,
    _VLM_WIDTH_GENERATED_KEYS,
    _as_numpy_vlm_field,
    _config_get,
    _finite_batch_schedule,
    _finite_text_pad_width,
    _get_processor_tokenizer,
    _mlx_rng_key,
    _normalize_mlx_messages,
    _normalize_seed,
    _prepare_vlm_batch_for_compile,
    _preserved_preprocessing_rng,
    _restore_mlx_rng_key,
    _processor_vlm_inputs,
    _right_pad_vlm_rows,
    _to_mx_vlm_batch,
    _vlm_batch_family,
    _vlm_family_divergence,
    _vlm_family_is_plannable,
    _vlm_pipeline_disposable_keys,
    _vlm_token_aligned_sidecars,
    _vlm_width_survey,
    encode_mlx_text,
)


GRPO_LOSS_TYPES = ("grpo", "bnpo", "dr_grpo", "dapo")
GRPO_SCALE_REWARDS = ("group", "batch", "none")

# One value per micro-step, averaged over the micro-steps since the last log.
GRPO_LOSS_METRICS = (
    "kl",
    "completion_length",
    "clip_ratio/low_mean",
    "clip_ratio/high_mean",
    "clip_ratio/region_mean",
)
GRPO_STATS_WIDTH = len(GRPO_LOSS_METRICS) + 1


@dataclass(frozen=True)
class GRPOObjective:
    loss_type: str = "dapo"
    beta: float = 0.001
    epsilon: float = 0.2
    epsilon_high: float | None = None
    delta: float | None = None
    importance_sampling_level: str = "token"
    temperature: float = 1.0
    max_completion_length: int = 256

    def __post_init__(self):
        if self.loss_type not in GRPO_LOSS_TYPES:
            raise ValueError(
                f"Unsloth MLX GRPO: loss_type={self.loss_type!r} is not one of "
                f"{', '.join(GRPO_LOSS_TYPES)}, the loss types TRL 0.24 supports."
            )
        if self.importance_sampling_level not in ("token", "sequence"):
            raise ValueError(
                "Unsloth MLX GRPO: importance_sampling_level must be 'token' or "
                f"'sequence', got {self.importance_sampling_level!r}."
            )
        if self.epsilon_high is None:
            object.__setattr__(self, "epsilon_high", self.epsilon)


def _is_conversational_prompt(prompt):
    return (
        isinstance(prompt, list) and bool(prompt)
        and isinstance(prompt[0], dict) and "role" in prompt[0] and "content" in prompt[0]
    )


def cut_completion(token_ids, finish_reason, *, eos_id, pad_id):
    """TRL's completion runs through the first EOS and reports whether it was truncated; the generation engine leaves the stopping id out of token_ids."""
    ids = [int(token) for token in token_ids]
    if finish_reason == "stop" and eos_id is not None:
        ids.append(int(eos_id))
    if eos_id is not None and eos_id in ids:
        ids = ids[:ids.index(eos_id) + 1]
    truncated = not ids or ids[-1] not in (eos_id, pad_id)
    return ids, truncated


def reward_prompts_and_completions(prompts, completion_texts):
    """What TRL hands reward functions: conversational completions become one assistant message, and a prompt ending in
    an assistant turn yields that turn as the completion's opening."""
    if not prompts or not _is_conversational_prompt(prompts[0]):
        return list(prompts), list(completion_texts)
    reward_prompts, completions = [], []
    for prompt, text in zip(prompts, completion_texts):
        prompt = copy.deepcopy(prompt)
        bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
        if isinstance(bootstrap, list):  # an image prompt's text parts; TRL 0.24 fails concatenating them
            bootstrap = "".join(part.get("text", "") for part in bootstrap if part.get("type") == "text")
        reward_prompts.append(prompt)
        completions.append([{"role": "assistant", "content": bootstrap + text}])
    return reward_prompts, completions


def reward_function_name(reward_func):
    return getattr(reward_func, "__name__", type(reward_func).__name__)


def calculate_rewards(
    reward_funcs, rows, *, prompts, completions, completion_ids, trainer_state,
):
    """(len(rows), len(reward_funcs)) float32 rewards, NaN where a function returned None; functions are called as TRL calls them."""
    columns = [
        key for key in rows[0] if key not in ("prompt", "completion", "completion_ids")
    ]
    kwargs = {key: [row[key] for row in rows] for key in columns}
    kwargs["trainer_state"] = trainer_state
    rewards = np.full((len(rows), len(reward_funcs)), np.nan, dtype=np.float32)
    for index, reward_func in enumerate(reward_funcs):
        output = reward_func(
            prompts=prompts, completions=completions,
            completion_ids=completion_ids, **kwargs,
        )
        output = list(output)
        if len(output) != len(rows):
            raise ValueError(
                f"Unsloth MLX GRPO: reward function "
                f"{reward_function_name(reward_func)!r} returned {len(output)} "
                f"rewards for {len(rows)} completions."
            )
        rewards[:, index] = [np.nan if value is None else value for value in output]
    for row in np.flatnonzero(np.isnan(rewards).all(axis=1)):
        details = {key: values[row] for key, values in kwargs.items() if key != "trainer_state"}
        warnings.warn(
            "Unsloth MLX GRPO: every reward function returned None for the "
            f"completion with {details}; it scores 0.",
            stacklevel=2,
        )
    return rewards


def group_advantages(rewards_per_func, reward_weights, num_generations, scale_rewards):
    """Advantages and the reward statistics TRL logs for one generation batch."""
    rewards = np.nansum(
        rewards_per_func * np.asarray(reward_weights, dtype=np.float32)[None, :], axis=1,
    ).astype(np.float32)
    groups = rewards.reshape(-1, num_generations)
    mean = np.repeat(groups.mean(axis=1), num_generations)
    advantages = rewards - mean
    if scale_rewards in ("group", "none"):
        std = np.repeat(groups.std(axis=1, ddof=1), num_generations)
    elif scale_rewards == "batch":
        std = np.full_like(rewards, rewards.std(ddof=1))
    else:
        raise ValueError(
            f"Unsloth MLX GRPO: scale_rewards must be one of "
            f"{', '.join(GRPO_SCALE_REWARDS)}, got {scale_rewards!r}."
        )
    std = std.astype(np.float32)
    if scale_rewards != "none":
        advantages = advantages / (std + np.float32(1e-4))
    return advantages.astype(np.float32), {
        "reward": float(mean.mean()),
        "reward_std": float(std.mean()),
        "frac_reward_zero_std": float(np.isclose(std, 0.0, rtol=1e-5, atol=1e-8).mean()),
    }


def scalerl_sample_weights(rewards, num_generations, lengths, exclude, prompt_level):
    """Unsloth's ScaleRL per-completion loss weights for one generation, and the fraction of groups excluded: zero-variance
    exclusion drops groups of equal rewards from the loss and its token count; prompt-level aggregation averages tokens
    within each group, then over groups."""
    groups = rewards.reshape(-1, num_generations)
    keep = ~np.isclose(groups.std(axis=1, ddof=1), 0.0) if exclude else np.ones(len(groups), dtype=bool)
    rows = np.repeat(keep, num_generations).astype(np.float64)
    tokens = (lengths * rows).reshape(-1, num_generations).sum(axis=1)
    if prompt_level:
        weights = rows / (np.repeat(np.maximum(tokens, 1.0), num_generations) * max(int(keep.sum()), 1))
    else:
        weights = rows / max(float(tokens.sum()), 1.0)
    return weights.astype(np.float32), float(1.0 - keep.mean())


def completion_logps(model, batch, temperature):
    """Per-token log probabilities of the batch's ids after the first, on logits divided by the sampling temperature as TRL scores them."""
    logits = _preference_logits(model, batch).astype(mx.float32)
    if temperature != 1.0:
        logits = logits / temperature
    targets = _preference_ids(batch)[:, 1:]
    # An image placeholder may lie outside the head (Mllama's) or be negative (Phi-3 Vision); none is scored.
    targets = mx.where((targets >= 0) & (targets < logits.shape[-1]), targets, 0)
    return -nn.losses.cross_entropy(logits, targets, reduction="none").reshape(targets.shape)


def _shared_prefix_hidden(model, batch, prefix):
    """_preference_hidden's states, zero over prefix: each group's leading tokens, run once for its consecutive rows."""
    from .utils import _PrefixKVSlot, _get_text_backbone

    backbone = _get_text_backbone(model)
    ids = _preference_ids(batch)
    groups, shared = prefix.shape
    recorded = [_PrefixKVSlot() for _ in backbone.layers]
    backbone(prefix, cache=recorded)
    extended = [_PrefixKVSlot(slot.recorded(), ids.shape[0] // groups, interleaved=True) for slot in recorded]
    hidden = backbone(ids[:, shared:-1], cache=extended)
    return mx.concatenate([mx.zeros((ids.shape[0], shared, hidden.shape[-1]), hidden.dtype), hidden], axis=1)


def grpo_logps(temperature, scorer=None):
    """(model, batch, lengths, indices=None, prefix=None) -> log-probs: the logits path, or scorer's runtime CCE with its
    own temperature over the flat positions in indices, through the shared prompt prefix; zero outside the response."""
    if scorer is None:
        return lambda model, batch, _lengths, _indices=None, _prefix=None: completion_logps(model, batch, temperature)

    def logps(model, batch, lengths, indices=None, prefix=None):
        supervised = _response_mask(_preference_ids(batch)[:, 1:], lengths) > 0
        shared = {} if prefix is None else {"hidden": _shared_prefix_hidden(model, batch, prefix)}
        if indices is None:
            return -scorer(model, batch, supervised, logit_sums=False, **shared)[0]
        return -scorer(model, batch, supervised, indices, logit_sums=False, **shared)[0]

    return logps


def grpo_cce_capacity(rows, width, min_prompt, max_completion):
    """Most completion positions a micro-batch at width holds, rounded up to 256 (a pure function of the shape, so no new compile key); None when that saves nothing."""
    count = rows * max(0, min(width - min_prompt, max_completion))
    capacity = max(256, -(-count // 256) * 256)
    return capacity if capacity < rows * (width - 1) else None


def grpo_cce_indices(lengths, width, capacity):
    """Flat ``rows * (width - 1)`` positions predicting completion tokens, padded with -1."""
    selected = [
        row * (width - 1) + step - 1
        for row, (prompt, end) in enumerate(lengths) for step in range(max(1, int(prompt)), int(end))
    ]
    indices = np.full(capacity, -1, dtype=np.int32)
    indices[:len(selected)] = selected
    return indices


def _scored_as_step(logps, step_compiled):
    """Score as the training step's forward, which MLX runs differently under its trace."""

    toggles, compiled, failed = {}, {}, []

    def traced(model, batch, lengths, indices=None, prefix=None):
        if step_compiled() and not failed:
            # Compiled like the step: its fused kernels differ from the eager ones in some vision towers (Qwen3.5's).
            # A trace keeps its model and the Python numbers it read, such as the adapter scales a reference turns
            # off, so each model and setting of them gets its own; the random state is threaded, as in the step.
            if id(model) not in toggles:
                toggles[id(model)] = [
                    (module, name) for _, module in model.named_modules()
                    for name in ("scale", "_neftune_noise_enabled")
                    if isinstance(getattr(module, name, None), (bool, int, float))
                ]
            key = (id(model), *(getattr(module, name) for module, name in toggles[id(model)]))
            rng = _mlx_rng_key()
            try:
                if key not in compiled:
                    run = lambda batch, lengths, indices, prefix: mx.value_and_grad(
                        lambda anchor: (anchor, logps(model, batch, lengths, indices, prefix)))(mx.array(0.0))[0][1]
                    compiled[key] = mx.compile(run, inputs=[model.state, mx.random.state], outputs=[mx.random.state])
                return compiled[key](batch, lengths, indices, prefix)
            except Exception as error:
                # Scoring runs while a batch is fetched, outside the step's compile fallback: score eagerly instead.
                failed.append(error)
                _restore_mlx_rng_key(rng)
                warnings.warn(f"Unsloth: compiling GRPO scoring failed ({error}); scoring eagerly.", RuntimeWarning)
        run = lambda: (mx.array(0.0), logps(model, batch, lengths, indices, prefix))
        return nn.value_and_grad(model, run)()[0][1]

    return traced


def _masked_mean(values, mask):
    if values.shape[1] == 1:
        return values.mean()
    return (values * mask).sum() / mx.maximum(mask.sum(), mx.array(1.0))


def grpo_terms(objective, new, advantages, mask, *, old=None, ref=None):
    """Unsloth's ``grpo_compute_loss`` per-token loss, KL and importance weight."""
    keep = mask > 0
    zero = mx.array(0.0, mx.float32)
    new = mx.where(keep, new, zero)
    old = None if old is None else mx.where(keep, old, zero)
    ref = None if ref is None else mx.where(keep, ref, zero)
    advantages = advantages[:, None]
    log_ratio = new - (mx.stop_gradient(new) if old is None else old)
    if objective.importance_sampling_level == "sequence":
        log_ratio = ((log_ratio * mask).sum(-1) / mx.maximum(mask.sum(-1), mx.array(1.0)))[:, None]
    coef_1 = mx.exp(log_ratio)
    coef_2 = mx.clip(coef_1, 1 - objective.epsilon, 1 + objective.epsilon_high)
    capped = coef_1 if objective.delta is None else mx.minimum(coef_1, mx.array(objective.delta, mx.float32))
    per_token = -mx.minimum(capped * advantages, coef_2 * advantages)
    if objective.beta != 0.0:
        log_ratio = mx.clip(ref - new, -20.0, 20.0)
        kl = mx.clip(mx.expm1(log_ratio) - log_ratio, -10.0, 10.0)
        per_token = per_token + objective.beta * kl
    else:
        kl = mx.zeros((new.shape[0], 1) if coef_1.shape[1] == 1 else new.shape, mx.float32)
    return per_token, kl, coef_1


def _reduce(objective, per_token, mask, normalizers, weights=None):
    """k times Unsloth's micro-step loss, k the window's micro-step count (it cancels); the trainer averages micro-step
    gradients while TRL sums them; sample weights replace every loss type's normalizer, as Unsloth's do."""
    steps_per_generation, generation_tokens = normalizers[0], normalizers[1]
    masked = per_token * mask
    if weights is not None:
        return steps_per_generation * (masked * weights[:, None]).sum()
    if objective.loss_type == "grpo":
        return (masked.sum(-1) / mx.maximum(mask.sum(-1), mx.array(1.0))).mean()
    if objective.loss_type == "dr_grpo":
        return masked.sum() / (per_token.shape[0] * objective.max_completion_length)
    if objective.loss_type == "bnpo":
        return masked.sum() / mx.maximum(mask.sum(), mx.array(1.0))
    return steps_per_generation * masked.sum() / mx.maximum(generation_tokens, mx.array(1.0))


def grpo_stats(objective, kl, coef_1, advantages, mask):
    """Metric values over the loss's own mask: a truncated row TRL masked counts no tokens."""
    rows = mask.sum(-1)
    if kl.shape[1] == 1:
        mean_kl = kl.mean()
    else:
        row_kl = (kl * mask).sum(-1) / mx.maximum(rows, mx.array(1.0))
        mean_kl = row_kl.sum() / mx.maximum((rows > 0).sum(), mx.array(1))
    advantages = advantages[:, None]
    low = mx.logical_and(coef_1 < 1 - objective.epsilon, advantages < 0).astype(mx.float32)
    high = mx.logical_and(coef_1 > 1 + objective.epsilon_high, advantages > 0).astype(mx.float32)
    region = mx.logical_or(low > 0, high > 0).astype(mx.float32)
    values = [
        mean_kl,
        rows.mean(),
        _masked_mean(low, mask),
        _masked_mean(high, mask),
        _masked_mean(region, mask),
        mx.array(1.0, mx.float32),
    ]
    return mx.stop_gradient(mx.stack([value.astype(mx.float32) for value in values]))


def grpo_metric_values(summed):
    count = summed[len(GRPO_LOSS_METRICS)]
    values = {name: summed[index] / count for index, name in enumerate(GRPO_LOSS_METRICS)}
    # One process: TRL's min / max over gathered processes is the value itself.
    values["clip_ratio/low_min"] = values["clip_ratio/low_mean"]
    values["clip_ratio/high_max"] = values["clip_ratio/high_mean"]
    return values


def make_grpo_loss_fn(objective, *, scorer=None):
    """GRPO loss over a micro-batch (tokens, lengths, normalizers, advantages, kept, old_logps, ref_logps, cce_indices, sample weights, shared prompt ids); kept zeroes rows TRL masks as truncated."""
    logps = grpo_logps(objective.temperature, scorer)

    def loss_fn(
        model, batch, lengths, normalizers, advantages, kept, old=None, ref=None, cce_indices=None, weights=None,
        prefix=None,
    ):
        new = logps(model, batch, lengths, cce_indices, prefix)
        mask = _response_mask(_preference_ids(batch)[:, 1:], lengths) * kept[:, None].astype(mx.float32)
        per_token, kl, coef_1 = grpo_terms(objective, new, advantages, mask, old=old, ref=ref)
        loss = _reduce(objective, per_token, mask, normalizers, weights)
        return loss, mx.array(1, dtype=mx.int32), grpo_stats(objective, kl, coef_1, advantages, mask)

    loss_fn._unsloth_supervised_tokens = _supervised_tokens
    loss_fn._unsloth_preference_metrics = GRPO_LOSS_METRICS
    loss_fn._unsloth_preference_stats_width = GRPO_STATS_WIDTH
    loss_fn._unsloth_preference_report = grpo_metric_values
    loss_fn._unsloth_cce_compaction = bool(getattr(scorer, "compaction", False))
    loss_fn._unsloth_grpo_scorer = scorer
    if scorer is not None:
        loss_fn._unsloth_cce_backend = "runtime-cce"
    return loss_fn


def make_grpo_cce_loss_fn(model, objective):
    """The GRPO loss scored through runtime CCE when the head allows it."""
    return make_grpo_loss_fn(objective, scorer=_make_preference_cce_scorer(
        model, temperature=objective.temperature,
    ))



def make_grpo_eval_fn(objective, *, scorer=None):
    """The GRPO loss of one eval batch, weighted by its rows as HF's ``eval_loss``."""
    loss_fn = make_grpo_loss_fn(objective, scorer=scorer)

    def eval_fn(model, batch, *rest):
        loss, _, stats = loss_fn(model, batch, *rest)
        return loss, mx.array(_preference_ids(batch).shape[0], dtype=mx.int32), stats

    eval_fn.__dict__.update(loss_fn.__dict__)
    return eval_fn

_TEMPLATE_RESERVED_COLUMNS = frozenset((
    "prompt", "chosen", "rejected", "completion", "messages", "label",
    "images", "image", "videos", "video", "audios", "audio",
))


def _grpo_images(row):
    """A row's images as TRL 0.24 reads them; an image cell may hold one image or a list, as Unsloth allows."""
    cell = row.get("images") if "images" in row else row.get("image")
    if cell is None:
        return ()
    return tuple(cell) if isinstance(cell, (list, tuple)) else (cell,)


def _trl_multimodal_messages(messages, num_images):
    """TRL 0.24's ``prepare_multimodal_messages``, on a copy."""
    structured, placed = [], False
    for message in messages:
        message = dict(message)
        if message["role"] not in ("system", "user", "assistant"):
            raise ValueError(
                f"Unsloth MLX GRPO: invalid role in message: {message['role']}. "
                "Expected 'user', 'assistant', or 'system'."
            )
        if isinstance(message["content"], str):
            parts = [{"type": "text", "text": message["content"]}]
            if message["role"] == "user" and not placed:
                parts, placed = [{"type": "image"}] * num_images + parts, True
            message["content"] = parts
        structured.append(message)
    return structured


def render_grpo_prompt(tokenizer, row, *, num_images=None, processor=None):
    """TRL's prompt text; a string column the chat template names is passed to it as a keyword; a vision dataset supplies num_images, and a VLM's processor renders."""
    prompt = row["prompt"]
    if not _is_conversational_prompt(prompt):
        if not isinstance(prompt, str):
            raise ValueError(
                "Unsloth MLX GRPO: a prompt must be a string or a list of chat messages."
            )
        return prompt
    renderer = tokenizer if processor is None else processor
    template = getattr(renderer, "chat_template", None)
    kwargs = {
        key: value for key, value in row.items()
        if isinstance(value, str) and key not in _TEMPLATE_RESERVED_COLUMNS
        and isinstance(template, str) and key in template
    }
    if num_images is not None:
        prompt = _trl_multimodal_messages(prompt, num_images)
    messages = _normalize_mlx_messages(prompt, is_vlm=processor is not None)
    role = messages[-1].get("role")
    if role == "user":
        mode = {"add_generation_prompt": True}
    elif role == "assistant":
        mode = {"continue_final_message": True}
    else:
        raise ValueError(
            "Unsloth MLX GRPO: a conversational prompt must end with a user or "
            "assistant message."
        )
    return _chat_template(
        renderer, messages, tools=None, kwargs=kwargs, is_vlm=processor is not None, **mode,
    )


def _encode_grpo_prompt(tokenizer, row, processor=None, vision=False):
    """``(ids, text, images, vision inputs)``: an image prompt through the processor."""
    images = _grpo_images(row) if vision else ()
    text = render_grpo_prompt(
        tokenizer, row, num_images=len(images) if vision else None, processor=processor,
    )
    if not images:
        ids = encode_mlx_text(tokenizer, text, add_special_tokens=False)
        return tuple(int(token) for token in ids), text, images, None
    inputs = _processor_vlm_inputs(
        processor, [text], [list(images)], None, truncation=False, padding_side="right",
    )
    ids = tuple(int(token) for token in _as_numpy_vlm_field(inputs, "input_ids")[0])
    return ids, text, images, {
        key: value for key, value in inputs.items()
        if key not in ("input_ids", "attention_mask") and not key.startswith("_unsloth_")
    }


def _grpo_vision_batch(processor, model_config, batch, lengths, media):
    """Packed rows as one vision batch from the processor's outputs for their media; token-aligned outputs are zero over
    each completion as TRL extends token_type_ids; image-less rows still form a dict since some vision models take no
    bare ids."""
    outputs = {"input_ids": batch}
    if media is not None and any(images for _, images in media):
        # Some processors (falcon, deepseek_vl_v2) left-pad whatever side is asked for.
        outputs = _right_pad_vlm_rows(_processor_vlm_inputs(
            processor, [text for text, _ in media], [list(images) for _, images in media], None,
            truncation=False, padding_side="right",
        ), processor)
    ids = _as_numpy_vlm_field(outputs, "input_ids")
    width = batch.shape[1]
    if any(
        ids[row, :prompt_length].tolist() != batch[row, :prompt_length].tolist()
        for row, (prompt_length, _) in enumerate(lengths)
    ):
        raise ValueError(
            "Unsloth MLX GRPO: the processor encodes a prompt differently beside "
            "others than alone, so its rollouts and scores would see different tokens."
        )
    inputs = {
        "input_ids": batch,
        "attention_mask": (np.arange(width)[None, :] < lengths[:, 1:]).astype(np.int32),
    }
    for name, values in _vlm_token_aligned_sidecars(outputs, ids.shape).items():
        inputs[name] = np.zeros(batch.shape, dtype=values.dtype)
        for row, (prompt_length, _) in enumerate(lengths):
            inputs[name][row, :prompt_length] = values[row, :prompt_length]
    rebuilt = (
        _VLM_WIDTH_GENERATED_KEYS
        if _config_get(model_config, "model_type") in _VLM_POSITION_GENERATING_MODEL_TYPES else ()
    )
    for key, value in outputs.items():
        if (
            key in ("input_ids", "attention_mask", "labels") or key in _VLM_RELOCATABLE_SIDECAR_KEYS
            or key in rebuilt or key.startswith("_unsloth_")
        ):
            continue
        shape = getattr(value, "shape", None)
        if key in _VLM_TOKEN_SPAN_KEYS or (
            shape is not None and len(shape) >= 2 and tuple(shape[:2]) == ids.shape
        ):
            raise NotImplementedError(
                f"Unsloth MLX GRPO: {type(processor).__name__} returns {key!r} aligned "
                "with the prompt as it laid it out, which a rollout re-lays. This "
                "model is not supported for GRPO on image prompts."
            )
        inputs[key] = value
    vision = _to_mx_vlm_batch(inputs)
    vision.pop(_RAW_INPUT_IDS_FOR_LABELS, None)
    vision = _prepare_vlm_batch_for_compile(vision, model_config)
    if tuple(vision["input_ids"].shape) != tuple(batch.shape):
        raise NotImplementedError(
            "Unsloth MLX GRPO: this model expands its image tokens after the "
            "processor, which moves the completion. It is not supported for GRPO "
            "on image prompts."
        )
    return vision


@dataclass(frozen=True)
class GRPOSampling:
    """Rollout sampling as TRL configures it; top_k and min_p None switch that filter off, and max_tokens is the resolved completion length."""

    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float = 1.0

    def __post_init__(self):
        valid = {
            "max_tokens": self.max_tokens >= 1,
            "temperature": self.temperature > 0,
            "top_p": 0 <= self.top_p <= 1,
            "top_k": self.top_k is None or self.top_k >= 0,
            "min_p": self.min_p is None or 0 <= self.min_p <= 1,
            "repetition_penalty": self.repetition_penalty > 0,
        }
        for name, ok in valid.items():
            if not ok:
                raise ValueError(
                    f"Unsloth MLX GRPO: {name}={getattr(self, name)!r} is out of "
                    "range for sampling."
                )

    @property
    def warps(self):
        return (
            self.temperature != 1.0 or self.top_p < 1.0 or bool(self.top_k)
            or bool(self.min_p) or self.repetition_penalty != 1.0
        )


def hf_logits_warper(sampling, prompt_ids):
    """HF sampling processors for one row: repetition penalty over the prompt and what followed it, then temperature, top-k, top-p and min-p."""
    # The engine shows a processor its last prompt token onward.
    context = mx.array(list(prompt_ids[:-1]), dtype=mx.int32)
    penalty = float(sampling.repetition_penalty)

    def warp(tokens, logits):
        logits = logits.astype(mx.float32)
        if penalty != 1.0:
            vocab = logits.shape[-1]
            seen = mx.concatenate([context, tokens.astype(mx.int32)])
            # Placeholders outside the head land on a dropped column.
            seen = mx.where((seen >= 0) & (seen < vocab), seen, vocab)
            logits = mx.concatenate([logits, mx.zeros_like(logits[:, :1])], axis=-1)
            picked = logits[:, seen]
            logits[:, seen] = mx.where(picked < 0, picked * penalty, picked / penalty)
            logits = logits[:, :vocab]
        if sampling.temperature != 1.0:
            logits = logits / sampling.temperature
        removed = mx.array(-float("inf"), mx.float32)
        vocab = logits.shape[-1]
        if sampling.top_k:
            kept = vocab - min(int(sampling.top_k), vocab)
            cutoff = mx.sort(logits, axis=-1)[:, kept:kept + 1]
            logits = mx.where(logits < cutoff, removed, logits)
        if sampling.top_p < 1.0:
            # Drops by sorted position as HF does, since a value cutoff keeps every tie; the most likely token always survives, as HF's min_tokens_to_keep=1.
            order = mx.argsort(logits, axis=-1)
            ordered = mx.take_along_axis(logits, order, axis=-1)
            below = mx.cumsum(mx.softmax(ordered, axis=-1), axis=-1) <= 1 - sampling.top_p
            below = mx.logical_and(below, mx.arange(vocab) < vocab - 1)
            dropped = mx.take_along_axis(below, mx.argsort(order, axis=-1), axis=-1)
            logits = mx.where(dropped, removed, logits)
        if sampling.min_p:
            probs = mx.softmax(logits, axis=-1)
            floor = sampling.min_p * mx.max(probs, axis=-1, keepdims=True)
            logits = mx.where(probs < floor, removed, logits)
        return logits

    return warp


def _derived_seed(*parts):
    return int(np.random.SeedSequence([int(part) % (2 ** 32) for part in parts])
               .generate_state(1, np.uint64)[0])


class _RepeatSamplerOrder:
    """TRL's RepeatSampler order: one generator seeded once, one permutation per epoch."""

    def __init__(self, size, seed, shuffle):
        self._size = size
        self._orders = []
        self._draw = None
        if not shuffle:
            return
        try:
            import torch
        except ImportError:
            warnings.warn(
                "Unsloth MLX GRPO: torch is not installed, so prompts are shuffled "
                "with numpy instead of TRL's torch.randperm; the order is seeded but "
                "differs from CUDA's.",
                stacklevel=3,
            )
            rng = np.random.default_rng(seed)
            self._draw = lambda: rng.permutation(size).tolist()
        else:
            generator = torch.Generator()
            generator.manual_seed(seed)
            self._draw = lambda: torch.randperm(size, generator=generator).tolist()

    def __call__(self, epoch):
        while len(self._orders) <= epoch:
            self._orders.append(
                list(range(self._size)) if self._draw is None else self._draw()
            )
        return self._orders[epoch]


def grpo_row_layout(order, block, key, classes=None):
    """order with each run of block rows stably sorted by key; a block is the rows of micro-steps that always share an
    optimizer step, so every step trains the same rows."""
    if classes is None:
        return np.concatenate([
            sorted(order[start:start + block], key=key) for start in range(0, len(order), block)
        ]).astype(order.dtype)
    laid = order.copy()
    for start in range(0, len(order), block):
        chunk = order[start:start + block]
        for kind in dict.fromkeys(classes[row] for row in chunk):
            positions = [start + i for i, row in enumerate(chunk) if classes[row] == kind]
            laid[positions] = sorted(order[positions], key=key)
    return laid


def _image_class(prompt, images):
    # Images alike in type, size and mode come out of the processor in the same shapes; others stay with their prompt.
    shapes = tuple((type(image).__name__, getattr(image, "shape", None) or getattr(image, "size", None),
                    getattr(image, "mode", None)) for image in images)
    return ("prompt", prompt) if any(shape is None for _, shape, _ in shapes) else shapes


def _media_signature(inputs, length):
    """One prompt's processor outputs as a batch's compile family sees them: array shapes and dtypes, integer values
    (grids, sizes) and other metadata, without token-aligned outputs, which batches rebuild; None if unkeyable."""
    def key(value):
        if isinstance(value, (list, tuple)):
            return tuple(key(item) for item in value)
        if isinstance(value, Mapping):
            return tuple(sorted((name, key(item)) for name, item in value.items()))
        if not hasattr(value, "shape"):
            return value
        shape, dtype = tuple(int(size) for size in value.shape), str(value.dtype)
        exact = "int" in dtype or "bool" in dtype
        return shape, dtype, hashlib.blake2b(np.asarray(value).tobytes(), digest_size=8).digest() if exact else None
    skipped = {"labels", *_VLM_RELOCATABLE_SIDECAR_KEYS, *_VLM_WIDTH_GENERATED_KEYS}
    try:
        signature = tuple(sorted(
            (name, key(value)) for name, value in inputs.items()
            if name not in skipped and tuple(getattr(value, "shape", ())) != (1, length)
        ))
        hash(signature)
    except TypeError:
        return None
    return signature


@dataclass
class _Generation:
    visit: int
    sequences: list
    prompt_lengths: np.ndarray
    advantages: np.ndarray
    kept: np.ndarray
    old: list | None
    ref: list | None
    generation_tokens: int
    media: list | None = None  # (prompt text, images) per row
    skipped: dict | None = None
    weights: np.ndarray | None = None
    eager: bool = False


class GRPOBatchPlan:
    """Finite GRPO schedule with one entry per micro-step naming its prompt chunk; a generation covers
    steps_per_generation * num_iterations micro-steps from start_at, so a resumed run generates at once, as TRL does
    with an empty buffer."""

    def __init__(
        self, dataset, tokenizer, prompt_ids, max_tokens, schedule, *, cycle_length,
        num_generations, batch_size, steps_per_generation, num_iterations,
        sampling, score_temperature, scale_rewards, mask_truncated_completions,
        with_old_logps, with_reference, max_seq_length, seed, counts_tokens=True,
        processor=None, model_config=None, vision=False, group_by_length=False, layout_block=1,
        skip_zero_advantage=False, sample_weighting=None, media_signatures=None, prefix_grouping=False,
    ):
        self._dataset = dataset
        self._media_signatures = media_signatures
        self._tokenizer = tokenizer
        self._processor = processor
        self._model_config = model_config
        self.vision = vision
        self._prompt_ids = prompt_ids
        self._max_tokens = max_tokens
        self._schedule = schedule
        self._cycle_length = cycle_length
        self.num_generations = num_generations
        self.batch_size = batch_size
        self.steps_per_generation = steps_per_generation
        self._repeats = steps_per_generation * num_iterations
        self.sampling = sampling
        self.score_temperature = score_temperature
        self.scale_rewards = scale_rewards
        self.mask_truncated_completions = mask_truncated_completions
        self.with_old_logps = with_old_logps
        self.with_reference = with_reference
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.counts_tokens = counts_tokens
        self.group_by_length = group_by_length
        self.skip_zero_advantage = skip_zero_advantage
        self.sample_weighting = sample_weighting
        self._prefix = prefix_grouping
        self._prefix_tokr = float(os.environ.get("UNSLOTH_GRPO_PREFIX_GROUPER_TOKR", "1.3"))
        self._layout_rows = batch_size * layout_block
        self.eos_id = int(tokenizer.eos_token_id)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        self.pad_id = self.eos_id if pad_id is None else int(pad_id)
        self._anchor = 0
        self._current = None
        self._model = None
        self._shape_plan = None
        self._cce_bounds = None
        self._families = None  # per schedule entry, once a compiled vision run surveys them
        self._forbidden = frozenset()
        self._family_drift = None
        self._eager_generation = False
        self._metrics = {}
        self.completions = None

    @property
    def schedule(self):
        return self._schedule

    @property
    def cycle_length(self):
        return self._cycle_length

    def __len__(self):
        return len(self._schedule)

    def batch_index_for_visit(self, absolute_visit):
        return int(absolute_visit) % len(self._schedule)

    def bind(
        self, model, reward_funcs, *, trainer_state, reward_weights=None,
        reference_policy=None, rollout_context=None, keep_completions=False, scorer=None,
        step_compiled=None,
    ):
        """Attach the policy, rewards and reference; trainer_state reaches reward functions and counts prompt and
        completion tokens as TRL does; keep_completions holds the latest texts; scorer is the loss's runtime CCE, so
        old and reference log-probs are scored as the loss scores."""
        if self.with_reference and reference_policy is None:
            raise ValueError("Unsloth MLX GRPO: this plan scores a reference policy.")
        if self._prefix and scorer is None:
            raise ValueError("Unsloth MLX GRPO: a shared-prompt plan needs the loss's runtime CCE scorer.")
        self._model = model
        self._reward_funcs = list(reward_funcs)
        self._reward_weights = (
            [1.0] * len(self._reward_funcs) if reward_weights is None
            else [float(weight) for weight in reward_weights]
        )
        self._reference_policy = reference_policy
        self._trainer_state = trainer_state
        self._rollout_context = rollout_context or nullcontext
        self._keep_completions = keep_completions
        self._logps = grpo_logps(self.score_temperature, scorer)
        self._step_compiled = step_compiled or (lambda: False)
        if step_compiled is not None:
            self._logps = _scored_as_step(self._logps, lambda: step_compiled() and not self._eager_generation)

    def start_at(self, visit):
        """Count generations from ``visit``, the first micro-step a run fetches."""
        self._anchor = int(visit)
        self._current = None

    def _generation_start(self, index):
        return self._anchor + ((index - self._anchor) // self._repeats) * self._repeats

    def _width(self, raw):
        return _finite_text_pad_width(
            raw, pad_to_multiple=32, minimum_width=2, max_seq_length=self.max_seq_length,
        )

    def width_rungs(self, index):
        """Every padded width a micro-step of this entry's generation can take."""
        chunk = self._schedule[self._generation_start(index) % len(self._schedule)]
        low = self._width(min(len(self._prompt_ids[prompt]) for prompt in chunk) + 1)
        high = self._width(max(
            len(self._prompt_ids[prompt]) + self._max_tokens[prompt] for prompt in chunk
        ))
        return tuple(sorted({self._clear_of_vision_extents(width) for width in (*range(low, high, 32), high)}))

    def _clear_of_vision_extents(self, width):
        # A width equal to an unpadded vision extent would reclassify the batch's family.
        while self._families is not None and width in self._forbidden:
            width += 1
        return width

    def _media_class(self, prompt, images):
        # Equal output signatures give a batch's family its same per-row shapes, unless batches record image token positions.
        if _config_get(self._model_config, "model_type") == "phi3_v":
            return ("prompt", prompt)
        signature = None if self._media_signatures is None else self._media_signatures[prompt]
        return _image_class(prompt, images) if signature is None else ("outputs", signature)

    def _generation_prompts(self, visit):
        return [prompt for prompt in self._schedule[visit % len(self._schedule)] for _ in range(self.num_generations)]

    def _shuffled(self, visit, rows):
        return np.random.default_rng(_derived_seed(self.seed, visit, 1)).permutation(rows)

    def _vision_family(self, inputs):
        _width, axes, padable, forbidden = _vlm_width_survey(
            inputs, disposable_keys=_vlm_pipeline_disposable_keys(self._model_config),
        )
        return _vlm_batch_family(inputs, symbolic_axes=axes if padable else None), padable, forbidden

    def ensure_descriptors(self):
        """Survey every entry's vision family for a compiled run, one processor call per micro-step at a prompt-only
        placeholder width over the seeded rows, which layouts later exchange only for rows with matching images.
        False when a family cannot be planned."""
        if self._families is not None:
            return True
        if self._processor is None:
            return False
        surveyed, entries = {}, []
        with _preserved_preprocessing_rng():
            for index in range(len(self._schedule)):
                start = self._generation_start(index)
                slot = (index - start) % self.steps_per_generation
                if (start, slot) not in surveyed:
                    surveyed[start, slot] = self._survey_slot(start, slot)
                entries.append(surveyed[start, slot])
        if not all(padable and _vlm_family_is_plannable(family) for family, padable, _ in entries):
            return False
        self._forbidden = frozenset().union(*(extents for _, _, extents in entries))
        self._families = tuple(family for family, _, _ in entries)
        return True

    def _survey_slot(self, start, slot):
        prompts = self._generation_prompts(start)
        chosen = [prompts[row] for row in self._shuffled(start, len(prompts))[slot * self.batch_size:][:self.batch_size]]
        media = {prompt: self._reencode(prompt, self._dataset[prompt])[1:3] for prompt in dict.fromkeys(chosen)} if self.vision else None
        width = max(self.width_rungs(start))
        while True:
            batch, lengths = self._pack(
                [self._prompt_ids[prompt] for prompt in chosen], np.array([len(self._prompt_ids[prompt]) for prompt in chosen]),
                width,
            )
            family, padable, forbidden = self._vision_family(_grpo_vision_batch(
                self._processor, self._model_config, batch, lengths, media and [media[prompt] for prompt in chosen],
            ))
            # A placeholder width that equals a vision extent cannot tell the two apart.
            if padable or width not in forbidden:
                return family, padable, forbidden
            width += 1

    def release_unplanned_descriptors(self):
        """Back to eager's layout and widths when the surveyed families were not compiled after all."""
        if self._shape_plan is None:
            self._families, self._forbidden = None, frozenset()

    def pop_family_drift(self):
        drift, self._family_drift = self._family_drift, None
        return drift

    def pop_metrics(self):
        """Generation-side metrics recorded since the last call, each averaged."""
        values = {name: float(np.mean(recorded)) for name, recorded in self._metrics.items()}
        self._metrics = {}
        return values

    def _record(self, name, value):
        self._metrics.setdefault(name, []).append(float(value))

    def _pack(self, sequences, prompt_lengths, width):
        batch = np.full((len(sequences), width), self.pad_id, dtype=np.int32)
        lengths = np.zeros((len(sequences), 2), dtype=np.int32)
        for row, (sequence, prompt_length) in enumerate(zip(sequences, prompt_lengths)):
            batch[row, :len(sequence)] = sequence
            lengths[row] = (prompt_length, len(sequence))
        return batch, lengths

    def _inputs(self, batch, lengths, media):
        if self._processor is None:
            return mx.array(batch)
        return _grpo_vision_batch(self._processor, self._model_config, batch, lengths, media)

    def configure_cce_compaction(self, enabled=True):
        self._cce_bounds = (min(map(len, self._prompt_ids)), max(self._max_tokens)) if enabled else None

    def prepare_cce_batch(self, index, batch):
        """Every fetch, an eager refetch included, already carries the indices old and reference log-probs were scored with."""
        return batch

    def _cce_indices(self, lengths, width):
        if self._cce_bounds is None:
            return None
        capacity = grpo_cce_capacity(len(lengths), width, *self._cce_bounds)
        return None if capacity is None else mx.array(grpo_cce_indices(lengths, width, capacity))

    def _prefix_ids(self, batch, lengths):
        """[groups, shared] leading prompt tokens each group's rows share, or None unless that cuts padded tokens by
        _prefix_tokr; shared, a multiple of 32, stops short of the last prompt token, which predicts the completion."""
        if not self._prefix:
            return None
        rows, width = batch.shape
        groups, shared = rows // self.num_generations, (int(lengths[:, 0].min()) - 1) // 32 * 32
        if shared <= 0 or rows % self.num_generations or (
            batch.reshape(groups, -1, width)[:, :, :shared] != batch[::self.num_generations, None, :shared]
        ).any():
            return None
        full, split = rows * (width - 1), groups * shared + rows * (width - 1 - shared)
        return mx.array(batch[::self.num_generations, :shared]) if full >= self._prefix_tokr * split else None

    def _score(self, scorer, visit, sequences, prompt_lengths, media, skipped=()):
        values = []
        for start in range(0, len(sequences), self.batch_size):
            chunk = sequences[start:start + self.batch_size]
            if start // self.batch_size in skipped:
                values.extend(np.zeros(len(sequence) - 1, dtype=np.float32) for sequence in chunk)
                continue
            inputs, lengths, width, prefix = self._slot_inputs(visit, start, chunk, prompt_lengths, media)
            logps = np.array(
                scorer(inputs, mx.array(lengths), self._cce_indices(lengths, width), prefix), dtype=np.float32,
            )
            values.extend(logps[row, :len(sequence) - 1] for row, sequence in enumerate(chunk))
        return values

    def _slot_inputs(self, visit, start, chunk, prompt_lengths, media):
        width = self._planned(self._width(max(len(sequence) for sequence in chunk)), visit + start // self.batch_size)
        batch, lengths = self._pack(chunk, prompt_lengths[start:start + self.batch_size], width)
        media = None if media is None else media[start:start + self.batch_size]
        return self._inputs(batch, lengths, media), lengths, width, self._prefix_ids(batch, lengths)

    def _generation_family_drift(self, visit, sequences, prompt_lengths, media):
        """The first micro-batch of a generation whose vision family left its survey, found before any is scored."""
        for start in range(0, len(sequences), self.batch_size):
            index = visit + start // self.batch_size
            if index >= len(self._families):
                continue
            inputs = self._slot_inputs(visit, start, sequences[start:start + self.batch_size], prompt_lengths, media)[0]
            divergence = _vlm_family_divergence(self._families[index], self._vision_family(inputs)[0])
            if divergence is not None:
                return f"batch {index}: {divergence}"
        return None

    def _decode(self, ids):
        try:
            return self._tokenizer.decode(list(ids), skip_special_tokens=True)
        except TypeError:
            return self._tokenizer.decode(list(ids))

    def _generate(self, visit):
        if self._model is None:
            raise RuntimeError("Unsloth MLX GRPO: bind the plan before fetching batches.")
        prompts = self._generation_prompts(visit)
        read = {prompt: self._dataset[prompt] for prompt in dict.fromkeys(prompts)}
        # Read again each generation and held only for it: the plan keeps no pixels.
        encoded = {prompt: self._reencode(prompt, row) for prompt, row in read.items()} if self.vision else {}
        requests = [
            GenerationRequest(
                prompt_token_ids=self._prompt_ids[prompt],
                vision_inputs=encoded[prompt][3] if encoded else None,
                max_tokens=self._max_tokens[prompt],
                sampling=SamplingParams(temperature=1.0, seed=_derived_seed(self.seed, visit, 0, row)),
                logits_processors=(
                    (hf_logits_warper(self.sampling, self._prompt_ids[prompt]),)
                    if self.sampling.warps else None
                ),
            )
            for row, prompt in enumerate(prompts)
        ]
        defaults = GenerationDefaults(
            max_tokens=self.sampling.max_tokens, stop_token_ids=(self.eos_id,),
        )
        with self._rollout_context():
            results = generate_module.generate_batch(
                self._model, self._processor or self._tokenizer, requests, defaults=defaults,
            )
        cut = [
            cut_completion(result.token_ids, result.finish_reason, eos_id=self.eos_id, pad_id=self.pad_id)
            for result in results
        ]
        completion_ids = [ids for ids, _ in cut]
        truncated = np.array([flag for _, flag in cut], dtype=bool)
        rows = [read[prompt] for prompt in prompts]
        texts = [self._decode(ids) for ids in completion_ids]
        lengths = np.array([len(ids) for ids in completion_ids])
        prompt_lengths = np.array([len(self._prompt_ids[prompt]) for prompt in prompts])
        # Counted before rewards, which read it through trainer_state as TRL's do.
        if self.counts_tokens:
            self._trainer_state.num_input_tokens_seen += int(prompt_lengths.sum() + lengths.sum())
        self._metrics["num_tokens"] = [self._trainer_state.num_input_tokens_seen]
        # TRL's placeholders reach the reward functions: it structures the prompts in place.
        reward_prompts, completions = reward_prompts_and_completions([
            _trl_multimodal_messages(read[prompt]["prompt"], len(encoded[prompt][2]))
            if encoded and _is_conversational_prompt(read[prompt]["prompt"]) else read[prompt]["prompt"]
            for prompt in prompts
        ], texts)
        rewards = calculate_rewards(
            self._reward_funcs, rows, prompts=reward_prompts, completions=completions,
            completion_ids=completion_ids, trainer_state=self._trainer_state,
        )
        advantages, reward_stats = group_advantages(
            rewards, self._reward_weights, self.num_generations, self.scale_rewards,
        )
        self._record_generation(lengths, truncated, rewards, reward_stats)
        kept = ~truncated if self.mask_truncated_completions else np.ones_like(truncated)
        weights = None
        if self.sample_weighting is not None:
            combined = np.nansum(rewards * np.asarray(self._reward_weights, dtype=np.float32)[None, :], axis=1)
            weights, excluded = scalerl_sample_weights(
                combined.astype(np.float32), self.num_generations, lengths * kept, *self.sample_weighting,
            )
            self._record("frac_reward_zero_std_excluded", excluded)
        if self._keep_completions:
            self.completions = {
                # Phi-3 Vision's negative placeholder ids do not decode.
                "prompt": [self._decode(i for i in self._prompt_ids[prompt] if i >= 0) for prompt in prompts],
                "completion": texts,
                "rewards": {
                    reward_function_name(reward_func): rewards[:, column].tolist()
                    for column, reward_func in enumerate(self._reward_funcs)
                },
                "advantage": advantages.tolist(),
            }
        # Shuffled before scoring so each old/reference chunk is one micro-batch at its own width, and a first iteration's ratio is exactly 1 when eager.
        order = self._shuffled(visit, len(prompts))
        # Exactly zero, not TRL's zero std: a rounded group mean leaves a tiny advantage that still has gradient.
        dead = (advantages == 0) & self.skip_zero_advantage
        eager = False
        if self._prefix:
            # Whole groups in each micro-batch; grpo_prefix_grouping admits only layouts whose block is a whole generation.
            size = self.num_generations
            dead_groups = dead.reshape(-1, size).all(axis=1)
            longest = (prompt_lengths + lengths).reshape(-1, size).max(axis=1) * self.group_by_length
            groups = np.array(list(dict.fromkeys(order // size)))
            groups = grpo_row_layout(groups, self._layout_rows // size, lambda g: (dead_groups[g], longest[g]))
            order = (groups[:, None] * size + np.arange(size)).reshape(-1)
        elif self.group_by_length or self.skip_zero_advantage:
            key = lambda row: (dead[row], prompt_lengths[row] + lengths[row] if self.group_by_length else 0)  # noqa: E731
            free = grpo_row_layout(order, self._layout_rows, key)
            if self._families is None:
                order = free
            else:
                # Rows move only between alike images, so families hold; a generation that skips less this way trains eagerly.
                classes = [self._media_class(prompt, encoded[prompt][2] if encoded else ()) for prompt in prompts]
                by_class = grpo_row_layout(order, self._layout_rows, key, classes)
                dead_steps = lambda rows: dead[rows].reshape(-1, self.batch_size).all(axis=1).sum()  # noqa: E731
                eager = bool(dead_steps(free) > dead_steps(by_class))
                order = free if eager else by_class
        skipped = {}
        if self.skip_zero_advantage:
            # The loss's stats for such rows: no KL, no clipping, and their masked completion lengths.
            skipped = {
                slot: np.array([0.0, (lengths * kept)[rows].mean(), 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                for slot, rows in enumerate(np.split(order, len(order) // self.batch_size)) if dead[rows].all()
            }
            self._record("frac_rows_skipped", len(skipped) * self.batch_size / len(order))
        sequences = [
            tuple(self._prompt_ids[prompts[row]]) + tuple(completion_ids[row]) for row in order
        ]
        prompt_lengths = prompt_lengths[order]
        media = [encoded[prompts[row]][1:3] for row in order] if encoded else None
        if self._families is not None:
            # A family that left its survey trains eagerly, so its old and reference log-probs are scored eagerly too.
            if not eager and self._step_compiled() and (self.with_old_logps or self.with_reference):
                self._family_drift = self._family_drift or self._generation_family_drift(
                    visit, sequences, prompt_lengths, media,
                )
                eager = self._family_drift is not None
            self._record("frac_generations_eager", eager)
        old = ref = None
        self._eager_generation = eager
        try:
            if self.with_old_logps:
                old = self._score(
                    lambda batch, lengths, indices, prefix: mx.stop_gradient(
                        self._logps(self._model, batch, lengths, indices, prefix)
                    ),
                    visit, sequences, prompt_lengths, media, skipped,
                )
            if self.with_reference:
                ref = self._score(
                    lambda batch, lengths, indices, prefix: self._reference_policy.forward(
                        self._model, batch, lengths,
                        response_scorer=lambda model, batch, lengths: self._logps(model, batch, lengths, indices, prefix),
                    ),
                    visit, sequences, prompt_lengths, media,
                )
        finally:
            self._eager_generation = False
        return _Generation(
            visit=visit, sequences=sequences, prompt_lengths=prompt_lengths,
            advantages=advantages[order], kept=kept[order].astype(np.int32), old=old, ref=ref,
            generation_tokens=int(lengths.sum()), media=media, skipped=skipped,
            weights=None if weights is None else weights[order], eager=eager,
        )

    def _reencode(self, prompt, row):
        encoded = _encode_grpo_prompt(self._tokenizer, row, self._processor, vision=True)
        if encoded[0] != self._prompt_ids[prompt]:
            raise ValueError(
                f"Unsloth MLX GRPO: dataset row {prompt} encodes to a different prompt "
                "than when the plan was built; its images must keep their size."
            )
        return encoded

    def _record_generation(self, lengths, truncated, rewards, reward_stats):
        terminated = lengths[~truncated] if (~truncated).any() else np.zeros(1)
        for name, values in (("length", lengths), ("terminated_length", terminated)):
            self._record(f"completions/mean_{name}", values.mean())
            self._record(f"completions/min_{name}", values.min())
            self._record(f"completions/max_{name}", values.max())
        self._record("completions/clipped_ratio", truncated.mean())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for column, reward_func in enumerate(self._reward_funcs):
                name = reward_function_name(reward_func)
                self._record(f"rewards/{name}/mean", np.nanmean(rewards[:, column]))
                self._record(f"rewards/{name}/std", np.nanstd(rewards[:, column], ddof=1))
        for name, value in reward_stats.items():
            self._record(name, value)

    def batch_family(self, index):
        rows = self.batch_size
        held = ((rows, "sequence-1"), "float32")
        # CCE indices are a function of the width, so they name no family of their own.
        family = (
            "grpo_tuple", ((rows, "sequence"), "int32"), ((rows, 2), "int32"),
            ((2,), "float32"), ((rows,), "float32"), ((rows,), "int32"),
            held if self.with_old_logps else None, held if self.with_reference else None,
        )
        if self.sample_weighting is not None:
            family = (*family, ((rows,), "float32"))
        return family if self._families is None else (*family, self._families[index])

    def set_shape_plan(self, shape_plan):
        if getattr(shape_plan.report, "action", None) not in ("exact", "bucket"):
            raise ValueError("only exact or bucket shape plans can be installed")
        self._shape_plan = shape_plan

    def _planned(self, width, index):
        # Old and reference log-probs are scored at the width the micro-batch trains at, so a first iteration's ratio
        # stays exactly 1, including after a compile fallback.
        width = self._clear_of_vision_extents(width)
        # A truncated schedule's last generation scores slots it never trains.
        if self._shape_plan is None or self._families is not None and index >= len(self._families):
            return width
        return self._shape_plan.endpoint_for(self.batch_family(index), width)

    def materialize(self, index, *, phase=None):
        """Micro-batch for schedule entry index at its planned width whatever the phase; refetching an entry reuses its generation instead of sampling again."""
        start = self._generation_start(index)
        if self._current is None or self._current.visit != start:
            self._current = self._generate(start)
        generation = self._current
        position = ((index - start) % self.steps_per_generation) * self.batch_size
        rows = range(position, min(position + self.batch_size, len(generation.sequences)))
        sequences = [generation.sequences[row] for row in rows]
        width = self._planned(self._width(max(len(sequence) for sequence in sequences)), index)
        batch, lengths = self._pack(sequences, generation.prompt_lengths[position:position + self.batch_size], width)

        def held(values):
            if values is None:
                return None
            table = np.zeros((len(sequences), width - 1), dtype=np.float32)
            for offset, row in enumerate(rows):
                table[offset, :len(values[row])] = values[row]
            return mx.array(table)

        media = None if generation.media is None else generation.media[position:position + self.batch_size]
        materialized = (
            self._inputs(batch, lengths, media), mx.array(lengths),
            mx.array([self.steps_per_generation, generation.generation_tokens], dtype=mx.float32),
            mx.array(generation.advantages[position:position + self.batch_size]),
            mx.array(generation.kept[position:position + self.batch_size]),
            held(generation.old), held(generation.ref),
        )
        if self._families is not None and phase is not None and self._family_drift is None and not generation.eager:
            divergence = _vlm_family_divergence(self._families[index], self._vision_family(materialized[0])[0])
            if divergence is not None:
                self._family_drift = f"batch {index}: {divergence}"
        indices = self._cce_indices(lengths, width)
        weights = generation.weights
        weights = None if weights is None else mx.array(weights[position:position + self.batch_size])
        if self._prefix:
            return (*materialized, indices, weights, self._prefix_ids(batch, lengths))
        if weights is not None:
            return (*materialized, indices, weights)
        return materialized if indices is None else (*materialized, indices)

    def runs_eager(self, index):
        """Whether a compiled run trains this micro-step eagerly: its generation laid out freely or drifted."""
        generation = self._current
        return generation is not None and generation.visit == self._generation_start(index) and generation.eager

    def skipped_stats(self, index):
        """Stats of a materialized micro-step whose rows all have advantage exactly 0, which the trainer counts unrun."""
        generation = self._current
        if generation is None or generation.visit != self._generation_start(index):
            return None
        return generation.skipped.get((index - generation.visit) % self.steps_per_generation)

    def __getitem__(self, index):
        return self.materialize(index)

    def __iter__(self):
        # One pass, sampling afresh: an evaluation never reuses an earlier one's rollouts.
        self._current = None
        return (self.materialize(index) for index in range(len(self)))


def _encode_grpo_prompts(
    dataset, tokenizer, sampling, max_seq_length, max_prompt_length, processor=None, vision=None,
):
    """Prompt ids, completion budgets and whether this is a vision dataset: as TRL reads it, an image column in the first row, here also holding an image."""
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise ValueError("Unsloth MLX GRPO: the tokenizer needs an eos_token_id to stop on.")
    if vision is None:
        first = dataset[0] if len(dataset) else None
        vision = processor is not None and isinstance(first, Mapping) and bool({"image", "images"} & set(first))
    prompt_ids, max_tokens, signatures = [], [], []
    long_prompts = capped = 0
    holds_images = False
    for index in range(len(dataset)):
        row = dataset[index]
        if not isinstance(row, Mapping) or "prompt" not in row:
            raise ValueError(
                f"Unsloth MLX GRPO: dataset row {index} has no prompt column."
            )
        if processor is not None and not vision and _grpo_images(row):
            raise ValueError(
                f"Unsloth MLX GRPO: dataset row {index} carries images, but the first row has no "
                "`image` or `images` field, which decides whether a dataset is text or vision. "
                "Give the first row an `images` field, empty if it has none."
            )
        # TRL tokenizes prompts with add_special_tokens=False: a chat template supplies its own BOS.
        ids, _, images, inputs = _encode_grpo_prompt(tokenizer, row, processor, vision)
        signatures.append(None if inputs is None else _media_signature(inputs, len(ids)))
        holds_images = holds_images or bool(images)
        if not ids or len(ids) >= max_seq_length:
            raise ValueError(
                f"Unsloth MLX GRPO: dataset row {index} encodes to a {len(ids)}-token "
                f"prompt, which leaves no room to generate within max_seq_length="
                f"{max_seq_length}. Filter the dataset or raise max_seq_length."
            )
        room = max_seq_length - len(ids)
        capped += room < sampling.max_tokens
        long_prompts += max_prompt_length is not None and len(ids) > max_prompt_length
        prompt_ids.append(ids)
        max_tokens.append(min(sampling.max_tokens, room))
    if vision and not holds_images:  # an empty image column: TRL trains such batches as text
        return _encode_grpo_prompts(
            dataset, tokenizer, sampling, max_seq_length, max_prompt_length, processor, False,
        )
    if long_prompts:
        warnings.warn(
            f"Unsloth MLX GRPO: {long_prompts} prompt(s) exceed max_prompt_length="
            f"{max_prompt_length}, which is not enforced (as on CUDA Unsloth and TRL "
            ">= 0.26); filter the dataset to bound prompt length.",
            stacklevel=3,
        )
    if capped:
        warnings.warn(
            f"Unsloth MLX GRPO: {capped} prompt(s) leave fewer than "
            f"{sampling.max_tokens} tokens within max_seq_length={max_seq_length}; "
            "their completions are capped to the room left.",
            stacklevel=3,
        )
    return prompt_ids, max_tokens, vision, signatures


def create_grpo_batch_plan(
    dataset, tokenizer, *, num_generations, batch_size, steps_per_generation,
    num_iterations, grad_accum, sampling, max_seq_length, score_temperature=None, scale_rewards="group",
    mask_truncated_completions=False, with_reference=False, max_prompt_length=None,
    shuffle=True, seed=None, num_batches=None, num_epochs=None, processor=None, model_config=None,
    group_by_length=False, skip_zero_advantage=False, sample_weighting=None, prefix_grouping=False,
):
    """Encode prompts once and lay out TRL's RepeatSampler schedule; batch_size counts completions, a generation samples
    batch_size * steps_per_generation of them, num_generations per prompt; score_temperature defaults to the sampling
    temperature; a VLM processor replaces tokenizer; within each optimizer step group_by_length sorts rows by length and
    skip_zero_advantage puts zero-advantage rows last; sample_weighting is grpo_sample_weighting's; prefix_grouping
    (grpo_prefix_grouping's) lays out whole groups per micro-batch and runs each shared prompt once."""
    if processor is not None:
        tokenizer = _get_processor_tokenizer(processor)
    prompt_ids, max_tokens, vision, signatures = _encode_grpo_prompts(
        dataset, tokenizer, sampling, max_seq_length, max_prompt_length, processor,
    )
    if batch_size * steps_per_generation % num_generations:
        raise ValueError(
            f"Unsloth MLX GRPO: a generation of {batch_size * steps_per_generation} "
            f"completions is not divisible by num_generations={num_generations}."
        )
    chunk = batch_size * steps_per_generation // num_generations
    repeats = steps_per_generation * num_iterations
    order = _RepeatSamplerOrder(len(prompt_ids), _normalize_seed(seed), shuffle)

    def batches_for_epoch(epoch):
        prompts = order(epoch)
        return tuple(
            tuple(prompts[start:start + chunk])
            for start in range(0, len(prompts) - chunk + 1, chunk)
            for _ in range(repeats)
        )

    schedule, cycle_length = _finite_batch_schedule(
        batches_for_epoch, num_batches=num_batches, num_epochs=num_epochs,
        grad_accum=grad_accum,
    )
    if not cycle_length:
        raise ValueError(
            f"Unsloth MLX GRPO: the dataset has {len(prompt_ids)} prompt(s), fewer "
            f"than the {chunk} one generation samples from."
        )
    return GRPOBatchPlan(
        dataset, tokenizer, prompt_ids, max_tokens, schedule,
        cycle_length=cycle_length, num_generations=num_generations,
        batch_size=batch_size, steps_per_generation=steps_per_generation,
        num_iterations=num_iterations, sampling=sampling,
        score_temperature=sampling.temperature if score_temperature is None else score_temperature,
        scale_rewards=scale_rewards,
        mask_truncated_completions=mask_truncated_completions,
        with_old_logps=grad_accum % repeats != 0, with_reference=with_reference,
        max_seq_length=max_seq_length, seed=_normalize_seed(seed),
        processor=processor, model_config=model_config, vision=vision,
        group_by_length=group_by_length, layout_block=math.gcd(steps_per_generation, grad_accum),
        skip_zero_advantage=skip_zero_advantage, sample_weighting=sample_weighting, media_signatures=signatures,
        prefix_grouping=prefix_grouping,
    )



def create_grpo_eval_plan(
    dataset, tokenizer, *, num_generations, batch_size, sampling, max_seq_length,
    score_temperature=None, scale_rewards="group", mask_truncated_completions=False,
    with_reference=False, max_prompt_length=None, seed=None, processor=None, model_config=None, sample_weighting=None,
):
    """TRL's eval RepeatSampler order, one generation per batch_size completions (last batch may be short), on its own
    seed stream without advancing num_input_tokens_seen; sample_weighting is training's, so eval_loss scores it."""
    if processor is not None:
        tokenizer = _get_processor_tokenizer(processor)
    prompt_ids, max_tokens, vision, _ = _encode_grpo_prompts(
        dataset, tokenizer, sampling, max_seq_length, max_prompt_length, processor,
    )
    seed = _normalize_seed(seed)
    order = _RepeatSamplerOrder(len(prompt_ids), seed, True)(0)
    chunk = batch_size // num_generations
    schedule = tuple(tuple(order[start:start + chunk]) for start in range(0, len(order), chunk))
    return GRPOBatchPlan(
        dataset, tokenizer, prompt_ids, max_tokens, schedule,
        cycle_length=len(schedule), num_generations=num_generations,
        batch_size=batch_size, steps_per_generation=1, num_iterations=1,
        sampling=sampling,
        score_temperature=sampling.temperature if score_temperature is None else score_temperature,
        scale_rewards=scale_rewards, mask_truncated_completions=mask_truncated_completions,
        with_old_logps=False, with_reference=with_reference, max_seq_length=max_seq_length,
        seed=_derived_seed(seed, 1), counts_tokens=False, sample_weighting=sample_weighting,
        processor=processor, model_config=model_config, vision=vision,
    )

_GENERATION_KWARGS = (
    "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "max_new_tokens",
    "do_sample",
)


def grpo_sampling(config):
    """Rollout sampling a GRPO config asks for; generation_kwargs override the named fields, as TRL updates its GenerationConfig with them."""
    kwargs = dict(config.generation_kwargs or {})
    unsupported = sorted(set(kwargs) - set(_GENERATION_KWARGS))
    if unsupported:
        raise ValueError(
            f"Unsloth MLX GRPO: generation_kwargs {unsupported} are not supported; "
            f"MLX rollouts take {', '.join(_GENERATION_KWARGS)}."
        )
    if kwargs.pop("do_sample", True) is not True:
        raise ValueError("Unsloth MLX GRPO: GRPO samples its completions, so do_sample must be True.")
    return GRPOSampling(
        max_tokens=kwargs.pop("max_new_tokens", config.max_completion_length),
        **{
            name: kwargs.get(name, getattr(config, name))
            for name in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")
        },
    )


def grpo_objective(config):
    # TRL scores at the configured temperature whatever generation_kwargs sample at.
    return GRPOObjective(
        loss_type=config.loss_type, beta=float(config.beta), epsilon=config.epsilon,
        epsilon_high=config.epsilon_high, delta=config.delta,
        importance_sampling_level=config.importance_sampling_level,
        temperature=config.temperature, max_completion_length=config.max_completion_length,
    )


def grpo_layout_options(objective, *, distributed=False):
    """Layouts keeping the window gradient: bnpo normalizes per micro-batch, so it is not regrouped; a zero-advantage row
    has no gradient only without a KL term, and a micro-step of such rows has loss 0 under every other normalizer."""
    return {
        "group_by_length": objective.loss_type != "bnpo",
        "skip_zero_advantage": (
            not distributed and objective.beta == 0 and objective.loss_type in ("grpo", "dapo", "dr_grpo")
        ),
    }


def grpo_prefix_grouping(config, *, distributed=False, model=None, announced=None, hint=True):
    """Whether whole groups per micro-batch keep every step's gradient, draw no per-row noise, and fit model. Warns, on
    every rank, why a requested prefix_grouper is refused, or, when off and given hint, prints that model and config
    would admit it; each message once per announced set (a trainer keeps one), every time without one."""
    if not config.prefix_grouper and model is None:
        return False
    batch_size, size = config.per_device_train_batch_size, config.num_generations
    checks = (
        (config.loss_type == "bnpo", "loss_type='bnpo' normalizes each micro-batch by its own tokens"),
        (batch_size % size, f"per_device_train_batch_size={batch_size} holds no whole groups of num_generations={size}"),
        (config.gradient_accumulation_steps % config.steps_per_generation,
         f"a generation's {config.steps_per_generation} micro-steps span several optimizer steps"),
        (distributed, "distributed processes split each generation"),
        (getattr(config, "neftune_noise_alpha", 0), "NEFTune draws its noise per row"),
    )
    problem = next((message for failed, message in checks if failed), None)
    if problem is None and model is not None:
        scorer = _make_preference_cce_scorer(model, temperature=config.temperature) if config.use_cce else None
        problem = grpo_prefix_problem(model, scorer, config.max_seq_length, dropout_disabled=config.disable_dropout)
    if config.prefix_grouper:
        message = problem and f"Unsloth MLX GRPO: prefix_grouper is off, since {problem}."
    else:
        message = None if problem or not hint else _PREFIX_HINT.format(size)
    if message and message not in (announced or ()):
        if not config.prefix_grouper:
            print(message)
        else:
            # A fresh registry leaves deduplication to announced, filled only once delivered (an "error" filter raises).
            # Every rank warns: raised on rank 0 alone, an "error" filter would leave the others hung in a collective.
            caller = sys._getframe(1)
            warnings.warn_explicit(
                message, UserWarning, caller.f_code.co_filename, caller.f_lineno,
                module=caller.f_globals.get("__name__"), registry={},
            )
        if announced is not None:
            announced.add(message)
    return bool(config.prefix_grouper) and problem is None


_PREFIX_HINT = (
    "Unsloth: prefix_grouper=True would run each shared prompt once for its {} completions, in micro-batches whose "
    "prompts share enough tokens; measured at num_generations=8 on Qwen3 text: 19% (short prompts) to 59% (long "
    "prompts) less GRPO training time."
)
# mlx-lm stacks whose attention applies RoPE at cache.offset and masks through cache.make_mask on every layer.
_PREFIX_MODEL_TYPES = ("llama", "qwen2", "qwen3")


def grpo_prefix_problem(model, scorer, max_seq_length, *, dropout_disabled=False):
    """Why a shared-prompt forward cannot run model, or None."""
    from . import utils

    if scorer is None:
        return "it scores through the runtime CCE head, which use_cce=False or this model's head rules out"
    backbone = utils._get_text_backbone(model)
    model_type = getattr(model, "model_type", None)
    if model_type not in _PREFIX_MODEL_TYPES or backbone is None:
        return f"model_type={model_type!r} is not one of {', '.join(_PREFIX_MODEL_TYPES)}"
    if utils._is_vlm_model(model) or getattr(model, "_unsloth_text_only_vlm", False):
        return "it is a vision model or a text-only load of one, which this forward does not cover"
    ropes = utils._length_dependent_ropes(backbone)
    if ropes:
        return f"its RoPE ({', '.join(ropes)}) may depend on the forward's length, not only the position"
    window = getattr(backbone, "sliding_window", None)
    if any(getattr(layer, "use_sliding", False) for layer in backbone.layers) and (window or 0) < max_seq_length:
        return f"its {window}-token sliding window is shorter than max_seq_length={max_seq_length}"
    if not dropout_disabled and any(
        utils._get_mlx_dropout_probability(getattr(module, "dropout", None)) > 0
        for _, module in utils.iter_mlx_lora_modules(model)
    ):
        return "LoRA dropout would draw one mask over each shared prompt instead of one per row"
    return None


def _is_torch_module(func):
    torch = sys.modules.get("torch")  # a caller holding a torch module has imported torch
    return torch is not None and isinstance(func, torch.nn.Module)


class _RewardModel:
    """A Hugging Face sequence-classification reward model scored as TRL scores one: logit 0 of each rendered prompt + completion, on the device it was given."""

    def __init__(self, model, tokenizer):
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        self.model, self.tokenizer = model, tokenizer
        self.__name__ = model.config._name_or_path.split("/")[-1]

    def __call__(self, prompts, completions, **_):
        import torch
        # The completion, not the prompt: an assistant-only prompt is empty once its turn moves into the completion.
        if _is_conversational_prompt(completions[0]):
            texts = [
                self.tokenizer.apply_chat_template(prompt + completion, tools=None, tokenize=False)
                for prompt, completion in zip(prompts, completions)
            ]
        else:
            texts = [prompt + completion for prompt, completion in zip(prompts, completions)]
        inputs = self.tokenizer(
            text=texts, return_tensors="pt", padding=True, padding_side="right",
            add_special_tokens=False,
        )
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            logits = self.model(**{key: value.to(device) for key, value in inputs.items()}).logits
        return logits[:, 0].float().tolist()


def _load_reward_model(name):
    try:
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification.from_pretrained(name, num_labels=1)
    except ImportError as error:
        raise ImportError(
            f"Unsloth MLX GRPO: reward function {name!r} names a Hugging Face "
            "sequence-classification model, which needs torch and transformers."
        ) from error


def grpo_reward_funcs(reward_funcs, reward_processing_classes=None):
    """TRL's reward_funcs: callables, torch reward models, or their ids, each model paired with its entry of reward_processing_classes."""
    funcs = list(reward_funcs) if isinstance(reward_funcs, (list, tuple)) else [reward_funcs]
    if not funcs:
        raise ValueError("Unsloth MLX GRPO: reward_funcs needs at least one reward function.")
    if reward_processing_classes is None:
        tokenizers = [None] * len(funcs)
    elif isinstance(reward_processing_classes, (list, tuple)):
        tokenizers = list(reward_processing_classes)
    else:
        tokenizers = [reward_processing_classes]
    if len(tokenizers) != len(funcs):
        raise ValueError(
            f"Unsloth MLX GRPO: {len(tokenizers)} reward_processing_classes for "
            f"{len(funcs)} reward functions."
        )
    for index, func in enumerate(funcs):
        if isinstance(func, str):
            func = _load_reward_model(func)
        if _is_torch_module(func):
            funcs[index] = _RewardModel(func, tokenizers[index])
        elif not callable(func) or hasattr(func, "parameters"):
            raise ValueError(
                f"Unsloth MLX GRPO: reward function {func!r} is neither a callable nor a "
                "torch reward model. Wrap an MLX model in a function that returns one "
                "reward per completion."
            )
    return funcs


_SCALERL_LOSS_TYPES = ("bnpo", "dapo")  # token-normalized, so exclusion alone is defined


def grpo_sample_weighting(config):
    """(exclude, prompt_level) for scalerl_sample_weights, or None when the config trains TRL's objective."""
    prompt_level = config.loss_aggregation == "prompt"
    return (bool(config.zero_variance_exclude), prompt_level) if config.zero_variance_exclude or prompt_level else None


def _resolve_scalerl_options(config):
    """CUDA Unsloth's ScaleRL environment variables are the fields' defaults; the GRPO flags it reads for vLLM rollouts
    have no MLX counterpart."""
    if config.zero_variance_exclude is None:
        config.zero_variance_exclude = os.environ.get("UNSLOTH_GRPO_ZERO_VARIANCE_EXCLUDE", "0") == "1"
        if config.zero_variance_exclude:
            warnings.warn(
                "Unsloth MLX GRPO: UNSLOTH_GRPO_ZERO_VARIANCE_EXCLUDE=1 sets zero_variance_exclude=True, as on CUDA "
                "Unsloth; set the field to choose explicitly.",
                stacklevel=4,
            )
    if config.loss_aggregation is None:
        value = os.environ.get("UNSLOTH_GRPO_LOSS_AGGREGATION", "").strip().lower()
        if value not in ("", "prompt"):
            raise ValueError(
                f"Unsloth MLX GRPO: UNSLOTH_GRPO_LOSS_AGGREGATION must be 'prompt' or unset, got {value!r}."
            )
        config.loss_aggregation = value or "token"
        if value:
            warnings.warn(
                "Unsloth MLX GRPO: UNSLOTH_GRPO_LOSS_AGGREGATION=prompt sets loss_aggregation='prompt', as on CUDA "
                "Unsloth; set the field to choose explicitly.",
                stacklevel=4,
            )
    if config.loss_aggregation not in ("token", "prompt"):
        raise ValueError(
            f"Unsloth MLX GRPO: loss_aggregation must be 'token' or 'prompt', got {config.loss_aggregation!r}."
        )
    alone = config.zero_variance_exclude and config.loss_aggregation == "token"
    if alone and config.loss_type not in _SCALERL_LOSS_TYPES:
        warnings.warn(
            "Unsloth MLX GRPO: zero_variance_exclude alone needs a token-normalized loss_type (bnpo, dapo), as on "
            f"CUDA Unsloth; it is ignored for loss_type={config.loss_type!r}.",
            stacklevel=4,
        )
        config.zero_variance_exclude = False
    for name in ("UNSLOTH_GRPO_TRUNCATED_SUPPORT", "UNSLOTH_GRPO_VLLM_IS_MODE"):
        if os.environ.get(name, "").strip() not in ("", "0"):
            warnings.warn(f"Unsloth MLX GRPO: {name} is not supported on MLX and is ignored.", stacklevel=4)


def _resolve_prefix_grouper(config):
    """CUDA Unsloth's UNSLOTH_GRPO_PREFIX_GROUPER is the field's default, which stays off when it is unset (CUDA's is on)."""
    if config.prefix_grouper is None:
        value = os.environ.get("UNSLOTH_GRPO_PREFIX_GROUPER", "0").strip().lower()
        config.prefix_grouper = value not in ("", "0", "false", "no", "off")
        if config.prefix_grouper:
            warnings.warn(
                f"Unsloth MLX GRPO: UNSLOTH_GRPO_PREFIX_GROUPER={value} sets prefix_grouper=True, as on CUDA Unsloth; "
                "set the field to choose explicitly.",
                stacklevel=4,
            )


def resolve_grpo_config(config):
    """CUDA Unsloth's GRPOConfig rewrites, then TRL 0.24's ``__post_init__``."""
    if config.use_vllm:
        raise ValueError("Unsloth MLX GRPO: use_vllm needs vLLM; MLX samples with its own engine.")
    if config.use_liger_loss:
        raise ValueError("Unsloth MLX GRPO: use_liger_loss selects a CUDA kernel.")
    default_loss_type = config.loss_type is None
    if default_loss_type:
        config.loss_type = "dapo"
        if config.beta is None:
            warnings.warn(
                "Unsloth MLX GRPO: loss_type defaults to TRL's 'dapo' with beta=0.0, as on "
                "CUDA Unsloth. Set loss_type='bnpo', beta=0.001 for the previous default.",
                stacklevel=3,
            )
    if isinstance(config.loss_type, str) and config.loss_type.lower() in ("dr_grpo", "dapo"):
        config.loss_type = config.loss_type.lower()
    if config.beta is None:
        config.beta = 0.0 if config.loss_type in ("dapo", "dr_grpo") else 0.001
    if config.loss_type == "dr_grpo":
        if config.scale_rewards is None:
            config.scale_rewards = True
        elif config.scale_rewards is True or config.scale_rewards == "group":
            warnings.warn(
                "Unsloth MLX GRPO: loss_type='dr_grpo' does not scale rewards, so "
                "scale_rewards is set to False, as on CUDA Unsloth. Set it to None to keep scaling.",
                stacklevel=3,
            )
            config.scale_rewards = False
    # The default keeps TRL's settings: masking truncated rows zeroes an all-truncated update.
    elif config.loss_type == "dapo" and not default_loss_type:
        for name, value in (("mask_truncated_completions", True), ("epsilon_high", 0.28)):
            if getattr(config, name) is None:
                warnings.warn(
                    f"Unsloth MLX GRPO: loss_type='dapo' sets {name}={value}, as on "
                    "CUDA Unsloth.",
                    stacklevel=3,
                )
                setattr(config, name, value)
        if config.beta != 0:
            warnings.warn(
                f"Unsloth MLX GRPO: loss_type='dapo' is usually trained with beta=0; "
                f"beta={config.beta} keeps its KL term.",
                stacklevel=3,
            )
    if config.mask_truncated_completions is None:
        config.mask_truncated_completions = False
    if not 0 < config.temperature < 10:
        raise ValueError(
            f"Unsloth MLX GRPO: temperature={config.temperature} must lie between 0 "
            "and 10, as on CUDA Unsloth."
        )
    batch_size, generations = config.per_device_train_batch_size, config.num_generations
    if (
        config.steps_per_generation is None and config.generation_batch_size is None
        and batch_size * config.gradient_accumulation_steps % generations
    ):
        warnings.warn(
            f"Unsloth MLX GRPO: per_device_train_batch_size * gradient_accumulation_steps "
            f"is not divisible by num_generations={generations}, so "
            f"per_device_train_batch_size is set to {generations}, as on CUDA Unsloth.",
            stacklevel=3,
        )
        config.per_device_train_batch_size = batch_size = generations
    config.scale_rewards = {True: "group", False: "none"}.get(config.scale_rewards, config.scale_rewards)
    if config.scale_rewards not in GRPO_SCALE_REWARDS:
        raise ValueError(
            f"Unsloth MLX GRPO: scale_rewards must be one of "
            f"{', '.join(GRPO_SCALE_REWARDS)}, got {config.scale_rewards!r}."
        )
    if config.generation_batch_size is not None and config.steps_per_generation is not None:
        raise ValueError(
            "Unsloth MLX GRPO: set generation_batch_size or steps_per_generation, not both."
        )
    if config.generation_batch_size is None:
        config.steps_per_generation = config.steps_per_generation or config.gradient_accumulation_steps
        config.generation_batch_size = batch_size * config.steps_per_generation
    elif config.generation_batch_size % batch_size:
        raise ValueError(
            f"Unsloth MLX GRPO: generation_batch_size={config.generation_batch_size} is "
            f"not divisible by per_device_train_batch_size={batch_size}."
        )
    else:
        config.steps_per_generation = config.generation_batch_size // batch_size
    if config.generation_batch_size % generations:
        raise ValueError(
            f"Unsloth MLX GRPO: generation_batch_size={config.generation_batch_size} is "
            f"not divisible by num_generations={generations}."
        )
    if generations < 2:
        raise ValueError(
            "Unsloth MLX GRPO: num_generations must be at least 2, since advantages "
            "compare completions within a group."
        )
    if config.top_entropy_quantile < 1:
        warnings.warn(
            "Unsloth MLX GRPO: top_entropy_quantile < 1 is ignored, as on CUDA Unsloth.",
            stacklevel=3,
        )
    _resolve_scalerl_options(config)
    _resolve_prefix_grouper(config)
    grpo_objective(config)
    grpo_sampling(config)


def format_completions(sample, step, limit=None):
    """Plain-text take on TRL's print_prompt_completions_sample table: the first limit rows, where TRL draws them at random."""
    names = list(sample["rewards"])
    count = len(sample["prompt"]) if limit is None else min(int(limit), len(sample["prompt"]))
    lines = [f"Step {step}", " | ".join(("Prompt", "Completion", *names, "Advantage"))]
    for row in range(count):
        lines.append(" | ".join((
            repr(sample["prompt"][row]), repr(sample["completion"][row]),
            *(f"{sample['rewards'][name][row]:.2f}" for name in names),
            f"{sample['advantage'][row]:.2f}",
        )))
    return "\n".join(lines)
