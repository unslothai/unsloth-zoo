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

"""GRPO for MLX text models: rollouts, rewards, group advantages and the loss."""

import inspect
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .generate import GenerationDefaults, GenerationRequest, SamplingParams, generate_batch
from .preference import _response_mask, _supervised_tokens
from .utils import _model_logits, _normalize_seed

# Rollout widths are padded up to this multiple so the compiled step sees a
# bounded set of shapes; padding sits past every row's end and is masked out.
ROLLOUT_WIDTH_MULTIPLE = 64


def _token_logps(model, batch, temperature):
    targets = batch[:, 1:]
    logits = _model_logits(model(batch[:, :-1]))
    # TRL scores every log-prob on the tempered distribution the rollout sampled.
    if temperature != 1.0:
        logits = logits / temperature
    return -nn.losses.cross_entropy(logits, targets, reduction="none").reshape(targets.shape)


def grpo_metric_layout(beta):
    """Logged names and, per name, the index of its denominator in the stats vector."""
    if beta:
        return ("reward", "kl", "completions/mean_length"), (3, 4, 3)
    return ("reward", "completions/mean_length"), (2, 2)


def make_grpo_loss_fn(*, beta, epsilon_low, epsilon_high, temperature, reference_policy=None):
    """TRL ``loss_type="grpo"``: per-sequence mean over completion tokens, then mean over rows.

    The accumulation weight is the row count, so a window of micro-batches
    averages rows exactly as one batch of the same rollouts would.
    """
    if beta and reference_policy is None:
        raise ValueError("Unsloth MLX GRPO: beta != 0 needs a reference policy.")

    def loss_fn(model, batch, lengths, advantages, rewards):
        mask = _response_mask(batch[:, 1:], lengths)
        logps = _token_logps(model, batch, temperature)
        ratio = mx.exp(logps - mx.stop_gradient(logps))
        clipped = mx.clip(ratio, 1 - epsilon_low, 1 + epsilon_high)
        advantage = advantages[:, None]
        per_token = -mx.minimum(ratio * advantage, clipped * advantage)
        tokens = mask.sum(axis=1)
        stats = [rewards.astype(mx.float32).sum()]
        if beta:
            with reference_policy.activate(model) as reference:
                reference_logps = mx.stop_gradient(_token_logps(reference, batch, temperature))
            delta = reference_logps - logps
            kl = mx.exp(delta) - delta - 1
            per_token = per_token + beta * kl
            stats.append(mx.stop_gradient((kl * mask).sum()))
        loss = ((per_token * mask).sum(axis=1) / mx.maximum(tokens, mx.array(1.0))).mean()
        rows = batch.shape[0]
        stats += [tokens.sum(), mx.array(float(rows))]
        if beta:
            stats.append(tokens.sum())
        return loss, mx.array(rows, dtype=mx.int32), mx.stack(stats)

    names, denominators = grpo_metric_layout(beta)
    loss_fn._unsloth_preference_metrics = names
    loss_fn._unsloth_preference_denominators = denominators
    loss_fn._unsloth_preference_stats_width = len(names) + (2 if beta else 1)
    loss_fn._unsloth_supervised_tokens = _supervised_tokens
    return loss_fn


def group_advantages(rewards, num_generations):
    """``(r - mean) / (std + 1e-4)`` per group, with TRL's unbiased (ddof=1) std."""
    grouped = np.asarray(rewards, dtype=np.float64).reshape(-1, num_generations)
    mean = grouped.mean(axis=1, keepdims=True)
    std = grouped.std(axis=1, ddof=1, keepdims=True)
    return ((grouped - mean) / (std + 1e-4)).reshape(-1)


def _accepts(func, name):
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _is_conversational(prompt):
    return (
        isinstance(prompt, list) and bool(prompt)
        and isinstance(prompt[0], dict) and "role" in prompt[0] and "content" in prompt[0]
    )


def render_prompt(tokenizer, prompt):
    """TRL's ``maybe_apply_chat_template`` for a prompt: continue a trailing assistant turn."""
    if not _is_conversational(prompt):
        return prompt
    last_role = prompt[-1]["role"]
    if last_role not in ("user", "assistant"):
        raise ValueError(
            f"Unsloth MLX GRPO: a chat prompt must end with a user or assistant turn, not {last_role!r}."
        )
    continues = last_role == "assistant"
    return tokenizer.apply_chat_template(
        prompt, tokenize=False,
        add_generation_prompt=not continues, continue_final_message=continues,
    )


def _reward_view(prompt, completions):
    """What TRL hands reward functions: chat prompts get assistant-message completions."""
    if not _is_conversational(prompt):
        return prompt, completions
    if prompt[-1]["role"] == "assistant":
        bootstrap, prompt = prompt[-1].get("content", ""), prompt[:-1]
    else:
        bootstrap = ""
    return prompt, [[{"role": "assistant", "content": bootstrap + text}] for text in completions]


def score_rewards(reward_funcs, reward_weights, examples, completions, completion_ids, trainer_state):
    """Weighted sum over reward functions, one score per completion; None is skipped as TRL's nansum does."""
    prompts, views = [], []
    for example, texts in zip(examples, completions):
        prompt, view = _reward_view(example["prompt"], texts)
        prompts += [prompt] * len(texts)
        views += view
    rows = [example for example, texts in zip(examples, completions) for _ in texts]
    ids = [row for group in completion_ids for row in group]
    columns = {
        key: [row[key] for row in rows]
        for key in examples[0] if key not in ("prompt", "completion", "completion_ids")
    }
    total = np.zeros(len(views), dtype=np.float64)
    for func, weight in zip(reward_funcs, reward_weights):
        kwargs = dict(columns, prompts=prompts, completions=views)
        if _accepts(func, "completion_ids"):
            kwargs["completion_ids"] = ids
        if _accepts(func, "trainer_state"):
            kwargs["trainer_state"] = trainer_state
        values = func(**kwargs)
        if len(values) != len(views):
            name = getattr(func, "__name__", type(func).__name__)
            raise ValueError(
                f"Unsloth MLX GRPO: reward function {name!r} returned {len(values)} "
                f"scores for {len(views)} completions; it must return one per completion."
            )
        for index, value in enumerate(values):
            if value is not None:
                total[index] += weight * float(value)
    return total


def rollout_prompt_indices(position, *, num_prompts, prompts_per_batch, rank, world, seed, shuffle):
    """Dataset rows micro-batch ``position`` rolls out on this rank; a pure function of ``position``.

    Each epoch visits a fresh seeded permutation (TRL's shuffling sampler), and a
    rank takes its own contiguous slice of every global micro-batch.
    """
    per_step = prompts_per_batch * world
    epoch_length = math.ceil(num_prompts / per_step)
    epoch, step = divmod(position, epoch_length)
    if shuffle:
        order = np.random.RandomState(
            (_normalize_seed(seed) + epoch) % (2 ** 32)
        ).permutation(num_prompts)
    else:
        order = np.arange(num_prompts)
    start = step * per_step + rank * prompts_per_batch
    return [int(order[(start + offset) % num_prompts]) for offset in range(prompts_per_batch)]


def rollout_seed(seed, rank, position, row):
    return int(np.random.SeedSequence(
        [_normalize_seed(seed), int(rank), int(position), int(row)]
    ).generate_state(1)[0])


def _pad_width(width, max_seq_length):
    padded = -(-(width - 1) // ROLLOUT_WIDTH_MULTIPLE) * ROLLOUT_WIDTH_MULTIPLE + 1
    return max(width, min(padded, max_seq_length))


def build_rollout_batch(
    model, tokenizer, encoder, examples, *, args, reward_funcs, reward_weights,
    seeds, pad_id, trainer_state,
):
    """Generate ``num_generations`` completions per example and score them.

    Returns ``(batch, lengths, advantages, rewards)``. A completion that stopped
    on an end token keeps that token, as TRL's completion mask does.
    """
    group = int(args.num_generations)
    top_p = float(args.top_p)
    requests, prompt_ids = [], []
    for example in examples:
        ids = encoder(render_prompt(tokenizer, example["prompt"]))
        room = int(args.max_seq_length) - len(ids)
        if room <= 0:
            raise ValueError(
                f"Unsloth MLX GRPO: a rendered prompt uses {len(ids)} tokens, leaving no "
                f"room for a completion within max_seq_length={args.max_seq_length}."
            )
        prompt_ids.append(ids)
        for _ in range(group):
            requests.append(GenerationRequest(
                prompt_token_ids=ids,
                max_tokens=min(int(args.max_completion_length), room),
                sampling=SamplingParams(
                    temperature=float(args.temperature),
                    top_p=0.0 if top_p >= 1.0 else top_p,
                    top_k=int(args.top_k or 0),
                    min_p=float(args.min_p or 0.0),
                    seed=seeds[len(requests)],
                ),
            ))
    results = generate_batch(
        model, tokenizer, requests,
        defaults=GenerationDefaults(
            max_tokens=int(args.max_completion_length),
            completion_batch_size=max(32, len(requests)),
        ),
    )
    rows, lengths, texts, completion_ids = [], [], [], []
    for index, result in enumerate(results):
        ids = prompt_ids[index // group]
        completion = [int(token) for token in result.token_ids]
        if result.finish_reason == "stop" and result.stop_token_id is not None:
            completion.append(int(result.stop_token_id))
        rows.append(ids + completion)
        # TRL's mask_truncated_completions: the row still counts, with no tokens.
        truncated = args.mask_truncated_completions and result.finish_reason == "length"
        lengths.append([len(rows[-1]) if truncated else len(ids), len(rows[-1])])
        texts.append(result.text)
        completion_ids.append(completion)
    rewards = score_rewards(
        reward_funcs, reward_weights, examples,
        [texts[i:i + group] for i in range(0, len(texts), group)],
        [completion_ids[i:i + group] for i in range(0, len(completion_ids), group)],
        trainer_state,
    )
    width = _pad_width(max(len(row) for row in rows), int(args.max_seq_length))
    batch = mx.array([row + [pad_id] * (width - len(row)) for row in rows], dtype=mx.int32)
    return (
        batch,
        mx.array(lengths, dtype=mx.int32),
        mx.array(group_advantages(rewards, group), dtype=mx.float32),
        mx.array(rewards, dtype=mx.float32),
    )
