"""MLX GRPO: completions, rewards, advantages, and the clipped policy objective.

The behaviour mirrors TRL 0.24's ``GRPOTrainer`` as Unsloth patches it on CUDA
(``unsloth_zoo.rl_replacements.grpo_compute_loss``).
"""

import copy
import warnings
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .preference import _response_mask, _supervised_tokens
from .utils import _model_logits


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
    loss_type: str = "bnpo"
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
    """TRL's completion: through the first EOS, and whether it was truncated.

    The generation engine leaves the stopping id out of ``token_ids``.
    """
    ids = [int(token) for token in token_ids]
    if finish_reason == "stop" and eos_id is not None:
        ids.append(int(eos_id))
    if eos_id is not None and eos_id in ids:
        ids = ids[:ids.index(eos_id) + 1]
    truncated = not ids or ids[-1] not in (eos_id, pad_id)
    return ids, truncated


def reward_prompts_and_completions(prompts, completion_texts):
    """What TRL hands reward functions: conversational completions become one
    assistant message, and a prompt ending in an assistant turn gives that turn
    up as the completion's opening."""
    if not prompts or not _is_conversational_prompt(prompts[0]):
        return list(prompts), list(completion_texts)
    reward_prompts, completions = [], []
    for prompt, text in zip(prompts, completion_texts):
        prompt = copy.deepcopy(prompt)
        bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
        reward_prompts.append(prompt)
        completions.append([{"role": "assistant", "content": bootstrap + text}])
    return reward_prompts, completions


def reward_function_name(reward_func):
    return getattr(reward_func, "__name__", type(reward_func).__name__)


def calculate_rewards(
    reward_funcs, rows, *, prompts, completions, completion_ids, trainer_state,
):
    """``(len(rows), len(reward_funcs))`` float32 rewards, NaN where a function
    returned None, called as TRL calls them."""
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


def completion_logps(model, batch, temperature):
    """Per-token log probabilities of ``batch[:, 1:]``, on logits divided by the
    sampling temperature as TRL scores them."""
    logits = _model_logits(model(batch[:, :-1])).astype(mx.float32)
    if temperature != 1.0:
        logits = logits / temperature
    targets = batch[:, 1:]
    return -nn.losses.cross_entropy(logits, targets, reduction="none").reshape(targets.shape)


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
        kl = mx.exp(ref - new) - (ref - new) - 1.0
        per_token = per_token + objective.beta * kl
    else:
        kl = mx.zeros((new.shape[0], 1) if coef_1.shape[1] == 1 else new.shape, mx.float32)
    return per_token, kl, coef_1


def _reduce(objective, per_token, mask, normalizers):
    """``k`` times TRL's micro-step loss, ``k`` the window's micro-step count: the
    trainer averages micro-step gradients, TRL sums them."""
    window_microbatches, generation_tokens = normalizers[0], normalizers[1]
    masked = per_token * mask
    if objective.loss_type == "grpo":
        return (masked.sum(-1) / mx.maximum(mask.sum(-1), mx.array(1.0))).mean()
    if objective.loss_type == "bnpo":
        return masked.sum() / mx.maximum(mask.sum(), mx.array(1.0))
    if objective.loss_type == "dr_grpo":
        return masked.sum() / (per_token.shape[0] * objective.max_completion_length)
    return window_microbatches * masked.sum() / generation_tokens


def grpo_stats(objective, kl, coef_1, advantages, response_mask):
    """Metric values measured as Unsloth does: over each completion's tokens,
    truncated or not."""
    rows = response_mask.sum(-1)
    if kl.shape[1] == 1:
        mean_kl = kl.mean()
    else:
        mean_kl = ((kl * response_mask).sum(-1) / mx.maximum(rows, mx.array(1.0))).mean()
    advantages = advantages[:, None]
    low = mx.logical_and(coef_1 < 1 - objective.epsilon, advantages < 0).astype(mx.float32)
    high = mx.logical_and(coef_1 > 1 + objective.epsilon_high, advantages > 0).astype(mx.float32)
    region = mx.logical_or(low > 0, high > 0).astype(mx.float32)
    values = [
        mean_kl,
        rows.mean(),
        _masked_mean(low, response_mask),
        _masked_mean(high, response_mask),
        _masked_mean(region, response_mask),
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


def make_grpo_loss_fn(objective):
    """GRPO loss over a micro-batch ``(tokens, lengths, normalizers, advantages,
    kept, old_logps, ref_logps)``; ``kept`` zeroes rows TRL masks as truncated."""

    def loss_fn(model, batch, lengths, normalizers, advantages, kept, old=None, ref=None):
        new = completion_logps(model, batch, objective.temperature)
        response_mask = _response_mask(batch[:, 1:], lengths)
        per_token, kl, coef_1 = grpo_terms(
            objective, new, advantages, response_mask, old=old, ref=ref,
        )
        mask = response_mask * kept[:, None].astype(mx.float32)
        loss = _reduce(objective, per_token, mask, normalizers)
        return loss, mx.array(1, dtype=mx.int32), grpo_stats(
            objective, kl, coef_1, advantages, response_mask,
        )

    loss_fn._unsloth_supervised_tokens = _supervised_tokens
    loss_fn._unsloth_preference_metrics = GRPO_LOSS_METRICS
    loss_fn._unsloth_preference_stats_width = GRPO_STATS_WIDTH
    loss_fn._unsloth_preference_report = grpo_metric_values
    loss_fn._unsloth_cce_compaction = False
    return loss_fn
