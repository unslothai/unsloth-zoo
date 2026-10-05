"""MLX GRPO: completions, rewards, advantages, and the clipped policy objective.

The behaviour mirrors TRL 0.24's ``GRPOTrainer`` as Unsloth patches it on CUDA
(``unsloth_zoo.rl_replacements.grpo_compute_loss``).
"""

import copy
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
    _chat_template, _make_preference_cce_scorer, _response_mask, _supervised_tokens,
)
from .utils import (
    _finite_batch_schedule,
    _finite_text_pad_width,
    _model_logits,
    _normalize_mlx_messages,
    _normalize_seed,
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


def grpo_logps(temperature, scorer=None):
    """``(model, batch, lengths) -> log-probs``: the logits path, or ``scorer``'s
    runtime CCE, whose own temperature then applies; it scores response positions
    only, and zero elsewhere."""
    if scorer is None:
        return lambda model, batch, _lengths: completion_logps(model, batch, temperature)

    def logps(model, batch, lengths):
        supervised = _response_mask(batch[:, 1:], lengths) > 0
        return -scorer(model, batch, supervised, logit_sums=False)[0]

    return logps


def _scored_as_step(logps, step_compiled):
    """Score as the training step's forward, which MLX runs differently under its trace."""

    def traced(model, batch, lengths):
        if step_compiled():
            run = lambda anchor: (anchor, logps(model, batch, lengths))
            return mx.value_and_grad(run)(mx.array(0.0))[0][1]
        run = lambda: (mx.array(0.0), logps(model, batch, lengths))
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


def _reduce(objective, per_token, mask, normalizers):
    """``k`` times Unsloth's micro-step loss, ``k`` the window's micro-step count (it cancels):
    the trainer averages micro-step gradients, TRL sums them."""
    steps_per_generation, generation_tokens = normalizers[0], normalizers[1]
    masked = per_token * mask
    if objective.loss_type == "grpo":
        return (masked.sum(-1) / mx.maximum(mask.sum(-1), mx.array(1.0))).mean()
    if objective.loss_type == "dr_grpo":
        return masked.sum() / (per_token.shape[0] * objective.max_completion_length)
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
    """GRPO loss over a micro-batch ``(tokens, lengths, normalizers, advantages,
    kept, old_logps, ref_logps)``; ``kept`` zeroes rows TRL masks as truncated."""
    logps = grpo_logps(objective.temperature, scorer)

    def loss_fn(model, batch, lengths, normalizers, advantages, kept, old=None, ref=None):
        new = logps(model, batch, lengths)
        mask = _response_mask(batch[:, 1:], lengths) * kept[:, None].astype(mx.float32)
        per_token, kl, coef_1 = grpo_terms(objective, new, advantages, mask, old=old, ref=ref)
        loss = _reduce(objective, per_token, mask, normalizers)
        return loss, mx.array(1, dtype=mx.int32), grpo_stats(objective, kl, coef_1, advantages, mask)

    loss_fn._unsloth_supervised_tokens = _supervised_tokens
    loss_fn._unsloth_preference_metrics = GRPO_LOSS_METRICS
    loss_fn._unsloth_preference_stats_width = GRPO_STATS_WIDTH
    loss_fn._unsloth_preference_report = grpo_metric_values
    loss_fn._unsloth_cce_compaction = False
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
        return loss, mx.array(batch.shape[0], dtype=mx.int32), stats

    eval_fn.__dict__.update(loss_fn.__dict__)
    return eval_fn

# Columns Unsloth never forwards to the chat template as keyword arguments.
_TEMPLATE_RESERVED_COLUMNS = frozenset((
    "prompt", "chosen", "rejected", "completion", "messages", "label",
    "images", "image", "videos", "video", "audios", "audio",
))


def render_grpo_prompt(tokenizer, row):
    """TRL's prompt text; a string column the chat template names is passed to
    it as a keyword, as Unsloth does."""
    prompt = row["prompt"]
    if not _is_conversational_prompt(prompt):
        if not isinstance(prompt, str):
            raise ValueError(
                "Unsloth MLX GRPO: a prompt must be a string or a list of chat messages."
            )
        return prompt
    template = getattr(tokenizer, "chat_template", None)
    kwargs = {
        key: value for key, value in row.items()
        if isinstance(value, str) and key not in _TEMPLATE_RESERVED_COLUMNS
        and isinstance(template, str) and key in template
    }
    messages = _normalize_mlx_messages(prompt, is_vlm=False)
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
    return _chat_template(tokenizer, messages, tools=None, kwargs=kwargs, **mode)


@dataclass(frozen=True)
class GRPOSampling:
    """Rollout sampling as TRL configures it; ``top_k`` / ``min_p`` None switch the
    filter off, and ``max_tokens`` is the resolved completion length."""

    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float = 1.0

    def __post_init__(self):
        # The ranges Hugging Face's sampling processors accept.
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
    """One row's HF sampling processors: repetition penalty over the prompt and
    what followed it, then temperature, top-k, top-p and min-p."""
    # The engine shows a processor its last prompt token onward.
    context = mx.array(list(prompt_ids[:-1]), dtype=mx.int32)
    penalty = float(sampling.repetition_penalty)

    def warp(tokens, logits):
        logits = logits.astype(mx.float32)
        if penalty != 1.0:
            seen = mx.concatenate([context, tokens.astype(mx.int32)])
            picked = logits[:, seen]
            logits[:, seen] = mx.where(picked < 0, picked * penalty, picked / penalty)
        if sampling.temperature != 1.0:
            logits = logits / sampling.temperature
        removed = mx.array(-float("inf"), mx.float32)
        vocab = logits.shape[-1]
        if sampling.top_k:
            kept = vocab - min(int(sampling.top_k), vocab)
            cutoff = mx.sort(logits, axis=-1)[:, kept:kept + 1]
            logits = mx.where(logits < cutoff, removed, logits)
        if sampling.top_p < 1.0:
            # By sorted position, as HF drops them: a value cutoff keeps every tie.
            # The most likely token always survives, as HF's min_tokens_to_keep=1.
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


class GRPOBatchPlan:
    """Finite GRPO schedule: one entry per micro-step, each naming the prompt chunk
    its generation samples from. A generation covers ``steps_per_generation *
    num_iterations`` micro-steps counted from ``start_at``, so a resumed run
    generates at once, as TRL does with an empty buffer."""

    def __init__(
        self, dataset, tokenizer, prompt_ids, max_tokens, schedule, *, cycle_length,
        num_generations, batch_size, steps_per_generation, num_iterations,
        sampling, score_temperature, scale_rewards, mask_truncated_completions,
        with_old_logps, with_reference, max_seq_length, seed, counts_tokens=True,
    ):
        self._dataset = dataset
        self._tokenizer = tokenizer
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
        self.eos_id = int(tokenizer.eos_token_id)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        self.pad_id = self.eos_id if pad_id is None else int(pad_id)
        self._anchor = 0
        self._current = None
        self._model = None
        self._shape_plan = None
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
        """Attach what generating needs: the policy, rewards and the reference.
        ``trainer_state`` reaches the reward functions and counts the prompt and
        completion tokens, as TRL's does; ``keep_completions`` holds the latest
        generation's texts in ``completions``; ``scorer`` is the loss's runtime CCE
        scorer, so old and reference log-probs are scored as the loss scores."""
        if self.with_reference and reference_policy is None:
            raise ValueError("Unsloth MLX GRPO: this plan scores a reference policy.")
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
        if step_compiled is not None:
            self._logps = _scored_as_step(self._logps, step_compiled)

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
        return tuple(sorted({*range(low, high, 32), high}))

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

    def _score(self, scorer, sequences, prompt_lengths):
        values = []
        for start in range(0, len(sequences), self.batch_size):
            chunk = sequences[start:start + self.batch_size]
            batch, lengths = self._pack(
                chunk, prompt_lengths[start:start + self.batch_size],
                self._planned(self._width(max(len(sequence) for sequence in chunk))),
            )
            logps = np.array(scorer(mx.array(batch), mx.array(lengths)), dtype=np.float32)
            values.extend(logps[row, :len(sequence) - 1] for row, sequence in enumerate(chunk))
        return values

    def _decode(self, ids):
        try:
            return self._tokenizer.decode(list(ids), skip_special_tokens=True)
        except TypeError:
            return self._tokenizer.decode(list(ids))

    def _generate(self, visit):
        if self._model is None:
            raise RuntimeError("Unsloth MLX GRPO: bind the plan before fetching batches.")
        prompts = [
            prompt for prompt in self._schedule[visit % len(self._schedule)]
            for _ in range(self.num_generations)
        ]
        requests = [
            GenerationRequest(
                prompt_token_ids=self._prompt_ids[prompt],
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
                self._model, self._tokenizer, requests, defaults=defaults,
            )
        cut = [
            cut_completion(result.token_ids, result.finish_reason, eos_id=self.eos_id, pad_id=self.pad_id)
            for result in results
        ]
        completion_ids = [ids for ids, _ in cut]
        truncated = np.array([flag for _, flag in cut], dtype=bool)
        rows = [self._dataset[prompt] for prompt in prompts]
        texts = [self._decode(ids) for ids in completion_ids]
        lengths = np.array([len(ids) for ids in completion_ids])
        prompt_lengths = np.array([len(self._prompt_ids[prompt]) for prompt in prompts])
        # Counted before rewards, which read it through trainer_state as TRL's do.
        if self.counts_tokens:
            self._trainer_state.num_input_tokens_seen += int(prompt_lengths.sum() + lengths.sum())
        self._metrics["num_tokens"] = [self._trainer_state.num_input_tokens_seen]
        reward_prompts, completions = reward_prompts_and_completions(
            [row["prompt"] for row in rows], texts,
        )
        rewards = calculate_rewards(
            self._reward_funcs, rows, prompts=reward_prompts, completions=completions,
            completion_ids=completion_ids, trainer_state=self._trainer_state,
        )
        advantages, reward_stats = group_advantages(
            rewards, self._reward_weights, self.num_generations, self.scale_rewards,
        )
        self._record_generation(lengths, truncated, rewards, reward_stats)
        if self._keep_completions:
            self.completions = {
                "prompt": [self._decode(self._prompt_ids[prompt]) for prompt in prompts],
                "completion": texts,
                "rewards": {
                    reward_function_name(reward_func): rewards[:, column].tolist()
                    for column, reward_func in enumerate(self._reward_funcs)
                },
                "advantage": advantages.tolist(),
            }
        kept = ~truncated if self.mask_truncated_completions else np.ones_like(truncated)
        # Shuffled before scoring, so each old/reference chunk is one micro-batch at
        # its own width and a first iteration's ratio is exactly 1 when eager.
        order = np.random.default_rng(_derived_seed(self.seed, visit, 1)).permutation(len(prompts))
        sequences = [
            tuple(self._prompt_ids[prompts[row]]) + tuple(completion_ids[row]) for row in order
        ]
        prompt_lengths = prompt_lengths[order]
        old = ref = None
        if self.with_old_logps:
            old = self._score(
                lambda batch, lengths: mx.stop_gradient(self._logps(self._model, batch, lengths)),
                sequences, prompt_lengths,
            )
        if self.with_reference:
            ref = self._score(
                lambda batch, lengths: self._reference_policy.forward(
                    self._model, batch, lengths, response_scorer=self._logps,
                ),
                sequences, prompt_lengths,
            )
        return _Generation(
            visit=visit, sequences=sequences, prompt_lengths=prompt_lengths,
            advantages=advantages[order], kept=kept[order].astype(np.int32), old=old, ref=ref,
            generation_tokens=int(lengths.sum()),
        )

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
        return (
            "grpo_tuple_7", ((rows, "sequence"), "int32"), ((rows, 2), "int32"),
            ((2,), "float32"), ((rows,), "float32"), ((rows,), "int32"),
            held if self.with_old_logps else None, held if self.with_reference else None,
        )

    def set_shape_plan(self, shape_plan):
        if getattr(shape_plan.report, "action", None) not in ("exact", "bucket"):
            raise ValueError("only exact or bucket shape plans can be installed")
        self._shape_plan = shape_plan

    def _planned(self, width):
        # Old and reference log-probs are scored at the width the micro-batch
        # trains at, so a first iteration's ratio stays exactly 1, including
        # after a compile fallback.
        if self._shape_plan is None:
            return width
        return self._shape_plan.endpoint_for(self.batch_family(0), width)

    def materialize(self, index, *, phase=None):
        """The micro-batch for schedule entry ``index``, at its planned width
        whatever the ``phase``. Refetching an entry reuses its generation instead
        of sampling again."""
        start = self._generation_start(index)
        if self._current is None or self._current.visit != start:
            self._current = self._generate(start)
        generation = self._current
        position = ((index - start) % self.steps_per_generation) * self.batch_size
        rows = range(position, min(position + self.batch_size, len(generation.sequences)))
        sequences = [generation.sequences[row] for row in rows]
        width = self._planned(self._width(max(len(sequence) for sequence in sequences)))
        batch, lengths = self._pack(sequences, generation.prompt_lengths[position:position + self.batch_size], width)

        def held(values):
            if values is None:
                return None
            table = np.zeros((len(sequences), width - 1), dtype=np.float32)
            for offset, row in enumerate(rows):
                table[offset, :len(values[row])] = values[row]
            return mx.array(table)

        return (
            mx.array(batch), mx.array(lengths),
            mx.array([self.steps_per_generation, generation.generation_tokens], dtype=mx.float32),
            mx.array(generation.advantages[position:position + self.batch_size]),
            mx.array(generation.kept[position:position + self.batch_size]),
            held(generation.old), held(generation.ref),
        )

    def __getitem__(self, index):
        return self.materialize(index)

    def __iter__(self):
        # One pass, sampling afresh: an evaluation never reuses an earlier one's rollouts.
        self._current = None
        return (self.materialize(index) for index in range(len(self)))


def _encode_grpo_prompts(dataset, tokenizer, sampling, max_seq_length, max_prompt_length):
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise ValueError("Unsloth MLX GRPO: the tokenizer needs an eos_token_id to stop on.")
    prompt_ids, max_tokens = [], []
    long_prompts = capped = 0
    for index in range(len(dataset)):
        row = dataset[index]
        if not isinstance(row, Mapping) or "prompt" not in row:
            raise ValueError(
                f"Unsloth MLX GRPO: dataset row {index} has no prompt column."
            )
        ids = tuple(int(token) for token in encode_mlx_text(tokenizer, render_grpo_prompt(tokenizer, row)))
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
    return prompt_ids, max_tokens


def create_grpo_batch_plan(
    dataset, tokenizer, *, num_generations, batch_size, steps_per_generation,
    num_iterations, grad_accum, sampling, max_seq_length, score_temperature=None, scale_rewards="group",
    mask_truncated_completions=False, with_reference=False, max_prompt_length=None,
    shuffle=True, seed=None, num_batches=None, num_epochs=None,
):
    """Encode every prompt once and lay out TRL's RepeatSampler schedule.

    ``batch_size`` counts completions; a generation samples ``batch_size *
    steps_per_generation`` of them, ``num_generations`` per prompt. Log-probs
    are scored at ``score_temperature``, the sampling temperature by default.
    """
    prompt_ids, max_tokens = _encode_grpo_prompts(
        dataset, tokenizer, sampling, max_seq_length, max_prompt_length,
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
    )



def create_grpo_eval_plan(
    dataset, tokenizer, *, num_generations, batch_size, sampling, max_seq_length,
    score_temperature=None, scale_rewards="group", mask_truncated_completions=False,
    with_reference=False, max_prompt_length=None, seed=None,
):
    """TRL's eval RepeatSampler order, one generation per ``batch_size`` completions
    (the last batch may be short), on a seed stream of its own and without
    advancing ``num_input_tokens_seen``."""
    prompt_ids, max_tokens = _encode_grpo_prompts(
        dataset, tokenizer, sampling, max_seq_length, max_prompt_length,
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
        seed=_derived_seed(seed, 1), counts_tokens=False,
    )

_GENERATION_KWARGS = (
    "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "max_new_tokens",
    "do_sample",
)


def grpo_sampling(config):
    """The rollout sampling a GRPO config asks for; ``generation_kwargs`` override
    the named fields, as TRL updates its GenerationConfig with them."""
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


def grpo_reward_funcs(reward_funcs):
    funcs = list(reward_funcs) if isinstance(reward_funcs, (list, tuple)) else [reward_funcs]
    if not funcs:
        raise ValueError("Unsloth MLX GRPO: reward_funcs needs at least one reward function.")
    for func in funcs:
        if not callable(func) or hasattr(func, "parameters"):
            raise ValueError(
                f"Unsloth MLX GRPO: reward function {func!r} is not a plain callable; "
                "reward models are not supported."
            )
    return funcs


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
    grpo_objective(config)
    grpo_sampling(config)


def format_completions(sample, step, limit=None):
    """A plain-text take on TRL's ``print_prompt_completions_sample`` table: the
    first ``limit`` rows, where TRL draws them at random."""
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
