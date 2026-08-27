"""MLX text preference tokenization, finite batching, and objectives."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np

from .utils import (
    _FiniteVisitMixin,
    _encode_mlx_prompt_completion,
    _finite_batch_schedule,
    _finite_row_schedule,
    _finite_text_pad_width,
    _get_mlx_dropout_probability,
    _model_logits,
    _normalize_mlx_messages,
    _normalize_seed,
    _torch_randperm_order,
    collect_mlx_lora_adapter_tensors,
    encode_mlx_text,
    iter_mlx_lora_modules,
)


@dataclass(frozen=True)
class TokenizedPreferenceRow:
    """Branch-specific prompt boundaries and response continuations."""

    chosen_prompt_ids: tuple
    chosen_ids: tuple
    rejected_prompt_ids: tuple
    rejected_ids: tuple

    @property
    def chosen(self):
        return self.chosen_prompt_ids + self.chosen_ids

    @property
    def rejected(self):
        return self.rejected_prompt_ids + self.rejected_ids


def _is_messages(value):
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(message, Mapping)
            and "role" in message
            and "content" in message
            for message in value
        )
    )


def _chat_template(tokenizer, messages, *, tools, kwargs, **mode):
    options = dict(kwargs or {})
    if tools is not None:
        options["tools"] = tools
    return tokenizer.apply_chat_template(
        messages, tokenize=False, **mode, **options,
    )


def _append_to_assistant(prompt, completion):
    if not completion or completion[0].get("role") != "assistant":
        raise ValueError(
            "Unsloth MLX preference: an assistant-ended prompt requires each "
            "completion to start with an assistant message."
        )
    merged = [dict(message) for message in prompt]
    merged[-1]["content"] = (
        str(merged[-1].get("content", ""))
        + str(completion[0].get("content", ""))
    )
    merged.extend(dict(message) for message in completion[1:])
    return merged


def _render_preference_row(tokenizer, row):
    prompt = row["prompt"]
    chosen = row["chosen"]
    rejected = row["rejected"]
    kinds = (_is_messages(prompt), _is_messages(chosen), _is_messages(rejected))
    if any(kinds) and not all(kinds):
        raise ValueError(
            "Unsloth MLX preference: prompt, chosen, and rejected must all be "
            "strings or all be message lists."
        )
    if not any(kinds):
        if not all(isinstance(value, str) for value in (prompt, chosen, rejected)):
            raise ValueError(
                "Unsloth MLX preference: prompt, chosen, and rejected must be strings."
            )
        return prompt, prompt + chosen, prompt + rejected

    prompt = _normalize_mlx_messages(prompt, is_vlm=False)
    chosen = _normalize_mlx_messages(chosen, is_vlm=False)
    rejected = _normalize_mlx_messages(rejected, is_vlm=False)
    role = prompt[-1].get("role")
    tools = row.get("tools")
    kwargs = row.get("chat_template_kwargs")
    if role == "user":
        prompt_text = _chat_template(
            tokenizer, prompt, tools=tools, kwargs=kwargs,
            add_generation_prompt=True,
        )
        chosen_messages = prompt + chosen
        rejected_messages = prompt + rejected
    elif role == "assistant":
        prompt_text = _chat_template(
            tokenizer, prompt, tools=tools, kwargs=kwargs,
            continue_final_message=True,
        )
        chosen_messages = _append_to_assistant(prompt, chosen)
        rejected_messages = _append_to_assistant(prompt, rejected)
    else:
        raise ValueError(
            "Unsloth MLX preference: a conversational prompt must end with a "
            "user or assistant message."
        )
    return (
        prompt_text,
        _chat_template(
            tokenizer, chosen_messages, tools=tools, kwargs=kwargs,
            add_generation_prompt=False,
        ),
        _chat_template(
            tokenizer, rejected_messages, tools=tools, kwargs=kwargs,
            add_generation_prompt=False,
        ),
    )


def _require_preference_fields(row):
    if not isinstance(row, Mapping):
        raise ValueError("Unsloth MLX preference: each dataset row must be a mapping.")
    missing = {"prompt", "chosen", "rejected"} - set(row)
    if missing:
        raise ValueError(
            "Unsloth MLX preference: every row requires explicit prompt, chosen, "
            f"and rejected fields; missing {sorted(missing)!r}."
        )


def tokenize_preference_row(
    tokenizer, row, *, max_seq_length, append_eos=True, rendered=None,
):
    """Tokenize one row and preserve a shared prompt boundary.

    ``rendered`` passes back a ``_render_preference_row`` result the caller
    already has: a chat template may carry state, so rendering twice is not
    guaranteed to give the same text.
    """
    _require_preference_fields(row)
    prompt_text, chosen_text, rejected_text = (
        _render_preference_row(tokenizer, row) if rendered is None else rendered
    )
    chosen = _encode_mlx_prompt_completion(
        tokenizer, prompt_text, chosen_text, append_eos=append_eos,
    )
    rejected = _encode_mlx_prompt_completion(
        tokenizer, prompt_text, rejected_text, append_eos=append_eos,
    )
    chosen_prompt = list(chosen.input_ids[:chosen.prompt_length])
    rejected_prompt = list(rejected.input_ids[:rejected.prompt_length])
    chosen_response = list(chosen.input_ids[chosen.prompt_length:])
    rejected_response = list(rejected.input_ids[rejected.prompt_length:])
    if not chosen_response or not rejected_response:
        raise ValueError(
            "Unsloth MLX preference: chosen and rejected must each contain at "
            "least one response token."
        )
    max_response = max(len(chosen_response), len(rejected_response))
    if max_response >= max_seq_length:
        chosen_response = chosen_response[:max_seq_length - 1]
        rejected_response = rejected_response[:max_seq_length - 1]
        chosen_prompt = chosen_prompt[-1:]
        rejected_prompt = rejected_prompt[-1:]
    else:
        prompt_budget = max_seq_length - max_response
        chosen_prompt = chosen_prompt[-prompt_budget:]
        rejected_prompt = rejected_prompt[-prompt_budget:]
    if (
        not chosen.prompt_ids
        or not rejected.prompt_ids
        or not chosen_response
        or not rejected_response
    ):
        raise ValueError(
            "Unsloth MLX preference: max_seq_length leaves no trainable prompt "
            "and response tokens."
        )
    return TokenizedPreferenceRow(
        tuple(chosen_prompt), tuple(chosen_response),
        tuple(rejected_prompt), tuple(rejected_response),
    )


def encode_generation_prompt(tokenizer, row, *, max_seq_length, max_new_tokens):
    """Encode one row's prompt for generation, reserving room for the sample.

    ``max_new_tokens`` comes out of ``max_seq_length`` and the prompt keeps its
    end. Not ``tokenize_preference_row``'s prompt: that boundary hands a merged
    prompt/completion token to the response span, and generation has no
    completion to merge with. Not the generation engine's own encoder either,
    which omits the special tokens training adds.
    """
    _require_preference_fields(row)
    prompt_text, _, _ = _render_preference_row(tokenizer, row)
    return encode_generation_prompt_text(
        tokenizer, prompt_text,
        max_seq_length=max_seq_length, max_new_tokens=max_new_tokens,
    )


def encode_generation_prompt_text(
    tokenizer, prompt_text, *, max_seq_length, max_new_tokens,
):
    """Encode an already-rendered prompt, reserving room for the sample."""
    budget = int(max_seq_length) - int(max_new_tokens)
    if budget < 1:
        raise ValueError(
            "Unsloth MLX preference: max_new_tokens must leave at least one "
            "prompt token within max_seq_length."
        )
    prompt_ids = [int(token) for token in encode_mlx_text(tokenizer, prompt_text)]
    if not prompt_ids:
        raise ValueError(
            "Unsloth MLX preference: a row rendered an empty prompt."
        )
    return prompt_text, tuple(prompt_ids[-budget:])


class FinitePreferenceBatchPlan(_FiniteVisitMixin):
    """CPU-backed preference schedule with lazy MLX materialization."""

    __slots__ = (
        "_rows", "_schedule", "_normalizers", "_widths", "_cycle_length",
        "max_seq_length", "pad_id", "_shape_plan", "_visit_policy",
        "_visit_seed", "_visit_epoch_cache",
    )

    def __init__(
        self, rows, schedule, *, normalizers, cycle_length, max_seq_length, pad_id,
    ):
        self._rows = tuple(rows)
        self._schedule = tuple(tuple(batch) for batch in schedule)
        self._normalizers = tuple(tuple(values) for values in normalizers)
        self._cycle_length = int(cycle_length)
        self.max_seq_length = int(max_seq_length)
        self.pad_id = int(pad_id)
        self._shape_plan = None
        self._visit_policy = "identity"
        self._visit_seed = None
        self._visit_epoch_cache = None
        self._widths = tuple(self._raw_width(batch) for batch in self._schedule)

    @property
    def rows(self):
        return self._rows

    @property
    def schedule(self):
        return self._schedule

    @property
    def widths(self):
        return self._widths

    @property
    def cycle_length(self):
        return self._cycle_length

    def __len__(self):
        return len(self._schedule)

    def _raw_width(self, batch):
        raw = max(
            max(len(self._rows[index].chosen), len(self._rows[index].rejected))
            for index in batch
        )
        return _finite_text_pad_width(
            raw, pad_to_multiple=32, minimum_width=2,
            max_seq_length=self.max_seq_length,
        )

    def batch_width(self, index):
        return self._widths[index]

    def batch_family(self, index):
        pairs = len(self._schedule[index])
        return (
            "preference_tuple_3",
            ((2 * pairs, "sequence"), "int32"),
            ((2 * pairs, 2), "int32"),
            ((3,), "int32"),
        )

    def set_shape_plan(self, shape_plan):
        if getattr(shape_plan.report, "action", None) not in ("exact", "bucket"):
            raise ValueError("only exact or bucket shape plans can be installed")
        self._shape_plan = shape_plan

    def materialize(self, index, *, phase=None):
        indices = self._schedule[index]
        rows = [self._rows[item] for item in indices]
        width = self._widths[index]
        if self._shape_plan is not None and phase is not None:
            family = self.batch_family(index)
            width = self._shape_plan.endpoint_for(family, width)
        batch = np.full((2 * len(rows), width), self.pad_id, dtype=np.int32)
        lengths = np.zeros((2 * len(rows), 2), dtype=np.int32)
        for offset, row in enumerate(rows):
            for output_row, values, prompt_length in (
                (offset, row.chosen, len(row.chosen_prompt_ids)),
                (
                    offset + len(rows), row.rejected,
                    len(row.rejected_prompt_ids),
                ),
            ):
                size = min(len(values), width)
                batch[output_row, :size] = values[:size]
                lengths[output_row] = (prompt_length, size)
        return mx.array(batch), mx.array(lengths), mx.array(
            self._normalizers[index], dtype=mx.int32,
        )

    def __getitem__(self, index):
        return self.materialize(index)

    def materialize_all(self):
        return [self[index] for index in range(len(self))]


def _window_normalizers(rows, schedule, cycle_length, grad_accum):
    values = []
    accum = max(1, int(grad_accum or 1))
    for position in range(len(schedule)):
        epoch_start = (position // cycle_length) * cycle_length
        epoch_end = min(epoch_start + cycle_length, len(schedule))
        window_start = epoch_start + ((position - epoch_start) // accum) * accum
        window_end = min(window_start + accum, epoch_end)
        batches = schedule[window_start:window_end]
        pair_count = sum(len(batch) for batch in batches)
        chosen_tokens = sum(
            max(0, len(rows[index].chosen) - 1)
            for batch in batches for index in batch
        )
        values.append((chosen_tokens, pair_count, window_end - window_start))
    return values


def create_preference_batch_plan(
    dataset,
    tokenizer,
    *,
    batch_size,
    max_seq_length,
    num_batches=None,
    num_epochs=None,
    grad_accum=None,
    dataset_order="default",
    preserve_dataset_order=False,
    seed=None,
    append_eos=True,
    formatting_func=None,
    prompt_sink=None,
):
    """Build the finite preference plan consumed by both objectives.

    ``prompt_sink`` receives each row's rendered prompt text as the plan is
    built, so a caller wanting it does not render the row a second time.
    """
    rows = []
    for raw in dataset:
        row = formatting_func(raw) if formatting_func is not None else raw
        if not isinstance(row, Mapping):
            raise ValueError(
                "Unsloth MLX preference: formatting_func must return a mapping."
            )
        rendered = None
        if prompt_sink is not None:
            _require_preference_fields(row)
            rendered = _render_preference_row(tokenizer, row)
            prompt_sink(rendered[0])
        rows.append(tokenize_preference_row(
            tokenizer, row, max_seq_length=max_seq_length, append_eos=append_eos,
            rendered=rendered,
        ))
    if not rows:
        raise ValueError("Unsloth MLX preference: the training dataset is empty.")
    order_mode = "sequential" if preserve_dataset_order else dataset_order
    if order_mode not in ("default", "sequential", "torch_randperm"):
        raise ValueError(
            f"Unsloth MLX preference: unsupported dataset_order {order_mode!r}."
        )

    def order_for_epoch(epoch):
        if order_mode == "torch_randperm":
            return _torch_randperm_order(
                len(rows), _normalize_seed(seed) + int(epoch),
            )
        order = list(range(len(rows)))
        if order_mode == "default":
            order.sort(key=lambda index: max(
                len(rows[index].chosen), len(rows[index].rejected),
            ))
        return order

    if order_mode == "default":
        def batches_for_epoch(epoch):
            order = order_for_epoch(0)
            blocks = [
                tuple(order[start : start + batch_size])
                for start in range(0, len(order), batch_size)
            ]
            rng = np.random.RandomState(_normalize_seed(seed) + int(epoch))
            rng.shuffle(blocks)
            return blocks

        schedule, cycle_length = _finite_batch_schedule(
            batches_for_epoch,
            num_batches=num_batches,
            num_epochs=num_epochs,
            grad_accum=grad_accum,
        )
    else:
        schedule, cycle_length = _finite_row_schedule(
            len(rows), batch_size, order_for_epoch=order_for_epoch,
            num_batches=num_batches, num_epochs=num_epochs, grad_accum=grad_accum,
        )
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    return FinitePreferenceBatchPlan(
        rows, schedule,
        normalizers=_window_normalizers(rows, schedule, cycle_length, grad_accum),
        cycle_length=cycle_length,
        max_seq_length=max_seq_length,
        pad_id=0 if pad_id is None else pad_id,
    )


def _response_mask(targets, lengths):
    steps = mx.arange(1, targets.shape[1] + 1)
    return mx.logical_and(
        steps >= lengths[:, 0:1], steps < lengths[:, 1:]
    ).astype(mx.float32)


def _supervised_tokens(batch_data):
    lengths = batch_data[1]
    return mx.maximum(
        lengths[:, 1] - mx.maximum(lengths[:, 0], mx.array(1)),
        mx.array(0),
    ).sum()


def _log_sigmoid(value):
    return -mx.logaddexp(mx.array(0.0, dtype=value.dtype), -value)


def _orpo_log_odds(chosen, rejected):
    chosen = chosen.astype(mx.float32)
    rejected = rejected.astype(mx.float32)
    floor = mx.array(1e-12, dtype=mx.float32)
    chosen_odds = chosen - mx.log(mx.maximum(-mx.expm1(chosen), floor))
    rejected_odds = rejected - mx.log(mx.maximum(-mx.expm1(rejected), floor))
    return chosen_odds - rejected_odds


def _orpo_terms(chosen, rejected):
    return -_log_sigmoid(_orpo_log_odds(chosen, rejected))


def make_orpo_loss_fn(beta=0.1):
    """Create an ORPO loss with exact logical-window normalization."""
    beta = float(beta)

    def loss_fn(model, batch, lengths, normalizers):
        targets = batch[:, 1:]
        ce = nn.losses.cross_entropy(
            _model_logits(model(batch[:, :-1])), targets, reduction="none",
        ).reshape(targets.shape)
        mask = _response_mask(targets, lengths)
        response_logp = -(ce * mask).sum(axis=1) / mx.maximum(
            mask.sum(axis=1), mx.array(1.0),
        )
        pairs = batch.shape[0] // 2
        nll_mask = (
            mx.arange(1, targets.shape[1] + 1) < lengths[:pairs, 1:]
        ).astype(mx.float32)
        nll_sum = (ce[:pairs] * nll_mask).sum()
        odds_sum = _orpo_terms(
            response_logp[:pairs], response_logp[pairs:],
        ).sum()
        nll_tokens, window_pairs, window_microbatches = normalizers
        loss = window_microbatches.astype(mx.float32) * (
            nll_sum / mx.maximum(nll_tokens, mx.array(1)).astype(mx.float32)
            + beta * odds_sum / mx.maximum(
                window_pairs, mx.array(1),
            ).astype(mx.float32)
        )
        return loss, mx.array(1, dtype=mx.int32)

    loss_fn._unsloth_supervised_tokens = _supervised_tokens
    return loss_fn


class PreferenceRunContext:
    """Run-scoped dropout state with exact restoration."""

    def __init__(self, model, enabled=True):
        self._saved = []
        if not enabled:
            return
        dropout_types = tuple(
            value for value in (
                getattr(nn, "Dropout", None),
                getattr(nn, "Dropout2d", None),
                getattr(nn, "Dropout3d", None),
            )
            if isinstance(value, type)
        )
        named_modules = getattr(model, "named_modules", None)
        try:
            for _, module in named_modules() if callable(named_modules) else ():
                if not (
                    (dropout_types and isinstance(module, dropout_types))
                    or getattr(module, "_p_1", None) is not None
                    or type(module).__name__.endswith("Dropout")
                ):
                    continue
                if _get_mlx_dropout_probability(module) <= 0:
                    continue
                state = {
                    "_unsloth_preference_original_probability": getattr(
                        module, "_unsloth_preference_original_probability", None,
                    )
                }
                self._saved.append((module, state))
                module._unsloth_preference_original_probability = (
                    _get_mlx_dropout_probability(module)
                )
                if hasattr(module, "_p_1"):
                    state["_p_1"] = module._p_1
                    module._p_1 = 1.0
                if hasattr(module, "p"):
                    state["p"] = module.p
                    module.p = 0.0
        except BaseException:
            self.restore()
            raise

    def restore(self):
        while self._saved:
            module, state = self._saved.pop()
            for name, value in state.items():
                if (
                    name == "_unsloth_preference_original_probability"
                    and value is None
                ):
                    delattr(module, name)
                else:
                    setattr(module, name, value)


class LoRAReferencePolicy:
    """Temporarily expose the run's frozen base policy for one forward."""

    def __init__(self, modules, neftune_modules=()):
        self.modules = tuple(modules)
        self.neftune_modules = tuple(neftune_modules)

    def forward(self, model, batch, lengths):
        scales = [module.scale for module in self.modules]
        noise = [
            getattr(module, "_neftune_noise_enabled", True)
            for module in self.neftune_modules
        ]
        try:
            for module in self.modules:
                module.scale = 0.0
            for module in self.neftune_modules:
                module._neftune_noise_enabled = False
            values = _response_logps(model, batch, lengths)
            return mx.stop_gradient(values)
        finally:
            for module, scale in zip(self.modules, scales):
                module.scale = scale
            for module, enabled in zip(self.neftune_modules, noise):
                module._neftune_noise_enabled = enabled


def _response_logps(model, batch, lengths):
    targets = batch[:, 1:]
    ce = nn.losses.cross_entropy(
        _model_logits(model(batch[:, :-1])), targets, reduction="none",
    ).reshape(targets.shape)
    return -(ce * _response_mask(targets, lengths)).sum(axis=1)


def make_dpo_loss_fn(
    *, beta=0.1, label_smoothing=0.0, reference_policy=None, reference_free=False,
):
    """Create sigmoid DPO with conservative preference-label smoothing."""
    beta = float(beta)
    epsilon = float(label_smoothing)

    def loss_fn(model, batch, lengths, normalizers):
        policy = _response_logps(model, batch, lengths)
        pairs = batch.shape[0] // 2
        if reference_free:
            reference = mx.zeros(policy.shape, dtype=policy.dtype)
        else:
            reference = reference_policy.forward(model, batch, lengths)
        logits = beta * (
            (policy[:pairs] - policy[pairs:])
            - (reference[:pairs] - reference[pairs:])
        )
        pair_loss = -(
            (1.0 - epsilon) * _log_sigmoid(logits)
            + epsilon * _log_sigmoid(-logits)
        )
        _, window_pairs, window_microbatches = normalizers
        loss = window_microbatches.astype(mx.float32) * pair_loss.sum() / mx.maximum(
            window_pairs, mx.array(1),
        ).astype(mx.float32)
        return loss, mx.array(1, dtype=mx.int32)

    loss_fn._unsloth_supervised_tokens = _supervised_tokens
    return loss_fn


_SHARED_EVAL_METRICS = (
    "rewards/chosen",
    "rewards/rejected",
    "rewards/accuracies",
    "rewards/margins",
    "logps/chosen",
    "logps/rejected",
    "logits/chosen",
    "logits/rejected",
)

# The names TRL logs for these objectives, so a run reads the same on
# either backend. ORPO adds the three terms its loss decomposes into.
PREFERENCE_EVAL_METRICS = {
    "dpo": _SHARED_EVAL_METRICS,
    "orpo": _SHARED_EVAL_METRICS + (
        "nll_loss", "log_odds_ratio", "log_odds_chosen",
    ),
}


def _masked_logit_sum(logits, mask):
    """Logit sum over the response positions, and the count it divides by.

    Separate so the eval set's mean is taken over all of its positions at once.
    """
    kept = mask.sum() * logits.shape[-1]
    # The float32 mask promotes the product, so this reduction is already exact.
    return (logits * mask[..., None]).sum(), kept


def _orpo_logit_sum(logits):
    """Logit sum over every position, and the count it divides by.

    Cast before reducing: mx.sum accumulates in the input dtype, and a bf16
    batch holds millions of logits, far past where an 8-bit mantissa stops
    registering the next addend. Casting the result would already be too late.
    """
    return logits.astype(mx.float32).sum(), mx.array(float(math.prod(logits.shape)))


# Metric index -> index of the stats entry it is divided by. Anything absent is
# a per-pair sum over the pair count. These are means over tokens, which no pair
# count recovers once batches hold different numbers of them.
PREFERENCE_EVAL_DENOMINATORS = {
    "dpo": {6: 8, 7: 9},
    "orpo": {6: 11, 7: 12, 8: 13},
}

# Reported metrics, then the denominators the trainer sums alongside them.
PREFERENCE_EVAL_STATS_WIDTH = {
    kind: len(names) + len(PREFERENCE_EVAL_DENOMINATORS[kind])
    for kind, names in PREFERENCE_EVAL_METRICS.items()
}


def _preference_forward(model, batch, lengths):
    """Logits, per-token cross entropy, and the response mask for one batch."""
    targets = batch[:, 1:]
    logits = model(batch[:, :-1])
    ce = nn.losses.cross_entropy(
        logits, targets, reduction="none",
    ).reshape(targets.shape)
    return logits, ce, _response_mask(targets, lengths)


def make_preference_eval_fn(
    kind, *, beta=0.1, label_smoothing=0.0,
    reference_policy=None, reference_free=False,
):
    """Score one preference batch as ``(loss, pairs, stats)``.

    ``stats`` holds a numerator per metric in ``PREFERENCE_EVAL_METRICS[kind]``
    order, then the denominators ``PREFERENCE_EVAL_DENOMINATORS[kind]`` names.
    Nothing is a per-batch mean: summing the vector across batches and dividing
    each numerator by its own total averages the eval set exactly, however
    unlike the batches are.
    """
    beta = float(beta)
    epsilon = float(label_smoothing)

    def eval_fn(model, batch, lengths, _normalizers=None):
        logits, ce, mask = _preference_forward(model, batch, lengths)
        pairs = batch.shape[0] // 2
        logps = -(ce * mask).sum(axis=1)
        # TRL is not consistent between its trainers: DPOTrainer averages its
        # logit metric over the completion positions only, ORPOTrainer over the
        # whole sequence. Follow each, so either compares against its own run.
        if kind == "orpo":
            chosen_logits, chosen_logit_count = _orpo_logit_sum(logits[:pairs])
            rejected_logits, rejected_logit_count = _orpo_logit_sum(logits[pairs:])
        else:
            chosen_logits, chosen_logit_count = _masked_logit_sum(
                logits[:pairs], mask[:pairs])
            rejected_logits, rejected_logit_count = _masked_logit_sum(
                logits[pairs:], mask[pairs:])
        if kind == "orpo":
            logps = logps / mx.maximum(mask.sum(axis=1), mx.array(1.0))
        chosen, rejected = logps[:pairs], logps[pairs:]
        if kind == "orpo":
            nll_mask = (
                mx.arange(1, batch.shape[1]) < lengths[:pairs, 1:]
            ).astype(mx.float32)
            nll_tokens = nll_mask.sum()
            nll_sum = (ce[:pairs] * nll_mask).sum()
            nll = nll_sum / mx.maximum(nll_tokens, mx.array(1.0))
            log_odds = _orpo_log_odds(chosen, rejected)
            ratio = _log_sigmoid(log_odds)
            loss = nll - beta * ratio.mean()
            chosen_rewards = beta * chosen
            rejected_rewards = beta * rejected
            extra = [nll_sum, ratio.sum(), log_odds.sum()]
            denominators = [
                chosen_logit_count, rejected_logit_count, nll_tokens,
            ]
        else:
            if reference_free:
                reference = mx.zeros(logps.shape, dtype=logps.dtype)
            else:
                reference = reference_policy.forward(model, batch, lengths)
            margin = beta * (
                (chosen - rejected) - (reference[:pairs] - reference[pairs:])
            )
            loss = -(
                (1.0 - epsilon) * _log_sigmoid(margin)
                + epsilon * _log_sigmoid(-margin)
            ).mean()
            chosen_rewards = beta * (chosen - reference[:pairs])
            rejected_rewards = beta * (rejected - reference[pairs:])
            extra = []
            denominators = [chosen_logit_count, rejected_logit_count]
        stats = [
            chosen_rewards.sum(),
            rejected_rewards.sum(),
            (chosen_rewards > rejected_rewards).astype(mx.float32).sum(),
            (chosen_rewards - rejected_rewards).sum(),
            chosen.sum(),
            rejected.sum(),
            chosen_logits,
            rejected_logits,
        ] + extra + denominators
        return (
            loss,
            mx.array(pairs, dtype=mx.int32),
            mx.stack([value.astype(mx.float32) for value in stats]),
        )

    eval_fn._unsloth_preference_metrics = PREFERENCE_EVAL_METRICS[kind]
    eval_fn._unsloth_preference_denominators = PREFERENCE_EVAL_DENOMINATORS[kind]
    eval_fn._unsloth_preference_stats_width = PREFERENCE_EVAL_STATS_WIDTH[kind]
    return eval_fn


def lora_modules_have_nonzero_delta(modules):
    def has_values(value):
        if value is not None and not hasattr(value, "shape"):
            value = getattr(value, "weight", None)
        if value is None or not hasattr(value, "shape"):
            return True
        try:
            result = mx.any(value != 0)
            mx.eval(result)
            return bool(result.item())
        except Exception:
            return True

    return any(
        has_values(module.lora_a) and has_values(module.lora_b)
        for module in modules
    )


def build_reference_policy(model, *, reference_free, resume_provenance, neftune=()):
    """Validate and construct the only supported referenced-DPO policy."""
    if reference_free:
        return None, {"kind": "reference_free"}
    named_modules = list(iter_mlx_lora_modules(model))
    if not named_modules:
        raise ValueError(
            "Unsloth MLX DPO: referenced training requires plain LoRA adapters. "
            "Use reference_free=True for full fine-tuning."
        )
    if any(type(module).__name__.startswith("DoRA") for _, module in named_modules):
        raise ValueError(
            "Unsloth MLX DPO: referenced training does not support DoRA because "
            "zeroing LoRA scale does not recover the initial policy."
        )
    adapters = sorted(collect_mlx_lora_adapter_tensors(model))
    trainable = sorted(dict(mlx.utils.tree_flatten(model.trainable_parameters())))
    if not trainable or not any(name in adapters for name in trainable):
        raise ValueError(
            "Unsloth MLX DPO: referenced training requires at least one "
            "trainable LoRA adapter tensor."
        )
    if any(name not in adapters for name in trainable):
        raise ValueError(
            "Unsloth MLX DPO: referenced training requires LoRA-only trainable "
            "parameters so the reference cannot drift."
        )
    provenance = {
        "kind": "plain_lora_base",
        "base_repo": getattr(model, "_hf_repo", None),
        "base_revision": getattr(model, "_unsloth_base_revision", None),
        "base_commit": getattr(model, "_unsloth_base_commit_hash", None),
        "adapter_modules": [name for name, _ in named_modules],
    }
    if resume_provenance is None:
        if lora_modules_have_nonzero_delta([module for _, module in named_modules]):
            raise ValueError(
                "Unsloth MLX DPO: referenced training requires a fresh zero-delta "
                "LoRA adapter or a checkpoint from this same run."
            )
    elif resume_provenance != provenance:
        raise ValueError(
            "Unsloth MLX DPO: the checkpoint reference provenance does not match "
            "this model and adapter layout."
        )
    return LoRAReferencePolicy(
        [module for _, module in named_modules], neftune,
    ), provenance
