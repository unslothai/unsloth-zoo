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

"""
MLXTrainer — drop-in trainer for Apple Silicon, mirroring SFTTrainer's API.

Usage mirrors TRL notebooks:

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    trainer = MLXTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=MLXTrainingConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=60,
            learning_rate=2e-4,
            use_cce=True,
        ),
    )
    trainer.train()
"""

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass, replace
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_reduce, tree_unflatten

_PAD_MULTIPLE = 32
SUPPORTED_MLX_OPTIMIZERS = ("adafactor", "adamw", "adam", "sgd", "muon", "lion")
SUPPORTED_MLX_LR_SCHEDULERS = ("linear", "cosine", "constant")


def _mlx_distributed_backend_from_env():
    """Return an explicit distributed backend implied by MLX launch env."""
    if os.environ.get("MLX_JACCL_COORDINATOR") and os.environ.get("MLX_IBV_DEVICES"):
        return "jaccl"
    return None


def _mlx_rank0_resolve_int(comm_group, resolver, context):
    """Resolve source metadata on rank 0 and synchronize its value or failure."""
    rank, world_size = _distributed_rank_size(comm_group)
    if world_size <= 1:
        return int(resolver())

    status = 0
    value = 0
    owner_error = None
    if rank == 0:
        try:
            value = int(resolver())
            status = 1
        except BaseException as exc:
            # Synchronize the failure (interrupts included) before it
            # propagates: peers block in the metadata collective and would hang
            # if rank 0 unwound past it. Re-raised on rank 0 below.
            owner_error = exc
            status = -1
    metadata = mx.array(
        [status, value] if rank == 0 else [0, 0], dtype=mx.int64,
    )
    metadata = mx.distributed.all_sum(metadata, group=comm_group)
    mx.eval(metadata)
    status, value = (int(item) for item in metadata.tolist())
    if status < 0:
        if owner_error is not None:
            raise owner_error
        raise RuntimeError(f"Unsloth MLX: rank 0 failed while {context}.")
    if status != 1:
        raise RuntimeError(
            f"Unsloth MLX: invalid rank-0 metadata status while {context}."
        )
    return value


class MLXTrainOutput(dict):
    """Dict-compatible train() result with HF Trainer-style attributes."""

    @property
    def metrics(self):
        return self

    @property
    def global_step(self):
        return self.get("train_steps", 0)

    @property
    def training_loss(self):
        return self.get("train_loss", 0.0)


@dataclass
class _MLXTrainerControl:
    """Torch-free subset of Hugging Face TrainerControl used by callbacks."""

    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False


@dataclass
class _MLXTrainerState:
    """Torch-free subset of Hugging Face TrainerState used by callbacks."""

    epoch: float | None = None
    global_step: int = 0
    max_steps: int = 0
    logging_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 500
    train_batch_size: int | None = None
    num_train_epochs: int = 0
    num_input_tokens_seen: int = 0
    total_flos: float = 0
    log_history: list = field(default_factory=list)
    best_metric: float | None = None
    best_global_step: int | None = None
    best_model_checkpoint: str | None = None
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    is_hyper_param_search: bool = False
    trial_name: str | None = None
    trial_params: dict | None = None
    stateful_callbacks: dict = field(default_factory=dict)


# Probed once from the installed DefaultFlowCallback; None until first asked.
_DEFAULT_FLOW_FINAL_STEP_EVAL = None


def _default_flow_evaluates_final_step():
    """Whether the installed DefaultFlowCallback forces a final-step evaluation.

    transformers 5.x evaluates again at ``state.global_step >= state.max_steps``
    when the step interval did not already land there; 4.x has no such block.
    Ask the shipped callback rather than pinning a version, so the loop's own
    cadence tracks whichever transformers is installed and a with-flow run and a
    without-flow run always agree.

    Probed by running the real callback: it reads plain attributes, so the
    torch-free _MLXTrainerState/_MLXTrainerControl stand in for HF's own and the
    probe adds no import to the module. No transformers -- or a future one that
    reads an argument this stand-in lacks -- keeps the 4.x answer, which is what
    the loop did before this cadence existed.
    """
    global _DEFAULT_FLOW_FINAL_STEP_EVAL
    if _DEFAULT_FLOW_FINAL_STEP_EVAL is None:
        _DEFAULT_FLOW_FINAL_STEP_EVAL = False
        try:
            from transformers.trainer_callback import DefaultFlowCallback

            class _FlowProbeArgs:
                eval_strategy = "steps"
                eval_delay = 0
                logging_strategy = "no"
                logging_first_step = False
                save_strategy = "no"

            # 6 % 4 != 0, so only the final-step block can raise the request.
            control = _MLXTrainerControl()
            DefaultFlowCallback().on_step_end(
                _FlowProbeArgs(),
                _MLXTrainerState(
                    global_step=6, max_steps=6,
                    logging_steps=4, eval_steps=4, save_steps=4,
                ),
                control,
            )
            _DEFAULT_FLOW_FINAL_STEP_EVAL = bool(control.should_evaluate)
        except Exception:
            pass
    return _DEFAULT_FLOW_FINAL_STEP_EVAL


def _resolve_interval_steps(value, total_steps):
    """Resolve an HF-style step interval to an absolute number of steps.

    HF accepts logging_steps / eval_steps / save_steps as a step count or a
    ratio in (0, 1) of the total steps, expanded in TrainerState.compute_steps.
    int(ratio) would turn 0.1 into 0, silently disabling the interval and
    making HF's DefaultFlowCallback divide by zero.
    """
    try:
        value = float(value or 0)
    except (TypeError, ValueError):
        return 0
    if value <= 0:
        return 0
    if value < 1:
        # max(1, ...) only guards total_steps == 0; ceil is already >= 1 otherwise.
        return max(1, math.ceil(float(total_steps) * value))
    return int(value)


class _MLXCallbackHandler:
    """Small HF-compatible callback dispatcher that keeps MLX imports Torch-free."""

    def __init__(self, callbacks, model, processing_class, optimizer, lr_scheduler):
        self.callbacks = []
        for callback in callbacks:
            self.add_callback(callback)
        self.model = model
        self.processing_class = processing_class
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = None
        self.eval_dataloader = None

    @property
    def callback_list(self):
        """Return callback class names for diagnostics."""
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def add_callback(self, callback):
        """Add a callback class or instance."""
        self.callbacks.append(callback() if isinstance(callback, type) else callback)

    def pop_callback(self, callback):
        """Remove and return a callback class or instance."""
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb
        return None

    def remove_callback(self, callback):
        """Remove a callback class or instance."""
        self.pop_callback(callback)

    def call_event(self, event, args, state, control, **kwargs):
        """Dispatch one callback event and return the latest control object."""
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method is None:
                continue
            result = method(
                args,
                state,
                control,
                model=self.model,
                processing_class=self.processing_class,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )
            if result is not None:
                control = result
        return control


class _MLXTokenizedDatasetView:
    """Lazy public dataset view that adds input_ids for SFTTrainer parity."""

    def __init__(
        self,
        dataset,
        tokenizer,
        max_seq_length,
        formatting_func=None,
        dataset_text_field="text",
        chat_template=None,
        model_name=None,
        model_type=None,
        append_eos=True,
    ):
        self._dataset = dataset
        self._tokenizer = normalize_mlx_chat_template(
            tokenizer,
            chat_template=chat_template,
            model_name=model_name,
            model_type=model_type,
            is_vlm=False,
            strict=False,
        )
        self._max_seq_length = max_seq_length
        self._formatting_func = formatting_func
        self._dataset_text_field = dataset_text_field
        self._append_eos = append_eos

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        for item in self._dataset:
            yield self._with_input_ids(item)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in ("input_ids", "attention_mask"):
                return [self._with_input_ids(self._dataset[idx])[key] for idx in range(len(self))]
            try:
                return self._dataset[key]
            except (KeyError, TypeError):
                values = []
                for idx in range(len(self)):
                    item = self._dataset[idx]
                    if not isinstance(item, dict) or key not in item:
                        raise
                    values.append(item[key])
                return values
        if not isinstance(key, int):
            return self._dataset[key]
        return self._with_input_ids(self._dataset[key])

    def _with_input_ids(self, item):
        if not isinstance(item, dict) or "input_ids" in item:
            return item

        source = self._formatting_func(item) if self._formatting_func is not None else item
        texts = collect_mlx_texts(
            self._tokenizer,
            source,
            dataset_text_field=self._dataset_text_field,
            is_vlm=False,
        )
        if not texts:
            return item

        encoded = encode_mlx_text(self._tokenizer, texts[0])
        eos_id = getattr(self._tokenizer, "eos_token_id", None)
        if self._append_eos and eos_id is not None and (not encoded or encoded[-1] != eos_id):
            encoded = list(encoded) + [eos_id]
        if self._max_seq_length and len(encoded) > self._max_seq_length:
            encoded = encoded[:self._max_seq_length]

        item = dict(item)
        item["input_ids"] = encoded
        item["attention_mask"] = [1] * len(encoded)
        return item


def _mlx_stream_declares_infinite(dataset):
    """Recognize explicit/common infinite iterable declarations without probing."""
    dataset = getattr(dataset, "_mlx_source_dataset", dataset)
    if bool(getattr(dataset, "_unsloth_mlx_infinite", False)):
        return True

    def _is_infinite(ex_iterable, seen):
        if ex_iterable is None or id(ex_iterable) in seen:
            return False
        seen = seen | {id(ex_iterable)}
        kind = type(ex_iterable).__name__
        if kind == "TakeExamplesIterable":
            return False
        if (
            kind == "RepeatExamplesIterable"
            and getattr(ex_iterable, "num_times", 0) is None
        ):
            return True
        # Private datasets internals: read defensively so a future rename only
        # loses detection instead of failing eval for every matching stream.
        if kind in (
            "VerticallyConcatenatedMultiSourcesExamplesIterable",
            "HorizontallyConcatenatedMultiSourcesExamplesIterable",
        ):
            # Vertical concat runs children in sequence; horizontal concat
            # drops each as it exhausts and ends with the LONGEST child. Both
            # are infinite when ANY child is.
            return any(
                _is_infinite(child, seen)
                for child in getattr(ex_iterable, "ex_iterables", ())
            )
        if kind in (
            "CyclingMultiSourcesExamplesIterable",
            "RandomlyCyclingMultiSourcesExamplesIterable",
        ):
            children = getattr(ex_iterable, "ex_iterables", ())
            probabilities = getattr(ex_iterable, "probabilities", None)
            stopping_strategy = getattr(ex_iterable, "stopping_strategy", "")
            if probabilities is not None:
                if (
                    stopping_strategy.startswith("all_exhausted")
                    and any(probability <= 0 for probability in probabilities)
                ):
                    return True
                children = [
                    child for child, probability in zip(children, probabilities)
                    if probability > 0
                ]
            infinite = [_is_infinite(child, seen) for child in children]
            return (
                any(infinite)
                if stopping_strategy.startswith("all_exhausted")
                else bool(infinite) and all(infinite)
            )
        return _is_infinite(getattr(ex_iterable, "ex_iterable", None), seen)

    ex_iterable = getattr(dataset, "_ex_iterable", None)
    seen = set()
    return _is_infinite(ex_iterable, seen)


class _MLXLazyEvalBatchView:
    """Restartable eval batch surface that constructs one lazy pass per use."""

    def __init__(self, dataset, factory, max_batches=None, comm_group=None):
        self._dataset = dataset
        self._factory = factory
        self._max_batches = max_batches
        self._comm_group = comm_group

    def __iter__(self):
        declared_infinite = False
        if self._max_batches is None:
            declared_infinite = bool(_mlx_rank0_resolve_int(
                self._comm_group,
                lambda: _mlx_stream_declares_infinite(self._dataset),
                "checking whether the streaming eval source is infinite",
            ))
        if declared_infinite:
            raise ValueError(
                "Unsloth MLX: an infinite streaming eval_dataset must set "
                "max_eval_batches to a positive value (or apply dataset.take) "
                "so evaluation has an explicit boundary."
            )
        iterator = iter(self._factory())
        try:
            if self._max_batches is None:
                yield from iterator
                return
            for _ in range(self._max_batches):
                try:
                    yield next(iterator)
                except StopIteration:
                    return
        finally:
            # Truncation and early consumer exits must release the owned
            # source cursors deterministically.
            close = getattr(iterator, "close", None)
            if callable(close):
                close()


def _mlx_declared_iterable_length(dataset):
    """Return a declared source length without probing a truly unsized source."""
    source = getattr(dataset, "_mlx_source_dataset", dataset)
    if not any("__len__" in cls.__dict__ for cls in type(source).__mro__):
        return None
    try:
        length = len(source)
    except (TypeError, AttributeError) as exc:
        raise ValueError(
            "Unsloth MLX: num_train_epochs requires a streaming text source "
            "whose declared __len__ returns the exact source row count. Use "
            "max_steps for a truly unsized iterable."
        ) from exc
    if length < 0:
        raise ValueError("Unsloth MLX: iterable dataset length cannot be negative.")
    return int(length)


from .utils import (
    make_cce_loss_fn,
    make_baseline_loss_fn,
    make_vlm_cce_loss_fn,
    make_vlm_baseline_loss_fn,
    FiniteTextBatchPlan,
    _FiniteTextRow,
    _create_text_batch_plan,
    _create_ordered_text_plan,
    _normalize_label_smoothing,
    create_batches,
    iterate_training_batches,
    _validate_streaming_length_window,
    _validate_streaming_prefetch,
    _is_mlx_lazy_text_source,
    _vlm_has_sized_index_space,
    _MLXIterableTokenizedDatasetView,
    create_vlm_batches,
    _create_vlm_batch_plan,
    _finite_text_pad_width,
    _vlm_family_is_plannable,
    FiniteVLMBatchPlan,
    _preserved_preprocessing_rng,
    iterate_vlm_training_batches,
    normalize_mlx_chat_template,
    normalize_vlm_processor_chat_template,
    encode_mlx_text,
    _get_vlm_ignore_token_ids,
    collect_mlx_texts,
    save_lora_adapters,
    save_trainable_adapters,
    save_optimizer_state,
    load_optimizer_state,
    save_trainer_state,
    load_trainer_state,
    collect_mlx_lora_adapter_tensors,
    iter_mlx_lora_modules,
    apply_gradient_checkpointing,
    remove_gradient_checkpointing,
    _is_vlm_model,
    _mlx_norm_path_part_is_norm,
    iter_mlx_norm_output_cast_classes,
    restore_mlx_norm_output_cast_state,
    set_mlx_norm_output_cast_to_input_dtype,
    snapshot_mlx_norm_output_cast_state,
    _get_text_model,
    _distributed_rank_size,
    _distributed_global_batch_size,
    _rank_slice_distributed_batch,
)
from .compile import (
    build_compile_policy,
    explain_compile_support,
    get_compile_qualification,
    model_has_gated_delta_layers,
    normalize_mlx_patch_mode,
    resolve_training_compile,
    trace_compile_application,
)
from .shape_guard import (
    AUTOMATIC_TEXT_COMPILE_CEILING,
    DDP_LOCAL_GRAD_SCOPE,
    FULL_STEP_SCOPE,
    TextShapeEvent,
    TextShapeGuardReport,
    build_text_shape_frontier,
    materialize_text_shape_frontier,
    phase_for_microstep,
    plan_text_shape_buckets,
    resolve_compile_max_variants,
    select_text_shape_padding_budget,
)

# Finite CPU-backed batch plans sharing one protocol (visit mapping,
# __getitem__/materialize, __len__).
_FINITE_BATCH_PLAN_TYPES = (FiniteTextBatchPlan, FiniteVLMBatchPlan)
# Plans a compile-failure fallback may refetch unpadded. The text plan rebuilds
# from stored token ids and touches no RNG. The VLM plan reruns the caller's
# processor, so a refetch would draw twice and offset every later batch; it
# reuses the materialized batch instead, whose planned padding is masked.
_EAGER_REFETCHABLE_PLAN_TYPES = (FiniteTextBatchPlan,)


def _is_hf_tokenizer(tokenizer):
    """Check whether a wrapper has already resolved to an HF tokenizer."""
    try:
        from transformers import PreTrainedTokenizerBase
    except Exception:
        return False
    return isinstance(tokenizer, PreTrainedTokenizerBase)


def _resolve_response_mask_tokenizer(tokenizer):
    """Return a callable HF tokenizer for the CUDA response-mask helper."""
    for _ in range(3):
        if _is_hf_tokenizer(tokenizer):
            return tokenizer

        processor_tokenizer = getattr(tokenizer, "tokenizer", None)
        if processor_tokenizer is not None and processor_tokenizer is not tokenizer:
            tokenizer = processor_tokenizer
            continue

        # mlx-lm TokenizerWrapper stores the HF tokenizer under _tokenizer.
        # HF fast tokenizers also expose _tokenizer, but that is the low-level
        # Rust tokenizer and is not callable like PreTrainedTokenizerBase.
        wrapped = getattr(tokenizer, "_tokenizer", None)
        if (
            wrapped is not None
            and wrapped is not tokenizer
            and (
                not hasattr(tokenizer, "convert_tokens_to_ids")
                or callable(wrapped)
            )
        ):
            tokenizer = wrapped
            continue

        break

    if not callable(tokenizer):
        raise TypeError(
            "Unsloth MLX: train_on_responses_only requires a callable "
            "Hugging Face tokenizer or a processor/tokenizer wrapper that "
            "contains one."
        )
    return tokenizer


def _looks_like_processor(obj):
    return obj is not None and (
        hasattr(obj, "image_processor")
        or (hasattr(obj, "tokenizer") and hasattr(obj, "apply_chat_template"))
    )


def _processor_ready_for_detect(obj):
    """Processor can drive detection: renders a template and has a callable inner tokenizer."""
    if not _looks_like_processor(obj):
        return False
    inner = getattr(obj, "tokenizer", None)
    if not _is_hf_tokenizer(inner):
        return False
    return (
        getattr(obj, "chat_template", None) is not None
        or getattr(inner, "chat_template", None) is not None
    )


def _model_type_of(trainer):
    config = getattr(getattr(trainer, "model", None), "_config", None)
    return config.get("model_type") if isinstance(config, dict) else None


def _clear_cached_marker_attrs(obj):
    """Drop Unsloth's cached instruction/response markers (on obj and its inner tokenizer)
    so a chat_template override forces re-detection instead of masking with markers from the
    old template."""
    for target in (obj, getattr(obj, "tokenizer", None)):
        if target is None:
            continue
        for attr in ("_unsloth_input_part", "_unsloth_output_part"):
            if hasattr(target, attr):
                try: delattr(target, attr)
                except Exception: pass
    return obj


def _resolve_autodetect_template_source(trainer, source, resolved_tokenizer, return_function=False):
    """Object to auto-detect (instruction_part, response_part) from.

    VLM templates live on the processor, so detection must see it (the HF helper unwraps to the
    inner tokenizer for matching). Detection must use the processor that will actually render the
    masked batches: when return_function=False the trainer renders them via _resolve_vlm_processor
    (trainer.processor / trainer.tokenizer / model._processor), so a tokenizer= override is not
    used by batching and must not drive detection; when return_function=True the caller applies the
    returned mask, so the explicit override is preferred. A configured chat_template override is
    applied (and any markers cached from a prior template dropped first) so detection matches the
    rendered batches. Falls back to resolved_tokenizer when no processor/override applies.
    """
    args = getattr(trainer, "args", None)
    model = getattr(trainer, "model", None)
    model_name = getattr(model, "_hf_repo", None)
    model_type = _model_type_of(trainer)

    if bool(getattr(trainer, "_is_vlm", False)):
        override = source if _looks_like_processor(source) else None
        trainer_tok = getattr(trainer, "tokenizer", None)
        # Mirror _resolve_vlm_processor's resolution (what batching renders through).
        batching = (
            getattr(trainer, "processor", None)
            or (trainer_tok if _looks_like_processor(trainer_tok) else None)
            or getattr(model, "_processor", None)
        )
        processor = (override or batching) if return_function else (batching or override)
        if processor is not None:
            try:
                if getattr(args, "vlm_chat_template", None) is not None:
                    _clear_cached_marker_attrs(processor)
                processor = normalize_vlm_processor_chat_template(
                    processor,
                    chat_template=getattr(args, "vlm_chat_template", None),
                    model_name=model_name,
                    model_type=model_type,
                    strict=False,
                )
            except Exception:
                pass
            if _processor_ready_for_detect(processor):
                return processor
        return resolved_tokenizer

    # Text: apply the chat_template override before detecting so markers match batches. Clear stale
    # markers BEFORE normalize: a raw Jinja override supplies none (so the HF helper re-detects),
    # while an Unsloth template name/tuple sets fresh correct markers that must be preserved.
    if args is not None and getattr(args, "chat_template", None) is not None:
        try:
            _clear_cached_marker_attrs(resolved_tokenizer)
            return normalize_mlx_chat_template(
                resolved_tokenizer,
                chat_template=args.chat_template,
                model_name=model_name,
                model_type=model_type,
                is_vlm=False,
                strict=False,
            )
        except Exception:
            pass
    if _processor_ready_for_detect(source):
        return source
    return resolved_tokenizer


def _text_completion_only_loss_arg(args):
    """Resolve SFT-compatible completion-only loss defaults."""
    value = getattr(args, "completion_only_loss", None)
    if value is not None:
        return value
    if bool(getattr(args, "train_on_completions", False)):
        return True
    return None


def _text_assistant_only_loss_arg(args):
    """Resolve SFT-compatible assistant-only loss setting."""
    return bool(getattr(args, "assistant_only_loss", False))


def _normalize_mlx_optimizer_name(name):
    if hasattr(name, "value"):
        name = name.value
    opt_name = str(name or "adamw").strip().lower()
    opt_name = opt_name.rsplit(".", 1)[-1].replace("-", "_")
    if opt_name in (
        "adamw_8bit",
        "paged_adamw_8bit",
        "adamw_bnb_8bit",
        "paged_adamw_32bit",
        "adamw_torch",
        "adamw_torch_fused",
        "paged_adamw",
        "adamw_32bit",
        "adamw_hf",
        "adamw_anyprecision",
        "adamw_apex_fused",
    ):
        opt_name = "adamw"
    if opt_name not in SUPPORTED_MLX_OPTIMIZERS:
        supported = ", ".join(SUPPORTED_MLX_OPTIMIZERS)
        raise ValueError(
            f"Unsloth: Unsupported MLX optimizer {name!r}. "
            f"Supported optimizers: {supported}."
        )
    return opt_name


_part_is_norm = _mlx_norm_path_part_is_norm
_iter_norm_output_cast_classes = iter_mlx_norm_output_cast_classes
_set_norm_output_cast_to_input_dtype = set_mlx_norm_output_cast_to_input_dtype


def _normalize_mlx_scheduler_type(name):
    if hasattr(name, "value"):
        name = name.value
    sched_type = str(name or "linear").strip().lower()
    sched_type = sched_type.rsplit(".", 1)[-1].replace("-", "_")
    if sched_type not in SUPPORTED_MLX_LR_SCHEDULERS:
        supported = ", ".join(SUPPORTED_MLX_LR_SCHEDULERS)
        raise ValueError(
            f"Unsloth: Unsupported MLX lr_scheduler_type {name!r}. "
            f"Supported schedulers: {supported}."
        )
    return sched_type


def _resolve_mlx_grad_clipping(args):
    """Resolve mutually exclusive MLX clipping knobs.

    Returns ``(max_grad_norm, max_grad_value, max_grad_leaf_norm, mode)``.
    ``max_grad_value`` keeps elementwise clamp semantics. ``max_grad_leaf_norm``
    is the cheap proportional alternative: cap each gradient leaf's L2 norm
    without a cross-tree global reduction.
    """
    max_grad_norm = float(getattr(args, "max_grad_norm", 0.0) or 0.0)
    raw_value = getattr(args, "max_grad_value", None)
    raw_leaf = getattr(args, "max_grad_leaf_norm", None)
    user_set_value = raw_value is not None
    user_set_leaf = raw_leaf is not None

    max_grad_value = float(raw_value or 0.0) if user_set_value else 0.0
    max_grad_leaf_norm = float(raw_leaf or 0.0) if user_set_leaf else 0.0

    if max_grad_value > 0:
        # Preserve the public meaning of max_grad_value as elementwise clamp.
        return 0.0, max_grad_value, 0.0, "value"

    if max_grad_leaf_norm > 0:
        return 0.0, 0.0, max_grad_leaf_norm, "leaf_norm"

    if max_grad_norm > 0:
        return max_grad_norm, 0.0, 0.0, "global_norm"

    if user_set_value or user_set_leaf:
        # Explicit 0.0 disables cheap clipping.
        return 0.0, 0.0, 0.0, "none"

    # MLX default: cheap proportional clipping without global norm memory cost.
    return 0.0, 0.0, 1.0, "leaf_norm"


def _clip_grad_by_value(grad, max_grad_value):
    """Elementwise clamp; preserves the historical max_grad_value contract."""
    return tree_map(lambda g: mx.clip(g, -max_grad_value, max_grad_value), grad)


def _clip_grad_by_leaf_norm(grad, max_grad_leaf_norm):
    """Scale each gradient leaf to a max L2 norm, preserving leaf direction."""
    def _clip_leaf_norm(g):
        g_f = g.astype(mx.float32)
        norm = mx.sqrt(mx.sum(g_f * g_f))
        scale = mx.minimum(max_grad_leaf_norm / (norm + 1e-6), 1.0)
        return g * scale.astype(g.dtype)

    return tree_map(_clip_leaf_norm, grad)


def _global_grad_norm_fp32(grad):
    """Fp32 L2 norm of a gradient tree (one cross-tree reduction)."""
    norm_squared = tree_reduce(
        lambda acc, g: acc + mx.sum(mx.square(g.astype(mx.float32))),
        grad,
        mx.array(0.0, dtype=mx.float32),
    )
    return mx.sqrt(norm_squared)


def _clip_grad_norm_fp32(grad, max_norm):
    """Global norm clipping with a float32 norm reduction.

    ``mlx.optimizers.clip_grad_norm`` reduces each leaf in its storage dtype.
    For bf16/fp16 VLMs, that can move the global scale away from PyTorch/HF,
    which computes the clipping norm in fp32. Keep clipped leaves in their
    original dtype, but compute the single global scale in fp32.
    """
    total_norm = _global_grad_norm_fp32(grad)
    scale = mx.minimum(
        mx.array(max_norm, dtype=mx.float32) / (
            total_norm + mx.array(1e-6, dtype=mx.float32)
        ),
        mx.array(1.0, dtype=mx.float32),
    )
    return tree_map(lambda g: g * scale.astype(g.dtype), grad), total_norm


def _validate_label_smoothing(value, is_vlm):
    """Configuration gate for ``label_smoothing_factor``: delegates the
    domain check to the shared loss-layer normalizer (config errors raise,
    unlike model-derived properties, which fall back) and adds the VLM rule."""
    eps = _normalize_label_smoothing(value)
    if is_vlm and eps > 0.0:
        raise ValueError(
            "label_smoothing_factor > 0 is not supported for VLM training on MLX."
        )
    return eps


def _require_complete_resume_checkpoint(resume_from):
    """Reject an incomplete resume directory and name the warm-start route.

    A saved adapter directory has only adapters.safetensors. Called from every
    path that reads resume state, so the streaming-prefetch early read raises
    the same guidance instead of a raw FileNotFoundError from trainer_state.json.
    """
    if not resume_from:
        return
    _missing_resume = [
        _f for _f in ("adapters.safetensors",
                      "optimizer_state.safetensors", "trainer_state.json")
        if not os.path.isfile(os.path.join(resume_from, _f))
    ]
    if _missing_resume:
        raise RuntimeError(
            f"Unsloth: resume_from_checkpoint={resume_from!r} is "
            f"missing resume state file(s) {_missing_resume}. Refusing "
            f"to silently restart from step 0. If this is a saved "
            f"adapter directory rather than a training checkpoint and "
            f"you meant to start a new run from it with a fresh "
            f"optimizer, load it with FastMLXModel.from_pretrained(<dir>) "
            f"and train without resume_from_checkpoint."
        )


def _prune_stale_checkpoints(output_dir, save_total_limit, keep_step=None):
    """Keep the newest ``save_total_limit`` checkpoint-* dirs (HF Trainer parity).

    ``keep_step`` is never rotated out, mirroring HF's _rotate_checkpoints, which
    protects best_model_checkpoint: without it a best result at an early step is
    deleted by a later worse save and the state field is left pointing at nothing.
    ``-1`` / ``0`` / ``None`` preserve the existing "no limit" contract.
    """
    if not save_total_limit or save_total_limit < 1:
        return
    import shutil
    from pathlib import Path

    checkpoints = []
    for child in Path(output_dir).glob("checkpoint-*"):
        # Only prune real step-checkpoint dirs the trainer created; never
        # follow symlinks or touch user paths that share the prefix.
        if child.is_symlink() or not child.is_dir():
            continue
        try:
            step = int(child.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        checkpoints.append((step, child))
    if len(checkpoints) <= save_total_limit:
        return
    checkpoints.sort()
    # Move the protected entry to the end before slicing, as HF does, so the
    # limit still binds: excluding it from the stale slice instead retained
    # save_total_limit + 1 directories from then on.
    protected = None if keep_step is None else int(keep_step)
    limit = save_total_limit
    if limit == 1 and protected is not None:
        # HF raises the limit to 2 here so the best and the latest both survive.
        limit = 2
    ordered = [child for step, child in checkpoints if step != protected]
    ordered += [child for step, child in checkpoints if step == protected]
    for stale in ordered[:max(0, len(ordered) - limit)]:
        try:
            shutil.rmtree(stale)
        except Exception as exc:
            print(f"  Unsloth: failed to prune old checkpoint {stale}: {exc}")
            continue
        print(f"  Unsloth: pruned old checkpoint {stale} "
              f"(save_total_limit={save_total_limit})")


def _mlx_batch_input_token_count(batch_data, mode="all", pad_token_id=None):
    """Input-token positions in a training microbatch (HF num_input_tokens_seen).

    HF's TrainerState.num_input_tokens_seen counts the main input tensor's numel
    (every forwarded position: prompt + response + padding), NOT just the
    supervised/label tokens the loss mask counts. Mirror that here so a callback
    reading state.num_input_tokens_seen for token-budget stopping or throughput
    reporting is not undercounted by the masked/prompt fraction (which for
    completion-only / assistant-only loss is most of the sequence).

    ``mode`` is HF's normalized include_num_input_tokens_seen value. "all" (and
    the un-normalized True a non-TrainingArguments config keeps) counts every
    forwarded position as above. "non_padding" follows HF's ladder instead --
    identical in transformers 4.57.x (the inline block in _inner_training_loop)
    and 5.x (Trainer._track_num_input_tokens): the attention mask when the batch
    carries one, else a pad-token comparison when the processing class exposes a
    pad_token_id, else every position. The text/preference/GRPO tuple batch has
    no attention_mask, but its ``lengths`` column 1 is the exclusive end of the
    real tokens (the same column the loss masks compare against, and what
    mlx-lm's iterate_batches emits), and rows are written at ``[0, length)``, so
    summing it is exactly that batch's attention-mask sum. DDP pad rows carry
    length 0 and therefore drop out, as they should.

    Uses ``.shape`` (a tuple under both real mlx and the torch test shim) rather
    than a backend-specific ``.size`` / ``.numel``. Handles the text/preference/
    GRPO tuple batch (input ids first) and the VLM dict batch (``input_ids`` key);
    returns 0 when no input-id tensor is present so the counter simply does not
    advance rather than raising.
    """
    import math
    attention_mask = None
    lengths = None
    if isinstance(batch_data, dict):
        arr = batch_data.get("input_ids")
        attention_mask = batch_data.get("attention_mask")
    elif isinstance(batch_data, (tuple, list)) and batch_data:
        arr = batch_data[0]
        lengths = batch_data[1] if len(batch_data) > 1 else None
    else:
        arr = None
    if arr is None or not hasattr(arr, "shape"):
        return 0
    if mode != "non_padding":
        return int(math.prod(arr.shape))
    if attention_mask is not None and hasattr(attention_mask, "shape"):
        return int(attention_mask.sum().item())
    if (
        lengths is not None
        and hasattr(lengths, "shape")
        and len(lengths.shape) == 2
        and lengths.shape[1] == 2
    ):
        return int(lengths[:, 1].sum().item())
    if pad_token_id is not None:
        return int((arr != pad_token_id).sum().item())
    # HF's last rung: no mask and no pad id, so every position is counted.
    return int(math.prod(arr.shape))


# Fields added after the original public MLXTrainingConfig surface. Keep them a
# suffix of the declaration order (append new ones at the end and list them
# here) so positional copies from older configs keep mapping correctly.
_MLX_CONFIG_OPTIONAL_COPY_FIELDS = (
    "max_eval_batches",
    "streaming_text_length_window_batches",
    "streaming_prefetch_batches",
    "logging_dir",
    "run_name",
)


@dataclass
class MLXTrainingConfig:
    """Training configuration mirroring SFTConfig / TrainingArguments field names."""

    # Core training
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 60
    num_train_epochs: int = -1  # -1 means use max_steps instead
    warmup_steps: int = 5
    warmup_ratio: float = 0.0
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "linear"  # "cosine", "linear", "constant"

    # Optimization
    optim: str = "adamw"  # "adafactor", "adamw", "adam", "sgd", "muon", "lion"
    weight_decay: float = 0.001
    adam_beta1: float | None = None
    adam_beta2: float | None = None
    # Global L2 norm clip (transformers/CUDA max_grad_norm). Disabled by
    # default on MLX: the per-leaf cap below is the default instead, since
    # global norm's cross-tree reduction costs more peak memory (measured
    # ~1 GB more at 3B, scaling with size). Set this for CUDA-exact clipping;
    # note per-leaf and global agree when no spike binds but diverge on
    # gradient spikes (per-leaf cannot see an aggregate norm spread across
    # many tensors).
    max_grad_norm: float = 0.0
    # Elementwise clip to `[-v, +v]`. None means "not requested";
    # positive values override other clipping modes to preserve API meaning.
    max_grad_value: float | None = None
    # Proportional per-leaf L2 norm cap and the MLX default (1.0 when no clip
    # knob is set). Preserves each tensor's direction and avoids max_grad_norm's
    # cross-tree memory overhead, but is not a drop-in for global max_grad_norm
    # (see above). None uses the 1.0 default unless another clip knob is explicit.
    max_grad_leaf_norm: float | None = None
    seed: int = 3407
    lora_plus_ratio: float = 0.0  # 0 = disabled, 16.0 = recommended
    embedding_learning_rate: float = 0.0  # 0 = disabled, 5e-5 = recommended

    # Logging & output
    logging_steps: int = 1
    output_dir: str = "./outputs"
    report_to: str = "none"
    save_steps: int = 0  # 0 = only save at end
    save_total_limit: int = -1  # -1 = no limit

    # Eval
    eval_steps: int = 0  # 0 = disabled
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 0  # 0 = disabled
    neftune_noise_alpha: float = 0.0  # 0 = disabled (text models only)

    # SFT-specific (from SFTConfig, for API compat)
    dataset_text_field: str = "text"
    max_seq_length: int = 2048
    packing: bool = False
    dataset_num_proc: int = 2
    chat_template: object = None  # Unsloth template name/tuple or raw Jinja string

    # MLX-specific
    use_cce: bool = True
    compile: bool = True
    compile_mode: str = "best_effort"  # "best_effort", "strict", "eager"
    compile_max_variants: int | None = None
    compile_arch_overrides: dict[str, str] | None = None
    compile_backend_overrides: dict[str, str] | None = None
    patch_mode: str = "patched"  # "patched" runs the MLX compile monkey patches, "unpatched" forces eager baseline mode.
    compile_auto_tune: bool = True
    compile_trace: bool = True
    gradient_checkpointing: bool = True
    streaming: bool = False  # Lazily consume unsized text/VLM sources
    dataset_order: str = "default"  # "default", "sequential", or "torch_randperm"
    preserve_dataset_order: bool = False  # Match Unsloth CUDA SequentialSampler order
    memory_limit_gb: float | None = None  # None = auto Metal guard (~85% of recommended working set); <= 0 disables
    cache_limit_gb: float | None = None  # Optional MLX Metal cache cap in GB; <= 0 disables override
    wired_limit_gb: float | None = None  # None = min(recommended working set, memory limit); <= 0 disables
    disable_memory_limits: bool = False
    cast_norm_output_to_input_dtype: bool = True  # fp32 norm storage/math, bf16/fp16 downstream activations
    append_eos: bool = True  # True = mlx-lm parity; Unsloth sets False (template owns EOS)

    # VLM / completion masking
    train_on_completions: bool = False  # Mask prompt tokens in loss
    completion_only_loss: bool | None = None  # None = SFT/VLM default; False trains on prompt+completion
    assistant_only_loss: bool = False  # Mask non-assistant tokens with chat-template assistant masks
    assistant_token_id: int = 0  # Token ID marking start of assistant response
    vlm_chat_template: object = None  # Unsloth template name/tuple or raw Jinja string
    per_device_eval_batch_size: int | None = None
    image_size: object = None  # VLM image resize override from UnslothVisionDataCollator(resize=...)
    # Appended by main after image_size; kept before the streaming fields so a
    # positional copy from a main config still maps correctly.
    label_smoothing_factor: float = 0.0  # HF LabelSmoother epsilon (text models only)

    # Opt-in true global grad-norm reporting when global-norm clipping is off; a
    # no-op when it is on, since clipping computes the pre-clip norm anyway.
    # Appended after the fields that predate it so each keeps its positional
    # index. Enabling it costs one cross-tree fp32 reduction per update, the same
    # class of peak-memory cost global clipping pays. When False (default)
    # grad_norm is absent everywhere and no reduction exists in the graph.
    # Reporting never changes numerics: the value is the fp32 norm of the
    # token-normalized gradient, after the accumulation divide and DDP reduction,
    # before clipping, decay, the update and the scoped-LR rescale.
    report_grad_norm: bool = False

    # Lazy-streaming fields, appended after every pre-existing field so
    # positional copies from older configs keep mapping correctly. Listed in
    # _MLX_CONFIG_OPTIONAL_COPY_FIELDS.
    max_eval_batches: int | None = None  # Bound an explicitly infinite lazy text eval stream
    # Lazy default-order text streams: pool this many global micro-batches,
    # length-sort, emit seeded-permuted batches. 1 = exact source order. Memory
    # scales with world size; the DDP owner retains one extra padding batch.
    streaming_text_length_window_batches: int = 8
    # Lazy text streams: prepare this many batches ahead on a producer thread
    # (0 = synchronous default; single-process, host-valued rows only). Queued
    # batches add to the window bound.
    streaming_prefetch_batches: int = 0

    # Callback-visible run metadata (HF TrainingArguments parity). Declared LAST
    # for the same reason as the fields above: the initializer binds positional
    # args by field order, so inserting them mid-list would shift the positional
    # slot of every field after it. Also listed in
    # _MLX_CONFIG_OPTIONAL_COPY_FIELDS so they stay an exact suffix of it.
    logging_dir: str | None = None
    run_name: str | None = None

    def __init__(self, *args, **kwargs):
        config_fields = [field for field in fields(type(self)) if field.init]
        if len(args) > len(config_fields):
            raise TypeError(
                f"MLXTrainingConfig expected at most {len(config_fields)} "
                f"positional arguments, got {len(args)}"
            )
        for field, value in zip(config_fields, args):
            if field.name in kwargs:
                raise TypeError(
                    f"MLXTrainingConfig got multiple values for argument "
                    f"{field.name!r}"
                )
            kwargs[field.name] = value

        provided = set(kwargs)
        unknown = provided - {field.name for field in config_fields}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"MLXTrainingConfig got unexpected arguments: {names}")

        for field in config_fields:
            if field.name in kwargs:
                value = kwargs[field.name]
            elif field.default is not MISSING:
                value = field.default
            elif field.default_factory is not MISSING:
                value = field.default_factory()
            else:
                raise TypeError(
                    f"MLXTrainingConfig missing required argument: {field.name!r}"
                )
            setattr(self, field.name, value)

        warmup_steps_default = type(self).warmup_steps
        warmup_ratio_default = type(self).warmup_ratio
        # A config copied or round-tripped from an older Unsloth may omit later
        # fields; still treat it as a wholesale copy for warmup semantics, or a
        # copied default warmup_steps would override a non-default warmup_ratio.
        # So tolerate every field appended since: the positional optional-copy
        # fields plus the later scalar additions.
        _appended_fields = set(_MLX_CONFIG_OPTIONAL_COPY_FIELDS) | {
            "compile_max_variants",
            "label_smoothing_factor",
            "report_grad_norm",
        }
        _field_names = {field.name for field in config_fields}
        copied_all_fields = (_field_names - _appended_fields) <= set(provided)
        copied_default_warmup_with_ratio = (
            copied_all_fields
            and getattr(self, "warmup_steps", None) == warmup_steps_default
            and getattr(self, "warmup_ratio", None) != warmup_ratio_default
        )
        self._unsloth_mlx_warmup_steps_explicit = (
            "warmup_steps" in provided and not copied_default_warmup_with_ratio
        )
        if self.compile_max_variants is not None:
            resolve_compile_max_variants(self.compile_max_variants)

    def to_dict(self):
        """Return a TrainingArguments-style dict for integration callbacks."""
        output = {}
        for key, value in vars(self).items():
            if is_dataclass(value):
                value = asdict(value)
            elif hasattr(value, "to_dict"):
                value = value.to_dict()
            output[key] = value
        return output

    def to_json_string(self):
        """Serialize this config like TrainingArguments.to_json_string()."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def to_sanitized_dict(self):
        """Serialize this config like TrainingArguments.to_sanitized_dict().

        The other integration callbacks read the config through to_dict() /
        to_json_string(); HF's NeptuneCallback reads it through this method
        instead, unguarded, so omitting it aborts on_train_begin. HF reports
        the resolved batch sizes next to the raw fields and stringifies
        anything a tracker cannot store, so mirror both. torch.Tensor is on
        HF's allow-list but cannot occur here: this module stays Torch-free.
        """
        output = self.to_dict()
        output["train_batch_size"] = self.per_device_train_batch_size
        output["eval_batch_size"] = (
            getattr(self, "per_device_eval_batch_size", None)
            or self.per_device_train_batch_size
        )
        # HF compares the exact type, so bool stays bool instead of widening.
        valid_types = (bool, int, float, str)
        return {
            key: value if type(value) in valid_types else str(value)
            for key, value in output.items()
        }


def _shape_guard_report(
    action,
    reason,
    cap,
    compile_scope="none",
    *,
    lazy_batches=True,
    cap_selection="not_applicable",
):
    return TextShapeGuardReport(
        action=action,
        reason=reason,
        cap=cap,
        compile_scope=compile_scope,
        raw_signatures=0,
        planned_signatures=None,
        raw_widths=0,
        lazy_batches=lazy_batches,
        configured_cap=cap,
        effective_cap=cap,
        cap_selection=cap_selection,
        budget_satisfied=False,
    )


def _mlx_epoch_microbatches(args, batches, *, includes_epochs=False):
    """Micro-batches in one epoch when epoch boundaries drive the schedule.

    None for streaming, for runs with no declared epoch count, and for max_steps
    runs whose batch source cannot report an exact one-pass length: those keep
    the flat accumulation model. Mirrors the epoch branches of
    MLXTrainer._callback_batches_per_epoch so the budget, the forced boundary
    update and the callback epoch events all land on the same micro-batch.

    num_train_epochs is a float in TrainingArguments/SFTConfig, so the declared
    count is read as one: truncating it sent 0 < num_train_epochs < 1 down the
    no-epoch flat path, which budgets a whole pass. Only the prebuilt-epochs
    split needs a whole number, and that branch keeps its integer guard.
    """
    if batches is None:
        return None
    total = len(batches)
    if total <= 0:
        return None
    # The plan's own one-pass length, when it can report one. Only that exact
    # count qualifies: the dataset-size approximation cannot see what batching
    # retained, so forcing updates on it would move optimizer steps onto
    # micro-batches that are not boundaries.
    plan_cycle = getattr(batches, "cycle_length", None)
    plan_cycle = max(1, int(plan_cycle)) if plan_cycle else None
    if int(getattr(args, "max_steps", 0) or 0) > 0:
        # HF's forced epoch-final update is not conditional on max_steps:
        # do_sync_step reads len(dataloader), so max_steps decides when the run
        # ends, never how an epoch's ragged tail is applied. Returning None here
        # left that tail pending across on_epoch_end and folded it into the next
        # epoch.
        return plan_cycle
    epochs = float(getattr(args, "num_train_epochs", 0) or 0)
    if epochs <= 0:
        return None
    if plan_cycle is not None:
        # why: a prebuilt fractional schedule holds a whole number of
        # micro-batches but a fractional number of passes, so int(epochs) does
        # not divide it. 1.5 epochs of 5 is 8 batches, and dividing by 1 reads
        # the lot as one epoch: the boundary at micro-batch 5 disappears, the
        # accumulation window never restarts there and the budget lands on 4
        # updates where HF takes 5.
        return plan_cycle
    whole_epochs = int(epochs)
    if includes_epochs and whole_epochs > 0 and total % whole_epochs == 0:
        return max(1, total // whole_epochs)
    return total


def _mlx_steps_per_epoch(epoch_microbatches, grad_accum):
    """Optimizer steps one epoch costs when its last micro-batch forces a step."""
    return max(1, math.ceil(
        int(epoch_microbatches) / max(1, int(grad_accum))
    ))


def _mlx_microstep_for_step(global_step, epoch_microbatches, grad_accum):
    """Micro-batches consumed once ``global_step`` optimizer steps have run.

    Epochs close on a forced step, so the mapping is per-epoch rather than flat
    (HF: epochs_trained * steps_in_epoch + global_step % updates_per_epoch *
    grad_accum). Equals global_step * grad_accum for a divisible epoch.
    """
    epoch_microbatches = int(epoch_microbatches)
    grad_accum = max(1, int(grad_accum))
    per_epoch = _mlx_steps_per_epoch(epoch_microbatches, grad_accum)
    epochs_done, steps_into_epoch = divmod(int(global_step), per_epoch)
    return epochs_done * epoch_microbatches + min(
        steps_into_epoch * grad_accum, epoch_microbatches,
    )


def _mlx_microstep_phase(
    compile_scope, grad_accum, index, epoch_microbatches=None,
):
    """Compiled-argument phase at one micro-batch, epoch flush included.

    Accumulation windows restart at every epoch boundary and the epoch's last
    micro-batch forces the update, so both the window position and the update
    flag are per-epoch once ``epoch_microbatches`` is known.
    """
    if not epoch_microbatches:
        return phase_for_microstep(compile_scope, grad_accum, index)
    epoch_microbatches = int(epoch_microbatches)
    position = int(index) % epoch_microbatches
    phase = phase_for_microstep(compile_scope, grad_accum, position)
    if (
        compile_scope != FULL_STEP_SCOPE
        or position != epoch_microbatches - 1
    ):
        return phase
    # Epoch-final micro-batch: it traces the updating signature even when it is
    # not the accumulation window's last position.
    if phase == "none_no_update":
        return "single"
    if phase == "tree_no_update":
        return "tree_update"
    return phase
class _VLMCompileDecisionError(RuntimeError):
    """A compile decision that mandates an abort, never maskable by
    best-effort degradation (per-architecture strict overrides set
    should_raise even while the base policy mode is best_effort)."""


def _effective_compile_mode(compile_policy, compile_decision):
    """Return the compile mode in force after arch/backend overrides.

    ``resolve_training_compile`` can resolve strict under a best_effort base
    policy, so strictness checks must follow the resolved mode or an
    override-selected strict run silently degrades to eager. Decisions
    predating the field fall back to the policy mode.
    """
    mode = getattr(compile_decision, "policy_mode", None)
    return mode if mode else compile_policy.mode


def _plan_single_process_vlm_shapes(
    batches,
    batch_iter,
    *,
    args,
    total_steps,
    distributed_world_size,
    compile_policy,
    compile_decision,
    install_plan=True,
):
    """Plan finite VLM shapes for the single-process compiled path.

    Must run only after compile qualification resolved: the descriptor
    survey materializes every scheduled batch once. Padable batches take
    the shared rounded width policy capped at the surveyed maximum final
    width (post-expansion widths legitimately exceed ``max_seq_length``,
    which is never consulted); batches the survey declined join at their
    exact raw widths. An unplannable family forces eager fallback for the
    run, since grouping it could span several compile keys.
    """
    configured_cap = getattr(args, "compile_max_variants", None)
    automatic = configured_cap is None
    cap = resolve_compile_max_variants(configured_cap)
    lazy = isinstance(batches, FiniteVLMBatchPlan)
    if compile_decision is not None and getattr(
        compile_decision, "should_raise", False,
    ):
        # Checked before EVERY applicability class (including streaming) so the
        # mandated abort surfaces inside the coordinated block, not rank-locally.
        raise _VLMCompileDecisionError(
            "Unsloth: strict mx.compile requested for VLM arch "
            f"'{getattr(compile_decision, 'arch', 'unknown')}', but compile "
            f"cannot be enabled "
            f"({getattr(compile_decision, 'reason', 'unqualified')})."
        )
    if batch_iter is not None:
        return None, _shape_guard_report(
            "not_applicable", "streaming", cap, lazy_batches=False,
        ), True, None
    if compile_policy.mode == "eager":
        return None, _shape_guard_report(
            "not_applicable", "compile_disabled", cap, lazy_batches=lazy,
        ), True, None
    if compile_decision is None or not getattr(
        compile_decision, "enabled", False,
    ):
        return None, _shape_guard_report(
            "not_applicable", "vlm_compile_unqualified", cap,
            lazy_batches=lazy,
        ), True, None
    # Strictness follows the RESOLVED mode: an arch/backend override can select
    # strict under a best_effort base policy, and that run must abort, not degrade.
    effective_mode = _effective_compile_mode(compile_policy, compile_decision)
    max_grad_norm = _resolve_mlx_grad_clipping(args)[0]
    if (
        distributed_world_size <= 1
        and max_grad_norm > 0
        and args.gradient_accumulation_steps > 1
    ):
        # Compilation is disabled later here, so skip the survey (it materializes
        # every batch) for a plan that could never be compiled.
        return None, _shape_guard_report(
            "not_applicable", "compile_ineligible_global_norm", cap,
            lazy_batches=lazy,
        ), False, None
    compile_scope = (
        DDP_LOCAL_GRAD_SCOPE
        if distributed_world_size > 1 else FULL_STEP_SCOPE
    )
    if not isinstance(batches, FiniteVLMBatchPlan):
        report = _shape_guard_report(
            "eager", "unsupported_batch_plan", cap, compile_scope,
            lazy_batches=False,
            cap_selection="not_applicable",
        )
        if effective_mode == "strict" and distributed_world_size <= 1:
            raise RuntimeError(
                "Unsloth: strict mx.compile requires a finite VLM batch plan."
            )
        return None, report, False, None

    batches.ensure_descriptors()
    # Admit only the batches the loop actually visits. The gradient-accumulation
    # floor drops the schedule's trailing micro-batches, and an unplannable
    # family confined to that tail would otherwise degrade the whole run to
    # eager (or abort strict mode) over a batch no compiled call can reach.
    # Resume is not resolved yet at this call site, so visits start at 0: that
    # is a superset of what a resumed loop runs, and a superset only ever
    # admits more than needed. The survey itself stays whole-schedule, because
    # planned_event_widths() reduces the maximum padable width and the union of
    # untouched extents over EVERY batch -- narrowing that input would silently
    # move the endpoints of the batches that do train.
    grad_accum = args.gradient_accumulation_steps
    # Same epoch-aware mapping the runtime loop uses, as the text planner does.
    # A ragged epoch turns its tail micro-batch into an update phase, and
    # cataloging the flat phase left that signature unadmitted, so compiled VLM
    # training aborted at the first ragged boundary.
    epoch_microbatches = _mlx_epoch_microbatches(args, batches)
    total_microsteps = (
        _mlx_microstep_for_step(total_steps, epoch_microbatches, grad_accum)
        if epoch_microbatches else total_steps * grad_accum
    )
    executed = sorted({
        batches.batch_index_for_visit(microstep)
        for microstep in range(total_microsteps)
    })
    unplannable = [
        index
        for index in executed
        if not _vlm_family_is_plannable(batches.batch_family(index))
    ]
    if unplannable:
        report = _shape_guard_report(
            "eager", "vlm_unplannable_family", cap, compile_scope,
            cap_selection="not_applicable",
        )
        if effective_mode == "strict":
            raise RuntimeError(
                "Unsloth: strict mx.compile cannot plan VLM batch "
                f"{unplannable[0]}: its compile-key family is not stable "
                "enough to group safely."
            )
        return None, report, False, None

    planned_widths = batches.planned_event_widths()

    event_counts = {}
    for microstep in range(total_microsteps):
        batch_index = batches.batch_index_for_visit(microstep)
        key = (
            batches.batch_family(batch_index),
            planned_widths[batch_index],
            _mlx_microstep_phase(
                compile_scope,
                grad_accum,
                microstep,
                epoch_microbatches,
            ),
            len(batches.schedule[batch_index]),
        )
        event_counts[key] = event_counts.get(key, 0) + 1
    events = tuple(
        TextShapeEvent(
            family=family,
            width=width,
            phase=phase,
            frequency=frequency,
            local_batch_size=batch_size,
        )
        for (family, width, phase, batch_size), frequency in event_counts.items()
    )
    frontier = None
    if automatic:
        frontier = build_text_shape_frontier(
            events, compile_scope=compile_scope,
        )
        # VLM catalogs keep every media family as its own endpoint group, so
        # budget compression drops too few signatures to pay for its padded
        # compute. Stay exact until the signature cap genuinely binds.
        shape_plan = select_text_shape_padding_budget(
            frontier,
            exact_signature_threshold=AUTOMATIC_TEXT_COMPILE_CEILING,
        )
    else:
        shape_plan = plan_text_shape_buckets(
            events,
            cap=cap,
            compile_scope=compile_scope,
        )
    if shape_plan.report.action == "eager":
        if effective_mode == "strict":
            raise RuntimeError(
                "Unsloth: strict mx.compile finite VLM shape planning failed "
                f"({shape_plan.report.reason})."
            )
        return shape_plan, shape_plan.report, False, frontier
    # Only widening writes the tokenizer pad id into a tail, so a pad id is
    # required by the BUILT plan rather than by the processor: uniform-width
    # and declined-only schedules pad nothing and stay compilable without one.
    # Same admitted set as above, or a tail batch nobody widens (it is only
    # ever built unpadded, by advance_preprocessing or a plain fetch) would
    # still demand a pad id the run never uses.
    if batches.pad_token_id is None and any(
        shape_plan.endpoint_for(
            batches.batch_family(index), planned_widths[index],
        ) > batches.batch_width(index)
        for index in executed
    ):
        report = _shape_guard_report(
            "eager", "vlm_pad_token_unavailable", cap, compile_scope,
            cap_selection="not_applicable",
        )
        if effective_mode == "strict":
            raise RuntimeError(
                "Unsloth: strict mx.compile cannot plan VLM widths without "
                "a tokenizer pad id."
            )
        return None, report, False, None
    if install_plan:
        batches.set_shape_plan(shape_plan, planned_widths)
    return shape_plan, shape_plan.report, True, frontier


def _plan_single_process_text_shapes(
    batches,
    batch_iter,
    *,
    args,
    total_steps,
    is_vlm,
    distributed_world_size,
    compile_policy,
    install_plan=True,
    includes_epochs=False,
    vlm_compile_decision=None,
):
    """Plan finite text shapes before optimizer or compiled-callable setup."""

    configured_cap = getattr(args, "compile_max_variants", None)
    automatic = configured_cap is None
    cap = resolve_compile_max_variants(configured_cap)
    if is_vlm:
        return _plan_single_process_vlm_shapes(
            batches,
            batch_iter,
            args=args,
            total_steps=total_steps,
            distributed_world_size=distributed_world_size,
            compile_policy=compile_policy,
            compile_decision=vlm_compile_decision,
            install_plan=install_plan,
        )
    if batch_iter is not None:
        return None, _shape_guard_report(
            "not_applicable", "streaming", cap, lazy_batches=False,
        ), True, None
    compile_scope = (
        DDP_LOCAL_GRAD_SCOPE
        if distributed_world_size > 1 else FULL_STEP_SCOPE
    )
    if compile_policy.mode == "eager":
        return None, _shape_guard_report(
            "not_applicable", "compile_disabled", cap,
        ), True, None
    if not isinstance(batches, FiniteTextBatchPlan):
        report = _shape_guard_report(
            "eager", "unsupported_batch_plan", cap, compile_scope,
            lazy_batches=False,
            cap_selection="not_applicable",
        )
        if compile_policy.mode == "strict" and distributed_world_size <= 1:
            raise RuntimeError(
                "Unsloth: strict mx.compile requires a finite CPU text batch plan."
            )
        return None, report, False, None

    grad_accum = args.gradient_accumulation_steps
    epoch_microbatches = _mlx_epoch_microbatches(
        args, batches, includes_epochs=includes_epochs,
    )
    if epoch_microbatches:
        # Epoch-count runs visit whole epochs and each epoch's last micro-batch
        # forces the update, so the stream is shorter than total_steps * accum.
        # A fractional num_train_epochs stops part-way through the final epoch,
        # and those micro-batches are still fetched, so the catalog counts them
        # via the same step -> micro-batch mapping the runtime fetch uses.
        # Whole epoch counts land on an epoch boundary and are unchanged.
        total_microsteps = _mlx_microstep_for_step(
            total_steps, epoch_microbatches, grad_accum,
        )
    else:
        total_microsteps = total_steps * grad_accum
    event_counts = {}
    for microstep in range(total_microsteps):
        # Same visit mapping as the runtime fetch, so the enumerated catalog
        # equals the actually visited (family, width, phase) sequence even for
        # epoch-permuted plans.
        batch_index = batches.batch_index_for_visit(microstep)
        family = batches.batch_family(batch_index)
        width = batches.batch_width(batch_index)
        phase = _mlx_microstep_phase(
            compile_scope,
            grad_accum,
            microstep,
            epoch_microbatches,
        )
        key = (family, width, phase, len(batches.schedule[batch_index]))
        event_counts[key] = event_counts.get(key, 0) + 1
    events = tuple(
        TextShapeEvent(
            family=family,
            width=width,
            phase=phase,
            frequency=frequency,
            local_batch_size=batch_size,
        )
        for (family, width, phase, batch_size), frequency in event_counts.items()
    )
    frontier = None
    if automatic:
        frontier = build_text_shape_frontier(
            events, compile_scope=compile_scope,
        )
        shape_plan = select_text_shape_padding_budget(frontier)
    else:
        shape_plan = plan_text_shape_buckets(
            events,
            cap=cap,
            compile_scope=compile_scope,
        )
    if shape_plan.report.action == "eager":
        if compile_policy.mode == "strict" and distributed_world_size <= 1:
            raise RuntimeError(
                "Unsloth: strict mx.compile finite text shape planning failed "
                f"({shape_plan.report.reason})."
            )
        return shape_plan, shape_plan.report, False, frontier
    if install_plan:
        batches.set_shape_plan(shape_plan)
    return shape_plan, shape_plan.report, True, frontier


def _resolve_training_steps(args, batches, batch_iter, *, includes_epochs=False):
    if batches is not None and not batches:
        raise ValueError(
            "No training batches created. Check your dataset and batch_size."
        )
    grad_accum = args.gradient_accumulation_steps
    if args.max_steps > 0:
        return args.max_steps
    if batches is not None:
        n_batches = len(batches)
        epoch_microbatches = _mlx_epoch_microbatches(
            args, batches, includes_epochs=includes_epochs,
        )
        if epoch_microbatches:
            # Each epoch's last micro-batch forces an optimizer step (HF's
            # do_sync_step), so an epoch costs the ceil and never the floor, which
            # dropped the ragged tail into the next epoch's window. num_train_epochs
            # is a float, so the product is rounded up as transformers does in
            # set_initial_training_values; truncating shortened 1.5 to one epoch and
            # stretched 0.5 to a full pass. Whole counts are unchanged.
            steps_per_epoch = _mlx_steps_per_epoch(
                epoch_microbatches, grad_accum,
            )
            if includes_epochs:
                # A prebuilt schedule may stop part-way through its last epoch,
                # and that tail costs its own windows, not a pro-rata share of a
                # full epoch's. Scaling steps_per_epoch by n_batches /
                # epoch_microbatches over-counts it: 1.5 epochs of 3 at accum 2
                # is 2 + 1 = 3 updates, where the ratio gives ceil(5/3 * 2) = 4.
                whole, tail = divmod(n_batches, epoch_microbatches)
                total_steps = whole * steps_per_epoch + math.ceil(
                    tail / grad_accum
                )
            else:
                total_steps = math.ceil(
                    float(args.num_train_epochs) * steps_per_epoch
                )
        else:
            total_steps = n_batches // grad_accum
        return max(1, total_steps)
    if args.num_train_epochs > 0:
        raise ValueError(
            "num_train_epochs requires a finite dataset (not streaming). "
            "Use max_steps instead, or disable streaming."
        )
    raise ValueError("max_steps must be > 0 when using streaming mode.")


class MLXTrainer:
    """MLX-native trainer for Apple Silicon, mirroring SFTTrainer's constructor API."""

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        dataset_text_field=None,
        max_seq_length=None,
        packing=None,
        data_collator=None,
        args=None,
        formatting_func=None,
        processor=None,
        callbacks=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataset = train_dataset
        self._mlx_train_dataset_for_batches = train_dataset
        self.eval_dataset = eval_dataset
        self.formatting_func = formatting_func
        # Use args or defaults
        self.args = args or MLXTrainingConfig()

        # Auto-detect VLM
        self._is_vlm = _is_vlm_model(model)

        # Constructor params override args if provided
        if dataset_text_field is not None:
            self.args.dataset_text_field = dataset_text_field
        if max_seq_length is not None:
            self.args.max_seq_length = max_seq_length
        if packing is not None:
            self.args.packing = packing

        if self.args.packing:
            print(
                "Unsloth: packing=True is not yet supported on MLX. "
                "Falling back to packing=False (standard padding)."
            )
            self.args.packing = False

        if (
            not self._is_vlm
            and self.train_dataset is not None
            and self.tokenizer is not None
            and self.args.streaming
            and _is_mlx_lazy_text_source(self.train_dataset)
        ):
            config = getattr(self.model, "_config", {})
            model_type = config.get("model_type") if isinstance(config, dict) else None
            self.tokenizer = normalize_mlx_chat_template(
                self.tokenizer,
                chat_template=getattr(self.args, "chat_template", None),
                model_name=getattr(self.model, "_hf_repo", None),
                model_type=model_type,
                is_vlm=False,
                strict=False,
            )
            self.train_dataset = _MLXIterableTokenizedDatasetView(
                self.train_dataset,
                self.tokenizer,
                dataset_text_field=self.args.dataset_text_field,
                formatting_func=self.formatting_func,
                append_eos=bool(getattr(self.args, "append_eos", True)),
                completion_only_loss=_text_completion_only_loss_arg(self.args),
                assistant_only_loss=_text_assistant_only_loss_arg(self.args),
                max_seq_length=self.args.max_seq_length,
            )
            self._mlx_train_dataset_for_batches = self.train_dataset
        elif (
            not self._is_vlm
            and self.train_dataset is not None
            and self.tokenizer is not None
            and hasattr(self.train_dataset, "__getitem__")
            and hasattr(self.train_dataset, "__len__")
        ):
            config = getattr(self.model, "_config", {})
            model_type = config.get("model_type") if isinstance(config, dict) else None
            self.train_dataset = _MLXTokenizedDatasetView(
                self.train_dataset,
                self.tokenizer,
                self.args.max_seq_length,
                formatting_func=self.formatting_func,
                dataset_text_field=self.args.dataset_text_field,
                chat_template=getattr(self.args, "chat_template", None),
                model_name=getattr(self.model, "_hf_repo", None),
                model_type=model_type,
                append_eos=bool(getattr(self.args, "append_eos", True)),
            )

        # Freeze non-LoRA params when LoRA is detected. Otherwise LayerNorm
        # weights stay trainable and adaptive optimizers NaN on step 1 (their
        # 1D second-moment init is numerically unstable).
        self._ensure_lora_frozen(model)

        # Training state. Per-run tracking lives in _reset_run_state (re-run at
        # each train() so a reused trainer starts clean); callbacks and
        # pre-created batches persist across runs and stay here.
        # stop_requested is cleared at train() entry, not in _reset_run_state, so
        # a cancel raised during THIS run's setup survives; set here so a trainer
        # inspected before train() has the attribute. _run_generation stamps each
        # request, so that clear only drops an EARLIER run's stop.
        self._run_generation = 0
        self.stop_requested = False
        self._reset_run_state()
        self._batches = None  # Pre-created batches (skips internal batch creation)
        self._step_callbacks = []  # Callbacks called after each logged step
        self._eval_callbacks = []  # Callbacks called after each eval

        # Hugging Face TrainerCallback support. The handler and TrainerState/
        # TrainerControl persist across train() runs (callbacks are not reset
        # in _reset_run_state); _init_callback_state re-seeds state at each run.
        self._ensure_callback_args_compat()
        self.callback_handler = _MLXCallbackHandler(
            callbacks or [],
            model=self.model,
            processing_class=self.processor or self.tokenizer,
            optimizer=None,
            lr_scheduler=None,
        )
        # Seed the real per-rank process-zero flags BEFORE on_init_end, as HF does.
        # Every rank constructs the trainer, so a callback gating file I/O on
        # is_world_process_zero would otherwise run once per rank: the defaults are
        # True everywhere and were corrected only later in _init_callback_state.
        # Each rank sets its own flag, so there is no lockstep concern.
        _is_main = bool(self.is_main_process)
        self.state = _MLXTrainerState(
            is_local_process_zero=_is_main,
            is_world_process_zero=_is_main,
        )
        self.control = _MLXTrainerControl()
        # Dispatch on_init_end under the same DDP failure consensus as _fire: now
        # that the flags are rank-specific, a callback raising on rank 0 only would
        # unwind that rank's __init__ while its peers proceed into train() and hang
        # at the next collective. OR-reduce so every rank aborts with the original
        # exception. on_init_end runs on all ranks, as in HF; single-process
        # re-raises unchanged.
        _init_error = None
        try:
            self.control = self.callback_handler.call_event(
                "on_init_end",
                self.args, self.state, self.control,
            )
        except BaseException as e:
            _init_error = e
        if self.distributed_world_size > 1:
            self._raise_distributed_failure(
                _init_error is not None, "on_init_end callback", _init_error,
            )
        elif _init_error is not None:
            raise _init_error

    @property
    def stop_requested(self):
        """True while an early stop is pending. Externally owned: a controller
        (e.g. the Studio cancel button) may set it at ANY time, even pre-train()."""
        return self._stop_requested

    @stop_requested.setter
    def stop_requested(self, value):
        self._stop_requested = value
        # Stamp the request with its run generation, bumped when a run finishes:
        # a stop latched by run N stamps N, a cancel raised after it stamps N+1.
        # train() clears only the former, so a pre-train() cancel is never lost.
        self._stop_requested_generation = getattr(self, "_run_generation", 0)

    def _stop_request_generation(self):
        """Generation the pending stop request was made in (0 when unset)."""
        return getattr(self, "_stop_requested_generation", 0)

    def _reset_run_state(self):
        """Per-run training/metric state. Reset from __init__ and inside
        _train_inner (after setup) so reusing a trainer for a second run starts
        clean; _early_stopped cleared so a run-1 early stop doesn't block run 2.
        stop_requested is deliberately NOT reset here: train() clears it once at
        entry (before any data prep / optimizer build), so an external cancel
        raised DURING this run's setup survives to the loop's top-of-loop
        _distributed_should_stop() check instead of being clobbered by this
        post-setup reset. Callbacks and pre-created batches persist across runs
        and aren't reset."""
        self._global_step = 0
        self._train_loss_history = []
        # Running token-weighted totals behind the returned train_loss.
        # _train_loss_history holds per-window means over unequal token counts, so
        # averaging those unweighted makes the result depend on the log cadence,
        # which any callback setting control.should_log can move.
        self._train_loss_token_sum = 0.0
        self._train_loss_token_total = 0
        self._train_loss_weighting_ok = True
        self._grad_norm_history = []
        self._tokens_per_second_history = []
        self._peak_memory_history = []
        self._step_times = []
        self._local_token_count_history = []
        self._global_token_count_history = []
        # Per-run eval metrics: cleared so a reused trainer that runs without
        # eval (eval_steps=0 or no eval dataset) does not report a prior run's
        # eval_loss/perplexity in its result. Repopulated by _evaluate.
        self._last_eval_metrics = {}
        self._early_stopped = False
        self._best_metric = None
        self._best_step = None
        self._es_patience_counter = 0
        # Restored from a checkpoint's saved num_input_tokens_seen by the resume
        # block; 0 on a fresh run so a reused trainer starts the counter clean.
        self._resume_num_input_tokens_seen = 0
        # Same contract for the checkpoint's callback-visible epoch: None on a
        # fresh run, so a reused trainer opens at HF's unstarted-epoch value.
        self._resume_epoch = None
        # Same contract for the checkpoint's ExportableState callback states, so a
        # reused trainer does not re-expose run-1's bookkeeping. HF rebuilds these
        # from the LIVE callbacks each run, reading trainer_state.json only on resume.
        self._resume_stateful_callbacks = {}
        # Same for the callback-visible log history: restored by the resume
        # block, empty on a fresh run so a reused trainer carries nothing over.
        self._resume_log_history = []
        # Same for TrainerState.best_metric: restored by the resume block, None
        # on a fresh run so a reused trainer drops run-1's watermark.
        self._resume_callback_best_metric = None
        self._resume_callback_best_step = None
        self._distributed_world = None
        self._distributed_initialized = False
        self._distributed_rank = 0
        self._distributed_world_size = 1
        self._distributed_is_main_process = True

    def _resolved_best_metric_name(self):
        """metric_for_best_model as it is looked up in eval metrics, mirroring
        HF Trainer: a present-but-None value falls back to eval_loss, and a
        bare name ("loss") gets the eval_ prefix eval metric keys carry."""
        name = getattr(self.args, "metric_for_best_model", None) or "eval_loss"
        return name if name.startswith("eval_") else f"eval_{name}"

    def _train_dataset_for_batches(self):
        """Return the internal dataset used for MLX batch construction."""
        return getattr(self, "_mlx_train_dataset_for_batches", self.train_dataset)

    def _ensure_distributed(self):
        """Initialize and cache MLX distributed metadata.

        MLX distributed collectives are no-ops at world size 1. The torch-backed
        MLX test shim returns ``None`` from ``mx.distributed.init()``, so keep a
        rank-0/world-size-1 fallback for non-real distributed runtimes.
        """
        if getattr(self, "_distributed_initialized", False):
            return getattr(self, "_distributed_world", None)

        world = None
        rank = 0
        world_size = 1
        distributed = getattr(mx, "distributed", None)
        init = getattr(distributed, "init", None) if distributed is not None else None
        if callable(init):
            backend = _mlx_distributed_backend_from_env()
            if backend is None:
                world = init()
            else:
                try:
                    world = init(backend=backend)
                except TypeError:
                    world = init()
            if world is not None:
                rank = int(world.rank())
                world_size = int(world.size())

        self._distributed_world = world
        self._distributed_rank = rank
        self._distributed_world_size = world_size
        self._distributed_is_main_process = rank == 0
        self._distributed_initialized = True
        return world

    @property
    def distributed_world(self):
        """Return the cached MLX distributed group, initializing it if needed."""
        return self._ensure_distributed()

    @property
    def distributed_rank(self):
        """Return this process rank in the MLX distributed group."""
        self._ensure_distributed()
        return self._distributed_rank

    @property
    def distributed_world_size(self):
        """Return the number of processes in the MLX distributed group."""
        self._ensure_distributed()
        return self._distributed_world_size

    @property
    def is_main_process(self):
        """Return whether this process should own user-visible side effects."""
        self._ensure_distributed()
        return self._distributed_is_main_process

    def _distributed_result_fields(self):
        """Fields included in training results for DDP inspection."""
        self._ensure_distributed()
        return {
            "distributed_world_size": self._distributed_world_size,
            "distributed_rank": self._distributed_rank,
            "distributed_is_main_process": self._distributed_is_main_process,
        }

    def _distributed_rank_vector(self, value, *, as_int=False):
        """Collect one scalar per rank into a rank-indexed list."""
        world = self._ensure_distributed()
        if as_int:
            local_value = int(value)
            dtype = mx.int64
        else:
            local_value = float(value)
            dtype = mx.float32
        if world is None or self._distributed_world_size <= 1:
            return [local_value]
        values = [0 for _ in range(self._distributed_world_size)]
        values[self._distributed_rank] = local_value
        gathered = self._distributed_all_sum(
            mx.array(values, dtype=dtype), stream=mx.cpu,
        )
        mx.eval(gathered)
        if as_int:
            return [int(item) for item in gathered.tolist()]
        return [float(item) for item in gathered.tolist()]

    def _distributed_rank_history(self, values, *, as_int=False):
        """Collect a scalar history from every rank without string/object gather."""
        world = self._ensure_distributed()
        values = list(values or [])
        lengths = self._distributed_rank_vector(len(values), as_int=True)
        max_len = max(lengths) if lengths else len(values)
        sentinel = -1 if as_int else -1.0
        if world is None or self._distributed_world_size <= 1:
            history = [[int(value) if as_int else float(value)] for value in values]
            return {
                "lengths": lengths,
                "values": history,
            }

        padded = [
            values[index] if index < len(values) else sentinel
            for index in range(max_len)
        ]
        empty = [0 for _ in range(max_len)]
        rank_rows = [
            padded if rank == self._distributed_rank else empty
            for rank in range(self._distributed_world_size)
        ]
        dtype = mx.int64 if as_int else mx.float32
        gathered = self._distributed_all_sum(
            mx.array(rank_rows, dtype=dtype), stream=mx.cpu,
        )
        mx.eval(gathered)
        per_rank = gathered.tolist()
        history = [
            [
                None if per_rank[rank][index] == sentinel else per_rank[rank][index]
                for rank in range(self._distributed_world_size)
            ]
            for index in range(max_len)
        ]
        return {
            "lengths": lengths,
            "values": history,
        }

    def _distributed_training_diagnostics(
        self,
        *,
        total_time,
        trained_tokens,
        compile_scope,
        compile_fallback_reason,
    ):
        """Return DDP diagnostics after training while all ranks are still live."""
        self._ensure_distributed()
        hostname = socket.gethostname()
        host_digest = int.from_bytes(
            hashlib.blake2b(hostname.encode("utf-8"), digest_size=7).digest(),
            "little",
        )
        host_digests = self._distributed_rank_vector(host_digest, as_int=True)
        pids = self._distributed_rank_vector(os.getpid(), as_int=True)
        peak_memory = mx.get_peak_memory() / 1e9
        per_rank_peak_memory = self._distributed_rank_vector(peak_memory)
        per_rank_runtime = self._distributed_rank_vector(total_time)
        # trained_tokens is the all-reduced global total, so gathering it would
        # report the same world_size-inflated figure for every rank. Use this
        # rank's local accumulated tokens (same logging cadence) so the
        # per-rank field reflects true per-rank work.
        local_trained_tokens = int(
            sum(getattr(self, "_local_token_count_history", []))
        )
        per_rank_tokens = self._distributed_rank_vector(
            local_trained_tokens, as_int=True,
        )
        host_rank_map = [
            {
                "rank": rank,
                "host_digest": digest,
                "hostname": hostname if digest == host_digest else None,
                "pid": pids[rank] if rank < len(pids) else None,
                "is_local_host": digest == host_digest,
            }
            for rank, digest in enumerate(host_digests)
        ]
        return {
            "distributed_local_hostname": hostname,
            "distributed_host_rank_map": host_rank_map,
            "distributed_train_runtime_per_rank": per_rank_runtime,
            "distributed_train_runtime_max": max(per_rank_runtime),
            "distributed_train_runtime_min": min(per_rank_runtime),
            "distributed_trained_tokens_per_rank": per_rank_tokens,
            "distributed_global_token_count_history": [
                int(value) for value in getattr(
                    self, "_global_token_count_history", []
                )
            ],
            "distributed_per_rank_token_count_history": (
                self._distributed_rank_history(
                    getattr(self, "_local_token_count_history", []),
                    as_int=True,
                )
            ),
            "distributed_tokens_per_second_history": [
                float(value) for value in getattr(
                    self, "_tokens_per_second_history", []
                )
            ],
            "distributed_per_rank_tokens_per_second_history": (
                self._distributed_rank_history(
                    getattr(self, "_tokens_per_second_history", []),
                )
            ),
            "distributed_step_time_history": [
                float(value) for value in getattr(self, "_step_times", [])
            ],
            "distributed_per_rank_step_time_history": (
                self._distributed_rank_history(
                    getattr(self, "_step_times", []),
                )
            ),
            "distributed_peak_memory_gb": max(per_rank_peak_memory),
            "distributed_peak_memory_gb_per_rank": per_rank_peak_memory,
            "eval_metrics": dict(getattr(self, "_last_eval_metrics", {})),
            "compile_fallback": compile_scope == "fallback_eager",
            "compile_fallback_reason": compile_fallback_reason or "",
        }

    def _distributed_all_sum(self, value, stream=None):
        """All-sum a scalar/array on the trainer's distributed group."""
        world = self._ensure_distributed()
        if world is None or self._distributed_world_size <= 1:
            return value
        return mx.distributed.all_sum(value, group=world, stream=stream)

    def _distributed_all_max(self, value, stream=None):
        """All-max a scalar/array on the trainer's distributed group."""
        world = self._ensure_distributed()
        if world is None or self._distributed_world_size <= 1:
            return value
        return mx.distributed.all_max(value, group=world, stream=stream)

    def _distributed_any_flag(self, flag):
        """Return whether any rank reported ``flag``."""
        return self._distributed_status_mask(int(bool(flag))) > 0

    def _coordinate_text_shape_guard(
        self,
        shape_plan,
        frontier,
        report,
        compile_allowed,
        compile_policy,
        *,
        automatic=False,
        local_error=None,
        keep_exact_local=False,
        compile_mode=None,
    ):
        """Require every DDP rank to admit its local finite shape plan.

        ``compile_mode`` is the resolved mode (an override can select strict
        under a best_effort base), so a strict run aborts rather than degrade to
        eager. ``keep_exact_local`` keeps an exact automatic plan out of the
        shared-cap re-materialization, whose maximum would force compressed
        peers to decompress; those ranks still contribute a neutral value and
        run every collective in order, so the schedule is unchanged.
        """
        strict_mode = (compile_mode or compile_policy.mode) == "strict"
        if self.distributed_world_size <= 1:
            if local_error is not None:
                raise local_error
            return shape_plan, report, compile_allowed
        failed_any = self._distributed_any_flag(
            local_error is not None or not compile_allowed
        )
        if failed_any:
            if strict_mode:
                error = RuntimeError(
                    "Unsloth: strict mx.compile finite text shape planning "
                    "failed on at least one DDP rank."
                )
                if local_error is not None:
                    raise error from local_error
                raise error
            reason = (
                report.reason if not compile_allowed
                else "peer_planner_failure"
            )
            return None, replace(
                report,
                action="eager",
                reason=reason,
                planned_signatures=None,
                planned_endpoints=(),
                padding_tokens=0,
                cap_selection="not_applicable",
                padding_work_fraction=0.0,
                max_width_stretch=1.0,
                budget_satisfied=False,
            ), False
        if not automatic or frontier is None:
            return shape_plan, report, compile_allowed

        keep_local = keep_exact_local and report.action == "exact"
        shared_cap = self._distributed_max_int(
            1 if keep_local else report.effective_cap
        )
        final_plan = None
        final_error = None
        try:
            if not 1 <= shared_cap <= AUTOMATIC_TEXT_COMPILE_CEILING:
                raise RuntimeError(
                    "automatic finite text cap synchronization exceeded "
                    f"{AUTOMATIC_TEXT_COMPILE_CEILING}"
                )
            if not keep_local:
                final_plan = materialize_text_shape_frontier(
                    frontier,
                    cap=shared_cap,
                    cap_selection=report.cap_selection,
                )
                if final_plan.report.action == "eager":
                    raise RuntimeError(final_plan.report.reason)
        except Exception as exc:
            final_error = exc
        final_failed_any = self._distributed_any_flag(final_error is not None)
        if not final_failed_any:
            if keep_local:
                return shape_plan, report, True
            return final_plan, final_plan.report, True
        if strict_mode:
            error = RuntimeError(
                "Unsloth: strict mx.compile finite text shared-cap "
                "materialization failed on at least one DDP rank."
            )
            if final_error is not None:
                raise error from final_error
            raise error
        failure_cap = (
            shared_cap
            if 1 <= shared_cap <= AUTOMATIC_TEXT_COMPILE_CEILING
            else AUTOMATIC_TEXT_COMPILE_CEILING
        )
        return None, replace(
            report,
            action="eager",
            reason=(
                "shared_cap_materialization_failed"
                if final_error is not None else "peer_planner_failure"
            ),
            cap=failure_cap,
            effective_cap=failure_cap,
            planned_signatures=None,
            planned_endpoints=(),
            padding_tokens=0,
            cap_selection="not_applicable",
            padding_work_fraction=0.0,
            max_width_stretch=1.0,
            budget_satisfied=False,
        ), False

    def _distributed_status_mask(self, mask):
        """All-sum a small integer status code across ranks."""
        local = mx.array(int(mask), dtype=mx.int32)
        total = self._distributed_all_sum(local, stream=mx.cpu)
        mx.eval(total)
        return int(total.item())

    def _distributed_max_int(self, value):
        """All-max a bounded integer preflight value across ranks."""
        local = mx.array(int(value), dtype=mx.int32)
        maximum = self._distributed_all_max(local, stream=mx.cpu)
        mx.eval(maximum)
        return int(maximum.item())

    def _raise_distributed_failure_from_any(self, failed_any, context, exc=None):
        """Abort this rank after a rank-wide failure consensus."""
        if not failed_any:
            return
        if exc is not None and not isinstance(exc, Exception):
            # Interrupts were captured only so this rank could join the
            # consensus. Re-raise unwrapped without mutating trainer state, so
            # a reused trainer does not inherit a stop request.
            raise exc
        self.stop_requested = True
        if exc is not None:
            raise RuntimeError(
                f"Unsloth MLX DDP: rank {self.distributed_rank} failed during "
                f"{context}: {exc}"
            ) from exc
        raise RuntimeError(
            f"Unsloth MLX DDP: a peer rank failed during {context}; "
            "aborting all ranks."
        )

    def _raise_distributed_failure(self, failed, context, exc=None):
        """Abort all ranks if any rank failed before the next collective section."""
        self._raise_distributed_failure_from_any(
            self._distributed_any_flag(failed),
            context,
            exc,
        )

    def _distributed_sum_gradient_tree(self, grad):
        """All-sum a gradient tree while preserving MLX's grouped all-reduce."""
        world = self._ensure_distributed()
        if world is None or self._distributed_world_size <= 1:
            return grad
        averaged = nn.average_gradients(grad, group=world)
        return tree_map(
            lambda value: value * mx.array(
                self._distributed_world_size, dtype=value.dtype,
            ),
            averaged,
        )

    def _distributed_should_stop(self):
        """Synchronize stop requests so all ranks leave loops together."""
        should_stop = self._distributed_any_flag(self.stop_requested)
        if should_stop:
            self.stop_requested = True
        return should_stop

    def _distributed_sync_control_actions(self):
        """OR the callback log/eval/save requests across ranks.

        Callbacks fire on every rank, but a rank-dependent one can still flip
        control.should_log / should_evaluate / should_save on a subset. Those
        actions run collective code (metric all-reduce, eval, rank-0-guarded
        saves), so every rank must agree before entering them or the peers
        deadlock at the collective. One packed all-sum keeps the flags in
        lockstep; a no-op at world size 1.
        """
        world = self._ensure_distributed()
        if world is None or self._distributed_world_size <= 1:
            return
        base = self._distributed_world_size + 1
        code = (
            int(bool(self.control.should_log))
            + base * int(bool(self.control.should_evaluate))
            + base * base * int(bool(self.control.should_save))
        )
        total = self._distributed_status_mask(code)
        self.control.should_log = (total % base) > 0
        self.control.should_evaluate = ((total // base) % base) > 0
        self.control.should_save = ((total // (base * base)) % base) > 0

    def _distributed_eval_status(self, failed=False):
        """Synchronize eval stop/failure state with one rank-wide collective."""
        status_base = self.distributed_world_size + 1
        status = self._distributed_status_mask(
            int(bool(self.stop_requested)) + status_base * int(bool(failed))
        )
        should_stop = (status % status_base) > 0
        failed_any = (status // status_base) > 0
        if should_stop:
            self.stop_requested = True
        return should_stop, failed_any

    def _validate_distributed_resume_checkpoint(self, resume_path):
        """Ensure DDP ranks agree on a complete resume checkpoint."""
        world = self._ensure_distributed()
        if world is None or self._distributed_world_size <= 1:
            return resume_path

        local_resume = mx.array(int(bool(resume_path)), dtype=mx.int32)
        resume_count = self._distributed_all_sum(local_resume, stream=mx.cpu)
        mx.eval(resume_count)
        if int(resume_count.item()) == 0:
            return None
        if int(resume_count.item()) != self._distributed_world_size:
            raise RuntimeError(
                "Unsloth MLX DDP: all ranks must either resume from the same "
                "checkpoint or all start fresh."
            )

        path = Path(resume_path).expanduser().resolve(strict=False)
        digest = int.from_bytes(
            hashlib.blake2b(str(path).encode("utf-8"), digest_size=7).digest(),
            "little",
        )
        digests = self._distributed_rank_vector(digest, as_int=True)
        required = (
            "adapters.safetensors",
            "optimizer_state.safetensors",
            "trainer_state.json",
        )
        missing = sum(
            0 if (path / filename).is_file() else 1
            for filename in required
        )
        missing_total = self._distributed_all_sum(
            mx.array(missing, dtype=mx.int32), stream=mx.cpu,
        )
        mx.eval(missing_total)
        if any(int(item) != digest for item in digests):
            raise RuntimeError(
                "Unsloth MLX DDP: all ranks must use the same "
                "resume_from_checkpoint path."
            )
        if int(missing_total.item()) > 0:
            # missing_total is all-reduced, so every rank enters this branch
            # together and raising here cannot strand a peer in a later
            # collective. A rank that can see adapters.safetensors but not the
            # rest is holding a saved adapter directory, so give it the same
            # warm-start guidance the single-process path gives; the plain
            # visibility failure keeps the coordinated message below.
            if (path / "adapters.safetensors").is_file():
                _require_complete_resume_checkpoint(str(path))
            raise RuntimeError(
                "Unsloth MLX DDP: resume checkpoint is incomplete or not "
                "visible on every rank. Expected adapters.safetensors, "
                "optimizer_state.safetensors, and trainer_state.json."
            )
        return str(path)

    def add_step_callback(self, fn):
        """Register a callback called after each logged step.

        fn(step, total_steps, loss, lr, tokens_sec, peak_gb, elapsed,
           num_tokens, grad_norm=None)
        grad_norm: fp32 pre-clip norm — a float when global-norm clipping
        is active or ``report_grad_norm=True``, otherwise None.
        """
        self._step_callbacks.append(fn)

    def add_eval_callback(self, fn):
        """Register a callback called after each evaluation.

        fn(step, eval_loss, perplexity)
        """
        self._eval_callbacks.append(fn)

    def add_callback(self, callback):
        """Add a Hugging Face TrainerCallback class or instance."""
        self.callback_handler.add_callback(callback)
        self._ensure_callback_args_compat()

    def remove_callback(self, callback):
        """Remove a Hugging Face TrainerCallback class or instance."""
        self.callback_handler.remove_callback(callback)

    def pop_callback(self, callback):
        """Remove and return a Hugging Face TrainerCallback class or instance."""
        return self.callback_handler.pop_callback(callback)

    def _suppress_torch_only_final_artifacts(self):
        """Disable HF final-model artifacts for one on_train_end dispatch.

        WandbCallback.on_train_end and DVCLiveCallback.on_train_end both log that
        artifact by building a Torch ``Trainer`` around ``args``/``model``, which
        raises AttributeError here (``full_determinism`` on 5.x,
        ``batch_eval_metrics`` on 4.57.x). Adapters are already saved by then, so
        the casualty is the caller's MLXTrainOutput -- and for DVCLive also the
        ``self.live.end()`` that trails the artifact block, leaving the tracked
        run unfinalized. Per-checkpoint artifacts are untouched: neither on_save
        builds a Trainer, and DVCLive's ``log_model="all"`` (its per-checkpoint
        mode) never enters the Trainer branch, which upstream gates on
        ``self._log_model is True``. Returns (callback, previous_mode) pairs for
        _restore_final_artifact_modes. Duck-typed so unsloth_zoo.mlx still
        imports without Torch.
        """
        suppressed = []
        for callback in getattr(self.callback_handler, "callbacks", ()):
            # Match the MRO, not just the concrete class: subclassing an
            # integration callback to customise logging is a common recipe and
            # inherits the same on_train_end.
            names = {base.__name__ for base in type(callback).__mro__}
            mode = getattr(callback, "_log_model", None)
            if "WandbCallback" in names:
                if mode is None or not getattr(mode, "is_enabled", False):
                    continue
                try:
                    # WandbLogModel is a str Enum, so the "false" member is
                    # reachable from the instance without importing transformers.
                    disabled = type(mode)("false")
                except Exception:
                    continue
                if getattr(disabled, "is_enabled", True):
                    continue
            elif "DVCLiveCallback" in names:
                # Mirror upstream's identity test, so "all" keeps logging its
                # per-checkpoint artifact (on_save, no Trainer) untouched.
                if mode is not True:
                    continue
                disabled = False
            else:
                continue
            callback._log_model = disabled
            suppressed.append((callback, mode))
        return suppressed

    def _restore_final_artifact_modes(self, suppressed):
        """Undo _suppress_torch_only_final_artifacts on the user's callbacks."""
        for callback, mode in suppressed:
            callback._log_model = mode

    def _suppress_torch_only_wandb_watch(self):
        """Neutralize WANDB_WATCH for one on_train_begin dispatch.

        WandbCallback.setup calls wandb.watch(model, ...) when WANDB_WATCH is
        gradients/parameters/all, and wandb.watch raises TypeError("Expected a
        pytorch model (torch.nn.Module)") on an mlx Module, so the opt-in aborts
        training during callback setup. Upstream reads the environment variable
        directly, so this is the lever that does not monkeypatch wandb. Returns
        the previous value for _restore_wandb_watch, or None when there is
        nothing to suppress.
        """
        previous = os.environ.get("WANDB_WATCH", "false")
        if previous not in ("all", "parameters", "gradients"):
            return None
        if not any(
            "WandbCallback" in {base.__name__ for base in type(callback).__mro__}
            for callback in getattr(self.callback_handler, "callbacks", ())
        ):
            return None
        os.environ["WANDB_WATCH"] = "false"
        return previous

    def _restore_wandb_watch(self, previous):
        """Undo _suppress_torch_only_wandb_watch."""
        if previous is not None:
            os.environ["WANDB_WATCH"] = previous

    def _ensure_callback_args_compat(self):
        """Populate TrainingArguments-style fields read by common callbacks."""
        args = self.args
        self._sync_synthesized_arg(
            "logging_strategy",
            "steps" if getattr(args, "logging_steps", 0) else "no",
        )
        self._sync_synthesized_arg(
            "eval_strategy", self._default_callback_eval_strategy(),
        )
        self._sync_synthesized_arg(
            "save_strategy",
            "steps" if getattr(args, "save_steps", 0) else "no",
        )
        if not hasattr(args, "logging_first_step"):
            args.logging_first_step = False
        if not hasattr(args, "eval_delay"):
            args.eval_delay = 0
        if not hasattr(args, "include_num_input_tokens_seen"):
            args.include_num_input_tokens_seen = False
        # Integration-facing fields, at HF's own TrainingArguments defaults.
        # TrackioCallback and SwanLabCallback read them directly in
        # on_train_begin, so a missing one aborts the run before step 1.
        if not hasattr(args, "project"):
            args.project = "huggingface"
        for _integration_arg in (
            "trackio_space_id", "trackio_bucket_id", "trackio_static_space_id",
            "hub_private_repo", "resume_from_checkpoint",
        ):
            if not hasattr(args, _integration_arg):
                setattr(args, _integration_arg, None)
        if getattr(args, "logging_dir", None) is None:
            args.logging_dir = os.path.join(args.output_dir, "runs")
        if getattr(args, "run_name", None) is None:
            args.run_name = args.output_dir

    def _sync_synthesized_arg(self, name, value):
        """Set a callback-compat arg we synthesized, refreshing it per run.

        These strategies derive from MLX knobs that stay writable after
        construction, so without a per-run refresh a trainer built without eval
        keeps eval_strategy="no" once eval is enabled later and HF's
        EarlyStoppingCallback asserts in on_train_begin. Only values this trainer
        wrote are refreshed; a real field or a user override is never clobbered.
        """
        args = self.args
        synthesized = getattr(self, "_synthesized_callback_args", None)
        if synthesized is None:
            synthesized = self._synthesized_callback_args = {}
        current = getattr(args, name, None)
        if name in synthesized:
            # Someone changed our value by hand: theirs wins from now on.
            if current != synthesized[name]:
                del synthesized[name]
                return
        elif hasattr(args, name) and current is not None:
            return
        setattr(args, name, value)
        synthesized[name] = value

    def _default_callback_eval_strategy(self):
        """Return the MLX-derived eval strategy for callback compatibility."""
        return (
            "steps"
            if self.eval_dataset is not None and getattr(self.args, "eval_steps", 0)
            else "no"
        )

    def _static_cadence_enabled(self, name):
        """Whether the loop's own step cadence applies, as HF decides it.

        _sync_synthesized_arg deliberately preserves a caller-supplied strategy
        (a real TrainingArguments/SFTConfig carries one, and a hand-set override
        wins over our derivation), so the loop must read the same field HF's
        DefaultFlowCallback reads instead of acting on the interval alone.
        transformers only raises a step-interval action under
        IntervalStrategy/SaveStrategy.STEPS (trainer_callback.
        DefaultFlowCallback.on_step_end), so "no" must not act at all and
        "epoch" must leave the cadence to on_epoch_end rather than adding a
        second action on top of it.

        Both are str Enums, so a plain "steps" and the member itself normalize
        here. A missing field means the args object never went through
        _ensure_callback_args_compat, so keep the legacy interval-only cadence
        rather than silently disabling the action.
        """
        strategy = getattr(self.args, name, None)
        if strategy is None:
            return True
        strategy = getattr(strategy, "value", strategy)
        return str(strategy).lower() == "steps"

    def _static_eval_cadence_enabled(self):
        """Whether the loop's own eval_steps cadence applies."""
        return self._static_cadence_enabled("eval_strategy")

    def _static_log_cadence_enabled(self):
        """Whether the loop's own logging_steps cadence applies."""
        return self._static_cadence_enabled("logging_strategy")

    def _static_save_cadence_enabled(self):
        """Whether the loop's own save_steps cadence applies."""
        return self._static_cadence_enabled("save_strategy")

    def _best_save_strategy_enabled(self):
        """Whether save_strategy asks for HF's save-on-improvement rule.

        SaveStrategy.BEST (transformers >= 4.47) is the one member
        DefaultFlowCallback never raises: its on_step_end and on_epoch_end act
        on STEPS and EPOCH only, so HF's Trainer core decides this one itself,
        immediately after the evaluation that produced the metric. Gating the
        static cadence on the strategy therefore left "best" with no cadence at
        all -- the same hole the epoch cadence fills for "epoch". Same str-Enum
        normalization as the other two; a missing field is not "best".
        """
        strategy = getattr(self.args, "save_strategy", None)
        if strategy is None:
            return False
        strategy = getattr(strategy, "value", strategy)
        return str(strategy).lower() == "best"

    def _epoch_cadence_enabled(self, name):
        """Whether a strategy field asks for HF's epoch cadence.

        The mirror of _static_cadence_enabled for the other half of the flow,
        DefaultFlowCallback.on_epoch_end, which raises its action under
        IntervalStrategy/SaveStrategy.EPOCH. Same str-Enum normalization; a
        missing field means the args object never went through
        _ensure_callback_args_compat, so it keeps the legacy interval-only
        cadence and gains no epoch action it never asked for.
        """
        strategy = getattr(self.args, name, None)
        if strategy is None:
            return False
        strategy = getattr(strategy, "value", strategy)
        return str(strategy).lower() == "epoch"

    def _request_epoch_cadence_actions(self):
        """Raise the epoch-strategy log/eval/save requests at a dataset boundary.

        transformers' Trainer always installs DefaultFlowCallback, so its
        on_epoch_end is what turns a caller's "epoch" strategy into an action.
        MLXTrainer installs no flow callback of its own, so gating the loop's
        static interval on the strategy (see _static_cadence_enabled) would
        otherwise leave a caller who hand-sets "epoch" and passes their own
        callbacks with NO periodic log, checkpoint or evaluation at all -- worse
        than the wrong-cadence bug that gating fixes. Raise the same requests
        here so the cadence holds either way.

        Deduplicated against an installed flow by construction: this sets
        exactly the control flags DefaultFlowCallback.on_epoch_end sets, so the
        two requests coalesce into the single boolean the loop already clears
        when it runs the action (CallbackHandler.on_log/on_evaluate/on_save
        clear theirs the same way) -- one on_save, one checkpoint-N write, one
        eval_loss in log_history per boundary. It adds no repeat of its own:
        every boundary that already dispatched on_epoch_end now carries the
        request, and no boundary is visited twice. Callers invoke this
        immediately BEFORE firing on_epoch_end, which is also where HF raises it
        (the flow sits at index 0 of the callback list), so the callbacks that
        follow observe the same control state they would under HF.

        This does not, and cannot, deduplicate ACROSS boundaries: when an epoch
        is shorter than one accumulation window (_mlx_epoch_microbatches returns
        None under max_steps for a source with no cycle_length, so the ragged
        tail is not forced to an optimizer step) the loop dispatches several
        on_epoch_end events at one global_step and repeats the action at each,
        exactly as it already does for a caller who installs the flow.

        DDP-safe with no collective of its own: args and state.epoch are
        rank-consistent, and every call site sits on the near side of the
        _distributed_sync_control_actions() that follows the on_epoch_end fire,
        which OR-reduces the flags before any rank enters the collective
        log/eval/save paths.
        """
        if self._epoch_cadence_enabled("logging_strategy"):
            self.control.should_log = True
        if self._epoch_cadence_enabled("eval_strategy"):
            # on_epoch_end compares eval_delay against state.epoch, not the step
            # its on_step_end gate uses. state.epoch is set by the caller before
            # this runs; the None fallback only covers helper paths with no
            # epoch length, which never reach a boundary anyway.
            epoch = getattr(self.state, "epoch", None)
            if epoch is None or self._eval_delay_satisfied(epoch):
                self.control.should_evaluate = True
        if self._epoch_cadence_enabled("save_strategy"):
            self.control.should_save = True

    def _request_step_cadence_actions(self):
        """Raise the non-interval requests DefaultFlowCallback.on_step_end makes.

        The loop's static interval (see _static_cadence_enabled) covers the
        periodic half of DefaultFlowCallback.on_step_end. This covers the rest:
        the logging_first_step log at step 1, and the save/eval HF forces once
        state.global_step reaches state.max_steps. MLXTrainer installs no flow
        callback of its own, so without these a run whose interval does not
        divide the step budget silently lost its LAST resumable checkpoint --
        checkpoint-<max_steps> and its on_save, the one an interrupted run
        resumes from, which the unconditional final save_model() cannot replace
        because it holds adapters only, no optimizer or trainer state -- and, on
        a transformers that forces one, the final evaluation that
        load_best_model_at_end and EarlyStoppingCallback read last.

        Deduplicated against an installed flow exactly like
        _request_epoch_cadence_actions: these are the control flags
        DefaultFlowCallback sets, so both requests coalesce into the single
        boolean the loop clears when it runs the action -- one on_save, one
        checkpoint-N write, one eval_loss in log_history. Callers raise it
        immediately BEFORE on_step_end, which is where the flow raises it (index
        0 of the callback list), so the callbacks that follow observe the same
        control state they would under HF.

        The final-step test reads state.max_steps, the same fixed field HF's
        flow tests, not the loop's live budget: a run whose budget
        _honor_epoch_stop_skip shrank stops short of max_steps and gets no
        forced final action, with or without a flow installed.

        DDP-safe with no collective of its own: args and state are
        rank-consistent, and every call site sits on the near side of the
        _distributed_sync_control_actions() that follows the on_step_end fire.
        """
        args = self.args
        current_step = int(getattr(self.state, "global_step", 0) or 0)
        # Deliberately not gated on logging_strategy: HF raises the first-step
        # log before it tests the strategy, so "no" and "epoch" log step 1 too.
        if current_step == 1 and getattr(args, "logging_first_step", False):
            self.control.should_log = True
        max_steps = int(getattr(self.state, "max_steps", 0) or 0)
        if max_steps <= 0 or current_step < max_steps:
            return
        # HF forces the final save from the strategy alone, with no save_steps
        # guard, so an interval that does not divide the budget still leaves a
        # checkpoint at the end.
        if self._static_save_cadence_enabled():
            self.control.should_save = True
        eval_steps = int(getattr(self.state, "eval_steps", 0) or 0)
        if (
            _default_flow_evaluates_final_step()
            # HF skips the forced eval when the interval already evaluated here.
            # Its modulo has no zero guard; a 0 interval means "never" for the
            # loop's own cadence, so keep it meaning that rather than raising.
            and eval_steps > 0
            and current_step % eval_steps != 0
            and self._static_eval_cadence_enabled()
            and self._eval_delay_satisfied(current_step)
        ):
            self.control.should_evaluate = True

    def _eval_delay_satisfied(self, current_step):
        """Whether HF's eval_delay allows the step cadence to evaluate yet.

        DefaultFlowCallback gates its step-strategy evaluation on
        `args.eval_delay <= state.global_step`, so the static cadence honors
        the same bound. _ensure_callback_args_compat defaults the field to 0,
        and an unparseable value falls back to "no delay" so a bad override
        cannot disable evaluation outright.
        """
        try:
            delay = float(getattr(self.args, "eval_delay", 0) or 0)
        except (TypeError, ValueError):
            return True
        return delay <= float(current_step)

    def _callback_num_train_epochs(self, total_steps, batches):
        """Return the epoch total HF's TrainerState reports for this run.

        Epoch-count runs use num_train_epochs directly. A max_steps run
        derives the total from the dataloader length as HF does: reporting 0
        while the loop dispatches epoch events contradicts state.epoch and
        divides by zero downstream.

        The count is CEILED, like HF's `num_train_epochs =
        math.ceil(args.num_train_epochs)` (transformers
        set_initial_training_values) and like the step budget, which already
        ceils. Truncating it reported 1 for num_train_epochs=1.5 while
        state.epoch still climbed to 1.5, so a callback normalizing progress by
        this total read 150 percent.
        """
        epochs = max(0, math.ceil(float(
            getattr(self.args, "num_train_epochs", 0) or 0
        )))
        # HF derives the total from max_steps whenever it is set, ignoring
        # num_train_epochs, which a real TrainingArguments leaves positive.
        if epochs > 0 and int(getattr(self.args, "max_steps", 0) or 0) <= 0:
            return epochs
        total_steps = int(total_steps or 0)
        if total_steps <= 0:
            return epochs
        micro_per_epoch = self._callback_batches_per_epoch(batches)
        if not micro_per_epoch:
            # Streaming: no known boundaries, so the loop fires no epoch events.
            return epochs
        grad_accum = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1) or 1))
        updates_per_epoch = max(1, math.ceil(micro_per_epoch / grad_accum))
        return max(1, math.ceil(total_steps / updates_per_epoch))

    def _init_callback_state(self, total_steps, resume_step, batches=None):
        """Initialize TrainerState for HF callback lifecycle events."""
        args = self.args
        # Expand HF's fractional (ratio-of-total-steps) intervals to absolute
        # counts before storing them, as TrainerState.compute_steps does.
        eval_steps = _resolve_interval_steps(
            getattr(args, "eval_steps", 0), total_steps,
        )
        # Reflect the real MLX rank so HF callbacks (loggers, savers) gate their
        # own I/O correctly. MLX distributed is single-node, so local and world
        # process-zero both track rank 0.
        is_main_process = self.is_main_process
        self.state = _MLXTrainerState(
            global_step=int(resume_step),
            # The checkpoint's epoch on resume, so the lifecycle events
            # dispatched before the loop rebuilds epoch progress carry the same
            # value HF restores from trainer_state.json. A fresh run starts at
            # 0.0, TrainerState's own default: leaving it None meant a run
            # cancelled before the loop (an on_train_begin stop, or an external
            # stop already pending) dispatched on_train_end with epoch=None, and
            # stock callbacks read it as a number -- NotebookProgressCallback
            # does int(state.epoch) there.
            epoch=getattr(self, "_resume_epoch", None) or 0.0,
            max_steps=int(total_steps),
            logging_steps=_resolve_interval_steps(
                getattr(args, "logging_steps", 0), total_steps,
            ),
            eval_steps=eval_steps,
            save_steps=_resolve_interval_steps(
                getattr(args, "save_steps", 0), total_steps,
            ),
            train_batch_size=int(getattr(args, "per_device_train_batch_size", 0) or 0),
            num_train_epochs=self._callback_num_train_epochs(total_steps, batches),
            num_input_tokens_seen=int(
                getattr(self, "_resume_num_input_tokens_seen", 0) or 0
            ),
            # Continue the checkpoint's history on resume (HF restores the whole
            # TrainerState from trainer_state.json); empty on a fresh run.
            log_history=list(getattr(self, "_resume_log_history", None) or []),
            is_local_process_zero=is_main_process,
            is_world_process_zero=is_main_process,
        )
        # Seed the callback-visible best-model fields from the restored native best
        # state, or a resumed run starts at best_metric=None and EarlyStoppingCallback
        # treats the first post-resume eval as the new best, overwriting the real one
        # and diverging from _run_best_tracking. Own checkpoint key, since it advances
        # even when native tracking is off; falls back to the native value on fresh
        # runs and pre-fix checkpoints.
        _cb_best_metric = getattr(self, "_resume_callback_best_metric", None)
        _cb_best_step = getattr(self, "_resume_callback_best_step", None)
        self.state.best_metric = (
            self._best_metric if _cb_best_metric is None else _cb_best_metric
        )
        self.state.best_global_step = (
            self._best_step if _cb_best_step is None else _cb_best_step
        )
        if self._best_step is not None:
            self.state.best_model_checkpoint = f"{args.output_dir}/best"
        self.control = _MLXTrainerControl()

    def _sync_callback_stop(self):
        """Mirror TrainerControl stop requests into MLXTrainer's loop flag."""
        if getattr(self.control, "should_training_stop", False):
            self.stop_requested = True

    def _call_callback_log(self, logs):
        """Record and dispatch a Hugging Face on_log callback event."""
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        output = dict(logs)
        output["step"] = self.state.global_step
        self.state.log_history.append(output)
        self.control.should_log = False
        self.control = self.callback_handler.call_event(
            "on_log",
            self.args, self.state, self.control, logs=logs,
        )

    def _call_callback_evaluate(self, metrics):
        """Dispatch a Hugging Face on_evaluate callback event."""
        self._call_callback_log(dict(metrics))
        self.control.should_evaluate = False
        self.control = self.callback_handler.call_event(
            "on_evaluate",
            self.args, self.state, self.control, metrics=metrics,
        )

    def _call_callback_save(self):
        """Dispatch a Hugging Face on_save callback event."""
        self.control.should_save = False
        self.control = self.callback_handler.call_event(
            "on_save",
            self.args, self.state, self.control,
        )

    def _callback_batches_per_epoch(self, batches):
        """Return the finite micro-batch count for one callback epoch."""
        if batches is None:
            return None
        total = len(batches)
        if total <= 0:
            return None
        # Epoch-count runs share this length with the step budget and the forced
        # epoch-final update, so both read it from one place.
        epoch_microbatches = _mlx_epoch_microbatches(
            self.args,
            batches,
            includes_epochs=getattr(
                self, "_prepared_batches_include_epochs", False,
            ),
        )
        if epoch_microbatches:
            return epoch_microbatches
        if getattr(self, "_prepared_batches_include_epochs", False):
            epochs = int(getattr(self.args, "num_train_epochs", 0) or 0)
            if epochs > 0 and total % epochs == 0:
                return max(1, total // epochs)
            return total
        # max_steps>0: `batches` is the whole cycled run, NOT one dataset pass, so
        # using its length as the epoch would make state.epoch climb to 1.0 across
        # the run and fire the epoch events once, starving epoch-based callbacks.
        # Approximate the per-epoch count from the dataset size and global batch,
        # matching HF's len(dataloader) accounting. Callback-visible only; never
        # touches the data or the gradient steps. Floored at 1 but not
        # upper-clamped, so a sub-one-pass run reports its true fraction.
        if int(getattr(self.args, "max_steps", 0) or 0) > 0:
            # Prefer the plan's own one-pass count: the approximation below
            # cannot see what batching retained (sub-two-token rows dropped, one
            # source item expanding into several, floored tail), so it fires the
            # epoch events at micro-batches that are not dataset boundaries.
            plan_cycle = getattr(batches, "cycle_length", None)
            if plan_cycle:
                return max(1, int(plan_cycle))
            per_device = int(getattr(self.args, "per_device_train_batch_size", 0) or 0)
            world = int(getattr(self, "_distributed_world_size", 1) or 1)
            ds = getattr(self, "_mlx_train_dataset_for_batches", None)
            if ds is None:
                ds = self.train_dataset
            try:
                n_examples = len(ds)
            except TypeError:
                n_examples = 0
            if per_device > 0 and n_examples > 0:
                one_pass = math.ceil(n_examples / (per_device * max(1, world)))
                return max(1, one_pass)
        return total

    def _metric_for_best_model_name(self, metrics=None, require=False):
        """Return the HF-normalized metric_for_best_model key."""
        metric_name = getattr(self.args, "metric_for_best_model", None)
        if not metric_name:
            return None
        metric_name = str(metric_name)
        if not metric_name.startswith("eval_"):
            metric_name = f"eval_{metric_name}"
        if metrics is not None and metric_name not in metrics:
            if require:
                raise ValueError(
                    f"metric_for_best_model={metric_name!r} not in eval "
                    f"metrics; available: {sorted(metrics)}"
                )
            return None
        return metric_name

    def _export_callback_states(self):
        """Populate state.stateful_callbacks from ExportableState callbacks.

        HF does this in _save_checkpoint, so a callback's internal bookkeeping
        (EarlyStoppingCallback.early_stopping_patience_counter) travels with the
        checkpoint. Duck-typed on a working `state()` to keep this module
        Torch-free. TrainerControl is excluded: MLX rebuilds control flags every
        train(), and rehydrating should_training_stop would end a resume at step 0.
        """
        exported = {}
        for cb in self.callback_handler.callbacks:
            state_fn = getattr(cb, "state", None)
            if not callable(state_fn):
                continue
            try:
                cb_state = state_fn()
            except NotImplementedError:
                continue  # ExportableState base, or a non-exporting callback
            if not isinstance(cb_state, dict):
                continue
            name = type(cb).__name__
            if name in exported:
                # HF stores duplicates of one class as a list, positionally
                # matched back to the callbacks on restore.
                if not isinstance(exported[name], list):
                    exported[name] = [exported[name]]
                exported[name].append(cb_state)
            else:
                exported[name] = cb_state
        self.state.stateful_callbacks = exported
        return exported

    def _restore_callback_states(self, stateful_callbacks):
        """Seed state.stateful_callbacks from a checkpoint. Rehydrating the live
        callbacks is HF's opt-in restore_callback_states_from_checkpoint path,
        which this config does not expose, so only the visible state is mirrored."""
        if isinstance(stateful_callbacks, dict) and stateful_callbacks:
            self.state.stateful_callbacks = dict(stateful_callbacks)

    def _update_callback_best_metric(self, metrics):
        """Update TrainerState.best_metric after eval callbacks inspect prior state.

        Returns whether this evaluation is a new best, like HF's
        _determine_best_metric, so save_strategy="best" can act on it. No
        configured metric and a NaN both mean "not a new best", which is also
        what HF concludes (metric_for_best_model None short-circuits its
        is_new_best_metric to False, and np.less/np.greater are False on NaN).
        """
        metric_name = self._metric_for_best_model_name(metrics, require=False)
        if metric_name is None:
            return False
        value = metrics[metric_name]
        if value != value:
            return False
        greater = bool(getattr(self.args, "greater_is_better", False))
        improved = (
            self.state.best_metric is None
            or (value > self.state.best_metric if greater else value < self.state.best_metric)
        )
        if improved:
            self.state.best_metric = value
            self.state.best_global_step = self.state.global_step
        return improved

    @staticmethod
    def _apply_compile_recommendations(args, decision):
        """Apply safe compile setting recommendations to the active args object."""

        applied = []
        if decision is None:
            return applied
        for rec in getattr(decision, "setting_recommendations", ()):
            if rec.setting == "gradient_checkpointing" and args.compile_auto_tune:
                if bool(getattr(args, "gradient_checkpointing", True)) is False:
                    args.gradient_checkpointing = bool(rec.recommended_value)
                    applied.append((rec.setting, rec.recommended_value, rec.reason))
        return applied

    @staticmethod
    def _ensure_lora_frozen(model):
        """Freeze accidentally trainable norm params when LoRA is active.

        LayerNorm/RMSNorm weights left trainable make adaptive optimizers NaN
        on 1D tensors at init (second-moment starts at 0 -> divide by ~eps).
        Only norms are frozen; projector/vision/other intentional non-LoRA
        params are left alone.
        """
        trainable = dict(tree_flatten(model.trainable_parameters()))
        if not trainable:
            return  # nothing trainable; stub models may lack model.parameters().
        adapter_tensors = collect_mlx_lora_adapter_tensors(model)
        has_lora = any(name in trainable for name in adapter_tensors)
        if not has_lora:
            return  # Not a LoRA model — don't touch

        # Only freeze accidentally-unfrozen norms; leave components the user
        # explicitly unfroze (train_projector, train_vision) alone.
        _NORM_FRAGMENTS = (".norm.", "norm.weight", "norm.bias",
                           ".ln_", "ln_f.weight", "ln_f.bias")
        _INTENTIONAL_COMPONENTS = (
            "multi_modal_projector", "mm_projector", "connector", "aligner",
            "vision_tower", "vision_model", "vision_encoder",
        )
        adapter_keys = set(adapter_tensors)
        suspect = [
            k for k in trainable
            if k not in adapter_keys
            and any(frag in k for frag in _NORM_FRAGMENTS)
            and not any(comp in k for comp in _INTENTIONAL_COMPONENTS)
        ]
        if not suspect:
            return  # No accidental norms — nothing to fix

        for key in suspect:
            parts = key.split(".")
            obj = model
            for p in parts[:-1]:
                try:
                    obj = obj[int(p)]
                except (ValueError, TypeError):
                    obj = getattr(obj, p)
            obj.freeze(keys=[parts[-1]], recurse=False)

        print(
            f"Unsloth: Froze {len(suspect)} accidentally trainable norm "
            f"parameters to prevent optimizer NaN."
        )

    def _resolve_warmup_steps(self, total_steps):
        get_warmup_steps = getattr(self.args, "get_warmup_steps", None)
        if callable(get_warmup_steps):
            return max(0, int(get_warmup_steps(total_steps)))

        warmup_steps = int(getattr(self.args, "warmup_steps", 0) or 0)
        warmup_ratio = getattr(self.args, "warmup_ratio", 0.0)
        if warmup_ratio is None:
            return max(0, warmup_steps)
        try:
            warmup_ratio = float(warmup_ratio)
        except (TypeError, ValueError):
            return max(0, warmup_steps)
        if warmup_ratio == 0.0:
            return max(0, warmup_steps)

        default_warmup_steps = getattr(type(self.args), "warmup_steps", 5)
        steps_explicit = getattr(
            self.args,
            "_unsloth_mlx_warmup_steps_explicit",
            warmup_steps != default_warmup_steps,
        )
        # HF get_warmup_steps parity: a zero warmup_steps never overrides a positive
        # warmup_ratio. warmup_steps == 0 means "use the ratio" even when explicitly
        # set, so only a positive explicit step count wins over the ratio.
        if steps_explicit and warmup_steps > 0:
            return max(0, warmup_steps)

        resolved = math.ceil(max(0.0, warmup_ratio) * max(0, int(total_steps)))
        return min(max(0, int(total_steps)), max(0, resolved))

    def _build_schedule(self, total_steps):
        """Build LR schedule from config. Returns a callable or float."""
        lr = self.args.learning_rate
        warmup = self._resolve_warmup_steps(total_steps)
        sched_type = _normalize_mlx_scheduler_type(self.args.lr_scheduler_type)

        if sched_type == "constant" and warmup == 0:
            return lr

        def warmup_multiplier(step):
            if warmup <= 0:
                return mx.array(1.0, dtype=mx.float32)
            return step / mx.array(max(warmup, 1), dtype=mx.float32)

        def decay_progress(step):
            return (
                step - mx.array(warmup, dtype=mx.float32)
            ) / mx.array(max(total_steps - warmup, 1), dtype=mx.float32)

        def schedule(step):
            # HF Trainer LR parity; `step` is zero-based optimizer-step index.
            step = mx.array(step).astype(mx.float32)
            if warmup > 0:
                warm = lr * warmup_multiplier(step)
            else:
                warm = mx.array(lr, dtype=mx.float32)

            progress = decay_progress(step)
            if sched_type == "cosine":
                decay = mx.array(0.5, dtype=mx.float32) * (
                    mx.array(1.0, dtype=mx.float32) + mx.cos(mx.array(math.pi) * progress)
                )
            elif sched_type == "linear":
                decay = mx.array(1.0, dtype=mx.float32) - progress
            else:  # constant with warmup
                decay = mx.array(1.0, dtype=mx.float32)
            decay = mx.maximum(decay, mx.array(0.0, dtype=mx.float32))
            main = mx.array(lr, dtype=mx.float32) * decay
            return mx.where(step < warmup, warm, main)

        return schedule

    @staticmethod
    def _schedule_value(schedule, step):
        if callable(schedule):
            return schedule(mx.array(step))
        return schedule

    def _set_optimizer_lr_for_step(self, optimizer, step):
        schedule = getattr(self, "_lr_schedule", None)
        if schedule is None:
            return
        optimizer.learning_rate = self._schedule_value(schedule, step)

    def _build_optimizer(self, total_steps):
        """Create MLX optimizer with LR schedule from config.

        For AdamW, MLX applies weight decay inside the leaf update without a
        parameter-group filter. Keep MLX AdamW's built-in decay disabled and
        apply decoupled decay ourselves so bias and norm parameters match
        HuggingFace Trainer behavior.
        """
        schedule = self._build_schedule(total_steps)
        initial_lr = self._schedule_value(schedule, 0)
        self._lr_schedule = schedule if callable(schedule) else None
        wd = self.args.weight_decay
        self._manual_weight_decay = 0.0
        self._coupled_weight_decay = 0.0
        adam_beta1 = getattr(self.args, "adam_beta1", None)
        adam_beta2 = getattr(self.args, "adam_beta2", None)
        adam_kwargs = {}
        if adam_beta1 is not None or adam_beta2 is not None:
            adam_kwargs["betas"] = (
                float(0.9 if adam_beta1 is None else adam_beta1),
                float(0.999 if adam_beta2 is None else adam_beta2),
            )

        opt_name = _normalize_mlx_optimizer_name(self.args.optim)
        if opt_name == "adafactor":
            unsupported = self._adafactor_unsupported_parameters(self.model)
            if unsupported:
                preview = ", ".join(
                    f"{name}{shape}" for name, shape in unsupported[:3]
                )
                if len(unsupported) > 3:
                    preview += f", +{len(unsupported) - 3} more"
                print(
                    "Unsloth: Adafactor does not support rank>2 trainable "
                    "parameters in MLX; using AdamW instead "
                    f"({preview})."
                )
                opt_name = "adamw"

        if opt_name == "adafactor":
            optimizer = optim.Adafactor(
                learning_rate=initial_lr,
                relative_step=False,
                scale_parameter=False,
            )
        elif opt_name == "adamw":
            # Match HF/PyTorch AdamW semantics. MLX defaults bias_correction
            # to False, which makes early warmup updates much larger.
            self._manual_weight_decay = float(wd or 0.0)
            optimizer = optim.AdamW(
                learning_rate=initial_lr,
                weight_decay=0.0,
                bias_correction=True,
                **adam_kwargs,
            )
        elif opt_name == "adam":
            optimizer = optim.Adam(
                learning_rate=initial_lr,
                bias_correction=True,
                **adam_kwargs,
            )
        elif opt_name == "sgd":
            # HF/PyTorch SGD couples weight decay into the gradient (and thus
            # momentum/Nesterov), unlike AdamW's decoupled shrink. Apply our
            # own bias/norm-aware coupled decay so the exemption matches HF
            # while keeping SGD's coupled dynamics.
            self._coupled_weight_decay = float(wd or 0.0)
            optimizer = optim.SGD(learning_rate=initial_lr, weight_decay=0.0)
        elif opt_name == "muon":
            self._manual_weight_decay = float(wd or 0.0)
            optimizer = optim.Muon(learning_rate=initial_lr, weight_decay=0.0)
        elif opt_name == "lion":
            self._manual_weight_decay = float(wd or 0.0)
            optimizer = optim.Lion(learning_rate=initial_lr, weight_decay=0.0)
        self._resolved_optimizer_name = opt_name
        return optimizer

    @staticmethod
    def _should_apply_weight_decay(name, parameter=None):
        """HF-style AdamW decay filter: decay weights, skip bias and norms."""
        parts = [part.lower() for part in str(name).split(".") if part]
        leaf = parts[-1] if parts else str(name).lower()
        if leaf == "bias":
            return False
        # Cover RMSNorm/LayerNorm via "norm" + GPT-2 style ln_1/ln_2/ln_f.
        if any(_part_is_norm(part) for part in parts):
            return False
        return True

    @staticmethod
    def _is_norm_parameter_name(name):
        return any(
            _part_is_norm(part.lower())
            for part in str(name).split(".")
            if part
        )

    @staticmethod
    def _is_lora_parameter_name(name):
        return any(
            "lora" in part.lower()
            for part in str(name).split(".")
            if part
        )

    def _apply_manual_weight_decay(self, model, optimizer, grad):
        """Decoupled HF-parity decay on trainable non-bias/non-norm leaves.

        Active for AdamW, Muon, and Lion. The underlying MLX optimizer is
        constructed with ``weight_decay=0.0`` so this helper owns the full
        update for the weight-decay term and matches what HF Trainer does
        via ``param_groups``. SGD uses coupled decay instead (see
        ``_apply_coupled_weight_decay``).
        """
        wd = float(getattr(self, "_manual_weight_decay", 0.0) or 0.0)
        if wd <= 0:
            return

        flat_grad = dict(tree_flatten(grad))
        decayed = []
        for name, parameter in tree_flatten(model.trainable_parameters()):
            if name not in flat_grad:
                continue
            if not self._should_apply_weight_decay(name, parameter):
                continue
            if not mx.issubdtype(parameter.dtype, mx.floating):
                continue
            lr_value = optimizer.learning_rate
            if hasattr(lr_value, "astype"):
                lr = lr_value.astype(mx.float32)
            else:
                lr = mx.array(lr_value, dtype=mx.float32)
            scale = mx.array(1.0, dtype=mx.float32) - lr * mx.array(wd, dtype=mx.float32)
            decayed.append((name, (parameter.astype(mx.float32) * scale).astype(parameter.dtype)))
        if decayed:
            model.update(tree_unflatten(decayed))

    def _apply_coupled_weight_decay(self, model, grad):
        """Fold HF/PyTorch-SGD coupled decay (wd * param) into the gradient.

        SGD adds ``weight_decay * parameter`` to the gradient before the
        momentum/Nesterov update, so it must be applied to ``grad`` rather
        than as a post-update parameter shrink. Keeps HF's bias/norm
        exemption. Returns a possibly-modified grad tree; the original is
        returned unchanged when no decay applies.
        """
        wd = float(getattr(self, "_coupled_weight_decay", 0.0) or 0.0)
        if wd <= 0:
            return grad

        params = dict(tree_flatten(model.trainable_parameters()))
        wd_arr = mx.array(wd, dtype=mx.float32)
        updated = []
        changed = False
        for name, value in tree_flatten(grad):
            parameter = params.get(name)
            if (
                parameter is not None
                and self._should_apply_weight_decay(name, parameter)
                and mx.issubdtype(parameter.dtype, mx.floating)
            ):
                decayed = value + (parameter.astype(value.dtype) * wd_arr.astype(value.dtype))
                updated.append((name, decayed))
                changed = True
            else:
                updated.append((name, value))
        if not changed:
            return grad
        return tree_unflatten(updated)

    @staticmethod
    def _adafactor_unsupported_parameters(model):
        """Return trainable params MLX Adafactor cannot update safely.

        It treats ndim >= 2 as factored and reconstructs via matmul (correct
        for 2-D), but rank-3/4 tensors from vision patch embeddings, convs, and
        some projectors fail or broadcast incorrectly.
        """
        unsupported = []
        try:
            trainable = tree_flatten(model.trainable_parameters())
        except Exception:
            return unsupported

        for name, value in trainable:
            ndim = getattr(value, "ndim", None)
            if ndim is not None and ndim > 2:
                unsupported.append((name, tuple(getattr(value, "shape", ()))))
        return unsupported

    def _fire_prediction_step(self):
        """Dispatch HF's on_prediction_step for one processed eval batch.

        transformers fires this once per evaluation batch from its evaluation
        loop (`self.control = self.callback_handler.on_prediction_step(args,
        self.state, self.control)` in Trainer.evaluation_loop), which is how
        stock ProgressCallback advances its evaluation bar and how per-batch
        evaluation instrumentation is notified. The handler passes its
        eval_dataloader through, so a sized batch list gives ProgressCallback a
        real total and a lazy view (no __len__) makes it no-op, like HF's own
        has_length guard.

        No-ops when the handler is absent: _evaluate is reachable from helper
        paths that never built one, and evaluation must not start depending on
        the callback bridge to produce a loss.
        """
        handler = getattr(self, "callback_handler", None)
        if handler is None:
            return
        self.control = handler.call_event(
            "on_prediction_step", self.args, self.state, self.control,
        )

    def _close_split_prediction_bars(self):
        """End per-split evaluation progress bars when eval_dataloader rotates.

        HF scores one eval_dataset split per evaluate() call, so its per-split
        on_evaluate closes ProgressCallback's prediction bar before the next
        split opens a new one sized to that split. MLX reports the whole dict as
        one evaluation, so without this the first split's bar keeps counting
        every later split's batch past its own total (2/2 climbing to 9/2 across
        three splits). The last split's bar is still closed by on_evaluate, as
        in HF. Duck-typed to keep this module Torch-free, and a no-op on ranks
        and callbacks that never opened a bar.
        """
        for callback in getattr(
            getattr(self, "callback_handler", None), "callbacks", ()
        ) or ():
            bar = getattr(callback, "prediction_bar", None)
            if bar is None:
                continue
            close = getattr(bar, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass  # a display failure must not abort the evaluation
            try:
                callback.prediction_bar = None
            except AttributeError:
                pass  # read-only attribute; the close above already ended it

    def _evaluate_batch_totals(self, eval_batches, loss_fn, is_vlm=False):
        """Accumulate weighted loss totals for one flat eval batch stream."""
        all_losses = mx.array(0.0)
        ntokens = mx.array(0)
        # A stop requested before evaluation must abort before the first pull:
        # an unsized source's next row can block, so cancellation could
        # otherwise never take effect. Rank-synchronized so peers return
        # together instead of diverging at the in-loop status collective.
        should_stop, _ = self._distributed_eval_status()
        if should_stop:
            return all_losses, ntokens
        iterator = iter(eval_batches)

        while True:
            failed = False
            error = None
            try:
                batch_data = next(iterator)
            except StopIteration:
                break
            except BaseException as exc:
                # Interrupts included: every rank must reach the eval status
                # collective below or the survivors hang in it.
                failed = True
                error = exc

            if not failed and not self.stop_requested:
                try:
                    if is_vlm:
                        loss, ntoks = loss_fn(self.model, batch_data)
                    else:
                        batch, lengths, labels = batch_data
                        loss, ntoks = loss_fn(self.model, batch, lengths, labels)
                    # Zero-token eval batches (distributed_pad_mode="empty" padding
                    # rows) make loss NaN; mask them so NaN * 0 does not poison the
                    # distributed all-sum. mx.where never selects the NaN branch.
                    all_losses += mx.where(ntoks > 0, loss * ntoks, 0.0)
                    ntokens += ntoks
                    mx.eval(all_losses, ntokens)
                    # HF dispatches on_prediction_step after each evaluation
                    # batch is folded into the running totals. Raised inside
                    # this try on purpose: a callback that fails on one rank
                    # then joins the same _distributed_eval_status consensus as
                    # a failed batch, so the peers abort together instead of
                    # hanging at the next collective.
                    self._fire_prediction_step()
                except BaseException as exc:
                    failed = True
                    error = exc

            should_stop, failed_any = self._distributed_eval_status(failed)
            self._raise_distributed_failure_from_any(
                failed_any,
                "evaluation",
                error,
            )
            if should_stop:
                break

        return all_losses, ntokens

    def _create_text_eval_batches(
        self,
        eval_dataset,
        eval_batch_size,
        completion_only_loss,
        assistant_only_loss,
    ):
        """Build eager or one-pass lazy text evaluation batches."""
        args = self.args
        config = getattr(self.model, "_config", {})
        model_type = config.get("model_type") if isinstance(config, dict) else None
        eval_tokenizer = getattr(
            self, "_mlx_response_mask_tokenizer", self.tokenizer,
        )
        common = dict(
            dataset=eval_dataset,
            tokenizer=eval_tokenizer,
            batch_size=eval_batch_size,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
            dataset_text_field=args.dataset_text_field,
            formatting_func=self.formatting_func,
            chat_template=getattr(args, "chat_template", None),
            model_name=getattr(self.model, "_hf_repo", None),
            model_type=model_type,
            append_eos=bool(getattr(args, "append_eos", True)),
            completion_only_loss=completion_only_loss,
            assistant_only_loss=assistant_only_loss,
        )
        if args.streaming and _is_mlx_lazy_text_source(eval_dataset):
            max_batches = getattr(args, "max_eval_batches", None)
            if max_batches is not None:
                # Reject rather than truncate: silent coercion would change how
                # much eval data is scored.
                coerced = None
                if not isinstance(max_batches, bool):
                    try:
                        coerced = int(max_batches)
                    except (TypeError, ValueError):
                        coerced = None
                if coerced is None or coerced != max_batches or coerced <= 0:
                    raise ValueError(
                        "Unsloth MLX: max_eval_batches must be a positive "
                        "integer when provided."
                    )
                max_batches = coerced

            def _factory():
                return iterate_training_batches(
                    **common,
                    response_mask_fn=getattr(self, "_mlx_response_mask_fn", None),
                    dataset_order="sequential",
                    comm_group=self.distributed_world,
                    require_replayable=True,
                    repeat=False,
                    distributed_pad_mode="empty",
                )

            return _MLXLazyEvalBatchView(
                eval_dataset,
                _factory,
                max_batches=max_batches,
                comm_group=self.distributed_world,
            )
        return create_batches(
            **common,
            comm_group=self.distributed_world,
            distributed_pad_mode="empty",
        )

    def _evaluate(self, eval_batches, loss_fn, is_vlm=False):
        """Run evaluation loop.

        Returns:
            (avg_loss, perplexity) tuple.
        """
        self.model.eval()
        metrics = {}
        if isinstance(eval_batches, dict):
            all_losses = mx.array(0.0)
            ntokens = mx.array(0)
            # HF evaluates one split at a time and rebuilds its eval_dataloader
            # per split, so on_prediction_step reports the split being consumed
            # rather than the dict of splits (whose len is the split count, and
            # would give ProgressCallback a nonsense bar total).
            handler = getattr(self, "callback_handler", None)
            outer_dataloader = getattr(handler, "eval_dataloader", None)
            try:
                for split_index, (split_name, split_batches) in enumerate(
                    eval_batches.items()
                ):
                    if handler is not None:
                        if split_index:
                            self._close_split_prediction_bars()
                        handler.eval_dataloader = split_batches
                    split_losses, split_tokens = self._evaluate_batch_totals(
                        split_batches, loss_fn, is_vlm=is_vlm,
                    )
                    split_losses = self._distributed_all_sum(split_losses, stream=mx.cpu)
                    split_tokens = self._distributed_all_sum(split_tokens, stream=mx.cpu)
                    all_losses += split_losses
                    ntokens += split_tokens
                    mx.eval(all_losses, ntokens)
                    split_loss = (
                        (split_losses / split_tokens).item()
                        if split_tokens.item() > 0 else 0.0
                    )
                    split_ppl = math.exp(min(split_loss, 100))
                    split_prefix = f"eval_{split_name}"
                    metrics[f"{split_prefix}_loss"] = split_loss
                    metrics[f"{split_prefix}_perplexity"] = split_ppl
                    if self._distributed_should_stop():
                        break
            finally:
                if handler is not None:
                    handler.eval_dataloader = outer_dataloader
        else:
            all_losses, ntokens = self._evaluate_batch_totals(
                eval_batches, loss_fn, is_vlm=is_vlm,
            )
            all_losses = self._distributed_all_sum(all_losses, stream=mx.cpu)
            ntokens = self._distributed_all_sum(ntokens, stream=mx.cpu)

        self.model.train()
        avg_loss = (all_losses / ntokens).item() if ntokens.item() > 0 else 0.0
        perplexity = math.exp(min(avg_loss, 100))
        metrics["eval_loss"] = avg_loss
        metrics["eval_perplexity"] = perplexity
        self._last_eval_metrics = metrics
        return avg_loss, perplexity

    @staticmethod
    def _bytes_to_gb(value):
        """Convert a byte count to decimal GB for user-facing memory logs."""
        try:
            return float(value) / 1e9
        except Exception:
            return None

    def _configure_memory_limits(self):
        """Apply conservative Metal memory caps so failed runs exit cleanly.

        Defaults to ~85% of Apple's recommended working-set size to avoid
        paging/kernel-panic on large multimodal runs. Disable shortcuts:
          - args.disable_memory_limits=True  ─► skip every cap (memory, cache, wired)
          - args.memory_limit_gb <= 0        ─► skip memory_limit AND wired_limit
          - args.wired_limit_gb  <= 0        ─► skip wired_limit only
          - args.cache_limit_gb  <= 0        ─► skip cache_limit only
        """
        if not mx.metal.is_available():
            return {}

        args = self.args
        if getattr(args, "disable_memory_limits", False):
            return {}

        info = mx.device_info()
        recommended_gb = self._bytes_to_gb(
            info.get("max_recommended_working_set_size")
        )
        if recommended_gb is None or recommended_gb <= 0:
            return {}

        configured = {}
        # Prior values are restored after training; the cap is process-global.
        self._prior_metal_limits = {}

        # memory_limit_gb: None → 85% of recommended; <= 0 → disable BOTH this
        # and the wired cap (wired default is min(recommended, memory_limit)).
        memory_limit_gb = getattr(args, "memory_limit_gb", None)
        memory_disabled = memory_limit_gb is not None and memory_limit_gb <= 0
        if memory_limit_gb is None:
            memory_limit_gb = recommended_gb * 0.85
        elif memory_disabled:
            memory_limit_gb = None
        if memory_limit_gb is not None:
            prev = mx.set_memory_limit(int(memory_limit_gb * 1e9))
            self._prior_metal_limits["memory"] = prev
            configured["memory_limit_gb"] = float(memory_limit_gb)

        cache_limit_gb = getattr(args, "cache_limit_gb", None)
        if cache_limit_gb is not None and cache_limit_gb > 0:
            prev = mx.set_cache_limit(int(cache_limit_gb * 1e9))
            self._prior_metal_limits["cache"] = prev
            configured["cache_limit_gb"] = float(cache_limit_gb)

        wired_limit_gb = getattr(args, "wired_limit_gb", None)
        if wired_limit_gb is None:
            # Inherit "disabled" from memory_limit so memory_limit_gb=-1
            # disables wired too.
            if memory_disabled:
                wired_limit_gb = None
            else:
                wired_limit_gb = min(
                    recommended_gb,
                    configured.get("memory_limit_gb", recommended_gb),
                )
        elif wired_limit_gb <= 0:
            wired_limit_gb = None
        if wired_limit_gb is not None:
            prev = mx.set_wired_limit(int(wired_limit_gb * 1e9))
            self._prior_metal_limits["wired"] = prev
            configured["wired_limit_gb"] = float(wired_limit_gb)

        configured["recommended_working_set_gb"] = float(recommended_gb)
        return configured

    def _restore_memory_limits(self):
        prior = getattr(self, "_prior_metal_limits", None)
        if not prior or not mx.metal.is_available():
            return
        try:
            if "memory" in prior and prior["memory"] is not None:
                mx.set_memory_limit(int(prior["memory"]))
            if "cache" in prior and prior["cache"] is not None:
                mx.set_cache_limit(int(prior["cache"]))
            if "wired" in prior and prior["wired"] is not None:
                mx.set_wired_limit(int(prior["wired"]))
        except Exception:
            pass
        self._prior_metal_limits = {}

    def _setup_report_to_callbacks(self):
        """Auto-register W&B / TensorBoard callbacks from report_to, mirroring
        Unsloth worker.py log keys so notebook and Unsloth runs chart identically."""
        raw = getattr(self.args, "report_to", "none")
        if not raw or raw == "none":
            return
        targets = raw if isinstance(raw, (list, tuple)) else [raw]
        targets = {str(t).lower() for t in targets}
        # "all" mirrors HF: enable every backend we support on MLX.
        if "all" in targets:
            targets |= {"wandb", "tensorboard"}
        unsupported = targets - {"wandb", "tensorboard", "all", "none"}
        if unsupported:
            print(f"Unsloth: report_to target(s) {sorted(unsupported)} are not "
                  f"supported on MLX; only 'wandb' and 'tensorboard' are logged.")

        wandb_run = None
        if "wandb" in targets:
            try:
                import wandb
                wandb_run = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "unsloth-mlx"),
                    config={k: v for k, v in vars(self.args).items()
                            if not k.startswith("_")},
                )
            except Exception as e:
                print(f"Unsloth: wandb init failed: {e}")
                wandb_run = None

        tb_writer = None
        if "tensorboard" in targets:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter
                except ImportError:
                    SummaryWriter = None
            if SummaryWriter is not None:
                try:
                    tb_writer = SummaryWriter(
                        log_dir=os.path.join(self.args.output_dir, "runs"))
                except Exception as e:
                    print(f"Unsloth: tensorboard init failed: {e}")
                    tb_writer = None

        if wandb_run is None and tb_writer is None:
            return

        def _on_step(step, total_steps, loss, lr, tokens_sec, peak_gb,
                     elapsed, num_tokens, grad_norm=None):
            if wandb_run is not None:
                try:
                    wandb_run.log({
                        "train/loss": loss,
                        "train/learning_rate": lr,
                        "train/tokens_per_sec": tokens_sec,
                        "train/peak_gb": peak_gb,
                        "train/num_tokens": num_tokens,
                        **({"train/grad_norm": grad_norm} if grad_norm is not None else {}),
                    }, step=step)
                except Exception:
                    pass
            if tb_writer is not None:
                try:
                    tb_writer.add_scalar("train/loss", loss, step)
                    tb_writer.add_scalar("train/learning_rate", lr, step)
                    tb_writer.add_scalar("train/tokens_per_sec", tokens_sec, step)
                    tb_writer.add_scalar("train/peak_gb", peak_gb, step)
                    if grad_norm is not None:
                        tb_writer.add_scalar("train/grad_norm", grad_norm, step)
                except Exception:
                    pass

        def _on_eval(step, eval_loss, perplexity):
            if wandb_run is not None:
                try:
                    wandb_run.log({"eval/loss": eval_loss,
                                   "eval/perplexity": perplexity}, step=step)
                except Exception:
                    pass
            if tb_writer is not None:
                try:
                    tb_writer.add_scalar("eval/loss", eval_loss, step)
                    tb_writer.add_scalar("eval/perplexity", perplexity, step)
                except Exception:
                    pass

        self.add_step_callback(_on_step)
        self.add_eval_callback(_on_eval)
        self._report_to_handles = (wandb_run, tb_writer)
        self._report_to_callbacks = (_on_step, _on_eval)

    def _install_neftune(self):
        """NEFTune: add scaled uniform noise to input embeddings during training.
        Text models only; no-op in eval. Uses __class__ reassignment (a real
        subclass) rather than a module swap, so the embedding object is
        unchanged -- .weight stays readable for tied LM-head models, and
        __call__ resolves on the subtype so interception actually fires."""
        alpha = float(getattr(self.args, "neftune_noise_alpha", 0.0) or 0.0)
        # Reject non-finite alpha: nan slips past `alpha <= 0` and would poison
        # every embedding with nan/inf noise from step 0.
        if not math.isfinite(alpha) or alpha <= 0:
            return
        if self._is_vlm:
            print("Unsloth: NEFTune (neftune_noise_alpha) is not yet supported "
                  "for VLM models on MLX; ignoring.")
            return
        try:
            tm = _get_text_model(self.model)
            backbone = getattr(tm, "model", tm)
            emb = backbone.embed_tokens
        except Exception as e:
            print(f"Unsloth: NEFTune could not locate embed_tokens ({e}); ignoring.")
            return
        if getattr(emb, "_unsloth_neftune_active", False):
            return

        _Base = type(emb)
        _alpha = alpha

        class _NEFTuneEmbed(_Base):
            _unsloth_neftune_active = True
            def __call__(self, x):
                out = _Base.__call__(self, x)
                if getattr(self, "training", False):
                    dim = out.shape[-1] * out.shape[-2]
                    scale = _alpha / (dim ** 0.5)
                    noise = mx.random.uniform(
                        low=-1.0, high=1.0, shape=out.shape
                    ).astype(out.dtype) * scale
                    return out + noise
                return out

        # Report the base class's name so the save-window DoRA detection
        # (`type(module).__name__.startswith("DoRA")` in mlx/utils.py) sees
        # through this transparent stand-in. An embedding-only DoRA adapter
        # (use_dora=True targets embed_tokens) is what NEFTune subclasses here,
        # so a bare "_NEFTuneEmbed" name would fail that check and silently
        # demote the DoRA adapter to plain LoRA on save. The quantization-map
        # scan is safe without this: it already keys on isinstance, not name.
        _NEFTuneEmbed.__name__ = _Base.__name__
        _NEFTuneEmbed.__qualname__ = getattr(_Base, "__qualname__", _Base.__name__)

        self._neftune_emb = emb
        self._neftune_base_cls = _Base
        emb.__class__ = _NEFTuneEmbed
        print(f"Unsloth: NEFTune enabled (noise_alpha={alpha}).")

    def _remove_neftune(self):
        emb = getattr(self, "_neftune_emb", None)
        base = getattr(self, "_neftune_base_cls", None)
        if emb is not None and base is not None:
            try:
                emb.__class__ = base
            except Exception:
                pass
        self._neftune_emb = None
        self._neftune_base_cls = None

    def _close_active_batch_iterator(self):
        """Best-effort release of an iterator owned by the training run."""
        batch_iter = getattr(self, "_active_batch_iter", None)
        self._active_batch_iter = None
        # Every exit path lands here: close the producer and persist a live
        # orphan so the next run's gate sees it.
        control = getattr(self, "_mlx_prefetch_control", None)
        prefetcher = control.get("prefetcher") if control else None
        try:
            close = getattr(batch_iter, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    # A cleanup error must not mask an in-flight training error
                    # or diverge ranks after a final collective. A propagating
                    # signal is still honored (see finally).
                    pass
            if prefetcher is not None:
                try:
                    prefetcher.close()
                except Exception:
                    pass
        finally:
            if prefetcher is not None:
                # close() marks a conservative orphan up front, so an unclean
                # close or a propagating interrupt leaves orphaned=True. Persist
                # it either way; the next run's gate drains and clears it.
                if prefetcher.orphaned:
                    self._mlx_prefetch_orphan = prefetcher
                if control is not None:
                    control["prefetcher"] = None


    def _quiesce_prefetcher_for_save(self, terminal=False):
        """Make serialization exclusive over shared preprocessing objects.

        ``terminal=True`` closes the producer for good; otherwise an active
        producer is PAUSED and returned so the caller can resume it after the
        save. Live persisted orphans always refuse.
        """
        orphan = getattr(self, "_mlx_prefetch_orphan", None)
        if orphan is not None:
            if orphan.orphan_alive():
                raise RuntimeError(
                    "Unsloth MLX: a prefetch producer is still blocked inside "
                    "its source and shares this trainer's tokenizer; refusing "
                    "to serialize concurrently. Wait for the thread to "
                    "terminate, then call save_model() again."
                )
            # Terminated orphan: close() drains its queued device tensors
            # before this save instead of waiting on the reference drop.
            orphan.close()
            self._mlx_prefetch_orphan = None
        control = getattr(self, "_mlx_prefetch_control", None)
        prefetcher = control.get("prefetcher") if control else None
        if prefetcher is None:
            return None
        if not terminal:
            prefetcher.quiesce()
            return prefetcher
        # Gate on close()'s return, not a fresh orphan_alive() read: it drained
        # the queue iff it reports clean, so a save can never proceed with
        # staged device tensors still queued.
        if not prefetcher.close():
            self._mlx_prefetch_orphan = prefetcher
            control["prefetcher"] = None
            raise RuntimeError(
                "Unsloth MLX: the prefetch producer is blocked inside its "
                "source and shares this trainer's tokenizer; refusing to "
                "serialize concurrently. Wait for the thread to terminate, "
                "then call save_model() again."
            )
        # Terminal close succeeded: drop the control reference so the ensuing
        # save_model() wrapper gate is a no-op.
        control["prefetcher"] = None
        return None

    def train(self, resume_from_checkpoint: str | None = None):
        """Run MLX-native training loop following mlx-lm's compiled-step pattern
        with gradient accumulation. Returns a dict of training metrics."""
        # Clear a PRIOR run's stale stop at the EARLIEST point, before any setup,
        # so a reused trainer starts un-stopped. Only a stop stamped by an EARLIER
        # generation is cleared, so a cancel raised DURING this run's setup (a
        # Studio cancel while _prepare_data materializes) survives to the loop's
        # _distributed_should_stop(). Local assignment only, no DDP collective.
        if self._stop_request_generation() < getattr(self, "_run_generation", 0):
            self.stop_requested = False
        # Then release the previous run's iterator/prefetch producer. Ordered
        # after the stop clear because that clear is a local assignment that
        # cannot fail, while this can block or propagate an interrupt; running it
        # first would leave a stale stop latched for the next train() call.
        self._close_active_batch_iterator()
        # Stash for _train_inner. None = fresh start, a path = resume.
        self._resume_from_checkpoint = resume_from_checkpoint
        self._ensure_distributed()
        args = self.args
        model = self.model
        self._text_shape_guard_preflight = None
        if (
            hasattr(self, "_batches")
            and not getattr(self, "_is_vlm", False)
            and not (
                getattr(self.args, "streaming", False)
                and self._batches is None
            )
        ):
            preflight_error = None
            try:
                if self._batches is None:
                    self._prepared_batches_include_epochs = False
                batches, batch_iter = self._prepare_data(False)
                total_steps = _resolve_training_steps(
                    args,
                    batches,
                    batch_iter,
                    includes_epochs=getattr(
                        self, "_prepared_batches_include_epochs", False,
                    ),
                )
                compile_policy = build_compile_policy(args=args)
            except BaseException as exc:
                preflight_error = exc
            if self.distributed_world_size > 1:
                self._raise_distributed_failure(
                    preflight_error is not None,
                    "preparing finite text shape guard",
                    preflight_error,
                )
            elif preflight_error is not None:
                raise preflight_error
            local_plan_error = None
            shape_plan = None
            frontier = None
            try:
                (
                    shape_plan,
                    report,
                    compile_allowed,
                    frontier,
                ) = _plan_single_process_text_shapes(
                    batches,
                    batch_iter,
                    args=args,
                    total_steps=total_steps,
                    is_vlm=False,
                    distributed_world_size=self.distributed_world_size,
                    compile_policy=compile_policy,
                    install_plan=False,
                    includes_epochs=getattr(
                        self, "_prepared_batches_include_epochs", False,
                    ),
                )
            except Exception as exc:
                local_plan_error = exc
                report = _shape_guard_report(
                    "eager",
                    "planner_error",
                    resolve_compile_max_variants(args.compile_max_variants),
                    (
                        DDP_LOCAL_GRAD_SCOPE
                        if self.distributed_world_size > 1
                        else FULL_STEP_SCOPE
                    ),
                    cap_selection="not_applicable",
                )
                compile_allowed = False
            (
                shape_plan,
                report,
                compile_allowed,
            ) = self._coordinate_text_shape_guard(
                shape_plan,
                frontier,
                report,
                compile_allowed,
                compile_policy,
                automatic=args.compile_max_variants is None,
                local_error=local_plan_error,
            )
            if compile_allowed and shape_plan is not None:
                batches.set_shape_plan(shape_plan)
            self._text_shape_guard_preflight = (
                batches, batch_iter, total_steps, report, compile_allowed,
            )

        self._install_neftune()
        is_main_process = self.is_main_process

        def _main_print(*print_args, **print_kwargs):
            if is_main_process:
                print(*print_args, **print_kwargs)

        cast_norm_output = bool(getattr(args, "cast_norm_output_to_input_dtype", True))
        _prev_norm_output_cast_state = snapshot_mlx_norm_output_cast_state(
            iter_mlx_norm_output_cast_classes(model)
        )
        # Save Qwen3-VL vision-block flag so finally restores it (not just False).
        _prev_qwen3_vision_cast = True
        try:
            from . import compile as _mlx_compile
            _prev_qwen3_vision_cast = bool(
                getattr(_mlx_compile, "_QWEN3_VISION_NORM_CAST_OUTPUT", True)
            )
        except Exception:
            pass
        # Patch INSIDE try/finally so any raise during setup still restores globals.
        try:
            from .loader import _keep_norm_parameters_float32
            _keep_norm_parameters_float32(model)
            _set_norm_output_cast_to_input_dtype(cast_norm_output, model)
            if cast_norm_output:
                _main_print("Unsloth: Casting MLX norm outputs back to activation dtype.")
            args.patch_mode = normalize_mlx_patch_mode(getattr(args, "patch_mode", "patched"))
            model._unsloth_patch_mode = args.patch_mode

            self._memory_limits_applied = self._configure_memory_limits()

            self._compile_decision = None
            self._compile_trace = None
            self._compile_auto_tune_applied = []
            if self._is_vlm and (args.compile or args.compile_trace):
                compile_policy = build_compile_policy(args=args)
                qual = getattr(model, "_unsloth_compile_qualification", None) or get_compile_qualification(model)
                if qual is not None:
                    model._unsloth_compile_qualification = qual
                self._compile_decision = resolve_training_compile(model, policy=compile_policy, args=args)
                model._unsloth_compile_decision = self._compile_decision
                if args.compile_trace:
                    self._compile_trace = trace_compile_application(model, policy=compile_policy, args=args)
                    model._unsloth_compile_trace = self._compile_trace
                    model._unsloth_compile_explain = explain_compile_support(model, policy=compile_policy, args=args)
                if args.compile_auto_tune:
                    self._compile_auto_tune_applied = self._apply_compile_recommendations(
                        args, self._compile_decision
                    )
                    for setting, value, reason in self._compile_auto_tune_applied:
                        _main_print(
                            f"Unsloth: Auto-tuned {setting}={value!r} for MLX compile "
                            f"({reason})"
                        )

            # Coordinated VLM shape-guard preflight. Runs AFTER compile
            # qualification (the survey materializes every batch) and after
            # auto-tuning (planning reads the tuned args), and BEFORE optimizer
            # or compiled-callable setup, so every rank agrees on failure, mode
            # and the shared cap while the run is still trivial to abort. The
            # setup in between is rank-deterministic and needs no collectives.
            # A strict abort here unwinds through train()'s finally, and the
            # state that persists is idempotent if train() is called again.
            if self._is_vlm and hasattr(self, "_batches"):
                preflight_error = None
                batches = batch_iter = None
                total_steps = 0
                compile_policy = build_compile_policy(args=args)
                self._deferred_vlm_all_masked_check = None
                try:
                    if self._batches is None:
                        self._prepared_batches_include_epochs = False
                    # Deferred checker: preparation stays collective-free, so the
                    # status reduction below is the FIRST collective on every rank
                    # and the checker's all-reduce runs strictly after it.
                    batches, batch_iter = self._prepare_data(
                        True, defer_vlm_checker=True,
                    )
                    total_steps = _resolve_training_steps(
                        args,
                        batches,
                        batch_iter,
                        includes_epochs=getattr(
                            self, "_prepared_batches_include_epochs", False,
                        ),
                    )
                except BaseException as exc:
                    # why: an interrupt on one rank must join the consensus
                    # below, or its peers block in a collective it never
                    # reaches.
                    preflight_error = exc
                if self.distributed_world_size > 1:
                    self._raise_distributed_failure(
                        preflight_error is not None,
                        "preparing finite VLM shape guard",
                        preflight_error,
                    )
                elif preflight_error is not None:
                    raise preflight_error
                deferred_check = self._deferred_vlm_all_masked_check
                self._deferred_vlm_all_masked_check = None
                if deferred_check is not None:
                    # Global counts, so an all-masked dataset raises symmetrically.
                    deferred_check()
                local_plan_error = None
                shape_plan = None
                frontier = None
                try:
                    (
                        shape_plan,
                        report,
                        compile_allowed,
                        frontier,
                    ) = _plan_single_process_vlm_shapes(
                        batches,
                        batch_iter,
                        args=args,
                        total_steps=total_steps,
                        distributed_world_size=self.distributed_world_size,
                        compile_policy=compile_policy,
                        compile_decision=self._compile_decision,
                        install_plan=False,
                    )
                except Exception as exc:
                    local_plan_error = exc
                    report = _shape_guard_report(
                        "eager",
                        "planner_error",
                        resolve_compile_max_variants(args.compile_max_variants),
                        (
                            DDP_LOCAL_GRAD_SCOPE
                            if self.distributed_world_size > 1
                            else FULL_STEP_SCOPE
                        ),
                        cap_selection="not_applicable",
                    )
                    compile_allowed = False
                # Synchronize the planning MODE before coordination: mixed
                # planning and benign non-planning ranks would run mismatched
                # collectives and diverge on compile eligibility later. Three
                # fixed reductions run on every rank; a genuinely MIXED state
                # discards plans and disables compile everywhere so downstream
                # participation stays symmetric. Uniform benign states are
                # args-derived, hence rank-identical, and keep legacy behavior.
                benign_not_planning = (
                    local_plan_error is None
                    and compile_allowed
                    and shape_plan is None
                )
                planning_locally = (
                    local_plan_error is None
                    and compile_allowed
                    and shape_plan is not None
                )
                decision_eligible = bool(
                    self._compile_decision is not None
                    and getattr(self._compile_decision, "enabled", False)
                )
                if self.distributed_world_size > 1:
                    # Three fixed reductions on every rank, always in this order.
                    # Eligibility divergence can hide inside ANY uniform
                    # applicability class, so it is synchronized directly: a mixed
                    # group disables compile everywhere and keeps later
                    # compiled-setup collectives symmetric.
                    any_ineligible = self._distributed_any_flag(
                        not decision_eligible,
                    )
                    any_benign = self._distributed_any_flag(
                        benign_not_planning,
                    )
                    any_planning = self._distributed_any_flag(
                        planning_locally,
                    )
                    mixed_eligibility = any_ineligible and decision_eligible
                    peer_split = any_benign and any_planning
                    if mixed_eligibility or peer_split:
                        if planning_locally:
                            report = _shape_guard_report(
                                "not_applicable",
                                "vlm_peer_not_planning",
                                resolve_compile_max_variants(
                                    args.compile_max_variants,
                                ),
                                lazy_batches=isinstance(
                                    batches, FiniteVLMBatchPlan,
                                ),
                            )
                            shape_plan = None
                            frontier = None
                        compile_allowed = False
                (
                    shape_plan,
                    report,
                    compile_allowed,
                ) = self._coordinate_text_shape_guard(
                    shape_plan,
                    frontier,
                    report,
                    compile_allowed,
                    compile_policy,
                    automatic=args.compile_max_variants is None,
                    local_error=local_plan_error,
                    keep_exact_local=True,
                    compile_mode=_effective_compile_mode(
                        compile_policy, self._compile_decision,
                    ),
                )
                # A should_raise decision must abort EVERY rank, not just the one
                # holding the error, or a lone exit strands peers at the next
                # training collective. One fixed reduction after coordination
                # keeps the schedule pairable in all states.
                decision_abort = isinstance(
                    local_plan_error, _VLMCompileDecisionError,
                )
                if self.distributed_world_size > 1:
                    any_decision_abort = self._distributed_any_flag(
                        decision_abort,
                    )
                else:
                    any_decision_abort = decision_abort
                if any_decision_abort:
                    if decision_abort:
                        raise local_plan_error
                    raise RuntimeError(
                        "Unsloth: a peer DDP rank's compile decision "
                        "mandates an abort for this VLM run."
                    )
                if (
                    compile_allowed
                    and shape_plan is not None
                    and isinstance(batches, FiniteVLMBatchPlan)
                ):
                    batches.set_shape_plan(
                        shape_plan, batches.planned_event_widths(),
                    )
                self._text_shape_guard_preflight = (
                    batches, batch_iter, total_steps, report, compile_allowed,
                )

            # (memory limits already applied above; just log what we configured)
            if self._memory_limits_applied:
                parts = []
                if "memory_limit_gb" in self._memory_limits_applied:
                    parts.append(
                        f"memory_limit={self._memory_limits_applied['memory_limit_gb']:.2f} GB"
                    )
                if "cache_limit_gb" in self._memory_limits_applied:
                    parts.append(
                        f"cache_limit={self._memory_limits_applied['cache_limit_gb']:.2f} GB"
                    )
                if "wired_limit_gb" in self._memory_limits_applied:
                    parts.append(
                        f"wired_limit={self._memory_limits_applied['wired_limit_gb']:.2f} GB"
                    )
                _main_print(
                    "Unsloth: MLX Metal memory guard enabled "
                    f"({', '.join(parts)})."
                )

            # Apply gradient checkpointing if requested
            if args.gradient_checkpointing:
                apply_gradient_checkpointing(model)
                _main_print("Unsloth: Using gradient checkpointing to reduce memory.")

            # Qwen3.5-specific fixes
            config = getattr(model, "_config", {})
            model_type = config.get("model_type", "") if isinstance(config, dict) else ""
            gated_delta_patched = False
            if "qwen3_5" in model_type:
                from .loader import _fix_qwen35_attention_cache, _disable_fused_mrope
                _fix_qwen35_attention_cache(model)
                _disable_fused_mrope(model)
                from ..gated_delta_vjp import patch_gated_delta, patch_gated_delta_vlm
                patch_gated_delta()
                patch_gated_delta_vlm()
                gated_delta_patched = True
            # Structural check: qwen3_next / kimi_linear also need the VJP.
            if not gated_delta_patched and model_has_gated_delta_layers(model):
                from ..gated_delta_vjp import patch_gated_delta
                patch_gated_delta()
            # Qwen2/2.5/3-VL language towers share the fused MRoPE kernel with
            # no VJP; flip it off so training takes the differentiable fallback.
            if any(t in model_type for t in ("qwen3_vl", "qwen2_vl", "qwen2_5_vl")):
                from .loader import _disable_fused_mrope
                _disable_fused_mrope(model)

            # Register W&B/TensorBoard reporters after arg auto-tuning so the
            # W&B config snapshot reflects the settings actually used (e.g. VLM
            # compile auto-tune can flip gradient_checkpointing before training).
            # Only rank 0 opens W&B / TensorBoard so DDP runs don't double-log.
            if self.is_main_process:
                self._setup_report_to_callbacks()
            return self._train_inner()
        finally:
            self._close_active_batch_iterator()
            _handles = getattr(self, "_report_to_handles", (None, None))
            _wb, _tb = _handles
            if _tb is not None:
                try: _tb.close()
                except Exception: pass
            if _wb is not None:
                try: _wb.finish()
                except Exception: pass
            for _cb in getattr(self, "_report_to_callbacks", ()):
                if _cb in self._step_callbacks: self._step_callbacks.remove(_cb)
                if _cb in self._eval_callbacks: self._eval_callbacks.remove(_cb)
            self._report_to_handles = (None, None)
            self._report_to_callbacks = ()
            self._remove_neftune()
            if args.gradient_checkpointing:
                try:
                    remove_gradient_checkpointing(model)
                except Exception:
                    pass
            try:
                self._restore_memory_limits()
            except Exception:
                pass
            # Restore the pre-run process-global norm patch state, even if setup failed mid-patch.
            try:
                restore_mlx_norm_output_cast_state(_prev_norm_output_cast_state)
            except Exception:
                pass
            # Restore Qwen3-VL vision-block flag to its pre-train value.
            try:
                from . import compile as _mlx_compile
                _mlx_compile.set_qwen3_vision_norm_cast_output(
                    _prev_qwen3_vision_cast
                )
            except Exception:
                pass
            self._text_shape_guard_preflight = None
            # Close this run's generation: a stop still latched from it is now
            # stale, while any later request belongs to the next run.
            self._run_generation = getattr(self, "_run_generation", 0) + 1

    def _train_inner(self):
        """Inner training loop, separated for GC cleanup in finally block."""
        args = self.args
        model = self.model
        is_vlm = self._is_vlm
        distributed_world_size = self.distributed_world_size
        is_main_process = self.is_main_process

        def _main_print(*print_args, **print_kwargs):
            if is_main_process:
                print(*print_args, **print_kwargs)

        # Pick loss function (returns (loss, ntoks)). Validate configuration
        # before any model-derived loss selection (config errors raise; model
        # properties fall back).
        use_cce = args.use_cce
        label_smoothing = _validate_label_smoothing(
            getattr(args, "label_smoothing_factor", 0.0), is_vlm,
        )
        _vlm_ignore_token_ids = None

        if is_vlm:
            processor = self._resolve_vlm_processor()
            # Backstop only; VLM collation already owns label masking.
            _vlm_ignore_token_ids = _get_vlm_ignore_token_ids(
                processor=processor,
                config=getattr(model, "_config", {}),
            )
            _atid = args.assistant_token_id if args.train_on_completions else 0
            if use_cce:
                loss_fn = make_vlm_cce_loss_fn(
                    model,
                    assistant_token_id=_atid,
                    ignore_token_ids=_vlm_ignore_token_ids,
                )
                cce_backend = getattr(loss_fn, "_unsloth_cce_backend", "unknown")
                if cce_backend == "baseline-fallback":
                    use_cce = False
                    _main_print(
                        "Unsloth: VLM CCE is unavailable for this model; using "
                        "standard cross-entropy loss.")
                else:
                    _main_print(
                        f"Unsloth: Using VLM CCE loss ({cce_backend}) "
                        "for memory-efficient training."
                    )
            else:
                loss_fn = make_vlm_baseline_loss_fn(
                    model,
                    assistant_token_id=_atid,
                    ignore_token_ids=_vlm_ignore_token_ids,
                )
                _main_print("Unsloth: Using VLM standard cross-entropy loss.")
        else:
            if use_cce:
                loss_fn = make_cce_loss_fn(model, label_smoothing=label_smoothing)
                cce_backend = getattr(loss_fn, "_unsloth_cce_backend", "unknown")
                if cce_backend == "baseline-fallback":
                    use_cce = False
                    # The factory already printed the specific reason (topology,
                    # head eligibility, or logit transform); keep this generic.
                    _main_print(
                        "Unsloth: fused CCE is unavailable for this model; "
                        "using standard cross-entropy loss.")
                else:
                    _main_print(
                        f"Unsloth: Using CCE loss ({cce_backend}) "
                        "for memory-efficient training."
                    )
            else:
                loss_fn = make_baseline_loss_fn(label_smoothing=label_smoothing)
                _main_print("Unsloth: Using standard cross-entropy loss.")

        # Prepare data and total_steps first. Keep any prebuilt flag from
        # train_on_responses_only: _prepare_data returns self._batches early
        # and never re-derives it for the completion-only text path.
        previous_orphan = getattr(self, "_mlx_prefetch_orphan", None)
        if previous_orphan is not None:
            if previous_orphan.orphan_alive():
                raise RuntimeError(
                    "Unsloth MLX: a previous prefetch producer is still "
                    "blocked inside its source and shares this trainer's "
                    "preprocessing objects. Wait for that thread to terminate "
                    "before training with this trainer again."
                )
            # Terminated orphan: close() drains its queued batches before the
            # reference is dropped.
            previous_orphan.close()
            self._mlx_prefetch_orphan = None
        self._mlx_resume_step_for_prefetch = 0
        self._mlx_resume_state_cache = None
        _wants_prefetch = bool(
            _validate_streaming_prefetch(
                getattr(self.args, "streaming_prefetch_batches", 0)
            )
            and self.distributed_world_size == 1
            and getattr(self.args, "streaming", False)
        )
        if _wants_prefetch and getattr(self, "_resume_from_checkpoint", None):
            # Single authority: this validated load feeds the producer skip now
            # and the scalar restoration later. World size is 1 here, so no
            # duplicated collectives. Failures stay hard.
            _early_resume = self._validate_distributed_resume_checkpoint(
                self._resume_from_checkpoint
            )
            if _early_resume:
                # Same completeness gate as the main resume block below, which
                # this early read would otherwise pre-empt with a raw
                # FileNotFoundError for trainer_state.json.
                _require_complete_resume_checkpoint(_early_resume)
                _early_state = load_trainer_state(_early_resume)
                self._mlx_resume_state_cache = (_early_resume, _early_state)
                self._mlx_resume_step_for_prefetch = int(
                    _early_state.get("global_step", 0)
                )
        compile_policy = build_compile_policy(args=args)
        preflight = getattr(self, "_text_shape_guard_preflight", None)
        if preflight is not None:
            (
                batches,
                batch_iter,
                total_steps,
                _compile_shape_guard_report,
                _shape_guard_compile_allowed,
            ) = preflight
        else:
            # Keep prebuilt completion/assistant batches and epoch metadata.
            if self._batches is None:
                self._prepared_batches_include_epochs = False
            batches, batch_iter = self._prepare_data(is_vlm)
            _stream_epochs = getattr(self, "_streaming_epoch_batch_count", None)
            if (
                batches is None
                and _stream_epochs is not None
                and args.max_steps <= 0
                and args.num_train_epochs > 0
            ):
                # Declared-length streaming epochs: total micro-batches come
                # from the per-pass trainable-row count. The finite-plan
                # resolver rejects streaming, so resolve it here.
                total_steps = max(1, (
                    _stream_epochs * args.num_train_epochs
                ) // args.gradient_accumulation_steps)
            else:
                total_steps = _resolve_training_steps(
                    args,
                    batches,
                    batch_iter,
                    includes_epochs=getattr(
                        self, "_prepared_batches_include_epochs", False,
                    ),
                )
            (
                _,
                _compile_shape_guard_report,
                _shape_guard_compile_allowed,
                _,
            ) = _plan_single_process_text_shapes(
                batches,
                batch_iter,
                args=args,
                total_steps=total_steps,
                is_vlm=is_vlm,
                distributed_world_size=distributed_world_size,
                compile_policy=compile_policy,
                includes_epochs=getattr(
                    self, "_prepared_batches_include_epochs", False,
                ),
                vlm_compile_decision=getattr(self, "_compile_decision", None),
            )
        # Shared by the preflight and prepared paths: batch_iter is the
        # streaming producer when active, None otherwise.
        _prefetch_active = bool(
            getattr(self, "_mlx_prefetch_control", None)
            and self._mlx_prefetch_control.get("eligible")
        )
        self._active_batch_iter = batch_iter
        grad_accum = args.gradient_accumulation_steps
        # Conceptual total micro-batches for an epoch-count run, used to shrink
        # the step budget when should_epoch_stop skips an epoch's tail. None for
        # max_steps and single-pass runs. The default path holds ONE pass revisited
        # num_train_epochs times; the torch_randperm path already holds every
        # epoch. Ceiled, matching HF's range(ceil(num_train_epochs)): truncating
        # made 1.5 epochs stopped in epoch 1 see a one-pass horizon and drop its
        # budget to zero. The tail stays bounded by the clamp in
        # _honor_epoch_stop_skip, and whole counts are unchanged.
        _epoch_stop_total_microbatches = None
        if args.max_steps <= 0 and batches is not None:
            _n_batches = len(batches)
            if getattr(self, "_prepared_batches_include_epochs", False):
                _epoch_stop_total_microbatches = _n_batches
            elif args.num_train_epochs > 0:
                _epoch_stop_total_microbatches = (
                    _n_batches * math.ceil(float(args.num_train_epochs))
                )
        elif (
            args.max_steps <= 0
            and batch_iter is not None
            and args.num_train_epochs > 0
        ):
            # Declared-length streaming epochs: _prepare_data resolved the source
            # length into _streaming_epoch_batch_count, so the horizon is that
            # per-pass count times the ceiled epoch count, as on the materialized
            # path. Without it a skipped epoch tail kept the full budget and the run
            # made it up out of the next pass, overtraining past num_train_epochs.
            # An unsized stream leaves the count 0 and stays None.
            _stream_epoch_batches = int(
                getattr(self, "_streaming_epoch_batch_count", 0) or 0
            )
            if _stream_epoch_batches > 0:
                _epoch_stop_total_microbatches = (
                    _stream_epoch_batches
                    * math.ceil(float(args.num_train_epochs))
                )
        # Micro-batches in one epoch, where the epoch's last micro-batch forces an
        # optimizer step (HF's do_sync_step). Set for epoch-count runs and for
        # max_steps runs whose plan reports an exact one-pass length; None for
        # streaming and for max_steps runs left on the dataset-size
        # approximation, which keep the flat model.
        _epoch_flush_microbatches = _mlx_epoch_microbatches(
            args,
            batches,
            includes_epochs=getattr(
                self, "_prepared_batches_include_epochs", False,
            ),
        )
        self._compile_shape_guard_report = _compile_shape_guard_report

        # Build optimizer with LR schedule
        optimizer = self._build_optimizer(total_steps)

        # Resume: adapters were already loaded into the model before train(), so
        # only optimizer and trainer state (step counter + loss history) are handled
        # here. The step offset is applied at loop start so the LR scheduler and the
        # dataloader fast-forward together.
        # Reset per-run state so reusing a trainer for a second train() without
        # resume starts clean (else run-1's early-stop flag breaks the loop at
        # step 0). The resume block below re-seeds the persisted fields.
        self._reset_run_state()

        _resume_step = 0
        _resume_from = getattr(self, "_resume_from_checkpoint", None)
        _resume_from = self._validate_distributed_resume_checkpoint(_resume_from)
        if _resume_from:
            # Up front: a missing file otherwise surfaces as a generic mx.load
            # RuntimeError that the handler below does not catch.
            _require_complete_resume_checkpoint(_resume_from)
            try:
                # 1. Load trained adapter weights into the model. The model
                #    already has LoRA wrappers applied (Unsloth pipeline does
                #    get_peft_model before training); strict=False ensures
                #    only the LoRA params match and base weights are untouched.
                model.load_weights(
                    f"{_resume_from}/adapters.safetensors", strict=False,
                )
                # 2. Restore optimizer state (Adam moments m,v, step counter).
                load_optimizer_state(optimizer, _resume_from)
                # 3. Restore trainer scalars (step counter, loss history, and
                #    best-model / early-stopping tracking). .get defaults keep
                #    pre-fix checkpoints (which lack these keys) resumable.
                _cached = getattr(self, "_mlx_resume_state_cache", None)
                ts = (
                    _cached[1]
                    if _cached is not None and _cached[0] == _resume_from
                    else load_trainer_state(_resume_from)
                )
                _resume_step = int(ts.get("global_step", 0))
                # Seed the live step counter from the checkpoint so a no-op
                # resume (checkpoint already at max_steps, loop body never runs)
                # still reports the reached step instead of the initial 0. The
                # loop overwrites this on every optimizer step of a real resume.
                self._global_step = _resume_step
                self._train_loss_history = list(ts.get("train_loss_history", []))
                # Restore the totals so a resumed run still spans both halves.
                # Older checkpoints carry the history but no weights: fall back to
                # the legacy mean rather than report only post-resume windows.
                if "train_loss_token_total" in ts:
                    self._train_loss_token_sum = float(
                        ts.get("train_loss_token_sum", 0.0) or 0.0
                    )
                    self._train_loss_token_total = int(
                        ts.get("train_loss_token_total", 0) or 0
                    )
                elif self._train_loss_history:
                    self._train_loss_weighting_ok = False
                self._best_metric = ts.get("best_metric", None)
                self._best_step = ts.get("best_step", None)
                self._es_patience_counter = int(ts.get("es_patience_counter", 0) or 0)
                # Restore the callback-visible input-token counter so a
                # token-budget stopping callback resumes from the accumulated
                # count rather than 0. .get default keeps pre-fix checkpoints
                # (no num_input_tokens_seen key) resumable.
                self._resume_num_input_tokens_seen = int(
                    ts.get("num_input_tokens_seen", 0) or 0
                )
                # Restore the callback-visible epoch so on_train_begin reports the
                # checkpoint's progress instead of None. Without it a no-op resume
                # (checkpoint already at max_steps, loop body never runs) leaves it
                # None through on_train_end, where HF's NotebookProgressCallback
                # does int(state.epoch). .get keeps pre-fix checkpoints resumable.
                _ckpt_epoch = ts.get("epoch", None)
                self._resume_epoch = (
                    None if _ckpt_epoch is None else float(_ckpt_epoch)
                )
                # Stash the checkpoint's ExportableState callback state; it is
                # applied after _init_callback_state, which rebuilds self.state.
                # The .get default keeps pre-fix checkpoints resumable.
                self._resume_stateful_callbacks = ts.get(
                    "stateful_callbacks", None
                ) or {}
                # Same for the callback-visible log history, which seeds
                # state.log_history. .get keeps pre-fix checkpoints resumable.
                self._resume_log_history = list(ts.get("log_history", None) or [])
                # best/ lives in output_dir, not in the checkpoint dir, so a
                # checkpoint resumed elsewhere (copied dir, new output_dir) can
                # carry best-model state whose weights aren't present. Keep the
                # state only when they are: an unloadable "best" would suppress
                # best-saves and early-stop against a model that
                # load_best_model_at_end can't restore.
                _best_path = f"{args.output_dir}/best/adapters.safetensors"
                _best_weights_missing = (
                    self._best_step is not None and not os.path.exists(_best_path)
                )
                if _best_weights_missing:
                    _main_print(
                        f"Unsloth: checkpoint carries best-model state (step "
                        f"{self._best_step}) but {args.output_dir}/best has no "
                        f"saved weights; restarting best-model tracking."
                    )
                    self._best_metric = None
                    self._best_step = None
                    self._es_patience_counter = 0
                # TrainerState.best_metric can be live while "best_metric" above
                # is null, so restore it from its own key, else
                # EarlyStoppingCallback calls the first post-resume eval a new
                # best. The restart above clears it too: HF keeps ONE watermark
                # for callbacks and best selection. Read after that branch so
                # pre-fix checkpoints keep the native fallback.
                self._resume_callback_best_metric = None if _best_weights_missing else ts.get(
                    "callback_best_metric", self._best_metric,
                )
                self._resume_callback_best_step = None if _best_weights_missing else ts.get(
                    "callback_best_step", self._best_step,
                )
                _main_print(
                    f"Unsloth: Resuming from {_resume_from} "
                    f"(step={_resume_step}, loss_history={len(self._train_loss_history)} entries)."
                )
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"Unsloth: resume_from_checkpoint={_resume_from!r} but "
                    f"resume state files are missing ({e}). Refusing to "
                    f"silently restart from step 0."
                ) from e

        self.callback_handler.optimizer = optimizer
        self.callback_handler.lr_scheduler = getattr(self, "_lr_schedule", None)
        self.callback_handler.processing_class = self.processor or self.tokenizer
        self._ensure_callback_args_compat()
        self._init_callback_state(total_steps, _resume_step, batches)
        # _init_callback_state rebuilds self.state, so seed the callback-visible
        # stateful_callbacks after it.
        self._restore_callback_states(
            getattr(self, "_resume_stateful_callbacks", None) or {}
        )

        # Build loss+grad function — returns ((loss, ntoks), grads)
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

        # Per-group learning rates (LoRA+, embedding LR) via post-update rescale
        lora_plus_ratio = args.lora_plus_ratio
        use_lora_plus = lora_plus_ratio > 0
        if use_lora_plus:
            _main_print(f"Unsloth: LoRA+ enabled (ratio={lora_plus_ratio}).")

        embedding_lr = args.embedding_learning_rate
        main_lr = args.learning_rate
        # Ratio < 1 slows embeddings down; 0 = disabled
        use_embedding_lr = embedding_lr > 0 and main_lr > 0
        embedding_lr_ratio = embedding_lr / main_lr if use_embedding_lr else 1.0
        if use_embedding_lr:
            _main_print(
                f"Unsloth: Embedding LR = {embedding_lr:.2e} "
                f"(ratio={embedding_lr_ratio:.3f} of main LR {main_lr:.2e})."
            )

        _scoped_lr_requested = use_lora_plus or use_embedding_lr

        # Per-group LR via post-update STEP rescale, not gradient scaling:
        # update-normalizing optimizers (AdamW/Lion/Adafactor, Muon rank>=2) are
        # invariant to a constant gradient scale. Rescaling the realized delta
        # (``param = pre + ratio*(post - pre)``) gives effective LR
        # ``ratio*base_lr`` for ANY optimizer, scales the decoupled decay with
        # the step, and adds no optimizer state. Scoped keys: LoRA+ -> lora_b;
        # embedding LR -> the CPT full-module keys, else a literal fallback.
        _cpt_full_keys = getattr(
            model, "_unsloth_cpt_full_module_weight_keys", None) or set()

        def _scoped_step_ratio(name):
            # mlx-lm may wrap the LoRA halves in nn.Linear children, flattening
            # lora_b to `...lora_b.weight`.
            if use_lora_plus and (
                name == "lora_b" or name.endswith(".lora_b")
                or name == "lora_b.weight" or name.endswith(".lora_b.weight")
            ):
                return lora_plus_ratio
            if use_embedding_lr:
                if name in _cpt_full_keys:
                    return embedding_lr_ratio
                _seg = name.split(".")
                if (len(_seg) >= 2 and _seg[-1] == "weight"
                        and _seg[-2] in ("embed_tokens", "lm_head")):
                    return embedding_lr_ratio
            return None

        # The trainable set is fixed after get_peft_model, so classify once.
        _scoped_ratios = {}
        if _scoped_lr_requested:
            for name, _value in tree_flatten(model.trainable_parameters()):
                r = _scoped_step_ratio(name)
                # ratio == 1.0 is a no-op; skip it so nothing large is snapshotted.
                if r is not None and r != 1.0:
                    _scoped_ratios[name] = r
        # A no-op ratio then neither snapshots anything nor disables the fast path.
        _needs_step_rescale = bool(_scoped_ratios)

        def _snapshot_scoped_params():
            """Pre-update values + ratio per scoped leaf, captured before
            decoupled decay so the rescale scales the decay too."""
            if not _scoped_ratios:
                return {}
            snap = {}
            for name, value in tree_flatten(model.trainable_parameters()):
                r = _scoped_ratios.get(name)
                if r is not None:
                    snap[name] = (value, r)
            return snap

        def _rescale_scoped_params(snap):
            if not snap:
                return
            live = dict(tree_flatten(model.trainable_parameters()))
            updates = []
            for name, (pre, ratio) in snap.items():
                post = live[name]
                r = mx.array(ratio, dtype=mx.float32).astype(post.dtype)
                updates.append((name, pre + r * (post - pre)))
            model.update(tree_unflatten(updates))

        # Build step functions following mlx-lm's pattern. `max_grad_value`
        # remains an elementwise clamp. MLX's cheap default is now the clearer
        # `max_grad_leaf_norm`, a proportional per-leaf norm cap that avoids
        # global norm clipping's cross-tree memory overhead.
        (
            max_grad_norm,
            max_grad_value,
            max_grad_leaf_norm,
            _grad_clip_mode,
        ) = _resolve_mlx_grad_clipping(args)
        _raw_mgln = getattr(args, "max_grad_leaf_norm", None)
        if max_grad_value > 0:
            conflicts = []
            if float(getattr(args, "max_grad_norm", 0.0) or 0.0) > 0:
                conflicts.append("max_grad_norm")
            if _raw_mgln is not None and float(_raw_mgln or 0.0) > 0:
                conflicts.append("max_grad_leaf_norm")
            if conflicts:
                _main_print(
                    "Unsloth: max_grad_value is elementwise and overrides "
                    f"{', '.join(conflicts)}."
                )
        elif (
            max_grad_leaf_norm > 0
            and float(getattr(args, "max_grad_norm", 0.0) or 0.0) > 0
        ):
            _main_print(
                "Unsloth: max_grad_leaf_norm is enabled; ignoring "
                "max_grad_norm to avoid double clipping."
            )
        _clip_grad_value = max_grad_value > 0
        _clip_grad_leaf_norm = max_grad_leaf_norm > 0
        # Construction-time Python constant: selects one of two step-graph
        # shapes for the whole run. Never a runtime or mx.array condition —
        # that would add a report/no-report compile trace signature.
        _report_grad_norm = bool(getattr(args, "report_grad_norm", False))
        _compute_report_norm = _report_grad_norm and max_grad_norm <= 0
        state = [model.state, optimizer.state, mx.random.state]
        # grad_accum==1 fast path: only for unclipped updates, since
        # clip_grad_norm can spike peak memory on bf16 VLM runs.
        _direct_single_step_update = (
            grad_accum == 1 and
            distributed_world_size <= 1 and
            not _needs_step_rescale and
            max_grad_norm <= 0 and
            not _clip_grad_value and
            not _clip_grad_leaf_norm
        )

        _restore_storage_after_norm_clip = max_grad_norm > 0
        _trainable_storage_dtypes = (
            {
                name: value.dtype
                for name, value in tree_flatten(model.trainable_parameters())
                if not self._is_norm_parameter_name(name)
                and not self._is_lora_parameter_name(name)
            }
            if _restore_storage_after_norm_clip
            else {}
        )

        def _restore_trainable_storage_dtypes():
            """Keep norm-clipped MLX updates from promoting base params."""
            if not _restore_storage_after_norm_clip:
                return
            recast = []
            needs_update = False
            for name, value in tree_flatten(model.trainable_parameters()):
                dtype = _trainable_storage_dtypes.get(name)
                if dtype is not None and value.dtype != dtype:
                    value = value.astype(dtype)
                    needs_update = True
                recast.append((name, value))
            if needs_update:
                model.update(tree_unflatten(recast))

        def _grad_leaf_scale(name, safe_toks_f, clip_scale=None, dtype=None):
            """Return the scalar applied to one grad leaf before update.

            Pass ``dtype`` (the leaf grad's dtype) so an fp32 scale doesn't
            promote a bf16/fp16 grad tree to fp32 (which would force
            optimizer.update to promote params/m/v too).
            """
            scale = mx.array(1.0, dtype=mx.float32) / safe_toks_f
            # Scoped ratios are NOT applied here: gradient scaling is a near
            # no-op under update-normalizing optimizers (see the step rescale).
            if clip_scale is not None:
                scale = scale * clip_scale
            if dtype is not None and scale.dtype != dtype:
                scale = scale.astype(dtype)
            return scale

        def _apply_update(grad, toks_f):
            """Scale accumulated grads by supervised-token count, apply the
            selected clipping mode, and update. Global-norm clipping reports
            its pre-clip norm; other modes report the same norm only when
            ``report_grad_norm`` opts in (default: no reporting reduction).
            """
            if distributed_world_size > 1:
                grad = self._distributed_sum_gradient_tree(grad)
                toks_f = self._distributed_all_sum(toks_f)
                if int(toks_f.item()) == 0:
                    return None
            safe_toks_f = mx.maximum(
                toks_f, mx.array(1.0, dtype=mx.float32)
            )
            flat_grad = tree_flatten(grad)
            grad_norm = None
            final_items = []
            for name, value in flat_grad:
                scaled = value * _grad_leaf_scale(
                    name, safe_toks_f, None, value.dtype
                )
                final_items.append((name, scaled))
            final_grad = tree_unflatten(final_items)
            if max_grad_norm > 0:
                final_grad, grad_norm = _clip_grad_norm_fp32(
                    final_grad, max_norm=max_grad_norm
                )
            elif _compute_report_norm:
                grad_norm = _global_grad_norm_fp32(final_grad)
            if _clip_grad_value:
                final_grad = _clip_grad_by_value(final_grad, max_grad_value)
            if _clip_grad_leaf_norm:
                final_grad = _clip_grad_by_leaf_norm(final_grad, max_grad_leaf_norm)
            # Snapshot BEFORE decay so the rescale covers decay + optimizer step.
            _scoped_snap = _snapshot_scoped_params() if _needs_step_rescale else None
            # Coupled (SGD) decay folds into the post-clip grad so it feeds
            # momentum; decoupled (AdamW-family) decay shrinks params directly.
            final_grad = self._apply_coupled_weight_decay(model, final_grad)
            self._apply_manual_weight_decay(model, optimizer, final_grad)
            optimizer.update(model, final_grad)
            if _scoped_snap:
                _rescale_scoped_params(_scoped_snap)
            _restore_trainable_storage_dtypes()
            return grad_norm

        def _apply_update_direct(grad, toks_f):
            """Fast exact path for ``grad_accum == 1`` with no per-leaf scaling.

            The raw grads already are the per-token average, so skip the
            ``*ntoks`` then ``/ntoks`` round-trip (which only promotes the tree
            to float32 and spikes peak memory) and clip/update directly.
            """
            grad_norm = None
            if max_grad_norm > 0:
                grad, grad_norm = _clip_grad_norm_fp32(grad, max_norm=max_grad_norm)
            elif _compute_report_norm:
                # Report the exact value the accumulated path computes by
                # emulating its token multiply/divide round-trip on a copy
                # used only for the norm; the fast-path update itself stays on
                # the raw gradients (reporting must not change numerics).
                safe_toks_f = mx.maximum(toks_f, mx.array(1.0, dtype=mx.float32))
                inv_toks = mx.array(1.0, dtype=mx.float32) / safe_toks_f
                def _rounded_for_norm(g):
                    if g.dtype == mx.float16:
                        # fp16 cannot represent token counts >= 65520 (the
                        # cast turns inf, zero grads nan): weight fp16 leaves
                        # in fp32 — exact norm rather than a rounding match.
                        return g.astype(mx.float32) * toks_f * inv_toks
                    return (g * toks_f.astype(g.dtype)) * inv_toks.astype(g.dtype)

                rounded = tree_map(_rounded_for_norm, grad)
                grad_norm = _global_grad_norm_fp32(rounded)
            if _clip_grad_value:
                grad = _clip_grad_by_value(grad, max_grad_value)
            if _clip_grad_leaf_norm:
                grad = _clip_grad_by_leaf_norm(grad, max_grad_leaf_norm)
            grad = self._apply_coupled_weight_decay(model, grad)
            self._apply_manual_weight_decay(model, optimizer, grad)
            optimizer.update(model, grad)
            _restore_trainable_storage_dtypes()
            return grad_norm

        def _loss_and_grad(batch_data):
            if isinstance(batch_data, dict):
                return loss_and_grad_fn(model, batch_data)
            return loss_and_grad_fn(
                model, batch_data[0], batch_data[1], batch_data[2]
            )

        def _accumulate_weighted_grad(grad, toks_f, prev_state):
            """Accumulate token-weighted grads without distributed collectives."""
            if prev_state is not None:
                prev_grad, prev_toks = prev_state
                # stop_gradient: accumulated grads are state, not something to
                # differentiate through; keeps CCE-style VJPs from corrupting
                # the carried bf16 accumulation graph.
                prev_grad = tree_map(mx.stop_gradient, prev_grad)
                prev_toks = mx.stop_gradient(prev_toks)
                grad = tree_map(
                    lambda g, p: p + g * toks_f.astype(g.dtype),
                    grad, prev_grad,
                )
                toks_f = toks_f + prev_toks
            else:
                grad = tree_map(
                    lambda g: g * toks_f.astype(g.dtype),
                    grad,
                )
            return grad, toks_f

        def _local_grad_step(batch_data, prev_state):
            """Local loss/grad accumulation step, safe to compile under DDP."""
            (lvalue, toks), grad = _loss_and_grad(batch_data)
            toks_f = toks.astype(mx.float32)
            grad, toks_f = _accumulate_weighted_grad(grad, toks_f, prev_state)
            # Carried as state across loop iterations, or reduced eagerly
            # outside mx.compile under DDP.
            grad = tree_map(mx.stop_gradient, grad)
            toks_f = mx.stop_gradient(toks_f)
            return lvalue, toks, (grad, toks_f)

        # Unified step for VLM (dict batch) and text (tuple batch) training.
        def step_fn(batch_data, prev_state, do_update):
            (lvalue, toks), grad = _loss_and_grad(batch_data)

            if _direct_single_step_update:
                grad_norm = _apply_update_direct(grad, toks.astype(mx.float32))
                return lvalue, toks, None, grad_norm

            toks_f = toks.astype(mx.float32)
            grad_norm = mx.array(0.0, dtype=mx.float32)
            grad, toks_f = _accumulate_weighted_grad(grad, toks_f, prev_state)

            if do_update:
                grad_norm = _apply_update(grad, toks_f)
                return lvalue, toks, None, grad_norm

            grad = tree_map(mx.stop_gradient, grad)
            toks_f = mx.stop_gradient(toks_f)
            return lvalue, toks, (grad, toks_f), None

        _compile_decision = getattr(self, "_compile_decision", None)
        _use_compile = (
            compile_policy.mode != "eager"
            and _shape_guard_compile_allowed
        )
        _ddp_compile_local_grad = _use_compile and distributed_world_size > 1
        if is_vlm and _use_compile:
            qual = getattr(model, "_unsloth_compile_qualification", None) or get_compile_qualification(model)
            if qual is not None:
                model._unsloth_compile_qualification = qual
            if _compile_decision is None:
                _compile_decision = resolve_training_compile(model, policy=compile_policy, args=args)
            model._unsloth_compile_decision = _compile_decision
            if getattr(args, "compile_trace", True):
                self._compile_trace = getattr(self, "_compile_trace", None) or trace_compile_application(
                    model,
                    policy=compile_policy,
                    args=args,
                )
                model._unsloth_compile_trace = self._compile_trace
                model._unsloth_compile_explain = explain_compile_support(
                    model,
                    policy=compile_policy,
                    args=args,
                )
            if _compile_decision.should_raise:
                raise ValueError(
                    f"Unsloth: strict mx.compile requested for VLM arch "
                    f"'{_compile_decision.arch}', but compile cannot be enabled "
                    f"({_compile_decision.reason})."
                )
            if not _compile_decision.enabled:
                _main_print(
                    f"Unsloth: mx.compile disabled for VLM arch "
                    f"'{_compile_decision.arch}' during training; using eager mode "
                    f"({_compile_decision.reason})."
                )
                if getattr(model, "_unsloth_compile_explain", None):
                    _main_print("Unsloth: Compile trace summary:")
                    for line in model._unsloth_compile_explain.splitlines():
                        _main_print(f"  {line}")
                _use_compile = False
        _ddp_compile_local_grad = _use_compile and distributed_world_size > 1
        _shape_guard_eager = _compile_shape_guard_report.action == "eager"
        _compile_scope = "fallback_eager" if _shape_guard_eager else "none"
        _compile_fallback_reason = (
            f"shape_guard:{_compile_shape_guard_report.reason}"
            if _shape_guard_eager else None
        )
        _compile_state = state
        class _DDPCompiledLocalGradError(RuntimeError):
            """Marks failures from the compiled DDP local-gradient graph."""

        def _is_compile_exception(exc):
            msg = str(exc).lower()
            return (
                "compile" in msg
                or "primitive" in msg
                or "trace" in msg
            )

        def _compile_fallback_allowed():
            return (
                _compile_decision.fallback_allowed
                if _compile_decision is not None
                else compile_policy.mode != "strict"
            )

        def _strict_compile_error(exc=None, peer=False):
            peer_text = " on a peer rank" if peer else ""
            error = RuntimeError(
                "Unsloth: strict mx.compile was enabled "
                f"and runtime fallback is disabled{peer_text}."
            )
            if exc is not None:
                raise error from exc
            raise error

        _ddp_update_outside_step = distributed_world_size > 1

        def _ddp_eager_local_step_fn(batch_data, prev_state, do_update):
            lvalue, toks, local_state = _local_grad_step(batch_data, prev_state)
            return lvalue, toks, local_state, None

        if _use_compile:
            _uncompiled_step_fn = step_fn
            if _ddp_compile_local_grad:
                _compile_state = [model.state, mx.random.state]
                _main_print(
                    "Unsloth: mx.compile enabled for MLX DDP local "
                    "loss/gradient accumulation; distributed collectives "
                    "remain eager."
                )
                _compiled_local_grad_step = None
                _compile_setup_error = None
                _compile_setup_abort = None
                try:
                    _compiled_local_grad_step = mx.compile(
                        _local_grad_step,
                        inputs=_compile_state,
                        outputs=_compile_state,
                    )
                except BaseException as e:
                    # Ordinary failures fall back to eager, but an interrupt
                    # must abort through the consensus instead of silently
                    # downgrading the run.
                    if isinstance(e, Exception):
                        _compile_setup_error = e
                    else:
                        _compile_setup_abort = e
                _setup_base = distributed_world_size + 1
                _setup_status = self._distributed_status_mask(
                    (1 if _compile_setup_error is not None else 0)
                    + _setup_base * (1 if _compile_setup_abort is not None else 0)
                )
                self._raise_distributed_failure_from_any(
                    (_setup_status // _setup_base) > 0,
                    "compile setup",
                    _compile_setup_abort,
                )
                if (_setup_status % _setup_base) > 0:
                    if not _compile_fallback_allowed():
                        _strict_compile_error(
                            _compile_setup_error,
                            peer=_compile_setup_error is None,
                        )
                    _main_print(
                        "Unsloth: mx.compile failed during setup; "
                        "falling back to eager mode."
                    )
                    _use_compile = False
                    _compile_scope = "fallback_eager"
                    _compile_fallback_reason = "setup_error"
                    step_fn = _uncompiled_step_fn
                    _ddp_compile_local_grad = False
                    _compiled_local_grad_step = None

                def _ddp_compiled_step_fn(batch_data, prev_state, do_update):
                    try:
                        lvalue, toks, local_state = _compiled_local_grad_step(
                            batch_data, prev_state,
                        )
                        mx.eval(
                            _compile_state,
                            lvalue,
                            toks,
                            local_state[0],
                            local_state[1],
                        )
                    except Exception as e:
                        if _is_compile_exception(e):
                            raise _DDPCompiledLocalGradError(str(e)) from e
                        raise
                    return lvalue, toks, local_state, None

                if _use_compile:
                    step_fn = _ddp_compiled_step_fn
                    _compile_scope = "ddp_local_grad"
            else:
                try:
                    step_fn = mx.compile(step_fn, inputs=state, outputs=state)
                except (ValueError, RuntimeError, TypeError) as e:
                    if not _compile_fallback_allowed():
                        _strict_compile_error(e)
                    _main_print(
                        "Unsloth: mx.compile failed during setup; "
                        "falling back to eager mode."
                    )
                    step_fn = _uncompiled_step_fn
                    _use_compile = False
                    _compile_scope = "fallback_eager"
                    _compile_fallback_reason = "setup_error"
                else:
                    _compile_scope = "full_step"

        if _ddp_update_outside_step and not _ddp_compile_local_grad:
            step_fn = _ddp_eager_local_step_fn

        # Prepare eval batches
        eval_batches = None
        text_completion_only_loss = _text_completion_only_loss_arg(args)
        text_assistant_only_loss = _text_assistant_only_loss_arg(args)

        def _prepare_eval_batches():
            """Materialize eval batches the first time evaluation is requested.

            Deferred so a callback can request evaluation even when the static
            eval cadence is off. Every rank enters this together (the eval
            control flag is rank-synced), so the collective create_batches call
            stays in lockstep.
            """
            nonlocal eval_batches
            if eval_batches is not None or self.eval_dataset is None:
                return eval_batches
            eval_batch_size = (
                getattr(args, "per_device_eval_batch_size", None)
                or args.per_device_train_batch_size
            )
            # Use pre-built labeled eval batches if available
            _labeled_eval = getattr(self, '_eval_batches_labeled', None)
            if _labeled_eval is not None:
                eval_batches = _labeled_eval
            else:
                def _create_eval_batches(eval_dataset):
                    """Build evaluation batches for one dataset split."""
                    if is_vlm:
                        if not _vlm_has_sized_index_space(eval_dataset):
                            raise ValueError(
                                "Unsloth MLX VLM: unsized streaming eval "
                                "datasets are not supported yet. Provide a "
                                "sized (__len__ + __getitem__) eval dataset; "
                                "lazy VLM evaluation is a planned follow-up."
                            )
                        processor = self._resolve_vlm_processor()
                        config = getattr(self.model, "_config", {})
                        _vlm_mask_fn = getattr(self, '_vlm_response_mask_fn', None)
                        return create_vlm_batches(
                            dataset=eval_dataset,
                            processor=processor,
                            config=config,
                            batch_size=eval_batch_size,
                            max_seq_length=args.max_seq_length,
                            image_size=getattr(args, "image_size", None),
                            seed=args.seed,
                            response_mask_fn=_vlm_mask_fn,
                            formatting_func=self.formatting_func,
                            completion_only_loss=text_completion_only_loss,
                            comm_group=self.distributed_world,
                            distributed_pad_mode="empty",
                        )
                    return self._create_text_eval_batches(
                        eval_dataset,
                        eval_batch_size,
                        text_completion_only_loss,
                        text_assistant_only_loss,
                    )

                def _create_every_eval_split():
                    """Build every eval split, in the order the user declared."""
                    if isinstance(self.eval_dataset, dict):
                        return {
                            key: _create_eval_batches(value)
                            for key, value in self.eval_dataset.items()
                        }
                    return _create_eval_batches(self.eval_dataset)

                if is_vlm:
                    # Eager VLM training batches used to be built before this
                    # point, so eval preprocessing could never reach the
                    # training augmentation stream. A lazy training plan builds
                    # nothing yet, so these eval builds would otherwise consume
                    # the draws the first training batch is owed; keep them out
                    # of that stream. ONE preservation spans every split: one
                    # per split would restore the same snapshot before each of
                    # them and replay a single draw sequence for all, where
                    # sequential construction advanced from split to split.
                    # It spans the process-global RNGs only, so state owned
                    # privately -- by the processor, or by a user's
                    # response_mask_fn, which the plan also calls per batch at
                    # materialize -- does still advance here. No snapshot of an
                    # arbitrary object's own counter exists to take.
                    with _preserved_preprocessing_rng():
                        eval_batches = _create_every_eval_split()
                else:
                    eval_batches = _create_every_eval_split()
            self.callback_handler.eval_dataloader = eval_batches
            _eval_steps = int(getattr(self.state, "eval_steps", 0) or 0)
            if eval_batches and _eval_steps > 0:
                lazy_eval = isinstance(eval_batches, _MLXLazyEvalBatchView) or (
                    isinstance(eval_batches, dict)
                    and any(
                        isinstance(value, _MLXLazyEvalBatchView)
                        for value in eval_batches.values()
                    )
                )
                if lazy_eval:
                    # A lazy view has no length to report.
                    _main_print(
                        f"Unsloth: Eval enabled every {_eval_steps} steps "
                        "(lazy text batches)."
                    )
                else:
                    eval_batch_count = (
                        sum(len(value) for value in eval_batches.values())
                        if isinstance(eval_batches, dict) else len(eval_batches)
                    )
                    _main_print(
                        f"Unsloth: Eval enabled every {_eval_steps} steps "
                        f"({eval_batch_count} eval batches)."
                    )
            return eval_batches

        def _fire(event, **kwargs):
            """Dispatch an HF callback event on every rank, like HF Trainer.

            HF invokes callbacks per process and expects host I/O to self-gate
            on state.is_world_process_zero (seeded per rank in
            _init_callback_state, same as on_init_end). Dispatching on rank 0
            alone leaves the peers' process-local training state un-mutated: an
            on_pre_optimizer_step callback overriding optimizer.learning_rate
            would update rank 0 only, so the peers apply the same all-reduced
            gradient with a different LR and the replicas silently diverge.
            Control-flag divergence is reconciled by the caller (_sync_stop for
            the stop flag, _distributed_sync_control_actions for log/eval/save).

            A callback that raises on one rank must not unwind that rank alone:
            the peers would return here and hang at the next collective. Route
            the failure through the distributed consensus path (all ranks call
            _fire in lockstep) so every rank aborts with the original error
            surfaced. Single-process keeps re-raising the original exception
            unchanged.

            Interrupts (KeyboardInterrupt/SystemExit and any other
            BaseException) are captured for the same reason the evaluation,
            batch-fetch and checkpoint paths capture them: a rank-local Ctrl-C
            delivered while a rank is inside a callback -- ordinary with a
            stock host-I/O callback that self-gates on is_world_process_zero,
            so only one rank spends time in it -- would otherwise skip the
            consensus below and strand the peers inside it with no timeout and
            no way to signal them out. _raise_distributed_failure_from_any
            re-raises a non-Exception unwrapped and without mutating trainer
            state, so the interrupted rank still exits with the original
            interrupt.
            """
            call_error = None
            try:
                self.control = self.callback_handler.call_event(
                    event, args, self.state, self.control, **kwargs,
                )
            except BaseException as e:
                call_error = e
            if distributed_world_size > 1:
                self._raise_distributed_failure(
                    call_error is not None,
                    f"{event} callback",
                    call_error,
                )
            elif call_error is not None:
                raise call_error

        def _sync_stop():
            """Propagate a callback stop request to every rank in lockstep.

            _sync_callback_stop copies control.should_training_stop into
            stop_requested on rank 0; _distributed_should_stop all-reduces the
            OR of stop_requested so no rank is left spinning while rank 0 exits.
            """
            self._sync_callback_stop()
            return self._distributed_should_stop()

        def _sync_epoch_stop():
            """OR-reduce control.should_epoch_stop across ranks (the epoch-scoped
            analogue of _sync_stop). Every rank sees the same result, so when the
            honoring below skips the rest of the epoch each rank skips the same
            micro-batches and DDP stays in lockstep."""
            return self._distributed_any_flag(self.control.should_epoch_stop)

        self.callback_handler.train_dataloader = batches if batches is not None else batch_iter
        self.control.should_training_stop = False
        _watch_mode = self._suppress_torch_only_wandb_watch()
        if _watch_mode:
            _main_print(
                f"Unsloth: WANDB_WATCH={_watch_mode} needs a Torch module, so "
                "gradient/parameter watching is off for this MLX run."
            )
        try:
            _fire("on_train_begin")
        finally:
            self._restore_wandb_watch(_watch_mode)
        _sync_stop()

        features = []
        if is_vlm:
            features.append("VLM")
        if use_cce:
            features.append("CCE")
        if args.gradient_checkpointing:
            features.append("GC")
        if _use_compile:
            features.append(
                "compile"
                if _compile_scope == "full_step"
                else f"compile={_compile_scope}"
            )
        elif _compile_decision is not None:
            features.append(f"compile={_compile_decision.support_state}")
        if use_lora_plus:
            features.append(f"LoRA+(r={lora_plus_ratio})")
        features.append(f"LR={args.lr_scheduler_type}")
        resolved_opt = getattr(self, "_resolved_optimizer_name", args.optim)
        if str(resolved_opt).lower() != str(args.optim).lower():
            features.append(f"opt={args.optim}->{resolved_opt}")
        else:
            features.append(f"opt={args.optim}")

        _main_print(
            f"Unsloth: Training for {total_steps} steps, "
            f"BS={args.per_device_train_batch_size}, "
            f"grad_accum={grad_accum}, "
            f"seq_len={args.max_seq_length}"
        )
        _main_print(f"Unsloth: Features: {', '.join(features)}")
        if _compile_decision is not None and _compile_decision.setting_recommendations:
            _main_print("Unsloth: Compile recommendations:")
            for rec in _compile_decision.setting_recommendations:
                _main_print(
                    f"  - {rec.setting}={rec.recommended_value!r}: {rec.reason}"
                )

        # Training loop — mlx-lm pattern
        model.train()
        # HF's include_num_input_tokens_seen gate: "no"/False (its default, and the
        # one _ensure_callback_args_compat applies) skips input-token counting
        # entirely. The remaining values select the counting MODE ("non_padding"
        # vs "all"/True). Read once so the branch is identical on every rank.
        input_token_mode = getattr(args, "include_num_input_tokens_seen", False)
        track_input_tokens = input_token_mode not in ("no", False)
        # HF reads the pad id off self.processing_class for the "non_padding"
        # fallback; this trainer's processing class is processor-or-tokenizer.
        input_token_pad_id = getattr(
            self.processor or self.tokenizer, "pad_token_id", None,
        ) if track_input_tokens else None
        start_time = time.perf_counter()
        # Metric accumulators are split into a "committed" window (loss/tokens
        # from optimizer steps that have already been APPLIED to the model but not
        # yet logged) and a "pending" window (the micro-batches of the current
        # accumulation window, which have NOT yet updated the model). HF only ever
        # folds a completed optimizer step's loss into the logged tr_loss and never
        # logs a not-yet-applied partial window (_maybe_log_save_evaluate logs only
        # when state.global_step > _globalstep_last_logged, and its last epoch batch
        # is force-applied as an optimizer step). Mirroring that split keeps a forced
        # log (epoch-end / callback) and the post-loop flush emitting only committed
        # metrics, while an abandoned partial window drops only its pending part.
        losses = 0
        n_tokens = 0
        steps = 0
        pending_losses = 0
        pending_n_tokens = 0
        pending_steps = 0
        trained_tokens = 0
        train_time = 0
        # Wall clock for the PENDING window, split like the loss/token counters
        # above: charging a forced log the pending micro-batches' time would
        # understate its tokens/s and hide that time from the window that owns it.
        pending_time = 0
        grad_accum_state = None
        accum_progress = 0
        batches_per_epoch = self._callback_batches_per_epoch(batches)
        # A streaming source materializes no `batches`, but a supported streaming
        # num_train_epochs run still has a known epoch length: _prepare_data
        # resolved it into _streaming_epoch_batch_count and rejected any source
        # whose per-pass micro-batches are not divisible by grad_accum, so every
        # boundary lands on an optimizer step. HF drives the epoch lifecycle off
        # exactly this quantity whenever the dataloader reports a length, which an
        # IterableDataset with __len__ does. Without it the epoch events never
        # fired and state.epoch stayed None for the whole run.
        if batches is None and batch_iter is not None and not batches_per_epoch:
            batches_per_epoch = int(
                getattr(self, "_streaming_epoch_batch_count", 0) or 0
            ) or None
        # A length-less stream has no dataset boundaries, but HF still runs ONE
        # conceptual epoch over a synthetic horizon of max_steps * grad_accum, since
        # num_train_epochs = sys.maxsize only means "re-iterate as needed" and its
        # loop body runs once (measured identical on 4.x and 5.x). state.epoch needs
        # the same quantity anyway: WandbCallback.on_save does round(state.epoch, 2)
        # and raises TypeError on None.
        stream_epoch_microbatches = None
        if not batches_per_epoch and batch_iter is not None:
            _stream_max_steps = int(getattr(args, "max_steps", 0) or 0)
            if _stream_max_steps > 0:
                stream_epoch_microbatches = _stream_max_steps * grad_accum
        # Epoch length the CALLBACK lifecycle uses: the real dataset pass when there
        # is one, HF's synthetic horizon otherwise. Deliberately a separate name from
        # batches_per_epoch, because honoring should_epoch_stop skips to the next
        # boundary and for a length-less stream that would drain a producer that
        # cannot replay (HF restarts its iterator instead), so _honor_epoch_stop_skip
        # stays on batches_per_epoch. Equal to it whenever it is set.
        epoch_event_microbatches = batches_per_epoch or stream_epoch_microbatches

        def _run_callback_epoch_begin(epoch_value):
            """Dispatch one epoch-begin event at a dataset boundary (rank 0)."""
            self.state.epoch = epoch_value
            # HF resets should_epoch_stop at on_epoch_begin so a per-epoch stop
            # from a prior epoch does not leak into this one.
            self.control.should_epoch_stop = False
            _fire("on_epoch_begin")
            _sync_stop()

        def _maybe_callback_epoch_begin(microstep):
            """Dispatch epoch-begin at finite dataset boundaries."""
            if (not epoch_event_microbatches
                    or (microstep - 1) % epoch_event_microbatches != 0):
                return False
            _run_callback_epoch_begin(
                (microstep - 1) / epoch_event_microbatches
            )
            return True

        def _maybe_callback_epoch_end(microstep, current_step, grad_norm):
            """Dispatch epoch-end after an optimizer step at a dataset boundary."""
            if (not epoch_event_microbatches
                    or microstep % epoch_event_microbatches != 0):
                return
            self.state.epoch = microstep / epoch_event_microbatches
            # The loop's own "epoch" strategy cadence, raised where HF's flow
            # raises it and folded into the same control flags, so it is a no-op
            # when a DefaultFlowCallback is installed to raise it too.
            self._request_epoch_cadence_actions()
            _fire("on_epoch_end")
            # A rank-dependent on_epoch_end callback can request log/eval/save on
            # a subset; sync before the collective actions so peers stay in lockstep.
            self._distributed_sync_control_actions()
            if self.control.should_log or self.control.should_evaluate or self.control.should_save:
                _run_callback_control_actions(current_step, grad_norm)
            _sync_stop()

        def _run_training_log(current_step, grad_norm):
            """Emit one MLX/HF training log from accumulated loss counters.

            The loss/token totals are all-reduced so every rank logs the same
            global figures. Printing and the native step callbacks run on rank 0;
            on_log fires on every rank and self-gates on is_world_process_zero.
            """
            nonlocal losses, n_tokens, steps, train_time, trained_tokens
            # Nothing accumulated since the last log: a callback can force
            # should_log again on a step that already logged, and the accumulators
            # are plain-int 0 after a reset, so .item() below would raise and a real
            # log would divide by zero. Skip like HF, which guards on
            # global_step > _globalstep_last_logged. steps advances identically on
            # every rank, so no rank reaches the all-sum without the others.
            if steps == 0:
                self.control.should_log = False
                return
            metric_losses = self._distributed_all_sum(losses, stream=mx.cpu)
            metric_tokens = self._distributed_all_sum(n_tokens, stream=mx.cpu)
            mx.eval(metric_losses, metric_tokens)
            train_loss = (
                (metric_losses / metric_tokens).item()
                if metric_tokens.item() > 0 else 0.0
            )
            local_tok_count = int(n_tokens.item())
            tok_count = int(metric_tokens.item())
            trained_tokens += tok_count
            lr_val = optimizer.learning_rate.item()
            tokens_sec = tok_count / train_time if train_time > 0 else 0
            peak_mem = mx.get_peak_memory() / 1e9

            self._train_loss_history.append(train_loss)
            # metric_losses is already sum(loss * tokens) over the window, so these
            # totals give the exact global mean whatever the boundaries were.
            self._train_loss_token_sum += float(metric_losses.item())
            self._train_loss_token_total += tok_count
            grad_norm_val = (
                float(grad_norm.item())
                if grad_norm is not None else None
            )
            if grad_norm_val is not None:
                self._grad_norm_history.append(grad_norm_val)
            self._tokens_per_second_history.append(tokens_sec)
            self._peak_memory_history.append(peak_mem)
            self._step_times.append(train_time / steps if steps > 0 else 0)
            self._local_token_count_history.append(local_tok_count)
            self._global_token_count_history.append(tok_count)

            reset_after = getattr(self, '_benchmark_reset_peak_after_step', 0)
            if reset_after > 0 and current_step == reset_after:
                mx.synchronize()
                mx.reset_peak_memory()

            elapsed_total = time.perf_counter() - start_time
            grad_text = (
                f"Grad: {grad_norm_val:.4f} | "
                if grad_norm_val is not None else ""
            )
            _main_print(
                f"  Step {current_step}/{total_steps} | "
                f"Loss: {train_loss:.4f} | "
                f"{grad_text}"
                f"LR: {lr_val:.2e} | "
                f"Tok/s: {tokens_sec:.0f} | "
                f"Peak: {peak_mem:.2f} GB"
            )

            if is_main_process:
                for cb in self._step_callbacks:
                    try:
                        cb(
                            current_step, total_steps, train_loss, lr_val,
                            tokens_sec, peak_mem, elapsed_total, trained_tokens,
                            grad_norm_val,
                        )
                    except Exception as e:
                        _main_print(f"Unsloth: step callback error: {e}")

            logs = {
                "loss": train_loss,
                "learning_rate": lr_val,
                "tokens_per_second": tokens_sec,
                "peak_memory_gb": peak_mem,
                "trained_tokens": trained_tokens,
            }
            if grad_norm_val is not None:
                logs["grad_norm"] = grad_norm_val
            # HF's Trainer.log stamps the epoch onto every payload, so a persisted
            # log_history entry keeps it after state.epoch has moved on.
            if self.state.epoch is not None:
                logs["epoch"] = self.state.epoch
            self.control.should_log = False
            # Every rank appends, like HF Trainer.log: on_log now fires on all
            # ranks, so a peer callback must see the entry it is notified about.
            record = dict(logs)
            record["step"] = self.state.global_step
            self.state.log_history.append(record)
            _fire("on_log", logs=logs)
            # on_log may itself request an eval or save (HF checks
            # should_evaluate/should_save after logging in the same step). Sync now:
            # the caller's should_eval / should_save branches run collective eval +
            # rank-0-guarded saves, so a request raised on a subset of ranks would
            # make those ranks enter _run_eval/_run_checkpoint while the rest skip
            # them and hang at the collective. No-op at world 1.
            self._distributed_sync_control_actions()

            losses = 0
            n_tokens = 0
            steps = 0
            train_time = 0

        def _run_eval(current_step):
            """Run eval and dispatch MLX/HF eval callbacks in DDP lockstep."""
            current_eval_batches = _prepare_eval_batches()
            if not current_eval_batches:
                self.control.should_evaluate = False
                return False
            # Pause the training prefetch producer for the eval: it shares the
            # tokenizer/processor this eval pass re-enters.
            _pf = (
                self._mlx_prefetch_control.get("prefetcher")
                if getattr(self, "_mlx_prefetch_control", None) else None
            )
            if _pf is not None:
                _pf.quiesce()
            _metrics_before_eval = self._last_eval_metrics
            try:
                val_loss, ppl = self._evaluate(
                    current_eval_batches, loss_fn, is_vlm=is_vlm)
            finally:
                if _pf is not None:
                    _pf.resume()
            model.train()
            # An external cancel makes _evaluate_batch_totals skip every remaining
            # batch, so eval_loss is 0.0 from an evaluation that never ran.
            # Dispatching it resets EarlyStoppingCallback's patience and latches a
            # best watermark no real evaluation can beat, both persisted by
            # _run_checkpoint. Drop it and restore the last real metrics. A
            # CALLBACK stop cannot reach here: it stays in should_training_stop
            # until after this step's actions, so it still gets its evaluation.
            # OR-reduced so a cancel landing on one rank cannot strand its peers
            # in _fire.
            if self._distributed_should_stop():
                self._last_eval_metrics = _metrics_before_eval
                self.control.should_evaluate = False
                return False
            _main_print(
                f"  Eval  {current_step}/{total_steps} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Perplexity: {ppl:.2f}"
            )
            if is_main_process:
                for cb in self._eval_callbacks:
                    try:
                        cb(current_step, val_loss, ppl)
                    except Exception as e:
                        _main_print(f"Unsloth: eval callback error: {e}")

            metrics = self._last_eval_metrics or {}
            # Same epoch stamp. Rebound rather than mutated so the checkpoint's
            # eval_metrics payload stays exactly as evaluated.
            if self.state.epoch is not None:
                metrics = {**metrics, "epoch": self.state.epoch}
            record = dict(metrics)
            record["step"] = self.state.global_step
            self.state.log_history.append(record)
            _fire("on_log", logs=dict(metrics))
            # Clear AFTER the eval on_log and just before on_evaluate, where HF
            # clears it. Clearing earlier lets an on_log callback's fresh
            # should_evaluate=True survive, so a boundary step evaluates twice.
            self.control.should_evaluate = False
            _fire("on_evaluate", metrics=metrics)
            # on_log/on_evaluate may each request a log/eval/save, and HF checks
            # should_save after on_evaluate in the same step. Sync the flags first:
            # both branches run collective code, so a request raised on a subset of
            # ranks would strand the rest. Mirrors the on_log sync in
            # _run_training_log.
            self._distributed_sync_control_actions()
            improved = self._update_callback_best_metric(metrics)
            # save_strategy="best": DefaultFlowCallback never raises this one, so
            # HF's Trainer core decides it right after the evaluation and ASSIGNS
            # rather than ORs (_maybe_log_save_evaluate, 4.x and 5.x alike). The
            # assignment matters: an evaluation that did not improve also clears a
            # save another source requested for this step, which is how "best"
            # writes only improving checkpoints. Rank-consistent without a
            # collective: metrics are all-reduced and best_metric advances
            # identically everywhere.
            if self._best_save_strategy_enabled():
                self.control.should_save = improved
            # An external cancel is OR-reduced here so every rank agrees on
            # stop_requested before the divergent best-model / early-stopping
            # branch in _run_best_tracking, whose rank-0-guarded save would
            # otherwise hang peers. A callback stop from on_step_end / on_evaluate
            # is deliberately NOT latched yet: HF runs _determine_best_metric and
            # writes this step's checkpoint BEFORE honoring should_training_stop,
            # so latching here would make _run_best_tracking (gated on not
            # stop_requested) skip a valid, improving eval and leave
            # load_best_model_at_end restoring a stale model. The caller's tail
            # _sync_stop() applies it after best tracking and the same-step save.
            self._distributed_should_stop()
            return True

        def _run_best_tracking(current_step):
            """Update native best-model + early-stopping state in DDP lockstep."""
            _track = not self.stop_requested and (
                getattr(args, "load_best_model_at_end", False)
                or int(getattr(args, "early_stopping_patience", 0) or 0) > 0
            )
            if not _track:
                return
            _metric_name = self._resolved_best_metric_name()
            _em = self._last_eval_metrics or {}
            if _metric_name not in _em:
                raise ValueError(
                    f"metric_for_best_model={_metric_name!r} not in eval "
                    f"metrics; available: {sorted(_em)}"
                )
            _cur = _em[_metric_name]
            _greater = bool(getattr(args, "greater_is_better", False))
            _improved = (
                _cur == _cur  # reject NaN: a diverged eval must never be "best"
                and (
                    self._best_metric is None
                    or (_cur > self._best_metric if _greater else _cur < self._best_metric)
                )
            )
            if _improved:
                self._best_metric = _cur
                self._best_step = current_step
                self.state.best_metric = _cur
                self.state.best_global_step = current_step
                self.state.best_model_checkpoint = f"{args.output_dir}/best"
                self._es_patience_counter = 0
                # Bookkeeping runs on every rank to keep early-stopping in
                # lockstep; only rank 0 writes output_dir/best. Sync save
                # failures so a rank-0 error does not hang peers at the next
                # collective.
                best_save_error = None
                if is_main_process:
                    try:
                        save_trainable_adapters(model, f"{args.output_dir}/best")
                    except ValueError as e:
                        _main_print(f"  Unsloth: skipped best-model save ({e})")
                    except BaseException as e:
                        best_save_error = e
                self._raise_distributed_failure(
                    best_save_error is not None,
                    "best-model save",
                    best_save_error,
                )
            else:
                self._es_patience_counter += 1
                _pat = int(getattr(args, "early_stopping_patience", 0) or 0)
                if _pat > 0 and self._es_patience_counter >= _pat:
                    _main_print(
                        f"Unsloth: early stopping at step {current_step} "
                        f"(no {_metric_name} improvement in {_pat} evals)."
                    )
                    self._early_stopped = True

        def _run_checkpoint(current_step):
            """Save a step checkpoint (rank 0) and dispatch HF on_save."""
            # Fold the committed-but-unlogged window into the totals WRITTEN to the
            # checkpoint, or a save cadence out of phase with the log cadence
            # persists totals covering fewer steps than its own global_step. Only
            # the payload is adjusted, so _run_training_log does not double count.
            # The all-sum runs on every rank before the rank-0 write guard, and is
            # skipped at steps == 0 where the accumulators are plain ints. The
            # pending window is excluded.
            ckpt_loss_sum = float(self._train_loss_token_sum)
            ckpt_loss_total = int(self._train_loss_token_total)
            ckpt_committed_steps = steps
            if ckpt_committed_steps > 0:
                _ckpt_losses = self._distributed_all_sum(losses, stream=mx.cpu)
                _ckpt_tokens = self._distributed_all_sum(n_tokens, stream=mx.cpu)
                mx.eval(_ckpt_losses, _ckpt_tokens)
                ckpt_loss_sum += float(_ckpt_losses.item())
                ckpt_loss_total += int(_ckpt_tokens.item())
            checkpoint_error = None
            checkpoint_written = False
            # Declared here too: the on_save guard below reads it on every rank,
            # and only rank 0 enters the write block that assigns it.
            checkpoint_complete = False
            if is_main_process:
                ckpt_dir = f"{args.output_dir}/checkpoint-{current_step}"
                try:
                    try:
                        save_trainable_adapters(model, ckpt_dir)
                    except ValueError as e:
                        _main_print(f"  Unsloth: skipped checkpoint ({e})")
                    else:
                        checkpoint_written = True
                        # Also write optimizer + trainer state so
                        # resume_from_checkpoint restores Adam moments, step
                        # counter, loss history, and best-model / early-stopping
                        # tracking. Best-effort: the adapter save already
                        # succeeded, so log failures but keep it.
                        try:
                            save_optimizer_state(optimizer, ckpt_dir)
                            save_trainer_state(
                                {
                                    "global_step": current_step,
                                    # HF checkpoints TrainerState wholesale, so
                                    # epoch travels with global_step and a resumed
                                    # run reports progress from on_train_begin.
                                    # Same key name as trainer_state.json.
                                    "epoch": self.state.epoch,
                                    "train_loss_history": list(
                                        self._train_loss_history
                                    ),
                                    "train_loss_token_sum": ckpt_loss_sum,
                                    "train_loss_token_total": ckpt_loss_total,
                                    "best_metric": self._best_metric,
                                    "best_step": self._best_step,
                                    "es_patience_counter": self._es_patience_counter,
                                    # Persist the callback-visible input-token
                                    # counter so a token-budget stopping callback
                                    # does not restart at 0 on resume (HF saves
                                    # num_input_tokens_seen in trainer_state.json).
                                    "num_input_tokens_seen": int(
                                        self.state.num_input_tokens_seen
                                    ),
                                    # HF's TrainerState watermark, persisted
                                    # wholesale by _save_checkpoint. It advances
                                    # on every eval, so it is non-null even when
                                    # "best_metric"/"best_step" above are null.
                                    "callback_best_metric": (
                                        None if self.state.best_metric is None
                                        else float(self.state.best_metric)
                                    ),
                                    "callback_best_step": (
                                        None if self.state.best_global_step is None
                                        else int(self.state.best_global_step)
                                    ),
                                    # HF writes this into every checkpoint;
                                    # without it the field stays permanently
                                    # empty and no later release can recover it.
                                    "stateful_callbacks":
                                        self._export_callback_states(),
                                    # HF's trainer_state.json carries the full
                                    # log_history; without it a resumed run
                                    # loses every pre-resume entry.
                                    "log_history": list(self.state.log_history),
                                },
                                ckpt_dir,
                            )
                            checkpoint_complete = True
                        except Exception as e:
                            _main_print(
                                "  Unsloth: checkpoint saved without "
                                f"resume state ({e})"
                            )
                        _main_print(f"  Saved checkpoint to {ckpt_dir}")
                        if checkpoint_complete:
                            _prune_stale_checkpoints(
                                args.output_dir,
                                args.save_total_limit,
                                keep_step=self.state.best_global_step,
                            )
                except BaseException as e:
                    checkpoint_error = e
            self._raise_distributed_failure(
                checkpoint_error is not None,
                "checkpoint save",
                checkpoint_error,
            )
            self.control.should_save = False
            # HF fires on_save only after _save_checkpoint writes to disk. A fully
            # frozen model raises ValueError and skips the write, and a failed
            # optimizer/trainer-state write leaves a directory that cannot be
            # resumed from, so neither should be advertised to hub uploaders or
            # checkpoint trackers. The write is rank 0 only, so broadcast the
            # outcome and fire on_save on every rank together, or the rank that
            # skips it strands its peers at the _fire consensus collective.
            checkpoint_written_any = self._distributed_status_mask(
                1 if (is_main_process and checkpoint_written and checkpoint_complete)
                else 0
            ) > 0
            if checkpoint_written_any:
                # HF's _save_checkpoint records the best checkpoint's path on
                # every save, so on_save and on_train_end callbacks can find it.
                # Without this the field stayed None for the whole run whenever
                # best tracking is callback-side rather than native, and no
                # integration could locate a checkpoint that was on disk.
                _best_step = self.state.best_global_step
                if _best_step:
                    _best_dir = f"{args.output_dir}/checkpoint-{int(_best_step)}"
                    self.state.best_model_checkpoint = (
                        _best_dir if os.path.isdir(_best_dir) else None
                    )
                _fire("on_save")

        def _run_callback_control_actions(current_step, grad_norm):
            """Run log/eval/save actions requested by callback control flags.

            Callers sync the control action flags across ranks before invoking
            this so the collective log/eval/save paths run in lockstep.
            """
            if self.control.should_log:
                _run_training_log(current_step, grad_norm)
            if self.control.should_evaluate:
                if _run_eval(current_step):
                    _run_best_tracking(current_step)
            if self.control.should_save:
                _run_checkpoint(current_step)

        # When resuming, start batch_idx at the resume position so the visit
        # mapping (plan-provided for finite plans, modulo for eager lists)
        # lands on the same batch the original run would have seen next. Once an
        # epoch's tail forces a step, global_step no longer maps flatly onto
        # micro-batches, so rebuild the cursor per epoch like HF does; otherwise
        # the resume skips the next epoch's opening micro-batch and cycles into
        # a pass num_train_epochs never authorised.
        if _epoch_flush_microbatches:
            _resume_microstep = _mlx_microstep_for_step(
                _resume_step, _epoch_flush_microbatches, grad_accum,
            )
        else:
            _resume_microstep = _resume_step * grad_accum
        batch_idx = _resume_microstep

        # Streaming mode: fast-forward the iterator to the resume position.
        # The seed is the same and create_batches/iterate_*_batches is
        # deterministic, so consuming N batches gives us the same data
        # ordering the killed run would have produced.
        if _resume_step > 0 and batch_iter is not None and not _prefetch_active:
            for _ in range(_resume_microstep):
                fast_forward_error = None
                try:
                    next(batch_iter)
                except StopIteration:
                    fast_forward_error = RuntimeError(
                        f"Unsloth: streaming dataset exhausted while "
                        f"fast-forwarding to resume step {_resume_step}. "
                        f"Dataset may be shorter than the killed run consumed."
                    )
                except BaseException as e:
                    fast_forward_error = e
                if distributed_world_size > 1:
                    self._raise_distributed_failure(
                        fast_forward_error is not None,
                        "fast-forwarding training batch",
                        fast_forward_error,
                    )
                elif fast_forward_error is not None:
                    raise fast_forward_error

        # Finite VLM plans: replay the skipped micro-batches' preprocessing.
        # The eager builder produced every scheduled batch up front, so the
        # killed run had already run the processor over the skipped region and
        # a stochastic preprocessing pipeline was past it. Rebuilding (and
        # discarding) them here keeps the resumed run on the same augmentation
        # stream an uninterrupted run used, exactly as the streaming branch
        # above fast-forwards its iterator.
        if _resume_step > 0 and batch_iter is None and isinstance(
            batches, FiniteVLMBatchPlan,
        ):
            fast_forward_error = None
            try:
                # The same cursor the fetch and the streaming fast-forward use:
                # the flat product over-advances the augmentation stream once an
                # epoch's tail has forced a step.
                batches.advance_preprocessing(_resume_microstep)
            except BaseException as e:
                fast_forward_error = e
            if distributed_world_size > 1:
                self._raise_distributed_failure(
                    fast_forward_error is not None,
                    "fast-forwarding VLM preprocessing",
                    fast_forward_error,
                )
            elif fast_forward_error is not None:
                raise fast_forward_error

        def _run_ddp_local_step(batch_data, prev_state, do_update):
            """Run local DDP work, then synchronize failures before collectives."""
            nonlocal step_fn, _use_compile, _compile_scope, _ddp_compile_local_grad, state
            nonlocal _compile_fallback_reason

            def _eval_local_result(step_result):
                lvalue, toks, local_state, _grad_norm = step_result
                if local_state is not None:
                    mx.eval(lvalue, toks, local_state[0], local_state[1])
                else:
                    mx.eval(lvalue, toks)

            local_error = None
            compile_error = None
            result = None
            rng_state_before = None
            if _ddp_compile_local_grad:
                rng_state_before = mx.array(
                    mx.random.state[0].tolist(),
                    dtype=mx.uint32,
                )
            try:
                result = step_fn(batch_data, prev_state, do_update)
                _eval_local_result(result)
            except BaseException as e:
                if isinstance(e, _DDPCompiledLocalGradError):
                    compile_error = e
                else:
                    local_error = e

            status_base = distributed_world_size + 1
            status = self._distributed_status_mask(
                (1 if local_error is not None else 0)
                + status_base * (1 if compile_error is not None else 0)
            )
            local_error_any = (status % status_base) > 0
            compile_error_any = (status // status_base) > 0
            self._raise_distributed_failure_from_any(
                local_error_any,
                "training step",
                local_error,
            )

            if compile_error_any:
                if not _compile_fallback_allowed():
                    _strict_compile_error(
                        compile_error,
                        peer=compile_error is None,
                    )
                if rng_state_before is not None:
                    mx.random.state[0] = rng_state_before
                _main_print(
                    "Unsloth: mx.compile failed at runtime; "
                    "falling back to eager mode on all DDP ranks."
                )
                step_fn = _ddp_eager_local_step_fn
                _use_compile = False
                _compile_scope = "fallback_eager"
                _compile_fallback_reason = "runtime_error"
                _ddp_compile_local_grad = False
                if isinstance(batches, _EAGER_REFETCHABLE_PLAN_TYPES):
                    batch_data = batches[scheduled_index]
                state = [model.state, optimizer.state, mx.random.state]
                local_error = None
                try:
                    result = step_fn(batch_data, prev_state, do_update)
                    _eval_local_result(result)
                except BaseException as e:
                    local_error = e
                self._raise_distributed_failure(
                    local_error is not None,
                    "training step after compile fallback",
                    local_error,
                )

            return result

        def _honor_epoch_stop_skip(it_val, current_step, grad_norm):
            """End the current epoch early for a should_epoch_stop callback.

            Fires the truncated epoch's on_epoch_end plus any synced log/eval/save,
            fast-forwards the batch cursor to the next epoch boundary, and shrinks
            the optimizer-step budget so the loop does not cycle
            batches[batch_idx % len(batches)] into extra data passes. Caller has
            confirmed _sync_epoch_stop() and a mid-epoch position
            (it_val % batches_per_epoch != 0). Returns the new `it` at the boundary.
            All arithmetic is rank-consistent (it_val, batches_per_epoch, batch_idx
            and _epoch_stop_total_microbatches are identical on every rank) and the
            on_epoch_end path reuses the lockstep collectives, so DDP stays in step.
            """
            nonlocal batch_idx, total_steps
            # Keep the callback-visible epoch FRACTIONAL for this truncated epoch's
            # on_epoch_end, mirroring HF: state.epoch = epoch + (step+1)/steps_in_epoch
            # is set at the last optimizer step and stays fractional when a callback
            # breaks the epoch mid-way (transformers _inner_training_loop fires
            # on_epoch_end without snapping state.epoch to the next integer). Snapping
            # to ceil(it_val / batches_per_epoch) here would report a full epoch (e.g.
            # 1.0) for a truncated one, making epoch-based integrations treat a
            # partial epoch as completed. it_val / batches_per_epoch is the same
            # fractional value the per-microstep update already set.
            self.state.epoch = it_val / batches_per_epoch
            # HF fires on_epoch_end for a should_epoch_stop-truncated epoch too,
            # so its flow raises the epoch action here; raise ours on the same
            # terms rather than skipping the boundary the callback just closed.
            self._request_epoch_cadence_actions()
            _fire("on_epoch_end")
            self._distributed_sync_control_actions()
            if (self.control.should_log or self.control.should_evaluate
                    or self.control.should_save):
                _run_callback_control_actions(current_step, grad_norm)
            next_boundary = (
                (it_val // batches_per_epoch) + 1
            ) * batches_per_epoch
            if batch_iter is None:
                batch_idx += next_boundary - it_val
            else:
                # A streaming producer has no index to fast-forward, so discard
                # the epoch's remaining micro-batches instead. The producer replays
                # passes back to back, so this lands on the next pass's first batch,
                # where HF lands too: it rebuilds the iterator every epoch and a
                # should_epoch_stop break abandons the rest of the current one. Only
                # declared-length streams reach here, so the count is finite, and it
                # is rank-consistent, so DDP stays in lockstep; a producer failure
                # takes the same consensus path as the loop's own fetch. batch_idx
                # is unused when streaming, so park the cursor on it.
                _drain_error = None
                try:
                    for _ in range(next_boundary - it_val):
                        next(batch_iter)
                except StopIteration:
                    # Exhausted early: the loop's own fetch reports it.
                    pass
                except BaseException as _drain_exc:
                    _drain_error = _drain_exc
                if distributed_world_size > 1:
                    self._raise_distributed_failure(
                        _drain_error is not None,
                        "skipping to the next streaming epoch boundary",
                        _drain_error,
                    )
                elif _drain_error is not None:
                    raise _drain_error
                batch_idx = next_boundary
            # An epoch-count run must shrink its budget after a skip, or the loop
            # cycles into extra passes and overtrains past num_train_epochs.
            # Recompute from the micro-batches that remain.
            # _epoch_stop_total_microbatches covers both layouts, the default
            # cycled pass and the torch_randperm every-epoch plan; the earlier
            # flag-gated len(batches) form silently skipped the default one.
            # max_steps runs keep their fixed budget (the total is None).
            if _epoch_stop_total_microbatches is not None:
                _remaining = _epoch_stop_total_microbatches - batch_idx
                if _epoch_flush_microbatches:
                    # batch_idx now sits on an epoch boundary, so what remains is
                    # a whole number of epochs and each costs a ceil'd step
                    # count; flooring the micro-batches shortens the epochs that
                    # were never truncated.
                    _shrunk = self._global_step + (
                        (_remaining // _epoch_flush_microbatches)
                        * _mlx_steps_per_epoch(
                            _epoch_flush_microbatches, grad_accum,
                        )
                    )
                else:
                    _shrunk = self._global_step + _remaining // grad_accum
                # Never grow the budget. The horizon counts every epoch HF would
                # enter, but a fractional run's final epoch is cut short by the step
                # budget itself, as HF stops such a run mid-epoch on
                # should_training_stop. Whole counts never reach the clamp: each
                # skipped epoch costs at most steps_per_epoch.
                total_steps = min(total_steps, _shrunk)
            return next_boundary

        # DDP-lockstep microstep loop. global_step advances only on optimizer
        # updates; _distributed_should_stop() OR-reduces stop_requested at the
        # top so an early stop (external cancel or an HF stop callback that ran
        # on a subset of ranks) drains every rank together before the next collective.
        microstep = _resume_microstep
        self._global_step = _resume_step
        # Resuming mid-epoch re-enters an epoch whose boundary already passed, so
        # the loop's predicate never fires its begin and a fresh callback sees an
        # unpaired on_epoch_end. HF dispatches on_epoch_begin for the resumed
        # partial epoch too, before skipping its trained batches. A stop set here
        # is drained by the loop's first _distributed_should_stop().
        if epoch_event_microbatches and microstep % epoch_event_microbatches:
            _run_callback_epoch_begin(
                float(microstep // epoch_event_microbatches)
            )
        while self._global_step < total_steps:
            it = microstep + 1
            if self._distributed_should_stop() or self._early_stopped:
                if self.stop_requested:
                    _main_print("Unsloth: Stop requested - ending training early.")
                break

            if _maybe_callback_epoch_begin(it):
                if self.stop_requested:
                    _main_print("Unsloth: Stop requested - ending training early.")
                    break

            if accum_progress == 0:
                self.control.should_log = False
                self.control.should_evaluate = False
                self.control.should_save = False
                _fire("on_step_begin")
                if _sync_stop():
                    _main_print("Unsloth: Stop requested - ending training early.")
                    break

            tic = time.perf_counter()

            # Get next batch
            batch_error = None
            batch_data = None
            try:
                if batch_iter is not None:
                    batch_data = next(batch_iter)
                else:
                    # Resolve the absolute visit exactly once; compiled
                    # materialization, eager access, and both compile-failure
                    # retries all reuse this resolved stored index.
                    scheduled_index = (
                        batches.batch_index_for_visit(batch_idx)
                        if isinstance(batches, _FINITE_BATCH_PLAN_TYPES)
                        else batch_idx % len(batches)
                    )
                    if (
                        _use_compile
                        and _compile_scope in (
                            FULL_STEP_SCOPE, DDP_LOCAL_GRAD_SCOPE,
                        )
                        # Phase-aware admission through the shared finite-plan
                        # protocol; a plan with no shape plan materializes unpadded.
                        and isinstance(batches, _FINITE_BATCH_PLAN_TYPES)
                    ):
                        batch_data = batches.materialize(
                            scheduled_index,
                            phase=_mlx_microstep_phase(
                                _compile_scope,
                                grad_accum,
                                it - 1,
                                _epoch_flush_microbatches,
                            ),
                        )
                    else:
                        batch_data = batches[scheduled_index]
                    batch_idx += 1
            except BaseException as e:
                batch_error = e
            if distributed_world_size > 1:
                self._raise_distributed_failure(
                    batch_error is not None,
                    "fetching training batch",
                    batch_error,
                )
            elif batch_error is not None:
                raise batch_error

            do_update = (accum_progress + 1 >= grad_accum)
            # HF forces a sync step on an epoch's last micro-batch, so the epoch is
            # fully applied before on_epoch_end and its tail never mixes into the
            # next window. It does this under max_steps too, but only an exact
            # one-pass length qualifies; the dataset-size approximation leaves
            # _epoch_flush_microbatches None and must not move steps. The run's last
            # authorized micro-batch closes a possibly-partial final epoch, so it
            # forces the update too: without it a ragged tail waits for a window
            # that never fills and pulls a row num_train_epochs never authorized.
            if _epoch_flush_microbatches and (
                it % _epoch_flush_microbatches == 0
                or it == _epoch_stop_total_microbatches
            ):
                do_update = True
            if do_update:
                # Keep callable scheduler evaluation outside mx.compile. The
                # compiled step reads the scalar LR already in optimizer state.
                self._set_optimizer_lr_for_step(optimizer, self._global_step)
                # HF fires this between clipping and optimizer.step(). MLX fuses
                # both into step_fn, so here is the last point with the update
                # un-applied. Nothing observable is lost: MLX parameters have no
                # .grad and the gradient pytree is local to the compiled function,
                # so HF's "monitor gradients" use is unreachable, while an LR
                # override still lands. As below, do NOT latch a callback stop here.
                _fire("on_pre_optimizer_step")
                self._distributed_should_stop()

            if _ddp_update_outside_step:
                lvalue, toks, grad_accum_state, grad_norm = _run_ddp_local_step(
                    batch_data, grad_accum_state, do_update,
                )
                if do_update:
                    grad, toks_f = grad_accum_state
                    grad_norm = _apply_update(grad, toks_f)
                    grad_accum_state = None
            else:
                # Compiled full step threads mx.random.state through its outputs;
                # snapshot it so an eager retry after a trace-time failure resumes
                # from the pre-call RNG (mirrors the DDP local-grad path). Guard on
                # the list form so the torch-sim test shim (callable state) is a no-op.
                rng_state_before = None
                _rng_state = mx.random.state
                if (
                    _use_compile
                    and not _ddp_compile_local_grad
                    and isinstance(_rng_state, list)
                    and _rng_state
                ):
                    rng_state_before = mx.array(
                        _rng_state[0].tolist(), dtype=mx.uint32,
                    )
                try:
                    lvalue, toks, grad_accum_state, grad_norm = step_fn(
                        batch_data, grad_accum_state, do_update,
                    )
                except (ValueError, RuntimeError, TypeError) as e:
                    _is_compile_failure = (
                        _use_compile
                        and not _ddp_compile_local_grad
                        and _is_compile_exception(e)
                    )
                    if _is_compile_failure:
                        if not _compile_fallback_allowed():
                            _strict_compile_error(e)
                        _main_print(
                            "Unsloth: mx.compile failed at runtime; "
                            "falling back to eager mode."
                        )
                        step_fn = _uncompiled_step_fn
                        _use_compile = False
                        _compile_scope = "fallback_eager"
                        _compile_fallback_reason = "runtime_error"
                        if isinstance(batches, _EAGER_REFETCHABLE_PLAN_TYPES):
                            batch_data = batches[scheduled_index]
                        if rng_state_before is not None:
                            mx.random.state[0] = rng_state_before
                        state = [model.state, optimizer.state, mx.random.state]
                        lvalue, toks, grad_accum_state, grad_norm = step_fn(
                            batch_data, grad_accum_state, do_update,
                        )
                    else:
                        raise

            # Accumulate into the PENDING window (this accumulation window's
            # micro-batches). Fold into the COMMITTED window only once the
            # optimizer step for the window has been applied (do_update), so a
            # forced log or the post-loop flush never reports a not-yet-applied
            # partial window, matching HF (a partial window is never logged and its
            # loss is folded into tr_loss only after the optimizer step).
            pending_losses += lvalue * toks
            pending_n_tokens += toks
            pending_steps += 1
            if do_update:
                # Window applied: fold pending into committed and reset pending.
                # Evaluating the committed accumulators here materializes the folded
                # pending contribution, so pending (now 0) needs no separate eval.
                losses += pending_losses
                n_tokens += pending_n_tokens
                steps += pending_steps
                pending_losses = 0
                pending_n_tokens = 0
                pending_steps = 0
                _metric_eval = (losses, n_tokens)
            else:
                # Substep: only the pending window changed; committed is unchanged
                # (already materialized at its last fold). Both are always arrays at
                # this point, so mx.eval never sees a plain-int accumulator.
                _metric_eval = (pending_losses, pending_n_tokens)
            # One evaluation boundary: the reported norm (when present) is
            # evaluated together with model/optimizer state and metric
            # accumulators, never as a separate earlier graph execution.
            eval_targets = [state, *_metric_eval]
            if grad_accum_state is not None:
                eval_targets.append(grad_accum_state[0])
                eval_targets.append(grad_accum_state[1])
            if grad_norm is not None:
                eval_targets.append(grad_norm)
            mx.eval(*eval_targets)
            global_toks = self._distributed_all_sum(toks, stream=mx.cpu)
            mx.eval(global_toks)
            if int(global_toks.item()) == 0:
                raise ValueError(
                    "Unsloth MLX: a training batch produced zero supervised "
                    "tokens after masking/truncation. Increase max_seq_length, "
                    "reduce image size, or check the chat template / labels."
                )
            # Global INPUT-token count for HF's num_input_tokens_seen, only when the
            # run opted in (track_input_tokens). global_toks above is the loss mask's
            # supervised-token count (used for the zero-token guard), not the
            # input-token count HF's field reports, so counting it would undercount
            # prompts and masked tokens. Sum the batch input positions the selected
            # mode counts and all-reduce that (same global gather semantics as HF
            # and as global_toks). The gate and the mode are rank-uniform config,
            # so every rank skips or runs it together.
            if track_input_tokens:
                global_input_toks = self._distributed_all_sum(
                    mx.array(
                        _mlx_batch_input_token_count(
                            batch_data,
                            mode=input_token_mode,
                            pad_token_id=input_token_pad_id,
                        ), dtype=mx.int32,
                    ),
                    stream=mx.cpu,
                )
                mx.eval(global_input_toks)
                # HF's num_input_tokens_seen is an all-rank count of INPUT tokens,
                # read directly by token-budget callbacks. Use the all-reduced input
                # count, not global_toks (label tokens) and not the rank-local value
                # (undercounts by ~world_size). Incremented BEFORE on_optimizer_step,
                # as HF advances it right after the forward, so a token-budget
                # callback sees this microbatch at the step it fires on.
                self.state.num_input_tokens_seen += int(global_input_toks.item())
            if do_update:
                _fire("on_optimizer_step")
                # Do NOT latch a callback should_training_stop here. HF runs
                # on_optimizer_step -> on_step_end -> _maybe_log_save_evaluate for
                # this step and only breaks after that block, so latching now would
                # make this step's _evaluate_batch_totals skip every eval batch and
                # report 0.0, corrupting best-model / early-stopping state.
                # OR-reduce only an external cancel; the tail _sync_stop() applies
                # the callback stop after the same-step actions.
                self._distributed_should_stop()
            # Charge this micro-batch to the PENDING window; it folds into
            # COMMITTED on an applied update, so train_time covers exactly the
            # micro-batches whose tokens are in n_tokens.
            pending_time += time.perf_counter() - tic
            if do_update:
                train_time += pending_time
                pending_time = 0

            # Only log/eval on actual optimizer steps
            if not do_update:
                accum_progress += 1
                _fire("on_substep_end")
                # Do NOT latch a callback should_training_stop yet. If this
                # non-update microstep is also an epoch boundary, on_epoch_end below
                # may run an eval, and latching now would make it skip every batch
                # and report 0.0, corrupting best-model / early-stopping state.
                # OR-reduce only an external cancel; the tail _sync_stop() applies
                # the callback stop after the epoch-end actions, as in the update
                # branch.
                self._distributed_should_stop()
                # An epoch boundary can fall on a non-update microstep when
                # batches-per-epoch is not a multiple of grad_accum (for example
                # 3 micro-batches with grad_accum=2). HF always fires on_epoch_end
                # once per epoch (its final batch forces an optimizer step), so
                # fire it here too; otherwise on_epoch_end and any log/eval/save/
                # stop it requests are dropped for that epoch. _maybe_callback_
                # epoch_end is a no-op away from a boundary and runs the same
                # collectives on every rank (it/batches_per_epoch are rank-
                # consistent), so DDP stays in lockstep.
                _maybe_callback_epoch_end(it, self._global_step, grad_norm)
                # Honor should_epoch_stop from on_substep_end. HF breaks right
                # after on_substep_end, before any deferred optimizer step, so the
                # partial window is abandoned rather than applied with the ended
                # epoch's tail (and, when batches_per_epoch is not a multiple of
                # grad_accum, wrapped next-epoch batches). Needs a known epoch
                # length; a length-less stream leaves batches_per_epoch None.
                # Gated on the all-reduced flag so every rank skips the same
                # micro-batches. A natural boundary already fired on_epoch_end
                # above, so only the mid-epoch skip fires it.
                if batches_per_epoch and _sync_epoch_stop():
                    if it % batches_per_epoch != 0:
                        # Mid-epoch: abandon the partial accumulation window like
                        # HF's mid-window break, then skip the epoch's remaining
                        # micro-batches. Only here -- at a boundary substep
                        # (it % batches_per_epoch == 0) there is no tail to skip and
                        # the normal loop carries this micro-batch's gradient into
                        # the next accumulation window, so discarding it would drop
                        # the epoch's final batch from the optimizer update while
                        # its loss/tokens were already counted.
                        grad_accum_state = None      # abandon the partial window
                        accum_progress = 0
                        # The abandoned micro-batches never updated the model, so
                        # drop ONLY the PENDING window, mirroring the discarded
                        # gradient; reporting un-applied data would misstate the
                        # logged loss and throughput. The COMMITTED window survives
                        # so a truncated epoch-end log still reports the completed
                        # update, as HF does.
                        pending_losses = 0
                        pending_n_tokens = 0
                        pending_steps = 0
                        # Drop the abandoned window's time with its tokens, else
                        # the next window's tokens/s is deflated by it.
                        pending_time = 0
                        it = _honor_epoch_stop_skip(
                            it, self._global_step, grad_norm)
                    self.control.should_epoch_stop = False
                # Apply any deferred callback stop (from on_substep_end or the
                # epoch-end callbacks) now that the epoch-end eval has run. This
                # tail _sync_stop() runs on every rank in the same order as the
                # update branch's tail, so DDP stays in lockstep; the continue then
                # routes to the top-of-loop _distributed_should_stop(), which drains
                # the stop on every rank before the next collective.
                _sync_stop()
                microstep = it
                continue

            current_step = self._global_step + 1
            self._global_step = current_step
            self.state.global_step = current_step
            accum_progress = 0
            # Advance the callback epoch only on an optimizer step, beside the
            # global_step it belongs to and just before on_step_end -- HF's
            # `state.global_step += 1; state.epoch = epoch + (step+1)/
            # steps_in_epoch; on_step_end` (transformers _inner_training_loop).
            # Updating it on every micro-batch instead left on_substep_end
            # reporting an epoch a micro-batch ahead of the last completed step.
            # Epoch boundaries are unaffected: _maybe_callback_epoch_end and the
            # truncated-epoch closes set the epoch themselves before firing.
            if epoch_event_microbatches:
                self.state.epoch = it / epoch_event_microbatches
            # The loop's own step cadence for the two requests the static
            # interval below cannot express -- logging_first_step and the
            # final-step force -- raised where HF's flow raises them and folded
            # into the same control flags, so an installed flow coalesces.
            self._request_step_cadence_actions()
            _fire("on_step_end")
            # on_step_end may request log/eval/save or a stop, and a rank-dependent
            # callback can do so on a subset. Sync those decisions before the
            # collective log/eval/save paths so every rank makes the same choice.
            self._distributed_sync_control_actions()
            # Do NOT copy a callback should_training_stop into stop_requested yet.
            # HF runs this step's log/evaluate/save before the loop breaks, so a
            # stop requested on on_step_end must not pre-empt a same-step eval:
            # _evaluate_batch_totals skips every eval batch while stop_requested
            # is set, which would report 0.0 loss and corrupt best-model /
            # early-stopping state. Only OR-reduce any external cancel here (a
            # rank-consistent hard stop); the callback stop is applied after the
            # same-step actions by the tail _sync_stop() below.
            self._distributed_should_stop()

            # Logging. Cadences come from self.state as HF-resolved absolute
            # counts, each guarded against 0; the synced control flag can also
            # force a log. _run_training_log all-reduces the totals.
            logging_steps = int(getattr(self.state, "logging_steps", 0) or 0)
            eval_steps = int(getattr(self.state, "eval_steps", 0) or 0)
            save_steps = int(getattr(self.state, "save_steps", 0) or 0)
            # The static interval mirrors DefaultFlowCallback's step-strategy
            # rule (strategy is STEPS, on a multiple of logging_steps); a
            # caller-supplied "no" must not log and "epoch" must leave the
            # cadence to on_epoch_end rather than logging on both. An explicit
            # callback request stays independent of the strategy, as does the
            # final-step flush that folds the run's last window into the
            # returned train_loss.
            should_log = (
                (
                    logging_steps > 0
                    and current_step % logging_steps == 0
                    and self._static_log_cadence_enabled()
                )
                or current_step == total_steps
                or self.control.should_log
            )
            if should_log:
                _run_training_log(current_step, grad_norm)

            # Eval (cadence or a synced callback request). _run_eval builds eval
            # batches lazily on every rank, runs the collective eval, then fires
            # on_evaluate on rank 0 and syncs any stop before best tracking.
            # The static cadence mirrors DefaultFlowCallback's step-strategy
            # rule (strategy is STEPS, on a multiple of eval_steps, past
            # eval_delay); an explicit callback request stays independent of
            # the strategy, exactly as HF honors control.should_evaluate
            # whoever raised it.
            should_eval = (
                self.eval_dataset is not None
                and (
                    (
                        eval_steps > 0
                        and current_step % eval_steps == 0
                        and self._static_eval_cadence_enabled()
                        and self._eval_delay_satisfied(current_step)
                    )
                    or self.control.should_evaluate
                )
            )
            if should_eval:
                if _run_eval(current_step):
                    _run_best_tracking(current_step)

            # Checkpointing (cadence or a synced callback request). _run_checkpoint
            # writes on rank 0 and syncs failures. The static cadence mirrors
            # DefaultFlowCallback's step-strategy rule: "no" must write no step
            # checkpoints even though save_steps keeps its default 500, and "epoch"
            # must leave the cadence to on_epoch_end rather than write checkpoint-N
            # twice per boundary. A callback request stays independent of the
            # strategy, which is also how the final-step checkpoint arrives.
            should_save = (
                (
                    save_steps > 0
                    and current_step % save_steps == 0
                    and self._static_save_cadence_enabled()
                )
                or self.control.should_save
            )
            if should_save:
                _run_checkpoint(current_step)

            _maybe_callback_epoch_end(it, current_step, grad_norm)
            # Honor control.should_epoch_stop, mirroring HF's `if should_epoch_stop:
            # break`: end this epoch and skip its remaining micro-batches so the
            # next iteration starts fresh. Applied at a clean optimizer-step
            # boundary, so no partial accumulation is abandoned. Needs a known epoch
            # length; a length-less stream fires no epoch events at all. The skip is
            # rank-consistent arithmetic gated on _sync_epoch_stop. A natural
            # boundary already fired on_epoch_end, so only skip when mid-epoch.
            if batches_per_epoch and _sync_epoch_stop():
                if it % batches_per_epoch != 0:
                    it = _honor_epoch_stop_skip(it, current_step, grad_norm)
                self.control.should_epoch_stop = False
            # Advance the completed-micro-batch counter before the stop check so a
            # callback-stop break still leaves `microstep` pointing at the step
            # just finished; the post-loop truncated-epoch on_epoch_end reads it to
            # decide whether the final epoch was left open.
            microstep = it
            # Propagate any stop set by the tail callbacks (on_step_end / on_log /
            # on_evaluate / on_save / on_epoch_end) to every rank before breaking,
            # so no rank is left waiting at the next collective.
            if _sync_stop():
                break

        # Close a truncated final epoch. Leaving off mid-epoch (max_steps off a
        # dataset boundary, or a should_training_stop break) means the natural
        # boundary never fired on_epoch_end, and HF fires it after its inner loop
        # breaks, so mirror that before the final flush and on_train_end. The guard
        # prevents a double fire, since a natural boundary and the should_epoch_stop
        # skip both leave microstep on a boundary. Both quantities are
        # rank-consistent, so the event and its synced actions stay in lockstep.
        # Streaming runs close their final epoch here on the same terms.
        if (epoch_event_microbatches
                and microstep % epoch_event_microbatches != 0):
            self.state.epoch = microstep / epoch_event_microbatches
            # Same for the truncated final epoch: HF's on_epoch_end after the
            # inner loop breaks feeds _maybe_log_save_evaluate, so an "epoch"
            # strategy still gets this boundary's action.
            self._request_epoch_cadence_actions()
            _fire("on_epoch_end")
            self._distributed_sync_control_actions()
            if (self.control.should_log or self.control.should_evaluate
                    or self.control.should_save):
                # A mid-epoch callback stop has already latched stop_requested, so
                # an epoch-end eval would hit _evaluate_batch_totals' gate, skip
                # every batch and dispatch a phantom 0.0 eval that corrupts
                # best-model / early-stopping state. HF runs a real epoch-end eval
                # for a callback stop, so lift it around these actions and restore
                # it after. A hard external cancel keeps its suppression and skips
                # the actions entirely. OR-reduced because should_training_stop is
                # rank-dependent; the synced flags above already put every rank
                # here, so the collective stays in lockstep.
                _callback_stop = self._distributed_any_flag(
                    getattr(self.control, "should_training_stop", False))
                if not self.stop_requested or _callback_stop:
                    _restore_stop = self.stop_requested
                    self.stop_requested = False
                    try:
                        _run_callback_control_actions(self._global_step, None)
                    finally:
                        self.stop_requested = _restore_stop

        # Flush a completed-but-unlogged window so the returned train_loss covers
        # steps that stopping between log points would otherwise drop, as HF folds
        # trailing tr_loss into _total_loss_scalar. Gated on the COMMITTED window:
        # a pending partial window never reached an optimizer update and lives in
        # pending_*, so only applied steps are reported. steps advances identically
        # on every rank, and anything that already logged reset it to 0, so this is
        # rank-consistent and never double-counts.
        if steps > 0:
            _run_training_log(self._global_step, None)

        # Token-weighted, so the returned loss is cadence-free like HF's
        # (_total_loss_scalar / effective_global_step). Falls back to the plain
        # window mean when the weights are unavailable (legacy resume, 0 tokens).
        if self._train_loss_weighting_ok and self._train_loss_token_total > 0:
            avg_loss = self._train_loss_token_sum / self._train_loss_token_total
        else:
            avg_loss = (
                sum(self._train_loss_history) / len(self._train_loss_history)
                if self._train_loss_history else 0.0
            )
        # Total wall-clock training time, consumed by the summary line and the
        # distributed diagnostics / train_runtime metrics below.
        total_time = time.perf_counter() - start_time

        # Report the step actually reached, which is < total_steps after an
        # early stop (self._global_step == total_steps on a full run).
        completed_steps = self._global_step

        _main_print(
            f"\nUnsloth: Training complete! "
            f"Avg loss: {avg_loss:.4f} | "
            f"Total time: {total_time:.1f}s | "
            f"Steps: {completed_steps} | "
            f"Tokens: {trained_tokens}"
        )

        # load_best_model_at_end: restore best adapters before the final save.
        restore_abort = None
        if getattr(args, "load_best_model_at_end", False) and self._best_step is not None:
            _best_path = f"{args.output_dir}/best/adapters.safetensors"
            if os.path.exists(_best_path):
                try:
                    model.load_weights(_best_path, strict=False)
                    _main_print(
                        f"Unsloth: Restored best model from step {self._best_step} "
                        f"({self._resolved_best_metric_name()}={self._best_metric:.4f})."
                    )
                except Exception as e:
                    _main_print(f"Unsloth: failed to restore best model ({e}).")
                except BaseException as e:
                    # Ordinary restore failures log and continue, but an
                    # interrupt must reach the consensus below before the
                    # diagnostics collective or peers hang in it.
                    restore_abort = e
        self._raise_distributed_failure(
            restore_abort is not None,
            "best-model restore",
            restore_abort,
        )

        distributed_diagnostics = self._distributed_training_diagnostics(
            total_time=total_time,
            trained_tokens=trained_tokens,
            compile_scope=_compile_scope,
            compile_fallback_reason=_compile_fallback_reason,
        )

        # Honor the documented save_steps=0 contract: save at end of training.
        final_save_error = None
        if is_main_process:
            try:
                self._quiesce_prefetcher_for_save(terminal=True)
                self.save_model()
            except ValueError as e:
                _main_print(f"Unsloth: skipped final save ({e})")
            except BaseException as e:
                final_save_error = e
            else:
                _main_print(f"Unsloth: Saved final adapters to {args.output_dir}")
        self._raise_distributed_failure(
            final_save_error is not None,
            "final save",
            final_save_error,
        )

        try:
            _rows_per_pass = len(self._train_dataset_for_batches())
        except Exception:
            _rows_per_pass = 0
        _train_samples = _rows_per_pass * float(self.state.epoch or 0)
        # The run's aggregate metrics, dispatched below and returned unchanged so
        # a caller reading MLXTrainOutput and a callback reading on_log agree.
        final_metrics = {
            "train_loss": avg_loss,
            "train_runtime": total_time,
            "train_steps": completed_steps,
            "total_train_steps": total_steps,
            "trained_tokens": trained_tokens,
            # Two rates, not one. This payload now reaches HF integrations, and
            # train_samples_per_second is their standard sample-throughput key, so
            # publishing the token rate under it recorded a length-dependent number
            # as samples/s. Samples follow HF's own approximation in speed_metrics
            # (rows in one pass times the epochs run); the token rate keeps its own
            # key, as HF does when num_tokens is available.
            "train_samples_per_second": (
                _train_samples / total_time if total_time > 0 else 0
            ),
            "train_tokens_per_second": (
                trained_tokens / total_time if total_time > 0 else 0
            ),
        }
        # HF logs these through Trainer.log immediately before on_train_end
        # (trainer.py _finalize_training on 5.x, the tail of
        # _inner_training_loop on 4.x), so they land in state.log_history and
        # reach on_log: that is how WandbCallback promotes train_loss /
        # train_runtime into the run summary and how a resumed run's history
        # keeps the previous run's totals. Unconditional there and here -- it is
        # not a DefaultFlowCallback cadence, so logging_strategy="no" still gets
        # this one payload. Every rank appends and fires, like _run_training_log.
        _final_log = dict(final_metrics)
        if self.state.epoch is not None:
            _final_log["epoch"] = self.state.epoch
        _final_record = dict(_final_log)
        _final_record["step"] = self.state.global_step
        self.state.log_history.append(_final_record)
        _fire("on_log", logs=_final_log)

        # HF's WandbCallback and DVCLiveCallback log their final-model artifact by
        # constructing a Torch Trainer around this (MLX) model, which raises
        # AttributeError and would throw away the finished run's result. Skip just
        # that artifact, keep the rest of on_train_end (DVCLive's live.end() trails
        # it), restore the user's callbacks afterwards.
        _final_artifact_modes = self._suppress_torch_only_final_artifacts()
        if _final_artifact_modes:
            _suppressed_names = sorted(
                {type(callback).__name__ for callback, _ in _final_artifact_modes}
            )
            _main_print(
                f"Unsloth: {', '.join(_suppressed_names)} final-model artifacts "
                "need a Torch Trainer and a torch.nn.Module, so they are skipped "
                "for MLX runs. Adapters were still saved to "
                f"{args.output_dir}; per-checkpoint artifacts are unaffected."
            )
        try:
            _fire("on_train_end")
        finally:
            self._restore_final_artifact_modes(_final_artifact_modes)
        _sync_stop()

        return MLXTrainOutput({
            **final_metrics,
            "compile_enabled": bool(_use_compile),
            "compile_support_state": (
                _compile_decision.support_state if _compile_decision is not None else "n/a"
            ),
            "compile_reason": (
                _compile_decision.reason if _compile_decision is not None else ""
            ),
            "compile_policy_mode": (
                _compile_decision.policy_mode if _compile_decision is not None else compile_policy.mode
            ),
            "compile_scope": _compile_scope,
            "compile_shape_guard": _compile_shape_guard_report.to_dict(),
            "patch_mode": getattr(self.args, "patch_mode", "patched"),
            "compile_trace": (
                asdict(self._compile_trace)
                if is_dataclass(getattr(self, "_compile_trace", None))
                else getattr(self, "_compile_trace", None)
            ),
            "compile_auto_tune_applied": list(getattr(self, "_compile_auto_tune_applied", [])),
            "memory_limits_applied": dict(getattr(self, "_memory_limits_applied", {})),
            "base_quantization_config": getattr(
                self.model, "_unsloth_quantization_config", None,
            ),
            "base_quantization_policy": getattr(
                self.model, "_unsloth_quantization_policy", None,
            ),
            "base_quantized_source": getattr(
                self.model, "_unsloth_quantized_source", None,
            ),
            **distributed_diagnostics,
            **self._distributed_result_fields(),
        })

    def _resolve_vlm_processor(self):
        """Resolve the processor used for VLM collation without mutating model."""
        args = self.args
        config = getattr(self.model, "_config", {})
        model_type = config.get("model_type") if isinstance(config, dict) else None
        model_name = getattr(self.model, "_hf_repo", None)

        processor = self.processor
        if processor is None and (
            hasattr(self.tokenizer, "image_processor")
            or (
                hasattr(self.tokenizer, "tokenizer")
                and hasattr(self.tokenizer, "apply_chat_template")
            )
        ):
            processor = self.tokenizer
        if processor is None:
            processor = getattr(self.model, "_processor", None)
        if processor is None:
            raise ValueError(
                "VLM training requires a processor. Pass processor= to MLXTrainer "
                "or load the model with FastLanguageModel.from_pretrained()."
            )

        processor = normalize_vlm_processor_chat_template(
            processor,
            chat_template=getattr(args, "vlm_chat_template", None),
            model_name=model_name,
            model_type=model_type,
            strict=False,
        )
        self.processor = processor
        return processor

    def _prepare_data(self, is_vlm, defer_vlm_checker=False):
        """Prepare training data. Returns (batches, batch_iter)."""
        args = self.args
        self._streaming_epoch_batch_count = None
        train_dataset = self._train_dataset_for_batches()
        config = getattr(self.model, "_config", {})
        model_type = config.get("model_type") if isinstance(config, dict) else None
        model_name = getattr(self.model, "_hf_repo", None)

        if is_vlm:
            processor = self._resolve_vlm_processor()
        else:
            self.tokenizer = normalize_mlx_chat_template(
                self.tokenizer,
                chat_template=getattr(args, "chat_template", None),
                model_name=model_name,
                model_type=model_type,
                is_vlm=False,
                strict=False,
            )

        if self._batches is not None:
            return self._batches, None

        total_batches_needed = (
            args.max_steps * args.gradient_accumulation_steps
            if args.max_steps > 0 else None
        )
        text_completion_only_loss = _text_completion_only_loss_arg(args)
        text_assistant_only_loss = _text_assistant_only_loss_arg(args)
        comm_group = self.distributed_world

        if is_vlm:
            if text_assistant_only_loss:
                raise ValueError(
                    "Unsloth MLX VLM: assistant_only_loss=True is not supported for "
                    "vision-language models. Set assistant_only_loss=False, or use "
                    "train_on_responses_only for response masking."
                )
            _vlm_mask_fn = getattr(self, '_vlm_response_mask_fn', None)
            vlm_dataset_order = (
                "sequential"
                if getattr(args, "preserve_dataset_order", False)
                else getattr(args, "dataset_order", "default")
            )
            vlm_num_epochs = (
                args.num_train_epochs
                if (
                    args.max_steps <= 0
                    and args.num_train_epochs > 0
                    and vlm_dataset_order == "torch_randperm"
                )
                else None
            )
            if args.streaming:
                vlm_prefetch_depth = _validate_streaming_prefetch(
                    getattr(args, "streaming_prefetch_batches", 0)
                )
                if vlm_prefetch_depth and self.distributed_world_size > 1:
                    if not getattr(self, "_mlx_prefetch_ddp_notice", False):
                        self._mlx_prefetch_ddp_notice = True
                        if getattr(self, "_distributed_is_main_process", True):
                            print(
                                "Unsloth: streaming_prefetch_batches is "
                                "single-process only; continuing "
                                "synchronously under DDP."
                            )
                    vlm_prefetch_depth = 0
                vlm_lazy = not _vlm_has_sized_index_space(train_dataset)
                if vlm_lazy and self.distributed_world_size > 1:
                    raise ValueError(
                        "Unsloth MLX VLM: DDP training with an unsized "
                        "streaming VLM source is not supported (every rank "
                        "would re-consume the global stream). Use a sized "
                        "dataset or single-process training; rank-owned lazy "
                        "VLM dispatch is a planned follow-up."
                    )
                vlm_require_replayable = bool(
                    getattr(self, "_resume_from_checkpoint", None)
                )
                vlm_expected_rows = None
                if vlm_lazy and args.max_steps <= 0 and args.num_train_epochs > 0:
                    declared = _mlx_declared_iterable_length(train_dataset)
                    if declared is None:
                        raise ValueError(
                            "Unsloth MLX VLM: num_train_epochs requires a "
                            "streaming iterable with an explicit reliable "
                            "__len__. Use max_steps for an unsized source."
                        )
                    if declared == 0:
                        raise ValueError(
                            "Unsloth MLX VLM: streaming iterable declares zero rows."
                        )
                    self._streaming_epoch_batch_count = math.ceil(
                        declared / args.per_device_train_batch_size
                    )
                    if (
                        self._streaming_epoch_batch_count
                        % args.gradient_accumulation_steps
                    ):
                        raise ValueError(
                            "Unsloth MLX: streaming num_train_epochs requires "
                            "the total epoch micro-batches to be divisible by "
                            "gradient_accumulation_steps. Use max_steps or "
                            "adjust the accumulation factor."
                        )
                    vlm_expected_rows = declared
                    vlm_require_replayable = True
                self._mlx_prefetch_control = {
                    "eligible": bool(vlm_prefetch_depth and vlm_lazy),
                }
                vlm_resume_skip = (
                    int(getattr(self, "_mlx_resume_step_for_prefetch", 0))
                    * args.gradient_accumulation_steps
                )
                return None, iterate_vlm_training_batches(
                    dataset=train_dataset,
                    processor=processor,
                    config=config,
                    batch_size=args.per_device_train_batch_size,
                    max_seq_length=args.max_seq_length,
                    image_size=getattr(args, "image_size", None),
                    seed=args.seed,
                    response_mask_fn=_vlm_mask_fn,
                    formatting_func=self.formatting_func,
                    dataset_order=vlm_dataset_order,
                    completion_only_loss=text_completion_only_loss,
                    comm_group=comm_group,
                    require_replayable=vlm_require_replayable,
                    expected_rows_per_pass=vlm_expected_rows,
                    prefetch_batches=vlm_prefetch_depth if vlm_lazy else 0,
                    prefetch_skip_batches=(
                        vlm_resume_skip
                        if vlm_prefetch_depth and vlm_lazy else 0
                    ),
                    prefetch_control=self._mlx_prefetch_control,
                )
            else:
                self._prepared_batches_include_epochs = vlm_num_epochs is not None
                plan = _create_vlm_batch_plan(
                    dataset=train_dataset,
                    processor=processor,
                    config=config,
                    batch_size=args.per_device_train_batch_size,
                    max_seq_length=args.max_seq_length,
                    image_size=getattr(args, "image_size", None),
                    num_batches=total_batches_needed,
                    seed=args.seed,
                    response_mask_fn=_vlm_mask_fn,
                    formatting_func=self.formatting_func,
                    dataset_order=vlm_dataset_order,
                    num_epochs=vlm_num_epochs,
                    completion_only_loss=text_completion_only_loss,
                    comm_group=comm_group,
                )
                batches = [] if plan is None else plan
                run_checker = _vlm_mask_fn is not None and len(batches) > 0
                if defer_vlm_checker:
                    # The coordinated preflight owns the collective schedule, so
                    # rank-local preparation must stay collective-free. Stash the
                    # pending check for the preflight to invoke.
                    self._deferred_vlm_all_masked_check = (
                        (lambda: _check_vlm_all_masked(
                            batches,
                            comm_group=comm_group,
                            world_size=self.distributed_world_size,
                        ))
                        if run_checker else None
                    )
                    return batches, None
                if run_checker:
                    _check_vlm_all_masked(
                        batches,
                        comm_group=comm_group,
                        world_size=self.distributed_world_size,
                    )
                return batches, None
        else:
            chat_tmpl = getattr(args, "chat_template", None)
            if args.streaming:
                text_dataset_order = (
                    "sequential"
                    if getattr(args, "preserve_dataset_order", False)
                    else getattr(args, "dataset_order", "default")
                )
                expected_rows_per_pass = None
                require_replayable = bool(
                    getattr(self, "_resume_from_checkpoint", None)
                )
                response_mask_fn = getattr(
                    train_dataset, "_response_mask_fn", None,
                ) or getattr(self, "_mlx_response_mask_fn", None)
                if (
                    _is_mlx_lazy_text_source(train_dataset)
                    and args.max_steps <= 0
                    and args.num_train_epochs > 0
                ):
                    def _resolve_source_length():
                        length = _mlx_declared_iterable_length(train_dataset)
                        return -1 if length is None else length

                    source_length = _mlx_rank0_resolve_int(
                        comm_group,
                        _resolve_source_length,
                        "resolving the streaming text source length",
                    )
                    if source_length < 0:
                        raise ValueError(
                            "Unsloth MLX: num_train_epochs requires a streaming "
                            "text iterable with an explicit reliable __len__. Use "
                            "max_steps for an unsized source."
                        )
                    if source_length == 0:
                        raise ValueError(
                            "Unsloth MLX: streaming text iterable declares zero rows."
                        )
                    global_batch_size = (
                        args.per_device_train_batch_size
                        * self.distributed_world_size
                    )
                    self._streaming_epoch_batch_count = math.ceil(
                        source_length / global_batch_size
                    )
                    if (
                        self._streaming_epoch_batch_count
                        % args.gradient_accumulation_steps
                    ):
                        raise ValueError(
                            "Unsloth MLX: streaming num_train_epochs requires "
                            "the total epoch micro-batches to be divisible by "
                            "gradient_accumulation_steps. Use max_steps or "
                            "adjust the accumulation factor."
                        )
                    expected_rows_per_pass = source_length
                    require_replayable = True
                prefetch_depth = _validate_streaming_prefetch(
                    getattr(args, "streaming_prefetch_batches", 0)
                )
                if prefetch_depth and self.distributed_world_size > 1:
                    if not getattr(self, "_mlx_prefetch_ddp_notice", False):
                        self._mlx_prefetch_ddp_notice = True
                        if getattr(self, "_distributed_is_main_process", True):
                            print(
                                "Unsloth: streaming_prefetch_batches is "
                                "single-process only; continuing "
                                "synchronously under DDP."
                            )
                    prefetch_depth = 0
                self._mlx_prefetch_control = {
                    "eligible": bool(
                        prefetch_depth
                        and _is_mlx_lazy_text_source(train_dataset)
                    ),
                }
                resume_skip = (
                    int(getattr(self, "_mlx_resume_step_for_prefetch", 0))
                    * args.gradient_accumulation_steps
                )
                return None, iterate_training_batches(
                    dataset=train_dataset,
                    tokenizer=self.tokenizer,
                    batch_size=args.per_device_train_batch_size,
                    max_seq_length=args.max_seq_length,
                    seed=args.seed,
                    dataset_text_field=args.dataset_text_field,
                    formatting_func=self.formatting_func,
                    chat_template=chat_tmpl,
                    model_name=model_name,
                    model_type=model_type,
                    append_eos=bool(getattr(args, "append_eos", True)),
                    completion_only_loss=text_completion_only_loss,
                    assistant_only_loss=text_assistant_only_loss,
                    response_mask_fn=response_mask_fn,
                    dataset_order=text_dataset_order,
                    comm_group=comm_group,
                    require_replayable=require_replayable,
                    expected_rows_per_pass=expected_rows_per_pass,
                    length_window_batches=_validate_streaming_length_window(
                        getattr(
                            args, "streaming_text_length_window_batches", 8,
                        )
                    ),
                    prefetch_batches=prefetch_depth,
                    prefetch_skip_batches=resume_skip if prefetch_depth else 0,
                    prefetch_control=self._mlx_prefetch_control,
                )
            else:
                batch_kwargs = dict(
                    dataset=train_dataset,
                    tokenizer=self.tokenizer,
                    batch_size=args.per_device_train_batch_size,
                    max_seq_length=args.max_seq_length,
                    num_batches=total_batches_needed,
                    seed=args.seed,
                    dataset_text_field=args.dataset_text_field,
                    formatting_func=self.formatting_func,
                    chat_template=chat_tmpl,
                    model_name=model_name,
                    model_type=model_type,
                    append_eos=bool(getattr(args, "append_eos", True)),
                    assistant_only_loss=text_assistant_only_loss,
                    comm_group=comm_group,
                )
                if (
                    getattr(args, "preserve_dataset_order", False)
                    or getattr(args, "dataset_order", "default") != "default"
                ):
                    text_dataset_order = (
                        "sequential"
                        if getattr(args, "preserve_dataset_order", False)
                        else getattr(args, "dataset_order", "default")
                    )
                    batch_kwargs["dataset_order"] = text_dataset_order
                    if (
                        args.max_steps <= 0
                        and args.num_train_epochs > 0
                        and text_dataset_order == "torch_randperm"
                    ):
                        batch_kwargs["num_epochs"] = args.num_train_epochs
                        # The builder quantizes a fractional epoch count to whole
                        # accumulation windows, as HF does, so it needs the factor.
                        batch_kwargs["grad_accum"] = (
                            args.gradient_accumulation_steps
                        )
                        self._prepared_batches_include_epochs = True
                    batch_kwargs["completion_only_loss"] = text_completion_only_loss
                    batches = _create_ordered_text_plan(**batch_kwargs)
                else:
                    batch_kwargs["completion_only_loss"] = text_completion_only_loss
                    batches = _create_text_batch_plan(**batch_kwargs)
                return batches, None

    def save_model(self, output_dir=None):
        """Save LoRA adapters or full merged model (if no LoRA)."""
        paused_prefetcher = self._quiesce_prefetcher_for_save()
        try:
            return self._save_model_impl(output_dir)
        finally:
            # A mid-training save pauses the producer for exclusivity and
            # resumes it; only end-of-training closes it terminally.
            if paused_prefetcher is not None:
                paused_prefetcher.resume()

    def _save_model_impl(self, output_dir=None):
        from .utils import (
            _coerce_mlx_lora_scale,
            _get_mlx_dropout_probability,
            _infer_mlx_lora_rank,
            save_merged_model,
        )
        output_dir = output_dir or self.args.output_dir

        # Detect LoRA from module structure so reloaded/frozen adapters
        # still take the adapter-save path.
        adapter_tensors = collect_mlx_lora_adapter_tensors(self.model)
        has_lora = bool(adapter_tensors)

        if has_lora:
            hf_repo = getattr(self.model, "_hf_repo", None) or ""


            # Infer rank/scale/dropout from the first reloadable module; leave
            # None on failure rather than persisting mis-scaling placeholders
            # (_enrich_mlx_adapter_config gets a second shot).
            _lora_rank = _lora_scale = _lora_dropout = None
            for _, m in iter_mlx_lora_modules(self.model):
                inferred_rank = _infer_mlx_lora_rank(m)
                if inferred_rank is None:
                    continue
                _lora_rank = inferred_rank
                # _coerce handles LoRASwitchLinear's per-expert mx.array where
                # raw float()/.item() raise.
                _lora_scale = _coerce_mlx_lora_scale(getattr(m, "scale", 1.0))
                _lora_dropout = _get_mlx_dropout_probability(
                    getattr(m, "dropout", None)
                )
                break

            from .utils import _get_transformer_layers
            layers = _get_transformer_layers(self.model)
            # mlx-lm.load_adapters() attr-accesses config.num_layers, so the
            # key MUST be present; -1 is the legacy "all layers" sentinel.
            try:
                _num_layers = len(layers) if layers is not None else -1
            except TypeError:
                _num_layers = -1
            if _num_layers <= 0:
                _num_layers = -1

            adapter_config = {
                "fine_tune_type": "lora",
                "peft_type": "LORA",
                "base_model_name_or_path": hf_repo,
                "learning_rate": self.args.learning_rate,
                "max_steps": self.args.max_steps,
                "max_seq_length": self.args.max_seq_length,
                "use_cce": self.args.use_cce,
                "base_quantization_config": getattr(
                    self.model, "_unsloth_quantization_config", None,
                ),
                "base_quantization_policy": getattr(
                    self.model, "_unsloth_quantization_policy", None,
                ),
                "base_quantized_source": getattr(
                    self.model, "_unsloth_quantized_source", None,
                ),
            }
            # Always emit num_layers for mlx-lm.load_adapters() attr-access.
            adapter_config["num_layers"] = _num_layers
            if _lora_rank is not None:
                adapter_config["lora_parameters"] = {
                    "rank": _lora_rank,
                    "scale": _lora_scale,
                    "dropout": _lora_dropout,
                }
                # mlx-vlm reads top-level rank/scale/dropout instead.
                adapter_config["rank"] = _lora_rank
                adapter_config["scale"] = _lora_scale
                adapter_config["dropout"] = _lora_dropout

            # Keep intentionally-trained non-LoRA tensors OUTSIDE any LoRA
            # module; drop wrapped base weights INSIDE one (else q_proj.weight
            # under a LoRA-wrapped q_proj re-leaks the Unsloth reload bug). Uses
            # the shared filter to match save_trainable_adapters / _merged.
            trainable = dict(tree_flatten(self.model.trainable_parameters()))
            adapter_keys = set(adapter_tensors)
            lora_module_prefixes = tuple(
                f"{name}." for name, _ in iter_mlx_lora_modules(self.model)
                if name
            )
            from .utils import _is_base_tensor_inside_lora_module
            has_root_lora_module = any(
                name == "" for name, _ in iter_mlx_lora_modules(self.model)
            )
            has_non_lora_trainable = any(
                key not in adapter_keys
                and not _is_base_tensor_inside_lora_module(
                    key, lora_module_prefixes, has_root_lora_module,
                )
                for key in trainable
            )
            if has_non_lora_trainable:
                save_trainable_adapters(
                    self.model, output_dir, adapter_config=adapter_config,
                )
            else:
                save_lora_adapters(
                    self.model, output_dir, adapter_config=adapter_config,
                )
            # VLM processors include the inner tokenizer; skip the separate
            # tokenizer save when the processor will cover it.
            _processor = self.processor or getattr(self.model, "_processor", None)
            _processor_saves_tokenizer = (
                _processor is not None and hasattr(_processor, "save_pretrained")
            )
            if not _processor_saves_tokenizer:
                self.tokenizer.save_pretrained(output_dir)

            # Copy base config.json so the checkpoint is loadable. Prefer the
            # mlx-vlm patched dir when materialized (e.g. DeepSeek OCR): _src_path
            # holds the original snapshot, whose unpatched model_type/auto_map
            # would break mlx-vlm routing on the saved adapter's reload.
            src_path = (
                getattr(self.model, "_config_src_path", None)
                or getattr(self.model, "_src_path", None)
            )
            if src_path is not None:
                import shutil
                from pathlib import Path
                src_config = Path(src_path) / "config.json"
                dst_config = Path(output_dir) / "config.json"
                if src_config.exists() and not dst_config.exists():
                    shutil.copy(str(src_config), str(dst_config))

            if _processor_saves_tokenizer:
                _processor.save_pretrained(output_dir)
            print(f"Unsloth: LoRA adapters saved to {output_dir}")
        else:
            save_merged_model(self.model, self.tokenizer, output_dir)


def _create_labeled_batches(dataset, tokenizer, mask_fn, batch_size,
                            max_seq_length, formatting_func=None,
                            dataset_text_field="text", num_batches=None,
                            seed=42, chat_template=None,
                            model_name=None, model_type=None,
                            append_eos=True, dataset_order="default",
                            preserve_dataset_order=False,
                            num_epochs=None, return_dataset=False,
                            comm_group=None, distributed_pad_mode="cycle",
                            return_plan=False):
    """Create padded batches with label masks for train_on_responses_only.

    Tokenizes each dataset item, applies the masking closure to get labels,
    sorts by length, and produces right-padded 3-tuple batches.

    Returns:
        List of (batch, lengths, labels) tuples where:
        - batch: mx.array (BS, padded_len) — input_ids padded with pad_token_id
        - lengths: mx.array of shape (BS, 2) holding [1, actual_len]
          per sequence. Right-half-open `[start, end)` matching the
          exclusive-end loss masks in `utils.py:360`, `:393`, `:429`,
          `:439`.
        - labels: mx.array (BS, padded_len) — labels padded with -100
    """
    eos_id = tokenizer.eos_token_id
    tokenizer = normalize_mlx_chat_template(
        tokenizer,
        chat_template=chat_template,
        model_name=model_name,
        model_type=model_type,
        is_vlm=False,
        strict=False,
    )
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_id = 0 if pad_id is None else int(pad_id)

    # 1. Gather all text strings (serial, fast)
    all_texts = []
    for item in dataset:
        if formatting_func is not None:
            result = formatting_func(item)
            texts = collect_mlx_texts(
                tokenizer, result, dataset_text_field=dataset_text_field,
                is_vlm=False,
            )
        else:
            texts = collect_mlx_texts(
                tokenizer, item, dataset_text_field=dataset_text_field,
                is_vlm=False,
            )

        for text in texts:
            if text:
                all_texts.append(text)

    # 2. Tokenize + mask in parallel (HF fast tokenizers are thread-safe).
    def _process_text(text):
        encoded = encode_mlx_text(tokenizer, text)
        # Mirror `_prepare_dataset`'s EOS contract; mismatch desyncs labeled vs unlabeled.
        if append_eos and eos_id is not None and (not encoded or encoded[-1] != eos_id):
            encoded.append(eos_id)
        if len(encoded) > max_seq_length:
            encoded = encoded[:max_seq_length]
        if len(encoded) < 2:
            return None
        result = mask_fn({"input_ids": [encoded]})
        labels = result["labels"]
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        return (encoded, labels[0])

    # Filter out samples where all labels are -100 (no valid training signal).
    # This can happen when truncation cuts off the response_part entirely,
    # e.g. long reasoning/analysis channels in GPT-OSS that exceed max_seq_length.
    # Such samples cause NaN loss since cross_entropy(mean) computes 0/0.
    def _has_valid_labels(labels):
        """Return whether a response-masked row still has trainable labels."""
        # Loss supervises labels[1:] (causal shift), so the first label never trains.
        return any(label != -100 for label in labels[1:])

    max_workers = min(4, os.cpu_count() or 1)
    all_items = []
    n_before_filter = 0
    n_removed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(_process_text, all_texts):
            if result is not None:
                n_before_filter += 1
                if _has_valid_labels(result[1]):
                    all_items.append(result)
                else:
                    n_removed += 1

    if n_removed > 0:
        print(
            f"Unsloth: Removed {n_removed} out of {n_before_filter} samples "
            f"from train_dataset where all labels were -100 "
            f"(no response found after truncation). "
            f"This prevents NaN loss during training."
        )

    if not all_items:
        raise ValueError(
            "No training data found after tokenization. "
            "Check your dataset and formatting_func."
        )

    # 2. Sample order; must agree with unlabeled `create_ordered_batches`
    # (utils.py:2845-2849) so `train_on_responses_only` sees the same stream.
    _order_requested = preserve_dataset_order or (
        dataset_order not in (None, "default")
    )
    if dataset_order not in (None, "default", "sequential", "torch_randperm"):
        raise ValueError(
            f"Unsloth MLX: unsupported dataset_order={dataset_order!r}. "
            "Expected one of: None, 'default', 'sequential', "
            "'torch_randperm'."
        )

    def _order_indices_for_epoch(epoch_idx):
        if preserve_dataset_order or dataset_order == "sequential":
            return list(range(len(all_items)))
        if dataset_order == "torch_randperm":
            from .utils import _torch_randperm_order, _normalize_seed
            # Reseed per epoch (matches `create_ordered_batches`). Normalize a
            # None seed first so seed=None does not raise on the int add.
            order = _torch_randperm_order(
                len(all_items), _normalize_seed(seed) + epoch_idx
            )
            return order
        # legacy default: length-sort once
        return sorted(range(len(all_items)), key=lambda i: len(all_items[i][0]))

    # 3. Build `num_epochs` blocks so `batches[i % len]` cycle reseeds correctly.
    _n_epochs_materialize = (
        max(1, int(num_epochs)) if num_epochs is not None else 1
    )
    from .utils import _finite_text_pad_width, _normalize_seed
    # Normalized so seed=None is deterministic (canonicalized) instead of
    # entropy-derived; explicit seeds are unchanged. Visits stay identity —
    # these plans carry explicitly materialized epoch blocks.
    rng = random.Random(_normalize_seed(seed))
    schedule = []
    widths = []
    cycle_length = None
    global_batch_size = _distributed_global_batch_size(batch_size, comm_group)
    for epoch_idx in range(_n_epochs_materialize):
        epoch_order = _order_indices_for_epoch(epoch_idx)
        epoch_schedule = []
        for start in range(0, len(epoch_order), global_batch_size):
            batch_indices = epoch_order[start:start + global_batch_size]
            batch_indices = _rank_slice_distributed_batch(
                batch_indices,
                batch_size,
                comm_group=comm_group,
                pad_source=epoch_order,
                pad_mode=distributed_pad_mode,
            )
            if not batch_indices:
                continue
            valid_indices = [i for i in batch_indices if i is not None]
            max_len = max(
                (len(all_items[i][0]) for i in valid_indices),
                default=2,
            )
            # +1 for autoregressive shift (mlx-lm iterate_batches parity).
            padded_len = _finite_text_pad_width(
                max_len,
                pad_to_multiple=_PAD_MULTIPLE,
                max_seq_length=max_seq_length,
            )
            epoch_schedule.append((tuple(batch_indices), padded_len))

        # 4. Legacy length-sort: shuffle batches so adjacent steps differ.
        if not _order_requested:
            rng.shuffle(epoch_schedule)
        for batch_indices, padded_len in epoch_schedule:
            schedule.append(batch_indices)
            widths.append(padded_len)
        # One dataset pass == this epoch's micro-batch count (pre-truncation).
        if cycle_length is None and len(epoch_schedule) > 0:
            cycle_length = len(epoch_schedule)

    # Limit if needed
    if num_batches is not None and len(schedule) > num_batches:
        schedule = schedule[:num_batches]
        widths = widths[:num_batches]

    plan = FiniteTextBatchPlan(
        tuple(
            _FiniteTextRow(
                tuple(int(token) for token in input_ids),
                offset=1,
                labels=tuple(int(label) for label in labels),
            )
            for input_ids, labels in all_items
        ),
        schedule,
        cycle_length=cycle_length,
        max_seq_length=max_seq_length,
        pad_id=pad_id,
        minimum_width=2,
        widths=widths,
        label_dtype="int32",
    )
    batches = plan if return_plan else plan.materialize_all()

    if return_dataset:
        return batches, _create_response_masked_dataset(all_items)
    return batches


def _create_response_masked_dataset(items):
    """Build a Dataset-like public view from tokenized response-masked rows."""
    rows = [
        {"input_ids": list(input_ids), "labels": list(labels)}
        for input_ids, labels in items
    ]
    try:
        from datasets import Dataset
    except ImportError:
        return rows
    return Dataset.from_list(rows)


def _check_all_masked(batches, max_check=100, comm_group=None, world_size=1):
    """Raise if all labels in the first N batches are -100 (mirrors
    fix_zero_training_loss from the HF path).

    In DDP ``batches`` is only this rank's shard, so the per-rank bad/good
    counts are all-summed before the decision. Otherwise a rank whose shard
    happens to be entirely masked would raise ZeroDivisionError alone while
    peers with trainable labels advance to the first collective and hang."""
    seen_bad = 0
    seen_good = 0
    checked = 0
    if isinstance(batches, FiniteTextBatchPlan):
        label_batches = (
            (
                (-100,)
                if row_index is None
                else batches.rows[int(row_index)].labels
                for row_index in batch_indices
            )
            for batch_indices in batches.schedule
        )
    else:
        label_batches = (
            batch_labels.tolist()
            for _batch_ids, _batch_lengths, batch_labels in batches
        )
    for labels_list in label_batches:
        for row in labels_list:
            unique = set(row or ())
            if unique == {-100}:
                seen_bad += 1
            else:
                seen_good += 1
            checked += 1
            if checked >= max_check:
                break
        if checked >= max_check:
            break

    # Reduce across ranks before deciding so every rank raises/warns together
    # (all ranks reach this collective; the early return below is post-reduce).
    if comm_group is not None and world_size > 1:
        counts = mx.distributed.all_sum(
            mx.array([seen_bad, seen_good], dtype=mx.int32),
            group=comm_group, stream=mx.cpu,
        )
        mx.eval(counts)
        seen_bad, seen_good = int(counts[0].item()), int(counts[1].item())

    if seen_bad == 0 and seen_good == 0:
        return
    ratio = seen_bad / (seen_bad + seen_good)
    # ZeroDivisionError matches fix_zero_training_loss in the HF/CUDA path
    if ratio == 1.0:
        raise ZeroDivisionError(
            "Unsloth: All labels in your dataset are -100. Training losses will be all 0.\n"
            "Are you sure you used `train_on_responses_only` correctly?\n"
            "Check that your instruction_part and response_part strings match "
            "the chat template used by your tokenizer."
        )
    elif ratio >= 0.9:
        import warnings
        warnings.warn(
            f"Unsloth: {seen_bad}/{seen_bad + seen_good} samples have all -100 labels "
            f"({ratio:.0%}). Your instruction_part / response_part may not match "
            f"the chat template correctly.",
            UserWarning,
        )


def _check_vlm_all_masked(batches, max_check=100, comm_group=None, world_size=1):
    """_check_all_masked for finite VLM batch plans (construction metadata).

    As in the text path, under DDP ``batches`` is only this rank's shard, so
    counts are all-summed before deciding: otherwise a rank whose shard is
    entirely masked would raise alone and hang its peers."""
    # The checker consumes construction-time plan metadata: no extra processor
    # work or materialization ahead of the collective below, or a failing rank
    # would strand its peers there.
    seen_bad, seen_good = batches.supervision_counts(max_check)

    # Reduce across ranks before deciding so every rank raises/warns together
    # (all ranks reach this collective; the early return below is post-reduce).
    if comm_group is not None and world_size > 1:
        counts = mx.distributed.all_sum(
            mx.array([seen_bad, seen_good], dtype=mx.int32),
            group=comm_group, stream=mx.cpu,
        )
        mx.eval(counts)
        seen_bad, seen_good = int(counts[0].item()), int(counts[1].item())

    if seen_bad == 0 and seen_good == 0:
        return
    ratio = seen_bad / (seen_bad + seen_good)
    # ZeroDivisionError matches fix_zero_training_loss in the HF/CUDA path
    if ratio == 1.0:
        raise ZeroDivisionError(
            "Unsloth: All VLM labels in your dataset are -100. Training losses will be all 0.\n"
            "Are you sure you used `train_on_responses_only` correctly?\n"
            "Check that your instruction_part and response_part strings match "
            "the chat template used by your processor."
        )
    elif ratio >= 0.9:
        import warnings
        warnings.warn(
            f"Unsloth: {seen_bad}/{seen_bad + seen_good} VLM samples have all -100 labels "
            f"({ratio:.0%}). Your instruction_part / response_part may not match "
            f"the chat template correctly.",
            UserWarning,
        )


def _prepare_response_labeled_eval_batches(
    trainer,
    tokenizer,
    mask_fn,
    *,
    sized_only=False,
):
    """Prepare response-masked text eval batches, optionally only when sized."""
    if trainer.eval_dataset is None:
        return False
    eval_datasets = (
        list(trainer.eval_dataset.values())
        if isinstance(trainer.eval_dataset, dict)
        else [trainer.eval_dataset]
    )
    lazy_splits = [
        _is_mlx_lazy_text_source(dataset) for dataset in eval_datasets
    ]
    if sized_only and all(lazy_splits):
        return False

    args = trainer.args
    eval_batch_size = (
        getattr(args, "per_device_eval_batch_size", None)
        or args.per_device_train_batch_size
    )
    comm_group = getattr(trainer, "distributed_world", None)

    def _create(eval_dataset):
        batches, response_masked_dataset = _create_labeled_batches(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            mask_fn=mask_fn,
            batch_size=eval_batch_size,
            max_seq_length=args.max_seq_length,
            formatting_func=trainer.formatting_func,
            dataset_text_field=args.dataset_text_field,
            seed=args.seed,
            chat_template=getattr(args, "chat_template", None),
            model_name=getattr(trainer.model, "_hf_repo", None),
            model_type=(
                getattr(trainer.model, "_config", {}).get("model_type")
                if isinstance(getattr(trainer.model, "_config", {}), dict)
                else None
            ),
            append_eos=bool(getattr(args, "append_eos", True)),
            dataset_order=getattr(args, "dataset_order", "default"),
            preserve_dataset_order=bool(
                getattr(args, "preserve_dataset_order", False)
            ),
            return_dataset=True,
            comm_group=comm_group,
            distributed_pad_mode="empty",
        )
        return batches, response_masked_dataset

    if isinstance(trainer.eval_dataset, dict):
        if sized_only and any(lazy_splits):
            for key, value in trainer.eval_dataset.items():
                if _is_mlx_lazy_text_source(value):
                    continue
                _unused_batches, split_dataset = _create(value)
                trainer.eval_dataset[key] = split_dataset
            return False
        eval_batches = {}
        for key, value in trainer.eval_dataset.items():
            split_batches, split_dataset = _create(value)
            eval_batches[key] = split_batches
            trainer.eval_dataset[key] = split_dataset
    else:
        eval_batches, trainer.eval_dataset = _create(trainer.eval_dataset)
    trainer._eval_batches_labeled = eval_batches
    return True


def train_on_responses_only(
    trainer,
    instruction_part=None,
    response_part=None,
    force_match=True,
    tokenizer=None,
    return_function=False,
    num_proc=None,
    last_response_only=False,
):
    """Mask instruction tokens from loss — train only on assistant responses.

    Call after MLXTrainer(...), before trainer.train(). Works for text and
    VLM models; mirrors the HF/unsloth API.

    Args:
        trainer: MLXTrainer (may be None when return_function=True and a
            tokenizer is given).
        instruction_part: String marking the start of user/instruction turns.
        response_part: String marking the start of assistant/response turns.
        force_match: Match newlines too (forwarded to the HF implementation).
        tokenizer: Optional override; defaults to trainer.tokenizer.
        return_function: If True, return the masking closure only.
        num_proc: Accepted for HF API compat, unused on MLX.
        last_response_only: If True, only the final assistant response is
            unmasked, matching the CUDA helper.

    Returns:
        The trainer (for chaining), or the closure if return_function=True.
    """
    from ..dataset_utils import (
        train_on_responses_only as _hf_train_on_responses_only,
    )

    # Resolve tokenizer: kwarg > trainer.tokenizer
    _source = tokenizer
    if _source is None and trainer is not None:
        _source = trainer.tokenizer
    if _source is None:
        raise ValueError(
            "Unsloth: A tokenizer must be provided either via the `tokenizer` "
            "kwarg or via trainer.tokenizer."
        )

    # Callable HF tokenizer for token matching and text batch encoding.
    _tokenizer = _resolve_response_mask_tokenizer(_source)
    _lazy_text_eval = False
    eval_dataset = getattr(trainer, "eval_dataset", None)
    if eval_dataset is not None:
        eval_datasets = (
            eval_dataset.values()
            if isinstance(eval_dataset, dict)
            else (eval_dataset,)
        )
        _lazy_text_eval = any(
            _is_mlx_lazy_text_source(dataset) for dataset in eval_datasets
        )
    if (
        not return_function
        and trainer is not None
        and not trainer._is_vlm
        and trainer.args.streaming
        and (
            _is_mlx_lazy_text_source(trainer._train_dataset_for_batches())
            or _lazy_text_eval
        )
    ):
        config = getattr(trainer.model, "_config", {})
        model_type = config.get("model_type") if isinstance(config, dict) else None
        _tokenizer = normalize_mlx_chat_template(
            _tokenizer,
            chat_template=getattr(trainer.args, "chat_template", None),
            model_name=getattr(trainer.model, "_hf_repo", None),
            model_type=model_type,
            is_vlm=False,
            strict=False,
        )

    # Omitted markers -> auto-detect from the right chat template (see helper).
    if instruction_part is None and response_part is None:
        _detect_source = _resolve_autodetect_template_source(
            trainer, _source, _tokenizer, return_function=return_function,
        )
    else:
        _detect_source = _tokenizer

    # Get masking closure from the HF/CUDA implementation
    mask_fn = _hf_train_on_responses_only(
        None,
        instruction_part=instruction_part,
        response_part=response_part,
        force_match=force_match,
        tokenizer=_detect_source,
        return_function=True,
        last_response_only=last_response_only,
    )

    if return_function:
        return mask_fn

    if trainer is None:
        raise ValueError(
            "trainer is required when return_function=False. "
            "Pass return_function=True to get the masking closure, "
            "or provide an MLXTrainer instance."
        )

    if trainer._is_vlm:
        # VLM path: store mask_fn for application during batch creation
        trainer._vlm_response_mask_fn = mask_fn
        print("Unsloth: train_on_responses_only enabled (VLM mode).")
    else:
        args = trainer.args
        train_dataset = trainer._train_dataset_for_batches()
        if args.streaming:
            trainer._mlx_response_mask_fn = mask_fn
            trainer._mlx_response_mask_tokenizer = _tokenizer
        if args.streaming and _is_mlx_lazy_text_source(train_dataset):
            if not isinstance(train_dataset, _MLXIterableTokenizedDatasetView):
                train_dataset = _MLXIterableTokenizedDatasetView(
                    train_dataset,
                    _tokenizer,
                    dataset_text_field=args.dataset_text_field,
                    formatting_func=trainer.formatting_func,
                    append_eos=bool(getattr(args, "append_eos", True)),
                    completion_only_loss=_text_completion_only_loss_arg(args),
                    assistant_only_loss=_text_assistant_only_loss_arg(args),
                    max_seq_length=args.max_seq_length,
                )
                trainer.train_dataset = train_dataset
                trainer._mlx_train_dataset_for_batches = train_dataset
            else:
                train_dataset.set_tokenizer(_tokenizer)
            train_dataset.set_response_mask(mask_fn)
            trainer._batches = None
            trainer._eval_batches_labeled = None
            _prepare_response_labeled_eval_batches(
                trainer,
                _tokenizer,
                mask_fn,
                sized_only=True,
            )
            print("Unsloth: train_on_responses_only enabled (lazy text mode).")
            return trainer

        # Eager/sized text path: tokenize, mask, and create batches now.
        total_batches_needed = (
            args.max_steps * args.gradient_accumulation_steps
            if args.max_steps > 0 else None
        )
        # Only materialize all epoch blocks for true epoch-based runs. Step-based
        # runs (max_steps>0) truncate to num_batches, so pre-building every epoch
        # just wastes tokenization/memory. Mirrors the unlabeled path's gate.
        labeled_num_epochs = (
            int(args.num_train_epochs)
            if (args.max_steps <= 0 and getattr(args, "num_train_epochs", -1) > 0)
            else None
        )
        comm_group = getattr(trainer, "distributed_world", None)
        batches, response_masked_dataset = _create_labeled_batches(
            dataset=train_dataset,
            tokenizer=_tokenizer,
            mask_fn=mask_fn,
            batch_size=args.per_device_train_batch_size,
            max_seq_length=args.max_seq_length,
            formatting_func=trainer.formatting_func,
            dataset_text_field=args.dataset_text_field,
            num_batches=total_batches_needed,
            seed=args.seed,
            chat_template=getattr(args, "chat_template", None),
            model_name=getattr(trainer.model, "_hf_repo", None),
            model_type=(
                getattr(trainer.model, "_config", {}).get("model_type")
                if isinstance(getattr(trainer.model, "_config", {}), dict)
                else None
            ),
            append_eos=bool(getattr(args, "append_eos", True)),
            dataset_order=getattr(args, "dataset_order", "default"),
            preserve_dataset_order=bool(getattr(args, "preserve_dataset_order", False)),
            num_epochs=labeled_num_epochs,
            return_dataset=True,
            comm_group=comm_group,
            return_plan=True,
        )
        trainer.train_dataset = response_masked_dataset
        trainer._mlx_train_dataset_for_batches = response_masked_dataset

        # Safety check: detect all-masked labels early. In DDP batches is this
        # rank's shard, so pass the group to reduce counts before deciding.
        _check_all_masked(
            batches,
            comm_group=comm_group,
            world_size=getattr(trainer, "distributed_world_size", 1),
        )
        trainer._prepared_batches_include_epochs = (
            labeled_num_epochs is not None
        )
        trainer._batches = batches

        _prepare_response_labeled_eval_batches(
            trainer,
            _tokenizer,
            mask_fn,
            sized_only=bool(args.streaming),
        )

        print(f"Unsloth: train_on_responses_only enabled "
              f"({len(batches)} batches prepared).")

    return trainer
