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

from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
import concurrent.futures
import hashlib
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


from .utils import (
    make_cce_loss_fn,
    make_baseline_loss_fn,
    make_vlm_cce_loss_fn,
    make_vlm_baseline_loss_fn,
    create_batches,
    create_preference_batches,
    _hf_encoding_tokenizer,
    make_orpo_loss_fn,
    make_dpo_loss_fn,
    make_grpo_loss_fn,
    create_ordered_batches,
    iterate_training_batches,
    create_vlm_batches,
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
    model_has_non_lora_trainable_params,
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


def _clip_grad_norm_fp32(grad, max_norm):
    """Global norm clipping with a float32 norm reduction.

    ``mlx.optimizers.clip_grad_norm`` reduces each leaf in its storage dtype.
    For bf16/fp16 VLMs, that can move the global scale away from PyTorch/HF,
    which computes the clipping norm in fp32. Keep clipped leaves in their
    original dtype, but compute the single global scale in fp32.
    """
    norm_squared = tree_reduce(
        lambda acc, g: acc + mx.sum(mx.square(g.astype(mx.float32))),
        grad,
        mx.array(0.0, dtype=mx.float32),
    )
    total_norm = mx.sqrt(norm_squared)
    scale = mx.minimum(
        mx.array(max_norm, dtype=mx.float32) / (
            total_norm + mx.array(1e-6, dtype=mx.float32)
        ),
        mx.array(1.0, dtype=mx.float32),
    )
    return tree_map(lambda g: g * scale.astype(g.dtype), grad), total_norm


def _prune_stale_checkpoints(output_dir, save_total_limit):
    """Keep the newest ``save_total_limit`` checkpoint-* dirs (HF Trainer parity).

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
    for _, stale in checkpoints[:-save_total_limit]:
        try:
            shutil.rmtree(stale)
        except Exception as exc:
            print(f"  Unsloth: failed to prune old checkpoint {stale}: {exc}")
            continue
        print(f"  Unsloth: pruned old checkpoint {stale} "
              f"(save_total_limit={save_total_limit})")


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
    loss_type: str = "sft"  # "sft" or "orpo"
    orpo_beta: float = 0.1  # ORPO odds-ratio weight (TRL default)
    dpo_beta: float = 0.1  # DPO beta (TRL default)
    reference_free: bool = False  # DPO: drop the reference term if True
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
    compile_arch_overrides: dict[str, str] | None = None
    compile_backend_overrides: dict[str, str] | None = None
    patch_mode: str = "patched"  # "patched" runs the MLX compile monkey patches, "unpatched" forces eager baseline mode.
    compile_auto_tune: bool = True
    compile_trace: bool = True
    gradient_checkpointing: bool = True
    streaming: bool = False  # Use streaming iterator instead of materializing batches
    dataset_order: str = "default"  # "default", "sequential", or "torch_randperm"
    preserve_dataset_order: bool = False  # Match Studio CUDA SequentialSampler order
    memory_limit_gb: float | None = None  # None = auto Metal guard (~85% of recommended working set); <= 0 disables
    cache_limit_gb: float | None = None  # Optional MLX Metal cache cap in GB; <= 0 disables override
    wired_limit_gb: float | None = None  # None = min(recommended working set, memory limit); <= 0 disables
    disable_memory_limits: bool = False
    cast_norm_output_to_input_dtype: bool = True  # fp32 norm storage/math, bf16/fp16 downstream activations
    append_eos: bool = True  # True = mlx-lm parity; Studio sets False (template owns EOS)

    # VLM / completion masking
    train_on_completions: bool = False  # Mask prompt tokens in loss
    completion_only_loss: bool | None = None  # None = SFT/VLM default; False trains on prompt+completion
    assistant_only_loss: bool = False  # Mask non-assistant tokens with chat-template assistant masks
    assistant_token_id: int = 0  # Token ID marking start of assistant response
    vlm_chat_template: object = None  # Unsloth template name/tuple or raw Jinja string
    per_device_eval_batch_size: int | None = None
    image_size: object = None  # VLM image resize override from UnslothVisionDataCollator(resize=...)

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
        copied_all_fields = len(provided) == len(config_fields)
        copied_default_warmup_with_ratio = (
            copied_all_fields
            and getattr(self, "warmup_steps", None) == warmup_steps_default
            and getattr(self, "warmup_ratio", None) != warmup_ratio_default
        )
        self._unsloth_mlx_warmup_steps_explicit = (
            "warmup_steps" in provided and not copied_default_warmup_with_ratio
        )


# init=False so the subclass keeps MLXTrainingConfig's custom __init__ rather
# than getting a dataclass-generated one. The base __init__ is the only place
# that records _unsloth_mlx_warmup_steps_explicit (and does the kwargs coercion
# / unknown-argument checks); a generated __init__ would skip it and silently
# treat an explicit warmup_steps that happens to equal the default as implicit,
# dropping it in favour of warmup_ratio. init=False still registers the new
# fields below in fields(), which the inherited __init__ iterates over.
@dataclass(init=False)
class MLXORPOConfig(MLXTrainingConfig):
    """ORPO config mirroring TRL's ORPOConfig. Presets loss_type='orpo';
    tune orpo_beta (inherited). Use with MLXORPOTrainer."""
    loss_type: str = "orpo"


@dataclass(init=False)
class MLXDPOConfig(MLXTrainingConfig):
    """DPO config mirroring TRL's DPOConfig. Presets loss_type='dpo';
    tune dpo_beta / reference_free (inherited). Use with MLXDPOTrainer."""
    loss_type: str = "dpo"


@dataclass(init=False)
class MLXGRPOConfig(MLXTrainingConfig):
    """GRPO config mirroring TRL's GRPOConfig. Presets loss_type='grpo'.
    Use with MLXGRPOTrainer (pass reward_funcs to the trainer)."""
    loss_type: str = "grpo"
    num_generations: int = 4          # completions per prompt (the group)
    grpo_beta: float = 0.04           # KL penalty weight (TRL default)
    grpo_epsilon: float = 0.2         # PPO clip epsilon (low and high)
    temperature: float = 1.0          # rollout sampling temperature
    max_completion_length: int = 128  # max new tokens per completion
    reference_free: bool = False      # drop the KL term if True


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
        self._reset_run_state()
        self.stop_requested = False  # Set True to stop training early
        self._batches = None  # Pre-created batches (skips internal batch creation)
        self._step_callbacks = []  # Callbacks called after each logged step
        self._eval_callbacks = []  # Callbacks called after each eval

    def _reset_run_state(self):
        """Per-run training/metric state. Reset from __init__ and at the start
        of each train() so reusing a trainer for a second run starts clean;
        _early_stopped cleared so a run-1 early stop doesn't block run 2.
        stop_requested is deliberately not reset: it is externally owned (a
        controller thread may set it at any moment, including during train()
        setup and batch prep), so clearing it here would silently drop an
        in-flight cancel. Callbacks and pre-created batches persist across
        runs and aren't reset."""
        self._global_step = 0
        self._train_loss_history = []
        self._kl_history = []  # GRPO: mean k3 KL per logged step
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

    def _distributed_any_flag(self, flag):
        """Return whether any rank reported ``flag``."""
        return self._distributed_status_mask(int(bool(flag))) > 0

    def _distributed_status_mask(self, mask):
        """All-sum a small integer status code across ranks."""
        local = mx.array(int(mask), dtype=mx.int32)
        total = self._distributed_all_sum(local, stream=mx.cpu)
        mx.eval(total)
        return int(total.item())

    def _raise_distributed_failure_from_any(self, failed_any, context, exc=None):
        """Abort this rank after a rank-wide failure consensus."""
        if not failed_any:
            return
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
        """
        self._step_callbacks.append(fn)

    def add_eval_callback(self, fn):
        """Register a callback called after each evaluation.

        fn(step, eval_loss, perplexity)
        """
        self._eval_callbacks.append(fn)

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

    def _evaluate_batch_totals(self, eval_batches, loss_fn, is_vlm=False):
        """Accumulate weighted loss totals for one flat eval batch stream."""
        all_losses = mx.array(0.0)
        ntokens = mx.array(0)
        iterator = iter(eval_batches)

        while True:
            failed = False
            error = None
            try:
                batch_data = next(iterator)
            except StopIteration:
                break
            except Exception as exc:
                failed = True
                error = exc

            if not failed and not self.stop_requested:
                try:
                    if is_vlm:
                        loss, ntoks = loss_fn(self.model, batch_data)
                    else:
                        batch, lengths, labels = batch_data
                        loss, ntoks = loss_fn(self.model, batch, lengths, labels)
                    all_losses += loss * ntoks
                    ntokens += ntoks
                    mx.eval(all_losses, ntokens)
                except Exception as exc:
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
            for split_name, split_batches in eval_batches.items():
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
        Studio worker.py log keys so notebook and Studio runs chart identically."""
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


    def train(self, resume_from_checkpoint: str | None = None):
        """Run MLX-native training loop following mlx-lm's compiled-step pattern
        with gradient accumulation. Returns a dict of training metrics."""
        # Stash for _train_inner. None = fresh start, a path = resume.
        self._resume_from_checkpoint = resume_from_checkpoint
        self._ensure_distributed()
        self._install_neftune()
        is_main_process = self.is_main_process

        def _main_print(*print_args, **print_kwargs):
            if is_main_process:
                print(*print_args, **print_kwargs)

        args = self.args
        model = self.model
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

        # Pick loss function (returns (loss, ntoks))
        use_cce = args.use_cce
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
            if getattr(args, "loss_type", "sft") == "orpo":
                _ob = getattr(args, "orpo_beta", 0.1)
                loss_fn = make_orpo_loss_fn(beta=_ob)
                print("Unsloth: Using ORPO loss (beta=" + str(_ob) + ").")
            elif getattr(args, "loss_type", "sft") == "dpo":
                _db = getattr(args, "dpo_beta", 0.1)
                _rf = bool(getattr(args, "reference_free", False))
                _lora_mods = [mod for _, mod in iter_mlx_lora_modules(model)]
                if (_lora_mods and not _rf
                        and model_has_non_lora_trainable_params(model)):
                    raise ValueError(
                        "Unsloth: DPO with a LoRA reference is not supported when "
                        "non-LoRA parameters (e.g. a directly-trained lm_head / "
                        "embed_tokens) are also trainable. The reference is obtained "
                        "by disabling LoRA adapters, but those non-LoRA tensors keep "
                        "moving during training, so the reference would no longer be "
                        "the frozen initial policy and the DPO gradient would be "
                        "wrong. Train LoRA adapters only, or pass "
                        "reference_free=True to train without a reference."
                    )
                loss_fn = make_dpo_loss_fn(beta=_db, lora_mods=_lora_mods, reference_free=_rf)
                print("Unsloth: Using DPO loss (beta=" + str(_db) +
                      (", reference_free" if _rf else "") + ").")
            elif getattr(args, "loss_type", "sft") == "grpo":
                _gb = getattr(args, "grpo_beta", 0.04)
                _ge = getattr(args, "grpo_epsilon", 0.2)
                _grf = bool(getattr(args, "reference_free", False))
                _lora_mods = [mod for _, mod in iter_mlx_lora_modules(model)]
                if (_lora_mods and _gb != 0.0 and not _grf
                        and model_has_non_lora_trainable_params(model)):
                    raise ValueError(
                        "Unsloth: GRPO KL regularization with a LoRA reference is "
                        "not supported when non-LoRA parameters (e.g. a "
                        "directly-trained lm_head / embed_tokens) are also "
                        "trainable. The KL reference is obtained by disabling LoRA "
                        "adapters, but those non-LoRA tensors keep moving during "
                        "training, so the reference would no longer be the frozen "
                        "initial policy and the KL term would be wrong. Train LoRA "
                        "adapters only, or set grpo_beta=0 (equivalently "
                        "reference_free=True) to train without the KL term."
                    )
                loss_fn = make_grpo_loss_fn(beta=_gb, lora_mods=_lora_mods,
                    reference_free=_grf, epsilon_low=_ge, epsilon_high=_ge)
                print("Unsloth: Using GRPO loss (beta=" + str(_gb) + ").")
            elif use_cce:
                loss_fn = make_cce_loss_fn(model)
                cce_backend = getattr(loss_fn, "_unsloth_cce_backend", "unknown")
                _main_print(
                    f"Unsloth: Using CCE loss ({cce_backend}) "
                    "for memory-efficient training."
                )
            else:
                loss_fn = make_baseline_loss_fn()
                _main_print("Unsloth: Using standard cross-entropy loss.")

        # Prepare data, determine total_steps first. Keep any prebuilt flag
        # from train_on_responses_only; _prepare_data returns self._batches
        # early and never re-derives it for the completion-only text path.
        if self._batches is None:
            self._prepared_batches_include_epochs = False
        batches, batch_iter = self._prepare_data(is_vlm)

        if batches is not None and not batches:
            raise ValueError(
                "No training batches created. Check your dataset and batch_size."
            )

        grad_accum = args.gradient_accumulation_steps
        if args.max_steps > 0:
            total_steps = args.max_steps
        elif batches is not None:
            n_batches = len(batches)
            if getattr(self, "_prepared_batches_include_epochs", False):
                total_steps = n_batches // grad_accum
            elif args.num_train_epochs > 0:
                # Epoch-based: total micro-batches = epochs * batches_per_epoch
                total_steps = (n_batches * args.num_train_epochs) // grad_accum
            else:
                total_steps = n_batches // grad_accum
            total_steps = max(1, total_steps)
        else:
            # Streaming mode — must have max_steps
            if args.num_train_epochs > 0:
                raise ValueError(
                    "num_train_epochs requires a finite dataset (not streaming). "
                    "Use max_steps instead, or disable streaming."
                )
            raise ValueError(
                "max_steps must be > 0 when using streaming mode."
            )

        # Build optimizer with LR schedule
        optimizer = self._build_optimizer(total_steps)

        # Resume from checkpoint: load adapter weights, optimizer state,
        # and trainer state (step counter + loss history). Adapters were
        # already loaded by the Studio worker into the model before train()
        # was called, so we only handle optimizer and trainer state here.
        # The step offset is applied below at loop start so the LR scheduler
        # and dataloader fast-forward to the right position.
        # Reset per-run state so reusing a trainer for a second train() without
        # resume starts clean (else run-1's early-stop flag breaks the loop at
        # step 0). The resume block below re-seeds the persisted fields.
        self._reset_run_state()

        _resume_step = 0
        _resume_from = getattr(self, "_resume_from_checkpoint", None)
        _resume_from = self._validate_distributed_resume_checkpoint(_resume_from)
        if _resume_from:
            try:
                # 1. Load trained adapter weights into the model. The model
                #    already has LoRA wrappers applied (Studio pipeline does
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
                ts = load_trainer_state(_resume_from)
                _resume_step = int(ts.get("global_step", 0))
                # Seed the live step counter from the checkpoint so a no-op
                # resume (checkpoint already at max_steps, loop body never runs)
                # still reports the reached step instead of the initial 0. The
                # loop overwrites this on every optimizer step of a real resume.
                self._global_step = _resume_step
                self._train_loss_history = list(ts.get("train_loss_history", []))
                self._best_metric = ts.get("best_metric", None)
                self._best_step = ts.get("best_step", None)
                self._es_patience_counter = int(ts.get("es_patience_counter", 0) or 0)
                # best/ lives in output_dir, not in the checkpoint dir, so a
                # checkpoint resumed elsewhere (copied dir, new output_dir) can
                # carry best-model state whose weights aren't present. Keep the
                # state only when they are: an unloadable "best" would suppress
                # best-saves and early-stop against a model that
                # load_best_model_at_end can't restore.
                _best_path = f"{args.output_dir}/best/adapters.safetensors"
                if self._best_step is not None and not os.path.exists(_best_path):
                    _main_print(
                        f"Unsloth: checkpoint carries best-model state (step "
                        f"{self._best_step}) but {args.output_dir}/best has no "
                        f"saved weights; restarting best-model tracking."
                    )
                    self._best_metric = None
                    self._best_step = None
                    self._es_patience_counter = 0
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

        # Build loss+grad function — returns ((loss, ntoks), grads)
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

        # Per-parameter gradient scaling (LoRA+, embedding LR)
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

        _needs_grad_scaling = use_lora_plus or use_embedding_lr
        _warned_skip_optimizer_state_grad_norm = False

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
        state = [model.state, optimizer.state, mx.random.state]
        # grad_accum==1 fast path: only for unclipped updates, since
        # clip_grad_norm can spike peak memory on bf16 VLM runs.
        _direct_single_step_update = (
            grad_accum == 1 and
            distributed_world_size <= 1 and
            not _needs_grad_scaling and
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
            # Suffix-anchor so lora_b_router.weight doesn't pick up the LoRA+ mult.
            if use_lora_plus and (name == "lora_b" or name.endswith(".lora_b")):
                scale = scale * lora_plus_ratio
            # Segment-anchor so not_lm_head_router.weight doesn't pick up embed LR.
            if use_embedding_lr:
                _segments = name.split(".")
                _is_embed_or_lm_head = (
                    "embed_tokens" in _segments
                    or "lm_head" in _segments
                )
                if _is_embed_or_lm_head:
                    scale = scale * embedding_lr_ratio
            if clip_scale is not None:
                scale = scale * clip_scale
            if dtype is not None and scale.dtype != dtype:
                scale = scale.astype(dtype)
            return scale

        optimizer_v_sum = None

        def _optimizer_v_total():
            total = mx.array(0.0, dtype=mx.float32)
            found = False
            for name, value in tree_flatten(getattr(optimizer, "state", {})):
                if name != "v" and not name.endswith(".v"):
                    continue
                found = True
                value_f = value.astype(mx.float32)
                total = total + mx.sum(value_f)
            return total if found else None

        def _grad_norm_from_optimizer_state():
            nonlocal optimizer_v_sum
            betas = getattr(optimizer, "betas", None)
            if not betas or len(betas) < 2:
                return None
            current_v_sum = _optimizer_v_total()
            if current_v_sum is None:
                return None
            previous_v_sum = (
                optimizer_v_sum
                if optimizer_v_sum is not None
                else mx.array(0.0, dtype=mx.float32)
            )
            beta2 = mx.array(float(betas[1]), dtype=mx.float32)
            denom = mx.maximum(
                mx.array(1.0, dtype=mx.float32) - beta2,
                mx.array(1e-30, dtype=mx.float32),
            )
            grad_norm_sq = mx.maximum(
                (current_v_sum - beta2 * previous_v_sum) / denom,
                mx.array(0.0, dtype=mx.float32),
            )
            grad_norm = mx.sqrt(grad_norm_sq)
            mx.eval(current_v_sum, grad_norm)
            optimizer_v_sum = current_v_sum
            return grad_norm

        def _can_report_optimizer_state_norm():
            # Adam-family: recover ||g|| from the second moment after update
            # (v_t = beta2*v_{t-1} + (1-beta2)*g_t^2), avoiding a second
            # consumer on the lazy backward graph.
            return getattr(optimizer, "betas", None)

        def _apply_update(grad, toks_f):
            """Scale accumulated grads by supervised-token count, apply the
            selected clipping mode, and update. Global-norm clipping reports
            its norm; non-global modes report after update from Adam state to
            keep the backward graph single-consumer.
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
            if _clip_grad_value:
                final_grad = _clip_grad_by_value(final_grad, max_grad_value)
            if _clip_grad_leaf_norm:
                final_grad = _clip_grad_by_leaf_norm(final_grad, max_grad_leaf_norm)
            # Coupled (SGD) decay folds into the post-clip grad so it feeds
            # momentum; decoupled (AdamW-family) decay shrinks params directly.
            final_grad = self._apply_coupled_weight_decay(model, final_grad)
            self._apply_manual_weight_decay(model, optimizer, final_grad)
            optimizer.update(model, final_grad)
            _restore_trainable_storage_dtypes()
            return grad_norm

        def _apply_update_direct(grad):
            """Fast exact path for ``grad_accum == 1`` with no per-leaf scaling.

            The raw grads already are the per-token average, so skip the
            ``*ntoks`` then ``/ntoks`` round-trip (which only promotes the tree
            to float32 and spikes peak memory) and clip/update directly.
            """
            grad_norm = None
            if max_grad_norm > 0:
                grad, grad_norm = _clip_grad_norm_fp32(grad, max_norm=max_grad_norm)
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
                grad_norm = _apply_update_direct(grad)
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

        compile_policy = build_compile_policy(args=args)
        _compile_decision = getattr(self, "_compile_decision", None)
        _use_compile = compile_policy.mode != "eager"
        _ddp_compile_local_grad = _use_compile and distributed_world_size > 1
        if (
            _use_compile
            and not _ddp_compile_local_grad
            and max_grad_norm > 0
            and grad_accum > 1
        ):
            _main_print(
                "Unsloth: mx.compile disabled because MLX global norm "
                "clipping is enabled with gradient accumulation."
            )
            _use_compile = False
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
        _compile_scope = "none"
        _compile_fallback_reason = None
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
                try:
                    _compiled_local_grad_step = mx.compile(
                        _local_grad_step,
                        inputs=_compile_state,
                        outputs=_compile_state,
                    )
                except Exception as e:
                    _compile_setup_error = e
                if self._distributed_any_flag(_compile_setup_error is not None):
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
        if (getattr(args, "loss_type", "sft") in ("orpo", "dpo", "grpo")
                and args.eval_steps > 0 and self.eval_dataset is not None):
            _main_print(f"Unsloth: eval is not yet supported for {args.loss_type}; skipping eval.")
        elif args.eval_steps > 0 and self.eval_dataset is not None:
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
                    """Materialize eval batches for one dataset split."""
                    if is_vlm:
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
                    return create_batches(
                        dataset=eval_dataset,
                        tokenizer=self.tokenizer,
                        batch_size=eval_batch_size,
                        max_seq_length=args.max_seq_length,
                        seed=args.seed,
                        dataset_text_field=args.dataset_text_field,
                        formatting_func=self.formatting_func,
                        chat_template=getattr(args, "chat_template", None),
                        model_name=getattr(self.model, "_hf_repo", None),
                        model_type=(
                            getattr(self.model, "_config", {}).get("model_type")
                            if isinstance(getattr(self.model, "_config", {}), dict)
                            else None
                        ),
                        append_eos=bool(getattr(args, "append_eos", True)),
                        completion_only_loss=text_completion_only_loss,
                        assistant_only_loss=text_assistant_only_loss,
                        comm_group=self.distributed_world,
                        distributed_pad_mode="empty",
                    )

                if isinstance(self.eval_dataset, dict):
                    eval_batches = {
                        key: _create_eval_batches(value)
                        for key, value in self.eval_dataset.items()
                    }
                else:
                    eval_batches = _create_eval_batches(self.eval_dataset)
            if eval_batches:
                eval_batch_count = (
                    sum(len(value) for value in eval_batches.values())
                    if isinstance(eval_batches, dict) else len(eval_batches)
                )
                _main_print(
                    f"Unsloth: Eval enabled every {args.eval_steps} steps "
                    f"({eval_batch_count} eval batches)."
                )

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
        start_time = time.perf_counter()
        losses = 0
        n_tokens = 0
        steps = 0
        trained_tokens = 0
        train_time = 0
        grad_accum_state = None
        # When resuming, start batch_idx at the resume position so
        # batches[batch_idx % len(batches)] lands on the same batch the
        # original run would have seen next.
        batch_idx = _resume_step * grad_accum

        # Streaming mode: fast-forward the iterator to the resume position.
        # The seed is the same and create_batches/iterate_*_batches is
        # deterministic, so consuming N batches gives us the same data
        # ordering the killed run would have produced.
        if _resume_step > 0 and batch_iter is not None:
            for _ in range(_resume_step * grad_accum):
                try:
                    next(batch_iter)
                except StopIteration:
                    raise RuntimeError(
                        f"Unsloth: streaming dataset exhausted while "
                        f"fast-forwarding to resume step {_resume_step}. "
                        f"Dataset may be shorter than the killed run consumed."
                    ) from None

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
            except Exception as e:
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
                state = [model.state, optimizer.state, mx.random.state]
                local_error = None
                try:
                    result = step_fn(batch_data, prev_state, do_update)
                    _eval_local_result(result)
                except Exception as e:
                    local_error = e
                self._raise_distributed_failure(
                    local_error is not None,
                    "training step after compile fallback",
                    local_error,
                )

            return result

        for it in range(_resume_step * grad_accum + 1, total_steps * grad_accum + 1):
            if self._distributed_should_stop() or self._early_stopped:
                if self.stop_requested:
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
                    batch_data = batches[batch_idx % len(batches)]
                    batch_idx += 1
            except Exception as e:
                batch_error = e
            if distributed_world_size > 1:
                self._raise_distributed_failure(
                    batch_error is not None,
                    "fetching training batch",
                    batch_error,
                )
            elif batch_error is not None:
                raise batch_error

            do_update = (it % grad_accum == 0)
            if do_update:
                # Keep callable scheduler evaluation outside mx.compile. The
                # compiled step reads the scalar LR already in optimizer state.
                self._set_optimizer_lr_for_step(optimizer, it // grad_accum - 1)

            # Pre-update GRPO KL probe (eager, outside the compiled step). Runs
            # only for steps that will be logged, on the SAME batch/lengths the
            # loss uses, and BEFORE the optimizer update -- so the KL reflects the
            # step's (pre-update) weights and is comparable to CUDA's logged KL.
            # Kept BEFORE the DDP dispatch/all-reduce below so grouped advantages
            # are computed on each rank before any collective op.
            self._pending_grpo_kl = None
            if (do_update and hasattr(self, "_grpo_mean_kl")
                    and not isinstance(batch_data, dict)):
                _prospective_step = it // grad_accum
                if (_prospective_step % args.logging_steps == 0
                        or _prospective_step == total_steps):
                    self._pending_grpo_kl = self._grpo_mean_kl(
                        batch_data[0], batch_data[1])

            if _ddp_update_outside_step:
                lvalue, toks, grad_accum_state, grad_norm = _run_ddp_local_step(
                    batch_data, grad_accum_state, do_update,
                )
                if do_update:
                    grad, toks_f = grad_accum_state
                    grad_norm = _apply_update(grad, toks_f)
                    grad_accum_state = None
            else:
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
                        state = [model.state, optimizer.state, mx.random.state]
                        lvalue, toks, grad_accum_state, grad_norm = step_fn(
                            batch_data, grad_accum_state, do_update,
                        )
                    else:
                        raise

            losses += lvalue * toks
            n_tokens += toks
            steps += 1
            if grad_norm is not None:
                mx.eval(grad_norm)
            if grad_accum_state is not None:
                mx.eval(state, losses, n_tokens, grad_accum_state[0], grad_accum_state[1])
            else:
                mx.eval(state, losses, n_tokens)
            if (
                do_update
                and grad_norm is None
                and max_grad_norm <= 0
                and _can_report_optimizer_state_norm()
            ):
                grad_norm = _grad_norm_from_optimizer_state()
            elif (
                do_update
                and grad_norm is None
                and max_grad_norm <= 0
                and not _can_report_optimizer_state_norm()
                and not _warned_skip_optimizer_state_grad_norm
            ):
                _main_print(
                    "Unsloth: skipping grad norm reporting for this MLX "
                    "optimizer/mode to avoid materializing the gradient graph."
                )
                _warned_skip_optimizer_state_grad_norm = True
            global_toks = self._distributed_all_sum(toks, stream=mx.cpu)
            mx.eval(global_toks)
            if int(global_toks.item()) == 0:
                raise ValueError(
                    "Unsloth MLX: a training batch produced zero supervised "
                    "tokens after masking/truncation. Increase max_seq_length, "
                    "reduce image size, or check the chat template / labels."
                )
            train_time += time.perf_counter() - tic

            # Only log/eval on actual optimizer steps
            if not do_update:
                continue

            self._global_step = it // grad_accum
            current_step = self._global_step

            # Logging
            if current_step % args.logging_steps == 0 or current_step == total_steps:
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
                if getattr(self, "_pending_grpo_kl", None) is not None:
                    self._kl_history.append(self._pending_grpo_kl)
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

                # Benchmark hook: reset peak memory after warmup
                reset_after = getattr(self, '_benchmark_reset_peak_after_step', 0)
                if reset_after > 0 and current_step == reset_after:
                    mx.synchronize()
                    mx.reset_peak_memory()

                elapsed_total = time.perf_counter() - start_time

                grad_text = (
                    f"Grad: {grad_norm_val:.4f} | "
                    if grad_norm_val is not None else ""
                )
                kl_text = (
                    f"KL: {self._pending_grpo_kl:.4f} | "
                    if getattr(self, "_pending_grpo_kl", None) is not None else ""
                )
                _main_print(
                    f"  Step {current_step}/{total_steps} | "
                    f"Loss: {train_loss:.4f} | "
                    f"{grad_text}"
                    f"{kl_text}"
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

                losses = 0
                n_tokens = 0
                steps = 0
                train_time = 0

            # Eval
            if (eval_batches and args.eval_steps > 0
                    and current_step % args.eval_steps == 0):
                val_loss, ppl = self._evaluate(
                    eval_batches, loss_fn, is_vlm=is_vlm)
                model.train()
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

                # Eval callbacks fire on rank 0 only, so a callback (or an
                # external cancel arriving mid-eval) that sets stop_requested is
                # initially visible on rank 0 alone. Sync it across ranks before
                # the best-model / early-stopping branch below: otherwise _track
                # diverges (rank 0 skips, peers enter) and the rank-0-guarded
                # best-model save collective in _track hangs the peer ranks.
                self._distributed_should_stop()

                # Best-model tracking + early stopping (Item-5). Skipped after
                # a stop request: an aborted eval leaves partial metrics (a
                # truncated eval_loss can beat any real best) and may lack the
                # tracked key entirely.
                _track = not self.stop_requested and (
                    getattr(args, "load_best_model_at_end", False)
                    or int(getattr(args, "early_stopping_patience", 0) or 0) > 0
                )
                if _track:
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
                        _cur == _cur  # reject NaN: a diverged eval must never become "best"
                        and (
                            self._best_metric is None
                            or (_cur > self._best_metric if _greater else _cur < self._best_metric)
                        )
                    )
                    if _improved:
                        self._best_metric = _cur
                        self._best_step = current_step
                        self._es_patience_counter = 0
                        # Bookkeeping runs on every rank to keep early-stopping
                        # in lockstep; only rank 0 writes output_dir/best. Sync
                        # save failures across ranks so a rank-0 error does not
                        # hang peers at the next collective.
                        best_save_error = None
                        if is_main_process:
                            try:
                                save_trainable_adapters(model, f"{args.output_dir}/best")
                            except ValueError as e:
                                print(f"  Unsloth: skipped best-model save ({e})")
                            except Exception as e:
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

            # Checkpointing
            checkpoint_due = (
                args.save_steps > 0
                and current_step % args.save_steps == 0
            )
            if checkpoint_due:
                checkpoint_error = None
                if is_main_process:
                    ckpt_dir = f"{args.output_dir}/checkpoint-{current_step}"
                    try:
                        try:
                            save_trainable_adapters(model, ckpt_dir)
                        except ValueError as e:
                            _main_print(f"  Unsloth: skipped checkpoint ({e})")
                        else:
                            # Also write optimizer + trainer state so
                            # resume_from_checkpoint restores Adam moments, step
                            # counter, loss history, and best-model / early-stopping
                            # tracking. Best-effort: the adapter save already
                            # succeeded, so log failures but keep it.
                            checkpoint_complete = False
                            try:
                                save_optimizer_state(optimizer, ckpt_dir)
                                save_trainer_state(
                                    {
                                        "global_step": current_step,
                                        "train_loss_history": list(
                                            self._train_loss_history
                                        ),
                                        "best_metric": self._best_metric,
                                        "best_step": self._best_step,
                                        "es_patience_counter": self._es_patience_counter,
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
                                )
                    except Exception as e:
                        checkpoint_error = e
                self._raise_distributed_failure(
                    checkpoint_error is not None,
                    "checkpoint save",
                    checkpoint_error,
                )

        total_time = time.perf_counter() - start_time
        avg_loss = (
            sum(self._train_loss_history) / len(self._train_loss_history)
            if self._train_loss_history else 0.0
        )

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
                self.save_model()
            except ValueError as e:
                _main_print(f"Unsloth: skipped final save ({e})")
            except Exception as e:
                final_save_error = e
            else:
                _main_print(f"Unsloth: Saved final adapters to {args.output_dir}")
        self._raise_distributed_failure(
            final_save_error is not None,
            "final save",
            final_save_error,
        )

        return MLXTrainOutput({
            "train_loss": avg_loss,
            "train_runtime": total_time,
            "train_steps": completed_steps,
            "total_train_steps": total_steps,
            "trained_tokens": trained_tokens,
            "train_samples_per_second": (
                trained_tokens / total_time if total_time > 0 else 0
            ),
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

    def _prepare_data(self, is_vlm):
        """Prepare training data. Returns (batches, batch_iter)."""
        args = self.args
        # GRPO needs the rollout data path (and reward_funcs), which only
        # MLXGRPOTrainer supplies via its own _prepare_data override. Reaching
        # here with loss_type='grpo' means the base MLXTrainer was used with a
        # GRPO config, which would feed SFT batches into the GRPO loss and
        # crash on advantages.reshape. Fail fast with a clear message instead.
        if getattr(args, "loss_type", "sft") == "grpo":
            raise ValueError(
                "Unsloth: GRPO training requires MLXGRPOTrainer (which supplies "
                "reward_funcs and generates rollouts). Use MLXGRPOTrainer instead "
                "of the base MLXTrainer for loss_type='grpo'."
            )
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

        if getattr(args, "loss_type", "sft") in ("orpo", "dpo"):
            if is_vlm:
                raise ValueError(
                    f"{args.loss_type.upper()} is not yet supported for VLM models on MLX."
                )
            batches = create_preference_batches(
                dataset=self.train_dataset,
                tokenizer=self.tokenizer,
                batch_size=args.per_device_train_batch_size,
                max_seq_length=args.max_seq_length,
                num_batches=total_batches_needed,
                seed=getattr(args, "seed", 42),
            )
            return batches, None

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
                )
            else:
                self._prepared_batches_include_epochs = vlm_num_epochs is not None
                batches = create_vlm_batches(
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
                if _vlm_mask_fn is not None and batches:
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
                    dataset_order=text_dataset_order,
                    comm_group=comm_group,
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
                        self._prepared_batches_include_epochs = True
                    batch_kwargs["completion_only_loss"] = text_completion_only_loss
                    batches = create_ordered_batches(**batch_kwargs)
                else:
                    batch_kwargs["completion_only_loss"] = text_completion_only_loss
                    batches = create_batches(**batch_kwargs)
                return batches, None

    def save_model(self, output_dir=None):
        """Save LoRA adapters or full merged model (if no LoRA)."""
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
            # under a LoRA-wrapped q_proj re-leaks the Studio reload bug). Uses
            # the shared filter to match save_trainable_adapters / _merged.
            if model_has_non_lora_trainable_params(self.model):
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


class MLXORPOTrainer(MLXTrainer):
    """ORPO trainer mirroring TRL's ORPOTrainer. Forces loss_type='orpo' so
    the class is authoritative regardless of the config passed."""
    def __init__(self, model, tokenizer, train_dataset, eval_dataset=None,
                 dataset_text_field=None, max_seq_length=None, packing=None,
                 data_collator=None, args=None, formatting_func=None, processor=None):
        if args is None:
            args = MLXORPOConfig()
        elif getattr(args, "loss_type", "sft") != "orpo":
            args.loss_type = "orpo"
        super().__init__(model, tokenizer, train_dataset, eval_dataset,
                         dataset_text_field, max_seq_length, packing,
                         data_collator, args, formatting_func, processor)


class MLXDPOTrainer(MLXTrainer):
    """DPO trainer mirroring TRL's DPOTrainer. Forces loss_type='dpo' so
    the class is authoritative regardless of the config passed."""
    def __init__(self, model, tokenizer, train_dataset, eval_dataset=None,
                 dataset_text_field=None, max_seq_length=None, packing=None,
                 data_collator=None, args=None, formatting_func=None, processor=None):
        if args is None:
            args = MLXDPOConfig()
        elif getattr(args, "loss_type", "sft") != "dpo":
            args.loss_type = "dpo"
        super().__init__(model, tokenizer, train_dataset, eval_dataset,
                         dataset_text_field, max_seq_length, packing,
                         data_collator, args, formatting_func, processor)


class MLXGRPOTrainer(MLXTrainer):
    """GRPO trainer mirroring TRL's GRPOTrainer. Forces loss_type='grpo'.

    Unlike SFT/DPO/ORPO (static data), GRPO generates rollouts on the fly:
    each step generates ``num_generations`` completions per prompt, scores
    them with ``reward_funcs``, computes group-relative advantages, and yields
    a (batch, lengths, advantages) tuple to the training loop. Rollout runs
    uncompiled (mlx-lm generation); only the grad step is compiled.

    reward_funcs: a callable or list of callables, each
        ``fn(completions, prompts=..., **kwargs) -> list[float]`` (TRL signature).
        Multiple reward functions are summed.
    """
    def __init__(self, model, tokenizer, train_dataset, reward_funcs,
                 eval_dataset=None, args=None, formatting_func=None, processor=None):
        if args is None:
            args = MLXGRPOConfig()
        elif getattr(args, "loss_type", "sft") != "grpo":
            args.loss_type = "grpo"
        # A base MLXTrainingConfig (or any non-GRPO config) coerced to
        # loss_type='grpo' lacks the GRPO-only fields (num_generations,
        # max_completion_length, ...). Fill them with the documented
        # MLXGRPOConfig defaults so rollout setup does not AttributeError.
        _grpo_defaults = MLXGRPOConfig()
        for _field in fields(MLXGRPOConfig):
            if not hasattr(args, _field.name):
                setattr(args, _field.name, getattr(_grpo_defaults, _field.name))
        if not isinstance(reward_funcs, (list, tuple)):
            reward_funcs = [reward_funcs]
        self.reward_funcs = list(reward_funcs)
        super().__init__(model, tokenizer, train_dataset, eval_dataset,
                         None, None, None, None, args, formatting_func, processor)

    def _grpo_prompts(self):
        """Extract prompt strings from the dataset (expects a 'prompt' column)."""
        prompts = []
        for ex in self.train_dataset:
            if "prompt" not in ex:
                raise ValueError("GRPO requires a 'prompt' column in the dataset.")
            prompts.append(ex["prompt"])
        return prompts

    def _grpo_render_prompt(self, prompt, hf):
        """Render a GRPO prompt to a string for rollout.

        Conversational prompts (a list of ``{"role","content"}`` message dicts,
        as passed by the GRPO notebooks) are rendered via the tokenizer's chat
        template with ``add_generation_prompt``, mirroring TRL's
        ``maybe_apply_chat_template`` (add_generation_prompt when the last role
        is 'user'; continue_final_message when it is 'assistant'). Plain-string
        prompts pass through unchanged (back-compat).
        """
        if (isinstance(prompt, list) and prompt
                and isinstance(prompt[0], dict)
                and "role" in prompt[0] and "content" in prompt[0]):
            last_role = prompt[-1]["role"]
            if last_role == "user":
                add_gen, cont = True, False
            elif last_role == "assistant":
                add_gen, cont = False, True
            else:
                raise ValueError(
                    f"GRPO chat prompt: invalid last-message role {last_role!r}; "
                    "expected 'user' or 'assistant'."
                )
            return hf.apply_chat_template(
                prompt, tokenize=False,
                add_generation_prompt=add_gen,
                continue_final_message=cont,
            )
        return prompt

    def _grpo_mean_kl(self, batch, lengths):
        """Eager masked-mean k3 KL for a GRPO batch at the CURRENT weights.

        Mirrors make_grpo_loss_fn's KL term: policy logp with adapters,
        reference logp via LoRA-disable, ``exp(d) - d - 1``, masked-mean over
        completion tokens then mean over the group. Used only for logging; runs
        eagerly OUTSIDE the compiled step. Toggles eval mode so dropout draws no
        RNG (keeps the training trajectory and the compiled step untouched).
        Returns a float, or None when KL is undefined for the config (no LoRA
        adapters, beta == 0, or reference_free).
        """
        import mlx.core as mx
        import mlx.nn as nn
        args = self.args
        if (getattr(args, "grpo_beta", 0.04) == 0.0
                or bool(getattr(args, "reference_free", False))):
            return None
        mods = [mod for _, mod in iter_mlx_lora_modules(self.model)]
        if not mods:
            return None

        def _logp_mask(b, ln):
            inp, tgt = b[:, :-1], b[:, 1:]
            logits = self.model(inp)
            steps = mx.arange(1, tgt.shape[1] + 1)
            mask = mx.logical_and(
                steps >= ln[:, 0:1], steps < ln[:, 1:]
            ).astype(mx.float32)
            return -nn.losses.cross_entropy(logits, tgt), mask

        was_training = self.model.training
        self.model.eval()
        try:
            pol, mask = _logp_mask(batch, lengths)
            saved = [md.scale for md in mods]
            try:
                for md in mods:
                    md.scale = 0.0
                ref, _ = _logp_mask(batch, lengths)
            finally:
                for md, s in zip(mods, saved):
                    md.scale = s
        finally:
            if was_training:
                self.model.train()
        d = ref - pol
        per_tok_kl = mx.exp(d) - d - 1
        denom = mx.maximum(mask.sum(-1), 1.0)
        kl = ((per_tok_kl * mask).sum(-1) / denom).mean()
        mx.eval(kl)
        return float(kl.item())

    def _grpo_rollout_generator(self):
        """Infinite generator: each next() does generate->reward->advantage
        and yields (batch, lengths, advantages). Runs uncompiled."""
        import inspect as _inspect
        from mlx_lm import batch_generate
        from mlx_lm.sample_utils import make_sampler
        # Newer mlx-lm (>= the return_token_ids addition) surfaces the exact
        # generated token IDs from batch_generate. Prefer them for scoring so the
        # GRPO loss/KL run on the sequence the policy actually sampled, rather
        # than on a decode->re-encode roundtrip of the completion text (which is
        # not identity: byte-fallback/cleanup tokens and prompt/completion
        # boundary merges can yield different IDs). Fall back to re-encoding on
        # older mlx-lm that lacks the flag.
        _supports_token_ids = (
            "return_token_ids" in _inspect.signature(batch_generate).parameters
        )
        args = self.args
        hf = _hf_encoding_tokenizer(self.tokenizer)
        pad_id = hf.eos_token_id if hf.eos_token_id is not None else 0
        N = args.num_generations
        # GRPO advantages are group-relative: (r - mean) / (std + eps). With a
        # single generation per prompt the group has one element, so mean == r
        # and std == 0, making every advantage exactly 0. The run would appear
        # to train but receive no reward-policy gradient (a silent no-op GRPO
        # objective). TRL's GRPOTrainer requires a group of at least 2 for the
        # same reason; fail fast instead of training a no-op.
        if N < 2:
            raise ValueError(
                "Unsloth: GRPO requires num_generations >= 2 (got "
                f"{N}). A single generation per prompt yields a zero-variance "
                "group, so every group-relative advantage is 0 and the policy "
                "receives no reward gradient. Set num_generations to 2 or more."
            )
        sampler = make_sampler(temp=getattr(args, "temperature", 1.0))
        prompts = self._grpo_prompts()
        # Full dataset rows, in the same order as prompts, so reward_kwargs can
        # pass through each row's other columns (e.g. 'answer'). prompt and
        # example are indexed by the SAME cycled index below to stay aligned.
        examples = list(self.train_dataset)
        idx = 0
        while True:
            j = idx % len(prompts)
            prompt = prompts[j]
            example = examples[j]
            idx += 1
            rendered = self._grpo_render_prompt(prompt, hf)
            # Chat templates (Llama/Qwen/Gemma-style) already emit a leading BOS
            # when rendering the prompt string, so tokenizing with the raw
            # encode()'s default add_special_tokens=True would prepend a SECOND
            # BOS and corrupt the rollout prompt distribution. Route through the
            # same double-BOS guard the SFT path uses (encode_mlx_text drops
            # add_special_tokens when the rendered text already starts with the
            # BOS token); plain-string prompts still get their single BOS.
            pids = encode_mlx_text(hf, rendered)
            # Rollout generation must sample from the deterministic policy, not a
            # dropout-perturbed one. The training loop has already put the model in
            # train() mode, so any LoRA/DoRA dropout layers are active, and mlx-lm's
            # batch_generate never toggles eval. Without this guard the sampled
            # completions -- and the log-probs the GRPO loss later scores them with
            # -- would come from a randomly masked sub-network, so the advantages
            # and the policy gradient would be computed for a different stochastic
            # policy than the one that generated the tokens. Generate in eval mode
            # and restore the prior training mode (mirrors _grpo_mean_kl /
            # _grpo_ref_kl). Dropout=0 models are unaffected (eval is a no-op there).
            was_training = self.model.training
            self.model.eval()
            try:
                _gen_kwargs = dict(
                    max_tokens=args.max_completion_length,
                    sampler=sampler, verbose=False,
                )
                if _supports_token_ids:
                    _gen_kwargs["return_token_ids"] = True
                resp = batch_generate(
                    self.model, self.tokenizer, prompts=[pids] * N,
                    **_gen_kwargs,
                )
            finally:
                if was_training:
                    self.model.train()
            comps = resp.texts
            # Exact generated IDs per completion (see _supports_token_ids note);
            # None when unavailable, in which case rows fall back to re-encoding.
            comp_ids = getattr(resp, "token_ids", None) if _supports_token_ids else None
            # rewards: sum across reward functions (TRL-style). Pass through the
            # dataset row's other columns as kwargs (repeated per generation to
            # align row-for-row with completions), mirroring
            # GRPOTrainer._calculate_rewards. 'prompt'/'completion'/
            # 'completion_ids' are excluded exactly as TRL does. completion_ids
            # are not passed to reward functions (and trainer_state is not
            # available: there is no transformers TrainerState here), so they are
            # left as follow-up rather than faked. Note the scoring path above
            # does consume the generated ids (comp_ids) when mlx-lm surfaces
            # them, so the loss runs on the sampled tokens.
            reward_kwargs = {
                k: [example[k]] * N
                for k in example
                if k not in ("prompt", "completion", "completion_ids")
            }
            total = [0.0] * N
            for rf in self.reward_funcs:
                vals = rf(completions=comps, prompts=[prompt] * N, **reward_kwargs)
                # TRL's reward-function contract is one score per completion:
                # len(vals) must equal N. A shorter list would leave the unfilled
                # positions at their initial 0.0 (silently fabricating rewards and
                # corrupting the group-relative advantages); a longer one would
                # index out of range below. Fail fast with a clear message.
                if len(vals) != N:
                    _rf_name = getattr(rf, "__name__", type(rf).__name__)
                    raise ValueError(
                        f"Unsloth: GRPO reward function {_rf_name!r} returned "
                        f"{len(vals)} scores for {N} completions. Each reward "
                        "function must return exactly one score per completion "
                        "(len(completions) == num_generations)."
                    )
                for i, v in enumerate(vals):
                    # TRL-style reward funcs may return None to skip a sample
                    # (multi-task rewards); TRL maps None to NaN and drops it
                    # from the sum. Skip None here instead of crashing on
                    # float(None).
                    if v is not None:
                        total[i] += float(v)
            rewards = mx.array(total)
            # TRL standardizes grouped rewards with the sample (Bessel-corrected)
            # std: torch.std defaults to unbiased=True. mx.std defaults to the
            # population std (ddof=0), which is systematically smaller and inflates
            # the advantage magnitude by sqrt(N/(N-1)) (~1.41x at num_generations=2,
            # ~1.15x at 4), mis-scaling the objective the PPO clip epsilon and KL
            # beta are calibrated against. Compute the ddof=1 std explicitly from
            # primitives (rather than mx.std(ddof=1)) so the expression is identical
            # under real mlx and the torch-backed test shim, whose std() signatures
            # differ. num_generations >= 2 is enforced above so N - 1 >= 1, and the
            # all-equal-rewards (std==0) case is still absorbed by the +1e-4.
            _mean = rewards.mean()
            _std = (((rewards - _mean) ** 2).sum() / (rewards.shape[0] - 1)) ** 0.5
            advantages = (rewards - _mean) / (_std + 1e-4)
            # build the concatenated group batch
            pe = len(pids)
            rows, lengths = [], []
            for i, c in enumerate(comps):
                if comp_ids is not None and comp_ids[i] is not None:
                    # Score the EXACT generated continuation: prompt ids followed
                    # by the sampled completion ids. This makes pe a true prefix
                    # length by construction and avoids the decode->re-encode
                    # roundtrip corrupting the tokens the policy generated.
                    comp = [int(t) for t in comp_ids[i]]
                    # mlx-lm's batch_generate strips the terminal EOS from the
                    # returned ids for a normally-terminating row (finish_reason
                    # == "stop"), keeping all max_completion_length tokens only
                    # when it truncated on length. TRL's GRPO completion mask is
                    # inclusive of that EOS (sequence_indices <= eos_idx), so the
                    # stop action gets advantage-weighted gradient/KL. Re-add the
                    # EOS when the row stopped normally (shorter than the cap) so
                    # the loss scores the model's probability of stopping and does
                    # not bias completion-length/EOS behavior. A row at the cap was
                    # truncated (no EOS emitted) and is left as-is, mirroring TRL's
                    # default mask_truncated_completions=False.
                    _eos_id = getattr(hf, "eos_token_id", None)
                    if _eos_id is not None and len(comp) < args.max_completion_length:
                        comp = comp + [int(_eos_id)]
                    full = (pids + comp)[: args.max_seq_length]
                else:
                    # Fallback (older mlx-lm without generated-id surfacing):
                    # guard the prompt+completion tokenization against the same
                    # double BOS as the prompt above, and with the SAME guard so
                    # the response boundary pe (from the guarded prompt encode)
                    # stays a true prefix length of this full sequence.
                    full = encode_mlx_text(hf, rendered + c)[: args.max_seq_length]
                rows.append(full)
                lengths.append([pe, len(full)])
            L = max(len(r) for r in rows)
            batch = mx.array([r + [pad_id] * (L - len(r)) for r in rows], dtype=mx.int32)
            yield batch, mx.array(lengths), advantages

    def _prepare_data(self, is_vlm):
        if is_vlm:
            raise ValueError("GRPO is not yet supported for VLM models on MLX.")
        # GRPO drives the loop with a rollout generator, not static batches.
        return None, self._grpo_rollout_generator()


def _create_labeled_batches(dataset, tokenizer, mask_fn, batch_size,
                            max_seq_length, formatting_func=None,
                            dataset_text_field="text", num_batches=None,
                            seed=42, chat_template=None,
                            model_name=None, model_type=None,
                            append_eos=True, dataset_order="default",
                            preserve_dataset_order=False,
                            num_epochs=None, return_dataset=False,
                            comm_group=None, distributed_pad_mode="cycle"):
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

    def _order_samples_for_epoch(items, epoch_idx):
        if preserve_dataset_order or dataset_order == "sequential":
            return list(items)
        if dataset_order == "torch_randperm":
            from .utils import _torch_randperm_order, _normalize_seed
            # Reseed per epoch (matches `create_ordered_batches`). Normalize a
            # None seed first so seed=None does not raise on the int add.
            order = _torch_randperm_order(
                len(items), _normalize_seed(seed) + epoch_idx
            )
            return [items[i] for i in order]
        # legacy default: length-sort once
        return sorted(items, key=lambda x: len(x[0]))

    # 3. Build `num_epochs` blocks so `batches[i % len]` cycle reseeds correctly.
    _n_epochs_materialize = (
        max(1, int(num_epochs)) if num_epochs is not None else 1
    )
    rng = random.Random(seed)
    batches = []
    global_batch_size = _distributed_global_batch_size(batch_size, comm_group)
    for epoch_idx in range(_n_epochs_materialize):
        epoch_items = _order_samples_for_epoch(all_items, epoch_idx)
        epoch_batches = []
        for start in range(0, len(epoch_items), global_batch_size):
            batch_items = epoch_items[start:start + global_batch_size]
            batch_items = _rank_slice_distributed_batch(
                batch_items,
                batch_size,
                comm_group=comm_group,
                pad_source=epoch_items,
                pad_mode=distributed_pad_mode,
            )
            if not batch_items:
                continue
            valid_items = [item for item in batch_items if item is not None]
            max_len = max((len(ids) for ids, _ in valid_items), default=2)
            # +1 for autoregressive shift (mlx-lm iterate_batches parity).
            padded_len = 1 + ((max_len + _PAD_MULTIPLE - 1) // _PAD_MULTIPLE) * _PAD_MULTIPLE
            padded_len = min(padded_len, max_seq_length)

            batch_ids = []
            batch_labels = []
            batch_lengths = []
            for item in batch_items:
                if item is None:
                    batch_ids.append([pad_id] * padded_len)
                    batch_labels.append([-100] * padded_len)
                    batch_lengths.append([0, 0])
                    continue
                ids, lbls = item
                L = min(len(ids), padded_len)
                pad_len = padded_len - L
                batch_ids.append(ids[:L] + [pad_id] * pad_len)
                batch_labels.append(lbls[:L] + [-100] * pad_len)
                # [start, end) matches loss masks in utils.py:360/:393/:429/:439.
                batch_lengths.append([1, L])

            epoch_batches.append((
                mx.array(batch_ids),
                mx.array(batch_lengths),
                mx.array(batch_labels),
            ))

        # 4. Legacy length-sort: shuffle batches so adjacent steps differ.
        if not _order_requested:
            rng.shuffle(epoch_batches)
        batches.extend(epoch_batches)

    # Limit if needed
    if num_batches is not None and len(batches) > num_batches:
        batches = batches[:num_batches]

    # Evaluate all tensors
    all_tensors = []
    for batch_arr, lengths_arr, labels_arr in batches:
        all_tensors.extend([batch_arr, lengths_arr, labels_arr])
    mx.eval(all_tensors)

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
    for batch_ids, batch_lengths, batch_labels in batches:
        labels_list = batch_labels.tolist()
        for row in labels_list:
            unique = set(row)
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
    """_check_all_masked for VLM batch dicts (a "labels" key, not a 3-tuple).

    As in the text path, in DDP ``batches`` is only this rank's shard, so the
    per-rank bad/good counts are all-summed before deciding. Otherwise a rank
    whose shard is entirely masked would raise ZeroDivisionError alone while
    peers advance to the first collective and hang."""
    seen_bad = 0
    seen_good = 0
    checked = 0
    for batch_dict in batches:
        labels = batch_dict.get("labels")
        if labels is None:
            continue
        labels_list = labels.tolist()
        for row in labels_list:
            unique = set(row)
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
        # Text path: tokenize, mask, and create batches now
        args = trainer.args
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
        train_dataset = trainer._train_dataset_for_batches()
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

        # Process eval dataset too
        if trainer.eval_dataset is not None:
            eval_batch_size = (
                getattr(args, "per_device_eval_batch_size", None)
                or args.per_device_train_batch_size
            )

            def _create_labeled_eval_batches(eval_dataset):
                """Build response-masked eval batches for one dataset split."""
                batches, response_masked_dataset = _create_labeled_batches(
                    dataset=eval_dataset,
                    tokenizer=_tokenizer,
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
                eval_batches = {}
                for key, value in trainer.eval_dataset.items():
                    split_batches, split_dataset = _create_labeled_eval_batches(value)
                    eval_batches[key] = split_batches
                    trainer.eval_dataset[key] = split_dataset
            else:
                eval_batches, trainer.eval_dataset = _create_labeled_eval_batches(
                    trainer.eval_dataset
                )
            trainer._eval_batches_labeled = eval_batches

        print(f"Unsloth: train_on_responses_only enabled "
              f"({len(batches)} batches prepared).")

    return trainer
