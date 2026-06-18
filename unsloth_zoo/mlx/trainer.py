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

from dataclasses import asdict, dataclass, is_dataclass
import concurrent.futures
import math
import os
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_reduce, tree_unflatten

_PAD_MULTIPLE = 32
SUPPORTED_MLX_OPTIMIZERS = ("adafactor", "adamw", "adam", "sgd", "muon", "lion")
SUPPORTED_MLX_LR_SCHEDULERS = ("linear", "cosine", "constant")

from .utils import (
    make_cce_loss_fn,
    make_baseline_loss_fn,
    make_vlm_cce_loss_fn,
    make_vlm_baseline_loss_fn,
    create_batches,
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
    iter_mlx_lora_modules,
    apply_gradient_checkpointing,
    remove_gradient_checkpointing,
    _is_vlm_model,
)
from .compile import (
    build_compile_policy,
    explain_compile_support,
    get_compile_qualification,
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


def _text_completion_only_loss_arg(args):
    """Resolve SFT-compatible completion-only loss defaults."""
    value = getattr(args, "completion_only_loss", None)
    if value is not None:
        return value
    if bool(getattr(args, "train_on_completions", False)):
        return True
    return None


def _normalize_mlx_optimizer_name(name):
    opt_name = str(name or "adamw").strip().lower()
    if opt_name not in SUPPORTED_MLX_OPTIMIZERS:
        supported = ", ".join(SUPPORTED_MLX_OPTIMIZERS)
        raise ValueError(
            f"Unsloth: Unsupported MLX optimizer {name!r}. "
            f"Supported optimizers: {supported}."
        )
    return opt_name


_NORM_OUTPUT_CAST_BASE_CLASSES = (nn.RMSNorm, nn.LayerNorm)
_NORM_OUTPUT_CAST_PATCHED_CLASSES = set()


def _part_is_norm(part: str) -> bool:
    # "norm" matches RMSNorm/LayerNorm/input_layernorm; ln_* covers GPT-2/GPT-OSS.
    return "norm" in part or part.startswith("ln_") or part == "ln_f"


def _is_norm_parameter_path(path) -> bool:
    parts = str(path).lower().split(".")
    return any(_part_is_norm(part) for part in parts[:-1])


def _is_norm_module_path(path) -> bool:
    return any(_part_is_norm(part) for part in str(path).lower().split("."))


def _has_norm_selected_floating_parameter(module_path, module) -> bool:
    try:
        parameters = module.parameters()
    except Exception:
        return False

    module_path_selected = _is_norm_module_path(module_path)
    try:
        for parameter_path, value in tree_flatten(parameters):
            if (
                hasattr(value, "dtype")
                and mx.issubdtype(value.dtype, mx.floating)
                and (
                    module_path_selected
                    or _is_norm_parameter_path(parameter_path)
                )
            ):
                return True
    except Exception:
        return False
    return False


def _has_floating_parameter(module) -> bool:
    try:
        parameters = module.parameters()
    except Exception:
        return False

    try:
        for _, value in tree_flatten(parameters):
            if hasattr(value, "dtype") and mx.issubdtype(value.dtype, mx.floating):
                return True
    except Exception:
        return False
    return False


def _has_parameterized_non_norm_children(module) -> bool:
    try:
        children = module.children()
    except Exception:
        return False

    try:
        for _, child in tree_flatten(children, is_leaf=nn.Module.is_module):
            if (
                isinstance(child, nn.Module)
                and "norm" not in type(child).__name__.lower()
                and _has_floating_parameter(child)
            ):
                return True
    except Exception:
        return False
    return False


def _norm_output_cast_input_dtype(args, kwargs):
    for value in args:
        if hasattr(value, "dtype"):
            return value.dtype
    for value in kwargs.values():
        if hasattr(value, "dtype"):
            return value.dtype
    return None


def _is_norm_output_cast_candidate(module_path, module) -> bool:
    """Return whether a custom module itself produces norm-like output."""
    norm_cls = type(module)
    if norm_cls in _NORM_OUTPUT_CAST_BASE_CLASSES:
        return True
    if "norm" not in norm_cls.__name__.lower():
        return False
    if _has_parameterized_non_norm_children(module):
        return False
    if not _has_norm_selected_floating_parameter(module_path, module):
        return False
    return True


def _iter_norm_output_cast_classes(model=None):
    norm_classes = []
    seen = set()

    for norm_cls in _NORM_OUTPUT_CAST_BASE_CLASSES:
        norm_classes.append(norm_cls)
        seen.add(norm_cls)

    if model is not None:
        try:
            named_modules = model.named_modules()
        except Exception:
            named_modules = ()
        for module_path, module in named_modules:
            if _is_norm_output_cast_candidate(module_path, module):
                norm_cls = type(module)
                if norm_cls not in seen:
                    norm_classes.append(norm_cls)
                    seen.add(norm_cls)

    return tuple(norm_classes)


def _set_norm_output_cast_to_input_dtype(enabled: bool, model=None) -> None:
    """Control whether norm outputs are cast back to activation dtype.

    Norm parameters can stay in fp32 for stability, but letting fp32 norm
    outputs flow through the rest of the graph promotes downstream
    intermediates and materially increases LoRA/QLoRA memory. Casting the
    result back matches PyTorch autocast behavior more closely: fp32 norm math,
    bf16/fp16 downstream activations.
    """
    # Sync Qwen3-VL vision-block patch with generic patcher (lazy import: cycle).
    try:
        from . import compile as _mlx_compile
        _mlx_compile.set_qwen3_vision_norm_cast_output(enabled)
    except Exception:
        pass

    norm_classes = list(_iter_norm_output_cast_classes(model))
    if not enabled:
        norm_classes.extend(
            norm_cls for norm_cls in _NORM_OUTPUT_CAST_PATCHED_CLASSES
            if norm_cls not in norm_classes
        )

    for norm_cls in norm_classes:
        patched = norm_cls in _NORM_OUTPUT_CAST_PATCHED_CLASSES
        if enabled:
            original_call = norm_cls.__call__
            if (
                patched
                or getattr(original_call, "_unsloth_norm_output_cast_wrapper", False)
            ):
                continue

            def norm_call_cast_output(self, *args, _original_call=original_call, **kwargs):
                input_dtype = _norm_output_cast_input_dtype(args, kwargs)
                out = _original_call(self, *args, **kwargs)
                if (
                    input_dtype is not None
                    and hasattr(out, "dtype")
                    and out.dtype != input_dtype
                ):
                    return out.astype(input_dtype)
                return out

            norm_call_cast_output._unsloth_norm_output_cast_wrapper = True
            norm_cls._unsloth_original_call = original_call
            norm_cls.__call__ = norm_call_cast_output
            norm_cls._unsloth_cast_output_to_input_dtype = True
            _NORM_OUTPUT_CAST_PATCHED_CLASSES.add(norm_cls)
        elif patched:
            original_call = getattr(norm_cls, "_unsloth_original_call", None)
            if original_call is not None:
                norm_cls.__call__ = original_call
            norm_cls._unsloth_original_call = None
            norm_cls._unsloth_cast_output_to_input_dtype = False
            _NORM_OUTPUT_CAST_PATCHED_CLASSES.discard(norm_cls)


def _normalize_mlx_scheduler_type(name):
    sched_type = str(name or "linear").strip().lower()
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
    assistant_token_id: int = 0  # Token ID marking start of assistant response
    vlm_chat_template: object = None  # Unsloth template name/tuple or raw Jinja string


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

        # Freeze non-LoRA params when LoRA is detected. Otherwise LayerNorm
        # weights stay trainable and adaptive optimizers NaN on step 1 (their
        # 1D second-moment init is numerically unstable).
        self._ensure_lora_frozen(model)

        # Training state
        self._global_step = 0
        self._train_loss_history = []
        self._grad_norm_history = []
        self._tokens_per_second_history = []
        self._peak_memory_history = []
        self._step_times = []
        self._batches = None  # Pre-created batches (skips internal batch creation)
        self._step_callbacks = []  # Callbacks called after each logged step
        self._eval_callbacks = []  # Callbacks called after each eval
        self.stop_requested = False  # Set True to stop training early

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

    def _build_schedule(self, total_steps):
        """Build LR schedule from config. Returns a callable or float."""
        lr = self.args.learning_rate
        warmup = self.args.warmup_steps
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

        for batch_data in eval_batches:
            if self.stop_requested:
                break
            if is_vlm:
                loss, ntoks = loss_fn(self.model, batch_data)
            else:
                batch, lengths, labels = batch_data
                loss, ntoks = loss_fn(self.model, batch, lengths, labels)
            all_losses += loss * ntoks
            ntokens += ntoks
            mx.eval(all_losses, ntokens)

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
                if self.stop_requested:
                    break
        else:
            all_losses, ntokens = self._evaluate_batch_totals(
                eval_batches, loss_fn, is_vlm=is_vlm,
            )

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

    def train(self, resume_from_checkpoint: str | None = None):
        """Run MLX-native training loop following mlx-lm's compiled-step pattern
        with gradient accumulation. Returns a dict of training metrics."""
        # Stash for _train_inner. None = fresh start, a path = resume.
        self._resume_from_checkpoint = resume_from_checkpoint
        args = self.args
        model = self.model
        cast_norm_output = bool(getattr(args, "cast_norm_output_to_input_dtype", True))
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
        _norm_cast_applied = False
        try:
            _set_norm_output_cast_to_input_dtype(cast_norm_output, model)
            _norm_cast_applied = True
            if cast_norm_output:
                print("Unsloth: Casting MLX norm outputs back to activation dtype.")
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
                        print(
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
                print(
                    "Unsloth: MLX Metal memory guard enabled "
                    f"({', '.join(parts)})."
                )

            # Apply gradient checkpointing if requested
            if args.gradient_checkpointing:
                apply_gradient_checkpointing(model)
                print("Unsloth: Using gradient checkpointing to reduce memory.")

            # Qwen3.5-specific fixes
            config = getattr(model, "_config", {})
            model_type = config.get("model_type", "") if isinstance(config, dict) else ""
            if "qwen3_5" in model_type:
                from .loader import _fix_qwen35_attention_cache, _disable_fused_mrope
                _fix_qwen35_attention_cache(model)
                _disable_fused_mrope(model)
                from ..gated_delta_vjp import patch_gated_delta, patch_gated_delta_vlm
                patch_gated_delta()
                patch_gated_delta_vlm()
            # Qwen2/2.5/3-VL language towers share the fused MRoPE kernel with
            # no VJP; flip it off so training takes the differentiable fallback.
            if any(t in model_type for t in ("qwen3_vl", "qwen2_vl", "qwen2_5_vl")):
                from .loader import _disable_fused_mrope
                _disable_fused_mrope(model)

            return self._train_inner()
        finally:
            if args.gradient_checkpointing:
                try:
                    remove_gradient_checkpointing(model)
                except Exception:
                    pass
            try:
                self._restore_memory_limits()
            except Exception:
                pass
            if _norm_cast_applied and cast_norm_output:
                # Undo the global norm-class patch; tolerate partial state.
                try:
                    _set_norm_output_cast_to_input_dtype(False, model)
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
                print(f"Unsloth: Using VLM CCE loss ({cce_backend}) for memory-efficient training.")
            else:
                loss_fn = make_vlm_baseline_loss_fn(
                    model,
                    assistant_token_id=_atid,
                    ignore_token_ids=_vlm_ignore_token_ids,
                )
                print("Unsloth: Using VLM standard cross-entropy loss.")
        else:
            if use_cce:
                loss_fn = make_cce_loss_fn(model)
                cce_backend = getattr(loss_fn, "_unsloth_cce_backend", "unknown")
                print(f"Unsloth: Using CCE loss ({cce_backend}) for memory-efficient training.")
            else:
                loss_fn = make_baseline_loss_fn()
                print("Unsloth: Using standard cross-entropy loss.")

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
        _resume_step = 0
        _resume_from = getattr(self, "_resume_from_checkpoint", None)
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
                # 3. Restore trainer scalars (step counter, loss history).
                ts = load_trainer_state(_resume_from)
                _resume_step = int(ts.get("global_step", 0))
                self._train_loss_history = list(ts.get("train_loss_history", []))
                print(
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
            print(f"Unsloth: LoRA+ enabled (ratio={lora_plus_ratio}).")

        embedding_lr = args.embedding_learning_rate
        main_lr = args.learning_rate
        # Ratio < 1 slows embeddings down; 0 = disabled
        use_embedding_lr = embedding_lr > 0 and main_lr > 0
        embedding_lr_ratio = embedding_lr / main_lr if use_embedding_lr else 1.0
        if use_embedding_lr:
            print(f"Unsloth: Embedding LR = {embedding_lr:.2e} "
                  f"(ratio={embedding_lr_ratio:.3f} of main LR {main_lr:.2e}).")

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
                print(
                    "Unsloth: max_grad_value is elementwise and overrides "
                    f"{', '.join(conflicts)}."
                )
        elif (
            max_grad_leaf_norm > 0
            and float(getattr(args, "max_grad_norm", 0.0) or 0.0) > 0
        ):
            print(
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

        # Unified step for VLM (dict batch) and text (tuple batch) training.
        def step_fn(batch_data, prev_state, do_update):
            if isinstance(batch_data, dict):
                (lvalue, toks), grad = loss_and_grad_fn(model, batch_data)
            else:
                (lvalue, toks), grad = loss_and_grad_fn(model, batch_data[0], batch_data[1], batch_data[2])

            if _direct_single_step_update:
                grad_norm = _apply_update_direct(grad)
                return lvalue, toks, None, grad_norm

            toks_f = toks.astype(mx.float32)
            grad_norm = mx.array(0.0, dtype=mx.float32)

            # Scale-and-accumulate per micro-batch, casting the scalar to each
            # leaf's dtype so bf16/fp16 grad trees avoid fp32 promotion.
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

            if do_update:
                grad_norm = _apply_update(grad, toks_f)
                return lvalue, toks, None, grad_norm

            grad = tree_map(mx.stop_gradient, grad)
            toks_f = mx.stop_gradient(toks_f)
            return lvalue, toks, (grad, toks_f), None

        compile_policy = build_compile_policy(args=args)
        _compile_decision = getattr(self, "_compile_decision", None)
        _use_compile = compile_policy.mode != "eager"
        if _use_compile and max_grad_norm > 0 and grad_accum > 1:
            print(
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
                print(
                    f"Unsloth: mx.compile disabled for VLM arch "
                    f"'{_compile_decision.arch}' during training; using eager mode "
                    f"({_compile_decision.reason})."
                )
                if getattr(model, "_unsloth_compile_explain", None):
                    print("Unsloth: Compile trace summary:")
                    for line in model._unsloth_compile_explain.splitlines():
                        print(f"  {line}")
                _use_compile = False
        if _use_compile:
            _uncompiled_step_fn = step_fn
            step_fn = mx.compile(step_fn, inputs=state, outputs=state)

        # Prepare eval batches
        eval_batches = None
        text_completion_only_loss = _text_completion_only_loss_arg(args)
        if args.eval_steps > 0 and self.eval_dataset is not None:
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
                            batch_size=args.per_device_train_batch_size,
                            max_seq_length=args.max_seq_length,
                            seed=args.seed,
                            response_mask_fn=_vlm_mask_fn,
                            formatting_func=self.formatting_func,
                            completion_only_loss=text_completion_only_loss,
                        )
                    return create_batches(
                        dataset=eval_dataset,
                        tokenizer=self.tokenizer,
                        batch_size=args.per_device_train_batch_size,
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
                print(f"Unsloth: Eval enabled every {args.eval_steps} steps "
                      f"({eval_batch_count} eval batches).")

        features = []
        if is_vlm:
            features.append("VLM")
        if use_cce:
            features.append("CCE")
        if args.gradient_checkpointing:
            features.append("GC")
        if _use_compile:
            features.append("compile")
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

        print(f"Unsloth: Training for {total_steps} steps, "
              f"BS={args.per_device_train_batch_size}, "
              f"grad_accum={grad_accum}, "
              f"seq_len={args.max_seq_length}")
        print(f"Unsloth: Features: {', '.join(features)}")
        if _compile_decision is not None and _compile_decision.setting_recommendations:
            print("Unsloth: Compile recommendations:")
            for rec in _compile_decision.setting_recommendations:
                print(f"  - {rec.setting}={rec.recommended_value!r}: {rec.reason}")

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

        for it in range(_resume_step * grad_accum + 1, total_steps * grad_accum + 1):
            if self.stop_requested:
                print("Unsloth: Stop requested — ending training early.")
                break

            tic = time.perf_counter()

            # Get next batch
            if batch_iter is not None:
                batch_data = next(batch_iter)
            else:
                batch_data = batches[batch_idx % len(batches)]
                batch_idx += 1

            do_update = (it % grad_accum == 0)
            if do_update:
                # Keep callable scheduler evaluation outside mx.compile. The
                # compiled step reads the scalar LR already in optimizer state.
                self._set_optimizer_lr_for_step(optimizer, it // grad_accum - 1)

            try:
                lvalue, toks, grad_accum_state, grad_norm = step_fn(
                    batch_data, grad_accum_state, do_update,
                )
            except (ValueError, RuntimeError) as e:
                _msg = str(e).lower()
                _is_compile_failure = _use_compile and (
                    "compile" in _msg
                    or "primitive" in _msg
                    or "trace" in _msg
                    or "eval" in _msg
                )
                if _is_compile_failure:
                    if _compile_decision is not None and not _compile_decision.fallback_allowed:
                        raise RuntimeError(
                            "Unsloth: strict mx.compile was enabled for this VLM "
                            "and runtime fallback is disabled."
                        ) from e
                    print(
                        "Unsloth: mx.compile failed at runtime; "
                        "falling back to eager mode."
                    )
                    step_fn = _uncompiled_step_fn
                    _use_compile = False
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
                print(
                    "Unsloth: skipping grad norm reporting for this MLX "
                    "optimizer/mode to avoid materializing the gradient graph."
                )
                _warned_skip_optimizer_state_grad_norm = True
            if int(toks.item()) == 0:
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
                train_loss = (losses / n_tokens).item()
                tok_count = n_tokens.item()
                trained_tokens += tok_count
                lr_val = optimizer.learning_rate.item()
                tokens_sec = tok_count / train_time if train_time > 0 else 0
                peak_mem = mx.get_peak_memory() / 1e9

                self._train_loss_history.append(train_loss)
                grad_norm_val = (
                    float(grad_norm.item())
                    if grad_norm is not None else None
                )
                if grad_norm_val is not None:
                    self._grad_norm_history.append(grad_norm_val)
                self._tokens_per_second_history.append(tokens_sec)
                self._peak_memory_history.append(peak_mem)
                self._step_times.append(train_time / steps if steps > 0 else 0)

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
                print(
                    f"  Step {current_step}/{total_steps} | "
                    f"Loss: {train_loss:.4f} | "
                    f"{grad_text}"
                    f"LR: {lr_val:.2e} | "
                    f"Tok/s: {tokens_sec:.0f} | "
                    f"Peak: {peak_mem:.2f} GB"
                )

                for cb in self._step_callbacks:
                    try:
                        cb(
                            current_step, total_steps, train_loss, lr_val,
                            tokens_sec, peak_mem, elapsed_total, trained_tokens,
                            grad_norm_val,
                        )
                    except Exception as e:
                        print(f"Unsloth: step callback error: {e}")

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
                print(
                    f"  Eval  {current_step}/{total_steps} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Perplexity: {ppl:.2f}"
                )
                for cb in self._eval_callbacks:
                    try:
                        cb(current_step, val_loss, ppl)
                    except Exception as e:
                        print(f"Unsloth: eval callback error: {e}")

            # Checkpointing
            if args.save_steps > 0 and current_step % args.save_steps == 0:
                ckpt_dir = f"{args.output_dir}/checkpoint-{current_step}"
                try:
                    save_trainable_adapters(model, ckpt_dir)
                except ValueError as e:
                    print(f"  Unsloth: skipped checkpoint ({e})")
                else:
                    # Also write optimizer + trainer state so resume_from_checkpoint
                    # can restore Adam moments, step counter, and loss history.
                    # Adapter save was successful -- treat the extra writes as
                    # best-effort: log on failure but don't undo the adapter save.
                    checkpoint_complete = False
                    try:
                        save_optimizer_state(optimizer, ckpt_dir)
                        save_trainer_state(
                            {
                                "global_step": current_step,
                                "train_loss_history": list(self._train_loss_history),
                            },
                            ckpt_dir,
                        )
                        checkpoint_complete = True
                    except Exception as e:
                        print(f"  Unsloth: checkpoint saved without resume state ({e})")
                    print(f"  Saved checkpoint to {ckpt_dir}")
                    if checkpoint_complete:
                        _prune_stale_checkpoints(args.output_dir, args.save_total_limit)

        total_time = time.perf_counter() - start_time
        avg_loss = (
            sum(self._train_loss_history) / len(self._train_loss_history)
            if self._train_loss_history else 0.0
        )

        print(f"\nUnsloth: Training complete! "
              f"Avg loss: {avg_loss:.4f} | "
              f"Total time: {total_time:.1f}s | "
              f"Steps: {total_steps} | "
              f"Tokens: {trained_tokens}")

        # Honor the documented save_steps=0 contract: save at end of training.
        try:
            self.save_model()
        except ValueError as e:
            print(f"Unsloth: skipped final save ({e})")
        else:
            print(f"Unsloth: Saved final adapters to {args.output_dir}")

        return {
            "train_loss": avg_loss,
            "train_runtime": total_time,
            "train_steps": total_steps,
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
        }

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

        if is_vlm:
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
                    dataset=self.train_dataset,
                    processor=processor,
                    config=config,
                    batch_size=args.per_device_train_batch_size,
                    max_seq_length=args.max_seq_length,
                    seed=args.seed,
                    response_mask_fn=_vlm_mask_fn,
                    formatting_func=self.formatting_func,
                    dataset_order=vlm_dataset_order,
                    completion_only_loss=text_completion_only_loss,
                )
            else:
                self._prepared_batches_include_epochs = vlm_num_epochs is not None
                batches = create_vlm_batches(
                    dataset=self.train_dataset,
                    processor=processor,
                    config=config,
                    batch_size=args.per_device_train_batch_size,
                    max_seq_length=args.max_seq_length,
                    num_batches=total_batches_needed,
                    seed=args.seed,
                    response_mask_fn=_vlm_mask_fn,
                    formatting_func=self.formatting_func,
                    dataset_order=vlm_dataset_order,
                    num_epochs=vlm_num_epochs,
                    completion_only_loss=text_completion_only_loss,
                )
                if _vlm_mask_fn is not None and batches:
                    _check_vlm_all_masked(batches)
                return batches, None
        else:
            chat_tmpl = getattr(args, "chat_template", None)
            if args.streaming:
                # Streaming has no index space; refuse explicit order requests.
                if (
                    getattr(args, "preserve_dataset_order", False)
                    or getattr(args, "dataset_order", "default") != "default"
                ):
                    raise ValueError(
                        "Unsloth MLX: preserve_dataset_order / dataset_order is not "
                        "supported with streaming=True for text training. Disable "
                        "streaming or materialize batches."
                    )
                return None, iterate_training_batches(
                    dataset=self.train_dataset,
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
                )
            else:
                batch_kwargs = dict(
                    dataset=self.train_dataset,
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
                    batches = create_ordered_batches(**batch_kwargs)
                else:
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

            # Copy base model's config.json so the checkpoint is loadable
            src_path = getattr(self.model, "_src_path", None)
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
                            num_epochs=None):
    """Create padded batches with label masks for train_on_responses_only.

    Tokenizes each dataset item, applies the masking closure to get labels,
    sorts by length, and produces right-padded 3-tuple batches.

    Returns:
        List of (batch, lengths, labels) tuples where:
        - batch: mx.array (BS, padded_len) — input_ids padded with 0
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
    for epoch_idx in range(_n_epochs_materialize):
        epoch_items = _order_samples_for_epoch(all_items, epoch_idx)
        epoch_batches = []
        for start in range(0, len(epoch_items), batch_size):
            batch_items = epoch_items[start:start + batch_size]
            if not batch_items:
                continue
            max_len = max(len(ids) for ids, _ in batch_items)
            # +1 for autoregressive shift (mlx-lm iterate_batches parity).
            padded_len = 1 + ((max_len + _PAD_MULTIPLE - 1) // _PAD_MULTIPLE) * _PAD_MULTIPLE
            padded_len = min(padded_len, max_seq_length)

            batch_ids = []
            batch_labels = []
            batch_lengths = []
            for ids, lbls in batch_items:
                L = min(len(ids), padded_len)
                pad_len = padded_len - L
                batch_ids.append(ids[:L] + [0] * pad_len)
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

    return batches


def _check_all_masked(batches, max_check=100):
    """Raise if all labels in the first N batches are -100 (mirrors
    fix_zero_training_loss from the HF path)."""
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


def _check_vlm_all_masked(batches, max_check=100):
    """_check_all_masked for VLM batch dicts (a "labels" key, not a 3-tuple)."""
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
    _tokenizer = tokenizer
    if _tokenizer is None and trainer is not None:
        _tokenizer = trainer.tokenizer
    if _tokenizer is None:
        raise ValueError(
            "Unsloth: A tokenizer must be provided either via the `tokenizer` "
            "kwarg or via trainer.tokenizer."
        )

    _tokenizer = _resolve_response_mask_tokenizer(_tokenizer)

    # Get masking closure from the HF/CUDA implementation
    mask_fn = _hf_train_on_responses_only(
        None,
        instruction_part=instruction_part,
        response_part=response_part,
        force_match=force_match,
        tokenizer=_tokenizer,
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
        batches = _create_labeled_batches(
            dataset=trainer.train_dataset,
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
        )

        # Safety check: detect all-masked labels early
        _check_all_masked(batches)
        trainer._prepared_batches_include_epochs = (
            labeled_num_epochs is not None
        )
        trainer._batches = batches

        # Process eval dataset too
        if trainer.eval_dataset is not None:
            def _create_labeled_eval_batches(eval_dataset):
                """Build response-masked eval batches for one dataset split."""
                return _create_labeled_batches(
                    dataset=eval_dataset,
                    tokenizer=_tokenizer,
                    mask_fn=mask_fn,
                    batch_size=args.per_device_train_batch_size,
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
                    preserve_dataset_order=bool(getattr(args, "preserve_dataset_order", False)),
                )

            if isinstance(trainer.eval_dataset, dict):
                eval_batches = {
                    key: _create_labeled_eval_batches(value)
                    for key, value in trainer.eval_dataset.items()
                }
            else:
                eval_batches = _create_labeled_eval_batches(trainer.eval_dataset)
            trainer._eval_batches_labeled = eval_batches

        print(f"Unsloth: train_on_responses_only enabled "
              f"({len(batches)} batches prepared).")

    return trainer
