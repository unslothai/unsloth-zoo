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

"""CPU defaults for transformers / TRL trainers. Installed only by the CPU backend."""

__all__ = ["install_cpu_training_defaults", "normalize_cpu_training_arguments"]

import functools
import json

import torch

from .runtime import cpu_bf16_supported, cpu_fp16_supported

# bitsandbytes 8-bit / paged optimizers have no CPU kernels; AdamW states for LoRA are tiny anyway.
_GPU_ONLY_OPTIMIZERS = (
    "adamw_8bit", "adamw_bnb_8bit", "paged_adamw_8bit", "paged_adamw_32bit", "paged_adamw",
    "adamw_torch_fused", "adamw_apex_fused", "adafactor_8bit", "lion_8bit", "paged_lion_8bit",
    "paged_lion_32bit", "ademamix_8bit", "paged_ademamix_8bit", "paged_ademamix_32bit",
)


def _name(value):
    return getattr(value, "value", value)


def normalize_cpu_training_arguments(args):
    args.use_cpu = True
    if getattr(args, "fp16", False):
        # bf16 needs no loss scaling, so it wins whenever it is native; fp16 autocast (with a CPU
        # GradScaler, see _patch_cpu_fp16_accelerator) only where fp16 is native and bf16 is not.
        if cpu_bf16_supported():
            args.fp16, args.bf16 = False, True
        elif not cpu_fp16_supported():
            args.fp16, args.bf16 = False, False
    elif getattr(args, "bf16", None) and not cpu_bf16_supported():
        # bf16 autocast without AMX / AVX512-BF16 / Arm BF16 is emulated and several times slower.
        args.bf16 = False
    optim = str(_name(getattr(args, "optim", "")) or "").lower()
    if optim in _GPU_ONLY_OPTIMIZERS or any(tag in optim for tag in ("8bit", "paged", "bnb", "npu", "xla")):
        print(f"Unsloth (CPU): optim = {optim!r} needs a GPU, using 'adamw_torch'.")
        args.optim = "adamw_torch"
    if getattr(args, "tf32", None):
        args.tf32 = None
    if hasattr(args, "dataloader_pin_memory"):
        args.dataloader_pin_memory = False
    # Packing / padding-free flatten a batch into one row with position_ids; SDPA on CPU then
    # attends across examples. Keep TRL's padded batches.
    if getattr(args, "padding_free", False):
        args.padding_free = False
    for key in ("packing", "eval_packing"):
        if getattr(args, key, False):
            print(f"Unsloth (CPU): {key} needs FlashAttention to keep packed examples apart, turning it off.")
            setattr(args, key, False)
    if getattr(args, "gradient_checkpointing", False):
        kwargs = getattr(args, "gradient_checkpointing_kwargs", None) or {}
        if isinstance(kwargs, str):
            # The CLI / JSON form, which TrainingArguments decodes only after this runs.
            kwargs = json.loads(kwargs)
        kwargs = dict(kwargs)
        kwargs.setdefault("use_reentrant", False)
        args.gradient_checkpointing_kwargs = kwargs
    return args


def _patch_post_init(cls):
    original = cls.__post_init__
    if getattr(original, "_unsloth_cpu", False):
        return

    @functools.wraps(original)
    def __post_init__(self, *args, **kwargs):
        # TRL resolves bf16=None to "not fp16" before the base class validates it, so normalise both
        # before (fp16/optim) and after (TRL's resolved bf16) the original runs.
        normalize_cpu_training_arguments(self)
        result = original(self, *args, **kwargs)
        if getattr(self, "bf16", False) and not cpu_bf16_supported():
            self.bf16 = False
        return result

    __post_init__._unsloth_cpu = True
    cls.__post_init__ = __post_init__


def _patch_cpu_fp16_accelerator(trainer_cls):
    """accelerate turns fp16 into a silent fp32 run on CPU (no autocast, no scaler). When
    normalize_cpu_training_arguments kept fp16 (native fp16, no native bf16), enable CPU autocast
    and a CPU GradScaler before the model and optimizer are prepared."""
    original = trainer_cls.create_accelerator_and_postprocess
    if getattr(original, "_unsloth_cpu", False):
        return

    @functools.wraps(original)
    def create_accelerator_and_postprocess(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        accelerator = getattr(self, "accelerator", None)
        if (
            accelerator is not None
            and getattr(self.args, "fp16", False)
            and accelerator.device.type == "cpu"
            and accelerator.state.mixed_precision == "fp16"
            and not accelerator.native_amp
        ):
            import torch
            accelerator.native_amp = True
            accelerator.scaler = torch.amp.GradScaler("cpu")
        return result

    create_accelerator_and_postprocess._unsloth_cpu = True
    trainer_cls.create_accelerator_and_postprocess = create_accelerator_and_postprocess


def _count_sft_tokens(trainer, model, inputs):
    """TRL's `num_tokens` bookkeeping from SFTTrainer.compute_loss, which the chunked path bypasses."""
    metrics = getattr(trainer, "_metrics", None)
    if metrics is None or not hasattr(trainer, "_total_train_tokens"):
        return
    mode = "train" if model.training else "eval"
    if mode == "train":
        mask = inputs.get("attention_mask", None)
        if mask is not None:
            trainer._total_train_tokens += trainer.accelerator.gather_for_metrics(mask.sum()).sum().item()
        elif inputs.get("position_ids", None) is not None:
            trainer._total_train_tokens += inputs["position_ids"].size(1)
    metrics[mode]["num_tokens"] = [trainer._total_train_tokens]


def _patch_sft_compute_loss(sft_trainer_cls):
    """Chunked CE returns no logits; TRL's entropy / token-accuracy metrics need them. Skip just those."""
    original = sft_trainer_cls.compute_loss
    if getattr(original, "_unsloth_cpu", False):
        return
    from transformers import Trainer

    @functools.wraps(original)
    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None, **kwargs):
        chunked = None
        unwrapped = self.accelerator.unwrap_model(model) if getattr(self, "accelerator", None) is not None else model
        for candidate in (unwrapped, getattr(unwrapped, "base_model", None), getattr(getattr(unwrapped, "base_model", None), "model", None)):
            # vars(), not getattr: PEFT forwards attribute reads to the inner model, but writes stay on the wrapper.
            if candidate is not None and vars(candidate).get("_unsloth_cpu_chunked_ce", False):
                chunked = candidate
                break
        if chunked is None:
            return original(self, model, inputs, return_outputs = return_outputs, num_items_in_batch = num_items_in_batch, **kwargs)
        if not torch.is_grad_enabled():
            # Evaluation / prediction: the stock forward returns logits, so keep TRL's entropy and accuracy metrics.
            return original(self, model, inputs, return_outputs = return_outputs, num_items_in_batch = num_items_in_batch, **kwargs)
        if str(_name(getattr(self.args, "loss_type", None) or "nll")) != "nll":
            # TRL's other SFT objectives (dft, ...) weight per-token logits: stock forward.
            return original(self, model, inputs, return_outputs = return_outputs, num_items_in_batch = num_items_in_batch, **kwargs)
        # Only this SFT step turns the chunked loss on; DPO / KTO / ORPO and plain forwards keep logits.
        inputs["use_cache"] = False
        chunked._unsloth_cpu_chunked_ce_active = True
        try:
            result = Trainer.compute_loss(self, model, inputs, return_outputs = return_outputs, num_items_in_batch = num_items_in_batch)
        finally:
            chunked._unsloth_cpu_chunked_ce_active = False
        _count_sft_tokens(self, model, inputs)
        return result

    compute_loss._unsloth_cpu = True
    sft_trainer_cls.compute_loss = compute_loss


# Arguments notebooks still pass to SFTTrainer itself. On GPU, unsloth's TRL rewrite accepts them;
# stock TRL (what the CPU backend uses) moved them onto SFTConfig.
_LEGACY_SFT_KWARGS = {
    "dataset_text_field": "dataset_text_field",
    "max_seq_length": "max_length",
    "packing": "packing",
    "dataset_num_proc": "dataset_num_proc",
}


def _make_cpu_sft_trainer(base):
    if getattr(base, "_unsloth_cpu", False):
        return base

    class SFTTrainer(base):
        _unsloth_cpu = True

        def __init__(self, *args, **kwargs):
            tokenizer = kwargs.pop("tokenizer", None)
            if tokenizer is not None and kwargs.get("processing_class") is None:
                kwargs["processing_class"] = tokenizer
            legacy = {k: kwargs.pop(k) for k in list(kwargs) if k in _LEGACY_SFT_KWARGS}
            if legacy:
                from trl import SFTConfig
                args = list(args)
                training_args = kwargs.get("args", args[1] if len(args) > 1 else None)
                if training_args is None:
                    training_args = SFTConfig(output_dir = "outputs")
                elif not isinstance(training_args, SFTConfig):
                    # TRL rebuilds a plain TrainingArguments from to_dict(), which drops attributes set here.
                    values = training_args.to_dict()
                    values["hub_token"] = getattr(training_args, "hub_token", None)
                    values.pop("push_to_hub_token", None)
                    training_args = SFTConfig(**values)
                if "args" in kwargs or len(args) <= 1:
                    kwargs["args"] = training_args
                else:
                    args[1] = training_args
                for key, value in legacy.items():
                    setattr(training_args, _LEGACY_SFT_KWARGS[key], value)
                normalize_cpu_training_arguments(training_args)
            super().__init__(*args, **kwargs)

    SFTTrainer.__module__ = base.__module__
    return SFTTrainer


def install_cpu_training_defaults():
    """Idempotent. Only ever called from a CPU-backend process."""
    import sys
    from transformers import Trainer, TrainingArguments
    _patch_post_init(TrainingArguments)
    if hasattr(Trainer, "create_accelerator_and_postprocess"):
        _patch_cpu_fp16_accelerator(Trainer)
    try:
        import trl
        from trl.trainer import sft_trainer
    except Exception:
        return
    for name in ("SFTConfig", "DPOConfig", "GRPOConfig", "ORPOConfig", "KTOConfig", "RewardConfig"):
        cls = getattr(trl, name, None)
        if cls is not None and "__post_init__" in vars(cls):
            _patch_post_init(cls)
    _patch_sft_compute_loss(sft_trainer.SFTTrainer)
    from .ops import patch_trl_fused_lm_head
    patch_trl_fused_lm_head()
    trainer = _make_cpu_sft_trainer(sft_trainer.SFTTrainer)
    for module in (sft_trainer, trl, sys.modules.get("trl.trainer")):
        if module is not None:
            setattr(module, "SFTTrainer", trainer)
