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

from .runtime import cpu_bf16_supported

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
        args.fp16 = False
        args.bf16 = cpu_bf16_supported()
    elif getattr(args, "bf16", None) and not cpu_bf16_supported():
        # bf16 autocast without AMX / AVX512-BF16 / Arm BF16 is emulated and several times slower.
        args.bf16 = False
    optim = str(_name(getattr(args, "optim", "")) or "").lower()
    if optim in _GPU_ONLY_OPTIMIZERS or "8bit" in optim or "paged" in optim or "bnb" in optim:
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
    if getattr(args, "gradient_checkpointing", False):
        kwargs = dict(getattr(args, "gradient_checkpointing_kwargs", None) or {})
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


def _patch_sft_compute_loss(sft_trainer_cls):
    """Chunked CE returns no logits; TRL's entropy / token-accuracy metrics need them. Skip just those."""
    original = sft_trainer_cls.compute_loss
    if getattr(original, "_unsloth_cpu", False):
        return
    from transformers import Trainer

    @functools.wraps(original)
    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None, **kwargs):
        unwrapped = getattr(model, "base_model", model)
        inner = getattr(unwrapped, "model", unwrapped)
        if not getattr(inner, "_unsloth_cpu_chunked_ce", False) and not getattr(model, "_unsloth_cpu_chunked_ce", False):
            return original(self, model, inputs, return_outputs = return_outputs, num_items_in_batch = num_items_in_batch, **kwargs)
        inputs["use_cache"] = False
        return Trainer.compute_loss(self, model, inputs, return_outputs = return_outputs, num_items_in_batch = num_items_in_batch)

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
                training_args = kwargs.get("args", args[1] if len(args) > 1 else None)
                if training_args is None:
                    from trl import SFTConfig
                    training_args = kwargs["args"] = SFTConfig(output_dir = "outputs")
                for key, value in legacy.items():
                    setattr(training_args, _LEGACY_SFT_KWARGS[key], value)
            super().__init__(*args, **kwargs)

    SFTTrainer.__module__ = base.__module__
    return SFTTrainer


def install_cpu_training_defaults():
    """Idempotent. Only ever called from a CPU-backend process."""
    import sys
    from transformers import TrainingArguments
    _patch_post_init(TrainingArguments)
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
    trainer = _make_cpu_sft_trainer(sft_trainer.SFTTrainer)
    for module in (sft_trainer, trl, sys.modules.get("trl.trainer")):
        if module is not None:
            setattr(module, "SFTTrainer", trainer)
