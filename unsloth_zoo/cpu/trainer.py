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

__all__ = ["install_cpu_training_defaults", "normalize_cpu_training_arguments", "cpu_packing_supported"]

import functools
import json
import os

import torch

from .runtime import cpu_bf16_supported, cpu_fp16_supported

# bitsandbytes 8-bit / paged optimizers have no CPU kernels; AdamW states for LoRA are tiny anyway.
_GPU_ONLY_OPTIMIZERS = (
    "adamw_8bit", "adamw_bnb_8bit", "paged_adamw_8bit", "paged_adamw_32bit", "paged_adamw",
    "adamw_apex_fused", "adafactor_8bit", "lion_8bit", "paged_lion_8bit",
    "paged_lion_32bit", "ademamix_8bit", "paged_ademamix_8bit", "paged_ademamix_32bit",
)


@functools.cache
def _cpu_adamw():
    """Fused AdamW has CPU kernels from torch 2.4; foreach is far slower over LoRA's many small tensors."""
    from packaging.version import Version
    return "adamw_torch_fused" if Version(torch.__version__.split("+")[0]) >= Version("2.4.0") else "adamw_torch"


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
        print(f"Unsloth (CPU): optim = {optim!r} needs a GPU, using {_cpu_adamw()!r}.")
        args.optim = _cpu_adamw()
    elif optim in ("adamw_torch", "adamw_torch_fused"):
        args.optim = _cpu_adamw()
    if getattr(args, "tf32", None):
        args.tf32 = None
    if hasattr(args, "dataloader_pin_memory"):
        args.dataloader_pin_memory = False
    # Packing / padding-free flatten a batch into one row with position_ids. transformers >= 4.53 on torch >= 2.6
    # turns those resets into a block-diagonal SDPA / eager mask; older stacks, and TRL's "wrapped" strategy (no
    # resets), attend across examples, so those keep TRL's padded batches.
    supported = cpu_packing_supported()
    wrapped = str(_name(getattr(args, "packing_strategy", "")) or "").lower() == "wrapped"
    if getattr(args, "padding_free", False) and not supported:
        print("Unsloth (CPU): padding_free needs transformers >= 4.53 and torch >= 2.6 to keep examples apart, "
              "turning it off.")
        args.padding_free = False
    for key in ("packing", "eval_packing"):
        if getattr(args, key, False) and (not supported or wrapped):
            reason = "the 'wrapped' packing strategy attends across examples without FlashAttention" if supported \
                else "keeping packed examples apart needs transformers >= 4.53 and torch >= 2.6 (or FlashAttention)"
            print(f"Unsloth (CPU): {key}: {reason}, turning it off.")
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


@functools.cache
def cpu_packing_supported():
    """transformers builds a block-diagonal mask from packed position_ids (masking_utils, 4.53+, needs torch 2.6), and
    TRL's padding-free collator leaves attention_mask out, which that detection requires. UNSLOTH_CPU_PACKING=0 opts out."""
    if os.environ.get("UNSLOTH_CPU_PACKING", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    try:
        from transformers import masking_utils
        if not hasattr(masking_utils, "find_packed_sequence_indices") or \
                not getattr(masking_utils, "_is_torch_greater_or_equal_than_2_6", False):
            return False
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
        batch = DataCollatorForLanguageModeling(pad_token_id = 0, padding_free = True)(
            [{"input_ids": [5, 6, 7]}, {"input_ids": [8, 9]}]
        )
        return "attention_mask" not in batch and torch.equal(batch["position_ids"][0], torch.tensor([0, 1, 2, 0, 1]))
    except Exception:
        return False


def _packed_examples_isolated(model):
    """A packed forward (position_ids reset, no attention_mask) gives the same logits as the examples alone: True
    when this model's attention keeps packed examples apart on CPU."""
    try:
        from .loader import _disable_bnb_cpu_weight_packing
        # bitsandbytes repacks 4-bit weights on a no-grad eval forward, which breaks their backward.
        _disable_bnb_cpu_weight_packing(model)
        config = getattr(model, "config", None)
        vocab = getattr(getattr(config, "get_text_config", lambda: config)(), "vocab_size", None) or 32
        device = next(model.parameters()).device
        generator = torch.Generator().manual_seed(0)
        a = torch.randint(1, max(2, min(vocab, 1000)), (1, 3), generator = generator).to(device)
        b = torch.randint(1, max(2, min(vocab, 1000)), (1, 4), generator = generator).to(device)
        position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3]], device = device)
        was_training = model.training
        model.eval()
        import contextlib
        set_stance = getattr(torch.compiler, "set_stance", None)
        # Compiled training layers would build an extra no-grad graph for these 3 calls.
        eager = set_stance("force_eager") if set_stance is not None else contextlib.nullcontext()
        try:
            with torch.no_grad(), eager:
                run = lambda **kw: model(use_cache = False, **kw).logits.float()  # noqa: E731
                separate = torch.cat([run(input_ids = a), run(input_ids = b)], dim = 1)
                packed = run(input_ids = torch.cat([a, b], dim = 1), position_ids = position_ids)
        finally:
            model.train(was_training)
        scale = separate.abs().amax().clamp(min = 1.0)
        return bool(torch.isfinite(packed).all()) and float((packed - separate).abs().amax() / scale) < 2e-2
    except Exception:
        return False


def _check_cpu_packing(model, training_args):
    """Per model: packing / padding_free stay on only when this model's attention keeps packed examples apart."""
    keys = [key for key in ("packing", "eval_packing", "padding_free") if getattr(training_args, key, False)]
    if not keys or not isinstance(model, torch.nn.Module):
        return None
    if _packed_examples_isolated(model):
        # Training forwards must not build a KV cache: a cache disables transformers' packed-sequence detection.
        config = getattr(model, "config", None)
        if config is not None:
            config.use_cache = False
        return True
    print(f"Unsloth (CPU): {type(model).__name__} attends across packed examples on CPU, turning {', '.join(keys)} off.")
    for key in keys:
        setattr(training_args, key, False)
    return False


def _patch_post_init(cls):
    original = cls.__post_init__
    if getattr(original, "_unsloth_cpu", False):
        return

    @functools.wraps(original)
    def __post_init__(self, *args, **kwargs):
        # TRL resolves bf16=None to not-fp16 before the base class validates it, so normalize fp16/optim before the
        # original runs and TRL's resolved bf16 after it.
        normalize_cpu_training_arguments(self)
        result = original(self, *args, **kwargs)
        if getattr(self, "bf16", False) and not cpu_bf16_supported():
            self.bf16 = False
        return result

    __post_init__._unsloth_cpu = True
    cls.__post_init__ = __post_init__


def _patch_cpu_fp16_accelerator(trainer_cls):
    """accelerate turns fp16 into a silent fp32 run on CPU (no autocast, no scaler); when
    normalize_cpu_training_arguments kept fp16 (no native bf16), enable CPU autocast and a CPU GradScaler before the
    model and optimizer are prepared."""
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


def _patch_max_seq_length_alias(config_cls):
    """TRL >= 0.20 dropped SFTConfig(max_seq_length), which notebooks and Studio pass: map it to max_length."""
    fields = getattr(config_cls, "__dataclass_fields__", {})
    if "max_seq_length" in fields or "max_length" not in fields:
        return
    original = config_cls.__init__
    if getattr(original, "_unsloth_cpu", False):
        return

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        if "max_seq_length" in kwargs:
            max_seq_length = kwargs.pop("max_seq_length")
            if max_seq_length is not None:
                kwargs["max_length"] = max_seq_length
        return original(self, *args, **kwargs)

    __init__._unsloth_cpu = True
    config_cls.__init__ = __init__


def _sft_loss_takes_num_items(trainer):
    """The CPU SFT loss (chunked CE or TRL's fused LM head) divides by num_items_in_batch itself, whatever the
    model's accepts_loss_kwargs says."""
    if getattr(trainer, "compute_loss_func", None) is not None or getattr(trainer, "label_smoother", None) is not None:
        return False
    loss_type = str(_name(getattr(trainer.args, "loss_type", None) or "nll"))
    model = trainer.model
    for candidate in (model, getattr(model, "base_model", None), getattr(getattr(model, "base_model", None), "model", None)):
        if candidate is not None and vars(candidate).get("_unsloth_cpu_chunked_ce", False) and loss_type == "nll":
            return True
    try:
        import trl.trainer.utils as trl_utils
    except Exception:
        return False
    return hasattr(trl_utils, "add_fused_lm_head") and loss_type in ("nll", "dft")


def _count_items_for_sft_loss(trainer):
    """Without the count each micro-batch loss is its own mean, so accumulation / DDP weight rows wrongly."""
    if getattr(trainer, "model_accepts_loss_kwargs", True) or not _sft_loss_takes_num_items(trainer):
        return False
    trainer.model_accepts_loss_kwargs = True
    return True


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
            # Configs write in-process as dataset_num_proc = 1, but datasets >= 4.1 builds a Pool(1) for it
            # (under spawn its child re-imports an unguarded __main__): stock TRL must see None while it maps.
            training_args = kwargs.get("args", args[1] if len(args) > 1 else None)
            if training_args is not None:
                _check_cpu_packing(kwargs.get("model", args[0] if args else None), training_args)
            num_proc = getattr(training_args, "dataset_num_proc", None)
            serial = isinstance(num_proc, int) and not isinstance(num_proc, bool) and 0 <= num_proc <= 1
            if serial:
                training_args.dataset_num_proc = None
            try:
                super().__init__(*args, **kwargs)
            finally:
                if serial:
                    training_args.dataset_num_proc = num_proc
            _count_items_for_sft_loss(self)

    # Pickled by reference: by value (TRL's tokenize map, dataset_num_proc > 1) fails on torch config modules.
    SFTTrainer.__module__ = __name__
    SFTTrainer.__qualname__ = "_CPUSFTTrainer"
    return SFTTrainer


def __getattr__(name):
    # Spawn workers unpickling the CPU class get it, not stock TRL's SFTTrainer.
    if name == "_CPUSFTTrainer":
        install_cpu_training_defaults()
        if name in globals():
            return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    if getattr(trl, "SFTConfig", None) is not None:
        _patch_max_seq_length_alias(trl.SFTConfig)
    _patch_sft_compute_loss(sft_trainer.SFTTrainer)
    from .ops import patch_trl_fused_lm_head
    patch_trl_fused_lm_head()
    trainer = _make_cpu_sft_trainer(sft_trainer.SFTTrainer)
    globals()["_CPUSFTTrainer"] = trainer
    for module in (sft_trainer, trl, sys.modules.get("trl.trainer")):
        if module is not None:
            setattr(module, "SFTTrainer", trainer)
