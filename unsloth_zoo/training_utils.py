# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import math
import datasets
from transformers import set_seed as transformers_set_seed
from transformers import get_scheduler as transformers_get_scheduler
from transformers import Trainer
from transformers.trainer_utils import seed_worker as trainer_utils_seed_worker
from tqdm import tqdm as ProgressBar
import time
from typing import Any, Optional, List, Dict, Tuple
from .utils import _get_dtype, Version
from .hf_utils import dtype_from_config
from .gradient_checkpointing import (
    unpatch_unsloth_gradient_checkpointing,
    unpatch_unsloth_smart_gradient_checkpointing,
)
import os
import re
import sys
import functools

__all__ = [
    "fix_zero_training_loss",
    "unsloth_train",
    "prepare_model_for_training",
]


@torch.inference_mode
def fix_zero_training_loss(model, tokenizer, train_dataset):
    """
    Sometimes the labels get masked by all -100s, causing the loss
    to be 0. We check for this!
    """
    # All Unsloth Zoo code licensed under LGPLv3
    if isinstance(train_dataset, datasets.IterableDataset):
        # Skip the check since the code below assumes
        # an indexable dataset
        return
    
    if len(train_dataset) == 0: return


    row = train_dataset[0]
    if type(row) is dict and "labels" in row:

        # Check the first 100 rows
        seen_bad  = 0
        seen_good = 0
        for i, row in enumerate(train_dataset):
            try:    check_tokens = list(set(row["labels"]))
            except: continue
            if len(check_tokens) == 1 and check_tokens[0] == -100: seen_bad += 1
            else: seen_good += 1
            if i >= 100: break
        pass

        # Check ratio
        if seen_bad == 0 and seen_good == 0: return

        elif seen_bad / (seen_bad + seen_good) == 1:
            raise ZeroDivisionError(
                "Unsloth: All labels in your dataset are -100. Training losses will be all 0.\n"\
                "For example, are you sure you used `train_on_responses_only` correctly?\n"\
                "Or did you mask our tokens incorrectly? Maybe this is intended?\n"\
                "Maybe you're using a Llama chat template on a non Llama model for example?"\
                "If you used `train_on_responses_only`, confirm your user and assistant parts are correct!"
            )
        elif seen_bad / (seen_bad + seen_good) >= 0.9:
            print(
                "Unsloth: Nearly all labels in your dataset are -100. Training losses will be all 0.\n"\
                "For example, are you sure you used `train_on_responses_only` correctly?\n"\
                "Or did you mask our tokens incorrectly? Maybe this is intended?\n"\
                "Maybe you're using a Llama chat template on a non Llama model for example?"\
                "If you used `train_on_responses_only`, confirm your user and assistant parts are correct!"
            )
    pass
pass


# Cache of generated autocast subclasses, keyed by (base_class, compute_dtype).
# Caching keeps the subclass identity stable across instances of the same base
# class and lets us register each subclass exactly once as a module-level
# symbol so pickle / torch.save(model) can resolve it by module + qualname.
_BF16_AUTOCAST_SUBCLASSES = {}


def _make_bf16_autocast_subclass(cls, compute_dtype):
    """Build (or fetch from cache) a subclass of `cls` whose `forward` is
    wrapped in `torch.amp.autocast(compute_dtype)`.

    The subclass is registered as a module-level attribute of this module so
    that `pickle` (and therefore `torch.save(model, ...)`) can resolve it by
    `module + qualname`. Without that registration, assigning a runtime
    `type(...)` class to `model.__class__` makes the model unpicklable
    (`PicklingError: attribute lookup ... failed`).
    """
    cached = _BF16_AUTOCAST_SUBCLASSES.get((cls, compute_dtype))
    if cached is not None:
        return cached

    _orig_forward = cls.forward

    @functools.wraps(_orig_forward)
    def _wrapped(self, *args, **kwargs):
        device_type = "cuda"
        for t in args:
            if torch.is_tensor(t):
                device_type = t.device.type
                break
        else:
            for t in kwargs.values():
                if torch.is_tensor(t):
                    device_type = t.device.type
                    break
        # Order matters: is_autocast_enabled raises on unsupported device
        # types (e.g. "meta"); is_autocast_available returns False cleanly.
        if not torch.amp.is_autocast_available(device_type):
            return _orig_forward(self, *args, **kwargs)
        if torch.is_autocast_enabled(device_type):
            return _orig_forward(self, *args, **kwargs)
        with torch.amp.autocast(device_type=device_type, dtype=compute_dtype):
            return _orig_forward(self, *args, **kwargs)

    name = cls.__name__ + "WithUnslothBf16Autocast"
    module = sys.modules[__name__]
    # Disambiguate if two different base classes share the same __name__ so
    # each registered symbol resolves back to exactly one subclass.
    if hasattr(module, name) and getattr(module, name) is not None:
        name = f"{name}_{len(_BF16_AUTOCAST_SUBCLASSES)}"

    new_cls = type(name, (cls,), {"forward": _wrapped, "__module__": __name__})
    new_cls.__qualname__ = name
    setattr(module, name, new_cls)
    _BF16_AUTOCAST_SUBCLASSES[(cls, compute_dtype)] = new_cls
    return new_cls


def _wrap_forward_in_bf16_autocast(model, compute_dtype):
    """For bf16 full-FT with fp32 norm weights, the norm forward returns an
    fp32 tensor which then meets the next bf16 linear and trips
    `F.linear`'s dtype-equality check. Wrap `model.forward` in
    `torch.amp.autocast(compute_dtype)` so linear/matmul inputs are
    downcast at the op boundary -- the standard PyTorch mixed-precision
    pattern PEFT and Accelerate already rely on. Idempotent; defers to an
    outer autocast context if one is already active.

    Implementation notes:
      - `functools.wraps(_orig_forward)` preserves the model's forward
        signature so `inspect.signature(model.forward)` still reports
        the real parameter names. HF `Trainer._set_signature_columns_if_needed`
        reads that signature to decide which dataset columns to keep under
        `remove_unused_columns=True`.
      - The wrap is installed by subclassing `type(model)` and overriding
        `forward` on the new class, rather than reassigning `model.forward`
        to a closure. `copy.deepcopy(model)` (used by EMA / model averaging
        / `Trainer._save_optimizer_and_scheduler` checkpoint paths) uses
        `obj.__class__` to reconstruct the copy, so the subclass survives
        and the copy's `forward` correctly binds to the copy's `self`.
      - The subclass is cached and registered as a module-level symbol (see
        `_make_bf16_autocast_subclass`) so `pickle` / `torch.save(model)` can
        resolve it by `module + qualname` instead of raising `PicklingError`.
      - `is_autocast_available(device_type)` is probed BEFORE
        `is_autocast_enabled(device_type)` because the latter raises on
        unsupported device types (e.g. "meta" tensors during materialise),
        whereas `is_autocast_available` returns False cleanly.
    """
    if compute_dtype in (None, torch.float32):
        return model
    if getattr(model, "_unsloth_bf16_autocast_wrapped", False):
        return model

    model.__class__ = _make_bf16_autocast_subclass(type(model), compute_dtype)
    model._unsloth_bf16_autocast_wrapped = True
    return model


@torch.no_grad
def prepare_model_for_training(
    model                      : Any,
    use_gradient_checkpointing : Optional = "unsloth",
    use_reentrant              : Optional[bool] = True,
    full_finetuning            : Optional[bool] = False,
    train_layernorms           : Optional[bool] = False,
    train_embedding            : Optional[bool] = False,
    train_lm_head              : Optional[bool] = False,
    float32_mixed_precision    : Optional[bool] = True,
    patch_modules_to_save      : Optional[bool] = False,
) -> Any:
    # All Unsloth Zoo code licensed under LGPLv3
    assert(use_gradient_checkpointing in (True, False, "unsloth",))
    assert(type(use_reentrant) is bool)
    assert(type(full_finetuning) is bool)
    assert(type(train_layernorms) is bool)
    assert(type(train_embedding) is bool)
    assert(type(train_lm_head) is bool)
    assert(type(float32_mixed_precision) is bool)

    dtype = _get_dtype(dtype_from_config(model.config))
    mixed_precision_dtype = torch.float32
    if dtype == torch.float16:
        # We need to upcast to float32
        mixed_precision_dtype = torch.float32
        os.environ["UNSLOTH_MIXED_PRECISION"] = "float32"
        # For full finetuning, update config dtype to match actual weight dtype.
        # The KV cache uses model.config.torch_dtype, but weights are upcast to float32.
        # Without this, generation fails with dtype mismatch in index_copy_().
        if full_finetuning:
            model._unsloth_original_dtype = dtype
            model.config.torch_dtype = torch.float32
    elif dtype == torch.bfloat16 and float32_mixed_precision:
        mixed_precision_dtype = torch.float32
        os.environ["UNSLOTH_MIXED_PRECISION"] = "float32"
        if full_finetuning:
            model._unsloth_original_dtype = dtype
            model.config.torch_dtype = torch.float32
    elif dtype == torch.bfloat16:
        mixed_precision_dtype = torch.bfloat16
        os.environ["UNSLOTH_MIXED_PRECISION"] = "bfloat16"
    else:
        mixed_precision_dtype = torch.float32
        os.environ["UNSLOTH_MIXED_PRECISION"] = "float32"
    pass
    # Defer to any external compute-dtype policy already in effect on norm
    # modules (e.g. UNSLOTH_HIGH_PRECISION_LAYERNORM which tags norm modules
    # with `_pre_set_compute_dtype`). Record those parameter ids so we leave
    # them alone instead of overwriting that policy.
    _externally_managed_param_ids = set()
    for _, _module in model.named_modules():
        if hasattr(_module, "_pre_set_compute_dtype"):
            for _, _p in _module.named_parameters(recurse=False):
                _externally_managed_param_ids.add(id(_p))
    # Rollback switch (defaults off = corrected behaviour). Set to 1 to keep
    # norm weights at their loaded dtype like the pre-fix code path.
    _disable_float32_norm_upcast = (
        os.environ.get("UNSLOTH_DISABLE_FLOAT32_UPCAST", "0") == "1")

    for name, param in model.named_parameters():
        upcast = False
        requires_grad = False
        if not full_finetuning:
            if ".lora_A." in name or ".lora_B." in name or ".lora_magnitude_vector" in name:
                upcast = True
                requires_grad = True
            else:
                requires_grad = False
        else:
            # Full finetuning: train everything by default at compute dtype.
            # Norm weights must be upcast to float32 for adam writeback
            # precision -- without this ~60% of bf16 adam updates round to
            # zero on writeback (measured on Qwen3-0.6B input_layernorm).
            # Previously a dangling `else:` attached to `if train_lm_head:`
            # silently clobbered the `upcast = True` set for norms above it.
            requires_grad = True
            upcast = False
            # Norm-name matcher catches the patterns we've seen in the wild:
            #   Llama/Qwen3:        "input_layernorm", "model.norm",
            #                       "self_attn.q_norm", "self_attn.k_norm"
            #   SigLIP/CLIP vision: "encoder.layers.0.layer_norm1/2"
            #   ViT/DINO/Qwen3-VL:  "visual.blocks.0.norm1/2"
            _is_norm_name = (
                "norm." in name
                or "_layernorm" in name
                or "layer_norm" in name
                or "norm1." in name
                or "norm2." in name
            )
            if (train_layernorms
                    and _is_norm_name
                    and id(param) not in _externally_managed_param_ids
                    and not _disable_float32_norm_upcast):
                upcast = True
        pass
        # Set training or not
        if requires_grad:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

        # Upcast to float32 if needed. Skip params owned by a module that
        # already has `_pre_set_compute_dtype` set (the external compute-dtype
        # policy will have already cast them) so we don't silently downcast
        # their fp32 storage back to the compute dtype here.
        if requires_grad and id(param) not in _externally_managed_param_ids:
            name = name.replace("base_model", "model", 1)
            while re.search(r'\.(\d+)\.', name) is not None:
                name = re.sub(r'\.(\d+)\.', r'[\1].', name)
            name = name.replace(".weight", "", 1)
            dtype = torch.float32 if upcast else mixed_precision_dtype
            try:
                # Try original name
                exec(f"{name}.to({str(dtype)})")
            except:
                # Maybe model.model
                exec(f"model.{name}.to({str(dtype)})")
        pass

        if ('norm.' in name or '_layernorm' in name) and os.environ.get("UNSLOTH_UPCAST_LAYERNORM", "0") == "1":
            try:
                name = name.replace("base_model", "model", 1)
                while re.search(r'\.(\d+)\.', name) is not None:
                    name = re.sub(r'\.(\d+)\.', r'[\1].', name)
                name = name.replace(".weight", "", 1)
                # Try original name
                exec(f"{name}.to({str(torch.float32)})")
            except:
                # Maybe model.model
                exec(f"model.{name}.to({str(torch.float32)})")
    pass

    # When the bf16 full-FT path now has fp32 norm weights, those norms'
    # forward returns fp32 tensors (e.g. `weight * downcast_hidden` in
    # transformers' Llama/Qwen3 RMSNorm). Without an autocast context, that
    # fp32 then enters the next bf16 linear and trips F.linear's dtype check.
    # Wrap forward in torch.amp.autocast(bf16) so linear/matmul inputs get
    # downcast at the op boundary -- the canonical PyTorch mixed-precision
    # pattern HF / PEFT / Accelerate all use. Cheap no-op when not needed.
    if (full_finetuning and not _disable_float32_norm_upcast
            and mixed_precision_dtype == torch.bfloat16):
        _wrap_forward_in_bf16_autocast(model, torch.bfloat16)

    # Gradient checkpointing
    # If the user requested vanilla GC (True/False), ensure any prior Unsloth patch is undone.
    if use_gradient_checkpointing != "unsloth":
        unpatch_unsloth_gradient_checkpointing()
        unpatch_unsloth_smart_gradient_checkpointing()
    m = model
    while hasattr(m, "model"):
        if use_gradient_checkpointing == "unsloth":
            m._offloaded_gradient_checkpointing = True
        if use_gradient_checkpointing == True and hasattr(m, "gradient_checkpointing_enable"):
            m.gradient_checkpointing_enable()
        m = m.model
    pass
    if use_gradient_checkpointing == "unsloth":
        m._offloaded_gradient_checkpointing = True
    if use_gradient_checkpointing == True and hasattr(m, "gradient_checkpointing_enable"):
        m.gradient_checkpointing_enable()

    # Also set HF version manually to stop failures
    if hasattr(model, "_set_gradient_checkpointing"):
        if use_gradient_checkpointing in (True, "unsloth"):
            model._set_gradient_checkpointing()
        else:
            # Ensure checkpointing stays disabled if explicitly requested.
            for module in model.modules():
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = False

    # If use_reentrant = True which is the Pytorch default, we just make the input requires_grad.
    if use_reentrant:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    pass

    # Upcast modules_to_save
    if patch_modules_to_save:
        try:
            from peft.utils import ModulesToSaveWrapper
        except:
            ModulesToSaveWrapper = None

        for name, module in model.named_modules():
            if type(module) is ModulesToSaveWrapper or "ModulesToSave" in name:
                if getattr(module, "original_module", None) is not None:
                    module.original_module.requires_grad_(False)
                if getattr(module, "modules_to_save", None) is not None:
                    for saved_module in module.modules_to_save.modules():
                        if hasattr(saved_module, "weight"):
                            if saved_module.weight.dtype == torch.float16:
                                print(f"Unsloth: Upcasting `{name}` from float16 to float32 since it's in `modules_to_save`. Also allowing gradients.")
                                saved_module.to(torch.float32)
                                saved_module.requires_grad_(True)
                            else:
                                print(f"Unsloth: Allowing gradients for `{name}` since it's in `modules_to_save`.")
                                saved_module.requires_grad_(True)
                    pass
                pass
            pass
        pass
    pass

    return model
pass


def get_max_steps(training_args, n_training_samples, train_dataset):
    # Approximately from https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L2092
    # Determines batch size, max steps, ga etc
    if training_args.world_size > 1:
        raise RuntimeError('Unsloth currently does not support multi GPU setups - but we are working on it!')
    pass

    bsz = training_args.per_device_train_batch_size
    ga  = training_args.gradient_accumulation_steps

    total_train_batch_size = bsz * ga
    max_steps = training_args.max_steps

    if max_steps > 0:
        total_samples_seen = total_train_batch_size * max_steps
        num_train_epochs = math.ceil(total_samples_seen / n_training_samples)
    else:
        num_train_epochs = training_args.num_train_epochs
        steps_per_epoch  = math.ceil(n_training_samples / total_train_batch_size)
        max_steps = math.ceil(steps_per_epoch * num_train_epochs)
        num_train_epochs = math.ceil(num_train_epochs)
    return total_train_batch_size, max_steps, num_train_epochs
pass


def set_training(model):
    # Start training
    model.training = True
    while hasattr(model, "model"):
        model = model.model
        model.training = True
    model.training = True
pass


def unset_training(model):
    # End training
    model.training = False
    while hasattr(model, "model"):
        model = model.model
        model.training = False
    model.training = False
pass


from dataclasses import dataclass
@dataclass
class Trainer_Stats:
    metrics: dict
pass

def unsloth_train(trainer):
    """
    Unsloth Trainer
    1. Fixes gradient accumulation
    2. Scaled down version of HF's trainer
    3. Much less feature complete
    """
    # All Unsloth Zoo code licensed under LGPLv3
    assert(hasattr(trainer, "args"))
    assert(hasattr(trainer, "model"))
    assert(hasattr(trainer, "train_dataset"))
    assert(hasattr(trainer, "data_collator"))

    model = trainer.model
    training_args = trainer.args
    data_collator = trainer.data_collator
    n_training_samples = len(trainer.train_dataset)
    set_training(model)
    transformers_set_seed(training_args.seed)

    if training_args.dataloader_drop_last:
        raise NotImplementedError(
            "Unsloth: Currently `dataloader_drop_last` is not yet implemented!"
        )
    pass

    if data_collator is None:
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer = trainer.tokenizer,
            mlm = False,
            pad_to_multiple_of = 4,
        )
    pass

    # Separate weight decay for parameters
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    decay_parameters = frozenset(Trainer.get_decay_parameter_names(None, model))
    yes_decay, no_decay = [], []
    n_parameters_to_train = 0
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name in decay_parameters: yes_decay.append(param)
        else: no_decay.append(param)
        n_parameters_to_train += param.numel()
    pass
    optimizer_grouped_parameters = [
        {"params" : yes_decay, "weight_decay" : training_args.weight_decay,},
        {"params" : no_decay,  "weight_decay" : 0,}
    ]
    trainable_parameters = \
        optimizer_grouped_parameters[0]["params"] + \
        optimizer_grouped_parameters[1]["params"]
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    total_train_batch_size, max_steps, num_train_epochs = \
        get_max_steps(training_args, n_training_samples, trainer.train_dataset)

    # Get LR scheduler
    lr_scheduler = transformers_get_scheduler(
        name = training_args.lr_scheduler_type,
        optimizer = optimizer,
        num_warmup_steps = training_args.get_warmup_steps(max_steps),
        num_training_steps = max_steps,
        **getattr(training_args, "lr_scheduler_kwargs", {}),
    )

    # Gradient accumulation and grad norm clipping
    max_grad_norm   = training_args.max_grad_norm
    clip_grad_norm_ = torch.nn.utils.clip_grad_norm_
    bsz = training_args.per_device_train_batch_size
    ga  = training_args.gradient_accumulation_steps
    # inverse_gradient_accumulation_steps = 1.0 / ga
    # inverse_gradient_accumulation_steps = \
    #     torch.FloatTensor([inverse_gradient_accumulation_steps])\
    #     .to(device = "cuda:0", non_blocking = True)[0]

    # Mixed precision scaling
    torch_version = torch.__version__
    config_dtype = dtype_from_config(model.config)
    if config_dtype == torch.float16:
        mixed_precision = "fp16"
        mixed_dtype = torch.float16
        # torch.cuda.amp.autocast is deprecated >= 2.4
        if Version(torch_version) < Version("2.4.0"):
            float16_scaler = torch.cuda.amp.GradScaler()
        else:
            float16_scaler = torch.amp.GradScaler("cuda")
    else:
        mixed_precision = "bf16"
        mixed_dtype = torch.bfloat16
        float16_scaler = None
    pass
    
    optimizer.zero_grad()

    # torch.cuda.amp.autocast is deprecated >= 2.4
    torch_version = torch.__version__
    if Version(torch_version) < Version("2.4.0"):
        autocast_context_manager = torch.cuda.amp.autocast(
            dtype = mixed_dtype,
            cache_enabled = False,
        )
    else:
        autocast_context_manager = torch.amp.autocast(
            device_type = "cuda",
            dtype = mixed_dtype,
            cache_enabled = False,
        )
    pass

    step = 0
    accumulated_loss = torch.zeros(1, device = "cuda:0", dtype = torch.float32)[0]
    debug_info = \
        f'==((====))==  Unsloth - 2x faster free finetuning | Num GPUs = {training_args.world_size}\n'\
        f'    \\   /|    Num examples = {n_training_samples:,} | Num Epochs = {num_train_epochs:,}\n'\
        f'O^O/ \\_/ \\    Batch size per device = {training_args.per_device_train_batch_size:,} | Gradient Accumulation steps = {training_args.gradient_accumulation_steps}\n'\
        f'\\        /    Total batch size = {total_train_batch_size:,} | Total steps = {max_steps:,}\n'\
        f' "-____-"     Number of trainable parameters = {n_parameters_to_train:,}'
    print(debug_info)

    # Get per epoch counter
    max_iters_per_epoch = math.ceil(n_training_samples / total_train_batch_size)
    leftover_samples = n_training_samples % total_train_batch_size
    # But also consider leftover steps
    leftover_ga = math.ceil(leftover_samples / bsz)
    if leftover_samples == 0: leftover_ga = ga

    logging_steps = training_args.logging_steps
    # Go through each epoch
    start_time = time.time()
    with ProgressBar(total = max_steps, dynamic_ncols = True) as progress_bar:
        for epoch in range(num_train_epochs):

            # We also need to shuffle the data loader every epoch!
            transformers_set_seed(training_args.seed + epoch)
            train_dataloader_iterator = iter(torch.utils.data.DataLoader(
                trainer.train_dataset,
                batch_size     = bsz,
                sampler        = torch.utils.data.SequentialSampler(trainer.train_dataset),
                num_workers    = training_args.dataloader_num_workers,
                collate_fn     = data_collator,
                pin_memory     = training_args.dataloader_pin_memory,
                drop_last      = training_args.dataloader_drop_last,
                worker_init_fn = trainer_utils_seed_worker,
            ))

            for j in range(max_iters_per_epoch):
                n_batches = leftover_ga if j == (max_iters_per_epoch-1) else ga
                batches = [next(train_dataloader_iterator) for j in range(n_batches)]

                # Count non zeros before loss calc
                n_items = torch.stack([
                    torch.count_nonzero(x["labels"][..., 1:] != -100) for x in batches
                ]).sum()

                # Gradient accumulation
                for batch in batches:
                    input_ids = batch["input_ids"].pin_memory().to(device = "cuda:0", non_blocking = True)
                    labels    = batch["labels"]   .pin_memory().to(device = "cuda:0", non_blocking = True)

                    with autocast_context_manager:
                        loss = model(input_ids = input_ids, labels = labels, n_items = n_items).loss
                        # loss = loss * inverse_gradient_accumulation_steps
                        accumulated_loss += loss.detach()
                    pass

                    if float16_scaler is None:  loss.backward()
                    else: float16_scaler.scale(loss).backward()
                pass

                if float16_scaler is None:
                    clip_grad_norm_(trainable_parameters, max_grad_norm)
                    optimizer.step()
                else:
                    float16_scaler.unscale_(optimizer)
                    clip_grad_norm_(trainable_parameters, max_grad_norm)
                    float16_scaler.step(optimizer)
                    float16_scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

                if step % logging_steps == 0:
                    progress_bar.write(f"{step}, {round(accumulated_loss.cpu().item(), 4)}")
                pass
                accumulated_loss.zero_()
                progress_bar.update(1)

                step += 1
                if step == max_steps: break
            pass
        pass
    pass
    unset_training(model)
    print("Unsloth: Finished training!")
    end_time = time.time()

    # Return stats
    trainer_stats = Trainer_Stats(metrics = {"train_runtime" : end_time - start_time})
    return trainer_stats
pass

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
