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
    "disable_use_cache",
    "restore_use_cache",
]


@torch.inference_mode
def fix_zero_training_loss(model, tokenizer, train_dataset):
    """Warn/raise when labels are all -100 (masked), which zeroes the loss."""
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


# Autocast subclasses cached by (base_class, compute_dtype) so pickle can resolve them.
_BF16_AUTOCAST_SUBCLASSES = {}


def _find_tensor_device_type(*values):
    """Device type of the first tensor found in args/kwargs (recurses dict/list/tuple)."""
    from collections.abc import Mapping
    stack = list(values)
    while stack:
        value = stack.pop()
        if torch.is_tensor(value):
            return value.device.type
        if isinstance(value, Mapping):
            stack.extend(value.values())
        elif isinstance(value, (tuple, list)):
            stack.extend(value)
    return None


def _call_forward_with_bf16_autocast(forward, model, args, kwargs, compute_dtype):
    """Run forward in autocast(compute_dtype); defer to an outer autocast or an
    unsupported device (e.g. meta). Device sniffed from inputs, else params, else cuda."""
    device_type = _find_tensor_device_type(*args, *kwargs.values())
    if device_type is None:
        try:
            device_type = next(model.parameters()).device.type
        except StopIteration:
            device_type = "cuda"
    # is_autocast_enabled raises on unsupported devices (meta), so check availability first.
    if not torch.amp.is_autocast_available(device_type):
        return forward(*args, **kwargs)
    if torch.is_autocast_enabled(device_type):
        return forward(*args, **kwargs)
    with torch.amp.autocast(device_type=device_type, dtype=compute_dtype):
        return forward(*args, **kwargs)


def _reconstruct_bf16_autocast_model(base_cls, compute_dtype, instance_mode=False):
    """Pickle/deepcopy reconstructor: rebuild the subclass from the importable base class
    so unpickling works in a fresh interpreter."""
    cls = _make_bf16_autocast_subclass(base_cls, compute_dtype, instance_mode)
    return cls.__new__(cls)


def _bf16_autocast_reduce(self):
    """__reduce__: serialize via the base class + nn.Module state, so no generated symbol
    is needed at unpickle."""
    cls = type(self)
    base_cls = cls.__dict__.get("_unsloth_autocast_base", cls.__mro__[1])
    compute_dtype = cls.__dict__.get("_unsloth_autocast_dtype", torch.bfloat16)
    instance_mode = cls.__dict__.get("_unsloth_autocast_instance_mode", False)
    getstate = getattr(self, "__getstate__", None)
    state = getstate() if getstate is not None else self.__dict__
    return (_reconstruct_bf16_autocast_model,
            (base_cls, compute_dtype, instance_mode), state)


def _bf16_autocast_instance_forward(self, *args, **kwargs):
    """Class-level forward (instance_mode): wrap the instance forward saved on
    self._unsloth_autocast_orig_forward. Module-scoped so it pickles by import path."""
    orig = self.__dict__["_unsloth_autocast_orig_forward"]
    compute_dtype = type(self).__dict__.get(
        "_unsloth_autocast_dtype", torch.bfloat16)
    return _call_forward_with_bf16_autocast(orig, self, args, kwargs, compute_dtype)


def _make_bf16_autocast_subclass(cls, compute_dtype, instance_forward=False):
    """Build/fetch a cached subclass of cls whose forward runs in autocast(compute_dtype).
    instance_forward=True wraps the per-instance forward instead of the base forward.
    Registered as a module-level symbol with __reduce__ via the base class, so the model
    pickles and torch.save loads in a fresh process."""
    cached = _BF16_AUTOCAST_SUBCLASSES.get((cls, compute_dtype, instance_forward))
    if cached is not None:
        return cached

    if instance_forward:
        # Per-instance forward unknown at class creation, so this rare path keeps a
        # generic signature (remove_unused_columns keeps all columns).
        _wrapped = _bf16_autocast_instance_forward
    else:
        _orig_forward = cls.forward

        @functools.wraps(_orig_forward)
        def _wrapped(self, *args, **kwargs):
            return _call_forward_with_bf16_autocast(
                lambda *a, **k: _orig_forward(self, *a, **k),
                self, args, kwargs, compute_dtype,
            )

    # Unique name for module-level registration; __name__ stays the base class so
    # save_pretrained records the base in architectures.
    pickle_name = cls.__name__ + "WithUnslothBf16Autocast"
    module = sys.modules[__name__]
    if hasattr(module, pickle_name) and getattr(module, pickle_name) is not None:
        pickle_name = f"{pickle_name}_{len(_BF16_AUTOCAST_SUBCLASSES)}"

    new_cls = type(pickle_name, (cls,), {
        "forward": _wrapped,
        "__module__": __name__,
        "__reduce__": _bf16_autocast_reduce,
        "_unsloth_autocast_base": cls,
        "_unsloth_autocast_dtype": compute_dtype,
        "_unsloth_autocast_instance_mode": instance_forward,
    })
    new_cls.__name__ = cls.__name__
    new_cls.__qualname__ = pickle_name
    setattr(module, pickle_name, new_cls)
    _BF16_AUTOCAST_SUBCLASSES[(cls, compute_dtype, instance_forward)] = new_cls
    return new_cls


def _wrap_forward_in_bf16_autocast(model, compute_dtype):
    """fp32 norm weights make the norm output fp32, tripping the next bf16 linear; wrap
    forward in autocast(compute_dtype) so matmul inputs downcast at the op boundary.
    Done by subclassing type(model) (not reassigning forward) so it survives
    deepcopy/pickle/torch.save and keeps the forward signature. Idempotent."""
    if compute_dtype in (None, torch.float32):
        return model
    if getattr(model, "_unsloth_bf16_autocast_wrapped", False):
        return model

    instance_forward = model.__dict__.get("forward")
    if instance_forward is not None:
        # An instance-level forward shadows a class override, so swapping __class__ alone
        # would not intercept it; move it off the instance and route via instance_mode so
        # forward stays a class attribute (picklable/deepcopy-safe).
        model._unsloth_autocast_orig_forward = instance_forward
        del model.__dict__["forward"]
        model.__class__ = _make_bf16_autocast_subclass(
            type(model), compute_dtype, instance_forward=True)
    else:
        model.__class__ = _make_bf16_autocast_subclass(type(model), compute_dtype)
    model._unsloth_bf16_autocast_wrapped = True
    return model


def _unwrap_forward_in_bf16_autocast(model):
    """Undo _wrap_forward_in_bf16_autocast so reusing a model across prepare modes
    leaves no stale autocast."""
    if not getattr(model, "_unsloth_bf16_autocast_wrapped", False):
        return model
    base_cls = getattr(type(model), "_unsloth_autocast_base", None)
    orig_forward = model.__dict__.pop("_unsloth_autocast_orig_forward", None)
    if base_cls is not None:
        model.__class__ = base_cls
    if orig_forward is not None:
        model.forward = orig_forward
    model._unsloth_bf16_autocast_wrapped = False
    return model


def _iter_configs(config):
    """Yield config and every nested transformers config (composite models
    like VLMs nest text_config / vision_config as attributes)."""
    try:
        from transformers import PreTrainedConfig
    except ImportError:
        from transformers import PretrainedConfig as PreTrainedConfig
    seen = set()
    stack = [config]
    while stack:
        cfg = stack.pop()
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        yield cfg
        for sub in vars(cfg).values():
            if isinstance(sub, PreTrainedConfig):
                stack.append(sub)


def disable_use_cache(model):
    """Set use_cache = False on every config of the model. KV cache is unused
    under gradient checkpointing. Original values are remembered on the model
    the first time so restore_use_cache can undo this for inference."""
    config = getattr(model, "config", None)
    if config is None:
        return
    originals = getattr(model, "_unsloth_use_cache_originals", None)
    record = originals is None
    if record:
        originals = []
    for cfg in _iter_configs(config):
        if getattr(cfg, "use_cache", None):
            if record:
                originals.append((cfg, cfg.use_cache))
            cfg.use_cache = False
    if record and originals:
        try:
            model._unsloth_use_cache_originals = originals
        except Exception:
            pass


def restore_use_cache(model):
    """Undo disable_use_cache by restoring the recorded use_cache values,
    e.g. when switching the model to inference. No-op if nothing was
    disabled. The record is kept so disable_use_cache can re-disable
    without re-recording when training resumes."""
    for cfg, value in getattr(model, "_unsloth_use_cache_originals", None) or ():
        cfg.use_cache = value


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
        # Full finetuning upcasts weights to float32; sync config dtype so the
        # KV cache matches and generation avoids a dtype mismatch in index_copy_().
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
    # Defer to an external norm dtype policy (_pre_set_compute_dtype); skip those params.
    _externally_managed_param_ids = set()
    # Also detect norms by owning-module class name (catches custom norms whose param
    # names lack a token, e.g. Gemma audio tower).
    _norm_class_re = re.compile(r"(?i)(rms_?norm|layer_?norm)")
    _norm_param_ids = set()
    for _, _module in model.named_modules():
        if hasattr(_module, "_pre_set_compute_dtype"):
            # External policy casts recursively, so all descendants are managed.
            for _p in _module.parameters(recurse=True):
                _externally_managed_param_ids.add(id(_p))
        if _norm_class_re.search(type(_module).__name__):
            for _, _p in _module.named_parameters(recurse=False):
                _norm_param_ids.add(id(_p))
    # Rollback switch (default off): 1 keeps norm weights at their loaded dtype (pre-fix).
    _disable_float32_norm_upcast = (
        os.environ.get("UNSLOTH_DISABLE_FLOAT32_UPCAST", "0") == "1")

    def _is_norm_parameter(nm, p):
        return (
            id(p) in _norm_param_ids
            or "norm." in nm
            or "_layernorm" in nm
            or "layer_norm" in nm
            or "norm1." in nm
            or "norm2." in nm
        )

    # Gate the bias branch to PEFT: non-PEFT nn.Linear biases default to
    # requires_grad=True and would all stay trainable on the LoRA path (#2343 review).
    _is_peft_model = hasattr(model, "peft_config")

    for name, param in model.named_parameters():
        original_name = name
        upcast = False
        requires_grad = False
        _keep_param_dtype = False
        _is_norm = _is_norm_parameter(original_name, param)
        if not full_finetuning:
            if ".lora_A." in name or ".lora_B." in name or ".lora_magnitude_vector" in name:
                upcast = True
                requires_grad = True
            elif (_is_peft_model and "bias" in name and param.requires_grad
                    and ".modules_to_save." not in name):
                # Respect PEFT's bias decision: bias="all"/"lora_only" marks biases
                # trainable; freezing them here disabled bias training (#2343).
                # _keep_param_dtype: keep the loaded dtype, since fp32 on a bf16/fp16
                # Linear breaks the matmul. modules_to_save is excluded so a saved head
                # with a frozen weight isn't partially trained via its bias (#2343 review).
                requires_grad = True
                _keep_param_dtype = True
            else:
                requires_grad = False
        else:
            # Norms need fp32 for adam writeback (~60% of bf16 norm updates round to
            # zero otherwise); a prior dangling else on train_lm_head had clobbered this.
            requires_grad = True
            upcast = False
            if (train_layernorms
                    and _is_norm
                    and id(param) not in _externally_managed_param_ids
                    and not _disable_float32_norm_upcast):
                upcast = True
        pass
        # Set training or not
        if requires_grad:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

        # Cast storage in place (keeps Parameter identity so tied weights stay tied);
        # skip externally-managed params (preserve their fp32 cast) and dtype-pinned
        # params (trainable biases must match their Linear weight).
        if (requires_grad
                and not _keep_param_dtype
                and id(param) not in _externally_managed_param_ids):
            dtype = torch.float32 if upcast else mixed_precision_dtype
            if param.dtype != dtype:
                param.data = param.data.to(dtype)
        pass

        # Legacy UNSLOTH_UPCAST_LAYERNORM path; fp32 norms it creates hit the wrapper gate below.
        if (_is_norm
                and id(param) not in _externally_managed_param_ids
                and os.environ.get("UNSLOTH_UPCAST_LAYERNORM", "0") == "1"):
            if param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
    pass

    # Wrap only if fp32 norms actually exist (our upcast, legacy env, or external policy);
    # gating on presence avoids wrapping a model that has none.
    _has_fp32_norms = any(
        _is_norm_parameter(nm, p) and p.dtype == torch.float32
        for nm, p in model.named_parameters()
    )
    if (full_finetuning
            and mixed_precision_dtype == torch.bfloat16
            and _has_fp32_norms):
        _wrap_forward_in_bf16_autocast(model, torch.bfloat16)
    else:
        _unwrap_forward_in_bf16_autocast(model)

    # Vanilla GC (True/False) requires undoing any prior Unsloth patch.
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

    # KV cache is unused under gradient checkpointing; disable it on every config.
    if use_gradient_checkpointing in (True, "unsloth"):
        disable_use_cache(model)

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
    model.training = True
    while hasattr(model, "model"):
        model = model.model
        model.training = True
    model.training = True
pass


def unset_training(model):
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
    Minimal trainer: a scaled-down HF Trainer that fixes gradient accumulation.
    Much less feature complete.
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
