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
from packaging.version import Version
import os
import math
import functools
from typing import Optional
torch_nn_functional_cross_entropy = torch.nn.functional.cross_entropy
from triton import __version__ as triton_version
from . import DEVICE_TYPE
from .temporary_patches.common import UNSLOTH_ENABLE_LOGGING, torch_compile_options, logger
import inspect

global HAS_CUT_CROSS_ENTROPY
global UNSLOTH_STUDIO_ENABLED
import importlib.util
if importlib.util.find_spec("unsloth_studio") is None:
    UNSLOTH_STUDIO_ENABLED = False
else:
    UNSLOTH_STUDIO_ENABLED = os.environ.get("UNSLOTH_STUDIO_DISABLED", "0") == "0"
pass
if UNSLOTH_STUDIO_ENABLED:
    from unsloth_studio.losses import (
        unsloth_efficient_ce_loss,
    )
pass

if DEVICE_TYPE == "cuda":
    major, minor = torch.cuda.get_device_capability()
    if (Version(torch.__version__) >= Version("2.4.0")) and \
        (not ((major <= 7) and (minor < 5))) and \
        (not (Version(triton_version) < Version("3.0.0"))):
        try:
            from cut_cross_entropy import linear_cross_entropy
            HAS_CUT_CROSS_ENTROPY = True
        except:
            HAS_CUT_CROSS_ENTROPY = False
    else:
        HAS_CUT_CROSS_ENTROPY = False
    pass
elif DEVICE_TYPE == "hip":
    try:
        from cut_cross_entropy import linear_cross_entropy
        HAS_CUT_CROSS_ENTROPY = True
    except:
        HAS_CUT_CROSS_ENTROPY = False
elif DEVICE_TYPE == "xpu":
    try:
        from cut_cross_entropy import linear_cross_entropy
        HAS_CUT_CROSS_ENTROPY = True
    except:
        HAS_CUT_CROSS_ENTROPY = False
else:
    HAS_CUT_CROSS_ENTROPY = False
pass

__all__ = [
    "patch_loss_functions",
    "post_patch_loss_function",
    "HAS_CUT_CROSS_ENTROPY",
    "fused_linear_cross_entropy",
    "fast_linear_cross_entropy",
    "_unsloth_get_batch_samples",
    "unsloth_fused_ce_loss",
]

from .fused_losses import unsloth_fused_ce_loss

def patch_loss_functions(_fast_cross_entropy_loss, torch_compile = True):
    # All Unsloth Zoo code licensed under LGPLv3
    try:
        import transformers.loss.loss_utils
    except:
        print("Unsloth: Cannot patch loss functions - update transformers for faster modules!")
        return None
    pass

    # Generic cross entropy loss
    def unsloth_fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
        if ignore_index == -100:
            loss = _fast_cross_entropy_loss(
                logits  = source,
                labels  = target,
                n_items = num_items_in_batch,
            )
        else:
            reduction = "sum" if num_items_in_batch is not None else "mean"
            loss = torch_nn_functional_cross_entropy(
                source,
                target,
                ignore_index = ignore_index,
                reduction    = reduction,
            )
            if reduction == "sum":
                # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
                if torch.is_tensor(num_items_in_batch):
                    num_items_in_batch = num_items_in_batch.to(loss.device)
                loss = loss / num_items_in_batch
        return loss
    pass
    
    # Causal LM loss
    def UnslothForCausalLMLoss(
        logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
    ):
        if labels is None: return None
        shift_logits = logits
        shift_labels = torch.empty_like(labels)
        shift_labels[..., :-1] = labels[..., 1:]
        shift_labels[..., -1] = ignore_index
        loss = unsloth_fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
        return loss
    pass

    if (Version(torch.__version__) < Version("2.4.0")):
        UnslothForCausalLMLoss = torch._disable_dynamo(UnslothForCausalLMLoss)
    
    elif torch_compile:
        UnslothForCausalLMLoss = torch.compile(
            UnslothForCausalLMLoss,
            dynamic = True,
            fullgraph = False,
            options = torch_compile_options,
        )
    pass

    # Now patch the losses!
    import transformers.modeling_utils
    LOSS_MAPPING = transformers.loss.loss_utils.LOSS_MAPPING
    LOSS_MAPPING["ForCausalLM"] = UnslothForCausalLMLoss

    # Remove @property and @lru_cache
    if hasattr(transformers.modeling_utils.PreTrainedModel.loss_function, "fget") and \
        hasattr(transformers.modeling_utils.PreTrainedModel.loss_function.fget, "__wrapped__"):
        transformers.modeling_utils.PreTrainedModel.loss_function = \
            transformers.modeling_utils.PreTrainedModel.loss_function.fget.__wrapped__
    pass
    print("Unsloth: Patched cross entropy losses.")
    os.environ["UNSLOTH_PATCHED"] = "1"
pass


def post_patch_loss_function(model):
    current_model = model
    while hasattr(current_model, "model"):
        try:
            # model.loss_function starts as a dict to a loss fx
            # We invoke it to save it
            current_model.loss_function = current_model.loss_function()
        except:
            # Failed means we already invoked it, and we need args to the loss fx
            pass
        pass
        current_model = current_model.model
    pass
    try: current_model.loss_function = current_model.loss_function()
    except: pass
    return model
pass


current_device = torch.xpu.device if DEVICE_TYPE == "xpu" else torch.cuda.device
def fused_linear_cross_entropy(
    hidden_states      : torch.Tensor,
    lm_weight          : torch.Tensor,
    labels             : torch.Tensor,
    num_items_in_batch : int = None,
    ignore_index       : int = -100,
    reduction          : str = "mean",
    logit_softcapping  : float = 0,
    accuracy_threshold : str = "auto",
):
    # All Unsloth Zoo code licensed under LGPLv3
    if num_items_in_batch is not None and torch.is_tensor(num_items_in_batch):
        num_items_in_batch = num_items_in_batch.to(hidden_states.device, non_blocking = True)

    reduction = "sum" if num_items_in_batch is not None else "mean"
    if logit_softcapping == 0: logit_softcapping = None

    with current_device(lm_weight.device):
        loss = linear_cross_entropy(
            hidden_states.to(lm_weight.dtype),
            lm_weight,
            targets      = labels,
            ignore_index = ignore_index,
            softcap      = logit_softcapping,
            reduction    = reduction,
            shift        = True,
            filter_eps   = accuracy_threshold,
        )
    if num_items_in_batch is not None: loss = loss / num_items_in_batch
    return loss
pass


def fast_linear_cross_entropy(
    hidden_states        : torch.Tensor,
    lm_head              : torch.nn.Linear,
    labels               : torch.Tensor,
    num_items_in_batch   : int = None,
    ignore_index         : int = -100,
    reduction            : str = "mean",
    logit_softcapping    : float = 0,
    logit_scale_multiply : float = 0,
    logit_scale_divide   : float = 0,
    attention_mask       : torch.Tensor = None,
):
    # All Unsloth Zoo code licensed under LGPLv3
    if num_items_in_batch is not None and torch.is_tensor(num_items_in_batch):
        num_items_in_batch = num_items_in_batch.to(hidden_states.device, non_blocking = True)

    reduction = "sum" if num_items_in_batch is not None else "mean"
    if logit_softcapping == 0: logit_softcapping = None
    if logit_scale_multiply != 0:
        logit_scale = logit_scale_multiply
    elif logit_scale_divide != 0:
        logit_scale = 1.0 / logit_scale_divide
    else:
        logit_scale = None

    loss = unsloth_efficient_ce_loss(
        hidden_states = hidden_states,
        lm_head = lm_head,
        labels = labels,
        shift = True,
        reduction = reduction,
        logit_scale = logit_scale,
        logit_softcapping = logit_softcapping,
        ignore_index = ignore_index,
        chunk_size = 512,
        attention_mask = attention_mask,
    )
    if num_items_in_batch is not None: loss = loss / num_items_in_batch
    return loss
pass

global ALLOWED_NUM_ITEMS_IN_BATCH
ALLOWED_NUM_ITEMS_IN_BATCH = dict()

global TRAINING_ITERATIONS
TRAINING_ITERATIONS = 0

# Check for DataParallel
#
# DataParallel uses scatter and gather
# cpu->0 scatter 0 --> 0 gather 0
# cpu->0 scatter 1 --> 1 gather 0
# cpu->0 scatter 2 --> 2 gather 0
#
# DistributedDataParallel is faster and launches multiple processes
# cpu->0 ------------> 0 gather 0
# cpu->1 ------------> 1 gather 0
# cpu->2 ------------> 2 gather 0
from transformers.training_args import ParallelMode

# Cannot use sadly
# import torch._dynamo.eval_frame as torch_dynamo_eval_frame
# torch_compiler_set_stance = torch.compiler.set_stance

mark_static  = torch._dynamo.mark_static
mark_dynamic = torch._dynamo.mark_dynamic

def _unsloth_get_batch_samples(self, epoch_iterator, num_batches, device = None, *args, **kwargs):
    # All Unsloth Zoo code licensed under LGPLv3
    batch_samples = []
    num_items_in_batch = None

    # Check if model allows **kwargs
    m = self.model
    if hasattr(m, "get_base_model"):
        # Removes PeftModelForCausalLM and gets internal model
        m = m.get_base_model()
    model_name = m.__class__.__name__
    global ALLOWED_NUM_ITEMS_IN_BATCH
    if model_name not in ALLOWED_NUM_ITEMS_IN_BATCH:

        has_kwargs = False
        is_vlm = False
        while True:
            # Stop when we encounter the name as ForConditionalGeneration or ForCausalLM
            if not hasattr(m, "forward"): break
            if not hasattr(m.forward, "__qualname__"): break
            forward = m.forward

            # Check double wrapped - for full finetuning
            if hasattr(forward, "__wrapped__"):
                __wrapped__ = forward.__wrapped__
                if hasattr(__wrapped__, "__wrapped__"):
                    __wrapped__ = __wrapped__.__wrapped__
                    if hasattr(__wrapped__, "__qualname__"):
                        forward = __wrapped__
            pass
            name = forward.__qualname__
            if "ForConditionalGeneration" in name or "VisionText2Text" in name:
                is_vlm = True
            if is_vlm or "CausalLM" in name or "_fast_forward" in name:
                signature = inspect.signature(forward).parameters.values()
                has_kwargs = tuple(signature)[-1].kind == inspect._VAR_KEYWORD
                break
            if not hasattr(m, "model"): break
            m = m.model
        pass
        ALLOWED_NUM_ITEMS_IN_BATCH[model_name] = (has_kwargs, is_vlm)
    else:
        has_kwargs, is_vlm = ALLOWED_NUM_ITEMS_IN_BATCH[model_name]
    pass

    # Iterate to find all batches
    for _ in range(num_batches):
        try:
            batch_samples += [next(epoch_iterator)]
        except StopIteration:
            break
    pass

    # Get num_items_in_batch
    if has_kwargs and len(batch_samples) > 0 and "labels" in batch_samples[0]:
        try:
            token_counts = []
            for x in batch_samples:
                labels = x["labels"]
                token_count = (labels[..., 1:] != -100)
                if "input_ids" in x:
                    input_ids = x["input_ids"]
                    mark_static (input_ids, 0)
                    mark_dynamic(input_ids, 1)
                if "attention_mask" in x:
                    attention_mask = x["attention_mask"]
                    mark_static (attention_mask, 0)
                    mark_dynamic(attention_mask, 1)
                    token_count &= (attention_mask[..., 1:] != 0)
                if "token_type_ids" in x:
                    token_type_ids = x["token_type_ids"]
                    mark_static (token_type_ids, 0)
                    mark_dynamic(token_type_ids, 1)
                count = token_count.sum()
                seq_lengths = x.get("packed_seq_lengths")
                if seq_lengths is not None:
                    # When packing N sequences, there are N-1 internal boundaries
                    # that shouldn't be counted as valid training positions
                    count -= torch.count_nonzero(seq_lengths > 0).item() - 1
                token_counts.append(count)
            pass
            num_items_in_batch = sum(token_counts)

            if self.args.average_tokens_across_devices:
                num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()
            if torch.is_tensor(num_items_in_batch):
                if device is not None:
                    num_items_in_batch = num_items_in_batch.to(device)
                if getattr(self.args, "n_gpu", 1) > 1 and self.args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
                    # Uses DataParallel scatter gather
                    # So we have to scatter num_items_in_batch to each GPU
                    num_items_in_batch = num_items_in_batch.unsqueeze(0).repeat(self.args.n_gpu)
        except Exception as exception:
            raise RuntimeError(exception)
    pass
    if UNSLOTH_ENABLE_LOGGING:
        logger.info(f"Unsloth: num_items_in_batch = {num_items_in_batch}")
    
    # [TODO] Unfortunately skip_guard_eval_unsafe = True fails
    # Increment counter and set compiler stance
    # if not hasattr(self.model, "vllm_engine"):
    #     # Only for non vLLM runs! Otherwise errors out
    #     global TRAINING_ITERATIONS
    #     if TRAINING_ITERATIONS == 16:
    #         # Skip guards after 16 warmup runs
    #         torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = True)
    #         if UNSLOTH_ENABLE_LOGGING:
    #             logger.info(f"Unsloth: Skipping torch.compile guards after 16 steps at TRAINING_ITERATIONS = {TRAINING_ITERATIONS}")
    #     elif torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe == False and TRAINING_ITERATIONS > 16:
    #         # Reset TRAINING_ITERATIONS
    #         torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
    #         TRAINING_ITERATIONS = 0
    #     TRAINING_ITERATIONS += 1
    return batch_samples, num_items_in_batch
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
