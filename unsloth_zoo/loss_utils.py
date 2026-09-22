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
from .utils import Version
import os
import math
import functools
from collections.abc import Mapping
from typing import Optional
torch_nn_functional_cross_entropy = torch.nn.functional.cross_entropy
from triton import __version__ as triton_version
from . import DEVICE_TYPE
from unsloth_zoo.temporary_patches.common import UNSLOTH_ENABLE_LOGGING, torch_compile_options, logger
import inspect
import re

global HAS_CUT_CROSS_ENTROPY
global UNSLOTH_STUDIO_ENABLED
import importlib.util
if importlib.util.find_spec("unsloth_studio") is None:
    UNSLOTH_STUDIO_ENABLED = False
else:
    UNSLOTH_STUDIO_ENABLED = os.environ.get("UNSLOTH_STUDIO_DISABLED", "0") == "0"
pass

# unsloth #2491: triton 3.3.0 and 3.3.1 cannot lower cut_cross_entropy's
# `_cce_lse_forward_kernel` for compute capability 7.5 (T4, RTX 2080 Ti). The TritonGPU to
# LLVM pass gives up on a `tt.fp_to_fp` and aborts the process:
#     error: Unsupported conversion from f16 to f16
#     LLVM ERROR: Unsupported rounding mode for conversion.
# (triton-lang/triton#6698, closed 2025-07-22.) The range is measured: the kernel was
# compiled ahead of time for cuda:75 on every release from 3.1.0 to 3.8.0 and only these
# two abort, for every block size, dot precision and accumulator dtype tried, while the
# sm_80 control compiles on all of them.
#
# Not expressible as a floor: torch pins triton exactly and the oldest supported torch,
# 2.6.0, requires `triton==3.2.0`.
_TRITON_CCE_BROKEN_ON_SM75 = ("3.3.0", "3.4.0")


def _triton_miscompiles_cce_on_sm75(major, minor, version = triton_version):
    if (major, minor) != (7, 5):
        return False
    low, high = _TRITON_CCE_BROKEN_ON_SM75
    try:
        installed = Version(version)
    except Exception:
        # An unparseable version is not evidence of a defect.
        return False
    return Version(low) <= installed < Version(high)
pass


def _triton_miscompiles_cce_on_any_visible_device():
    """Every visible CUDA device this triton would miscompile the CCE kernel for, as
    [(index, major, minor), ...].

    Asked of all of them because `torch.cuda.get_device_capability()` with no argument
    describes the CURRENT device, while a `device_map` can place `lm_head` anywhere. On a
    heterogeneous host that turns the import-time reading into a guess about the wrong GPU,
    and the consequence of guessing wrong is not a slow path: the kernel aborts the process
    with `LLVM ERROR: Unsupported rounding mode for conversion`, which nothing can catch.
    A host with no affected device is unchanged, since the list is then empty.
    """
    affected = []
    try:
        count = torch.cuda.device_count()
    except Exception:
        return affected
    for index in range(count):
        try:
            device_major, device_minor = torch.cuda.get_device_capability(index)
        except Exception:
            continue
        # `triton_version` passed rather than left to the default, which binds at
        # definition time and so cannot be substituted by a caller or a test.
        if _triton_miscompiles_cce_on_sm75(device_major, device_minor, triton_version):
            affected.append((index, device_major, device_minor))
    return affected
pass


if DEVICE_TYPE == "cuda" and not torch.cuda.is_available():
    # UNSLOTH_ALLOW_CPU=1 keeps DEVICE_TYPE "cuda" on driverless hosts, so ask
    # whether a device is present before asking what it can do. Cut cross
    # entropy is a Triton GPU kernel and cannot run here regardless, so False is
    # both the conservative answer and the only correct one.
    HAS_CUT_CROSS_ENTROPY = False
elif DEVICE_TYPE == "cuda":
    major, minor = torch.cuda.get_device_capability()
    # Every VISIBLE device, not just the current one: a device_map can place lm_head on
    # another GPU, so the kernel can run on an sm_75 while device 0 is an sm_80. The
    # failure aborts the process rather than raising, so be conservative.
    _miscompiling_devices = _triton_miscompiles_cce_on_any_visible_device()
    if (Version(torch.__version__) >= Version("2.4.0")) and \
        (not ((major <= 7) and (minor < 5))) and \
        (not (Version(triton_version) < Version("3.0.0"))) and \
        (not _miscompiling_devices):
        try:
            from cut_cross_entropy import linear_cross_entropy
            HAS_CUT_CROSS_ENTROPY = True
        except:
            HAS_CUT_CROSS_ENTROPY = False
    else:
        HAS_CUT_CROSS_ENTROPY = False
    pass
    if _miscompiling_devices:
        _affected = ", ".join(
            f"cuda:{index} (sm_{device_major}{device_minor})"
            for index, device_major, device_minor in _miscompiling_devices
        )
        logger.warning(
            f"Unsloth: triton=={triton_version} miscompiles the cut cross entropy "
            f"kernel for {_affected}, so it is disabled and "
            f"the standard loss is used instead, at a higher memory cost. torch 2.6.0, "
            f"which carries triton 3.2.0, restores it. torch 2.8.0 and later carry a "
            f"triton that compiles the kernel, but do NOT restore it here: unsloth_zoo "
            f"sets UNSLOTH_ENABLE_CCE=0 for torch 2.8 and above over a separate shared "
            f"memory failure, and both compiled branches require that flag as well as "
            f"HAS_CUT_CROSS_ENTROPY. Leaving it enabled aborts the process with "
            f"'LLVM ERROR: Unsupported rounding mode for conversion'."
        )
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

from unsloth_zoo.fused_losses import unsloth_fused_ce_loss

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
        # The stock loss also takes (tokens, vocab) logits with 1-D labels (Ling's MTP head
        # calls it that way); treat them as one row so the shift matches.
        if logits.dim() == 2 and labels.dim() == 1:
            logits = logits.unsqueeze(0)
            labels = labels.unsqueeze(0)
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
    # Patch every key still aliased to the stock ForCausalLMLoss. PreTrainedModel
    # resolves loss_type by regex on the class name, so e.g.
    # Qwen3_5ForConditionalGeneration / CsmForConditionalGeneration land on keys
    # pointing at the stock loss; without this sweep they keep the un-patched
    # loss and OOM via logits.float() at large vocab sizes.
    for _key, _fn in list(LOSS_MAPPING.items()):
        if getattr(_fn, "__name__", "") == "ForCausalLMLoss":
            LOSS_MAPPING[_key] = UnslothForCausalLMLoss

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


def fast_linear_cross_entropy(*args, **kwargs):
    raise RuntimeError(
        "Unsloth: `fast_linear_cross_entropy` has been deprecated. "
        "Please update Unsloth and Unsloth Zoo via:\n"
        "pip install --upgrade --no-cache-dir --no-deps unsloth_zoo unsloth"
    )
pass


global ALLOWED_NUM_ITEMS_IN_BATCH
ALLOWED_NUM_ITEMS_IN_BATCH = dict()

# Heads whose labels are not shifted next-token targets, so the counter below cannot
# describe them. Matched on the top level class name, since the walk below descends
# into the backbone and would judge the head by the decoder it sits on.
NON_CAUSAL_HEADS = (
    "ForSequenceClassification",
    "ForTokenClassification",
    "ForQuestionAnswering",
    "ForMultipleChoice",
)

# The loss_types that shift labels, snapshotted AT IMPORT: patch_loss_functions() above
# swaps every causal entry for UnslothForCausalLMLoss, so testing against the live
# mapping would read False for every causal model once unsloth has loaded.
try:
    from transformers.loss.loss_utils import LOSS_MAPPING as _LOSS_MAPPING, ForCausalLMLoss
    CAUSAL_LOSS_TYPES = frozenset(
        key for key, fn in _LOSS_MAPPING.items() if fn is ForCausalLMLoss
    )
except Exception:
    CAUSAL_LOSS_TYPES = frozenset()


def _loss_shifts_labels(trainer, model, is_encoder_decoder):
    # All Unsloth Zoo code licensed under LGPLv3
    """Does this model's loss shift labels, so labels[..., 1:] is the right count?

    A positive signal, not a list of excluded suffixes: a name cannot tell
    BertForMaskedLM (2D token aligned labels, column 0 supervised) from a causal LM.
    5.x already computed this on the Trainer; older ones get the same answer from the
    loss_type every PreTrainedModel derives from its class name. A hand written
    nn.Module has neither, and stays out: without a signal we decline rather than
    guess, which is today's behaviour for it anyway.
    """
    shifts = getattr(trainer, "_loss_shifts_labels", None)
    if isinstance(shifts, bool): return shifts
    return getattr(model, "loss_type", None) in CAUSAL_LOSS_TYPES \
        and not is_encoder_decoder
pass

global TRAINING_ITERATIONS
TRAINING_ITERATIONS = 0

# ParallelMode distinguishes DataParallel (scatter/gather from one process)
# from the faster DistributedDataParallel (one process per device).
from transformers.training_args import ParallelMode

# Cannot use sadly
# import torch._dynamo.eval_frame as torch_dynamo_eval_frame
# torch_compiler_set_stance = torch.compiler.set_stance

mark_static  = torch._dynamo.mark_static
mark_dynamic = torch._dynamo.mark_dynamic


def _normalize_packed_seq_lengths(seq_lengths):
    # All Unsloth Zoo code licensed under LGPLv3
    """Coerce collator supplied packed sequence lengths to a 1D int64 CPU tensor.

    Returns None when the metadata is missing, unusable, describes a single
    document (one document has no internal boundary to drop), or when the running
    execution mode cannot evaluate the filter below. Kept on CPU:
    these are tens of elements, it avoids MPS / XPU integer op gaps, and it keeps
    the counting path free of a device sync.

    Deliberately duplicated from unsloth/utils/packing.py rather than imported.
    The dependency arrow runs unsloth -> unsloth_zoo and must never run back.

    Anything unconvertible is treated as absent rather than raised: the caller
    counts inside a try that re-raises as RuntimeError, so a plain list of ints
    used to end the run outright.
    """
    if seq_lengths is None: return None
    try:
        if isinstance(seq_lengths, torch.Tensor):
            lengths = seq_lengths.detach().to(device = "cpu", dtype = torch.long)
        else:
            lengths = torch.as_tensor(seq_lengths, dtype = torch.long)
        if lengths.ndim != 1:
            lengths = lengths.reshape(-1)
        # Filtering is mandatory (a 0 length duplicates a start, a negative one
        # corrupts every later cumsum) and must stay inside the try: the mask
        # lowers to aten.nonzero, which raises DynamicOutputShapeException under
        # FakeTensorMode and raises under vmap. Escaping into the caller's
        # `except Exception: raise RuntimeError(...)` kills the run.
        lengths = lengths[lengths > 0]
        if lengths.numel() <= 1: return None
    except Exception:
        return None
    return lengths
pass


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

    # Read off the top level model, before the walk below reassigns m: unsloth patches
    # LlamaModel.forward class-wide, so LlamaForSequenceClassification descends into
    # .model, matches "_fast_forward" and never consults the head it trains. Not
    # cached, since is_encoder_decoder belongs to the instance, not the class name.
    top_model = m
    is_encoder_decoder = bool(getattr(getattr(m, "config", None), "is_encoder_decoder", False))
    is_non_causal_head = any(head in model_name for head in NON_CAUSAL_HEADS)

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
            # Fall back to the class name: patched forwards (temporary_patches)
            # may have qualnames lacking identifiers like "CausalLM".
            class_name = type(m).__name__
            if "ForConditionalGeneration" in name or "ForConditionalGeneration" in class_name \
                or "VisionText2Text" in name:
                is_vlm = True
            if is_vlm or "CausalLM" in name or "CausalLM" in class_name or "_fast_forward" in name:
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

    # Get num_items_in_batch. Two questions, as in stock transformers:
    #   has a consumer: `model_accepts_loss_kwargs or compute_loss_func is not None`,
    #     same on 4.57.6 and 5.17.0. has_kwargs is our first half, #1217 the second.
    #   is countable: everything below counts SHIFTED CAUSAL targets (stock 5.17.0's
    #     self._loss_shifts_labels). This lives on the base Trainer class, so
    #     classification and seq2seq subclasses inherit it and their labels mean
    #     something else: they raise here, or come back quietly short.
    # The guards differ in width on purpose. is_non_causal_head gates both routes; the
    # new route also demands a POSITIVE shifted-label signal, so Whisper and Florence2
    # keep today's count rather than being rescaled here. A sample has to be a Mapping
    # before "labels" can be read off it, and then have a .ndim before any tensor op
    # touches it: TRL's GRPO collator is the identity, so a sample is a LIST of dicts
    # and `.get` on it raises. Both used to kill a run stock trains through, which is
    # why the pre-#1217 spelling asked `"labels" in batch_samples[0]`.
    labels_are_countable = getattr(
        batch_samples[0].get("labels") if len(batch_samples) > 0
        and isinstance(batch_samples[0], Mapping) else None, "ndim", None,
    ) is not None
    if (not is_non_causal_head) and labels_are_countable \
            and (has_kwargs or (getattr(self, "compute_loss_func", None) is not None
                                and _loss_shifts_labels(self, top_model, is_encoder_decoder))):
        try:
            token_counts = []
            # Shape only, so no device sync and still traceable. Acted on after the
            # collectives below, never by breaking out: skipping accelerator.gather on
            # one rank alone would hang the others.
            degenerate = False
            # One column leaves labels[..., 1:] empty. Only fatal when EVERY microbatch
            # is short, since then the total is 0 and a sum/count loss divides by it. A
            # short member of a mixed group just contributes 0, and voiding the group
            # for it would lose GA invariance.
            all_short = True
            for x in batch_samples:
                labels = x["labels"]
                if labels.shape[-1] >= 2: all_short = False
                token_count = (labels[..., 1:] != -100)
                if "input_ids" in x:
                    input_ids = x["input_ids"]
                    mark_static (input_ids, 0)
                    mark_dynamic(input_ids, 1)
                    # One label per row against 2D input_ids is a classification
                    # batch, whatever the class is called.
                    if labels.ndim != input_ids.ndim: degenerate = True
                if "attention_mask" in x:
                    attention_mask = x["attention_mask"]
                    mark_static (attention_mask, 0)
                    mark_dynamic(attention_mask, 1)
                    # Only AND a mask describing these same targets: a seq2seq mask is
                    # the encoder's, a different length from the decoder labels, which
                    # used to raise. Causal shapes always match, so nothing changes.
                    if attention_mask.shape != labels.shape:
                        degenerate = True
                    else:
                        token_count &= (attention_mask[..., 1:] != 0)
                if "token_type_ids" in x:
                    token_type_ids = x["token_type_ids"]
                    mark_static (token_type_ids, 0)
                    mark_dynamic(token_type_ids, 1)
                seq_lengths = _normalize_packed_seq_lengths(x.get("packed_seq_lengths"))
                if seq_lengths is not None and token_count.ndim in (1, 2) and token_count.shape[-1] != 0:
                    # Packing N documents leaves N-1 internal boundaries that are
                    # not valid training positions. Zero those exact slots rather
                    # than subtract N-1: many collators already mask them (TRL
                    # >= 0.23.1 labels[position_ids == 0] = -100, transformers'
                    # DataCollatorWithFlattening, completion_only_loss /
                    # assistant_masks), so subtracting double counts them and
                    # inflates loss and grads. Zeroing is idempotent and cannot
                    # go below zero; subtracting had no lower bound and drove the
                    # count to zero or negative on small batches.
                    #
                    # labels[..., 1:] already dropped column 0 of every row, so a
                    # document starting at flat index s sits at column s - 1, and
                    # one starting at a row boundary is already gone. cumsum[:-1]
                    # drops the trailing boundary, making a single document a
                    # provable no-op and keeping truncated metadata harmless.
                    #
                    # The data-dependent reads below (rows[keep], rows.numel())
                    # are unprotected, and safe only because
                    # _normalize_packed_seq_lengths already returned None under
                    # any mode that cannot evaluate them. Keep that ordering.
                    n_shift = token_count.shape[-1]
                    n_rows  = token_count.numel() // n_shift
                    starts  = torch.cumsum(seq_lengths, dim = 0)[:-1]
                    rows    = torch.div(starts, n_shift + 1, rounding_mode = "floor")
                    cols    = starts - rows * (n_shift + 1)
                    keep    = (cols > 0) & (rows < n_rows)
                    rows, cols = rows[keep], cols[keep]
                    if rows.numel() != 0:
                        if rows.device != token_count.device:
                            rows = rows.to(token_count.device)
                            cols = cols.to(token_count.device)
                        # Reassign: reshape can copy on a non contiguous input.
                        token_count = token_count.reshape(n_rows, n_shift)
                        token_count[rows, cols - 1] = False
                    pass
                pass
                count = token_count.sum()
                token_counts.append(count)
            pass
            num_items_in_batch = sum(token_counts)

            if self.args.average_tokens_across_devices:
                if getattr(self.args, "world_size", 1) > 1:
                    # One collective, not two: the two rank-local fallback flags ride
                    # along with the count, so this path keeps exactly the single
                    # gather it has always had. ANY degenerate rank means the layout
                    # is uncountable everywhere; only an ALL-short world has a truly
                    # zero total, since a short rank beside healthy ones has simply
                    # contributed 0 to a total that is still its right divisor.
                    count = torch.as_tensor(num_items_in_batch).reshape(()).to(torch.int64)
                    flags = torch.tensor([int(degenerate), int(all_short)],
                                         dtype = torch.int64, device = count.device)
                    packed = torch.cat([count.reshape(1), flags])
                    packed = self.accelerator.gather(packed).reshape(-1, 3)
                    num_items_in_batch = packed[:, 0].sum()
                    degenerate = bool(packed[:, 1].any())
                    all_short  = bool(packed[:, 2].all())
                else:
                    num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()
            if torch.is_tensor(num_items_in_batch):
                if device is not None:
                    num_items_in_batch = num_items_in_batch.to(device)
                if getattr(self.args, "n_gpu", 1) > 1 and self.args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
                    # Uses DataParallel scatter gather
                    # So we have to scatter num_items_in_batch to each GPU
                    num_items_in_batch = num_items_in_batch.unsqueeze(0).repeat(self.args.n_gpu)
            # Discard a count these labels could not support. Last, so every collective
            # above ran on every rank, and reduced above so no rank drops a divisor its
            # peers keep.
            if degenerate or all_short: num_items_in_batch = None
        except Exception as exception:
            raise RuntimeError(exception)
    pass

    # num_items_in_batch is set from the forward signature, but training_step
    # divides by grad-accum off self.model_accepts_loss_kwargs. Counting while
    # that flag is False normalises twice (TRL chunked_nll and our fused CE each
    # divide by it), scaling loss and grads by 1/GA. Like stock
    # Trainer._get_num_items_in_batch, only count when a consumer exists; these
    # losses fall back to a mean when it is None.
    if (num_items_in_batch is not None
            and not getattr(self, "model_accepts_loss_kwargs", True)
            and getattr(self, "compute_loss_func", None) is None):
        num_items_in_batch = None

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
