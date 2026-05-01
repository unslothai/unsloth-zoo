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

__all__ = [
    "unsloth_fused_ce_loss",
    "apply_autograd_function",
    "compute_fused_ce_loss",
]

import torch
from typing import Optional, Tuple, Callable, Dict
import inspect
import functools
import math
import os
from ..temporary_patches.common import UNSLOTH_ENABLE_LOGGING, torch_compile_options, logger
from ..device_type import DEVICE_TYPE
        

TARGET_GB = os.environ.get("UNSLOTH_CE_LOSS_TARGET_GB", None)
N_CHUNKS = os.environ.get("UNSLOTH_CE_LOSS_N_CHUNKS", None)

# Register grad_and_value_impl in trace_rules as defense-in-depth.
# grad_impl is registered but grad_and_value_impl is not, which can cause
# GB0149 "Unsupported functorch tracing attempt" in some configurations.
try:
    from torch._dynamo.trace_rules import manual_torch_name_rule_map as _trace_map
    from torch._dynamo.variables.higher_order_ops import FunctorchHigherOrderVariable as _FHOV
    _key = "torch._functorch.eager_transforms.grad_and_value_impl"
    if _key not in _trace_map:
        _trace_map[_key] = _FHOV
        torch._dynamo.trace_rules.get_torch_obj_rule_map.cache_clear()
except Exception:
    pass

# Module-level flag: None = untested, True = works, False = skip compile.
_FUSED_CE_COMPILE_SUPPORTED = None if \
    os.environ.get("UNSLOTH_FUSED_CE_COMPILE_DISABLE", "0") != "1" else False

@functools.cache
def _get_mapping(autograd):
    parameters = inspect.signature(getattr(autograd, "forward")).parameters
    parameters = dict(parameters)
    parameters.pop("ctx", None)
    return tuple(parameters.keys()), tuple([x.default for x in parameters.values()])
pass

def apply_autograd_function(autograd, mapping):
    parameters, defaults = _get_mapping(autograd)
    return getattr(autograd, "apply")(*(
        mapping.get(old_key, default) \
        for old_key, default in zip(parameters, defaults)
    ))
pass

def compute_fused_ce_loss(
    hidden_states  : torch.Tensor,
    lm_head_weight : torch.Tensor,
    lm_head_bias   : Optional[torch.Tensor],
    labels         : torch.Tensor,
    n_items        : Optional[torch.Tensor] = None,
    scaling        : Optional[float] = None,
    shift_labels   : bool = True,
    **kwargs,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor,],]:
    """
    Computes cross_entropy_loss(X @ W + b, labels)
    * shift_labels does hidden_states[..., :-1] and labels[..., 1:]
    * If n_items is not given, does mean(ce_loss), otherwise sum(ce_loss)/n_items
    * Allows scaling factor from mixed precision fp16, fp8
    * Upcasts to float32 and allows kwargs to have:
    1) logit_scale_multiply (X = X * logit_scale_multiply)
    2) logit_scale_divide   (X = X / logit_scale_divide)
    3) logit_softcapping    (X = tanh(X / logit_softcapping) * logit_softcapping)
    """
    device = lm_head_weight.device
    if shift_labels:
        # Get shifted labels first
        _labels = torch.empty_like(labels, device = device)
        _labels[..., :-1] = labels[..., 1:]
        _labels[..., -1] = -100
        labels = _labels
    pass

    logits = torch.nn.functional.linear(
        hidden_states.to(dtype = lm_head_weight.dtype, device = device),
        lm_head_weight,
        lm_head_bias,
    )
    vocab_size = lm_head_weight.shape[0]

    # Apply softcapping and other functions
    logit_scale_multiply = kwargs.get("logit_scale_multiply", None)
    logit_scale_divide = kwargs.get("logit_scale_divide", None)
    logit_softcapping = kwargs.get("logit_softcapping", None)
    if logit_scale_multiply != 0 and logit_scale_multiply is not None:
        logits = logits * logit_scale_multiply
    if logit_scale_divide != 0 and logit_scale_divide is not None:
        logits = logits / logit_scale_divide
    if logit_softcapping != 0 and logit_softcapping is not None:
        logits = logits / logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * logit_softcapping

    # Calculate cross entropy loss
    reduction = "sum" if n_items is not None else "mean"
    loss = torch.nn.functional.cross_entropy(
        input  = logits.view(-1, vocab_size).float().contiguous(),
        target = labels.view(-1).to(device).contiguous(),
        reduction = reduction,
    )
    loss = loss / n_items if n_items is not None else loss
    # Scale loss if needed for mixed precision training
    scaled_loss = loss * scaling if scaling is not None else loss
    # Must add .loss.detach otherwise autograd uses 2x VRAM
    return scaled_loss, (loss.detach(),)
pass


@functools.cache
def _get_chunk_multiplier(vocab_size, target_gb = None):
    """ Gets chunk size that fits the target max memory usage (1GB) """
    if target_gb is None:
        # Find current VRAM left in the GPU, and use 50% or less of it
        free, total = torch.xpu.mem_get_info(0) if DEVICE_TYPE == "xpu" else torch.cuda.mem_get_info(0)
        free_gb = free / 1024 / 1024 / 1024
        free_gb = free_gb * 0.5
        target_gb = free_gb
    pass

    # Prevent ZeroDivisionError when GPU memory is exhausted
    if target_gb <= 1e-9: # Use a small epsilon for float comparison
        raise RuntimeError("Unsloth: No or negligible GPU memory available for fused cross entropy.")

    multiplier = (vocab_size * 4 / 1024 / 1024 / 1024) / (target_gb)
    multiplier = multiplier / 4 # Output only multiples of 4
    return multiplier
pass

def get_chunk_size(bsz, qlen, vocab_size, target_gb = None):
    """ Gets chunk size that fits the target max memory usage (1GB) """
    multiplier = _get_chunk_multiplier(vocab_size, target_gb)
    n_splits = (bsz*qlen) * multiplier
    # n_splits = max(round(n_splits / 4) * 4, 1) # Output only multiples of 4
    n_splits = max(round(n_splits) * 4, 1)
    return n_splits
pass

class UnslothFusedLoss(torch.autograd.Function):
    # One-time flag so the "scaling=0" info message is logged at most once per
    # process, even if the condition triggers on every backward call.
    _scaling_zero_logged = False

    @staticmethod
    def forward(
        ctx,
        loss_function  : Callable,
        hidden_states  : torch.Tensor,
        lm_head_weight : torch.Tensor,
        lm_head_bias   : Optional[torch.Tensor],
        labels         : torch.Tensor,
        mask           : Optional[torch.Tensor] = None,
        n_items        : Optional[torch.Tensor] = None,
        scaling        : Optional[float] = None,
        shift_labels   : Optional[bool] = True,
        target_gb      : Optional[int] = None,
        torch_compile  : Optional[bool] = True,
        overwrite      : Optional[bool] = False,
        extra_kwargs   : Optional[Dict] = None,
    ):
        """
        Computes chunked fused loss_function(chunk(X) @ W + b, chunk(labels))
        * If n_items is not given, does mean(loss), otherwise sum(loss)/n_items
        * shift_labels does hidden_states[..., :-1] and labels[..., 1:]
        * Allows scaling factor from mixed precision fp16, fp8
        * target_gb specifies the max GB memory the fused loss can use - default detects VRAM left
        * overwrite allows hidden_states to be overwritten with gradients
        * Place extra args in extra_kwargs which will be passed to (loss_function)
        """
        device = lm_head_weight.device
        if extra_kwargs is None: extra_kwargs = {}

        # Get shifted labels first
        if shift_labels:
            _labels = torch.empty_like(labels, device = device)
            _labels[..., :-1] = labels[..., 1:]
            # Also check mask
            if mask is not None:
                mask = mask.to(device = device)
                _labels[..., :-1][mask[..., 1:] == 0] = -100
            pass
            _labels[..., -1] = -100
            _labels = _labels.view(-1)
            labels = _labels
        pass

        # N items divisor
        divisor = n_items if n_items is not None else (labels != -100).sum()
        # Counteract DataParallel having multiple items since it does scatter & gather
        if divisor.numel() != 1: divisor = divisor.ravel()[0]
        divisor = divisor.to(dtype = torch.float32, device = device)
        # Check what needs gradients
        lm_head_requires_grad = lm_head_weight is not None and lm_head_weight.requires_grad
        lm_head_bias_requires_grad = lm_head_bias is not None and lm_head_bias.requires_grad
        vocab_size = lm_head_weight.shape[0]

        # Create backwards output
        grad_inputs = torch.empty_like(hidden_states, device = device) if not overwrite else hidden_states
        grad_lm_head = torch.zeros_like(lm_head_weight, device = device) if lm_head_requires_grad else None
        grad_lm_head_bias = torch.zeros_like(lm_head_bias, device = device) if lm_head_bias_requires_grad else None

        bsz, qlen, hd = hidden_states.shape
        accumulated_loss = torch.zeros(1, device = device)[0]
        # Chunk hidden_states and labels
        if "n_chunks" in extra_kwargs:
            n_chunks = extra_kwargs.pop("n_chunks")
        else:
            n_chunks = get_chunk_size(bsz, qlen, vocab_size, target_gb = target_gb)
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(f"Fused CE Loss [bsz={bsz}][qlen={qlen}][vocab_size={vocab_size}][n_chunks={n_chunks}]")
        __shift_labels = torch.chunk(labels,                     n_chunks, dim = 0)
        __shift_states = torch.chunk(hidden_states.reshape(-1, hd), n_chunks, dim = 0)
        __grad_inputs  = torch.chunk(grad_inputs.view(-1, hd),   n_chunks, dim = 0)

        def accumulate_chunk(
            n_chunks,
            grad_inputs_j,
            grad_lm_head,
            grad_lm_head_bias,
            hidden_states_j,
            lm_head_weight,
            lm_head_bias,
            labels_j,
            divisor = None,
            scaling = None,
            shift_labels = False,
            **kwargs,
        ):
            if lm_head_requires_grad and lm_head_bias_requires_grad:
                (chunk_grad_input, chunk_grad_lm_head, chunk_grad_lm_head_bias,), \
                (chunk_loss, (unscaled_loss,)) = \
                torch.func.grad_and_value(
                    loss_function,
                    argnums = (0, 1, 2,),
                    has_aux = True,
                )(
                    hidden_states_j,
                    lm_head_weight,
                    lm_head_bias,
                    labels_j,
                    divisor,
                    scaling,
                    not shift_labels, # Already label shifted
                    **kwargs,
                )
                grad_lm_head.add_(chunk_grad_lm_head)
                grad_lm_head_bias.add_(chunk_grad_lm_head_bias)
            elif lm_head_requires_grad:
                (chunk_grad_input, chunk_grad_lm_head,), \
                (chunk_loss, (unscaled_loss,)) = torch.func.grad_and_value(
                    loss_function,
                    argnums = (0, 1,),
                    has_aux = True,
                )(
                    hidden_states_j,
                    lm_head_weight,
                    lm_head_bias,
                    labels_j,
                    divisor,
                    scaling,
                    not shift_labels, # Already label shifted
                    **kwargs,
                )
                grad_lm_head.add_(chunk_grad_lm_head)
            elif lm_head_bias_requires_grad:
                (chunk_grad_input, chunk_grad_lm_head_bias,), \
                (chunk_loss, (unscaled_loss,)) = torch.func.grad_and_value(
                    loss_function,
                    argnums = (0, 2,),
                    has_aux = True,
                )(
                    hidden_states_j,
                    lm_head_weight,
                    lm_head_bias,
                    labels_j,
                    divisor,
                    scaling,
                    not shift_labels, # Already label shifted
                    **kwargs,
                )
                grad_lm_head_bias.add_(chunk_grad_lm_head_bias)
            else:
                (chunk_grad_input,), \
                (chunk_loss, (unscaled_loss,)) = torch.func.grad_and_value(
                    loss_function,
                    argnums = (0,),
                    has_aux = True,
                )(
                    hidden_states_j,
                    lm_head_weight,
                    lm_head_bias,
                    labels_j,
                    divisor,
                    scaling,
                    not shift_labels, # Already label shifted
                    **kwargs,
                )
            pass
            accumulated_loss.add_(unscaled_loss)
            grad_inputs_j[:] = chunk_grad_input
        pass
        global _FUSED_CE_COMPILE_SUPPORTED
        uncompiled_accumulate_chunk = accumulate_chunk

        chunks = []
        for grad_inputs_j, hidden_states_j, labels_j in zip(__grad_inputs, __shift_states, __shift_labels):
            if bool((labels_j != -100).any().item()):
                chunks.append((grad_inputs_j, hidden_states_j, labels_j,))
            else:
                grad_inputs_j.zero_()
        pass

        if torch_compile and _FUSED_CE_COMPILE_SUPPORTED is not False:
            try:
                accumulate_chunk = torch.compile(
                    accumulate_chunk,
                    dynamic = True,
                    fullgraph = True,
                    options = torch_compile_options,
                )
            except Exception:
                _FUSED_CE_COMPILE_SUPPORTED = False
                accumulate_chunk = uncompiled_accumulate_chunk

        # Probe path: first-ever forward pass, test if compiled version works
        if _FUSED_CE_COMPILE_SUPPORTED is None and torch_compile and \
            accumulate_chunk is not uncompiled_accumulate_chunk:

            _iter = iter(chunks)
            try:
                grad_inputs_j, hidden_states_j, labels_j = next(_iter)
            except StopIteration:
                _FUSED_CE_COMPILE_SUPPORTED = True
            else:
                try:
                    accumulate_chunk(
                        n_chunks = n_chunks,
                        grad_inputs_j = grad_inputs_j,
                        grad_lm_head = grad_lm_head,
                        grad_lm_head_bias = grad_lm_head_bias,
                        hidden_states_j = hidden_states_j,
                        lm_head_weight = lm_head_weight,
                        lm_head_bias = lm_head_bias,
                        labels_j = labels_j,
                        divisor = divisor,
                        scaling = scaling,
                        shift_labels = shift_labels,
                        **extra_kwargs,
                    )
                    _FUSED_CE_COMPILE_SUPPORTED = True
                except Exception:
                    _FUSED_CE_COMPILE_SUPPORTED = False
                    torch._dynamo.reset()
                    accumulated_loss.zero_()
                    if not overwrite:
                        grad_inputs.zero_()
                    if grad_lm_head is not None: grad_lm_head.zero_()
                    if grad_lm_head_bias is not None: grad_lm_head_bias.zero_()
                    accumulate_chunk = uncompiled_accumulate_chunk
                    accumulate_chunk(
                        n_chunks = n_chunks,
                        grad_inputs_j = grad_inputs_j,
                        grad_lm_head = grad_lm_head,
                        grad_lm_head_bias = grad_lm_head_bias,
                        hidden_states_j = hidden_states_j,
                        lm_head_weight = lm_head_weight,
                        lm_head_bias = lm_head_bias,
                        labels_j = labels_j,
                        divisor = divisor,
                        scaling = scaling,
                        shift_labels = shift_labels,
                        **extra_kwargs,
                    )
                # Process remaining chunks via fast path
                for (grad_inputs_j, hidden_states_j, labels_j,) in _iter:
                    accumulate_chunk(
                        n_chunks = n_chunks,
                        grad_inputs_j = grad_inputs_j,
                        grad_lm_head = grad_lm_head,
                        grad_lm_head_bias = grad_lm_head_bias,
                        hidden_states_j = hidden_states_j,
                        lm_head_weight = lm_head_weight,
                        lm_head_bias = lm_head_bias,
                        labels_j = labels_j,
                        divisor = divisor,
                        scaling = scaling,
                        shift_labels = shift_labels,
                        **extra_kwargs,
                    )
        else:
            # Fast path: compile status already known, original main branch loop
            for (grad_inputs_j, hidden_states_j, labels_j,) in chunks:
                accumulate_chunk(
                    n_chunks = n_chunks,
                    grad_inputs_j = grad_inputs_j,
                    grad_lm_head = grad_lm_head,
                    grad_lm_head_bias = grad_lm_head_bias,
                    hidden_states_j = hidden_states_j,
                    lm_head_weight = lm_head_weight,
                    lm_head_bias = lm_head_bias,
                    labels_j = labels_j,
                    divisor = divisor,
                    scaling = scaling,
                    shift_labels = shift_labels,
                    **extra_kwargs,
                )
        pass
        ctx.save_for_backward(grad_inputs, grad_lm_head, grad_lm_head_bias)
        ctx.scaling = scaling
        return accumulated_loss
    pass

    @staticmethod
    def backward(ctx, grad_output,):
        # DDP can scale grad_output by world size; normalize to expected scaling.
        scaling = ctx.scaling if ctx.scaling is not None else 1.0
        (grad_inputs, grad_lm_head, grad_lm_head_bias, ) = ctx.saved_tensors

        # Collapse tensor scaling to a Python float at the boundary. All current
        # callers pass a Python float (GradScaler.get_scale() returns float); a
        # future tensor caller pays a single .item() sync here and then takes
        # the scalar path. This keeps one code path, one semantics.
        if torch.is_tensor(scaling):
            scaling = float(scaling.detach().item())

        # scaling == 0 lost the saved gradient: forward's grad_and_value
        # differentiated scaled_loss = loss * scaling, so saved = scaling *
        # d(loss)/d(hidden) = 0. The Function returns the unscaled loss though,
        # so the correct answer is grad_output * d(loss)/d(hidden) - which we
        # cannot recover from saved=0. Only safe when grad_output is also 0
        # (chain rule: 0 * anything = 0); otherwise raise.
        if scaling == 0.0:
            if torch.is_tensor(grad_output):
                go_is_zero = bool(torch.all(grad_output == 0).item())
            else:
                go_is_zero = float(grad_output) == 0.0
            if not go_is_zero:
                raise RuntimeError(
                    "Fused CE loss: scaling=0 with non-zero grad_output. The "
                    "saved gradient was zeroed by scaling in the forward pass "
                    "and the unscaled gradient cannot be recovered. Likely a "
                    "misconfigured GradScaler."
                )
            if UNSLOTH_ENABLE_LOGGING and not UnslothFusedLoss._scaling_zero_logged:
                UnslothFusedLoss._scaling_zero_logged = True
                logger.info(
                    "Fused CE loss: scaling=0 with grad_output=0; returning zero "
                    "gradients. This message is logged once per process."
                )
            return (
                None, grad_inputs, grad_lm_head, grad_lm_head_bias,
                None, None, None, None, None, None, None, None, None,
            )

        if torch.is_tensor(grad_output):
            grad_scale = grad_output.detach().float().mean()
        else:
            grad_scale = torch.tensor(float(grad_output), device=grad_inputs.device, dtype=grad_inputs.dtype)

        scale_factor = grad_scale / scaling

        if UNSLOTH_ENABLE_LOGGING:
            if torch.is_tensor(grad_output):
                grad_scale_val = float(grad_scale.detach().cpu().item())
            else:
                grad_scale_val = float(grad_output)
            scale_factor_val = float(scale_factor.detach().cpu().item())
            if scale_factor_val == 1.0:
                torch._assert(
                    torch.all(grad_output == scaling),
                    f"Fused losses expect grad_output to be all {scaling}, but got {grad_output.ravel()[:10]}",
                )
            else:
                world_size = None
                try:
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized():
                        world_size = dist.get_world_size()
                except Exception:
                    world_size = None
                if world_size is not None:
                    logger.info(
                        f"Fused losses grad_output scaled by {scale_factor_val} (got {grad_scale_val}, expected {scaling} or {scaling * world_size})"
                    )
                else:
                    logger.info(
                        f"Fused losses grad_output scaled by {scale_factor_val} (got {grad_scale_val}, expected {scaling})"
                    )

        # Out-of-place mul so that ctx.saved_tensors' version counter does not
        # bump; this keeps retain_graph / double-backward-capable flows working.
        # Measured peak-memory delta vs. in-place mul is <3 MB across 14
        # configurations (LoRA, full-FT, MoE, vision, bsz up to 16, seq up to
        # 8192) because the temporary is freed before peak-setting allocations.
        grad_inputs = grad_inputs * scale_factor
        if grad_lm_head is not None: grad_lm_head = grad_lm_head * scale_factor
        if grad_lm_head_bias is not None: grad_lm_head_bias = grad_lm_head_bias * scale_factor

        return (None, grad_inputs, grad_lm_head, grad_lm_head_bias, None, None, None, None, None, None, None, None, None,)
    pass
pass

def unsloth_fused_ce_loss(
    trainer,
    hidden_states  : torch.Tensor,
    lm_head_weight : torch.Tensor,
    lm_head_bias   : Optional[torch.Tensor],
    labels         : torch.Tensor,
    mask           : Optional[torch.Tensor] = None,
    n_items        : Optional[torch.Tensor] = None,
    scaling        : Optional[float] = None,
    target_gb      : Optional[int] = None,
    torch_compile  : Optional[bool] = True,
    overwrite      : Optional[bool] = False,
    **kwargs,
):
    """
    Computes chunked fused cross_entropy_loss(chunk(X) @ W + b, chunk(labels))
    * If n_items is not given, does mean(ce_loss), otherwise sum(ce_loss)/n_items
    * Auto does shift of labels ie hidden_states[..., :-1] and labels[..., 1:]
    * Allows scaling factor from mixed precision fp16, fp8
    * target_gb specifies the max GB memory the fused loss can use - default detects VRAM left
    * Upcasts to float32 and allows kwargs to have:
    1) logit_scale_multiply (X = X * logit_scale_multiply)
    2) logit_scale_divide   (X = X / logit_scale_divide)
    3) logit_softcapping    (X = tanh(X / logit_softcapping) * logit_softcapping)
    """
    scaler = trainer.accelerator.scaler if trainer is not None else None
    # Get mixed precision scaling if seen
    scaling = scaler.get_scale() if scaler is not None else scaling
    if hasattr(scaling, "get_scale"): scaling = scaling.get_scale()
    if TARGET_GB: target_gb = float(TARGET_GB)
    elif N_CHUNKS: kwargs["n_chunks"] = max(int(N_CHUNKS), 1)

    # Move hidden_states to lm_head's device if they differ (e.g. multi-GPU
    # device_map="balanced"). torch.func.grad_and_value wraps inputs and fails
    # with "Cannot access storage of TensorWrapper" when tensors span devices.
    # Autograd tracks .to() and moves gradients back to the original device.
    device = lm_head_weight.device
    if hidden_states.device != device:
        hidden_states = hidden_states.to(device = device)

    return apply_autograd_function(UnslothFusedLoss, dict(
        loss_function = compute_fused_ce_loss,
        hidden_states = hidden_states,
        lm_head_weight = lm_head_weight,
        lm_head_bias = lm_head_bias,
        labels = labels,
        mask = mask,
        n_items = n_items,
        scaling = scaling,
        shift_labels = True,
        target_gb = target_gb,
        torch_compile = torch_compile,
        overwrite = overwrite,
        extra_kwargs = kwargs,
    ))
pass

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
