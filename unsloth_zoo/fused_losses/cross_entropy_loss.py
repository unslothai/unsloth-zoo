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
from unsloth_zoo.temporary_patches.common import UNSLOTH_ENABLE_LOGGING, torch_compile_options, logger
from unsloth_zoo.device_type import DEVICE_TYPE
        

TARGET_GB = os.environ.get("UNSLOTH_CE_LOSS_TARGET_GB", None)
N_CHUNKS = os.environ.get("UNSLOTH_CE_LOSS_N_CHUNKS", None)

# Register grad_and_value_impl in trace_rules (grad_impl is registered but
# grad_and_value_impl is not, which can cause GB0149 "Unsupported functorch
# tracing attempt" in some configurations).
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
    4) ignore_index         (passed to F.cross_entropy; defaults to -100)
    5) label_smoothing      (passed to F.cross_entropy; defaults to 0.0)
    """
    ignore_index = int(kwargs.get("ignore_index", -100))
    label_smoothing = float(kwargs.get("label_smoothing", 0.0))
    device = lm_head_weight.device
    if shift_labels:
        # Get shifted labels first
        _labels = torch.empty_like(labels, device = device)
        _labels[..., :-1] = labels[..., 1:]
        _labels[..., -1] = ignore_index
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
        ignore_index = ignore_index,
        label_smoothing = label_smoothing,
    )
    loss = loss / n_items if n_items is not None else loss
    # Scale loss if needed for mixed precision training
    scaled_loss = loss * scaling if scaling is not None else loss
    # Must add .loss.detach otherwise autograd uses 2x VRAM
    return scaled_loss, (loss.detach(),)
pass


# Per (token x vocab) element over the whole eager chain: bf16 logits + float32
# upcast + saved log_softmax + backward gradient = 14, and 16 under softcapping.
# 4 counted the logits alone.
_CE_BYTES_PER_LOGIT = 16.0

# ---------------------------------------------------------------------------
# Per-chunk memory budget policy.
#
# target_gb is "how much transient memory one chunk of the lm_head + CE
# computation may use". A smaller target means more chunks, which lowers peak
# memory but is strictly slower: every extra chunk re-reads the whole lm_head
# weight and relaunches the matmul + softmax + backward chain.
#
# The policy used to be a constant cap:
#     target_gb = min(free_gb * 0.5, 4.0)
# The 4.0 GiB constant is deliberate. On a very large GPU, half the free pool
# rounds the chunk count down to a single chunk, which materializes the full
# float32 logits at once and then dominates peak memory. That reasoning is
# correct on a 16 GiB card, but the constant does not scale up: on a 183 GiB
# B200 at vocab 248320 / hidden 2048 / 2048 tokens the ENTIRE transient is only
# ~7.6 GiB, and the cap still forces 4 chunks - measured 13.46 ms instead of
# 8.84 ms (1.52x slower) to save 0.71 GiB on a card with ~175 GiB spare.
#
# Setting UNSLOTH_CE_VRAM_AWARE_CAP=1 opts in to a cap that scales with how big
# the card actually is, instead of being a constant:
#     cap_gb = clamp(total_vram_gb / _CE_CAP_VRAM_DIVISOR, 4.0, 16.0)
#
# This is OPT-IN and off by default, because it is not a universal win. It is
# clearly worth it when the lm_head is trained: chunking cost is dominated by the
# lm_head weight gradient, so with lm_head frozen (ordinary LoRA) the curve is
# much flatter and the extra memory buys far less. Measured at
# vocab 248320 / hidden 2048 / 2048 tokens on a B200:
#     lm_head trained: 4 chunks 13.50 ms -> 1 chunk 8.81 ms   (+0.71 GiB)
#     lm_head frozen : 4 chunks  6.99 ms -> 1 chunk 4.99 ms   (+1.42 GiB)
# and on a 40 GiB A100 at vocab 151936 / hidden 896 with lm_head frozen the
# single-chunk landing was slightly SLOWER than 4 chunks while using ~1.9x the
# peak, because the optimum there is 2 chunks rather than 1. Until the policy
# targets the knee of that curve rather than the fewest chunks that fit, the
# default stays exactly where it is.
#
# _CE_CAP_VRAM_DIVISOR = 6.0 is picked so that the cap is EXACTLY 4.0 at 24 GiB.
# That matters for backwards compatibility:
#   * every card <= 24 GiB (T4 16, P100 16, V100 16/32->see below, L4 22.5,
#     3090/4090 24, A10 24) clamps to the 4.0 floor, i.e. bit-for-bit the old
#     policy. Nothing small can regress, and a 16 GiB T4 sees no change at all.
#   * 40 GiB A100 -> 6.67, 80 GiB A100/H100 -> 13.33, >= 96 GiB -> 16.0.
#   * the 16.0 GiB ceiling keeps the cap BOUNDED. A bigger card never degenerates
#     into "one chunk of unbounded size": any transient above 16 GiB still gets
#     chunked, which is the property the original cap was protecting.
# free_gb * 0.5 remains the other half of the min(), so a card that is large but
# already mostly full still chunks aggressively.
#
# The free-VRAM half is read LIVE on every sizing call (see get_chunk_size).
# _get_chunk_multiplier is memoized on (vocab_size, target_gb, fixed_gb), so if
# the live reading were left as the `None` sentinel the first observation would
# be frozen for the whole process - the chunk count would never react to memory
# pressure that appeared later. get_chunk_size therefore resolves None to a
# concrete number BEFORE the memoized call, so the reading is part of the key.
# Because a live float key can take unboundedly many values (on a small or busy
# card the free half binds and drifts every step), the memo is an LRU with a
# bounded size rather than an unbounded functools.cache. It still exposes
# .cache_clear() / .cache_info(), which existing tests rely on.
_CE_CAP_MIN_GB       = 4.0   # floor: the historical constant
_CE_CAP_MAX_GB       = 16.0  # ceiling: keeps the cap bounded on huge GPUs
_CE_CAP_VRAM_DIVISOR = 6.0   # total_gb / 6 == 4.0 exactly at 24 GiB
_CE_MULTIPLIER_CACHE_SIZE = 1024


def _device_mem_info(index = 0):
    """(free_bytes, total_bytes) for the accelerator, or None if unavailable.

    Covers CUDA, ROCm/HIP (torch.cuda.* is the ROCm API too) and XPU. Returns
    None on CPU / MPS / MLX builds, on driverless CI, and on any backend whose
    mem_get_info is missing or raises - callers then fall back to the historical
    constant rather than crashing.
    """
    if index is None: index = 0
    try:
        if DEVICE_TYPE == "xpu":
            return torch.xpu.mem_get_info(index)
        return torch.cuda.mem_get_info(index)
    except Exception:
        return None
pass


def _device_index_of(device):
    """Accelerator index of `device`, or None if it is not an indexed accelerator.

    The budget must be read from the card the logits will actually land on. With
    device_map="balanced" the lm_head can sit on cuda:3 while the old code always
    asked cuda:0 - harmless while the cap was a constant, but not once the cap is
    derived from that card's own size and free pool.
    """
    try:
        if not isinstance(device, torch.device): device = torch.device(device)
        if device.type in ("cpu", "meta"): return None
        return 0 if device.index is None else int(device.index)
    except Exception:
        return None
pass


def _vram_aware_cap_enabled():
    """Whether the VRAM-scaled cap is switched on. Off by default.

    Read live rather than captured at import so it can be toggled inside a
    process (tests, and A/B measurement in a single run).
    """
    return os.environ.get("UNSLOTH_CE_VRAM_AWARE_CAP", "0") == "1"
pass


def _auto_target_gb_from(free_bytes, total_bytes, vram_aware = None):
    """Pure policy: per-chunk budget in GiB from a (free, total) memory pair.

    Split out from the device read so it can be reasoned about and tested
    without a GPU.

    With vram_aware off (the default) this is exactly the historical
    `min(free_gb * 0.5, 4.0)`. With it on, the 4.0 constant becomes a cap that
    scales with the size of the card.
    """
    if vram_aware is None: vram_aware = _vram_aware_cap_enabled()
    free_gb  = free_bytes  / 1024 / 1024 / 1024
    cap_gb   = _CE_CAP_MIN_GB
    if vram_aware:
        total_gb = total_bytes / 1024 / 1024 / 1024
        # Cap scales with the card, clamped to [_CE_CAP_MIN_GB, _CE_CAP_MAX_GB].
        cap_gb = total_gb / _CE_CAP_VRAM_DIVISOR
        if cap_gb < _CE_CAP_MIN_GB: cap_gb = _CE_CAP_MIN_GB
        if cap_gb > _CE_CAP_MAX_GB: cap_gb = _CE_CAP_MAX_GB
    pass
    # Still never use more than 50% of what is actually free right now. On every
    # card <= 24 GiB cap_gb is exactly _CE_CAP_MIN_GB == 4.0, so even with the
    # flag on this reduces to the previous `min(free_gb * 0.5, 4.0)` bit-for-bit.
    return min(free_gb * 0.5, cap_gb)
pass


def _auto_target_gb(index = 0):
    """Resolve the automatic per-chunk budget from live device memory.

    Returns the historical 4.0 GiB constant when the device cannot report its
    memory, so an unknown backend degrades to exactly the previous behaviour.
    """
    if index is None: index = 0
    info = _device_mem_info(index)
    if info is None:
        return _CE_CAP_MIN_GB
    return _auto_target_gb_from(info[0], info[1])
pass


@functools.lru_cache(maxsize = _CE_MULTIPLIER_CACHE_SIZE)
def _get_chunk_multiplier(vocab_size, target_gb = None, fixed_gb = 0.0):
    """Chunk multiplier sized to fit target max memory usage."""
    if target_gb is None:
        # Legacy/direct callers may still pass None. get_chunk_size resolves it
        # before this point so the hot path never freezes a VRAM reading here.
        target_gb = _auto_target_gb()
    pass

    # Prevent ZeroDivisionError when GPU memory is exhausted
    if target_gb <= 1e-9: # Use a small epsilon for float comparison
        raise RuntimeError("Unsloth: No or negligible GPU memory available for fused cross entropy.")

    # Unchunkable allocations share the budget; if they alone exceed the target
    # no chunk count helps, so keep the full budget instead.
    if 0.0 < fixed_gb < target_gb:
        target_gb = target_gb - fixed_gb
    pass

    multiplier = (vocab_size * _CE_BYTES_PER_LOGIT / 1024 / 1024 / 1024) / (target_gb)
    multiplier = multiplier / 4 # Output only multiples of 4
    return multiplier
pass

def get_chunk_size(bsz, qlen, vocab_size, target_gb = None, fixed_gb = 0.0,
                   device_index = None):
    """Number of chunks that fits the target max memory usage.

    device_index is optional and trailing, so every existing positional or
    keyword call site keeps working unchanged; when omitted the budget is read
    from device 0 exactly as before.
    """
    # Resolve the automatic budget HERE, not inside the memoized multiplier:
    # _get_chunk_multiplier is cached on its arguments, so passing the None
    # sentinel through would pin the very first VRAM observation for the
    # lifetime of the process. Resolving first makes the live reading part of
    # the cache key, so the cache stays correct as memory pressure changes.
    if target_gb is None:
        target_gb = _auto_target_gb(device_index)
    multiplier = _get_chunk_multiplier(vocab_size, target_gb, fixed_gb)
    n_splits = (bsz*qlen) * multiplier
    # n_splits * 4 == (chunk transient GiB) / target. Round UP: nearest-rounding
    # (round(0.5) -> 0) collapses a large transient into one uncapped chunk.
    exact = n_splits * 4
    if exact <= 1.0 + 1e-9:
        return 1
    n_chunks = math.ceil(exact / 4 - 1e-9) * 4
    return min(n_chunks, bsz*qlen)
pass

class UnslothFusedLoss(torch.autograd.Function):
    # Log the "scaling=0" info message at most once per process.
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
        # Thread ignore_index through label-shift and the inner CE call.
        ignore_index = int(extra_kwargs.get("ignore_index", -100))

        # Get shifted labels first
        if shift_labels:
            _labels = torch.empty_like(labels, device = device)
            _labels[..., :-1] = labels[..., 1:]
            # Also check mask
            if mask is not None:
                mask = mask.to(device = device)
                _labels[..., :-1][mask[..., 1:] == 0] = ignore_index
            pass
            _labels[..., -1] = ignore_index
            _labels = _labels.view(-1)
            labels = _labels
        else:
            # Caller already shifted (e.g. trl padding_free passes
            # shift_labels=<tensor>). Flatten so chunking aligns with
            # hidden_states.reshape(-1, hd).
            labels = labels.contiguous().view(-1).to(device = device)
        pass

        # N items divisor
        divisor = n_items if n_items is not None else (labels != ignore_index).sum()
        if not torch.is_tensor(divisor):
            divisor = torch.tensor(divisor, dtype = torch.float32, device = device)
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
            # Memory no chunk count can shrink. Under overwrite grad_inputs
            # aliases hidden_states; the head gradient counts twice (per chunk).
            fixed_bytes = 0
            if not overwrite:
                fixed_bytes += grad_inputs.numel() * grad_inputs.element_size()
            if grad_lm_head is not None:
                fixed_bytes += 2 * grad_lm_head.numel() * grad_lm_head.element_size()
            if grad_lm_head_bias is not None:
                fixed_bytes += 2 * grad_lm_head_bias.numel() * grad_lm_head_bias.element_size()
            n_chunks = get_chunk_size(
                bsz, qlen, vocab_size, target_gb = target_gb,
                fixed_gb = fixed_bytes / 1024 / 1024 / 1024,
                # Size against the card the logits actually land on, not cuda:0.
                device_index = _device_index_of(device),
            )
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
                    False, # Outer pre-shifted (or caller did); inner skips
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
                    False, # Outer pre-shifted (or caller did); inner skips
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
                    False, # Outer pre-shifted (or caller did); inner skips
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
                    False, # Outer pre-shifted (or caller did); inner skips
                    **kwargs,
                )
            pass
            accumulated_loss.add_(unscaled_loss)
            grad_inputs_j[:] = chunk_grad_input
        pass
        global _FUSED_CE_COMPILE_SUPPORTED
        uncompiled_accumulate_chunk = accumulate_chunk

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

            _iter = iter(zip(__grad_inputs, __shift_states, __shift_labels))
            grad_inputs_j, hidden_states_j, labels_j = next(_iter)
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
            for (grad_inputs_j, hidden_states_j, labels_j,) in \
                zip(__grad_inputs, __shift_states, __shift_labels,):
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

        # Out-of-place mul so ctx.saved_tensors' version counter doesn't bump,
        # keeping retain_graph / double-backward flows working. Measured peak
        # memory delta vs in-place is <3 MB across 14 configs.
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
    shift_labels   : bool = True,
    **kwargs,
):
    """
    Computes chunked fused cross_entropy_loss(chunk(X) @ W + b, chunk(labels))
    * If n_items is not given, does mean(ce_loss), otherwise sum(ce_loss)/n_items
    * shift_labels=True (default) shifts internally: hidden_states[..., :-1] and labels[..., 1:].
      Set False when caller already pre-shifted (e.g. trl padding_free).
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
        shift_labels = shift_labels,
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
