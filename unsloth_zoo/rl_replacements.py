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

__all__ = [
    "RL_REPLACEMENTS"
]

import torch
import inspect
import os
import sys
import math
import logging
from typing import Union, Callable, Optional, List, Dict
from .device_type import DEVICE_TYPE, device_synchronize
from unsloth_zoo.temporary_patches.common import (
    torch_compile_options,
    _maybe_compile,
)
from unsloth_zoo.log import logger



RL_REPLACEMENTS = dict()

# https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L1674
@_maybe_compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def selective_log_softmax(logits, index):
    logits = logits.to(torch.float32)
    selected_logits = torch.gather(logits, dim = -1, index = index.unsqueeze(-1)).squeeze(-1)
    logsumexp_values = torch.logsumexp(logits, dim = -1)
    per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    return per_token_logps
pass

def _truncated_support_mask_impl(logits: torch.Tensor, top_k: int, top_p: float, min_p: float) -> torch.Tensor:
    if min_p > 0.0:
        probs = torch.softmax(logits, dim = -1)
        logits = logits.masked_fill(probs < min_p * probs.amax(dim = -1, keepdim = True), float("-inf"))
    if top_k > 0 and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim = -1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        # Per sorted position, as vLLM does: a cutoff inside a run of equal logits keeps only part of it.
        sorted_logits, sorted_idx = torch.sort(logits, dim = -1)
        dropped = torch.cumsum(torch.softmax(sorted_logits, dim = -1), dim = -1) <= (1.0 - top_p)
        dropped[:, -1] = False
        logits = logits.scatter(-1, sorted_idx, sorted_logits.masked_fill(dropped, float("-inf")))
    return torch.isfinite(logits)
pass

# Opaque to Inductor: with dynamic chunk rows its lowering of these reductions raises CantSplit
# (torch 2.13), and the set is a constant of the backward anyway.
try:
    _truncated_support_mask = torch.library.custom_op(
        "unsloth_zoo::truncated_support_mask", mutates_args = (),
    )(_truncated_support_mask_impl)
    _truncated_support_mask.register_fake(
        lambda logits, top_k, top_p, min_p: torch.empty_like(logits, dtype = torch.bool)
    )
except Exception:
    _truncated_support_mask = _truncated_support_mask_impl
pass

# Mirrors vLLM v1 processed_logprobs: renormalised over what survives min_p, then top_k, then top_p
# (sampler.py `sample`, topk_topp_sampler.py `apply_top_k_top_p`). The scored token is always kept
# so a boundary disagreement between the trainer's recomputed set and the sampler's stays finite.
# A scored token tied with the smallest kept logit but dropped by the sort counts as one of the kept
# tokens (vLLM keeps an arbitrary subset of a tied run): it takes the place of one kept tied token,
# so both the value and its gradient are those of a kept token.
def _truncated_support_logps(chunk_logits, chunk_index, top_k, top_p, min_p):
    index = chunk_index.unsqueeze(-1)
    selected_logits = torch.gather(chunk_logits, dim = -1, index = index)
    detached = chunk_logits.detach()
    keep = _truncated_support_mask(detached, top_k, top_p, min_p)
    selected_kept = torch.gather(keep, dim = -1, index = index)
    smallest_kept = detached.masked_fill(~keep, float("inf")).amin(dim = -1, keepdim = True)
    tied = (~selected_kept) & (selected_logits.detach() == smallest_kept)
    boundary = (keep & (detached == smallest_kept)).to(torch.int32).argmax(dim = -1, keepdim = True)
    keep = keep.scatter(-1, boundary, torch.gather(keep, dim = -1, index = boundary) & ~tied)
    keep = keep.scatter(-1, index, torch.ones_like(selected_kept))
    kept = chunk_logits.masked_fill(~keep, float("-inf"))
    return (selected_logits - torch.logsumexp(kept, dim = -1, keepdim = True)).squeeze(-1)
pass

# Memory-efficient chunked variant of the above on (bsz+qlen); exactly equivalent.
@_maybe_compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_selective_log_softmax(
    logits,
    index,
    temperature: float = 1.0,
    chunks: int = 4,
    support_top_k: int = 0,
    support_top_p: float = 1.0,
    support_min_p: float = 0.0,
):
    # Fixed `chunks` row ranges; see chunked_hidden_states_selective_log_softmax.
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_index = index.reshape(-1)
    # Fewer rows than chunks: one row per range, as torch.chunk did, never empty ranges.
    chunks = max(1, min(chunks, flat_logits.shape[0]))
    bounds = [(i * flat_logits.shape[0]) // chunks for i in range(chunks + 1)]
    chunked_logits = [flat_logits[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    chunked_index  = [flat_index[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    all_per_token_logps = []
    # Per-chunk selective_log_softmax.
    for chunk_logits, chunk_index in zip(chunked_logits, chunked_index):
        chunk_logits = chunk_logits.to(torch.float32)
        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature
        if support_top_k > 0 or support_top_p < 1.0 or support_min_p > 0.0:
            all_per_token_logps.append(_truncated_support_logps(
                chunk_logits, chunk_index, support_top_k, support_top_p, support_min_p,
            ))
            continue
        selected_logits = torch.gather(chunk_logits, dim = -1, index = chunk_index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.logsumexp(chunk_logits, dim = -1)
        per_token_logps = selected_logits - logsumexp_values
        all_per_token_logps.append(per_token_logps)
    pass
    all_per_token_logps = torch.concat(all_per_token_logps)
    all_per_token_logps = all_per_token_logps.reshape((logits.shape[0], logits.shape[1]))
    return all_per_token_logps
pass

RL_REPLACEMENTS["selective_log_softmax"] = chunked_selective_log_softmax

@_maybe_compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_hidden_states_selective_log_softmax(
    hidden_states: torch.Tensor,
    lm_head: torch.Tensor,
    index: torch.Tensor,
    chunks: int = 4,
    logit_scale_multiply: float = 0.0,
    logit_scale_divide: float = 0.0,
    logit_softcapping: float = 0.0,
    temperature: float = 1.0,
    # Rows per chunk cap. Read HERE, in the default, not in the body: the body
    # is traced with fullgraph = True, and `os.environ` is an unsupported op
    # there on torch 2.4 -- Dynamo raises
    #   torch._dynamo.exc.Unsupported: const method call bytes.decode
    # from os._Environ.__getitem__, which the eager fallback does not catch
    # (it only catches recompile-limit and disabled-hook breaks), so the very
    # first call would die even with the variable unset. A default is evaluated
    # once when the def runs, which is import time, outside any traced region.
    # It is also a plain int argument, so Dynamo guards on it instead of
    # constant-folding an unguarded read (2.7+ never notice a later change).
    # A non-numeric value is ignored rather than raised on, so a typo cannot
    # break the import. 0 keeps the previous chunk boundaries exactly.
    max_rows_per_chunk: int = (
        int(os.environ.get("UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK", "0").strip())
        if os.environ.get("UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK", "0").strip().isdigit()
        else 0
    ),
    lm_head_bias: Optional[torch.Tensor] = None,
    support_top_k: int = 0,
    support_top_p: float = 1.0,
    support_min_p: float = 0.0,
    # Read in the default for the same Dynamo reason as max_rows_per_chunk. Pairs with vLLM's
    # head_dtype=float32 that load_vllm requests under the same flag.
    fp32_lm_head: bool = (
        os.environ.get("UNSLOTH_VLLM_FP32_LM_HEAD", "0").strip().lower() in ("1", "true", "on", "yes")
    ),
) -> torch.Tensor:
    # All Unsloth Zoo code licensed under AGPL3
    # Reshape on this tensor's own last dim: a no-op, so a wrong-width caller
    # cannot have its row count silently rewritten and instead fails at the
    # matmul below, which prints both operands. Do not swap in a bare
    # torch._check: it reports only "Expected cond to be True", naming neither
    # operand, and Dynamo rejects a message-carrying one. Callers dispatch on
    # the width first -- see `compute_logprobs_chunk`, the packed path and
    # `_pg_grad_forward`.
    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_index = index.reshape(-1)

    # Each chunk materialises rows x vocab logits and then a float32 copy of
    # them, all on the device holding the output head. With a large vocabulary
    # and a fixed chunk count that grows with the batch, so the peak scales with
    # the batch rather than staying bounded. max_rows_per_chunk caps the rows
    # per chunk instead, which is pure loop splitting: more, smaller chunks,
    # same concatenated result. 0 (the default) keeps the previous chunk
    # boundaries exactly.
    if max_rows_per_chunk > 0:
        n_rows = flat_hidden_states.shape[0]
        chunks = max(chunks, -(-n_rows // max_rows_per_chunk))
        chunks = min(chunks, max(n_rows, 1))

    # A one-row chunk's head gradient is a K=1 matmul Inductor mis-lowers under dynamic shapes (illegal
    # memory access or NaN head gradients, torch 2.13), so keep chunks at 2+ rows when the head trains.
    # Exactly `chunks` row ranges, so the unrolled loop length never depends on T: torch.chunk
    # returns ceil(T / ceil(T / chunks)) pieces and recompiled per new count; tensor_split guards
    # on T % chunks. At most rows // 2 chunks when the head trains keeps every piece at 2+ rows.
    lm_head_grad = torch.is_grad_enabled() and lm_head.requires_grad
    if lm_head_grad and flat_hidden_states.shape[0] < 2 * chunks:
        chunks = max(flat_hidden_states.shape[0] // 2, 1)

    n_rows = flat_hidden_states.shape[0]
    chunks = max(1, min(chunks, n_rows))
    bounds = [(i * n_rows) // chunks for i in range(chunks + 1)]
    chunked_hidden_states = [flat_hidden_states[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    chunked_index = [flat_index[a:b] for a, b in zip(bounds[:-1], bounds[1:])]

    # Cast once, outside the loop, so every chunk shares one fp32 head copy (also for autograd).
    if fp32_lm_head:
        lm_head = lm_head.to(torch.float32)
        if lm_head_bias is not None:
            lm_head_bias = lm_head_bias.to(torch.float32)

    all_per_token_logps = []

    for chunk_hidden_states, chunk_index in zip(chunked_hidden_states, chunked_index):
        # When the model is dispatched over several devices, the output head can
        # sit on a different one from the hidden states, because accelerate
        # places the tail of the model on the last device it fills. Co-locate on
        # the head's device before the matmul, otherwise this raises
        #   Unhandled FakeTensor Device Propagation for aten.mm.default,
        #   found two different devices cuda:0, cuda:1
        # On a single device every .to() here is a no-op and the result is
        # bit-identical to before.
        chunk_hidden_states = chunk_hidden_states.to(device = lm_head.device, dtype = lm_head.dtype)
        chunk_index = chunk_index.to(lm_head.device)
        if fp32_lm_head:
            # The GRPO loss runs this under bf16 / fp16 autocast, which would downcast the matmul.
            with torch.autocast(device_type = chunk_hidden_states.device.type, enabled = False):
                chunk_logits = torch.nn.functional.linear(chunk_hidden_states, lm_head, lm_head_bias)
        elif lm_head_bias is None:
            chunk_logits = chunk_hidden_states @ lm_head.t()
        else:
            chunk_logits = torch.nn.functional.linear(chunk_hidden_states, lm_head, lm_head_bias)

        if logit_scale_multiply != 0.0:
            chunk_logits = chunk_logits * logit_scale_multiply
        if logit_scale_divide != 0.0:
            chunk_logits = chunk_logits / logit_scale_divide
        if logit_softcapping != 0.0:
            chunk_logits = logit_softcapping * torch.tanh(chunk_logits / logit_softcapping)

        chunk_logits = chunk_logits.to(torch.float32)

        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature

        if support_top_k > 0 or support_top_p < 1.0 or support_min_p > 0.0:
            per_token_logps = _truncated_support_logps(
                chunk_logits, chunk_index, support_top_k, support_top_p, support_min_p,
            )
        else:
            selected_logits = torch.gather(chunk_logits, dim=-1, index=chunk_index.unsqueeze(-1)).squeeze(-1)
            logsumexp_values = torch.logsumexp(chunk_logits, dim=-1)
            per_token_logps = selected_logits - logsumexp_values
        # Return to the caller's device so the concatenation below and every
        # downstream consumer see the device they started on.
        all_per_token_logps.append(per_token_logps.to(hidden_states.device))

    all_per_token_logps = torch.concat(all_per_token_logps)

    all_per_token_logps = all_per_token_logps.reshape((hidden_states.shape[0], hidden_states.shape[1]))
    return all_per_token_logps

RL_REPLACEMENTS["grpo_selective_log_softmax"] = chunked_hidden_states_selective_log_softmax

# Keeps only per-row logsumexp and recomputes each chunk's logits in backward (as verl's
# FusedLinearForPPOFunction). Self-contained (torch, os): its source is copied into trainers.
def chunked_hidden_states_selective_log_softmax_recompute(
    hidden_states: torch.Tensor,
    lm_head: torch.Tensor,
    index: torch.Tensor,
    chunks: int = 4,
    logit_scale_multiply: float = 0.0,
    logit_scale_divide: float = 0.0,
    logit_softcapping: float = 0.0,
    temperature: float = 1.0,
    max_rows_per_chunk: int = (
        int(os.environ.get("UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK", "0").strip())
        if os.environ.get("UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK", "0").strip().isdigit()
        else 0
    ),
    lm_head_bias: Optional[torch.Tensor] = None,
    return_entropy: bool = False,
    # Same flag and default as chunked_hidden_states_selective_log_softmax, which scores old / ref
    # logps: entropy and recompute scoring must project the head in the same precision.
    fp32_lm_head: bool = (
        os.environ.get("UNSLOTH_VLLM_FP32_LM_HEAD", "0").strip().lower() in ("1", "true", "on", "yes")
    ),
):
    # All Unsloth Zoo code licensed under AGPL3
    self_fn = chunked_hidden_states_selective_log_softmax_recompute
    function = getattr(self_fn, "_autograd_function", None)
    if function is None:
        def post_forward(logits, index, scale_multiply, scale_divide, softcapping, temperature, with_entropy):
            # The scale / softcap run in the matmul dtype, then float32, exactly as the plain path.
            if scale_multiply != 0.0:
                logits = logits * scale_multiply
            if scale_divide != 0.0:
                logits = logits / scale_divide
            if softcapping != 0.0:
                logits = softcapping * torch.tanh(logits / softcapping)
            logits = logits.to(torch.float32)
            if temperature != 1.0:
                logits = logits / temperature
            logsumexp = torch.logsumexp(logits, dim = -1)
            selected = torch.gather(logits, dim = -1, index = index.unsqueeze(-1)).squeeze(-1)
            if with_entropy:
                # -sum(p * log p) on log p = z - logsumexp: logsumexp - sum(p * z) cancels badly.
                log_p = logits - logsumexp.unsqueeze(-1)
                entropy = -(torch.exp(log_p) * log_p).sum(-1)
            else:
                entropy = torch.zeros_like(logsumexp)
            return selected - logsumexp, logsumexp, entropy

        def post_backward(
            logits, index, logsumexp, entropy, grad_logps, grad_entropy,
            scale_multiply, scale_divide, softcapping, temperature, with_entropy,
        ):
            # z exactly as the forward made it, so p matches the saved logsumexp; the softcap
            # derivative in at least float32 (autograd would take it in the matmul dtype).
            z = logits
            if scale_multiply != 0.0:
                z = z * scale_multiply
            if scale_divide != 0.0:
                z = z / scale_divide
            if softcapping != 0.0:
                z = softcapping * torch.tanh(z / softcapping)
                x = logits.to(torch.promote_types(logits.dtype, torch.float32))
                if scale_multiply != 0.0:
                    x = x * scale_multiply
                if scale_divide != 0.0:
                    x = x / scale_divide
                tanh = torch.tanh(x / softcapping)
            z = z.to(torch.float32)
            if temperature != 1.0:
                z = z / temperature
            log_p = z - logsumexp.unsqueeze(-1)
            p = torch.exp(log_p)
            # d logp / dz = onehot - p, d H / dz = -p * (log p + H)
            if with_entropy:
                grad_z = -p * (grad_logps.unsqueeze(-1) + grad_entropy.unsqueeze(-1) * (log_p + entropy.unsqueeze(-1)))
            else:
                grad_z = -p * grad_logps.unsqueeze(-1)
            # A fused one-hot rather than scatter_add, which would materialise grad_z in float32.
            onehot = torch.arange(z.shape[-1], device = z.device) == index.unsqueeze(-1)
            grad_z = grad_z + onehot * grad_logps.unsqueeze(-1)
            if temperature != 1.0:
                grad_z = grad_z / temperature
            if softcapping != 0.0:
                grad_z = grad_z * (1.0 - tanh * tanh)
            if scale_divide != 0.0:
                grad_z = grad_z / scale_divide
            if scale_multiply != 0.0:
                grad_z = grad_z * scale_multiply
            return grad_z.to(logits.dtype)

        try:
            from unsloth_zoo.temporary_patches.common import _maybe_compile as compile_function
        except ImportError:
            compile_function = torch.compile
        compiled_forward  = compile_function(dynamic = True, fullgraph = True)(post_forward)
        compiled_backward = compile_function(dynamic = True, fullgraph = True)(post_backward)

        def run(kernel, eager, *args):
            # getattr: the cached autograd function can outlive the flag (a test's monkeypatch undo).
            if getattr(self_fn, "_use_compiled", True):
                try:
                    return kernel(*args)
                except Exception:
                    self_fn._use_compiled = False
            return eager(*args)

        # aten::mm.dtype (torch >= 2.8) has CUDA / ROCm kernels only.
        mm_out_dtype = {"cuda": hasattr(torch.ops.aten.mm, "dtype")}

        def mm_float32(a, b):
            # fp16 GEMM with float32 output where it exists, else a float32 matmul of the chunk.
            if mm_out_dtype.get(a.device.type, False):
                try:
                    return torch.mm(a, b, out_dtype = torch.float32)
                except torch.cuda.OutOfMemoryError:
                    raise
                except (TypeError, RuntimeError, NotImplementedError):
                    mm_out_dtype[a.device.type] = False
            return torch.mm(a.float(), b.float())

        def chunk_logits(chunk_hidden_states, lm_head, lm_head_bias):
            chunk_hidden_states = chunk_hidden_states.to(device = lm_head.device, dtype = lm_head.dtype)
            if lm_head_bias is None:
                return chunk_hidden_states @ lm_head.t()
            return torch.nn.functional.linear(chunk_hidden_states, lm_head, lm_head_bias)

        class ChunkedSelectiveLogSoftmax(torch.autograd.Function):
            @staticmethod
            def forward(ctx, hidden_states, lm_head, lm_head_bias, index, bounds, transforms):
                device_type = lm_head.device.type
                ctx.autocast_kwargs = dict(
                    device_type = device_type,
                    enabled = torch.is_autocast_enabled(device_type),
                    dtype = torch.get_autocast_dtype(device_type),
                    cache_enabled = torch.is_autocast_cache_enabled(),
                )
                logps, logsumexps, entropies = [], [], []
                for a, b in zip(bounds[:-1], bounds[1:]):
                    logits = chunk_logits(hidden_states[a:b], lm_head, lm_head_bias)
                    chunk_index = index[a:b].to(lm_head.device)
                    logp, logsumexp, entropy = run(
                        compiled_forward, post_forward, logits, chunk_index, *transforms,
                    )
                    logps.append(logp)
                    logsumexps.append(logsumexp)
                    entropies.append(entropy)
                logps = torch.cat(logps).to(hidden_states.device)
                entropies = torch.cat(entropies).to(hidden_states.device)
                ctx.save_for_backward(hidden_states, lm_head, lm_head_bias, index, torch.cat(logsumexps), entropies)
                ctx.bounds = bounds
                ctx.transforms = transforms
                return logps, entropies

            @staticmethod
            def backward(ctx, grad_logps, grad_entropies):
                hidden_states, lm_head, lm_head_bias, index, logsumexps, entropies = ctx.saved_tensors
                need_hidden, need_head, need_bias = ctx.needs_input_grad[:3]
                device = lm_head.device
                if grad_logps is None:
                    grad_logps = torch.zeros_like(entropies)
                if grad_entropies is None:
                    grad_entropies = torch.zeros_like(entropies)
                grad_logps = grad_logps.to(device = device, dtype = torch.float32)
                grad_entropies = grad_entropies.to(device = device, dtype = torch.float32)
                entropies = entropies.to(device)
                grad_hidden = torch.empty_like(hidden_states) if need_hidden else None
                grad_head = None
                grad_bias = torch.zeros(lm_head_bias.shape, dtype = torch.float32, device = device) if need_bias else None
                head = None
                for a, b in zip(ctx.bounds[:-1], ctx.bounds[1:]):
                    chunk_hidden_states = hidden_states[a:b]
                    with torch.no_grad(), torch.autocast(**ctx.autocast_kwargs):
                        logits = chunk_logits(chunk_hidden_states, lm_head, lm_head_bias)
                    grad_logits = run(
                        compiled_backward, post_backward,
                        logits, index[a:b].to(device), logsumexps[a:b], entropies[a:b],
                        grad_logps[a:b], grad_entropies[a:b], *ctx.transforms,
                    )
                    del logits
                    if head is None:
                        head = lm_head if lm_head.dtype == grad_logits.dtype else lm_head.to(grad_logits.dtype)
                        half_to_float32 = grad_logits.dtype == torch.float16
                    if need_hidden:
                        if half_to_float32 and hidden_states.dtype != torch.float16:
                            # A bf16 / fp32 hidden state can hold a row sum an fp16 product cannot.
                            grad_chunk = mm_float32(grad_logits, head)
                        else:
                            grad_chunk = grad_logits @ head
                        grad_hidden[a:b] = grad_chunk.to(device = hidden_states.device, dtype = hidden_states.dtype)
                        del grad_chunk
                    if need_head:
                        x = chunk_hidden_states.to(device = device, dtype = grad_logits.dtype)
                        if grad_head is None:
                            # Accumulate in place in the head's own dtype when the matmul ran in it,
                            # as autograd does, except fp16: its chunk products and running sum can
                            # leave the fp16 range before opposite signed chunks cancel. fp16 and a
                            # float32 head under autocast accumulate in float32, cast once at the end.
                            same_dtype = lm_head.dtype == grad_logits.dtype and not half_to_float32
                            grad_head = torch.zeros(
                                lm_head.shape, device = device,
                                dtype = grad_logits.dtype if same_dtype else torch.float32,
                            )
                        if grad_head.dtype == grad_logits.dtype:
                            grad_head.addmm_(grad_logits.t(), x)
                        elif half_to_float32:
                            grad_head.add_(mm_float32(grad_logits.t(), x))
                        else:
                            grad_head.add_(grad_logits.t() @ x)
                    if need_bias:
                        grad_bias.add_(grad_logits.to(torch.float32).sum(0))
                    del grad_logits
                if need_head:
                    grad_head = grad_head.to(lm_head.dtype)
                if need_bias:
                    grad_bias = grad_bias.to(lm_head_bias.dtype)
                return grad_hidden, grad_head, grad_bias, None, None, None

        self_fn._use_compiled = getattr(self_fn, "_use_compiled", True)
        self_fn._autograd_function = function = ChunkedSelectiveLogSoftmax

    if hidden_states is None:
        # Build-only call, made once at import: a cold outer torch.compile(fullgraph = True) cannot
        # trace the setup above (class creation, compile wrappers), only the cached function.
        return function
    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_index = index.reshape(-1)
    n_rows = flat_hidden_states.shape[0]
    if max_rows_per_chunk > 0:
        chunks = max(chunks, -(-n_rows // max_rows_per_chunk))
        chunks = min(chunks, max(n_rows, 1))
    if torch.is_grad_enabled() and lm_head.requires_grad and n_rows < 2 * chunks:
        chunks = max(n_rows // 2, 1)
    chunks = max(1, min(chunks, n_rows))
    bounds = tuple((i * n_rows) // chunks for i in range(chunks + 1))
    transforms = (
        float(logit_scale_multiply), float(logit_scale_divide),
        float(logit_softcapping), float(temperature), bool(return_entropy),
    )
    if fp32_lm_head:
        # One fp32 head copy (autograd casts its gradient back), and no autocast, which would
        # downcast the matmul; forward saves the disabled state, so backward recomputes in fp32 too.
        lm_head = lm_head.to(torch.float32)
        if lm_head_bias is not None:
            lm_head_bias = lm_head_bias.to(torch.float32)
        with torch.autocast(device_type = lm_head.device.type, enabled = False):
            logps, entropies = function.apply(flat_hidden_states, lm_head, lm_head_bias, flat_index, bounds, transforms)
    else:
        logps, entropies = function.apply(flat_hidden_states, lm_head, lm_head_bias, flat_index, bounds, transforms)
    logps = logps.reshape(hidden_states.shape[0], hidden_states.shape[1])
    if return_entropy:
        return logps, entropies.reshape(hidden_states.shape[0], hidden_states.shape[1])
    return logps
pass
chunked_hidden_states_selective_log_softmax_recompute(None, None, None)
RL_REPLACEMENTS["grpo_selective_log_softmax_recompute"] = chunked_hidden_states_selective_log_softmax_recompute

def calculate_pad_tokens_in_prompt(
    input_ids: torch.Tensor,
    logits_to_keep: int,
    pad_token_id: int
) -> torch.Tensor:
    """Count left-padded tokens per sequence, e.g. [pad, pad, pad, cat] -> 3."""
    if logits_to_keep >= input_ids.shape[1]:
        raise ValueError("logits_to_keep must be smaller than the sequence length.")

    prompt_section = input_ids[:, :-logits_to_keep]

    padding_mask = (prompt_section == pad_token_id)

    pad_token_counts = padding_mask.sum(dim=1)

    return pad_token_counts
pass
RL_REPLACEMENTS["calculate_pad_tokens_in_prompt"] = calculate_pad_tokens_in_prompt


def create_completion_attention_mask(
    completion_input_ids: torch.Tensor,
    left_pad_tokens_per_prompt: torch.Tensor,
    max_left_pad: int,
    pad_token_id: int
) -> torch.Tensor:
    """Build a completion mask that zeros leading prompt and trailing pad tokens.

    For [p,p,p,c,c,c,pad,pad,pad] (p=sliced prompt, c=completion, pad=padding)
    this returns [0,0,0,1,1,1,0,0,0].
    """
    batch_size, completion_len = completion_input_ids.shape
    device = completion_input_ids.device

    num_tokens_to_mask = max_left_pad - left_pad_tokens_per_prompt

    indices = torch.arange(completion_len, device=device).unsqueeze(0)
    shift_mask = indices >= num_tokens_to_mask.unsqueeze(1)

    non_padding_mask = (completion_input_ids != pad_token_id)

    final_mask = shift_mask & non_padding_mask

    return final_mask
pass
RL_REPLACEMENTS["create_completion_attention_mask"] = create_completion_attention_mask


# Rebuild Qwen-style mm_token_type_ids from full input_ids after GRPO generation changes sequence length.
# This is primarily towards VLMs that use MRoPE
def _unsloth_get_mm_token_id(processing_class, attr_name, token):
    tokenizer = getattr(processing_class, "tokenizer", processing_class)
    token_id = getattr(processing_class, attr_name, None)
    if token_id is None:
        token_id = getattr(tokenizer, attr_name, None)

    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if token_id is None and convert_tokens_to_ids is not None:
        token_id = convert_tokens_to_ids(token)

    if type(token_id) is int and token_id >= 0:
        if token_id != getattr(tokenizer, "unk_token_id", None):
            return token_id
    return None
pass


def _unsloth_fix_mm_token_type_ids(
    processing_class, input_ids, mm_token_type_ids = None, completion_ids = None
):
    image_token_id = _unsloth_get_mm_token_id(
        processing_class, "image_token_id", "<|image_pad|>"
    )
    video_token_id = _unsloth_get_mm_token_id(
        processing_class, "video_token_id", "<|video_pad|>"
    )

    if image_token_id is not None or video_token_id is not None:
        rebuilt = input_ids.new_zeros(input_ids.shape)
        if image_token_id is not None:
            rebuilt = rebuilt.masked_fill(input_ids == image_token_id, 1)
        if video_token_id is not None:
            rebuilt = rebuilt.masked_fill(input_ids == video_token_id, 2)
        return rebuilt

    if (
        mm_token_type_ids is not None
        and completion_ids is not None
        and mm_token_type_ids.shape[0] == input_ids.shape[0]
        and mm_token_type_ids.shape[1] + completion_ids.shape[1] == input_ids.shape[1]
    ):
        return torch.cat(
            [mm_token_type_ids, mm_token_type_ids.new_zeros(completion_ids.shape)],
            dim = 1,
        )
    return mm_token_type_ids
pass

def left_pack_padding(tensor: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Move all padding tokens in each sequence to the right."""
    mask = (tensor != pad_id)
    # stable=True since the binary mask is unordered.
    sorted_indices = torch.argsort(mask, dim=1, descending=True, stable=True)
    packed_tensor = torch.gather(tensor, 1, sorted_indices)
    return packed_tensor
pass
RL_REPLACEMENTS["left_pack_padding"] = left_pack_padding

def align_logprobs_with_mask(
    logprob_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_value: float = 0.0
) -> torch.Tensor:
    """Align a log probability tensor with a given attention mask."""

    device = logprob_tensor.device
    batch_size, logprob_seq_len = logprob_tensor.shape
    mask_seq_len = attention_mask.shape[1]

    padded_logprobs = torch.full(
        attention_mask.shape,
        fill_value=pad_value,
        dtype=logprob_tensor.dtype,
        device=device
    )

    left_pad_counts = torch.argmax(attention_mask, dim=1)

    cols = torch.arange(logprob_seq_len, device=device)


    dest_indices = left_pad_counts.unsqueeze(1) + cols

    # Destination row indices, shape [batch_size, logprob_seq_len].
    row_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(dest_indices)

    # Keep only in-bounds destinations, then scatter via advanced indexing.
    valid_mask = dest_indices < mask_seq_len
    valid_rows = row_indices[valid_mask]
    valid_cols = dest_indices[valid_mask]
    valid_vals = logprob_tensor[valid_mask]
    padded_logprobs[valid_rows, valid_cols] = valid_vals

    return padded_logprobs

RL_REPLACEMENTS["align_logprobs_with_mask"] = align_logprobs_with_mask

def align_completion_tool_mask(
    tool_mask: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Align a raw completion-length tool/env mask with Unsloth's repacked loss mask."""
    if tool_mask is None:
        return completion_mask
    if tool_mask.shape[0] != completion_mask.shape[0]:
        raise ValueError("tool_mask batch size must match completion_mask batch size.")

    tool_mask = tool_mask.to(device=completion_mask.device)
    if tool_mask.shape == completion_mask.shape:
        aligned_tool_mask = tool_mask
    else:
        aligned_tool_mask = align_logprobs_with_mask(
            tool_mask,
            completion_mask,
            pad_value=0,
        )
    return completion_mask * aligned_tool_mask.to(dtype=completion_mask.dtype)
pass
RL_REPLACEMENTS["align_completion_tool_mask"] = align_completion_tool_mask

def autotune_batch_and_chunks(
    total_input_rows,
    seq_len,
    hidden_size,
    vocab_size,
    dtype_bytes=16,
    multiplier=None
):
    if multiplier is None:
        final_m = max(4, seq_len // 4096)
    else:
        final_m = multiplier

    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        limit_gb = (free_bytes / (1024**3))*.80
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        # XPU: estimate free memory as total - reserved.
        total_mem = torch.xpu.get_device_properties(0).total_memory
        reserved_mem = torch.xpu.memory_reserved()
        free_bytes = total_mem - reserved_mem
        limit_gb = (free_bytes / (1024**3)) * 0.80
    else:
        # Fallback: assume 8GB available.
        limit_gb = 8.0

    bytes_to_gb = 1024**3

    b_vals = torch.arange(total_input_rows, 0, -1, device='cpu', dtype=torch.float32)

    hidden_gb = (b_vals * seq_len * hidden_size * dtype_bytes) / bytes_to_gb

    base_logits = ((b_vals/total_input_rows) * b_vals * seq_len * vocab_size * dtype_bytes) / bytes_to_gb
    logits_gb = base_logits / final_m

    total_mem_gb = hidden_gb + logits_gb

    valid_mask = total_mem_gb <= limit_gb
    valid_indices = torch.nonzero(valid_mask, as_tuple=False)

    if valid_indices.shape[0] == 0:
        #This means your GPU will OOM
        # Capped at the row count: unsloth's no-grad pass divides rows by this without max(1, ...).
        return max(1, min(4, total_input_rows)), final_m

    best_idx = valid_indices[0].item()
    final_b = int(b_vals[best_idx].item())

    return final_b, final_m

RL_REPLACEMENTS["grpo_autotune_batch_and_chunks"] = autotune_batch_and_chunks


def grpo_update_SamplingParams(SamplingParams, generation_kwargs, vllm_sampling_params = None):
    good_sampling_params_keys = inspect.signature(SamplingParams).parameters.keys()

    new_generation_kwargs = {}
    for key in generation_kwargs.keys():
        if key in good_sampling_params_keys:
            new_generation_kwargs[key] = generation_kwargs[key]
    generation_kwargs = new_generation_kwargs

    if vllm_sampling_params is not None:
        for key in good_sampling_params_keys:
            if hasattr(vllm_sampling_params, key):
                overwrited_key = getattr(vllm_sampling_params, key)
                if overwrited_key is not None and (type(overwrited_key) in (list, tuple,) and len(overwrited_key) != 0):
                    generation_kwargs[key] = overwrited_key
    return generation_kwargs
pass
RL_REPLACEMENTS["grpo_update_SamplingParams"] = grpo_update_SamplingParams


def sanitize_logprob(logprob):
    """Local port of trl.scripts.vllm_serve.sanitize_logprob.
    Filters NaN logprobs from vLLM outputs."""
    value = logprob.logprob
    if math.isnan(value):
        logging.getLogger(__name__).warning(
            f"Generated NaN logprob, token logprob '{logprob}' will be ignored"
        )
        return None
    return value

RL_REPLACEMENTS["sanitize_logprob"] = sanitize_logprob
# Custom compiled GRPO loss - creates 3 Triton kernels
def grpo_compute_loss(
    ref,
    new,
    old,
    sampling_per_token_logps,
    input_ids,
    mask,
    beta,
    advantages,
    **kwargs
):
    # All Unsloth Zoo code licensed under AGPL3
    # Optional argument defaults.
    loss_type = kwargs.get("loss_type", "grpo")
    epsilon_low = kwargs.get("epsilon_low", 0.2)
    epsilon_high = kwargs.get("epsilon_high", 0.2)
    max_completion_length = kwargs.get("max_completion_length", 8192)
    delta = kwargs.get("delta", None)
    importance_sampling_level = kwargs.get("importance_sampling_level", "token")
    num_items_in_batch = kwargs.get("num_items_in_batch", None)
    current_gradient_accumulation_steps = kwargs.get("current_gradient_accumulation_steps", 1)
    steps_per_generation = kwargs.get("steps_per_generation", None)
    num_processes = kwargs.get("num_processes", 1)
    use_vllm = kwargs.get("use_vllm", False)
    # The off-policy mask uses vLLM sampling logprobs whenever the batch supplies them (matching TRL);
    # the vLLM importance-sampling ratio is applied to the loss only when this flag is on.
    vllm_importance_sampling_correction = kwargs.get("vllm_importance_sampling_correction", False)
    vllm_importance_sampling_mode = kwargs.get("vllm_importance_sampling_mode", "sequence_mask")
    vllm_importance_sampling_cap = kwargs.get("vllm_importance_sampling_cap", 2.0)
    vllm_importance_sampling_clip_min = kwargs.get("vllm_importance_sampling_clip_min", None)
    vllm_importance_sampling_clip_max = kwargs.get("vllm_importance_sampling_clip_max", 3.0)
    get_sapo_token_loss = kwargs.get("get_sapo_token_loss", None)
    sapo_temperature_pos = kwargs.get("sapo_temperature_pos", 1.0)
    sapo_temperature_neg = kwargs.get("sapo_temperature_neg", 1.05)
    get_gamma_weights = kwargs.get("get_gamma_weights", None)
    vespo_k_pos = kwargs.get("vespo_k_pos", 2.0)
    vespo_lambda_pos = kwargs.get("vespo_lambda_pos", 3.0)
    vespo_k_neg = kwargs.get("vespo_k_neg", 3.0)
    vespo_lambda_neg = kwargs.get("vespo_lambda_neg", 2.0)
    get_off_policy_mask = kwargs.get("get_off_policy_mask", None)
    off_policy_mask_threshold  = kwargs.get("off_policy_mask_threshold", None)
    # TRL top_entropy_quantile: the policy term (not the KL) on high-entropy tokens only.
    entropy_mask = kwargs.get("entropy_mask", None)
    # Parsed (option, lower_log, upper) tuples from grpo_parse_rollout_rs, or None.
    rollout_rs = kwargs.get("rollout_rs", None)
    # Per-completion weights from unsloth's generation step (ScaleRL zero-variance exclusion /
    # prompt-level aggregation), normalised over the whole generation batch; replaces the loss_type normaliser.
    sample_weight = kwargs.get("sample_weight", None)
    # Only direct callers see this fallback; the trainer always forwards an explicit value.
    use_bias_correction_kl = kwargs.get("use_bias_correction_kl", False)
    input_ids = input_ids.unsqueeze(-1)

    importance_sampling_ratio = None
    # Bool (B, T) keep mask from rollout_rs; None when rejection is off.
    rs_keep = None

    # exp(new - old) and exp(ref - new) below are taken before `mask` is applied. A sequence-packed
    # logp path leaves the masked (prompt/pad) columns at 0 while a padded one fills them with a real
    # logp, so when new and old/ref disagree there those ratios can overflow to inf and inf * 0 (the
    # masked-out loss) becomes nan. Force new/old/ref to share 0 on the masked columns so both ratios
    # are exp(0) = 1 there; every loss term below multiplies by `mask`, so this changes nothing.
    if mask is not None:
        _keep = mask.to(torch.bool)
        new = torch.where(_keep, new, 0.0)
        if old is not None: old = torch.where(_keep, old, 0.0)
        if ref is not None: ref = torch.where(_keep, ref, 0.0)

    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)

    if off_policy_mask_threshold is not None:
        # DeepSeek-V3.2 off-policy mask. The mismatch logprobs are sampling_per_token_logps (vLLM
        # sampling logprobs) if present, else old, else new.detach() when both are absent
        # (num_iterations == 1 with no vLLM). This mirrors TRL, which defaults old_per_token_logps to
        # per_token_logps.detach() so get_off_policy_mask never receives None (it computes
        # mismatch - per_token_logps.detach(), so new.detach() yields a zero-KL keep-all mask). The
        # callable is a signature-stable adapter installed in grpo_accumulated_loss, so this stays
        # fixed across TRL versions with no signature introspection inside this compiled function.
        off_policy_mask = get_off_policy_mask(
            advantages=advantages,
            per_token_logps=new,
            sampling_per_token_logps=sampling_per_token_logps if sampling_per_token_logps is not None else (old if old is not None else new.detach()),
            mask=mask,
            off_policy_threshold=off_policy_mask_threshold,
        )

    with torch.no_grad():
        if use_vllm and sampling_per_token_logps is not None and vllm_importance_sampling_correction:
            # Filter out extra leading prompt tokens after left-padding input_ids.
            # Match TRL: aggregate log-ratios then exp (product), not sum of exp ratios.
            importance_sampling_ratio = (old - sampling_per_token_logps) * mask
            # Unscored vLLM tokens arrive as nan and nan * 0 survives the mask: ratio 1, as TRL does.
            importance_sampling_ratio = torch.nan_to_num(importance_sampling_ratio, nan = 0.0)

            if vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]:
                importance_sampling_ratio = importance_sampling_ratio.sum(dim=-1, keepdim=True)
            elif vllm_importance_sampling_mode in ["geometric_mask", "geometric_truncate"]:
                # Geometric mean of the token ratios (verl seq_mean_k1): the product blows up on long rows.
                importance_sampling_ratio = importance_sampling_ratio.sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

            importance_sampling_ratio = torch.exp(importance_sampling_ratio)

            if vllm_importance_sampling_mode in ["token_truncate", "sequence_truncate", "geometric_truncate"]:
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, 
                    min=vllm_importance_sampling_clip_min,
                    max=vllm_importance_sampling_clip_max
                )
            elif vllm_importance_sampling_mode in ["token_mask", "sequence_mask", "geometric_mask"]:
                min_val = (
                    vllm_importance_sampling_clip_min
                    if vllm_importance_sampling_clip_min is not None
                    else -math.inf
                )

                max_val = (
                    vllm_importance_sampling_clip_max
                    if vllm_importance_sampling_clip_max is not None
                    else math.inf
                )

                invalid_mis_mask = (importance_sampling_ratio < min_val) | (
                        importance_sampling_ratio > max_val
                )

                importance_sampling_ratio = importance_sampling_ratio.masked_fill(
                        invalid_mis_mask, value=0.0
                )
            else:
                raise ValueError(
                        f"Unknown vLLM importance sampling mode: {vllm_importance_sampling_mode}. Possible values are 'token_truncate', 'token_mask', 'sequence_truncate', 'sequence_mask', 'geometric_truncate' and 'geometric_mask'."
                )

        # verl compute_rollout_rejection_mask: every option must pass. verl then replaces
        # response_mask with response_mask * keep, so a rejected token leaves the policy term
        # and the KL loss (both selected out below). It stays in the loss normaliser like
        # TRL's *_mask modes, since num_items_in_batch is counted before the loss.
        # Log ratio is pi_old / pi_rollout; with rollout logps as pi_old (verl bypass mode) it is
        # pi_theta / pi_rollout, verl compute_policy_loss_bypass_mode passing old_log_prob=log_prob.
        rs_log_ratio = None
        if rollout_rs is not None:
            if use_vllm and sampling_per_token_logps is not None and vllm_importance_sampling_correction:
                rs_log_ratio = old - sampling_per_token_logps
            elif kwargs.get("rollout_logps_as_old", False) and old is not None:
                rs_log_ratio = new.detach() - old
        if rs_log_ratio is not None:
            rs_log_ratio = torch.clamp(torch.nan_to_num(rs_log_ratio * mask, nan = 0.0), min = -20.0, max = 20.0)
            rs_mask = mask.to(rs_log_ratio.dtype)
            rs_count = rs_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            rs_keep = torch.ones_like(rs_log_ratio)
            for rs_option, rs_lower, rs_upper in rollout_rs:
                if rs_option.endswith("k1"):
                    rs_stat = -rs_log_ratio
                elif rs_option.endswith("k2"):
                    rs_stat = 0.5 * rs_log_ratio * rs_log_ratio
                else:
                    rs_stat = torch.expm1(rs_log_ratio) - rs_log_ratio
                rs_stat = rs_stat * rs_mask
                if rs_option.startswith("seq_sum"):
                    rs_stat = rs_stat.sum(dim=-1, keepdim=True)
                elif rs_option.startswith("seq_mean"):
                    rs_stat = rs_stat.sum(dim=-1, keepdim=True) / rs_count
                elif rs_option.startswith("seq_max"):
                    # k2, k3 >= 0 and masked entries are 0, so this is the masked max (0 on an empty row).
                    rs_stat = rs_stat.amax(dim=-1, keepdim=True)
                if rs_lower is not None:
                    rs_pass = (rs_stat >= rs_lower) & (rs_stat <= rs_upper)
                else:
                    rs_pass = rs_stat <= rs_upper
                rs_keep = rs_keep * rs_pass.to(rs_keep.dtype)
            rs_keep = rs_keep > 0
            if importance_sampling_ratio is not None:
                # Select, not multiply: an unclipped ratio can overflow to inf and inf * 0 is nan.
                importance_sampling_ratio = torch.where(rs_keep, importance_sampling_ratio, 0.0)
    pass

    # Must detach when old is None: exp(new - new.detach()) == 1 but keeps grads correct.
    if old is not None:
        log_ratio = new - old
    else:
        log_ratio = new - new.detach()
    if kwargs.get("rollout_logps_as_old", False):
        # Rollout gap can pass 88 nats (float32 exp overflow, 0 * inf backward); verl
        # compute_policy_loss_vanilla clamps the log ratio to [-20, 20] too.
        log_ratio = torch.clamp(log_ratio, min = -20.0, max = 20.0)

    if importance_sampling_level == "token":
        log_importance_weights = log_ratio
    elif importance_sampling_level == "sequence":
        log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        log_importance_weights = log_importance_weights.unsqueeze(-1)
    else:
        raise ValueError(
            f"Unknown importance sampling level: {importance_sampling_level}. Possible values are 'token' "
            "and 'sequence'."
        )

    coef_1 =  torch.exp(log_importance_weights)

    # Reverse KL: low-variance low-bias estimator as used in the GRPO paper.
    if beta != 0.0:
        # Clamped like verl "low_var_kl" / SkyRL "k3" (TRL does not): log-ratio to [-20, 20] so expm1
        # cannot overflow into a nan gradient, then k3 to [-10, 10], before bias correction and beta.
        kl_log_ratio = torch.clamp(ref - new, min = -20.0, max = 20.0)
        kl_i = torch.clamp(torch.expm1(kl_log_ratio) - kl_log_ratio, min = -10.0, max = 10.0)
        # TRL order: pre-clamp non-detached coef_1, before the loss_type dispatch.
        if use_bias_correction_kl:
            kl_i = kl_i * coef_1
    else:
        # Zeros with the correct shape.
        if importance_sampling_level == "sequence":
            kl_i = new.new_zeros(new.size(0), 1)
        else:
            kl_i = torch.zeros_like(new)

    if loss_type == "cispo":
        clamped_ratios = torch.clamp(coef_1, max=epsilon_high).detach()
        loss_i = -clamped_ratios * advantages * new
    elif loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)

        if delta is not None:
            loss_1 = torch.clamp(coef_1, max=delta) * advantages
        else:
            loss_1 = coef_1 * advantages
        pass
        loss_2 = coef_2 * advantages
        loss_i = -torch.min(loss_1, loss_2)
    elif loss_type == "sapo":
        temperatures = torch.where(advantages > 0, sapo_temperature_pos, sapo_temperature_neg)
        soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
        loss_i = -soft_coef_1 * advantages
    elif loss_type == "vespo":
        if get_gamma_weights is None:
            raise Exception("vespo is only available in TRL 0.26.0+")
        phi_seq = get_gamma_weights(
            advantages=advantages,
            log_ratio_per_token=log_ratio,
            mask=mask,
            importance_sampling_ratio=importance_sampling_ratio,
            k_pos=vespo_k_pos,
            lambda_pos=vespo_lambda_pos,
            k_neg=vespo_k_neg,
            lambda_neg=vespo_lambda_neg,
        )
        loss_i = -phi_seq * advantages * new
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    if off_policy_mask_threshold is not None:
        loss_i = loss_i * off_policy_mask

    if entropy_mask is not None:
        loss_i = loss_i * entropy_mask

    if use_vllm and sampling_per_token_logps is not None and vllm_importance_sampling_correction:
        # vespo applies the IS ratio inside get_gamma_weights, so skip it here.
        if loss_type != "vespo":
            loss_i = loss_i * importance_sampling_ratio
        # delta for the metric.
        with torch.no_grad():
            delta = torch.abs(old - sampling_per_token_logps)
            delta = delta * mask
            # Zeroed rather than filtered like TRL, to keep the returned shape.
            delta = torch.nan_to_num(delta, nan = 0.0, posinf = math.inf)
            flat_is_ratio = importance_sampling_ratio * mask
    else:
        delta = torch.tensor([]).detach()
        flat_is_ratio = torch.tensor([]).detach()
    if beta != 0.0:
        loss_i = loss_i + beta * kl_i
    if rs_keep is not None:
        # Rejected tokens drop out of every loss term, KL included (verl rollout_rs). A select, so
        # an inf or nan there neither reaches the loss nor turns into a nan gradient.
        loss_i = torch.where(rs_keep, loss_i, 0.0)

    mask = mask.to(torch.float32)
    n_mask_per_reward = mask.sum(1)

    # https://github.com/huggingface/trl/blob/e8b8499f1f8d76838155b515e414ee98f757d6d5/trl/trainer/grpo_trainer.py#L1624
    if sample_weight is not None:
        # Same process / accumulation rescale as the dapo branch, whose weights are 1 / num_items_in_batch.
        normalizer = 1.0 / num_processes
        if steps_per_generation:
            normalizer = normalizer * current_gradient_accumulation_steps / steps_per_generation
        loss = (loss_i * mask * sample_weight.to(loss_i.dtype).unsqueeze(-1)).sum() / normalizer
    elif loss_type in ["grpo", "sapo"]:
        loss = ((loss_i * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
        loss = loss / current_gradient_accumulation_steps
    elif loss_type == "bnpo" and num_items_in_batch is None:
        # TRL < 0.22 passes no global token count: per micro-batch, so the loss depends on how the batch is split.
        loss = (loss_i * mask).sum() / mask.sum().clamp(min=1.0)
        loss = loss / current_gradient_accumulation_steps
    elif loss_type == "dr_grpo":
        loss = (loss_i * mask).sum() / (loss_i.size(0) * max_completion_length)
        loss = loss / current_gradient_accumulation_steps
    elif loss_type in ["bnpo", "cispo", "dapo", "vespo"]:
        # bnpo too: a per micro-batch token mean changes with the GPU / accumulation split, the global count does not.
        # Floor at 1 like TRL: a fully masked batch (mask_truncated_completions) is 0/0 = nan otherwise.
        if torch.is_tensor(num_items_in_batch):
            normalizer = num_items_in_batch.clamp(min = 1.0) / num_processes
        else:
            normalizer = max(float(num_items_in_batch), 1.0) / num_processes
        # num_items_in_batch spans the whole generation batch; rescale to one accumulation window like TRL.
        if steps_per_generation:
            normalizer = normalizer * current_gradient_accumulation_steps / steps_per_generation
        loss = (loss_i * mask).sum() / normalizer
    elif loss_type == "luspo":
        # loss_i is (B, T) unless sequence level with beta 0, so mask elementwise (TRL >= 1.10).
        loss = (loss_i * mask).sum(-1).mean()
        normalizer = current_gradient_accumulation_steps
        loss = loss / normalizer
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Folded metrics.
    def masked_batch_mean(x):
        with torch.inference_mode():
            completion_length = n_mask_per_reward.mean()
            # Sequence level only if x is narrower than mask: one-token completions are (B, 1) too.
            if x.shape[1] == 1 and mask.shape[1] != 1:
                return completion_length, x.mean()
            else:
                # Rows with no tokens (mask_truncated_completions) are left out, as in TRL.
                mean_kl_per_reward = (x * mask).sum(1) / n_mask_per_reward.clamp(min = 1.0)
                kept_rows = (n_mask_per_reward > 0).sum()
                mean_kl = torch.where(
                    kept_rows == n_mask_per_reward.numel(),
                    mean_kl_per_reward.mean(),
                    mean_kl_per_reward.sum() / kept_rows.clamp(min = 1),
                )
                return completion_length, mean_kl
    # Rejected tokens add nothing to the reported KL either; the denominator keeps every completion
    # token, like the loss normaliser.
    kl_metric = kl_i if rs_keep is None or beta == 0.0 else torch.where(rs_keep, kl_i, 0.0)
    completion_length, mean_kl = masked_batch_mean(kl_metric)
    return loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1, mask
pass
RL_REPLACEMENTS["grpo_compute_loss"]      = grpo_compute_loss
# Same eager fallback as every other fullgraph region: a bare decorator leaves
# cache exhaustion fatal under fullgraph.
RL_REPLACEMENTS["grpo_compute_loss_slow"] = \
    f"from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback\n"\
    f"@torch_compile_with_fallback(dynamic = True, fullgraph = True, options = torch_compile_options)\n"\
    f"{inspect.getsource(grpo_compute_loss)}"
RL_REPLACEMENTS["grpo_compute_loss_slow"] = \
    RL_REPLACEMENTS["grpo_compute_loss_slow"].replace(
        "def grpo_compute_loss",
        "def grpo_compute_loss_slow",
)

# Unsloth's memory efficient GRPO implementation
class UnslothEfficientGRPO(torch.autograd.Function):
    # All Unsloth Zoo code licensed under AGPL3
    @staticmethod
    def forward(ctx, _new_logps, _old_logps, _ref_logps, _sampling_per_token_logps, lm_head, _input_ids, _mask, _advantages, beta, scaler = None, n_chunks = 1, extra_kwargs=None, upstream_scale = None):
        if extra_kwargs is None:
            extra_kwargs = {}
        def compute_loss(new_logps, old_logps, ref_logps, sampling_per_token_logps, input_ids, mask, advantages, scaling):
            loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1, _mask  = grpo_compute_loss(
                ref_logps,
                new_logps,
                old_logps,
                sampling_per_token_logps,
                input_ids,
                mask,
                beta,
                advantages,
                **extra_kwargs,
            )

            # Scale for mixed precision; return loss.detach() or autograd uses 2x VRAM.
            scaled_loss = loss * scaling
            return scaled_loss, (loss.detach(), completion_length, mean_kl, delta, flat_is_ratio, coef_1)
        pass

        device =_new_logps.device
        grad_inputs = torch.empty_like(_new_logps)
        accumulated_loss              = torch.zeros(1, device = device)[0]
        accumulated_completion_length = torch.zeros(1, device = device)[0]
        accumulated_mean_kl           = torch.zeros(1, device = device)[0]
        accumulated_delta             = []
        accumulated_flat_is_ratio     = []
        accumulated_coef_1            = []

        def accumulate_chunk(
            new_logps_j,
            old_logps_j,
            ref_logps_j,
            sampling_per_token_logps_j,
            input_ids_j,
            mask_j,
            advantages_j,
            scaling,
            grad_inputs_j,
        ):
            (chunk_grad_input,), (chunk_loss, (unscaled_loss, chunk_completion_length, chunk_mean_kl, chunk_delta, chunk_flat_is_ratio, chunk_coef_1)) = torch.func.grad_and_value(
                compute_loss,
                argnums = (0,),
                has_aux = True,
            )(new_logps_j, old_logps_j, ref_logps_j, sampling_per_token_logps_j, input_ids_j, mask_j, advantages_j, scaling)
            accumulated_loss             .add_(unscaled_loss)
            accumulated_completion_length.add_(chunk_completion_length)
            accumulated_mean_kl          .add_(chunk_mean_kl)
            accumulated_delta            .append(chunk_delta)
            accumulated_flat_is_ratio    .append(chunk_flat_is_ratio)
            accumulated_coef_1           .append(chunk_coef_1)
            grad_inputs_j[:] = chunk_grad_input
        pass

        from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback
        accumulate_chunk = torch_compile_with_fallback(
            fullgraph = True,
            # [TODO] Dynamic marking causes torch.compile errors if sequence length is long
            dynamic = True,
            options = torch_compile_options,
        )(accumulate_chunk)

        grad_inputs_chunks = torch.chunk(grad_inputs,        chunks = n_chunks, dim = 0)
        new_logps  = torch.chunk(_new_logps, chunks = n_chunks, dim = 0)
        if _old_logps is not None:
            old_logps  = torch.chunk(_old_logps, chunks = n_chunks, dim = 0)
        else:
            old_logps = [None] * n_chunks
        if _ref_logps is not None:
            ref_logps  = torch.chunk(_ref_logps, chunks = n_chunks, dim = 0)
        else:
            ref_logps = [None] * n_chunks
        if _sampling_per_token_logps is not None:
            sampling_per_token_logps  = torch.chunk(_sampling_per_token_logps, chunks = n_chunks, dim = 0)
        else:
            sampling_per_token_logps = [None] * n_chunks
        input_ids          = torch.chunk(_input_ids,         chunks = n_chunks, dim = 0)
        mask               = torch.chunk(_mask,              chunks = n_chunks, dim = 0)
        advantages         = torch.chunk(_advantages,        chunks = n_chunks, dim = 0)

        # Mixed precision scaling if present.
        scaling = scaler.get_scale() if scaler is not None else 1.0

        for (grad_inputs_j, new_logps_j, old_logps_j, ref_logps_j, sampling_per_token_logps_j, input_ids_j, mask_j, advantages_j, ) in \
            zip(grad_inputs_chunks, new_logps, old_logps, ref_logps, sampling_per_token_logps, input_ids, mask, advantages):

            # [TODO] Dynamic marking causes torch.compile errors if sequence length is long

            # mark_dynamic(new_hidden_states_j)
            # mark_dynamic(ref_hidden_states_j)
            # if old_hidden_states_j is not None:
            #     mark_dynamic(old_hidden_states_j)
            # mark_dynamic(input_ids_j)
            # mark_dynamic(mask_j)
            accumulate_chunk(
                new_logps_j,
                old_logps_j,
                ref_logps_j,
                sampling_per_token_logps_j,
                input_ids_j,
                mask_j,
                advantages_j,
                scaling,
                grad_inputs_j,
            )
        pass

        grad_inputs                  .div_(n_chunks)
        accumulated_loss             .div_(n_chunks)
        accumulated_completion_length.div_(n_chunks)
        accumulated_mean_kl          .div_(n_chunks)

        if _sampling_per_token_logps is not None:
            accumulated_delta = torch.cat(accumulated_delta, dim=0)
            accumulated_flat_is_ratio = torch.cat(accumulated_flat_is_ratio, dim=0)
        else:
            accumulated_delta = None
            accumulated_flat_is_ratio = None
        accumulated_coef_1  = torch.cat(accumulated_coef_1, dim=0)
        ctx.save_for_backward(grad_inputs)
        ctx.upstream_scale = upstream_scale
        return (
            accumulated_loss,
            accumulated_completion_length,
            accumulated_mean_kl,
            accumulated_delta,
            accumulated_flat_is_ratio,
            accumulated_coef_1
        )
    pass

    @staticmethod
    def backward(ctx, grad_output, dcompletion_length, dmean_kl, ddelta, ddflat_is_ratio, dcoef_1):
        (grad_input,) = ctx.saved_tensors
        if ctx.upstream_scale is not None:
            grad_input = grad_input * (grad_output * ctx.upstream_scale)
        return (grad_input, None, None, None, None, None, None, None, None, None, None, None, None)
    pass
pass
RL_REPLACEMENTS["UnslothEfficientGRPO"] = UnslothEfficientGRPO


class UnslothGRPOEntropyBonus(torch.autograd.Function):
    """sum(entropies * weights), with the gradient scaled exactly as UnslothEfficientGRPO scales
    the policy term's: by the mixed precision scale, ignoring grad_output unless DeepSpeed's
    upstream_scale is set. Plain autograd would see grad_output instead, which the policy term
    never does, and the entropy coefficient would then mean something else than in TRL."""
    @staticmethod
    def forward(ctx, entropies, weights, scaling, upstream_scale = None):
        ctx.save_for_backward(weights)
        ctx.scaling = scaling
        ctx.upstream_scale = upstream_scale
        return (entropies.to(torch.float32) * weights).sum()

    @staticmethod
    def backward(ctx, grad_output):
        (weights,) = ctx.saved_tensors
        grad = weights * ctx.scaling
        if ctx.upstream_scale is not None:
            grad = grad * (grad_output * ctx.upstream_scale)
        return grad, None, None, None
pass
RL_REPLACEMENTS["UnslothGRPOEntropyBonus"] = UnslothGRPOEntropyBonus


def grpo_entropy_bonus_weights(completion_mask, entropy_mask, entropy_coef, loss_kwargs, trl_version = None, training = True):
    """Per-token weights w with sum(w * entropies) == -entropy_coef * TRL's entropy_loss.

    TRL >= 1.10: the mean entropy over the tokens the policy term acts on, divided by the
    accumulation steps (1 in eval) for every loss_type. TRL 1.8 and 1.9 instead divided cispo /
    dapo / vespo by the policy term's own global token normalizer, without the mean.
    """
    weights = completion_mask.to(torch.float32)
    if entropy_mask is not None:
        weights = weights * entropy_mask
    sample_weight = loss_kwargs.get("sample_weight", None)
    if sample_weight is not None:
        # Zero-variance exclusion drops a completion from the whole loss, the entropy bonus included.
        weights = weights * (sample_weight.to(weights.device) > 0).to(weights.dtype).unsqueeze(-1)
    if trl_version is None:
        try:
            import trl
            trl_version = trl.__version__
        except Exception:
            trl_version = "0"
    from packaging.version import Version
    loss_type = loss_kwargs.get("loss_type", "grpo")
    if Version(trl_version) < Version("1.10.0") and loss_type in ("cispo", "dapo", "vespo"):
        if sample_weight is not None:
            # The policy term is normalised by sample_weight (kept tokens only, grpo_compute_loss),
            # and TRL 1.8 / 1.9 divide the bonus by the policy's own normaliser: the same here, so
            # excluded rows do not dilute it. Without exclusion this equals the branch below.
            normalizer = 1.0 / loss_kwargs.get("num_processes", 1)
            steps_per_generation = loss_kwargs.get("steps_per_generation", None)
            if steps_per_generation:
                normalizer = normalizer * loss_kwargs.get("current_gradient_accumulation_steps", 1) / steps_per_generation
            row_weight = sample_weight.to(device = weights.device, dtype = weights.dtype).unsqueeze(-1)
            return weights * row_weight * (-entropy_coef / normalizer)
        # TRL 1.8 / 1.9 used num_items_in_batch / num_processes as is, before 1.10 rescaled it to
        # one accumulation window (grpo_compute_loss always rescales the policy term).
        num_items_in_batch = loss_kwargs.get("num_items_in_batch", None)
        if torch.is_tensor(num_items_in_batch):
            normalizer = num_items_in_batch.clamp(min = 1.0) / loss_kwargs.get("num_processes", 1)
        else:
            normalizer = max(float(num_items_in_batch), 1.0) / loss_kwargs.get("num_processes", 1)
        return weights * (-entropy_coef / normalizer)
    # Unsloth before #11337 passes the training window's accumulation steps into eval too.
    accumulation_steps = loss_kwargs.get("current_gradient_accumulation_steps", 1) if training else 1
    return weights * (-entropy_coef / weights.sum().clamp(min = 1.0) / accumulation_steps)
pass
RL_REPLACEMENTS["grpo_entropy_bonus_weights"] = grpo_entropy_bonus_weights


_HIGH_ENTROPY_MASK_PADS = {}
def _high_entropy_mask_pads(method):
    # TRL >= 0.24 pads every rank to one length before the gather; TRL 0.22 / 0.23 gather the
    # raw valid tokens and return early on a rank that has none
    func = getattr(method, "__func__", method)
    pads = _HIGH_ENTROPY_MASK_PADS.get(func, None)
    if pads is None:
        try:
            source = inspect.getsource(func)
            pads = ("pad_across_processes" in source) or ("accelerator.gather" not in source)
        except Exception:
            pads = True
        _HIGH_ENTROPY_MASK_PADS[func] = pads
    return pads
pass


def grpo_high_entropy_mask(trainer, entropies, mask, threshold, sample_weight = None):
    """trainer.get_high_entropy_mask, except on a multi-process run whose method gathers without
    padding (TRL < 0.24): a rank with no valid tokens returns before the gather the other ranks
    enter, so training hangs, and uneven token counts break the gather. Those runs take TRL >= 0.24's
    padded gather, which sees the same quantile population; a single process is unchanged.
    sample_weight: rows with weight 0 (zero-variance exclusion) leave the loss, so they are left
    out of the quantile population too."""
    if sample_weight is not None:
        mask = mask * (sample_weight.to(mask.device) > 0).to(mask.dtype).unsqueeze(-1)
    accelerator = getattr(trainer, "accelerator", None)
    if getattr(accelerator, "num_processes", 1) > 1 and not _high_entropy_mask_pads(trainer.get_high_entropy_mask):
        local = entropies[mask.bool()].float()
        # entropies are >= 0, so a negative sentinel cannot collide with a real value
        pad_value = -1e9
        padded = accelerator.pad_across_processes(local, dim = 0, pad_index = pad_value)
        gathered = accelerator.gather(padded)
        gathered = gathered[gathered != pad_value]
        if gathered.numel() == 0:
            return torch.zeros_like(entropies, dtype = torch.bool)
        entropy_threshold = torch.quantile(gathered, threshold)
        entropy_mask = (entropies * mask.float()) >= entropy_threshold
        return entropy_mask & mask.bool()
    return trainer.get_high_entropy_mask(entropies, mask, threshold)
pass


def _warn_unsupported_grpo_options(trainer):
    """Warn once per trainer about TRL GRPOConfig options this path ignores, so setting
    them is not silently dropped. Adaptive entropy control (use_adaptive_entropy) is
    unimplemented, and with it its entropy bonus; top_entropy_quantile is unsupported only on a
    TRL without get_high_entropy_mask. Their TRL defaults are off, so only non-defaults warn.
    top_entropy_quantile, a fixed entropy_coef and use_bias_correction_kl are supported.
    """
    if getattr(trainer, "_unsloth_grpo_unsupported_warned", False):
        return
    args = getattr(trainer, "args", None)

    unsupported = []
    top_entropy_quantile = getattr(args, "top_entropy_quantile", 1.0)
    if top_entropy_quantile is not None and top_entropy_quantile < 1.0 and not hasattr(trainer, "get_high_entropy_mask"):
        unsupported.append(f"top_entropy_quantile={top_entropy_quantile}")
    if getattr(args, "use_adaptive_entropy", False):
        # The adaptive controller's bonus is dropped whole, including its starting entropy_coef.
        entropy_coef = getattr(args, "entropy_coef", 0.0)
        if entropy_coef:
            unsupported.append(f"entropy_coef={entropy_coef}")
        unsupported.append("use_adaptive_entropy=True")
    # These only act on vLLM sampling logprobs, which exist with vllm_importance_sampling_correction.
    if not (getattr(trainer, "use_vllm", False) and getattr(trainer, "vllm_importance_sampling_correction", False)):
        _names = ("UNSLOTH_GRPO_VLLM_IS_MODE", "UNSLOTH_GRPO_VLLM_IS_CLIP", "UNSLOTH_GRPO_ROLLOUT_RS", "UNSLOTH_GRPO_ROLLOUT_METRICS")
        if os.environ.get("UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD", "0") == "1":
            # Rejection and the mismatch metrics run on pi_theta / pi_rollout there (verl bypass mode).
            _names = _names[:2]
        for name in _names:
            if os.environ.get(name, "") not in ("", "0"):
                unsupported.append(f"{name}={os.environ[name]} (needs use_vllm and vllm_importance_sampling_correction=True)")

    if unsupported:
        message = (
            "Unsloth: GRPOConfig option(s) " + ", ".join(unsupported) + " are set but "
            "are not supported by Unsloth's optimized GRPO path and will be ignored "
            "(training proceeds as standard GRPO)."
        )
        try:
            logger.warning(message)
        except Exception:
            import warnings as _warnings
            _warnings.warn(message)
        # Only latch after warning, so a config changed mid-run still gets one warning.
        try:
            trainer._unsloth_grpo_unsupported_warned = True
        except Exception:
            pass
    return
pass
RL_REPLACEMENTS["_warn_unsupported_grpo_options"] = _warn_unsupported_grpo_options


def _grpo_rollout_sampling_params(trainer, keys):
    """Later sources override earlier ones as they do for the SamplingParams: GRPOConfig fields,
    then generation_kwargs, then Unsloth's vllm_sampling_params."""
    args = getattr(trainer, "args", None)
    params = {key: getattr(args, key, None) for key in keys}
    vllm_sampling_params = getattr(args, "vllm_sampling_params", None)
    vllm_overrides = getattr(vllm_sampling_params, "_set_kwargs", None)
    if vllm_overrides is None and vllm_sampling_params is not None:
        # A raw vllm.SamplingParams: take exactly the fields generation applies from it, through the
        # grpo_update_SamplingParams the trainer's generated module calls (unsloth >= #11595 copies
        # the fields that differ from SamplingParams(), older ones use this module's, lists only)
        update = grpo_update_SamplingParams
        for klass in type(trainer).__mro__:
            candidate = getattr(sys.modules.get(klass.__module__, None), "grpo_update_SamplingParams", None)
            if callable(candidate):
                update = candidate
                break
        try:
            vllm_overrides = update(type(vllm_sampling_params), {}, vllm_sampling_params)
        except Exception:
            vllm_overrides = grpo_update_SamplingParams(type(vllm_sampling_params), {}, vllm_sampling_params)
    for overrides in (
        getattr(args, "generation_kwargs", None),
        vllm_overrides,
    ):
        if isinstance(overrides, dict):
            for key in params:
                if overrides.get(key, None) is not None:
                    params[key] = overrides[key]
    return params
pass


# vLLM v1 applies these to the logits before temperature, min_p and top_k / top_p
# (Sampler.apply_logits_processors), so its kept set and logprobs come from logits the trainer never sees.
_PRE_TRUNCATION_NEUTRAL = {
    "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "min_tokens": 0,
}
_PRE_TRUNCATION_OPTIONAL = (
    "logit_bias", "bad_words", "allowed_token_ids", "logits_processors",
    "structured_outputs", "guided_decoding", "vllm_structured_outputs_regex", "vllm_guided_decoding_regex",
)


def grpo_truncated_support_blockers(trainer):
    """Rollout settings under which the recomputed kept set cannot match vLLM's."""
    params = _grpo_rollout_sampling_params(
        trainer, ("temperature",) + tuple(_PRE_TRUNCATION_NEUTRAL) + _PRE_TRUNCATION_OPTIONAL,
    )
    blockers = []
    for key, neutral in _PRE_TRUNCATION_NEUTRAL.items():
        if params[key] is not None and float(params[key]) != neutral:
            blockers.append(f"{key}={params[key]}")
    for key in _PRE_TRUNCATION_OPTIONAL:
        value = params[key]
        if value is not None and not (isinstance(value, (dict, list, tuple, set, str)) and len(value) == 0):
            blockers.append(key)
    # The loss scales by the trainer's temperature; vLLM truncates at the rollout's.
    trainer_temperature = getattr(trainer, "temperature", None)
    if trainer_temperature is None:
        trainer_temperature = getattr(getattr(trainer, "args", None), "temperature", None)
    rollout_temperature = params["temperature"]
    if rollout_temperature is not None and trainer_temperature is not None \
            and float(rollout_temperature) != float(trainer_temperature):
        blockers.append(f"rollout temperature={rollout_temperature} != trainer temperature={trainer_temperature}")
    return blockers
pass


def grpo_sampling_support(trainer):
    """(top_k, top_p, min_p) the rollouts were truncated with, or None for full-vocabulary sampling."""
    params = _grpo_rollout_sampling_params(trainer, ("top_k", "top_p", "min_p"))
    # top_k None / 0 / -1 all mean no top-k in TRL and vLLM.
    top_k = max(int(params["top_k"] or 0), 0)
    top_p = 1.0 if params["top_p"] is None else float(params["top_p"])
    min_p = 0.0 if params["min_p"] is None else float(params["min_p"])
    if top_k == 0 and top_p >= 1.0 and min_p <= 0.0:
        return None
    return top_k, top_p, min_p
pass


def grpo_truncated_support_kwargs(trainer):
    """support_* kwargs for the chunked logprob helpers when UNSLOTH_GRPO_TRUNCATED_SUPPORT=1 and
    the vLLM rollouts were truncated, else {}. With the opt-in off, warns once if the truncated
    vLLM logprobs feed the IS correction or the off-policy mask."""
    if not getattr(trainer, "use_vllm", False):
        return {}
    support = grpo_sampling_support(trainer)
    if support is None:
        return {}
    top_k, top_p, min_p = support
    enabled = os.environ.get("UNSLOTH_GRPO_TRUNCATED_SUPPORT", "0").strip().lower() in ("1", "true", "yes", "on")
    if not enabled:
        args = getattr(trainer, "args", None)
        consumed = (
            getattr(trainer, "vllm_importance_sampling_correction", False)
            or getattr(args, "off_policy_mask_threshold", None) is not None
        )
        if consumed and not getattr(trainer, "_unsloth_truncated_support_warned", False):
            logger.warning(
                f"Unsloth: rollouts sample with top_k={top_k}, top_p={top_p}, min_p={min_p}, so "
                "vLLM's logprobs are renormalised over the kept tokens while the trainer scores the "
                "full vocabulary. The importance sampling ratio / off-policy mask then sees a "
                "systematic gap of log(kept probability mass) per token. Set "
                "UNSLOTH_GRPO_TRUNCATED_SUPPORT=1 to score the trainer over the same kept set."
            )
            trainer._unsloth_truncated_support_warned = True
        return {}
    blockers = grpo_truncated_support_blockers(trainer)
    if blockers:
        if not getattr(trainer, "_unsloth_truncated_support_blocked_warned", False):
            logger.warning(
                "Unsloth: ignoring UNSLOTH_GRPO_TRUNCATED_SUPPORT=1 because the rollouts use "
                f"{', '.join(blockers)}, which vLLM applies before truncating, so the trainer cannot "
                "rebuild the same kept set. Scoring over the full vocabulary instead."
            )
            trainer._unsloth_truncated_support_blocked_warned = True
        return {}
    return {"support_top_k": top_k, "support_top_p": top_p, "support_min_p": min_p}
pass
RL_REPLACEMENTS["grpo_truncated_support_kwargs"] = grpo_truncated_support_kwargs


_n_chunks_deprecation_warned = False

def _warn_deprecated_n_chunks(n_chunks):
    """Warn once per process when unsloth_num_chunks is set to a non-default value.
    The parameter stays because unsloth's generated trainer passes it, but the value
    never reaches UnslothEfficientGRPO.apply, which is always called with 1.
    """
    global _n_chunks_deprecation_warned
    if _n_chunks_deprecation_warned:
        return
    if n_chunks is None or n_chunks == -1 or n_chunks == 1:
        return
    _n_chunks_deprecation_warned = True
    message = (
        "Unsloth: unsloth_num_chunks is deprecated and is ignored; the GRPO loss now "
        "always runs as a single chunk, since memory is managed by "
        "unsloth_grpo_mini_batch and unsloth_logit_chunk_multiplier."
    )
    try:
        logger.warning(message)
    except Exception:
        import warnings as _warnings
        _warnings.warn(message)
    return
pass
RL_REPLACEMENTS["_warn_deprecated_n_chunks"] = _warn_deprecated_n_chunks


# The multimodal keys TRL's GRPO trainer puts in the inputs dict. Both Unsloth logprob
# paths read this one tuple: they must forward the same inputs, or the importance ratio
# compares two different policies. unslothai/unsloth#6960.
GRPO_VISION_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_attention_mask",
    "image_sizes",
    "spatial_shapes",
    "num_tiles",
    # One Gemma 4 field under both TRL names: pixel_position_ids in 1.0.x, renamed in
    # 1.1.0. Only ever one of the two is present.
    "image_position_ids",
    "pixel_position_ids",
    "num_images",
    "token_type_ids",
    "mm_token_type_ids",
    # Qwen2-VL style video: patch rows by video, one grid row (and Qwen2.5-VL time step) per video.
    "pixel_values_videos",
    "video_grid_thw",
    "second_per_grid_ts",
    "num_videos",
)


def grpo_get_vision_inputs(source):
    """Collect the GRPO multimodal inputs out of a kwargs or inputs mapping."""
    if source is None:
        return {}
    get = getattr(source, "get", None)
    if get is None:
        return {}
    return {key: get(key, None) for key in GRPO_VISION_KEYS}
pass
RL_REPLACEMENTS["grpo_get_vision_inputs"] = grpo_get_vision_inputs


# What every released unsloth up to 2026.9.4 forwards on the no-grad side, hard coded in its
# own `_get_per_token_logps_and_entropies` replacement. The conservative answer when the
# installed companion is present but cannot be read: forwarding MORE than this on the gradient
# side alone is what makes the two policies differ, so an unknown companion is assumed to be
# one of those rather than assumed to be current.
GRPO_RELEASED_VISION_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_attention_mask",
    "image_sizes",
    "num_images",
    "token_type_ids",
    "mm_token_type_ids",
)

# The names whose presence in the installed unsloth's no-grad replacement means it reads THIS
# module's key tuple rather than a list of its own. Either one is enough: unslothai/unsloth#11031
# collects the keys in a module level helper and names only the chunker inside the replacement.
# One constant rather than two literals per site, because the integration test has to ask the
# same question the gate asks and a second copy of the answer drifts.
GRPO_SHARED_HELPER_MARKERS = ("grpo_get_vision_inputs", "grpo_vision_chunks")

# Memo for grpo_companion_vision_keys. Set once per process; tests reset it to None.
_GRPO_COMPANION_VISION_KEYS = None


def grpo_companion_vision_keys():
    """Which of GRPO_VISION_KEYS the installed unsloth also forwards on the no-grad side.

    unsloth ships separately from this package, and up to 2026.9.4 its
    ``_get_per_token_logps_and_entropies`` replacement, which computes the old and reference
    logprobs, carries its own hard coded list of seven keys. Forwarding more than that on the
    gradient side alone would let the current policy see image metadata the reference policy
    never saw, so the importance ratio and the KL term would compare two different policies.
    Agreeing on the smaller set is what makes the ratio meaningful, and it is still strictly
    better than before, since the shared chunker no longer drops pixel_values outright.

    Read out of the installed unsloth's source rather than its version, so an unsloth carrying
    the companion change lifts the restriction on its own: either shared name appearing in that
    replacement means the no-grad pass reads this module's tuple. Only sys.modules is consulted:
    at this point unsloth is what is driving the run, and importing it from here is circular.
    Only an answer READ off a patcher is memoized: a module that is not there yet is a call
    that came too early, not an unsloth without the companion.
    """
    global _GRPO_COMPANION_VISION_KEYS
    if _GRPO_COMPANION_VISION_KEYS is not None:
        return _GRPO_COMPANION_VISION_KEYS

    keys = GRPO_VISION_KEYS
    module = sys.modules.get("unsloth.models.rl_replacements")
    patcher = getattr(module, "grpo_trainer__get_per_token_logps_and_entropies", None)
    if patcher is None:
        # Nothing to classify yet, and an absence here is not a fact about the process: this
        # module can be imported before unsloth installs its RL replacements, and the first
        # caller would otherwise memoize "no companion" permanently, so the restriction could
        # never be applied to the run that actually needs it. Treat it like the unreadable
        # case and leave it unmemoized, so a later call gets to look again once the import has
        # happened. The full tuple is the right answer while nothing is known to restrict it.
        return GRPO_VISION_KEYS

    source = None
    unreadable = False
    try:
        source = inspect.getsource(patcher)
    except (OSError, TypeError):
        # Source stripped, frozen, or dynamically wrapped. The companion is THERE -- the
        # patcher exists -- and the only thing missing is the ability to see which keys it
        # forwards. Leaving the full tuple there was the unsafe half of the guess: the
        # gradient pass would forward spatial_shapes, num_tiles and the position ids that
        # every released companion omits, and the importance ratio and KL term would then
        # compare two different policies with nothing to show for it. Assume the released
        # set instead, which is the answer for every unsloth that does not carry the
        # companion change, and do not memoize it: this is a failure to look, not a fact
        # about the process, so a later call gets to look again.
        source = None
        unreadable = True
    shares = source is not None and any(
        marker in source for marker in GRPO_SHARED_HELPER_MARKERS
    )
    if source is not None and not shares:
        named = tuple(
            key for key in GRPO_VISION_KEYS
            if '"%s"' % key in source or "'%s'" % key in source
        )
        # A no-grad pass that names no pixel_values at all is one this reader does not
        # understand; leave the full set rather than silently turning vision off.
        if "pixel_values" in named and len(named) != len(GRPO_VISION_KEYS):
            keys = named
            logger.warning(
                "Unsloth: the installed unsloth computes GRPO reference logprobs without "
                f"{', '.join(k for k in GRPO_VISION_KEYS if k not in named)}. Holding the "
                "gradient pass to the same inputs so both policies match. Upgrade unsloth to "
                "forward every vision kwarg on both paths."
            )
    if unreadable:
        logger.warning(
            "Unsloth: the installed unsloth's GRPO reference-logprob pass cannot be read, so "
            "which vision kwargs it forwards is unknown. Holding the gradient pass to the "
            "keys every released unsloth forwards, so both policies still match."
        )
        return GRPO_RELEASED_VISION_KEYS
    _GRPO_COMPANION_VISION_KEYS = keys
    return keys
pass
RL_REPLACEMENTS["grpo_companion_vision_keys"] = grpo_companion_vision_keys


# Warned once per process, not per step: this is a property of the installed packages.
_GRPO_COMPANION_PIXELS_WARNED = False


def grpo_shared_vision_inputs(source):
    """grpo_get_vision_inputs, restricted to what both logprob passes actually forward.

    Two restrictions, not one. The key list is the first: a companion with its own hard coded
    tuple never sees the keys outside it. The second is a SHAPE, and intersecting names cannot
    express it -- a companion that does not share the chunker slices the pixels itself, and its
    loop drops ``pixel_values`` outright unless ``image_grid_thw`` is there to slice them by
    (``pixel_values_chunks.append(None)`` in its else branch). For a VLM that carries no grid,
    which is Gemma 3, InternVL and LFM2-VL, forwarding pixels on the gradient side alone would
    leave the current policy looking at images the reference policy never saw, and the
    importance ratio and the KL term would compare two different policies however carefully the
    key names were matched.

    That second restriction is NOT applied here, and the layer matters. ``pixel_values`` is a
    sentinel in the caller as well as a model input: ``grpo_accumulated_loss`` reads it to
    decide whether to left-pack the batch (recomputing ``max_left_pad``, repacking
    ``input_ids`` and rebuilding ``completion_mask``) and whether to take the sequence-packing
    path. The companion chooses those on its own ``pixel_values``, which is a real tensor --
    it appends None per chunk INSIDE its loop, after the branch is already chosen -- so
    blanking the key here would stop the two passes disagreeing about pixels and start them
    disagreeing about how the sequences are arranged, which is the worse comparison of the
    two. The suppression therefore happens one layer down, in ``grpo_vision_chunks``, on the
    chunk that is actually forwarded. See ``grpo_companion_drops_pixels``.
    """
    keys = grpo_companion_vision_keys()
    if len(keys) == len(GRPO_VISION_KEYS):
        return grpo_get_vision_inputs(source)
    return {key: value for key, value in grpo_get_vision_inputs(source).items() if key in keys}
pass
RL_REPLACEMENTS["grpo_shared_vision_inputs"] = grpo_shared_vision_inputs


def grpo_companion_drops_pixels(vision):
    """Whether the installed companion's no-grad loop forwards no pixels for THIS batch.

    Its else branch is ``pixel_values_chunks.append(None)``: with no ``image_grid_thw`` to
    slice the pixels by, that loop has nothing to forward, so for a VLM that carries no grid
    -- Gemma 3, InternVL, LFM2-VL -- the reference policy sees text where the gradient pass
    would see images. Matching it is what keeps the importance ratio and the KL term a
    comparison of one policy with itself.

    False whenever the companion shares this module's chunker, because then both passes slice
    with the same code and nothing needs to be given up.
    """
    if vision.get("pixel_values", None) is None:
        return False
    if vision.get("image_grid_thw", None) is not None:
        return False
    return len(grpo_companion_vision_keys()) != len(GRPO_VISION_KEYS)
pass
RL_REPLACEMENTS["grpo_companion_drops_pixels"] = grpo_companion_drops_pixels


def grpo_vision_chunks(vision, total_samples, batch_size):
    """Slice the GRPO multimodal inputs into per-chunk forward kwargs, one dict per chunk.

    One implementation for both logprob paths, so they cannot index the same tensors
    differently. The axis per family mirrors TRL's own ``_get_per_token_logps_and_entropies``:

    * ``image_grid_thw`` (Qwen2-VL): ``pixel_values`` by patch row, the grid by image.
    * ``image_position_ids`` (Gemma 4): both by image.
    * ``spatial_shapes`` (LFM2-VL): ``pixel_values``, ``pixel_attention_mask`` and
      ``spatial_shapes`` by tile, with ``num_tiles`` giving the tiles per sample.
    * ``num_tiles`` alone (InternVL): ``pixel_values`` by tile.
    * anything else: by image when there is one row per image, else by sample.

    An unrecognised model still gets its ``pixel_values``, as stock TRL does; dropping
    them silently recomputes the reference logprobs from the text alone.
    """
    def _as_int_list(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().reshape(-1).tolist()
        try:
            return [int(n) for n in value]
        except TypeError:
            return None

    def _first_dim_len(value):
        if value is None:
            return None
        if hasattr(value, "shape"):
            return value.shape[0]
        try:
            return len(value)
        except TypeError:
            return None

    pixel_values = vision.get("pixel_values", None)
    image_grid_thw = vision.get("image_grid_thw", None)
    pixel_attention_mask = vision.get("pixel_attention_mask", None)
    image_sizes = vision.get("image_sizes", None)
    spatial_shapes = vision.get("spatial_shapes", None)
    # One local for both spellings, but it must leave under the name it arrived with:
    # the model kwarg is named the same as the inputs key.
    image_position_ids = vision.get("image_position_ids", None)
    position_ids_key = "image_position_ids"
    if image_position_ids is None:
        image_position_ids = vision.get("pixel_position_ids", None)
        position_ids_key = "pixel_position_ids"
    token_type_ids = vision.get("token_type_ids", None)
    mm_token_type_ids = vision.get("mm_token_type_ids", None)
    num_images = _as_int_list(vision.get("num_images", None))
    num_tiles = _as_int_list(vision.get("num_tiles", None))

    # A count list is only a per-sample cumulative index when it has one entry per row.
    if num_images is not None and len(num_images) != total_samples:
        num_images = None
    if num_tiles is not None and len(num_tiles) != total_samples:
        num_tiles = None

    cum_imgs = None if num_images is None else torch.tensor([0] + num_images).cumsum(0)
    cum_tiles = None if num_tiles is None else torch.tensor([0] + num_tiles).cumsum(0)

    cum_rows = None
    if image_grid_thw is not None and pixel_values is not None and num_images is not None:
        rows_per_image = image_grid_thw.prod(dim = -1)
        rows_per_sample = torch.split(rows_per_image, num_images)
        rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
        # .item() in the loop below, so keep it on CPU or every chunk pays a sync.
        cum_rows = torch.cat(
            [
                torch.tensor([0], device = rows_per_sample.device),
                rows_per_sample.cumsum(0),
            ]
        ).cpu()

    total_images = None if num_images is None else sum(num_images)

    pixel_values_videos = vision.get("pixel_values_videos", None)
    video_grid_thw = vision.get("video_grid_thw", None)
    second_per_grid_ts = vision.get("second_per_grid_ts", None)
    num_videos = _as_int_list(vision.get("num_videos", None))
    if num_videos is not None and len(num_videos) != total_samples:
        num_videos = None
    cum_videos = cum_video_rows = None
    if pixel_values_videos is not None and video_grid_thw is not None:
        if num_videos is None or sum(num_videos) != video_grid_thw.shape[0]:
            raise ValueError(
                "Unsloth: GRPO cannot tell which sample each video belongs to: "
                f"{video_grid_thw.shape[0]} videos over {total_samples} samples and no matching num_videos."
            )
        cum_videos = torch.tensor([0] + num_videos).cumsum(0)
        cum_video_rows = torch.cat([
            torch.zeros(1, dtype = torch.long),
            video_grid_thw.prod(dim = -1).cumsum(0).cpu(),
        ])

    def _row_axis_is_images(value, flat_ndim):
        """Whether this tensor's first dimension counts IMAGES rather than samples.

        Equal lengths are not enough on their own. A model that pads its image tensors to the
        widest sample -- SmolVLM, Idefics -- keeps the sample axis first, and a batch whose
        image counts happen to sum to the number of samples (``num_images = [2, 0]`` over two
        samples) makes the two readings numerically identical. Slicing such a batch by image
        sends both of the first sample's rows to one chunk and an empty tensor to the next.

        The layouts differ in RANK, which the counts cannot express: the padded one carries an
        explicit per-sample image axis (``[B, max_images, C, H, W]`` for pixels,
        ``[B, max_images, 2]`` for image sizes), one dimension more than the flattened form
        this branch is for.
        """
        if value is None or total_images is None:
            return False
        if _first_dim_len(value) != total_images:
            return False
        if total_images != total_samples:
            # The length has already settled it. Rank can only add anything inside the region
            # where the image axis and the sample axis are the same length; outside it a
            # genuinely flattened tensor is free to outrank flat_ndim, and LLaVA-NeXT and
            # LLaVA-OneVision do exactly that -- they pad on the PATCH axis and store
            # [n_images, max_patches, C, H, W], one row per image with a rank of 5. Applying
            # the rank test there sent each sample the rows of whichever sample shares its
            # index, and dropped the surplus images entirely.
            return True
        ndim = getattr(value, "ndim", None)
        if isinstance(ndim, int) and ndim > flat_ndim:
            return False
        return True

    def _image_sizes_slice(start, end, img_start, img_end):
        if image_sizes is None:
            return None
        if img_start is not None and _row_axis_is_images(image_sizes, 2):
            return image_sizes[img_start:img_end]
        return image_sizes[start:end]

    # Asked once per call, not per chunk: it reads the installed companion's source.
    drop_pixels = grpo_companion_drops_pixels(vision)
    global _GRPO_COMPANION_PIXELS_WARNED
    if drop_pixels and not _GRPO_COMPANION_PIXELS_WARNED:
        _GRPO_COMPANION_PIXELS_WARNED = True
        logger.warning(
            "Unsloth: the installed unsloth computes GRPO reference logprobs with a chunk "
            "loop of its own, and that loop forwards no pixel_values for a model without "
            "image_grid_thw. Holding the gradient pass to the same inputs, so both policies "
            "match, which means this run trains on the text of these samples. Upgrade "
            "unsloth to train on the images."
        )

    chunks = []
    current_pixel_idx = 0
    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        chunk = {}
        if token_type_ids is not None:
            chunk["token_type_ids"] = token_type_ids[start:end]
        if mm_token_type_ids is not None:
            chunk["mm_token_type_ids"] = mm_token_type_ids[start:end]

        img_start = img_end = None
        if cum_imgs is not None:
            img_start, img_end = int(cum_imgs[start]), int(cum_imgs[end])
        vid_start = vid_end = 0
        if cum_videos is not None:
            vid_start, vid_end = int(cum_videos[start]), int(cum_videos[end])
        if vid_end > vid_start:
            chunk["pixel_values_videos"] = pixel_values_videos[
                int(cum_video_rows[vid_start]) : int(cum_video_rows[vid_end])
            ]
            chunk["video_grid_thw"] = video_grid_thw[vid_start:vid_end]
            if second_per_grid_ts is not None:
                chunk["second_per_grid_ts"] = second_per_grid_ts[vid_start:vid_end]

        if pixel_values is None:
            if image_sizes is not None:
                chunk["image_sizes"] = _image_sizes_slice(start, end, img_start, img_end)
            chunks.append(chunk)
            continue

        if image_grid_thw is not None:
            if num_images is None:
                grid_slice = image_grid_thw[start:end]
                batch_pixel_count = grid_slice.prod(dim = -1).sum().item()
                start_pixel_idx = current_pixel_idx
                end_pixel_idx = current_pixel_idx + batch_pixel_count
                current_pixel_idx = end_pixel_idx
            else:
                start_pixel_idx = cum_rows[start].item()
                end_pixel_idx = cum_rows[end].item()
                grid_slice = image_grid_thw[img_start:img_end]
            chunk["image_grid_thw"] = grid_slice
            chunk["pixel_values"] = pixel_values[start_pixel_idx:end_pixel_idx]
            if pixel_attention_mask is not None:
                if img_start is not None and pixel_attention_mask.shape[0] == image_grid_thw.shape[0]:
                    chunk["pixel_attention_mask"] = pixel_attention_mask[img_start:img_end]
                elif (
                    pixel_attention_mask.shape[0] == pixel_values.shape[0]
                    and pixel_attention_mask.shape[0] != total_samples
                ):
                    chunk["pixel_attention_mask"] = pixel_attention_mask[start_pixel_idx:end_pixel_idx]
                else:
                    chunk["pixel_attention_mask"] = pixel_attention_mask[start:end]
            if image_sizes is not None:
                chunk["image_sizes"] = _image_sizes_slice(start, end, img_start, img_end)
        elif image_position_ids is not None:
            if img_start is None:
                chunk["pixel_values"] = pixel_values[start:end]
                chunk[position_ids_key] = image_position_ids[start:end]
            else:
                chunk["pixel_values"] = pixel_values[img_start:img_end]
                chunk[position_ids_key] = image_position_ids[img_start:img_end]
            if pixel_attention_mask is not None:
                chunk["pixel_attention_mask"] = pixel_attention_mask[start:end]
            if image_sizes is not None:
                chunk["image_sizes"] = _image_sizes_slice(start, end, img_start, img_end)
        elif spatial_shapes is not None:
            if cum_tiles is not None:
                tile_start, tile_end = int(cum_tiles[start]), int(cum_tiles[end])
            elif img_start is not None and _first_dim_len(spatial_shapes) == total_images:
                tile_start, tile_end = img_start, img_end
            else:
                tile_start, tile_end = start, end
            chunk["pixel_values"] = pixel_values[tile_start:tile_end]
            chunk["spatial_shapes"] = spatial_shapes[tile_start:tile_end]
            if pixel_attention_mask is not None:
                chunk["pixel_attention_mask"] = pixel_attention_mask[tile_start:tile_end]
            if image_sizes is not None:
                chunk["image_sizes"] = _image_sizes_slice(start, end, img_start, img_end)
        elif cum_tiles is not None:
            tile_start, tile_end = int(cum_tiles[start]), int(cum_tiles[end])
            chunk["pixel_values"] = pixel_values[tile_start:tile_end]
            if pixel_attention_mask is not None:
                chunk["pixel_attention_mask"] = pixel_attention_mask[start:end]
            if image_sizes is not None:
                chunk["image_sizes"] = _image_sizes_slice(start, end, img_start, img_end)
        else:
            # Not a bare length comparison: see _row_axis_is_images. A flattened image tensor
            # is one row per image, [N, C, H, W]; a padded one keeps the sample axis in front
            # of it and must be sliced by sample like everything else in this chunk.
            if img_start is not None and _row_axis_is_images(pixel_values, 4):
                chunk["pixel_values"] = pixel_values[img_start:img_end]
            else:
                chunk["pixel_values"] = pixel_values[start:end]
            if pixel_attention_mask is not None:
                chunk["pixel_attention_mask"] = pixel_attention_mask[start:end]
            if image_sizes is not None:
                chunk["image_sizes"] = _image_sizes_slice(start, end, img_start, img_end)
        if drop_pixels:
            # Here, on the forwarded chunk, rather than on the mapping the caller branches on.
            # The mask goes with the pixels: the companion appends None for it in the same
            # branch, and a mask with nothing to mask is not a shape any of these models take.
            chunk.pop("pixel_values", None)
            chunk.pop("pixel_attention_mask", None)
        chunks.append(chunk)
    return chunks
pass
RL_REPLACEMENTS["grpo_vision_chunks"] = grpo_vision_chunks


GRPO_ROLLOUT_RS_OPTIONS = (
    "token_k1", "token_k2", "token_k3",
    "seq_sum_k1", "seq_sum_k2", "seq_sum_k3",
    "seq_mean_k1", "seq_mean_k2", "seq_mean_k3",
    "seq_max_k2", "seq_max_k3",
)

def grpo_parse_is_clip(spec, mode):
    """Parse UNSLOTH_GRPO_VLLM_IS_CLIP "lower_upper" (either side may be empty) into (lower, upper).
    Applied after GRPOConfig validation, so it checks the bounds itself: reversed bounds collapse
    every truncated ratio to one value and make the mask modes reject every token."""
    if "_" not in str(spec):
        # A bare "2.0" would otherwise become a lower bound and floor / reject every on-policy ratio.
        raise ValueError(
            f"Unsloth: UNSLOTH_GRPO_VLLM_IS_CLIP must be 'lower_upper' (either side may be empty, "
            f"e.g. '_{spec}' for an upper cap), got {spec!r}."
        )
    lower, _, upper = str(spec).partition("_")
    lower = float(lower) if lower.strip() else None
    upper = float(upper) if upper.strip() else None
    if (lower is not None and not (0 <= lower < math.inf)) or (upper is not None and not (0 < upper < math.inf)):
        raise ValueError(f"Unsloth: UNSLOTH_GRPO_VLLM_IS_CLIP needs a finite lower >= 0 and upper > 0, got {spec!r}.")
    if lower is not None and upper is not None and lower >= upper:
        # TRL GRPOConfig rejects clip_min >= clip_max: equal bounds make truncate a constant weight
        # and the mask modes keep only ratios exactly equal to it.
        raise ValueError(f"Unsloth: UNSLOTH_GRPO_VLLM_IS_CLIP lower bound must be below the upper bound in {spec!r}.")
    if lower is None and upper is None and str(mode).endswith("_truncate"):
        raise ValueError(f"Unsloth: UNSLOTH_GRPO_VLLM_IS_CLIP={spec!r} gives no bound for {mode}.")
    return lower, upper

def grpo_parse_rollout_rs(options, thresholds):
    """Parse verl rollout_rs / rollout_rs_threshold strings (comma separated, one threshold shared
    or one per option; k1 takes "lower_upper" ratio bounds, default lower = 1 / upper; k2 / k3 a
    single upper bound) into (option, lower_log, upper) tuples, k1 bounds in log space.
    Returns None when options is empty."""
    if options is None or not str(options).strip():
        return None
    names = []
    for name in str(options).split(","):
        name = name.strip()
        if not name: continue
        if name not in GRPO_ROLLOUT_RS_OPTIONS:
            raise ValueError(f"Unsloth: unknown rollout rejection option {name}; choose from {GRPO_ROLLOUT_RS_OPTIONS}.")
        if name not in names: names.append(name)
    if not names:
        return None
    if thresholds is None or not str(thresholds).strip():
        raise ValueError("Unsloth: UNSLOTH_GRPO_ROLLOUT_RS needs UNSLOTH_GRPO_ROLLOUT_RS_THRESHOLD.")
    specs = [x.strip() for x in str(thresholds).split(",") if x.strip()]
    if len(specs) not in (1, len(names)):
        raise ValueError(f"Unsloth: give one rollout rejection threshold or {len(names)}, got {len(specs)}.")
    if len(specs) == 1: specs = specs * len(names)
    parsed = []
    for name, spec in zip(names, specs):
        if name.endswith("k1"):
            if "_" in spec:
                lower, upper = (float(x) for x in spec.split("_", 1))
            else:
                upper = float(spec)
                lower = 1.0 / upper
            if not (0 < lower <= upper < math.inf):
                raise ValueError(f"Unsloth: rollout rejection bounds for {name} must be finite with 0 < lower <= upper, got {spec}.")
            parsed.append((name, math.log(lower), math.log(upper)))
        else:
            if "_" in spec:
                raise ValueError(f"Unsloth: {name} takes a single upper bound, got {spec}.")
            upper = float(spec)
            if not (0 < upper < math.inf):
                raise ValueError(f"Unsloth: rollout rejection threshold for {name} must be positive and finite, got {spec}.")
            parsed.append((name, None, upper))
    return tuple(parsed)
pass
RL_REPLACEMENTS["grpo_parse_rollout_rs"] = grpo_parse_rollout_rs


def grpo_rollout_rs_keep(log_ratio, mask, rollout_rs):
    """Bool (B, T) keep mask of grpo_compute_loss's rollout_rs rejection, for the caller's metric
    and entropy weights. grpo_compute_loss inlines the same steps (its source is copied into the
    compiled trainer, so it cannot call this); log_ratio is old - sampling, or new - rollout with
    rollout logprobs as pi_old."""
    with torch.no_grad():
        rs_log_ratio = torch.clamp(torch.nan_to_num(log_ratio * mask, nan = 0.0), min = -20.0, max = 20.0)
        rs_mask = mask.to(rs_log_ratio.dtype)
        rs_count = rs_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        rs_keep = torch.ones_like(rs_log_ratio)
        for rs_option, rs_lower, rs_upper in rollout_rs:
            if rs_option.endswith("k1"):
                rs_stat = -rs_log_ratio
            elif rs_option.endswith("k2"):
                rs_stat = 0.5 * rs_log_ratio * rs_log_ratio
            else:
                rs_stat = torch.expm1(rs_log_ratio) - rs_log_ratio
            rs_stat = rs_stat * rs_mask
            if rs_option.startswith("seq_sum"):
                rs_stat = rs_stat.sum(dim=-1, keepdim=True)
            elif rs_option.startswith("seq_mean"):
                rs_stat = rs_stat.sum(dim=-1, keepdim=True) / rs_count
            elif rs_option.startswith("seq_max"):
                rs_stat = rs_stat.amax(dim=-1, keepdim=True)
            if rs_lower is not None:
                rs_pass = (rs_stat >= rs_lower) & (rs_stat <= rs_upper)
            else:
                rs_pass = rs_stat <= rs_upper
            rs_keep = rs_keep * rs_pass.to(rs_keep.dtype)
        return rs_keep > 0
pass
RL_REPLACEMENTS["grpo_rollout_rs_keep"] = grpo_rollout_rs_keep


def grpo_rollout_mismatch_stats(old_logps, sampling_logps, mask, is_weights):
    """Sums for the rollout / trainer mismatch metrics over valid tokens, so processes and
    micro-batches can be pooled before dividing. Returns a float64 vector:
    [tokens, sum k1, sum k3, max k3, sum w, sum w^2, rejected tokens, rows, rejected rows],
    k1 = log pi_rollout - log pi_train and k3 = r - 1 - log r (r = pi_train / pi_rollout) estimate
    KL(rollout || train) as verl compute_offpolicy_metrics; w = the applied IS weights."""
    with torch.no_grad():
        log_ratio = old_logps.to(torch.float64) - sampling_logps.to(torch.float64)
        # Tokens vLLM could not score (NaN) carry no divergence information: leave them out.
        keep = mask.to(torch.bool) & ~torch.isnan(log_ratio)
        maskd = keep.to(torch.float64)
        log_ratio = torch.nan_to_num(log_ratio, nan = 0.0) * maskd
        # verl compute_offpolicy_metrics does not clamp: a +-20 clamp capped K1 at 20 and K3 near
        # exp(20). Only keep the float64 sums finite (and -inf logps out of inf - inf): exp(670)
        # leaves room for 1e15 saturated tokens below the float64 max (exp(709.8)).
        log_ratio = torch.clamp(log_ratio, min = -670.0, max = 670.0)
        k3 = (torch.expm1(log_ratio) - log_ratio) * maskd
        w = torch.broadcast_to(is_weights.to(torch.float64), maskd.shape) * maskd
        rejected = ((w == 0) & keep).to(torch.float64)
        rows = (maskd.sum(-1) > 0).to(torch.float64)
        return torch.stack([
            maskd.sum(),
            (-log_ratio).sum(),
            k3.sum(),
            k3.amax() if k3.numel() > 0 else k3.new_zeros(()),
            w.sum(),
            (w * w).sum(),
            rejected.sum(),
            rows.sum(),
            ((rejected.sum(-1) > 0).to(torch.float64) * rows).sum(),
        ])
pass
RL_REPLACEMENTS["grpo_rollout_mismatch_stats"] = grpo_rollout_mismatch_stats


def grpo_rollout_mismatch_metrics(stats):
    """Metric dict from (summed, k3 max maxed) grpo_rollout_mismatch_stats. ESS is the normalised
    effective sample size (sum w)^2 / (N sum w^2) of the applied IS weights, verl's
    rollout_is_eff_sample_size."""
    stats = [float(x) for x in stats.tolist()]
    n, k1, k3, k3_max, w1, w2, rej, rows, rej_rows = stats
    return {
        "sampling/rollout_kl_k1/mean": k1 / max(n, 1.0),
        "sampling/rollout_kl_k3/mean": k3 / max(n, 1.0),
        "sampling/rollout_kl_k3/max": k3_max,
        "sampling/importance_sampling_ess": (w1 * w1) / (n * w2) if n > 0 and w2 > 0 else 0.0,
        "sampling/importance_sampling_rejected_frac": rej / max(n, 1.0),
        "sampling/importance_sampling_rejected_seq_frac": rej_rows / max(rows, 1.0),
    }
pass
RL_REPLACEMENTS["grpo_rollout_mismatch_metrics"] = grpo_rollout_mismatch_metrics


def grpo_rollout_logps_as_old(sampling_logps, old_logps = None):
    """UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD (verl rollout_correction bypass_mode with loss_type
    ppo_clip, vime --use-rollout-logprobs): the rollout engine's logprobs are pi_old, so the PPO
    ratio is pi_theta / pi_rollout and its clip is the off-policy correction. Returns the
    (old, sampling) pair for grpo_compute_loss: sampling is dropped so no importance weight
    multiplies that ratio a second time (verl compute_policy_loss_bypass_mode: "would be
    double-counting"; vime refuses use_tis with use_rollout_logprobs)."""
    if sampling_logps is None:
        raise ValueError("Unsloth: rollout_logps_as_old needs the rollout engine's sampling logprobs.")
    if old_logps is not None:
        raise ValueError("Unsloth: rollout_logps_as_old batches must not carry trainer-scored old logprobs.")
    return sampling_logps.to(torch.float32), None
pass
RL_REPLACEMENTS["grpo_rollout_logps_as_old"] = grpo_rollout_logps_as_old


def grpo_rollout_old_fill_unscored(old_logps, new_logps):
    """Tokens the rollout engine could not score (vLLM returns NaN) take the current policy's own
    logprob as pi_old, so their ratio is exactly 1: no correction, as the vLLM importance-sampling
    path treats them (TRL zeroes their log ratio). Returns (old, unscored mask)."""
    unscored = ~torch.isfinite(old_logps)
    filled = torch.where(unscored, new_logps.detach().to(old_logps.dtype), old_logps)
    return filled, unscored
pass
RL_REPLACEMENTS["grpo_rollout_old_fill_unscored"] = grpo_rollout_old_fill_unscored


def grpo_rollout_as_old_ratio_metrics(coef_1, mask, log_ratio = None):
    """(|log r|, r) zero-filled outside the mask, r = pi_theta / pi_rollout per token, for the
    sampling_logp_difference / importance_sampling_ratio metrics when pi_old is the rollout policy.
    verl measures the bypass mismatch the same way (compute_rollout_correction_and_rejection_mask
    with old_log_prob = the current log_prob). None for a sequence-level ratio. log_ratio, when
    given, is the unclamped new - rollout for |log r|: coef_1 is the loss ratio, clamped to
    [-20, 20] in log space, and verl compute_offpolicy_metrics reads the raw difference for its
    KL metrics. r stays coef_1, which matches verl's SAFETY_BOUND clamped IS weights."""
    if coef_1 is None or coef_1.shape != mask.shape:
        return None, None
    with torch.no_grad():
        keep = mask.to(torch.bool)
        ratio = coef_1.detach().to(torch.float32)
        if log_ratio is not None and log_ratio.shape == mask.shape:
            log_ratio = torch.nan_to_num(log_ratio.detach().to(torch.float32), nan = 0.0, posinf = 670.0, neginf = -670.0)
        else:
            log_ratio = torch.nan_to_num(torch.log(ratio), nan = 0.0, posinf = 670.0, neginf = -670.0)
        # where, not * mask: an overflowed ratio (inf) on a padded column would leave inf * 0 = nan.
        zero = torch.zeros_like(ratio)
        return torch.where(keep, log_ratio.abs(), zero), torch.where(keep, ratio, zero)
pass
RL_REPLACEMENTS["grpo_rollout_as_old_ratio_metrics"] = grpo_rollout_as_old_ratio_metrics


def grpo_accumulated_loss(
    trainer,
    input_ids,
    attention_mask,
    logits_to_keep,
    completion_mask,
    advantages,
    old_logps,
    ref_logps,
    n_chunks = -1,
    tool_mask = None,
    **kwargs,
):
    # All Unsloth Zoo code licensed under AGPL3
    # Body-local import so the copy inlined into the generated trainer cache resolves.
    try:
        from unsloth_zoo.rl_replacements import _warn_unsupported_grpo_options
        _warn_unsupported_grpo_options(trainer)
    except Exception:
        pass

    # Body-local: this source is copied into the generated trainer without its imports.
    from unsloth_zoo.rl_replacements import (
        grpo_shared_vision_inputs as _grpo_get_vision_inputs,
        grpo_vision_chunks as _grpo_vision_chunks,
        chunked_hidden_states_selective_log_softmax,
        chunked_selective_log_softmax,
        grpo_rollout_logps_as_old,
        grpo_rollout_as_old_ratio_metrics,
        grpo_rollout_old_fill_unscored,
    )
    if os.environ.get("UNSLOTH_GRPO_LOGPROB_RECOMPUTE", "0") == "1":
        from unsloth_zoo.rl_replacements import (
            chunked_hidden_states_selective_log_softmax_recompute as chunked_hidden_states_selective_log_softmax,
        )
    # Adaptive entropy control is not implemented, so it disables the bonus (and warns).
    _entropy_quantile = getattr(trainer.args, "top_entropy_quantile", 1.0)
    _entropy_mask_on = (
        _entropy_quantile is not None and _entropy_quantile < 1.0
        and hasattr(trainer, "get_high_entropy_mask")
    )
    _entropy_coef = float(getattr(trainer, "entropy_coef", getattr(trainer.args, "entropy_coef", 0.0)) or 0.0)
    _entropy_bonus_on = _entropy_coef != 0.0 and not getattr(trainer.args, "use_adaptive_entropy", False)
    _need_entropy = _entropy_mask_on or _entropy_bonus_on
    if _need_entropy:
        from unsloth_zoo.rl_replacements import (
            chunked_hidden_states_selective_log_softmax_recompute as _entropy_log_softmax,
            UnslothGRPOEntropyBonus as _UnslothGRPOEntropyBonus,
            grpo_entropy_bonus_weights as _grpo_entropy_bonus_weights,
            grpo_high_entropy_mask as _grpo_high_entropy_mask,
        )
    vision_inputs = _grpo_get_vision_inputs(kwargs)
    pixel_values = vision_inputs.get('pixel_values', None)
    image_grid_thw = vision_inputs.get('image_grid_thw', None)
    video_grid_thw = vision_inputs.get('video_grid_thw', None)
    if pixel_values is None:
        # Only a sentinel below (the chunks carry the tensors): video rows must take the vision
        # path too, since left and sequence packing move M-RoPE video positions.
        pixel_values = vision_inputs.get('pixel_values_videos', None)
    # Released unsloth 2026.9.4 decides whether multi-image GRPO is supported by grepping
    # inspect.getsource(grpo_accumulated_loss) for "num_images", so moving the handling into
    # grpo_vision_chunks makes that probe answer no and raise "Please upgrade unsloth_zoo" at
    # the user who just did. The chunker reads num_images out of vision_inputs itself; this
    # binding is what the released probe looks for, and it keeps the name meaningful here.
    num_images = vision_inputs.get('num_images', None)
    # Transformers 5.x requires token_type_ids/mm_token_type_ids for some vision models
    token_type_ids = vision_inputs.get('token_type_ids', None)
    mm_token_type_ids = vision_inputs.get('mm_token_type_ids', None)
    if mm_token_type_ids is not None or image_grid_thw is not None or video_grid_thw is not None:
        mm_token_type_ids = _unsloth_fix_mm_token_type_ids(
            trainer.processing_class, input_ids, mm_token_type_ids
        )
        vision_inputs['mm_token_type_ids'] = mm_token_type_ids
    # Thread vLLM sampling logprobs when something actually consumes them: the off-policy mask
    # (off_policy_mask_threshold) or the IS ratio (vllm_importance_sampling_correction). The mask
    # needs them regardless of IS correction (matching TRL, which feeds them to get_off_policy_mask
    # either way); the IS ratio stays gated on the correction flag inside grpo_compute_loss. On the
    # plain vLLM path (neither active) they are dropped so nothing pays for an unused aligned/compiled
    # input and grpo_compute_loss returns None (not empty) delta/flat_is_ratio.
    # Set by unsloth when UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD=1 skipped the old-policy forward.
    _rollout_logps_as_old = bool(kwargs.pop("rollout_logps_as_old", False))
    _sampling_logps_used = (
        getattr(trainer, "vllm_importance_sampling_correction", False)
        or getattr(trainer.args, "off_policy_mask_threshold", None) is not None
        or _rollout_logps_as_old
    )
    sampling_per_token_logps = kwargs.get("sampling_per_token_logps", None) if _sampling_logps_used else None
    temperature = kwargs.get("temperature", 1.0)
    logit_scale_multiply = kwargs.get("logit_scale_multiply", 0.0)
    logit_scale_divide   = kwargs.get("logit_scale_divide", 0.0)
    logit_softcapping    = kwargs.get("logit_softcapping", 0.0)
    prev_max_left_pad    = kwargs.get("max_left_pad", 0) # max_left_pad for LLM training, enabled by default.

    # Pop from kwargs to avoid downstream issues.
    _ = kwargs.pop("sampling_per_token_logps", None)
    kwargs["vllm_importance_sampling_cap"] = getattr(trainer.args, "vllm_importance_sampling_cap", None)
    # Older TRL lacks this arg; fall back to token_truncate (legacy clamp(max=cap) behavior).
    kwargs["vllm_importance_sampling_mode"] = getattr(trainer.args, "vllm_importance_sampling_mode", None) or "token_truncate"
    # Opt-in for TRL versions without the field or the geometric modes, e.g. geometric_mask.
    if os.environ.get("UNSLOTH_GRPO_VLLM_IS_MODE", ""):
        kwargs["vllm_importance_sampling_mode"] = os.environ["UNSLOTH_GRPO_VLLM_IS_MODE"]
    # Like TRL, only check the mode where it is used: with the correction off it stays a no-op.
    if getattr(trainer, "use_vllm", False) and getattr(trainer, "vllm_importance_sampling_correction", False) and kwargs["vllm_importance_sampling_mode"] not in (
        "token_truncate", "token_mask", "sequence_truncate", "sequence_mask", "geometric_truncate", "geometric_mask",
    ):
        raise ValueError(f"Unsloth: unknown vLLM importance sampling mode {kwargs['vllm_importance_sampling_mode']}.")
    # TRL validates this in __init__, before the env override: get_gamma_weights sums a per-token log ratio.
    if (
        kwargs.get("loss_type", getattr(trainer, "loss_type", None)) == "vespo"
        and getattr(trainer, "use_vllm", False)
        and getattr(trainer, "vllm_importance_sampling_correction", False)
        and kwargs["vllm_importance_sampling_mode"] not in ("token_truncate", "token_mask")
    ):
        raise ValueError(
            "Unsloth: VESPO loss requires vllm_importance_sampling_mode to be 'token_truncate' or 'token_mask'. "
            f"Got: {kwargs['vllm_importance_sampling_mode']}."
        )
    kwargs["vllm_importance_sampling_clip_min"] = getattr(trainer.args, "vllm_importance_sampling_clip_min", None)
    kwargs["vllm_importance_sampling_clip_max"] = getattr(trainer.args, "vllm_importance_sampling_clip_max", None)
    kwargs["get_sapo_token_loss"] = trainer.get_sapo_token_loss if hasattr(trainer, "get_sapo_token_loss") else None
    kwargs["sapo_temperature_pos"] = trainer.args.sapo_temperature_pos if hasattr(trainer.args, "sapo_temperature_pos") else None
    kwargs["sapo_temperature_neg"] = trainer.args.sapo_temperature_neg if hasattr(trainer.args, "sapo_temperature_neg") else None
    kwargs["get_gamma_weights"] = trainer.get_gamma_weights if hasattr(trainer, "get_gamma_weights") else None
    kwargs["vespo_k_pos"] = trainer.args.vespo_k_pos if hasattr(trainer.args, "vespo_k_pos") else 2.0
    kwargs["vespo_k_neg"] = trainer.args.vespo_k_neg if hasattr(trainer.args, "vespo_k_neg") else 3.0
    kwargs["vespo_lambda_pos"] = trainer.args.vespo_lambda_pos if hasattr(trainer.args, "vespo_lambda_pos") else 3.0
    kwargs["vespo_lambda_neg"] = trainer.args.vespo_lambda_neg if hasattr(trainer.args, "vespo_lambda_neg") else 2.0
    off_policy_mask_threshold = trainer.args.off_policy_mask_threshold if hasattr(trainer.args, "off_policy_mask_threshold") else None
    kwargs["off_policy_mask_threshold"] = off_policy_mask_threshold
    # get_off_policy_mask exists on TRL >= 0.27.0; its 3rd parameter was `old_per_token_logps` in 0.27.0
    # and renamed to `sampling_per_token_logps` in 0.27.1 (huggingface/trl#4857), so a fixed keyword call
    # crashes on one side of the rename. Wrap it in a signature-stable adapter here, outside the compiled
    # loss, so grpo_compute_loss always calls it with one keyword. Detect the real name once via inspect
    # and cache the adapter on the trainer; a fresh closure every step would re-trigger torch.compile.
    _off_policy_mask_fn = trainer.get_off_policy_mask if hasattr(trainer, "get_off_policy_mask") else None
    if _off_policy_mask_fn is None or off_policy_mask_threshold is None:
        kwargs["get_off_policy_mask"] = None
    else:
        _adapter = getattr(trainer, "_unsloth_off_policy_mask_adapter", None)
        # Compare by value, not identity: trainer.get_off_policy_mask returns a fresh bound-method
        # object on every access, so `is not` would always miss and rebuild the adapter each step
        # (re-triggering torch.compile). `!=` on bound methods compares __self__ and __func__, so the
        # cached adapter is reused; the first call still rebuilds since None != the bound method.
        if getattr(_adapter, "_unsloth_wrapped", None) != _off_policy_mask_fn:
            import inspect as _inspect
            try:
                _params = _inspect.signature(_off_policy_mask_fn).parameters
            except (TypeError, ValueError):
                _params = {}
            if "old_per_token_logps" in _params and "sampling_per_token_logps" not in _params:
                # TRL 0.27.0 named the mismatch-logprobs parameter old_per_token_logps.
                def _adapter(advantages, per_token_logps, sampling_per_token_logps, mask, off_policy_threshold):
                    return _off_policy_mask_fn(
                        advantages=advantages,
                        per_token_logps=per_token_logps,
                        old_per_token_logps=sampling_per_token_logps,
                        mask=mask,
                        off_policy_threshold=off_policy_threshold,
                    )
            else:
                # TRL >= 0.27.1 / 1.7.x use sampling_per_token_logps (also the default going forward).
                def _adapter(advantages, per_token_logps, sampling_per_token_logps, mask, off_policy_threshold):
                    return _off_policy_mask_fn(
                        advantages=advantages,
                        per_token_logps=per_token_logps,
                        sampling_per_token_logps=sampling_per_token_logps,
                        mask=mask,
                        off_policy_threshold=off_policy_threshold,
                    )
            _adapter._unsloth_wrapped = _off_policy_mask_fn
            trainer._unsloth_off_policy_mask_adapter = _adapter
        kwargs["get_off_policy_mask"] = _adapter
    # Read inside the compiled loss to gate the IS ratio, which the off-policy mask must not gate.
    kwargs["vllm_importance_sampling_correction"] = getattr(trainer, "vllm_importance_sampling_correction", False)
    # Follows TRL's own value; older TRL has no such field and False is correct there.
    kwargs["use_bias_correction_kl"] = getattr(trainer.args, "use_bias_correction_kl", False)
    kwargs["use_vllm"] = trainer.use_vllm
    # Eval batches are not split or accumulated, so keep the full count.
    _training = getattr(getattr(trainer, "model", None), "training", True)
    kwargs["steps_per_generation"] = getattr(trainer.args, "steps_per_generation", None) if _training else None
    # Generated trainers still pass unsloth_num_chunks; nothing downstream reads it.
    try:
        from unsloth_zoo.rl_replacements import _warn_deprecated_n_chunks
        _warn_deprecated_n_chunks(n_chunks)
    except Exception:
        pass

    if kwargs["vllm_importance_sampling_clip_max"] is None and kwargs["vllm_importance_sampling_cap"] is not None:
        kwargs["vllm_importance_sampling_clip_min"] = 0
        kwargs["vllm_importance_sampling_clip_max"] = kwargs["vllm_importance_sampling_cap"]
    # "lower_upper" (either side may be empty), e.g. "0.999_1.001" for the geometric modes.
    _is_clip_env = os.environ.get("UNSLOTH_GRPO_VLLM_IS_CLIP", "")
    if _is_clip_env:
        from unsloth_zoo.rl_replacements import grpo_parse_is_clip as _grpo_parse_is_clip
        kwargs["vllm_importance_sampling_clip_min"], kwargs["vllm_importance_sampling_clip_max"] = _grpo_parse_is_clip(
            _is_clip_env, kwargs["vllm_importance_sampling_mode"],
        )
    # verl rollout_rs rejection (token_k3, seq_mean_k3, seq_mean_k1, ...), applied with the vLLM IS ratio.
    from unsloth_zoo.rl_replacements import grpo_parse_rollout_rs as _grpo_parse_rollout_rs
    _rollout_rs = _grpo_parse_rollout_rs(
        os.environ.get("UNSLOTH_GRPO_ROLLOUT_RS", ""), os.environ.get("UNSLOTH_GRPO_ROLLOUT_RS_THRESHOLD", ""),
    )
    if _rollout_rs is not None:
        kwargs["rollout_rs"] = _rollout_rs
    _rollout_metrics_on = os.environ.get("UNSLOTH_GRPO_ROLLOUT_METRICS", "0") == "1"

    if not hasattr(trainer, '_autocast_dtype'):
        # "no" is float32 training (a T4 / V100 without bfloat16): autocasting it to bfloat16 raises there.
        _mixed_precision = os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16')
        trainer._autocast_dtype = None if _mixed_precision == 'no' else (torch.float16 if _mixed_precision == 'fp16' else torch.bfloat16)
        if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1': trainer._autocast_dtype = None
    pass
    # Restored in `finally`: an OOM or interrupt below must not leave later forwards returning hidden states.
    _unsloth_prior_hidden_states = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        output_embeddings = trainer.model.get_output_embeddings()
        lm_head = output_embeddings.weight
        lm_head_bias = getattr(output_embeddings, "bias", None)
        # ZeRO-3 released the head after the forward: score with one gathered copy per call.
        _zero3_head = None
        from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_zero3_partitioned, unsloth_zero3_gathered
        if unsloth_zero3_partitioned(lm_head) or unsloth_zero3_partitioned(lm_head_bias):
            _zero3_head = (unsloth_zero3_gathered(lm_head), unsloth_zero3_gathered(lm_head_bias))
            lm_head, lm_head_bias = _zero3_head

        def _refresh_lm_head():
            # FSDP2's forward unshard swaps the Parameters: the bindings above are metadata-only until then.
            nonlocal lm_head, lm_head_bias
            if _zero3_head is not None:
                lm_head, lm_head_bias = _zero3_head
                return
            from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fsdp_unshard
            unsloth_fsdp_unshard(output_embeddings)
            lm_head = output_embeddings.weight
            lm_head_bias = getattr(output_embeddings, "bias", None)

        from unsloth_zoo.rl_replacements import grpo_truncated_support_kwargs
        _support_kwargs = grpo_truncated_support_kwargs(trainer)
        # old / ref logps come from Unsloth's _get_per_token_logps, which marks the trainer when
        # it scored them over the same set; mixing the two normalisations would bias every ratio.
        if _support_kwargs and (old_logps is not None or ref_logps is not None) \
                and not getattr(trainer, "_unsloth_truncated_support_scored", False):
            if not getattr(trainer, "_unsloth_truncated_support_refused", False):
                from unsloth_zoo.log import logger as _zoo_logger
                _zoo_logger.warning(
                    "Unsloth: UNSLOTH_GRPO_TRUNCATED_SUPPORT=1 needs an unsloth build that scores the "
                    "old / reference logprobs over the same kept set. Ignoring it: pip install -U unsloth"
                )
                trainer._unsloth_truncated_support_refused = True
            _support_kwargs = {}
        if _support_kwargs and getattr(trainer, "_unsloth_truncated_support_refused", False):
            # Unsloth found the scored policy is not the sampler (an unsynced vLLM sampler, a reference
            # model, reused rollouts): the current policy's own kept set would not be vLLM's either.
            _support_kwargs = {}
        if _support_kwargs:
            from functools import partial
            if os.environ.get("UNSLOTH_GRPO_LOGPROB_RECOMPUTE", "0") == "1":
                # The recompute helper scores the full vocabulary only, so the kept set uses the plain helper
                from unsloth_zoo.rl_replacements import chunked_hidden_states_selective_log_softmax
                if not getattr(trainer, "_unsloth_truncated_support_recompute_warned", False):
                    from unsloth_zoo.log import logger as _zoo_logger
                    _zoo_logger.warning(
                        "Unsloth: UNSLOTH_GRPO_LOGPROB_RECOMPUTE=1 is not used with UNSLOTH_GRPO_TRUNCATED_SUPPORT=1; "
                        "the logprobs are scored with the standard helper over the kept set."
                    )
                    trainer._unsloth_truncated_support_recompute_warned = True
            chunked_hidden_states_selective_log_softmax = partial(
                chunked_hidden_states_selective_log_softmax, **_support_kwargs,
            )
            chunked_selective_log_softmax = partial(chunked_selective_log_softmax, **_support_kwargs)
        # Unsloth keeps _autocast_dtype set when it turns autocast off (float32 training on a GPU without bfloat16).
        _autocast_on = trainer._autocast_dtype is not None and getattr(trainer, "_autocast_enabled", True)
        dtype_bytes = 16 if _autocast_on and trainer._autocast_dtype in [torch.float16, torch.bfloat16] else 32

        total_rows = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        hidden_dim = lm_head.shape[1]
        vocab_dim = lm_head.shape[0]

        if trainer.args.unsloth_grpo_mini_batch is None:
            # Size per call, as unsloth's copy does: caching in args froze the first step's plan.
            B, multiplier = autotune_batch_and_chunks(
                total_rows, seq_len, hidden_dim, vocab_dim, dtype_bytes, trainer.args.unsloth_logit_chunk_multiplier
            )
            B = max(1, total_rows//B)
        else:
            if trainer.args.unsloth_grpo_mini_batch > total_rows:
                B = total_rows
            else:
                B = trainer.args.unsloth_grpo_mini_batch

            if trainer.args.unsloth_logit_chunk_multiplier is None:
                multiplier = max(4, seq_len // 4096)
            else:
                multiplier = trainer.args.unsloth_logit_chunk_multiplier

        # The text path rebuilds completion_mask from token ids, which would undo TRL's
        # mask_truncated_completions row zeroing; keep TRL's rows and reapply them before the loss.
        kept_completion_rows = None
        if (
            pixel_values is None
            and getattr(trainer, "mask_truncated_completions", False)
            and torch.is_tensor(completion_mask)
            and completion_mask.dim() == 2
            and completion_mask.shape[0] == input_ids.shape[0]
        ):
            kept_completion_rows = completion_mask.sum(dim = 1, keepdim = True) > 0

        if pixel_values is None:
            left_pad_tokens_per_prompt = calculate_pad_tokens_in_prompt(input_ids, logits_to_keep, trainer.processing_class.pad_token_id)

            # Determine max_left_pad from precomputed logprobs shape for consistency
            if old_logps is not None:
                max_left_pad = old_logps.shape[1] - logits_to_keep
            elif ref_logps is not None:
                max_left_pad = ref_logps.shape[1] - logits_to_keep
            else:
                max_left_pad = torch.max(left_pad_tokens_per_prompt).item()

            input_ids = left_pack_padding(input_ids, trainer.processing_class.pad_token_id)

            completion_input_ids = input_ids[:, -(logits_to_keep +max_left_pad):]
            completion_mask = create_completion_attention_mask(completion_input_ids, left_pad_tokens_per_prompt, max_left_pad, trainer.processing_class.pad_token_id).to(attention_mask.dtype)

            if (trainer.use_vllm or _rollout_logps_as_old) and sampling_per_token_logps is not None:
                sampling_per_token_logps = align_logprobs_with_mask(sampling_per_token_logps, completion_mask)
            else:
                sampling_per_token_logps = None
            completion_mask = align_completion_tool_mask(tool_mask, completion_mask)
            attention_mask =  input_ids != trainer.processing_class.pad_token_id
            attention_mask = attention_mask.to(attention_mask.dtype)
        else:
            completion_input_ids = input_ids[:, -logits_to_keep:]
            completion_mask = align_completion_tool_mask(tool_mask, completion_mask)

        if _rollout_logps_as_old:
            old_logps, sampling_per_token_logps = grpo_rollout_logps_as_old(sampling_per_token_logps, old_logps)

        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper = False)

        for module in unwrapped_model.modules():
            if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "io_same_decice"):
                module._hf_hook.io_same_decice = False
        pass

        all_logprobs_list = []
        all_entropies_list = []

        # No width dispatch in these helpers: unsloth rewrites every
        # `if <x>.shape[-1] == lm_head.shape[1]:` in this function and refuses any other spelling.
        def _hidden_logps_and_entropies(hidden_states, ids, chunks):
            logps, entropies = _entropy_log_softmax(
                hidden_states, lm_head, ids, chunks,
                logit_scale_multiply, logit_scale_divide, logit_softcapping, temperature,
                lm_head_bias = lm_head_bias, return_entropy = True,
            )
            if _support_kwargs:
                # Entropies stay full-vocabulary as TRL's; the logprobs take the kept set
                logps = chunked_hidden_states_selective_log_softmax(
                    hidden_states, lm_head, ids, chunks,
                    logit_scale_multiply, logit_scale_divide, logit_softcapping, temperature,
                    lm_head_bias = lm_head_bias,
                )
            return logps, entropies

        def _raw_logits_entropies(logits):
            # The forward already scaled / softcapped these; same temperature as the logprobs.
            if _entropy_bonus_on:
                log_p = torch.log_softmax(logits.to(torch.float32) / temperature, dim = -1)
                return -(log_p.exp() * log_p).sum(-1)
            # Mask only (top_entropy_quantile): TRL scores entropies under no_grad, so no graph is kept,
            # and rows go in slices so the fp32 copies stay near 64M elements, not the whole [B, T, V].
            with torch.no_grad():
                flat = logits.reshape(-1, logits.shape[-1])
                out = torch.empty(flat.shape[0], dtype = torch.float32, device = logits.device)
                step = max(1, (1 << 26) // max(int(flat.shape[-1]), 1))
                for lo in range(0, flat.shape[0], step):
                    log_p = torch.log_softmax(flat[lo : lo + step].to(torch.float32) / temperature, dim = -1)
                    out[lo : lo + step] = -(log_p.exp() * log_p).sum(-1)
                    del log_p
                return out.reshape(logits.shape[:-1])

        import math
        total_samples = input_ids.shape[0]
        batch_size = math.ceil(total_samples / B)
        input_ids_chunks = []
        attention_mask_chunks = []
        completion_ids_chunks = []
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            input_ids_chunks.append(input_ids[start:end])
            attention_mask_chunks.append(attention_mask[start:end])
            completion_ids_chunks.append(completion_input_ids[start:end])

        # Shared with the no-grad pass, so the two cannot slice the same tensors differently.
        vision_chunks = _grpo_vision_chunks(vision_inputs, total_samples, batch_size)

        zipped_inputs = zip(
            input_ids_chunks,
            attention_mask_chunks,
            vision_chunks,
            completion_ids_chunks,
        )

        # Bound in the body, not at module scope, for the reason spelled out just below: this
        # function's source is copied into the generated UnslothGRPOTrainer cache without
        # unsloth_zoo's module imports, so a module-level import reaches the import path and
        # not the one that actually runs in production.
        from contextlib import nullcontext

        if not _autocast_on:
            autocaster = nullcontext()
        else:
            autocaster = torch.amp.autocast(device_type = trainer.model.device.type, dtype = trainer._autocast_dtype)

        # PrefixGrouper grad path. This function's source is copied into the generated
        # UnslothGRPOTrainer cache without unsloth_zoo's module imports, so bind names
        # inside the body; the prefix_grouper import stays lazy + guarded (circular import,
        # may be absent) and a failed import just leaves PG off.
        from unsloth_zoo.temporary_patches.common import UNSLOTH_ENABLE_LOGGING

        # Memoize env gate + import once per process on the function object (which survives
        # into the cache). Env gate checked first so =0 never imports PG code; () = PG off.
        _pg_funcs = getattr(grpo_accumulated_loss, "_pg_funcs", None)
        if _pg_funcs is None:
            _pg_funcs = ()
            if os.environ.get("UNSLOTH_GRPO_PREFIX_GROUPER", "1").lower() not in (
                "0", "false", "no", "off",
            ):
                try:
                    from unsloth.utils.prefix_grouper import (
                        build_group_layout as _pg_build_layout,
                        prefix_grouper_enabled as _pg_enabled_fn,
                        verify_on as _pg_verify_on,
                        tol_ok as _pg_tol_ok,
                        TOL_KILL as _PG_TOL_KILL,
                    )
                    _pg_funcs = (
                        _pg_build_layout, _pg_enabled_fn, _pg_verify_on, _pg_tol_ok, _PG_TOL_KILL,
                    )
                except Exception:
                    _pg_funcs = ()
            grpo_accumulated_loss._pg_funcs = _pg_funcs
        # Skip PG under vLLM (fast_inference=True): rollout dominates the step, so the
        # saving is small and the first-use self-verify is net overhead.
        _pg_engage = bool(_pg_funcs) and not getattr(trainer, "use_vllm", False) and not _need_entropy

        # ---- PrefixGrouper (GRPO shared-prompt dedup; UNSLOTH_GRPO_PREFIX_GROUPER=0 disables) ----
        # Each prompt's G completions share the prefix; PG forwards it once + the G suffixes
        # (FlexAttention shared-prefix mask), cutting G*(P+R) tokens to P+G*R. First-use
        # self-verify vs the full-row packed new_logprobs; grads flow through the shared stream
        # (prefix grad once = sum of G repeats, identical math). Off/failed/unverified ->
        # full-row packed path runs as before.
        _pg_result = None
        _pg_use = False
        _pg_skip_pack = False
        _pg_num_gen = getattr(trainer, "num_generations", None)
        # Runtime gate; broad except -> engage False.
        if _pg_engage and _pg_funcs:
            try:
                _pg_build_layout, _pg_enabled_fn, _pg_verify_on, _pg_tol_ok, _PG_TOL_KILL = _pg_funcs
                # Exclusions: softcap models (gemma2) - the FlexAttention kernel skips
                # attn_logit_softcapping; hybrid SSM (FalconH1) and MoE (Qwen3-MoE) - their
                # decoders do not thread prefix_seg_info, so state would leak across suffixes.
                _pg_cfg = getattr(unwrapped_model, "config", None)
                _pg_engage = (
                    _pg_enabled_fn()
                    and pixel_values is None
                    and token_type_ids is None
                    and mm_token_type_ids is None
                    and _pg_num_gen is not None
                    and _pg_num_gen >= 2
                    and not getattr(_pg_cfg, "attn_logit_softcapping", None)
                    and not any(
                        getattr(_pg_cfg, _pg_a, None) is not None
                        for _pg_a in ("mamba_d_ssm", "mamba_d_state", "mamba_expand")
                    )
                    and not any(
                        getattr(_pg_cfg, _pg_a, None) is not None
                        for _pg_a in (
                            "num_experts", "num_experts_per_tok", "num_local_experts",
                            "n_routed_experts", "moe_intermediate_size",
                        )
                    )
                )
            except Exception:
                _pg_engage = False
        else:
            _pg_engage = False
        _pg_layout = None
        _pg_trusted = False   # signature already verified -> skip the full-row forward this step
        if _pg_engage:
            try:
                _pg_pad_id = trainer.processing_class.pad_token_id
                # Build the layout from the left-packed input_ids with the original left-pad
                # counts so the prefix/suffix split matches the packed path (_pack_cstart) and
                # the verify is apples-to-apples. Cap the PG span at any sliding window,
                # mirroring the packed _pack_sw guard.
                _pg_sw = getattr(getattr(unwrapped_model, "config", None), "sliding_window", None)
                if not (isinstance(_pg_sw, int) and _pg_sw > 0):
                    _pg_sw = None
                _pg_layout = _pg_build_layout(
                    input_ids, logits_to_keep, _pg_pad_id, _pg_num_gen, left_pad_tokens_per_prompt,
                    max_segment_cap = _pg_sw,
                )
                _pg_unsafe = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_unsafe", None)
                if _pg_unsafe is None:
                    _pg_unsafe = set()
                if _pg_layout is not None and _pg_layout.signature in _pg_unsafe:
                    _pg_layout = None
                elif _pg_layout is not None:
                    _pg_layout.W = logits_to_keep + max_left_pad
                    _pg_verified = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_verified", None)
                    # trust only if the verified envelope covers this batch's lengths
                    # (re-verify when T or the longest segment grows)
                    _pg_T = int(_pg_layout.flat_ids.shape[1])
                    _pg_maxseg = int(_pg_layout.position_ids.max()) + 1
                    _pg_env = (
                        _pg_verified.get(_pg_layout.signature)
                        if isinstance(_pg_verified, dict) else None
                    )
                    if (not _pg_verify_on()) or (
                        _pg_env is not None and _pg_T <= _pg_env[0] and _pg_maxseg <= _pg_env[1]
                    ):
                        _pg_trusted = True
                        _pg_skip_pack = True   # trusted shape -> skip the full-row forward
            except Exception as _pg_err:
                _pg_layout = None
                _pg_trusted = False
                _pg_skip_pack = False
                if isinstance(_pg_err, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
                if UNSLOTH_ENABLE_LOGGING:
                    print(f"[Unsloth] GRPO PrefixGrouper (grad) disabled (fell back to packed): {_pg_err!r}", flush = True)

        # ---- Sequence packing (default-on; disable with UNSLOTH_GRPO_SEQ_PACKING=0) ----
        # One varlen [1, sum L] block-diagonal forward replaces the padded [B, Lmax] loop: the exact per-row
        # result, and it fixes the padded path's left-pad RoPE error. Loss/gradients flow through it. Self-
        # verified against the per-row forward (shape/RoPE-aware, re-checked as T grows); falls back if a
        # backend ignores packed_seq_lengths. lm_head runs on completion positions only.
        new_logprobs = None
        _pack_result = None
        _pack_use = False
        _pack_T = None
        # True only while the packed forward and its scatter run, so an OOM elsewhere (setup, per-row verifier)
        # is not read as a length limit
        _pack_in_packed = False
        _pack_enabled = os.environ.get("UNSLOTH_GRPO_SEQ_PACKING", "1").lower() not in ("0", "false", "no", "off")
        _pack_ok = getattr(unwrapped_model, "_unsloth_seq_packing_grad_ok", None)
        if (_pack_enabled and not _pg_skip_pack and pixel_values is None
                and token_type_ids is None and mm_token_type_ids is None and _pack_ok is not False):
            try:
                _pack_pad_id = trainer.processing_class.pad_token_id
                _pack_keep = input_ids != _pack_pad_id
                _pack_lengths = _pack_keep.sum(dim = 1)
                _pack_lengths_cpu = _pack_lengths.tolist()                 # single GPU->CPU sync, reused below
                _pack_nz_cpu = [_n for _n in _pack_lengths_cpu if _n > 0]
                _pack_T = sum(_pack_nz_cpu)
                _pack_L = input_ids.shape[1]
                _pack_W = logits_to_keep + max_left_pad
                _pack_maxseg = max(_pack_nz_cpu) if _pack_nz_cpu else 0
                # sliding-window models lose the per-sequence local window in a packed stream
                _pack_sw = getattr(getattr(unwrapped_model, "config", None), "sliding_window", None)
                _pack_sw_ok = not (isinstance(_pack_sw, int) and _pack_sw > 0 and _pack_maxseg > _pack_sw)
                _pack_unsafe = getattr(unwrapped_model, "_unsloth_seq_packing_grad_unsafe_T", None)
                # skip the whole packed forward for a known-unsafe length region (a prior moderate mismatch or a
                # packed OOM), before the flatten and the active-row read, so a capped batch costs one sync
                _pack_capped = _pack_unsafe is not None and _pack_T >= _pack_unsafe
                if not _pack_capped:
                    _pack_flat_ids = input_ids[_pack_keep].unsqueeze(0)
                    _pack_active = int((completion_mask.sum(dim = 1) > 0).sum())
                if (not _pack_capped) and _pack_T >= 2 and len(_pack_nz_cpu) > 0 and _pack_sw_ok \
                        and (_pack_ok is True or _pack_active >= 2):
                    _pack_psl = torch.tensor(_pack_nz_cpu, dtype = torch.int32, device = input_ids.device)
                    # reset 0-based position_ids per segment
                    _pack_pos = (_pack_keep.cumsum(dim = 1) - 1)[_pack_keep].unsqueeze(0)
                    _pack_chunks = max(1, total_rows * multiplier)
                    _pack_nz_idx = _pack_keep.nonzero(as_tuple = False)            # [T, 2] = (row, col)
                    _pack_within = _pack_nz_idx[1:, 0] == _pack_nz_idx[:-1, 0]     # [T-1]
                    # completion start is per-row after left-packing: (L - logits_to_keep) minus that
                    # row's left-pad (matches create_completion_attention_mask exactly)
                    _pack_cstart = (_pack_L - logits_to_keep) - left_pad_tokens_per_prompt  # [rows]
                    _pack_ctgt = (_pack_nz_idx[1:, 1] >= _pack_cstart[_pack_nz_idx[1:, 0]]) & _pack_within
                    _pack_in_packed = True
                    with autocaster:
                        # use_cache=False: a KV cache silently disables varlen packing
                        _pack_hidden = unwrapped_model(
                            input_ids = _pack_flat_ids,
                            position_ids = _pack_pos,
                            packed_seq_lengths = _pack_psl,
                            use_cache = False,
                        ).logits
                        _refresh_lm_head()
                        # `.logits` carries hidden states only when the forward is the
                        # Unsloth generated one honouring UNSLOTH_RETURN_HIDDEN_STATES;
                        # otherwise it is real [T, vocab] logits and the lm_head matmul
                        # dies. Dispatch on width, as the padded path already does.
                        _pack_h   = _pack_hidden[0, :-1, :][_pack_ctgt].unsqueeze(0)
                        _pack_tid = _pack_flat_ids[0, 1:][_pack_ctgt].unsqueeze(0)
                        if _pack_h.shape[-1] == lm_head.shape[1]:
                            if _need_entropy:
                                _pack_sel, _pack_ent = _hidden_logps_and_entropies(_pack_h, _pack_tid, _pack_chunks)
                                _pack_sel, _pack_ent = _pack_sel[0], _pack_ent[0]
                            else:
                                _pack_sel = chunked_hidden_states_selective_log_softmax(
                                    _pack_h, lm_head, _pack_tid, _pack_chunks,
                                    logit_scale_multiply, logit_scale_divide, logit_softcapping, temperature,
                                    lm_head_bias = lm_head_bias,
                                )[0]
                        else:
                            # Raw logits: the forward already applied scale/softcap.
                            _pack_sel = chunked_selective_log_softmax(
                                _pack_h, _pack_tid,
                                temperature = temperature, chunks = _pack_chunks,
                            )[0]
                            if _need_entropy:
                                _pack_ent = _raw_logits_entropies(_pack_h)[0]
                    # GPT-OSS offload race guard (matches the padded loop)
                    device_synchronize()
                    # scatter each completion logprob back to its (row, col) so [:, -_pack_W:] matches padded
                    _pack_tgt = (_pack_nz_idx[1:, 0] * _pack_L + _pack_nz_idx[1:, 1])[_pack_ctgt]
                    _pack_result = torch.zeros(
                        total_rows * _pack_L, dtype = torch.float32, device = input_ids.device,
                    ).index_put((_pack_tgt,), _pack_sel.to(torch.float32)).view(total_rows, _pack_L)[:, -_pack_W:]
                    if _need_entropy:
                        _pack_entropies = torch.zeros(
                            total_rows * _pack_L, dtype = torch.float32, device = input_ids.device,
                        ).index_put((_pack_tgt,), _pack_ent.to(torch.float32)).view(total_rows, _pack_L)[:, -_pack_W:]
                    _pack_in_packed = False
                    # trust decision: re-verify when T or the longest segment grows past what was verified
                    # (a LongRoPE cache switch can change the result)
                    _pack_vT = int(getattr(unwrapped_model, "_unsloth_seq_packing_grad_verified_T", 0))
                    _pack_vS = int(getattr(unwrapped_model, "_unsloth_seq_packing_grad_verified_seg", 0))
                    _pack_force_verify = os.environ.get("UNSLOTH_GRPO_SEQ_PACKING_VERIFY", "0") == "1"
                    if (not _pack_force_verify) and _pack_ok is True and _pack_T <= _pack_vT and _pack_maxseg <= _pack_vS:
                        _pack_use = True                                           # already verified for this shape
                    else:
                        # verify against the per-row clean forward (exact ground truth; no grad, value check)
                        _pack_ref = torch.zeros_like(_pack_result)
                        with torch.no_grad(), autocaster:
                            for _pack_i in range(total_rows):
                                _pack_ni = _pack_lengths_cpu[_pack_i]
                                if _pack_ni < 2: continue
                                _pack_rmask = _pack_keep[_pack_i]
                                _pack_real = input_ids[_pack_i][_pack_rmask].unsqueeze(0)
                                _pack_rpos = torch.arange(_pack_ni, device = input_ids.device).unsqueeze(0)
                                _pack_rh = unwrapped_model(input_ids = _pack_real, position_ids = _pack_rpos, use_cache = False).logits
                                _refresh_lm_head()
                                # same width dispatch as the packed call above: this forward
                                # returns raw logits whenever that one did, and the first
                                # packed batch always lands here
                                if _pack_rh.shape[-1] == lm_head.shape[1]:
                                    _pack_rsel = chunked_hidden_states_selective_log_softmax(
                                        _pack_rh[:, :-1, :], lm_head, _pack_real[:, 1:], 1,
                                        logit_scale_multiply, logit_scale_divide, logit_softcapping, temperature,
                                        lm_head_bias = lm_head_bias,
                                    )[0]
                                else:
                                    _pack_rsel = chunked_selective_log_softmax(
                                        _pack_rh[:, :-1, :], _pack_real[:, 1:],
                                        temperature = temperature, chunks = 1,
                                    )[0]
                                _pack_rcols = _pack_rmask.nonzero(as_tuple = False).squeeze(1)[1:] - (_pack_L - _pack_W)
                                _pack_rkeep = _pack_rcols >= 0
                                _pack_ref[_pack_i, _pack_rcols[_pack_rkeep]] = _pack_rsel[_pack_rkeep].to(torch.float32)
                        device_synchronize()
                        # compare over the exact loss-mask region (same mask the loss uses; pure
                        # create_completion_attention_mask, before any tool_mask is applied)
                        _pack_cm = create_completion_attention_mask(
                            input_ids[:, -_pack_W:], left_pad_tokens_per_prompt, max_left_pad, _pack_pad_id
                        ).float()
                        _pack_diff = float(((_pack_result.detach() - _pack_ref).abs() * _pack_cm).max())
                        if UNSLOTH_ENABLE_LOGGING:
                            print(f"[Unsloth] GRPO seq-packing (grad) verify: T={_pack_T} maxseg={_pack_maxseg} packed-vs-perrow max|d|={_pack_diff:.4f}", flush = True)
                        # floor ~0.25 through different kernels; cross-sample contamination is >= 2.4
                        if _pack_diff < 7e-1:
                            unwrapped_model._unsloth_seq_packing_grad_ok = True
                            # only widen the trusted shape when >= 2 completion rows actually exercised
                            # cross-sample packing; a < 2 row pass proves nothing, so keep re-verifying
                            # larger shapes until a real multi-row batch clears them
                            if _pack_active >= 2:
                                unwrapped_model._unsloth_seq_packing_grad_verified_T = max(_pack_vT, _pack_T)
                                unwrapped_model._unsloth_seq_packing_grad_verified_seg = max(_pack_vS, _pack_maxseg)
                            _pack_ok = True
                            _pack_use = True
                        else:
                            _pack_use = False
                            if _pack_diff >= 1.5:
                                # large mismatch = contamination (attention ignores the packed mask, e.g.
                                # some MoE): disable packing for this model
                                unwrapped_model._unsloth_seq_packing_grad_ok = False
                            else:
                                # moderate mismatch -> likely a length boundary (LongRoPE): mark unsafe but
                                # keep packing for smaller shapes
                                unwrapped_model._unsloth_seq_packing_grad_unsafe_T = (
                                    _pack_T if _pack_unsafe is None else min(_pack_unsafe, _pack_T)
                                )
                            if UNSLOTH_ENABLE_LOGGING:
                                print(f"[Unsloth] GRPO seq-packing (grad) fell back at T={_pack_T} (diff={_pack_diff:.3f})", flush = True)
            except Exception as _pack_err:
                # any failure -> drop intermediates, use the padded loop, do not retry
                _pack_hidden = None
                _pack_sel = None
                _pack_result = None
                _pack_use = False
                if isinstance(_pack_err, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                if isinstance(_pack_err, torch.cuda.OutOfMemoryError) and _pack_in_packed and _pack_T is not None:
                    # an OOM in the packed forward means this length does not fit, not that packing is wrong
                    _pack_unsafe = getattr(unwrapped_model, "_unsloth_seq_packing_grad_unsafe_T", None)
                    _pack_unsafe = _pack_T if _pack_unsafe is None else min(_pack_unsafe, _pack_T)
                    unwrapped_model._unsloth_seq_packing_grad_unsafe_T = _pack_unsafe
                    if UNSLOTH_ENABLE_LOGGING:
                        print(f"[Unsloth] GRPO sequence-packing OOM at T={_pack_T}: packing capped below T={_pack_unsafe}, this batch uses the padded loop: {_pack_err!r}", flush = True)
                else:
                    unwrapped_model._unsloth_seq_packing_grad_ok = False
                    if UNSLOTH_ENABLE_LOGGING:
                        print(f"[Unsloth] GRPO sequence-packing disabled (fell back to padded): {_pack_err!r}", flush = True)
        # ---- PrefixGrouper resolution + first-use self-verify (grad) ----
        # Verify runs under no_grad, then a separate grad forward builds new_logprobs,
        # so no inference tensors are saved for backward.
        def _pg_grad_forward():
            _pg_chunks = max(1, total_rows * multiplier)
            with autocaster:
                _h = unwrapped_model(
                    input_ids = _pg_layout.flat_ids,
                    position_ids = _pg_layout.position_ids,
                    prefix_seg_info = _pg_layout.prefix_seg_info,
                    use_cache = False,
                ).logits
                _refresh_lm_head()
                # Same width dispatch as the packed path and compute_logprobs_chunk.
                # `.logits` carries hidden states only when the forward is the Unsloth
                # generated one honouring UNSLOTH_RETURN_HIDDEN_STATES; otherwise it is
                # real [T, vocab] logits. extract_logps always calls its helper as
                # (hidden, lm_head, ids, chunks, ...), so pass a raw-logits helper with
                # that same signature, which skips the lm_head matmul and the scale /
                # softcap the forward already applied.
                _pg_fn = chunked_hidden_states_selective_log_softmax
                if lm_head_bias is not None:
                    from functools import partial
                    _pg_fn = partial(_pg_fn, lm_head_bias = lm_head_bias)
                if _h.shape[-1] != lm_head.shape[1]:
                    def _pg_fn(_pg_h, _pg_lm, _pg_ids, _pg_n, _pg_lsm, _pg_lsd, _pg_lsc, _pg_t):
                        return chunked_selective_log_softmax(
                            _pg_h, _pg_ids, temperature = _pg_t, chunks = _pg_n,
                        )
                _pg_lp = _pg_layout.extract_logps(
                    _h, lm_head, _pg_fn,
                    _pg_chunks, logit_scale_multiply, logit_scale_divide,
                    logit_softcapping, temperature,
                )  # [total_rows, W] with grad
                # GPT-OSS offload race guard
                device_synchronize()
                return _pg_lp

        if _pg_layout is not None:
            # A verify-phase OOM (packed graph co-resident) does not prove PG alone cannot fit;
            # only an OOM after the packed graph is freed is worth marking unsafe.
            _pg_phase_verify = False
            try:
                if not _pg_trusted:
                    # first use: verify vs the packed new_logprobs. < tol_ok -> trust;
                    # >= TOL_KILL -> unsafe forever; borderline -> fall back this shape.
                    if _pack_use and _pack_result is not None:
                        _pg_phase_verify = True   # packed graph still co-resident
                        with torch.no_grad():
                            _pg_ref = _pg_grad_forward()
                        _pg_W2 = logits_to_keep + max_left_pad
                        _pg_cm = create_completion_attention_mask(
                            input_ids[:, -_pg_W2:], left_pad_tokens_per_prompt, max_left_pad,
                            trainer.processing_class.pad_token_id,
                        ).float()
                        _pg_a = _pg_ref[:, -_pg_W2:].float()
                        _pg_b = _pack_result.detach()[:, -_pg_W2:].float()
                        _pg_diff = float(((_pg_a - _pg_b).abs() * _pg_cm).max())
                        if UNSLOTH_ENABLE_LOGGING:
                            print(
                                f"[Unsloth] GRPO PrefixGrouper (grad) verify: sig={_pg_layout.signature} "
                                f"shared-prefix vs full-row-packed max|d|={_pg_diff:.4f}", flush = True,
                            )
                        if _pg_diff < _pg_tol_ok():
                            _pg_v = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_verified", None)
                            if not isinstance(_pg_v, dict):
                                _pg_v = {}
                            _pg_vT = int(_pg_layout.flat_ids.shape[1])
                            _pg_vS = int(_pg_layout.position_ids.max()) + 1
                            _pg_old = _pg_v.get(_pg_layout.signature, (0, 0))
                            _pg_v[_pg_layout.signature] = (
                                max(_pg_vT, _pg_old[0]), max(_pg_vS, _pg_old[1]),
                            )
                            unwrapped_model._unsloth_prefix_grouper_grad_verified = _pg_v
                            _pg_trusted = True
                        else:
                            _pg_u = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_unsafe", None)
                            if _pg_u is None:
                                _pg_u = set()
                            if _pg_diff >= _PG_TOL_KILL:
                                _pg_u.add(_pg_layout.signature)
                                unwrapped_model._unsloth_prefix_grouper_grad_unsafe = _pg_u
                            _pg_trusted = False
                    # else: no packed reference -> cannot verify -> fall back.
                if _pg_trusted:
                    # free the packed graph BEFORE the grad forward: holding both can OOM when
                    # PG alone would fit, and on PG failure the padded loop recomputes anyway.
                    _pack_hidden = _pack_sel = _pack_result = None
                    _pg_phase_verify = False   # packed freed: an OOM below is PG-alone
                    _pg_result = _pg_grad_forward()
                    _pg_use = True
            except Exception as _pg_err2:
                _pg_use = False
                os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
                # untrust this signature so the next batch runs the packed path again
                _pg_v = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_verified", None)
                if isinstance(_pg_v, dict):
                    _pg_v.pop(_pg_layout.signature, None)
                if isinstance(_pg_err2, torch.cuda.OutOfMemoryError):
                    # mark unsafe only for a PG-alone OOM (deterministic at these lengths);
                    # a verify-phase OOM (packed co-resident) proves nothing, just retry.
                    if not _pg_phase_verify:
                        _pg_u = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_unsafe", None)
                        if _pg_u is None:
                            _pg_u = set()
                        _pg_u.add(_pg_layout.signature)
                        unwrapped_model._unsloth_prefix_grouper_grad_unsafe = _pg_u
                    torch.cuda.empty_cache()
                if UNSLOTH_ENABLE_LOGGING:
                    print(f"[Unsloth] GRPO PrefixGrouper (grad) forward failed -> packed/padded fallback: {_pg_err2!r}", flush = True)

        if _pg_use and _pg_result is not None:
            new_logprobs = _pg_result            # PrefixGrouper verified -> skip the loop
            zipped_inputs = []
        elif _pack_use and _pack_result is not None:
            new_logprobs = _pack_result          # verified -> skip the loop
            if _need_entropy:
                all_entropies_list = [_pack_entropies]
            zipped_inputs = []
        else:
            # packing rejected/unused: drop the packed graph and its index buffers before the padded loop so
            # neither co-resides with it
            _pack_hidden = _pack_sel = _pack_result = None
            _pack_ent = _pack_entropies = None
            if _pack_T is not None:
                _pack_keep = _pack_lengths = _pack_flat_ids = _pack_pos = _pack_psl = None
                _pack_nz_idx = _pack_within = _pack_cstart = _pack_ctgt = _pack_tgt = None
                _pack_h = _pack_tid = _pack_ref = _pack_cm = None

        def to_device(tensor, device, non_blocking=True):
            if tensor is None: return None
            return tensor.to(device, non_blocking=non_blocking)

        def _offload_device_module(tensor_or_device):
            # Stream/Event module for the offload copy. torch.cuda is also the HIP
            # backend, so ROCm reports is_cuda and needs no branch of its own; XPU has
            # its own namespace, matching gradient_checkpointing.py. Anything else
            # (CPU, MPS, ...) returns None and takes the pageable copy.
            device = getattr(tensor_or_device, "device", tensor_or_device)
            if device.type == "cuda": return torch.cuda
            if device.type == "xpu": return getattr(torch, "xpu", None)
            return None

        class Unsloth_Offloaded_Log_Softmax(torch.autograd.Function):
            """Manual gradient checkpointing / CPU offloading for log softmax."""
            @staticmethod
            def forward(ctx, hidden_states, lm_head, index, chunks,
                        logit_scale_multiply, logit_scale_divide,
                        logit_softcapping, temperature, lm_head_bias = None):
                # Detach so we don't keep the graph (and extra memory) on CPU.
                detached_hidden_states = hidden_states.detach().contiguous()
                ctx.device = hidden_states.device
                ctx.copy_event = None
                # Backward runs outside autocast; recompute must match forward (torch.utils.checkpoint does the same).
                ctx.autocast_kwargs = dict(
                    device_type = lm_head.device.type,
                    enabled = torch.is_autocast_enabled(lm_head.device.type),
                    dtype = torch.get_autocast_dtype(lm_head.device.type),
                    cache_enabled = torch.is_autocast_cache_enabled(),
                )

                # Always offload: this path only runs when the caller is already memory bound
                # (long completions / large batches), so the win is overlapping the copy.
                saved_hidden_states = None
                device_module = _offload_device_module(detached_hidden_states)
                if device_module is not None:
                    # Async D2H on a side stream; backward MUST wait on copy_event before
                    # the H2D reload or it races the copy.
                    try:
                        pinned_buffer = torch.empty_like(detached_hidden_states, device = "cpu", pin_memory = True)
                        if pinned_buffer is not None:
                            current_stream = device_module.current_stream(detached_hidden_states.device)
                            copy_stream = device_module.Stream(device = detached_hidden_states.device)
                            copy_stream.wait_stream(current_stream)
                            with device_module.stream(copy_stream):
                                pinned_buffer.copy_(detached_hidden_states, non_blocking = True)
                            # Keeps the GPU storage alive until the side-stream copy finishes.
                            detached_hidden_states.record_stream(copy_stream)
                            copy_event = device_module.Event()
                            copy_event.record(copy_stream)
                            saved_hidden_states = pinned_buffer
                            ctx.copy_event = copy_event
                    except (RuntimeError, OSError, AttributeError):
                        # Any accelerator that cannot do pinned side-stream copies falls
                        # back below; correctness never depends on this path.
                        saved_hidden_states = None
                        ctx.copy_event = None
                if saved_hidden_states is None:
                    # No accelerator, or the async copy is unavailable: pageable copy.
                    saved_hidden_states = detached_hidden_states.to("cpu", non_blocking = True)
                ctx.saved_hidden_states = saved_hidden_states
                # Drop the clone before the log-softmax below. hidden_states is usually a
                # [:, :-1, :] slice, so .contiguous() allocated a full copy; holding the
                # reference across the forward would keep it resident alongside the chunk
                # logits. record_stream still blocks reuse until the D2H lands, so the
                # allocator reclaims it mid-compute rather than at the end of forward.
                del detached_hidden_states

                ctx.lm_head = lm_head
                ctx.lm_head_requires_grad = lm_head.requires_grad
                ctx.lm_head_bias = lm_head_bias
                ctx.bias_requires_grad = lm_head_bias is not None and lm_head_bias.requires_grad
                ctx.index = index
                ctx.args = (chunks, logit_scale_multiply, logit_scale_divide, logit_softcapping, temperature)

                bias_kwargs = {} if lm_head_bias is None else {"lm_head_bias": lm_head_bias}
                with torch.no_grad():
                    output = chunked_hidden_states_selective_log_softmax(
                        hidden_states, lm_head, index, *ctx.args, **bias_kwargs,
                    )

                return output

            @staticmethod
            def backward(ctx, grad_output):
                if ctx.copy_event is not None:
                    # The offload copy must land before the H2D reload.
                    device_module = _offload_device_module(ctx.device)
                    ctx.copy_event.wait(device_module.current_stream(ctx.device))
                hidden_states = to_device(ctx.saved_hidden_states, ctx.device)
                hidden_states.requires_grad_(True)

                lm_head = ctx.lm_head
                if ctx.lm_head_requires_grad:
                    # Recompute against a private leaf. A Tensor.register_hook on the real
                    # lm_head fires for tensors named in autograd.grad's inputs, so reusing
                    # it here would run a user's grad mask / scaler once on this local
                    # gradient and again when the returned gradient reaches lm_head.
                    lm_head = lm_head.detach().requires_grad_(True)
                lm_head_bias = ctx.lm_head_bias
                if ctx.bias_requires_grad:
                    lm_head_bias = lm_head_bias.detach().requires_grad_(True)
                index = ctx.index

                bias_kwargs = {} if lm_head_bias is None else {"lm_head_bias": lm_head_bias}
                with torch.enable_grad(), torch.autocast(**ctx.autocast_kwargs):
                    output = chunked_hidden_states_selective_log_softmax(
                        hidden_states, lm_head, index, *ctx.args, **bias_kwargs,
                    )

                # autograd.grad, not backward: backward writes into leaf .grad, which the
                # outer AccumulateGrad would then double-count.
                inputs = [hidden_states]
                if ctx.lm_head_requires_grad:
                    inputs.append(lm_head)
                if ctx.bias_requires_grad:
                    inputs.append(lm_head_bias)
                grad_inputs = torch.autograd.grad(output, inputs, grad_output)

                return (
                    grad_inputs[0],
                    grad_inputs[1] if ctx.lm_head_requires_grad else None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    grad_inputs[-1] if ctx.bias_requires_grad else None,
                )

        def efficient_log_softmax(hidden_states, lm_head, index, chunks=32,
                                logit_scale_multiply=0.0, logit_scale_divide=0.0,
                                logit_softcapping=0.0, temperature=1, batch_size=8,
                                lm_head_bias=None):
            if (index.shape[1] <= 1024 and batch_size <= 8) or batch_size==1:
                # Normal path is faster / saves a GB under these conditions.
                return chunked_hidden_states_selective_log_softmax(
                    hidden_states,
                    lm_head,
                    index,
                    chunks,
                    logit_scale_multiply,
                    logit_scale_divide,
                    logit_softcapping,
                    temperature,
                    lm_head_bias = lm_head_bias,
                )
            else:
                return Unsloth_Offloaded_Log_Softmax.apply(
                    hidden_states, lm_head, index, chunks,
                    logit_scale_multiply, logit_scale_divide,
                    logit_softcapping, temperature, lm_head_bias,
                )

        def compute_logprobs_chunk(new_hidden_states_chunk, completion_ids, input_ids_chunk):
            _refresh_lm_head()
            # Hidden states -> lm_head matmul path; raw logits -> skip matmul and
            # skip scale/softcap (model forward already applied them).
            chunks = input_ids_chunk.shape[0] * multiplier
            if new_hidden_states_chunk.shape[-1] == lm_head.shape[1]:
                if _need_entropy:
                    logps, entropies = _hidden_logps_and_entropies(new_hidden_states_chunk, completion_ids, chunks)
                    all_entropies_list.append(entropies)
                    return logps
                return efficient_log_softmax(
                    new_hidden_states_chunk,
                    lm_head,
                    completion_ids,
                    chunks = chunks,
                    logit_scale_multiply = logit_scale_multiply,
                    logit_scale_divide = logit_scale_divide,
                    logit_softcapping = logit_softcapping,
                    temperature = temperature,
                    batch_size = B,
                    lm_head_bias = lm_head_bias,
                )
            if _need_entropy:
                all_entropies_list.append(_raw_logits_entropies(new_hidden_states_chunk))
            return chunked_selective_log_softmax(
                new_hidden_states_chunk,
                completion_ids,
                temperature = temperature,
                chunks = chunks,
            )


        for (
            input_ids_chunk,
            attention_mask_chunk,
            vision_chunk,
            completion_ids
        ) in zipped_inputs:
                with autocaster:
                    if pixel_values is None:
                        new_hidden_states_chunk = unwrapped_model(
                            input_ids = input_ids_chunk,
                            attention_mask = attention_mask_chunk,
                            **vision_chunk,
                        ).logits

                        new_hidden_states_chunk = new_hidden_states_chunk[:, -(logits_to_keep + max_left_pad + 1): , :]
                        new_hidden_states_chunk = new_hidden_states_chunk[:, :-1, :]
                        logprobs_chunk = compute_logprobs_chunk(new_hidden_states_chunk, completion_ids, input_ids_chunk)
                    else:
                        new_hidden_states_chunk = unwrapped_model(
                            input_ids = input_ids_chunk,
                            attention_mask = attention_mask_chunk,
                            logits_to_keep = logits_to_keep + 1,
                            **vision_chunk,
                        ).logits

                        new_hidden_states_chunk = new_hidden_states_chunk[:, :-1, :]
                        logprobs_chunk = compute_logprobs_chunk(new_hidden_states_chunk, completion_ids, input_ids_chunk)
                    # Avoids race conditions with GPT OSS offload_embbed=True; no measurable slowdown.
                    device_synchronize()
                all_logprobs_list.append(logprobs_chunk)

        if new_logprobs is None:
            # padded fallback (packing disabled / unsupported / not verified for this length)
            new_logprobs = torch.cat(all_logprobs_list, dim=0)

        if kept_completion_rows is not None:
            completion_mask = completion_mask * kept_completion_rows.to(
                device = completion_mask.device, dtype = completion_mask.dtype,
            )

        entropy_mask = None
        if _need_entropy:
            new_entropies = torch.cat(all_entropies_list, dim = 0)
            if _entropy_mask_on:
                # TRL's own method, so its quantile population, ties and empty-batch rule match per version
                # (multi-process TRL < 0.24 takes the padded gather so an empty rank cannot hang the rest).
                entropy_mask = _grpo_high_entropy_mask(
                    trainer, new_entropies.detach(), completion_mask, 1 - _entropy_quantile,
                    sample_weight = kwargs.get("sample_weight", None),
                )
                kwargs["entropy_mask"] = entropy_mask

        # DeepSpeed (no scaler) only: undo training_step's 1/GAS, which transformers >= 5.19 skips when
        # loss_is_scaled_for_ga (TRL >= 1.15 GRPO) and older ones when compute_loss_func is set.
        upstream_scale = None
        if trainer.accelerator.scaler is None and getattr(trainer, "is_deepspeed_enabled", False):
            import transformers as _unsloth_transformers
            loss_is_scaled_for_ga = None
            if hasattr(_unsloth_transformers.Trainer, "loss_is_scaled_for_ga"):
                loss_is_scaled_for_ga = getattr(trainer, "loss_is_scaled_for_ga", None)
            if loss_is_scaled_for_ga is None:
                loss_is_scaled_for_ga = getattr(trainer, "compute_loss_func", None) is not None
            upstream_scale = 1.0
            if not loss_is_scaled_for_ga:
                upstream_scale = float(kwargs.get("current_gradient_accumulation_steps", 1))

        _rollout_unscored = None
        if _rollout_logps_as_old:
            old_logps, _rollout_unscored = grpo_rollout_old_fill_unscored(old_logps, new_logprobs)
            # Only on this path: grpo_compute_loss clamps pi_theta / pi_rollout's log to [-20, 20] like verl.
            kwargs["rollout_logps_as_old"] = True
        # The rollout_rs keep mask grpo_compute_loss applies, for the IS metric weights and the entropy bonus.
        _rs_keep = None
        if kwargs.get("rollout_rs", None) is not None and old_logps is not None and old_logps.shape == completion_mask.shape:
            _rs_log_ratio = None
            if _rollout_logps_as_old:
                _rs_log_ratio = new_logprobs.detach() - old_logps
            elif trainer.use_vllm and sampling_per_token_logps is not None and kwargs.get("vllm_importance_sampling_correction", False):
                _rs_log_ratio = old_logps - sampling_per_token_logps
            if _rs_log_ratio is not None:
                from unsloth_zoo.rl_replacements import grpo_rollout_rs_keep as _grpo_rollout_rs_keep
                _rs_keep = _grpo_rollout_rs_keep(_rs_log_ratio, completion_mask, kwargs["rollout_rs"])
        with autocaster:
            loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1 = UnslothEfficientGRPO.apply(
                new_logprobs,
                old_logps,
                ref_logps,
                sampling_per_token_logps,
                lm_head,
                completion_input_ids,
                completion_mask,
                advantages,
                trainer.beta,
                trainer.accelerator.scaler,
                1,
                kwargs,
                upstream_scale,
            )
        _rollout_stats_args = None
        if _rollout_logps_as_old:
            # Unclamped log(pi_theta / pi_rollout) for the KL diagnostics; the loss ratio is clamped to [-20, 20].
            _raw_log_ratio = new_logprobs.detach().to(torch.float64) - old_logps.detach().to(torch.float64)
            if _raw_log_ratio.shape != completion_mask.shape:
                _raw_log_ratio = torch.log(coef_1.detach().to(torch.float64))
            delta, flat_is_ratio = grpo_rollout_as_old_ratio_metrics(coef_1, completion_mask, _raw_log_ratio)
            if _rs_keep is not None and flat_is_ratio is not None:
                # Rejected tokens get IS weight 0, as on the vLLM IS path: rejected_frac counts them, ESS drops them.
                flat_is_ratio = torch.where(_rs_keep, flat_is_ratio, torch.zeros_like(flat_is_ratio))
            if _rollout_metrics_on and flat_is_ratio is not None:
                _log_ratio = torch.nan_to_num(_raw_log_ratio, nan = 0.0, posinf = 670.0, neginf = -670.0)
                _log_ratio = torch.where(completion_mask.to(torch.bool), _log_ratio, torch.zeros_like(_log_ratio))
                if _rollout_unscored is not None:
                    # Unscored tokens carry no mismatch information: NaN leaves them out of the sums.
                    _log_ratio = torch.where(_rollout_unscored, torch.full_like(_log_ratio, float("nan")), _log_ratio)
                # An overflowed ratio is capped (finite squares) so ESS tends to its 1 / N limit, not nan.
                _ess_weights = torch.clamp(flat_is_ratio.to(torch.float64), max = 1e100)
                _rollout_stats_args = (_log_ratio, torch.zeros_like(_log_ratio), completion_mask, _ess_weights)
        elif (
            _rollout_metrics_on and old_logps is not None and sampling_per_token_logps is not None
            and flat_is_ratio is not None and flat_is_ratio.numel() > 0
        ):
            _rollout_stats_args = (old_logps, sampling_per_token_logps, completion_mask, flat_is_ratio)
        if _rollout_stats_args is not None:
            from unsloth_zoo.rl_replacements import (
                grpo_rollout_mismatch_stats as _grpo_rollout_mismatch_stats,
                grpo_rollout_mismatch_metrics as _grpo_rollout_mismatch_metrics,
            )
            _rollout_stats = _grpo_rollout_mismatch_stats(*_rollout_stats_args)
            _rollout_stats = trainer.accelerator.gather(_rollout_stats.reshape(1, -1)).reshape(-1, _rollout_stats.numel())
            _rollout_stats = torch.cat([_rollout_stats[:, :3].sum(0), _rollout_stats[:, 3:4].amax(0), _rollout_stats[:, 4:].sum(0)])
            _metrics = trainer._metrics
            _bucket = "train" if _training else "eval"
            if "train" in _metrics:
                _metrics = _metrics[_bucket]
            # Pool the sums over every micro-batch since the last log instead of averaging
            # per-micro-batch ratios: TRL's log() clears _metrics, which restarts the window.
            _rollout_acc = getattr(trainer, "_unsloth_rollout_mismatch_stats", None)
            if not isinstance(_rollout_acc, dict):
                _rollout_acc = {}
                trainer._unsloth_rollout_mismatch_stats = _rollout_acc
            _prev = _rollout_acc.get(_bucket)
            if _prev is not None and "sampling/rollout_kl_k1/mean" in _metrics:
                _prev = _prev.to(_rollout_stats.device)
                _rollout_stats = torch.cat([
                    _prev[:3] + _rollout_stats[:3],
                    torch.maximum(_prev[3:4], _rollout_stats[3:4]),
                    _prev[4:] + _rollout_stats[4:],
                ])
            _rollout_acc[_bucket] = _rollout_stats.detach()
            for _name, _value in _grpo_rollout_mismatch_metrics(_rollout_stats).items():
                _metrics[_name] = [_value]
        if _entropy_bonus_on:
            _entropy_weights = _grpo_entropy_bonus_weights(
                completion_mask, entropy_mask, _entropy_coef, kwargs, training = _training,
            )
            if _rs_keep is not None:
                # Rejected tokens leave the entropy bonus too (verl drops them from response_mask);
                # its normaliser keeps them, like the policy term's.
                _entropy_weights = torch.where(_rs_keep.to(_entropy_weights.device), _entropy_weights, 0.0)
            _scaler = trainer.accelerator.scaler
            loss = loss + _UnslothGRPOEntropyBonus.apply(
                new_entropies, _entropy_weights.to(new_entropies.device),
                _scaler.get_scale() if _scaler is not None else 1.0, upstream_scale,
            )
    finally:
        if _unsloth_prior_hidden_states is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = _unsloth_prior_hidden_states

    # Force logits (not hidden states) again or output is gibberish.
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"

    return loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1, completion_mask
pass
RL_REPLACEMENTS["grpo_accumulated_loss"] = grpo_accumulated_loss

from .dataset_utils import sft_prepare_dataset
RL_REPLACEMENTS["sft_prepare_dataset"] = sft_prepare_dataset


def _distillation_project_logits(
    hidden_states,
    lm_head,
    lm_head_bias = None,
    logit_scale: float = 1.0,
    logit_softcapping: float = 0.0,
):
    """``hidden_states @ lm_head.T`` plus the per model logit post processing.

    ``logit_scale`` is Cohere's multiplier (Muse Glimmer's ``output_multiplier``)
    and ``logit_softcapping`` is Gemma's, both read off the model's own config so a
    chunk matches that model's full forward. Co-located on the head's device
    because an accelerate dispatch can separate the head from the hidden states.
    """
    hidden_states = hidden_states.to(device = lm_head.device, dtype = lm_head.dtype)
    logits = (hidden_states @ lm_head.t()).float()
    if lm_head_bias is not None:
        logits = logits + lm_head_bias.float()
    if logit_scale != 1.0:
        logits = logits * logit_scale
    if logit_softcapping is not None and logit_softcapping != 0.0:
        logits = logit_softcapping * torch.tanh(logits / logit_softcapping)
    return logits


def _distillation_generalized_jsd(student_log_probs, teacher_log_probs, beta: float):
    """``beta`` = 0 forward KL, 1 reverse KL, anything between generalized JSD.

    The endpoints need their own branches: substituted into the interior
    expression they collapse the mixture onto one distribution and degenerate to
    exactly zero. ``F.kl_div(input, target)`` is ``target * (log target - input)``,
    which is why the arguments read swapped against the paper.
    """
    if beta == 0.0:
        return torch.nn.functional.kl_div(
            student_log_probs, teacher_log_probs, reduction = "none", log_target = True,
        )
    if beta == 1.0:
        return torch.nn.functional.kl_div(
            teacher_log_probs, student_log_probs, reduction = "none", log_target = True,
        )
    beta_t = torch.tensor(beta, dtype = student_log_probs.dtype, device = student_log_probs.device)
    mixture_log_probs = torch.logsumexp(
        torch.stack([
            student_log_probs + torch.log1p(-beta_t),
            teacher_log_probs + torch.log(beta_t),
        ]),
        dim = 0,
    )
    kl_teacher = torch.nn.functional.kl_div(
        mixture_log_probs, teacher_log_probs, reduction = "none", log_target = True,
    )
    kl_student = torch.nn.functional.kl_div(
        mixture_log_probs, student_log_probs, reduction = "none", log_target = True,
    )
    return beta_t * kl_teacher + (1 - beta_t) * kl_student


def _distillation_jsd_chunk(
    student_hidden_states, student_lm_head, student_lm_head_bias,
    teacher_hidden_states, teacher_lm_head, teacher_lm_head_bias,
    valid, divisor,
    student_logit_scale, student_final_logit_softcapping,
    teacher_logit_scale, teacher_final_logit_softcapping,
    beta, temperature, grad_scale,
):
    """One normalized chunk, scaled before ``torch.func.grad_and_value``. The positional order IS the argnums order: student hidden states, then head, then bias. The unscaled loss and entropy are auxiliary outputs."""
    student_logits = _distillation_project_logits(
        student_hidden_states, student_lm_head, student_lm_head_bias,
        student_logit_scale, student_final_logit_softcapping,
    )
    # The teacher is a fixed target. Inert inside the grad_and_value transform, which does not differentiate these arguments; kept for direct callers outside it.
    with torch.no_grad():
        teacher_logits = _distillation_project_logits(
            teacher_hidden_states, teacher_lm_head, teacher_lm_head_bias,
            teacher_logit_scale, teacher_final_logit_softcapping,
        )

    # Distillation temperature, NOT the sampling one, and no T**2 rescale; applied after each model's own scaling and softcapping.
    if temperature != 1.0:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

    student_log_probs = torch.nn.functional.log_softmax(student_logits, dim = -1)
    teacher_log_probs = torch.nn.functional.log_softmax(teacher_logits, dim = -1)

    # Reduce on the STUDENT head's device. Each projection above left its result
    # on its own head's device, which under an accelerate dispatch that splits
    # the two models is not the same device, and the divergence then raises. The
    # student's side is the right target because the gradient accumulates there.
    # Both moves are no-ops when the heads are co-located, which is every
    # single-device run.
    if teacher_log_probs.device != student_log_probs.device:
        teacher_log_probs = teacher_log_probs.to(student_log_probs.device)
    if valid.device != student_log_probs.device:
        valid = valid.to(student_log_probs.device)

    jsd = _distillation_generalized_jsd(student_log_probs, teacher_log_probs, beta)

    # The final chunk's tail holds positions packed out of the valid prefix.
    per_token_jsd = jsd.sum(dim = -1) * valid
    per_token_entropy = -(student_log_probs.exp() * student_log_probs).sum(dim = -1) * valid
    loss = per_token_jsd.sum() / divisor
    return loss * grad_scale, (loss, per_token_entropy.sum())
pass


# grad_and_value_impl is missing from Dynamo's trace rules where grad_impl is present, which surfaces as GB0149 "Unsupported functorch tracing attempt". Idempotent, and fused_losses.cross_entropy_loss makes the same registration.
try:
    from torch._dynamo.trace_rules import manual_torch_name_rule_map as _trace_map
    from torch._dynamo.variables.higher_order_ops import FunctorchHigherOrderVariable as _FHOV
    _key = "torch._functorch.eager_transforms.grad_and_value_impl"
    if _key not in _trace_map:
        _trace_map[_key] = _FHOV
        torch._dynamo.trace_rules.get_torch_obj_rule_map.cache_clear()
except Exception:
    pass

# Keyed by argnums, so Dynamo traces once per gradient shape rather than per call.
_DISTILLATION_JSD_GRAD_FNS = {}


def _distillation_jsd_grad_fn(argnums):
    """``grad_and_value`` over one chunk, compiled the way the other chunked
    losses in this repo are.

    ``_maybe_compile`` reads the compile-disable flag for us and, under fullgraph,
    routes through ``torch_compile_with_fallback`` so cache exhaustion degrades to
    eager instead of raising mid step.

    Memoized through a module dict rather than ``functools.lru_cache``: the
    compiler copies a function's decorators verbatim into the generated trainer
    modules and emits no ``import functools`` for them, so the decorator form
    becomes a ``NameError`` in the generated module the moment anything splices
    this in (tests/test_generated_trainer_is_installed.py).
    """
    cached = _DISTILLATION_JSD_GRAD_FNS.get(argnums)
    if cached is not None:
        return cached
    eager = torch.func.grad_and_value(
        _distillation_jsd_chunk, argnums = argnums, has_aux = True,
    )
    # dynamic = False, unlike the other compiled regions here, because the caller
    # pads every chunk to exactly chunk_size: the traced shapes never move, and
    # asking for dynamic shapes anyway measured 18.0 ms against 11.4 ms on a B200
    # at a 151936 token vocabulary. Only chunk_size, the hidden width and the
    # vocabulary reach the trace, and none of those change within a run.
    compiled = _maybe_compile(
        dynamic = False, fullgraph = True, options = torch_compile_options,
    )(eager)
    _DISTILLATION_JSD_GRAD_FNS[argnums] = compiled
    return compiled
pass


# Keep ``lru_cache``'s one useful method so callers that need a fresh compile
# decision, notably anything toggling the compile-disable flag mid process, do
# not have to know the memo is a dict now.
_distillation_jsd_grad_fn.cache_clear = _DISTILLATION_JSD_GRAD_FNS.clear


class _UnslothDistillationJSD(torch.autograd.Function):
    # All Unsloth Zoo code licensed under AGPL3
    """Chunked generalized JSD that differentiates in the forward pass.

    Gradient checkpointing would be the shorter way to bound the logits, but it
    recomputes each chunk's projection during the backward, and that projection is
    the expensive part. Measured on a B200 at a 151936 token vocabulary, that
    version came out slower than materializing the full logits it was meant to
    replace, 56.0 ms against 29.5 ms, so it bought memory with throughput.
    Accumulating the gradient while the chunk is already live runs the body once
    and wins on both, which is the same reason ``UnslothEfficientGRPO`` and the
    fused cross entropy loss are built this way.
    """

    @staticmethod
    def forward(
        ctx,
        student_hidden_states, teacher_hidden_states,
        student_lm_head, teacher_lm_head,
        completion_mask, beta, chunk_size, num_items_in_batch,
        student_lm_head_bias, teacher_lm_head_bias,
        student_logit_scale, teacher_logit_scale,
        student_final_logit_softcapping, teacher_final_logit_softcapping,
        temperature, scaling,
    ):
        flat_student = student_hidden_states.reshape(-1, student_hidden_states.shape[-1])
        flat_teacher = teacher_hidden_states.reshape(-1, teacher_hidden_states.shape[-1])
        # The mask decides the packing order for BOTH models, so it has to answer
        # on the student's device; a dispatched model leaves it wherever the
        # batch was built, and indexing rejects an index from another device.
        valid = completion_mask.reshape(-1).to(flat_student.device) != 0
        n_valid = valid.sum()

        # Pack the valid positions to the front so the masked ones form whole
        # trailing chunks that can be skipped. argsort on the mask is a static
        # shape op, unlike flat_student[valid], whose output shape is data
        # dependent and cannot be traced.
        order = valid.to(torch.int8).argsort(descending = True, stable = True)
        flat_student = flat_student[order]
        # The teacher can sit on another device entirely; index it with its own
        # copy of the order rather than dragging its hidden states across.
        flat_teacher = flat_teacher[order.to(flat_teacher.device)]
        sorted_valid = valid[order]

        # At least one chunk always runs: a fully masked batch still has to reach
        # every trainable parameter, or backward and the gradient sync that
        # follows it hang.
        n_padded = int(max(1, -(-int(n_valid) // chunk_size)) * chunk_size)
        # Pad the packed rows out to a whole number of chunks so every chunk is
        # exactly chunk_size and the compiled region sees one static shape. The
        # tail would otherwise be short whenever batch * seq is not a multiple of
        # chunk_size, and a second shape means a second Dynamo trace on the first
        # step plus another on every sequence length after that. The padding rows
        # carry valid = 0, so they contribute nothing to either the loss or the
        # gradient; at most chunk_size extra rows are ever added.
        n_rows = flat_student.shape[0]
        if n_rows < n_padded:
            pad = n_padded - n_rows
            flat_student = torch.cat([flat_student, flat_student.new_zeros(pad, flat_student.shape[-1])])
            flat_teacher = torch.cat([flat_teacher, flat_teacher.new_zeros(pad, flat_teacher.shape[-1])])
            sorted_valid = torch.cat([sorted_valid, sorted_valid.new_zeros(pad)])

        if num_items_in_batch is None:
            # Clamped for the same reason a chunk always runs below: a fully
            # masked batch has to reduce to a finite zero rather than 0 / 0.
            divisor = n_valid.clamp(min = 1).to(torch.float32)
        else:
            if isinstance(num_items_in_batch, torch.Tensor):
                divisor = num_items_in_batch.to(device = flat_student.device, dtype = torch.float32)
            else:
                divisor = torch.tensor(
                    float(num_items_in_batch), device = flat_student.device, dtype = torch.float32,
                )
            # A supplied count is legitimately zero when the whole global batch is
            # masked, and 0 / 0 is a NaN loss AND NaN gradients, which poison the
            # optimizer state for the rest of the run rather than costing one step.
            # Clamp it exactly as the implicit count above is clamped: the
            # numerator is zero in that case, so the quotient is still zero.
            divisor = divisor.clamp(min = 1.0)

        head_needs_grad = student_lm_head.requires_grad
        bias_needs_grad = student_lm_head_bias is not None and student_lm_head_bias.requires_grad
        # Build the argnums from what actually needs a gradient. A frozen head with
        # a trainable bias is a real configuration (bias-only tuning, or LoRA with
        # bias = "all"), and enumerating the combinations by hand had missed it.
        argnums = (0,) + ((1,) if head_needs_grad else ()) + ((2,) if bias_needs_grad else ())
        # grads[i] answers to argnums[i], so read the positions rather than
        # assuming the head is always index 1 and the bias always index 2.
        head_position = argnums.index(1) if head_needs_grad else None
        bias_position = argnums.index(2) if bias_needs_grad else None
        grad_fn = _distillation_jsd_grad_fn(argnums)

        grad_hidden = torch.zeros_like(flat_student)
        grad_head = torch.zeros_like(student_lm_head) if head_needs_grad else None
        grad_bias = torch.zeros_like(student_lm_head_bias) if bias_needs_grad else None

        loss = flat_student.new_zeros((), dtype = torch.float32)
        entropy_sum = flat_student.new_zeros((), dtype = torch.float32)
        # Scale before differentiation: an unscaled saved FP16 gradient underflows before GradScaler's backward.
        grad_scale = loss.new_tensor(scaling)
        ctx.scaling = scaling

        for start in range(0, n_padded, chunk_size):
            stop = start + chunk_size
            grads, (_, (chunk_loss, chunk_entropy)) = grad_fn(
                flat_student[start : stop], student_lm_head, student_lm_head_bias,
                flat_teacher[start : stop], teacher_lm_head, teacher_lm_head_bias,
                sorted_valid[start : stop].to(flat_student.dtype), divisor,
                student_logit_scale, student_final_logit_softcapping,
                teacher_logit_scale, teacher_final_logit_softcapping,
                beta, temperature, grad_scale,
            )
            grad_hidden[start : stop] = grads[0]
            if head_position is not None:
                grad_head.add_(grads[head_position])
            if bias_position is not None:
                grad_bias.add_(grads[bias_position])
            loss = loss + chunk_loss
            entropy_sum = entropy_sum + chunk_entropy
        pass

        # Undo the packing so the caller gets its own row order back, dropping the
        # padding rows first: they have no counterpart in the caller's tensor.
        grad_hidden = grad_hidden[: n_rows]
        unsorted = torch.zeros_like(grad_hidden)
        unsorted[order] = grad_hidden

        ctx.save_for_backward(
            unsorted.reshape(student_hidden_states.shape),
            grad_head if grad_head is not None else flat_student.new_zeros(0),
            grad_bias if grad_bias is not None else flat_student.new_zeros(0),
        )
        ctx.mark_non_differentiable(entropy_sum, n_valid)
        return loss, entropy_sum, n_valid

    @staticmethod
    def backward(ctx, grad_output, _grad_entropy, _grad_n_valid):
        grad_hidden, grad_head, grad_bias = ctx.saved_tensors
        grad_output = grad_output / ctx.scaling
        # Saved as empty rather than None: save_for_backward rejects None, and an
        # empty tensor is the cheap way to say "this input was not trainable".
        return (
            grad_output * grad_hidden,
            None,
            grad_output * grad_head if grad_head.numel() != 0 else None,
            None, None, None, None, None,
            grad_output * grad_bias if grad_bias.numel() != 0 else None,
            None, None, None, None, None, None, None,
        )
pass


def distillation_chunked_jsd(
    student_hidden_states,
    teacher_hidden_states,
    student_lm_head,
    teacher_lm_head,
    completion_mask,
    beta: float = 0.5,
    chunk_size: int = 256,
    num_items_in_batch = None,
    student_lm_head_bias = None,
    teacher_lm_head_bias = None,
    student_logit_scale: float = 1.0,
    teacher_logit_scale: float = 1.0,
    student_final_logit_softcapping: float = 0.0,
    teacher_final_logit_softcapping: float = 0.0,
    temperature: float = 1.0,
    scaler = None,
):
    """Memory efficient generalized JSD between a student and a frozen teacher.

    Both models are projected ``chunk_size`` positions at a time with the gradient
    accumulated while each chunk is live, so peak logit memory is
    ``2 * chunk_size * vocab`` rather than ``2 * batch * seq * vocab``. A frozen
    student head, the ordinary LoRA case, allocates no dense ``(vocab, hidden)``
    gradient either. Student and teacher must share a vocabulary but not a hidden
    width: each is projected through its own head.

    The objective is the one TRL's ``DistillationTrainer`` settled on: pure soft
    loss with no hard cross entropy term, no implicit ``T**2`` gradient rescale,
    and the mixture ``M = (1 - beta) * p_student + beta * p_teacher``. Other
    implementations differ on all three, so the tests check this against a dense
    reference rather than against any one of them.

    Args:
        student_hidden_states: ``(batch, seq, hidden)`` before the student's head.
        teacher_hidden_states: ``(batch, seq, hidden)`` for the same positions.
        student_lm_head: ``(vocab, hidden)`` student output head weight.
        teacher_lm_head: ``(vocab, hidden)`` teacher output head weight.
        completion_mask: ``(batch, seq)``, non zero where a position is trained on.
        beta: 0 forward KL, 1 reverse KL, in between generalized JSD.
        chunk_size: valid positions per chunk. Peak memory scales with this.
        num_items_in_batch: total valid tokens across the global batch. When given
            the reduction is ``sum / num_items_in_batch``, which is what makes
            gradient accumulation exact; when ``None`` it is the local mean.
        scaler: current GradScaler, applied before the chunk gradients are
            computed. The returned loss remains unscaled.
    Returns:
        ``(loss, entropy_sum, n_valid_tokens)``. The last two are raw local sums so
        a distributed caller can reduce them itself.
    """
    return _UnslothDistillationJSD.apply(
        student_hidden_states, teacher_hidden_states,
        student_lm_head, teacher_lm_head,
        completion_mask, beta, chunk_size, num_items_in_batch,
        student_lm_head_bias, teacher_lm_head_bias,
        student_logit_scale, teacher_logit_scale,
        student_final_logit_softcapping, teacher_final_logit_softcapping,
        temperature, scaler.get_scale() if scaler is not None else 1.0,
    )
pass
RL_REPLACEMENTS["distillation_chunked_jsd"] = distillation_chunked_jsd

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
