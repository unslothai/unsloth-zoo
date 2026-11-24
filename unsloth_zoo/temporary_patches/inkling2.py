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
"""Inkling 2 (inkling2 / inkling2_loop, transformers >= 5.24) MoE experts: softcapped SwiGLU over
gate and up, tanh(g / c_g) * c_g * sigmoid(g) * tanh(u / c_u) * c_u, with fp32 routed outputs;
both expert classes and fp32 reference must declare _apply_gate, else generic backends use
act_fn(gate) * up."""

import contextlib
import functools
from typing import List

import torch

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import UNSLOTH_DECODE_COMPILE, logger, torch_compiler_disable_unless_decode

__all__ = [
    "patch_inkling2_moe_experts",
    "INKLING2_EXPERT_CLASSES",
    "INKLING2_DEQUANT_SCRATCH_FRACTION",
    "inkling2_residual_eager_unless_decode",
    "inkling2_block_residual_reference",
    "inkling2_block_residual_op",
    "patch_hybrid_sliding_generate_masks",
]

# Training dequantizes a frozen 4-bit expert stack per projection call: a whole Mini gate_up stack is 1.27 GiB, a
# Small one 16 GiB. Where a stack exceeds twice this share of the card and does not fit the free memory, read it in
# chunks of at most this share (moe_utils._auto_dequant_chunk_experts); the chunks give the whole-stack values bit
# for bit.
INKLING2_DEQUANT_SCRATCH_FRACTION = 0.025

INKLING2_EXPERT_CLASSES = (
    ("transformers.models.inkling2.modeling_inkling2", "Inkling2Experts"),
    ("transformers.models.inkling2_loop.modeling_inkling2_loop", "Inkling2LoopExperts"),
)


def patch_inkling2_moe_experts():
    import importlib

    for module_name, class_name in INKLING2_EXPERT_CLASSES:
        try:
            klass = getattr(importlib.import_module(module_name), class_name)
        except Exception:
            continue
        if not callable(getattr(klass, "_apply_gate", None)):
            # Layout changed upstream: keep the native forward rather than guess the activation.
            continue
        klass._unsloth_own_apply_gate = True
        klass._unsloth_moe_fp32_reference = True
        klass._unsloth_dequant_scratch_fraction = INKLING2_DEQUANT_SCRATCH_FRACTION
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(f"Unsloth: {class_name} uses its own softcapped SwiGLU in the MoE backends.")


TEMPORARY_PATCHES.append(patch_inkling2_moe_experts)


# Block attention residuals. Compiled with Inductor (dynamic = True, CUDA), the one-query form over 3+ stacked blocks
# returns correct values but wrong input grads, so the compiler keeps them eager. A torch.compiler.disable boundary
# does that outside any graph, but inside an enclosing compiled graph it is a graph break, and under fullgraph = True
# a hard Unsupported. The opaque custom ops below keep the same eager forward and backward inside the graph instead.
def inkling2_block_residual_reference(block_states, partial_block_states, query, eps, sdpa = False):
    """transformers' Inkling2{,Loop}BlockAttentionResidual.forward (sdpa, else eager attention), stacked as
    [..., num_queries, hidden] instead of unbound."""
    stacked = torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim = -2).float()
    stacked = stacked * torch.rsqrt(stacked.square().mean(-1, keepdim = True) + eps)
    hidden = stacked.shape[-1]
    q = query.float().view(-1, hidden)
    if sdpa:
        # The module's own call, on the operands autocast would give it (each cast on its own, after the expand, so
        # the grads accumulate in fp32 as there). Cast by hand: torch.func.vjp through the math backend under an
        # active autocast fails ("expected scalar type Float but found Half", torch 2.10).
        q, k, v = q.expand(*stacked.shape[:-2], -1, -1), stacked, stacked
        device_type = stacked.device.type
        autocast, autocast_dtype = _autocast_state(device_type)
        if autocast:
            q, k, v = q.to(autocast_dtype), k.to(autocast_dtype), v.to(autocast_dtype)
        with torch.autocast(device_type, enabled = False):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale = hidden ** -0.5)
    # Eager: the scale goes on the fp32 query before the matmul (the module scales after it), so under fp16 autocast
    # the score (up to hidden * |q| unscaled) cannot overflow to inf.
    scores = torch.matmul(q * hidden ** -0.5, stacked.transpose(-1, -2))  # [..., num_queries, blocks]
    return torch.matmul(torch.softmax(scores, dim = -1, dtype = torch.float32), stacked)


def _autocast_state(device_type):
    # The caller's autocast for this device. The op body runs below the dispatcher's autocast handling and, compiled,
    # outside the enclosing autocast region, so the state is passed in and re-entered explicitly.
    try:
        return torch.is_autocast_enabled(device_type), torch.get_autocast_dtype(device_type)
    except Exception:  # older torch: per-device getters
        if device_type == "cpu":
            return torch.is_autocast_cpu_enabled(), torch.get_autocast_cpu_dtype()
        return torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype()


def _residual_under(device_type, autocast, autocast_dtype, sdpa, block_states, partial_block_states, query, eps):
    # The eager module under the same autocast: its attention autocasts (output in autocast_dtype), the rest is fp32.
    with torch.autocast(device_type, dtype = autocast_dtype, enabled = autocast):
        out = inkling2_block_residual_reference(block_states, partial_block_states, query, eps, sdpa)
    return out.to(autocast_dtype if autocast else torch.float32)  # the dtype the fake promises


@contextlib.contextmanager
def _without_saved_tensors_hooks():
    # Pop the thread's saved tensor default hooks for the block and push them back after, in order.
    autograd = torch._C._autograd
    top_hooks = getattr(autograd, "_top_saved_tensors_default_hooks", None)  # torch < 2.7 lacks it
    if top_hooks is None:
        yield
        return
    stack = []
    try:
        while (top := top_hooks(True)) is not None:
            stack.append(top)
            autograd._pop_saved_tensors_default_hooks()
        yield
    finally:
        for pack, unpack in reversed(stack):
            autograd._push_saved_tensors_default_hooks(pack, unpack)


_RESIDUAL_OP = None
try:
    @torch.library.custom_op("unsloth_zoo::inkling2_block_residual", mutates_args = ())
    def _residual_op(block_states: torch.Tensor, partial_block_states: torch.Tensor, query: torch.Tensor,
                     eps: float, device_type: str, autocast: bool, autocast_dtype: torch.dtype,
                     sdpa: bool) -> torch.Tensor:
        with torch.no_grad():
            return _residual_under(
                device_type, autocast, autocast_dtype, sdpa, block_states, partial_block_states, query, eps,
            ).contiguous()

    @_residual_op.register_fake
    def _residual_fake(block_states, partial_block_states, query, eps, device_type, autocast, autocast_dtype, sdpa):
        hidden = partial_block_states.shape[-1]
        return partial_block_states.new_empty(
            (*partial_block_states.shape[:-1], query.numel() // hidden, hidden),
            dtype = autocast_dtype if autocast else torch.float32)

    @torch.library.custom_op("unsloth_zoo::inkling2_block_residual_backward", mutates_args = ())
    def _residual_backward_op(grad: torch.Tensor, block_states: torch.Tensor, partial_block_states: torch.Tensor,
                              query: torch.Tensor, eps: float, device_type: str, autocast: bool,
                              autocast_dtype: torch.dtype, sdpa: bool) -> List[torch.Tensor]:
        # The op body runs below autograd: torch.func.vjp carries its own grad transform. The recompute re-enters the
        # forward's autocast, whatever the backward pass runs under, and sets aside the caller's saved tensor hooks
        # (save_on_cpu offloading), which torch.func refuses and this local recompute has no use for.
        with _without_saved_tensors_hooks():
            _, vjp = torch.func.vjp(
                lambda b, p, q: _residual_under(device_type, autocast, autocast_dtype, sdpa, b, p, q, eps),
                block_states, partial_block_states, query)
            vjp_grads = vjp(grad)
        taken = {t.untyped_storage().data_ptr() for t in (grad, block_states, partial_block_states, query)}
        grads = []
        for g in vjp_grads:
            # Contiguous, as the fake promises; op outputs may not alias its inputs or each other.
            g = g.contiguous()
            ptr = g.untyped_storage().data_ptr()
            grads.append(g.clone() if ptr in taken else g)
            taken.add(ptr)
        return grads

    @_residual_backward_op.register_fake
    def _residual_backward_fake(grad, block_states, partial_block_states, query, eps, device_type, autocast,
                                autocast_dtype, sdpa):
        return [t.new_empty(t.shape) for t in (block_states, partial_block_states, query)]

    def _residual_setup_context(ctx, inputs, output):
        block_states, partial_block_states, query, eps, device_type, autocast, autocast_dtype, sdpa = inputs
        ctx.save_for_backward(block_states, partial_block_states, query)
        ctx.state = (eps, device_type, autocast, autocast_dtype, sdpa)

    def _residual_backward(ctx, grad):
        block_states, partial_block_states, query = ctx.saved_tensors
        grads = torch.ops.unsloth_zoo.inkling2_block_residual_backward(
            grad.contiguous(), block_states, partial_block_states, query, *ctx.state)
        return grads[0], grads[1], grads[2], None, None, None, None, None

    _residual_op.register_autograd(_residual_backward, setup_context = _residual_setup_context)
    _RESIDUAL_OP = _residual_op
except Exception:  # torch without torch.library.custom_op: the plain compiler boundary stays
    _RESIDUAL_OP = None


def inkling2_block_residual_op(block_states, partial_block_states, query, eps, sdpa = False):
    """The opaque residual under the caller's autocast, as the eager module computes it."""
    device_type = partial_block_states.device.type
    autocast, autocast_dtype = _autocast_state(device_type)
    return _RESIDUAL_OP(block_states, partial_block_states, query, eps, device_type, autocast, autocast_dtype,
                        bool(sdpa))


def _residual_inputs(args, kwargs):
    # The compiler's generated forward passes both by name.
    if not kwargs and len(args) == 2:
        return args
    if len(args) + len(kwargs) != 2:
        return None
    names = ("block_states", "partial_block_states")
    got = dict(zip(names, args))
    for name, value in kwargs.items():
        if name not in names or name in got:
            return None
        got[name] = value
    return got["block_states"], got["partial_block_states"]


def _has_hooks(module):
    # The op computes the norm inline, so a hook on it (or a global module hook) would silently not run.
    names = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
    if any(len(getattr(module, name, None) or ()) for name in names):
        return True
    names = ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks",
             "_global_backward_pre_hooks")
    return any(len(getattr(torch.nn.modules.module, name, None) or ()) for name in names)


def _residual_op_eps(module, args):
    # Only the layout the reference reproduces: two tensor inputs, an unweighted RMSNorm, eager or sdpa attention
    # (transformers 5.23 calls sdpa directly and keeps no config on the module).
    if _RESIDUAL_OP is None or args is None:
        return None
    query, norm = getattr(module, "query", None), getattr(module, "norm", None)
    eps = getattr(norm, "eps", None)
    if not isinstance(query, torch.Tensor) or not type(norm).__name__.endswith("UnweightedRMSNorm"):
        return None
    if not isinstance(eps, float) or len(list(norm.parameters())) != 0 or _has_hooks(norm):
        return None
    config = getattr(module, "config", None)
    if config is not None and getattr(config, "_attn_implementation", None) not in ("eager", "sdpa"):
        return None
    if not all(isinstance(x, torch.Tensor) for x in args) or args[0].shape[-1] != query.shape[-1]:
        return None
    return eps


def inkling2_residual_eager_unless_decode(func):
    """torch_compiler_disable_unless_decode for the block attention residuals, except that inside an enclosing
    compiled training graph the residual runs as an opaque op (eager forward and backward, no graph break)."""
    disabled = torch_compiler_disable_unless_decode(func)
    plain = getattr(func, "_unsloth_undisabled", func)

    @functools.wraps(plain)
    def forward(self, *args, **kwargs):
        if torch.compiler.is_compiling() and not UNSLOTH_DECODE_COMPILE[0]:
            inputs = _residual_inputs(args, kwargs)
            eps = _residual_op_eps(self, inputs)
            if eps is not None:
                config = getattr(self, "config", None)
                sdpa = config is None or config._attn_implementation == "sdpa"
                return inkling2_block_residual_op(inputs[0], inputs[1], self.query, eps, sdpa).unbind(-2)
        return disabled(self, *args, **kwargs)
    forward._unsloth_undisabled = plain
    return forward


def patch_hybrid_sliding_generate_masks():
    """generate prebuilds the attention masks for a compileable cache (StaticCache) from the layer pattern table, which
    in transformers 5.24 gives the sliding side of "hybrid_sliding" layers (inkling, inkling2, inkling2_loop, zaya) a
    full causal mask. Those models build a sliding window mask in their own forward, and the full one is sized to the
    whole static buffer while the sliding layers return window bounded states, so a prefill longer than the window
    fails with a size mismatch. Point that entry at the sliding window mask, as the models do; a no-op where
    transformers already maps it so."""
    try:
        import transformers.masking_utils as masking_utils
    except Exception:
        return
    mapping = getattr(masking_utils, "LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING", None)
    entry = mapping.get("hybrid_sliding") if isinstance(mapping, dict) else None
    causal = getattr(masking_utils, "create_causal_mask", None)
    sliding = getattr(masking_utils, "create_sliding_window_causal_mask", None)
    if not isinstance(entry, dict) or causal is None or sliding is None:
        return
    if entry.get("sliding_attention") is causal:
        entry["sliding_attention"] = sliding
        if UNSLOTH_ENABLE_LOGGING:
            logger.info("Unsloth: hybrid_sliding layers get a sliding window mask in generate.")


TEMPORARY_PATCHES.append(patch_hybrid_sliding_generate_masks)
