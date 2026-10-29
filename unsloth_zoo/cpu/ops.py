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

"""Pure-torch CPU replacements. No Triton, no CUDA."""

__all__ = [
    "chunked_linear_cross_entropy",
    "patch_chunked_causal_lm_loss",
]

import functools
import math
import os
import types

import torch
import torch.nn.functional as F


def _cpu_fp16_autocast():
    try:
        return torch.is_autocast_enabled("cpu") and torch.get_autocast_dtype("cpu") == torch.float16
    except TypeError:
        # Older torch: the device-less CPU autocast API.
        return torch.is_autocast_cpu_enabled() and torch.get_autocast_cpu_dtype() == torch.float16


def _softmax_without_collapse(z, y, keep_logits = False):
    """(softmax(z) with the target column zeroed, log p_t, shifted z_t); shifts z in place.
    Sums the other classes apart from exp(0) = 1, else p_t rounds to 1 past a ~16.6 lead (softmax collapse, arXiv 2610.31642)."""
    z.sub_(z.amax(-1, keepdim = True))
    target = z.gather(-1, y)
    p = torch.exp(z) if keep_logits else z.exp_()
    others = p.scatter_(-1, y, 0.0).sum(-1, keepdim = True)
    log_p_target = -F.softplus(torch.log(others) - target)
    p.div_(others + torch.exp(target))
    return p, log_p_target, target


class _ChunkedLinearCrossEntropy(torch.autograd.Function):
    """sum CE(hidden @ weight.T, labels) / divisor without materialising the (rows, vocab) logits.

    Gradients are formed chunk by chunk inside forward (the loss is a scalar, so backward only
    rescales them): peak extra memory is one (chunk, vocab) fp32 block instead of the full logits
    plus their gradient, which for a 150k vocab is the largest tensor of a CPU training step.
    """

    @staticmethod
    def forward(ctx, hidden, weight, labels, divisor, chunk_rows, ignore_index):
        rows = hidden.shape[0]
        need_hidden_grad = hidden.requires_grad
        need_weight_grad = weight.requires_grad
        # fp16 cannot hold these gradients before GradScaler multiplies them: keep them fp32 until backward.
        exact = hidden.dtype == torch.float16 or _cpu_fp16_autocast()
        grad_dtype = torch.float32 if exact else hidden.dtype
        grad_hidden = torch.empty(hidden.shape, dtype = grad_dtype) if need_hidden_grad else None
        grad_weight = torch.zeros(weight.shape, dtype = torch.float32) if need_weight_grad else None
        total = torch.zeros((), dtype = torch.float32)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            y = labels[start:end]
            valid = y != ignore_index
            # Zero ignored rows first: fp16 can overflow to inf in the projection, and NaN * 0 stays NaN.
            h = torch.where(valid.unsqueeze(-1), hidden[start:end], hidden.new_zeros(()))
            logits = (h @ weight.t()).float()
            safe_y = torch.where(valid, y, torch.zeros_like(y)).unsqueeze(-1)
            logits, log_p_target, _ = _softmax_without_collapse(logits, safe_y)
            total -= (log_p_target.squeeze(-1) * valid).sum()
            if need_hidden_grad or need_weight_grad:
                # softmax - onehot on valid rows; p_t - 1 = expm1(log p_t).
                logits.scatter_(-1, safe_y, torch.expm1(log_p_target))
                logits.mul_(valid.unsqueeze(-1))
                if need_hidden_grad:
                    if exact:
                        with torch.autocast("cpu", enabled = False):
                            grad_hidden[start:end] = logits @ weight.float()
                    else:
                        grad_hidden[start:end] = logits.to(h.dtype) @ weight
                if need_weight_grad:
                    grad_weight.addmm_(logits.t(), h.float())
        saved = (
            grad_hidden if grad_hidden is not None else torch.empty(0),
            grad_weight if grad_weight is not None else torch.empty(0),
        )
        # num_items_in_batch == 0 (a fully masked accumulation group): a finite 0, as TRL's clamped divisor gives.
        if torch.is_tensor(divisor):
            # Keep tensor counts as tensors: .item() breaks fullgraph tracing.
            loss = total / torch.where(divisor == 0, torch.ones_like(divisor), divisor)
            ctx.divisor = None
            saved = saved + (divisor,)
        else:
            loss = total / (1.0 if divisor == 0 else divisor)
            ctx.divisor = divisor
        ctx.save_for_backward(*saved)
        ctx.has_hidden = need_hidden_grad
        ctx.has_weight = need_weight_grad
        ctx.weight_dtype = weight.dtype
        ctx.hidden_dtype = hidden.dtype
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_hidden, grad_weight = ctx.saved_tensors[:2]
        # Every label ignored: the loss is NaN like F.cross_entropy(mean), but its gradient is zero.
        # Also num_items_in_batch == 0 (a fully masked accumulation group): stock CE's gradient is zero there.
        if ctx.divisor is None:
            divisor = ctx.saved_tensors[2]
            empty = torch.isnan(divisor) | (divisor == 0)
            scale = torch.where(empty, torch.zeros_like(grad_output),
                                grad_output / torch.where(empty, torch.ones_like(divisor), divisor))
        else:
            empty = math.isnan(ctx.divisor) or ctx.divisor == 0
            scale = torch.zeros_like(grad_output) if empty else grad_output / ctx.divisor
        dh = (grad_hidden * scale.to(grad_hidden.dtype)).to(ctx.hidden_dtype) if ctx.has_hidden else None
        dw = (grad_weight * scale).to(ctx.weight_dtype) if ctx.has_weight else None
        return dh, dw, None, None, None, None


_LOGPROB_OUTPUTS = ("entropy", "log_sum_sq_probs", "mean_logits", "is_top1")


def _logprob_piece(hidden, weight, bias, targets, inv_t, softcap, logit_scale, want_entropy):
    logits = (hidden @ weight.t()).float()
    if bias is not None:
        logits = logits + bias.float()
    logits = logits * logit_scale
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    logits = logits * inv_t
    y = targets.unsqueeze(-1)
    target = logits.gather(-1, y).squeeze(-1)
    # Softplus form: keeps 1 - p_t past a ~16.6 lead (softmax collapse).
    log_others = torch.logsumexp(logits.scatter(-1, y, float("-inf")), dim = -1)
    log_probs = -F.softplus(log_others - target)
    log_z = target - log_probs
    if not want_entropy:
        return log_probs, log_z
    entropy = log_z - (torch.softmax(logits, dim = -1) * logits).sum(-1)
    return log_probs, entropy


def _transformed_logits(hidden, weight, bias, inv_t, softcap, logit_scale):
    """fp32 (hidden @ W.T + bias) * scale, soft-capped, / temperature, and d(final)/d(raw) (a scalar without a cap)."""
    logits = (hidden @ weight.t()).float()
    if bias is not None:
        logits = logits + bias.float()
    logits = logits * logit_scale
    slope = logit_scale * inv_t
    if softcap is not None:
        capped = torch.tanh(logits / softcap)
        slope = (1 - capped * capped) * slope
        logits = capped * softcap
    return logits * inv_t, slope


class _ChunkedLogProbs(torch.autograd.Function):
    """One vocabulary pass per row chunk; d(log_prob)/d(hidden) is per row, so backward only rescales."""

    @staticmethod
    def forward(ctx, hidden, weight, bias, targets, inv_t, softcap, logit_scale, rows, outputs, need_grad):
        ctx.set_materialize_grads(False)
        n = hidden.shape[0]
        exact = hidden.dtype == torch.float16  # fp16 gradients overflow before GradScaler: keep them fp32
        grad_lp = torch.empty(hidden.shape, dtype = torch.float32 if exact else hidden.dtype) if need_grad else None
        new = lambda name, dtype = torch.float32: torch.empty(n, dtype = dtype) if name in outputs else None  # noqa: E731
        log_probs, entropy, sq, mean = new("log_probs"), new("entropy"), new("log_sum_sq_probs"), new("mean_logits")
        top1 = new("is_top1", torch.bool)
        rowdot = lambda x, y: torch.einsum("ij,ij->i", x, y)  # noqa: E731
        for start in range(0, n, rows):
            end = min(start + rows, n)
            y = targets[start:end].unsqueeze(-1)
            z, slope = _transformed_logits(hidden[start:end], weight, bias, inv_t, softcap, logit_scale)
            if mean is not None:
                mean[start:end] = z.mean(-1)
            if top1 is not None:
                top1[start:end] = z.argmax(-1) == targets[start:end]
            p, log_p, target = _softmax_without_collapse(z, y, keep_logits = entropy is not None)
            log_probs[start:end] = log_p.squeeze(-1)
            p_target = torch.exp(log_p).squeeze(-1)  # p's target column is zero
            if entropy is not None:
                # H = log(total) - sum p * z, the target term apart.
                entropy[start:end] = (target - log_p).squeeze(-1) - rowdot(p, z) - p_target * target.squeeze(-1)
            if sq is not None:
                sq[start:end] = torch.log(rowdot(p, p) + p_target * p_target)
            if need_grad:
                p.neg_().scatter_(-1, y, -torch.expm1(log_p))
                if not (isinstance(slope, float) and slope == 1.0):
                    p.mul_(slope)
                if exact:
                    with torch.autocast("cpu", enabled = False):
                        grad_lp[start:end] = p @ weight.float()
                else:
                    grad_lp[start:end] = p.to(weight.dtype) @ weight
        ctx.save_for_backward(hidden, weight, bias, targets, grad_lp)
        ctx.args = (inv_t, softcap, logit_scale, rows)
        result = (log_probs, entropy, sq, mean, top1)
        ctx.mark_non_differentiable(*[x for x in result[2:] if x is not None])
        return result

    @staticmethod
    def backward(ctx, d_log_probs, d_entropy, *unused):
        hidden, weight, bias, targets, grad_lp = ctx.saved_tensors
        if grad_lp is None or (d_log_probs is None and d_entropy is None):
            return (None,) * 10
        dh = torch.zeros(hidden.shape, dtype = grad_lp.dtype)
        if d_log_probs is not None:
            dh += grad_lp * d_log_probs.to(grad_lp.dtype).unsqueeze(-1)
        if d_entropy is not None and bool(torch.any(d_entropy != 0)):
            # Entropy is only a metric in TRL SFT: recompute dH/dz = -p (z - E_p[z]) when it gets a gradient.
            inv_t, softcap, logit_scale, rows = ctx.args
            for start in range(0, hidden.shape[0], rows):
                end = min(start + rows, hidden.shape[0])
                z, slope = _transformed_logits(hidden[start:end], weight, bias, inv_t, softcap, logit_scale)
                p = torch.softmax(z, dim = -1)
                g = -p * (z - (p * z).sum(-1, keepdim = True)) * slope
                g = g * d_entropy[start:end].float().unsqueeze(-1)
                dh[start:end] += (g @ weight.float()).to(dh.dtype)
        return (dh.to(hidden.dtype),) + (None,) * 9


def _checkpointed_log_probs(hidden, weight, bias, targets, inv_t, softcap, logit_scale, rows, outputs):
    """Trainable LM head / bias: row chunks under activation checkpointing, so autograd forms every gradient."""
    from torch.utils.checkpoint import checkpoint
    want_entropy = "entropy" in outputs
    grad = torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad)
    log_probs, entropy = [], []
    for start in range(0, hidden.shape[0], rows):
        args = (hidden[start:start + rows], weight, bias, targets[start:start + rows], inv_t,
                softcap, logit_scale, want_entropy)
        lp, ent = checkpoint(_logprob_piece, *args, use_reentrant = False) if grad else _logprob_piece(*args)
        log_probs.append(lp)
        entropy.append(ent)
    log_probs = torch.cat(log_probs)
    entropy = torch.cat(entropy) if want_entropy else None
    extras = {}
    if any(name in outputs for name in ("log_sum_sq_probs", "mean_logits", "is_top1")):
        # Metrics only (TRL marks them non-differentiable): one more pass without autograd.
        with torch.no_grad():
            sq, mean, top1 = [], [], []
            for start in range(0, hidden.shape[0], rows):
                z, _ = _transformed_logits(hidden[start:start + rows], weight, bias, inv_t, softcap, logit_scale)
                sq.append(torch.logsumexp(2 * torch.log_softmax(z, dim = -1), dim = -1))
                mean.append(z.mean(-1))
                top1.append(z.argmax(-1) == targets[start:start + rows])
            extras = {"log_sum_sq_probs": torch.cat(sq), "mean_logits": torch.cat(mean), "is_top1": torch.cat(top1)}
    return log_probs, entropy, extras.get("log_sum_sq_probs"), extras.get("mean_logits"), extras.get("is_top1")


def cpu_chunked_log_probs(hidden, weight, bias, targets, temperature, chunk_size,
                          final_logit_softcapping = None, logit_scale = 1.0, outputs = _LOGPROB_OUTPUTS):
    """Torch twin of TRL 1.x's Triton fused-LM-head `ChunkedLogProbFunction`, in row chunks (no [N, V] logits)."""
    if torch.is_autocast_enabled(hidden.device.type):
        dtype = torch.get_autocast_dtype(hidden.device.type)
        hidden, weight = hidden.to(dtype), weight.to(dtype)
        bias = bias.to(dtype) if bias is not None else None
    if hidden.shape[0] == 0:
        empty = (hidden @ weight.t()).float().sum(-1) * 0
        pick = lambda name, value: value if name in outputs else None  # noqa: E731
        return (empty, pick("entropy", empty), pick("log_sum_sq_probs", empty.detach()),
                pick("mean_logits", empty.detach()), pick("is_top1", empty.detach().bool()))
    rows = max(1, int(os.environ.get("UNSLOTH_CPU_CE_CHUNK_ELEMENTS", 16 * 1024 * 1024)) // max(1, weight.shape[0]))
    inv_t = 1.0 / temperature
    head_grad = torch.is_grad_enabled() and (weight.requires_grad or (bias is not None and bias.requires_grad))
    if head_grad:
        result = _checkpointed_log_probs(hidden, weight, bias, targets, inv_t, final_logit_softcapping,
                                         logit_scale, rows, outputs)
    else:
        with torch.autocast(hidden.device.type, enabled = False):
            result = _ChunkedLogProbs.apply(
                hidden, weight.detach(), bias.detach() if bias is not None else None, targets, inv_t,
                final_logit_softcapping, logit_scale, rows, ("log_probs", *outputs),
                torch.is_grad_enabled() and hidden.requires_grad,
            )
    names = ("log_probs", "entropy", "log_sum_sq_probs", "mean_logits", "is_top1")
    return tuple(value if name == "log_probs" or name in outputs else None for name, value in zip(names, result))


def patch_trl_fused_lm_head():
    """TRL 1.x SFTTrainer always computes its loss through a fused LM head whose kernel is Triton
    (CUDA / XPU only) and is None where Triton is missing, so CPU training and evaluate() fail. Keep
    the kernel on those devices and route every other device to cpu_chunked_log_probs. Idempotent."""
    try:
        from trl.trainer import utils as trl_utils
    except Exception:
        return False
    if not hasattr(trl_utils, "_ChunkedLogProbFunction"):
        return False  # TRL without the fused head: nothing to route.
    kernel = trl_utils._ChunkedLogProbFunction
    if getattr(kernel, "_unsloth_cpu_dispatch", False):
        return True

    class _DeviceDispatch:
        _unsloth_cpu_dispatch = True
        _kernel = kernel

        @staticmethod
        def apply(hidden, *args, **kwargs):
            if kernel is not None and hidden.device.type in ("cuda", "xpu"):
                return kernel.apply(hidden, *args, **kwargs)
            return cpu_chunked_log_probs(hidden, *args, **kwargs)

    trl_utils._ChunkedLogProbFunction = _DeviceDispatch
    return True


class _Bf16StoredLinearFunction(torch.autograd.Function):
    """F.linear with a frozen bf16 weight upcast per call; saves the bf16 weight, backward is dX = dY @ W."""

    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(weight)
        return F.linear(x, weight.to(x.dtype), bias)

    @staticmethod
    def backward(ctx, grad):
        (weight,) = ctx.saved_tensors
        return grad @ weight.to(grad.dtype), None, None


class Bf16StoredLinear(torch.nn.Linear):
    """nn.Linear whose frozen weight stays in bf16 while activations, LoRA and the matmul are fp32."""

    def forward(self, x):
        trainable = self.weight.requires_grad or (self.bias is not None and self.bias.requires_grad)
        if self.weight.dtype == x.dtype or (trainable and torch.is_grad_enabled()):
            # Trainable weight / bias (LoRA bias="all") or an upcast copy (merged fp32): plain autograd.
            return F.linear(x, self.weight.to(x.dtype), self.bias)
        return _Bf16StoredLinearFunction.apply(x, self.weight, self.bias)


# Only projections always called as modules: Mamba dt_proj, Nemotron-H out_proj, MLA read .weight directly.
_BF16_STORED_PROJECTIONS = frozenset((
    "q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "gate_proj", "up_proj", "down_proj", "gate_up_proj",
    "in_proj_qkv", "in_proj_z",
))


def store_frozen_linears_in_bf16(model):
    """fp32 training of a bf16 checkpoint: frozen attention / MLP projections stay bf16 (lossless), the rest fp32."""
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    stored = set()
    for name, module in model.named_modules():
        if type(module) is torch.nn.Linear and module is not head and module.weight.dtype == torch.bfloat16 \
                and name.rsplit(".", 1)[-1] in _BF16_STORED_PROJECTIONS:
            module.__class__ = Bf16StoredLinear
            stored.add(id(module.weight))
    for param in model.parameters():
        if id(param) not in stored and param.dtype == torch.bfloat16:
            param.data = param.data.float()
    for buffer in model.buffers():
        if buffer.dtype == torch.bfloat16:
            buffer.data = buffer.data.float()
    return len(stored)


def cpu_causal_conv1d(hidden_states, weight, bias = None, activation = None, **kwargs):
    """Causal depthwise conv1d as K shifted multiply-adds: transformers' grouped F.conv1d fallback is slow without oneDNN."""
    x = hidden_states.to(weight.dtype)
    kernel, length = weight.shape[-1], x.shape[-1]
    padded = F.pad(x, (kernel - 1, 0))
    out = padded[..., :length] * weight[:, :1]
    for j in range(1, kernel):
        out = out.addcmul(padded[..., j:j + length], weight[:, j:j + 1])
    if bias is not None:
        out = out + bias[:, None]
    if activation is not None:
        from transformers.activations import ACT2FN
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def patch_causal_conv1d_fallback(model):
    """Without oneDNN (macOS wheels), route modeling modules' `causal_conv1d_fn(hidden_states, weight, bias,
    activation)` to cpu_causal_conv1d when a probe agrees; with oneDNN the grouped conv is faster."""
    import inspect
    import sys
    if torch.backends.mkldnn.is_available():
        return 0
    patched = 0
    for name in {type(module).__module__ for module in model.modules()}:
        module = sys.modules.get(name)
        fn = getattr(module, "causal_conv1d_fn", None)
        if fn is None or fn is cpu_causal_conv1d:
            continue
        try:
            params = list(inspect.signature(fn).parameters)[:4]
            if params != ["hidden_states", "weight", "bias", "activation"]:
                continue
            gen = torch.Generator().manual_seed(0)
            x, w, b = (torch.randn(shape, generator = gen) for shape in ((2, 8, 11), (8, 4), (8,)))
            if not torch.allclose(fn(x, w, b, activation = "silu"), cpu_causal_conv1d(x, w, b, activation = "silu"),
                                  rtol = 1e-5, atol = 1e-6):
                continue
        except Exception:
            continue
        module.causal_conv1d_fn = cpu_causal_conv1d
        patched += 1
    return patched


def chunked_linear_cross_entropy(hidden, weight, labels, num_items_in_batch = None,
                                 ignore_index = -100, shift_labels = None, chunk_rows = None):
    """Causal-LM loss with the same shift and normalisation as transformers' ForCausalLMLoss."""
    if shift_labels is None:
        labels = F.pad(labels, (0, 1), value = ignore_index)
        shift_labels = labels[..., 1:]
    hidden = hidden.reshape(-1, hidden.shape[-1])
    shift_labels = shift_labels.reshape(-1).to(hidden.device)
    if num_items_in_batch is None:
        count = (shift_labels != ignore_index).sum().to(torch.float32)
        # Matches F.cross_entropy(reduction="mean"): NaN when every label is ignored.
        divisor = torch.where(count == 0, torch.full_like(count, float("nan")), count)
    elif torch.is_tensor(num_items_in_batch):
        divisor = num_items_in_batch.detach().to(device = hidden.device, dtype = torch.float32).reshape(())
    else:
        divisor = float(num_items_in_batch)
    # Skip ignored rows' LM head work. Eager only: the sync breaks compile; chunk loop zeroes them anyway.
    if not torch.compiler.is_compiling():
        valid = shift_labels != ignore_index
        if bool(valid.any()) and not bool(valid.all()):
            rows = valid.nonzero().squeeze(-1)
            hidden, shift_labels = hidden.index_select(0, rows), shift_labels.index_select(0, rows)
    if chunk_rows is None:
        # ~64 MB of fp32 logits per chunk.
        chunk_rows = max(16, int(os.environ.get("UNSLOTH_CPU_CE_CHUNK_ELEMENTS", 16 * 1024 * 1024)) // weight.shape[0])
    return _ChunkedLinearCrossEntropy.apply(hidden, weight, shift_labels, divisor, chunk_rows, ignore_index)


# Architectures whose causal-LM loss is plain CE over lm_head(hidden). Others (soft caps, logit scales,
# z-loss, router losses, ...) keep their own forward.
_CHUNKED_CE_MODEL_TYPES = frozenset(("llama", "mistral", "qwen2", "qwen3", "phi3", "gemma", "olmo2", "smollm3"))


def _plain_lm_head(model):
    """Only models whose logits are exactly lm_head(hidden): no softcap, scale or bias."""
    config = getattr(model, "config", None)
    text_config = getattr(config, "get_text_config", lambda: config)()
    if getattr(text_config, "model_type", None) not in _CHUNKED_CE_MODEL_TYPES:
        return False
    for name in ("final_logit_softcapping", "logits_soft_cap", "logit_scale", "logits_scaling", "output_multiplier_scale"):
        value = getattr(text_config, name, None)
        if value not in (None, 1, 1.0):
            return False
    # MoE heads add a router load-balancing loss inside their own forward.
    if getattr(text_config, "output_router_logits", False) or getattr(text_config, "num_experts", None) \
            or getattr(text_config, "num_local_experts", None) or getattr(text_config, "n_routed_experts", None):
        return False
    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if not isinstance(lm_head, torch.nn.Linear) or lm_head.bias is not None:
        return False
    inner = getattr(model, getattr(model, "base_model_prefix", "model"), None)
    return isinstance(inner, torch.nn.Module)


def _plain_head_weight(head):
    """The weight that alone produces this head's logits, or None (e.g. a LoRA-wrapped lm_head)."""
    if type(head) is torch.nn.Linear and head.bias is None:
        return head.weight
    if type(head).__name__ == "ModulesToSaveWrapper":
        adapter = getattr(head, "active_adapter", None)
        adapter = adapter[0] if isinstance(adapter, (list, tuple)) else adapter
        copies = getattr(head, "modules_to_save", None)
        active = copies[adapter] if copies is not None and adapter in copies else None
        return _plain_head_weight(active if active is not None else getattr(head, "original_module", None))
    return None


def patch_chunked_causal_lm_loss(model):
    """Route training forwards with labels through chunked_linear_cross_entropy.

    Instance-level, so only this model changes. The returned logits are None in that case;
    unsloth_zoo.cpu.trainer makes TRL skip its logits-only metrics for it.
    Returns True when installed.
    """
    if os.environ.get("UNSLOTH_CPU_CHUNKED_CE", "1") == "0" or not _plain_lm_head(model):
        return False
    if getattr(model, "_unsloth_cpu_chunked_ce", False):
        return True
    from transformers.modeling_outputs import CausalLMOutputWithPast

    original_forward = model.forward
    # Call the class forward on `self`, not the bound method: deepcopy (TRL reference models,
    # merged copies) rebinds the outer method to the copy, and a bound original would still run
    # the source model's weights.
    original_func = original_forward.__func__ if getattr(original_forward, "__self__", None) is model else None
    inner_name = getattr(model, "base_model_prefix", "model")

    def forward(self, input_ids = None, attention_mask = None, position_ids = None,
                past_key_values = None, inputs_embeds = None, labels = None, use_cache = None,
                logits_to_keep = 0, **kwargs):
        shift_labels = kwargs.pop("shift_labels", None)
        head_weight = _plain_head_weight(self.get_output_embeddings())
        if (labels is None and shift_labels is None) or not torch.is_grad_enabled() \
                or not getattr(self, "_unsloth_cpu_chunked_ce_active", False) or head_weight is None:
            if shift_labels is not None:
                kwargs["shift_labels"] = shift_labels
            call = functools.partial(original_func, self) if original_func is not None else original_forward
            return call(
                input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids,
                past_key_values = past_key_values, inputs_embeds = inputs_embeds, labels = labels,
                use_cache = use_cache, logits_to_keep = logits_to_keep, **kwargs,
            )
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)
        ignore_index = kwargs.pop("ignore_index", -100)
        kwargs.pop("return_dict", None)
        outputs = getattr(self, inner_name)(
            input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids,
            past_key_values = past_key_values, inputs_embeds = inputs_embeds, use_cache = use_cache,
            **kwargs,
        )
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        loss = chunked_linear_cross_entropy(
            hidden, head_weight, labels,
            num_items_in_batch = num_items_in_batch, shift_labels = shift_labels, ignore_index = ignore_index,
        )
        return CausalLMOutputWithPast(
            loss = loss, logits = None,
            past_key_values = getattr(outputs, "past_key_values", None),
            hidden_states = getattr(outputs, "hidden_states", None),
            attentions = getattr(outputs, "attentions", None),
        )

    model.forward = types.MethodType(forward, model)
    model._unsloth_cpu_chunked_ce = True
    return True
