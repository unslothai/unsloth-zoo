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
import inspect
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


def _cpu_autocast_state():
    try:
        return torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu")
    except TypeError:
        return torch.is_autocast_cpu_enabled(), torch.get_autocast_cpu_dtype()


def _softmax_without_collapse(z, y, keep_logits = False):
    """Returns softmax(z) with target column zeroed, log p_t and shifted z_t, shifting z in place; sums other classes
    apart from exp(0)=1, else p_t rounds to 1 past a ~16.6 lead (softmax collapse, arXiv 2610.31642)."""
    z.sub_(z.amax(-1, keepdim = True))
    target = z.gather(-1, y)
    p = torch.exp(z) if keep_logits else z.exp_()
    others = p.scatter_(-1, y, 0.0).sum(-1, keepdim = True)
    log_p_target = -F.softplus(torch.log(others) - target)
    p.div_(others + torch.exp(target))
    return p, log_p_target, target


def _chunked_ce_forward_impl(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor,
                        chunk_rows: int, ignore_index: int, need_hidden_grad: bool,
                        need_weight_grad: bool, exact: bool, amp: bool,
                        amp_dtype: torch.dtype, softcap: float = 0.0,
                        logit_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Opaque to Dynamo/AOT: the runtime loop keeps logits bounded without specialising symbolic rows.
    # Compiled graphs run with autocast off, so the caller's CPU autocast is passed in and restored here.
    with torch.autocast("cpu", dtype = amp_dtype, enabled = amp):
        valid = labels != ignore_index
        if bool(valid.all()):
            return _chunked_ce_forward_loop(
                hidden, weight, labels, chunk_rows, ignore_index, need_hidden_grad, need_weight_grad, exact,
                amp_dtype if amp else None, softcap, logit_scale,
            )
        # Ignored rows skip the LM head GEMMs (compiled calls too: this op is opaque); their hidden grads are zero.
        keep = valid.nonzero().squeeze(-1)
        total, grad_hidden, grad_weight = _chunked_ce_forward_loop(
            hidden.index_select(0, keep), weight, labels.index_select(0, keep), chunk_rows, ignore_index,
            need_hidden_grad, need_weight_grad, exact, amp_dtype if amp else None, softcap, logit_scale,
        )
        if need_hidden_grad:
            grad_hidden = grad_hidden.new_zeros(hidden.shape).index_copy_(0, keep, grad_hidden)
        return total, grad_hidden, grad_weight


def _chunked_ce_forward_loop(hidden, weight, labels, chunk_rows, ignore_index, need_hidden_grad,
                             need_weight_grad, exact, amp_dtype = None, softcap = 0.0, logit_scale = 1.0):
    """CE over softcap * tanh(hidden @ W.T * logit_scale / softcap) (no tanh if softcap is 0)."""
    rows = hidden.shape[0]
    # A detached head misses autocast's cast cache, so cast it once instead of per chunk.
    mm_weight = weight
    if amp_dtype is not None and weight.is_floating_point() and weight.dtype not in (amp_dtype, torch.float64):
        mm_weight = weight.to(amp_dtype)
    grad_dtype = torch.float32 if exact else hidden.dtype
    grad_hidden = torch.empty(hidden.shape, dtype = grad_dtype) if need_hidden_grad else torch.empty(0)
    # The head gradient is [V, H] fp32 (622 MB for a 151k vocab): the first chunk's product becomes the buffer, so
    # it is never zero filled first; later chunks add into it in place.
    grad_weight = None
    total = torch.zeros((), dtype = torch.float32)
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        y = labels[start:end]
        valid = y != ignore_index
        # Zero ignored rows first: fp16 can overflow to inf in the projection, and NaN * 0 stays NaN.
        h = torch.where(valid.unsqueeze(-1), hidden[start:end], hidden.new_zeros(()))
        logits = (h @ mm_weight.t()).float()
        if logit_scale != 1.0:
            logits.mul_(logit_scale)
        slope = None
        if softcap > 0:
            logits.div_(softcap).tanh_()
            if need_hidden_grad or need_weight_grad:
                slope = logits.square().neg_().add_(1)
            logits.mul_(softcap)
        safe_y = torch.where(valid, y, torch.zeros_like(y)).unsqueeze(-1)
        logits, log_p_target, _ = _softmax_without_collapse(logits, safe_y)
        total -= (log_p_target.squeeze(-1) * valid).sum()
        if need_hidden_grad or need_weight_grad:
            # softmax - onehot on valid rows; p_t - 1 = expm1(log p_t).
            logits.scatter_(-1, safe_y, torch.expm1(log_p_target))
            logits.mul_(valid.unsqueeze(-1))
            if slope is not None:
                logits.mul_(slope)
                del slope
            if logit_scale != 1.0:
                logits.mul_(logit_scale)
            if need_hidden_grad:
                if exact:
                    with torch.autocast("cpu", enabled = False):
                        grad_hidden[start:end] = logits @ weight.float()
                else:
                    grad_hidden[start:end] = logits.to(h.dtype) @ mm_weight
            if need_weight_grad:
                if grad_weight is None:
                    grad_weight = torch.empty(weight.shape, dtype = torch.float32)
                    with torch.autocast("cpu", enabled = False):
                        torch.mm(logits.t(), h.float(), out = grad_weight)
                else:
                    grad_weight.addmm_(logits.t(), h.float())
        # Free logits before the next projection, else two chunks' logits coexist beside the head copy.
        del logits
    if not need_weight_grad:
        grad_weight = torch.empty(0)
    elif grad_weight is None:  # no rows
        grad_weight = torch.zeros(weight.shape, dtype = torch.float32)
    return total, grad_hidden, grad_weight


_chunked_ce_forward = torch.library.custom_op(
    "unsloth_zoo::cpu_chunked_ce_forward", _chunked_ce_forward_impl, mutates_args = (),
)


@_chunked_ce_forward.register_fake
def _chunked_ce_forward_fake(hidden, weight, labels, chunk_rows, ignore_index,
                             need_hidden_grad, need_weight_grad, exact, amp, amp_dtype, softcap = 0.0,
                             logit_scale = 1.0):
    grad_dtype = torch.float32 if exact else hidden.dtype
    return (
        torch.empty((), dtype = torch.float32),
        torch.empty(hidden.shape, dtype = grad_dtype) if need_hidden_grad else torch.empty(0),
        torch.empty(weight.shape, dtype = torch.float32) if need_weight_grad else torch.empty(0),
    )


# torch 2.8+: whether this backward pass keeps the graph (retain_graph); older torch scales out of place.
_KEEP_GRAPH = getattr(torch._C._autograd, "_get_current_graph_task_keep_graph", None)
_TOP_SAVED_HOOKS = getattr(torch._C._autograd, "_top_saved_tensors_default_hooks", None)


def _owned_ptr(buffer):
    """The forward's head-gradient address, or None under compile / a saved-tensor hook (which may hand back a
    shared or substituted buffer that must not be scaled in place)."""
    if torch.compiler.is_compiling():
        return None
    try:
        if _TOP_SAVED_HOOKS is not None and _TOP_SAVED_HOOKS(True) is not None:
            return None
        return buffer.data_ptr()
    except (RuntimeError, TypeError):  # fake / functional tensors have no storage
        return None


def _scale_in_place_ok(owned, buffer):
    """In place only in an eager backward that frees the graph (a retained graph runs backward again on the same
    buffer), without create_graph, on the exact buffer forward made."""
    if owned is None or _KEEP_GRAPH is None or torch.compiler.is_compiling() or torch.is_grad_enabled():
        return False
    if os.environ.get("UNSLOTH_CPU_HEAD_GRAD_REUSE", "1") == "0":
        return False
    if type(buffer) is not torch.Tensor or buffer.requires_grad or not buffer.is_contiguous():
        return False
    try:
        if buffer.data_ptr() != owned or _KEEP_GRAPH():
            return False
    except RuntimeError:  # not inside a backward pass
        return False
    return True


_GRAPH_TASK_ID = getattr(torch._C, "_current_graph_task_id", None)
_WILL_EXECUTE = getattr(torch._C, "_will_engine_execute_node", None)


def _will_execute(node):
    try:
        return node is not None and bool(_WILL_EXECUTE(node))
    except (RuntimeError, TypeError):
        return False


class _TiedHeadGradLink:
    """Moves the chunked CE's head gradient to the tied embedding's backward in the same pass. The CE then returns
    no weight gradient and the embedding returns the head gradient with its own rows added, so whatever else reaches
    the shared weight (a second forward, a regulariser) still accumulates through autograd as usual."""

    __slots__ = ("weight", "node", "grad", "task")

    def __init__(self, weight):
        self.weight, self.node, self.grad, self.task = weight, None, None, None

    def offer(self, grad):
        self.grad, self.task = grad, _GRAPH_TASK_ID()

    def claim(self):
        grad, self.grad = self.grad, None  # never keep a reference: autograd adopts the buffer only if it is the last
        if grad is None or self.task != _GRAPH_TASK_ID():
            return None
        return grad


class _TiedEmbedding(torch.autograd.Function):
    """F.embedding that returns the tied head gradient plus its rows when the chunked CE handed it over."""

    @staticmethod
    def forward(ctx, input_ids, weight, padding_idx, link):
        ctx.save_for_backward(input_ids)
        ctx.num_weights, ctx.padding_idx, ctx.link = weight.shape[0], padding_idx, link
        return F.embedding(input_ids, weight, padding_idx)

    @staticmethod
    def backward(ctx, grad_output):
        input_ids, = ctx.saved_tensors
        padding_idx = -1 if ctx.padding_idx is None else ctx.padding_idx
        head_grad = ctx.link.claim()
        if head_grad is None:
            if not ctx.needs_input_grad[1]:
                return None, None, None, None
            return None, torch.ops.aten.embedding_dense_backward(
                grad_output, input_ids, ctx.num_weights, padding_idx, False), None, None
        ids = input_ids.reshape(-1)
        grad = grad_output.reshape(ids.numel(), -1)
        if padding_idx >= 0:
            keep = ids != padding_idx
            ids, grad = ids[keep], grad[keep]
        # The same kernel over the distinct tokens sums each row in the same order as the dense backward, and each row
        # is then added once: bitwise what autograd's head grad + embedding grad gives, without the [V, H] zeros.
        rows, inverse = torch.unique(ids, return_inverse = True)
        summed = torch.ops.aten.embedding_dense_backward(grad, inverse, rows.numel(), -1, False)
        head_grad.index_add_(0, rows, summed.to(head_grad.dtype))
        return None, head_grad, None, None


def _tied_embedding_link(model, head_weight, input_ids, inputs_embeds):
    """A link when the training forward can fold the tied input embedding's gradient into the head gradient, else
    None: eager only, a plain nn.Embedding sharing the trainable leaf head weight, no hooks beyond transformers'
    enable_input_require_grads (a no-op here: the output already requires grad)."""
    if os.environ.get("UNSLOTH_CPU_HEAD_GRAD_REUSE", "1") == "0" or None in (_GRAPH_TASK_ID, _KEEP_GRAPH, _WILL_EXECUTE):
        return None
    if input_ids is None or inputs_embeds is not None or torch.compiler.is_compiling() or not torch.is_grad_enabled():
        return None
    if head_weight is None or not head_weight.requires_grad or not head_weight.is_leaf or head_weight.is_sparse:
        return None
    embed = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if type(embed) is not torch.nn.Embedding or embed.weight is not head_weight or "forward" in vars(embed):
        return None
    if embed.max_norm is not None or embed.scale_grad_by_freq or embed.sparse or input_ids.dtype not in (torch.long, torch.int):
        return None
    if embed._forward_pre_hooks or embed._backward_hooks or getattr(embed, "_backward_pre_hooks", None):
        return None
    if any(getattr(hook, "__name__", "") != "make_inputs_require_grads" for hook in embed._forward_hooks.values()):
        return None
    if torch.nn.modules.module._global_forward_hooks or torch.nn.modules.module._global_forward_pre_hooks:
        return None
    return _TiedHeadGradLink(head_weight)


class _ChunkedLinearCrossEntropy(torch.autograd.Function):
    """Sums CE(hidden @ weight.T, labels) / divisor; grads are formed per chunk in forward, so peak extra memory
    is one (chunk, vocab) fp32 block in both eager and compiled calls."""

    @staticmethod
    def forward(ctx, hidden, weight, labels, divisor, chunk_rows, ignore_index, softcap = 0.0, logit_scale = 1.0,
                head_grad_link = None):
        need_hidden_grad = hidden.requires_grad
        need_weight_grad = weight.requires_grad
        # fp16 cannot hold these gradients before GradScaler multiplies them: keep them fp32 until backward.
        exact = hidden.dtype == torch.float16 or _cpu_fp16_autocast()
        # The surrounding autograd.Function supplies backward; these are precomputed, non-differentiable buffers.
        compute = _chunked_ce_forward if torch.compiler.is_compiling() else _chunked_ce_forward_impl
        amp, amp_dtype = _cpu_autocast_state()
        total, grad_hidden, grad_weight = compute(
            hidden.detach(), weight.detach(), labels, chunk_rows, ignore_index,
            need_hidden_grad, need_weight_grad, exact, amp, amp_dtype, float(softcap), float(logit_scale),
        )
        saved = (grad_hidden, grad_weight)
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
        ctx.owned_weight_grad = _owned_ptr(grad_weight) if need_weight_grad else None
        ctx.head_grad_link = head_grad_link if head_grad_link is not None and head_grad_link.weight is weight else None
        ctx.embed_node = None
        if head_grad_link is not None:
            # The embedding's node lives on this ctx, not on the link the embedding holds (that would be a cycle).
            ctx.embed_node, head_grad_link.node = head_grad_link.node, None
        ctx.has_hidden = need_hidden_grad
        ctx.has_weight = need_weight_grad
        ctx.weight_dtype = weight.dtype
        ctx.hidden_dtype = hidden.dtype
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_hidden, grad_weight = ctx.saved_tensors[:2]
        # All labels ignored: NaN loss like mean CE, zero grad; num_items_in_batch == 0 too.
        if ctx.divisor is None:
            divisor = ctx.saved_tensors[2]
            empty = torch.isnan(divisor) | (divisor == 0)
            scale = torch.where(empty, torch.zeros_like(grad_output),
                                grad_output / torch.where(empty, torch.ones_like(divisor), divisor))
        else:
            empty = math.isnan(ctx.divisor) or ctx.divisor == 0
            scale = torch.zeros_like(grad_output) if empty else grad_output / ctx.divisor
        dh = (grad_hidden * scale.to(grad_hidden.dtype)).to(ctx.hidden_dtype) if ctx.has_hidden else None
        dw = None
        if ctx.has_weight:
            if _scale_in_place_ok(ctx.owned_weight_grad, grad_weight):
                # Scale the op's own [V, H] buffer in place (bitwise equal to the product) and hand it on, so it is
                # adopted as .grad instead of a fresh 622 MB copy; a unit scale (plain backward()) skips the pass.
                if not bool(scale == 1):
                    grad_weight.mul_(scale)
                dw = grad_weight.to(ctx.weight_dtype)
                # Hand the gradient to the tied embedding only if its backward runs in this pass (it is reachable and
                # needed); it then returns head + embedding rows as one gradient.
                if ctx.head_grad_link is not None and _will_execute(ctx.embed_node):
                    ctx.head_grad_link.offer(dw)
                    dw = None
            else:
                dw = (grad_weight * scale).to(ctx.weight_dtype)
        return dh, dw, None, None, None, None, None, None, None


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
    from unsloth_zoo.gradient_checkpointing import _torch_checkpoint
    checkpoint = _torch_checkpoint()
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
    metrics = {
        "log_sum_sq_probs": lambda z, y: torch.logsumexp(2 * torch.log_softmax(z, dim = -1), dim = -1),
        "mean_logits": lambda z, y: z.mean(-1),
        "is_top1": lambda z, y: z.argmax(-1) == y,
    }
    metrics = {name: metric for name, metric in metrics.items() if name in outputs}  # only the requested ones
    if metrics:
        # Metrics only (TRL marks them non-differentiable): one more pass without autograd.
        with torch.no_grad():
            parts = {name: [] for name in metrics}
            for start in range(0, hidden.shape[0], rows):
                z, _ = _transformed_logits(hidden[start:start + rows], weight, bias, inv_t, softcap, logit_scale)
                for name, metric in metrics.items():
                    parts[name].append(metric(z, targets[start:start + rows]))
            extras = {name: torch.cat(values) for name, values in parts.items()}
    return log_probs, entropy, extras.get("log_sum_sq_probs"), extras.get("mean_logits"), extras.get("is_top1")


def cpu_chunked_log_probs(hidden, weight, bias, targets, temperature, chunk_size,
                          final_logit_softcapping = None, logit_scale = 1.0, outputs = _LOGPROB_OUTPUTS):
    """Torch twin of TRL 1.x's Triton fused-LM-head `ChunkedLogProbFunction`, in row chunks (no [N, V] logits)."""
    if torch.is_autocast_enabled(hidden.device.type):
        dtype = torch.get_autocast_dtype(hidden.device.type)
        hidden, weight = hidden.to(dtype), weight.to(dtype)
        bias = bias.to(dtype) if bias is not None else None
    elif weight.dtype != hidden.dtype or (bias is not None and bias.dtype != hidden.dtype):
        # Autocast off (fp32 trainable lm_head, bf16=False eval): compute in hidden's dtype, as TRL's kernel does.
        # `.to` is differentiable, so a trainable head still gets its gradient in its own dtype.
        weight = weight.to(hidden.dtype)
        bias = bias.to(hidden.dtype) if bias is not None else None
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


_TRL_LOGPROB_ARGS = ("hidden", "weight", "bias", "targets", "temperature", "chunk_size",
                     "final_logit_softcapping", "logit_scale", "outputs")
# Appended args -> extra outputs (TRL 1.17).
_TRL_LOGPROB_EXTRA_ARGS = {"support_ids": 0, "score_ids": 1}


def _trl_logprob_contract(kernel):
    """(argument names after ctx, output count, known)."""
    try:
        names = tuple(inspect.signature(kernel.forward).parameters)[1:]
    except (AttributeError, TypeError, ValueError):
        return _TRL_LOGPROB_ARGS, 5, True  # No kernel (Triton missing) or no readable forward: the 1.x contract.
    head, extra = names[:len(_TRL_LOGPROB_ARGS)], names[len(_TRL_LOGPROB_ARGS):]
    known = head == _TRL_LOGPROB_ARGS and all(name in _TRL_LOGPROB_EXTRA_ARGS for name in extra)
    return names, 5 + sum(_TRL_LOGPROB_EXTRA_ARGS.get(name, 0) for name in extra), known


def patch_trl_fused_lm_head():
    """TRL 1.x SFTTrainer always computes loss through a fused LM head whose Triton kernel is None without Triton, so
    CPU training and evaluate() fail; keep it on CUDA/XPU, route other devices to cpu_chunked_log_probs. Idempotent.
    TRL 1.17 unpacks six outputs, so the torch path pads; id-set calls or an unknown contract stay on TRL's kernel."""
    try:
        from trl.trainer import utils as trl_utils
    except Exception:
        return False
    if not hasattr(trl_utils, "_ChunkedLogProbFunction"):
        return False  # TRL without the fused head: nothing to route.
    kernel = trl_utils._ChunkedLogProbFunction
    if getattr(kernel, "_unsloth_cpu_dispatch", False):
        return True
    names, n_outputs, known = _trl_logprob_contract(kernel)

    class _DeviceDispatch:
        _unsloth_cpu_dispatch = True
        _kernel = kernel

        @staticmethod
        def apply(hidden, *args, **kwargs):
            if kernel is not None and (hidden.device.type in ("cuda", "xpu") or not known):
                return kernel.apply(hidden, *args, **kwargs)
            bound = dict(zip(names[1:], args))
            bound.update(kwargs)
            if kernel is not None and any(bound.get(name) is not None for name in _TRL_LOGPROB_EXTRA_ARGS):
                return kernel.apply(hidden, *args, **kwargs)
            result = cpu_chunked_log_probs(hidden, *args[:len(_TRL_LOGPROB_ARGS) - 1],
                                           **{k: v for k, v in kwargs.items() if k in _TRL_LOGPROB_ARGS})
            if isinstance(result, tuple) and len(result) < n_outputs:
                result = result + (None,) * (n_outputs - len(result))
            return result

    trl_utils._ChunkedLogProbFunction = _DeviceDispatch
    return True


def patch_cpu_sdpa_gqa():
    """CPU SDPA takes enable_gqa masks (torch >= 2.5) to skip K/V repeats; CUDA rejects them."""
    try:
        from transformers.integrations import sdpa_attention
    except Exception:
        return False
    original = getattr(sdpa_attention, "use_gqa_in_sdpa", None)
    if original is None:
        return False
    if getattr(original, "_unsloth_cpu", False):
        return True
    from packaging.version import Version
    if Version(torch.__version__.split("+")[0]) < Version("2.5.0"):
        return False

    @functools.wraps(original)
    def use_gqa_in_sdpa(attention_mask, key, *args, **kwargs):
        if original(attention_mask, key, *args, **kwargs):
            return True
        value = args[0] if args else kwargs.get("value", key)
        return (
            isinstance(key, torch.Tensor) and not isinstance(key, torch.fx.Proxy) and key.device.type == "cpu"
            and isinstance(value, torch.Tensor) and key.shape[-1] == value.shape[-1]
        )

    use_gqa_in_sdpa._unsloth_cpu = True
    sdpa_attention.use_gqa_in_sdpa = use_gqa_in_sdpa
    return True


# Training attention calls that ran in float32 (tests and benchmarks read it).
SDPA_FP32_ENGAGED = [0]


def _bf16_sdpa_in_fp32_pays():
    """Arm CPUs: torch's bf16 CPU flash attention backward is far slower than the fp32 one. On a Linux aarch64
    Neoverse N2 (Cobalt 100, 4 threads) a Qwen3-0.6B-shaped forward + backward took 142 ms in bf16 against 7.9 ms
    in fp32 at 128 tokens, and 5.4 s against 0.27 s at 1024; bf16 SFT / DPO / full FT steps were 23-36% faster with
    attention in fp32, and a 512-token SFT step 10.6x. On x86 with native bf16 (EPYC 9V45 / 9V74) the bf16 kernel is
    the faster one (161 against 257 ms at 1024 tokens) and fp32 made that SFT step 2-11% slower, so x86 keeps it.
    UNSLOTH_CPU_SDPA_FP32=0 / 1 turns it off / on anywhere."""
    setting = os.environ.get("UNSLOTH_CPU_SDPA_FP32", "").strip().lower()
    if setting in ("0", "false", "no", "off"):
        return False
    if setting in ("1", "true", "yes", "on"):
        return True
    import platform
    return platform.machine().lower() in ("aarch64", "arm64")


def _bf16_to_fp32(value):
    return value.float() if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16 else value


def _fp32_attention_call(module, query, key, kwargs):
    """A call that needs a gradient, or one a no-grad reference forward makes the same way: full-sequence
    attention, or non-causal attention (encoder self-attention, cross-attention). Left out: causal self-attention
    with fewer queries than keys, i.e. a decode step against a KV cache."""
    if query.requires_grad and torch.is_grad_enabled():
        return True
    if query.shape[-2] == key.shape[-2]:
        return True
    causal = kwargs.get("is_causal", None)
    if causal is None:
        causal = getattr(module, "is_causal", True)
    return not causal


def patch_cpu_bf16_sdpa_fp32():
    """bf16 attention on CPU computed by the fp32 SDPA kernel: bf16 query / key / value (and a floating mask) are
    upcast exactly, the output is rounded back to bf16 and gradients flow back through the casts in bf16. Taken by
    every call whose query needs a gradient and by the no-grad calls a reference forward makes (full-sequence and
    non-causal attention), so no-grad reference / old-policy forwards of DPO and GRPO see the same attention as the
    policy forward they are compared with. Decode steps against a KV cache keep the bf16 kernel: the cache would
    need an upcast per token."""
    if not _bf16_sdpa_in_fp32_pays():
        return False
    try:
        from transformers.integrations import sdpa_attention
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    except Exception:
        return False
    if getattr(original, "_unsloth_cpu_fp32", False):
        return True

    @functools.wraps(original)
    def sdpa_attention_forward(module, query, key, value, attention_mask, *args, **kwargs):
        if not (isinstance(query, torch.Tensor) and isinstance(key, torch.Tensor) and query.device.type == "cpu"
                and query.dtype == torch.bfloat16 and _fp32_attention_call(module, query, key, kwargs)):
            return original(module, query, key, value, attention_mask, *args, **kwargs)
        if kwargs.get("cache", None) is not None:  # a paged cache would be written with fp32 keys / values
            return original(module, query, key, value, attention_mask, *args, **kwargs)
        if not torch.compiler.is_compiling():
            SDPA_FP32_ENGAGED[0] += 1
        if isinstance(attention_mask, torch.Tensor) and attention_mask.is_floating_point():
            attention_mask = attention_mask.float()
        # Other bf16 tensors (T5's position_bias) join the mask or the scores, so they are upcast too.
        args = tuple(_bf16_to_fp32(arg) for arg in args)
        kwargs = {name: _bf16_to_fp32(arg) for name, arg in kwargs.items()}
        with torch.autocast("cpu", enabled = False):
            output, weights = original(module, query.float(), key.float(), value.float(), attention_mask,
                                       *args, **kwargs)
        return output.to(query.dtype), weights

    sdpa_attention_forward._unsloth_cpu_fp32 = True
    # The registry's local entry wins over its global one (patch_sdpa_bool_causal_mask writes it too); a patch
    # installed later wraps the module attribute, so it chains to this one.
    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_attention_forward
    if getattr(sdpa_attention, "sdpa_attention_forward", None) is original:
        sdpa_attention.sdpa_attention_forward = sdpa_attention_forward
    return True


# torch's CPU 2-norm of one float32 tensor sums every square into float32 partials: on a 151936 x 1024 embedding
# gradient it is 2% off the float64 norm (6e-5 at 3072 x 1024) and runs at ~10 GB/s. One float32 norm per row (a
# row is at most a few thousand elements) combined in float64 is accurate to 1e-9 and reads the gradient faster.
_ROWWISE_NORM_MIN_NUMEL = 1 << 20


def _rowwise_eligible(grad):
    return (
        isinstance(grad, torch.Tensor) and grad.device.type == "cpu" and grad.dtype == torch.float32
        and grad.layout == torch.strided and grad.dim() >= 2 and grad.numel() >= _ROWWISE_NORM_MIN_NUMEL
        and type(grad) is torch.Tensor
    )


def patch_cpu_grad_total_norm():
    """clip_grad_norm_ (Trainer's max_grad_norm, via accelerate) computes the total 2-norm with one torch norm per
    gradient. For large float32 CPU gradients that norm is inexact (2% on Qwen3-0.6B's embedding) and slow (22% of a
    full finetuning step). Those gradients get row-wise float32 norms combined in float64; every other tensor, device,
    dtype and norm type keeps torch's own norm, and the result keeps torch's dtype and non-finite check. Idempotent;
    returns True when installed (torch >= 2.6, where clip_grad_norm_ calls clip_grad._get_total_norm)."""
    try:
        from torch.nn.utils import clip_grad
    except Exception:
        return False
    original = getattr(clip_grad, "_get_total_norm", None)
    if original is None:
        return False
    if getattr(original, "_unsloth_cpu", False):
        return True

    @functools.wraps(original)
    def _get_total_norm(tensors, norm_type = 2.0, error_if_nonfinite = False, foreach = None):
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        tensors = list(tensors)
        try:
            two = float(norm_type) == 2.0
        except (TypeError, ValueError):
            two = False
        if not two or foreach or not any(_rowwise_eligible(t) for t in tensors) or \
                any(not isinstance(t, torch.Tensor) or t.device.type != "cpu" or t.is_complex() for t in tensors):
            return original(tensors, norm_type, error_if_nonfinite, foreach)
        with torch.no_grad():
            return _rowwise_total_norm(tensors, norm_type, error_if_nonfinite)

    _get_total_norm._unsloth_cpu = True
    _get_total_norm._unsloth_original = original
    clip_grad._get_total_norm = _get_total_norm
    return True


def _square_sum_float64(norms):
    """Sum of squares of a list of norms in float64, in a constant number of ops (one cat, one cast, one dot)."""
    if len({n.dtype for n in norms}) > 1:
        norms = [n.double() for n in norms]
    flat = torch.cat([n.reshape(-1) for n in norms]).double()
    return torch.dot(flat, flat)


def _rowwise_total_norm(tensors, norm_type, error_if_nonfinite):
    # One norm op per tensor, as torch's own path: row norms for the large float32 ones, a plain norm for the rest.
    rows, others = [], []
    for t in tensors:
        if _rowwise_eligible(t):
            rows.append(torch.linalg.vector_norm(t.reshape(-1, t.shape[-1]), 2.0, dim = -1))
        else:
            others.append(torch.linalg.vector_norm(t, 2.0))
    square = _square_sum_float64(rows + others)
    # torch promotes the per-tensor norm dtypes; for the real tensors that reach here they equal the inputs' dtypes.
    dtype = functools.reduce(torch.promote_types, (n.dtype for n in rows + others))
    total_norm = square.sqrt().to(dtype)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from `parameters` is non-finite, so it cannot be "
            "clipped. To disable this error and scale the gradients by the non-finite norm anyway, set "
            "`error_if_nonfinite=False`"
        )
    return total_norm


def _lora_input_cast_is_undone(layer, x, dtype):
    """True when lora_A would only cast x back: a bf16 / fp16 CPU input, fp32 adapters, CPU autocast in x's dtype,
    and one vanilla adapter behind an identity dropout (DoRA and friends read the fp32 input themselves; several
    active adapters share PEFT's fp32 copy, so their input grads are summed in fp32)."""
    if not isinstance(x, torch.Tensor) or dtype != torch.float32 or x.dtype not in (torch.bfloat16, torch.float16):
        return False
    if x.device.type != "cpu" or not torch.is_autocast_enabled("cpu") or torch.get_autocast_dtype("cpu") != x.dtype:
        return False
    variants = getattr(layer, "lora_variant", None) or {}
    dropouts = getattr(layer, "lora_dropout", None)
    lora_A = getattr(layer, "lora_A", None)
    if dropouts is None or lora_A is None:
        return False
    active = [adapter for adapter in layer.active_adapters if adapter in lora_A]
    if len(active) != 1:
        return False
    adapter = active[0]
    return adapter not in variants and type(lora_A[adapter]) is torch.nn.Linear and \
        type(dropouts[adapter]) is torch.nn.Identity


def patch_peft_lora_autocast_input():
    """Skip PEFT's fp32 LoRA input upcast that CPU autocast casts back exactly (extra copies)."""
    try:
        from peft.tuners.lora.layer import Linear as LoraLinear
    except Exception:
        return False
    original = getattr(LoraLinear, "_cast_input_dtype", None)
    if original is None:
        return False
    if getattr(original, "_unsloth_cpu", False):
        return True

    @functools.wraps(original)
    def _cast_input_dtype(self, x, dtype):
        if _lora_input_cast_is_undone(self, x, dtype):
            return x
        return original(self, x, dtype)

    _cast_input_dtype._unsloth_cpu = True
    LoraLinear._cast_input_dtype = _cast_input_dtype
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


def set_bf16_stored_dtype(model, dtype, frozen_only = False):
    """Bf16StoredLinear to dtype (lossless); frozen upcasts revert; modules_to_save kept exact. frozen_only: upcast
    frozen weights only, so a trainable one (modules_to_save) keeps its dtype and its optimizer state's."""
    moved = 0
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if not isinstance(module, Bf16StoredLinear) or weight is None or weight.dtype == dtype:
            continue
        if dtype == torch.bfloat16:
            if not getattr(module, "_unsloth_upcast", False) or weight.requires_grad:
                continue
            module._unsloth_upcast = False
        elif weight.dtype == torch.bfloat16 and not (frozen_only and weight.requires_grad):
            module._unsloth_upcast = True
        else:
            continue
        weight.data = weight.data.to(dtype)
        moved += 1
    return moved


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


def patch_cpu_router_aux_ga(model):
    """Weight the MoE router aux loss a stock transformers forward adds in full (router_aux_loss_coef x aux_loss)
    by each micro-batch's token share, local / num_items_in_batch, as zoo's GPU forwards do
    (fused_losses/aux_loss.py). Unweighted, every one of G accumulated micro-batches carried the whole term, so
    gradient accumulation trained with G times the coefficient of one batch of the same rows. Per instance; only
    forwards whose source adds exactly that term (zoo's own rewriter recognises it). Returns True when installed."""
    if os.environ.get("UNSLOTH_CPU_ROUTER_AUX_GA", "1") == "0":
        return False
    # Absolute, and here rather than at module level: fused_losses stays out of the CPU backend's import.
    from unsloth_zoo.fused_losses.aux_loss import rewrite_aux_loss_ga, unsloth_ga_scale_aux_loss
    target = next((m for m in model.modules() if "router_aux_loss_coef" in vars(m)), None)
    if target is None or vars(target).get("_unsloth_cpu_router_aux_ga", False):
        return False
    class_forward = type(target).forward
    try:
        source = inspect.getsource(inspect.unwrap(class_forward))
    except (OSError, TypeError):
        return False
    if rewrite_aux_loss_ga(source) == source:
        return False

    def forward(self, *args, **kwargs):
        out = class_forward(self, *args, **kwargs)
        labels = kwargs.get("labels")
        n_items = kwargs.get("num_items_in_batch")
        aux = getattr(out, "aux_loss", None)
        coef = getattr(self, "router_aux_loss_coef", None)
        if getattr(out, "loss", None) is None or aux is None or labels is None or n_items is None or not coef:
            return out
        weighted = unsloth_ga_scale_aux_loss(aux, labels, kwargs.get("attention_mask"), kwargs, n_items, model = self)
        out.loss = out.loss + (coef * (weighted - aux)).to(device = out.loss.device, dtype = out.loss.dtype)
        return out

    forward.__wrapped__ = class_forward
    # A deepcopy (TRL reference models) rebinds the method to the copy.
    target.forward = types.MethodType(forward, target)
    target._unsloth_cpu_router_aux_ga = True
    return True


def patch_causal_conv1d_fallback(model):
    """Without oneDNN (macOS wheels), routes modeling modules' causal_conv1d_fn to cpu_causal_conv1d when a probe agrees; with oneDNN the grouped conv is faster."""
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
                                 ignore_index = -100, shift_labels = None, chunk_rows = None,
                                 softcap = None, logit_scale = 1.0, head_grad_link = None):
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
    elif torch.compiler.is_compiling():
        # A Python count becomes symbolic once it changes between steps; float()/math.isnan on it cannot be traced.
        divisor = torch.full((), num_items_in_batch, dtype = torch.float32, device = hidden.device)
    else:
        divisor = float(num_items_in_batch)
    # Skip ignored rows' LM head work here in eager; compiled calls drop them inside the opaque forward op.
    if not torch.compiler.is_compiling():
        valid = shift_labels != ignore_index
        if bool(valid.any()) and not bool(valid.all()):
            rows = valid.nonzero().squeeze(-1)
            hidden, shift_labels = hidden.index_select(0, rows), shift_labels.index_select(0, rows)
    if chunk_rows is None:
        # ~64 MB of fp32 logits per chunk.
        chunk_rows = max(16, int(os.environ.get("UNSLOTH_CPU_CE_CHUNK_ELEMENTS", 16 * 1024 * 1024)) // weight.shape[0])
    return _ChunkedLinearCrossEntropy.apply(
        hidden, weight, shift_labels, divisor, chunk_rows, ignore_index, float(softcap or 0.0), float(logit_scale),
        head_grad_link,
    )


# Model types whose causal-LM loss is plain CE over lm_head(hidden); others (soft caps, logit scales, z-loss, router losses) keep their own forward.
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
    return _plain_head_and_inner(model)


def _plain_head_and_inner(model):
    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if not isinstance(lm_head, torch.nn.Linear) or lm_head.bias is not None:
        return False
    return _inner_model_name(model) is not None


def _inner_model_name(model):
    """base_model_prefix, else `model` (transformers 4.x Gemma 3 text declares `language_model`)."""
    for name in (getattr(model, "base_model_prefix", "model"), "model"):
        if isinstance(getattr(model, name, None), torch.nn.Module):
            return name
    return None


# Causal LMs whose forward transforms lm_head(hidden) before the same CE: class -> (model_type, knob, how).
_CHUNKED_CE_TRANSFORMED = {
    "Gemma2ForCausalLM": ("gemma2", "final_logit_softcapping", "softcap"),
    "Gemma3ForCausalLM": ("gemma3_text", "final_logit_softcapping", "softcap"),
    "CohereForCausalLM": ("cohere", "logit_scale", "multiply"),
    "Cohere2ForCausalLM": ("cohere2", "logit_scale", "multiply"),
    "GraniteForCausalLM": ("granite", "logits_scaling", "divide"),
}
_LOGIT_KNOBS = ("final_logit_softcapping", "logits_soft_cap", "logit_scale", "logits_scaling", "output_multiplier_scale")


def _lm_head_transform(model):
    """(softcap, logit_scale) of the logits the model's forward feeds its CE, or None when chunked CE cannot match."""
    if _plain_lm_head(model):
        return None, 1.0
    if type(model).__name__ not in _CHUNKED_CE_TRANSFORMED or not _plain_head_and_inner(model):
        return None
    return _config_transform(model)


def _config_transform(model):
    """The transform read from the config as the forward reads it on every call; None if chunked CE cannot match."""
    config = getattr(model, "config", None)
    model_type, knob, how = _CHUNKED_CE_TRANSFORMED[type(model).__name__]
    # Top-level config only: a composite (vision) config keeps its own forward.
    if getattr(config, "model_type", None) != model_type:
        return None
    for name in _LOGIT_KNOBS:
        if name != knob and getattr(config, name, None) not in (None, 1, 1.0):
            return None
    if getattr(config, "output_router_logits", False) or getattr(config, "num_experts", None) \
            or getattr(config, "num_local_experts", None) or getattr(config, "n_routed_experts", None):
        return None
    value = getattr(config, knob, None)
    if how == "multiply":
        value = getattr(model, "logit_scale", value)  # Cohere's forward reads self.logit_scale
    if value is None and how == "softcap":
        return None, 1.0
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return None
    value = float(value)
    if how == "softcap":
        return value, 1.0
    return None, (value if how == "multiply" else 1.0 / value)


def _plain_head_weight(head):
    """The weight that alone produces this head's logits, or None (e.g. a LoRA-wrapped lm_head)."""
    if not isinstance(head, torch.nn.Module) or "forward" in vars(head) or any(
        getattr(head, name, None) for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
    ):
        return None  # hooks / an instance forward (user code, DeepSpeed ZeRO-3 gathers) act on the module call
    if type(head) is torch.nn.Linear and head.bias is None:
        return head.weight
    if type(head).__name__ == "ModulesToSaveWrapper":
        original = getattr(head, "original_module", None)
        if getattr(head, "disable_adapters", False):
            return _plain_head_weight(original)  # PEFT's disable_adapter(): the wrapper runs the base head
        adapter = getattr(head, "active_adapter", None)
        adapter = adapter[0] if isinstance(adapter, (list, tuple)) else adapter
        copies = getattr(head, "modules_to_save", None)
        active = copies[adapter] if copies is not None and adapter in copies else None
        return _plain_head_weight(active if active is not None else original)
    return None


@functools.lru_cache(maxsize = None)
def _forward_takes_index_logits_to_keep(cls):
    """True when cls.forward slices with transformers' `slice(-k, None) if isinstance(logits_to_keep, int) else
    logits_to_keep`, or (newer transformers, after 5.18) hands it to `slice_logits_to_keep`, which indexes with a
    1D tensor the same way; i.e. it also takes a tensor of positions."""
    try:
        source = inspect.getsource(cls.forward)
    except (OSError, TypeError):
        return False
    if "slice_logits_to_keep(hidden_states, logits_to_keep)" in source:
        return True
    return "isinstance(logits_to_keep, int)" in source and "slice(-logits_to_keep, None)" in source


def _logits_to_keep_as_index(model, input_ids, inputs_embeds, logits_to_keep):
    """Int logits_to_keep is a strided view torch < 2.14 won't fold into one GEMM for bf16/fp16
    heads (matmul copies head per row); an index tensor avoids it, bit-identical; fp32 folds."""
    if not isinstance(logits_to_keep, int) or isinstance(logits_to_keep, bool) or logits_to_keep <= 0:
        return logits_to_keep
    tokens = input_ids if input_ids is not None else inputs_embeds
    if tokens is None or tokens.dim() < 2 or tokens.shape[1] <= logits_to_keep:
        return logits_to_keep  # decode steps (one new token) and nothing-to-slice calls stop here
    weight = getattr(model.get_output_embeddings(), "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.dtype not in (torch.bfloat16, torch.float16) \
            or weight.device.type != "cpu" or not _forward_takes_index_logits_to_keep(type(model)):
        return logits_to_keep
    length = tokens.shape[1]
    return torch.arange(length - logits_to_keep, length, device = tokens.device)


def patch_chunked_causal_lm_loss(model):
    """Routes training forwards with labels through chunked_linear_cross_entropy, per instance; logits come back None,
    so unsloth_zoo.cpu.trainer makes TRL skip logits-only metrics. Returns True when installed."""
    if os.environ.get("UNSLOTH_CPU_CHUNKED_CE", "1") == "0":
        return False
    if _lm_head_transform(model) is None:
        return False
    if getattr(model, "_unsloth_cpu_chunked_ce", False):
        return True
    transformed = not _plain_lm_head(model)
    from transformers.modeling_outputs import CausalLMOutputWithPast

    original_forward = model.forward
    # Call class forward on self: deepcopy rebinds the outer method, else source weights run.
    original_func = original_forward.__func__ if getattr(original_forward, "__self__", None) is model else None
    inner_name = _inner_model_name(model)
    try:
        params = inspect.signature(original_forward).parameters
    except (TypeError, ValueError):
        params = {}
    cache_kwarg = "cache" if "cache" in params else "past_key_values"
    output_cache_field = "cache" if "cache" in getattr(CausalLMOutputWithPast, "__dataclass_fields__", {}) \
        else "past_key_values"

    def forward(self, input_ids = None, attention_mask = None, position_ids = None,
                past_key_values = None, inputs_embeds = None, labels = None, use_cache = None,
                logits_to_keep = 0, **kwargs):
        cache = kwargs.pop("cache", None)
        if past_key_values is None:
            past_key_values = cache
        shift_labels = kwargs.pop("shift_labels", None)
        head_weight = _plain_head_weight(self.get_output_embeddings())
        # Re-read per call: a cap or scale changed after patching must not train on the old one.
        transform = _config_transform(self) if transformed else (None, 1.0)
        if transform is None:
            head_weight = None  # the stock forward
        softcap, logit_scale = transform or (None, 1.0)
        logps_targets = vars(self).get("_unsloth_cpu_logps_targets", None)
        if logps_targets is not None and head_weight is not None and labels is None and shift_labels is None:
            # Head runs in forward so DDP / FSDP see a trainable lm_head as used (trainer.py).
            kwargs.pop("return_dict", None)
            outputs = getattr(self, inner_name)(
                input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids,
                **{cache_kwarg: past_key_values}, inputs_embeds = inputs_embeds, use_cache = use_cache, **kwargs,
            )
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            batch_size, length = logps_targets.shape
            # targets[:, t] is scored by position t: L - 1 targets (ORPO) drop the last position, L (DPO / KTO) keep it.
            hidden = hidden[:, :length].to(head_weight.dtype)
            wanted = ("mean_logits", "log_sum_sq_probs") if vars(self).get("_unsloth_cpu_logps_sq", False) else ("mean_logits",)
            log_probs, _, sq, mean_logits, _ = cpu_chunked_log_probs(
                hidden.reshape(batch_size * length, hidden.shape[-1]), head_weight, None, logps_targets.reshape(-1), 1.0, None,
                final_logit_softcapping = softcap, logit_scale = logit_scale, outputs = wanted,
            )
            result = {"log_probs": log_probs.view(batch_size, length), "mean_logits": mean_logits.view(batch_size, length)}
            if sq is not None:
                result["log_sum_sq_probs"] = sq.view(batch_size, length)
            return result
        if (labels is None and shift_labels is None) or not torch.is_grad_enabled() \
                or not getattr(self, "_unsloth_cpu_chunked_ce_active", False) or head_weight is None:
            if shift_labels is not None:
                kwargs["shift_labels"] = shift_labels
            logits_to_keep = _logits_to_keep_as_index(self, input_ids, inputs_embeds, logits_to_keep)
            call = functools.partial(original_func, self) if original_func is not None else original_forward
            return call(
                input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids,
                **{cache_kwarg: past_key_values}, inputs_embeds = inputs_embeds, labels = labels,
                use_cache = use_cache, logits_to_keep = logits_to_keep, **kwargs,
            )
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)
        ignore_index = kwargs.pop("ignore_index", -100)
        kwargs.pop("return_dict", None)
        # Full fine-tuning with tied embeddings: the embedding's gradient rows go into the head gradient buffer.
        link = _tied_embedding_link(self, head_weight, input_ids, inputs_embeds)
        if link is not None:
            embed = self.get_input_embeddings()
            inputs_embeds, input_ids = _TiedEmbedding.apply(input_ids, head_weight, embed.padding_idx, link), None
            link.node = inputs_embeds.grad_fn
        outputs = getattr(self, inner_name)(
            input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids,
            **{cache_kwarg: past_key_values}, inputs_embeds = inputs_embeds, use_cache = use_cache,
            **kwargs,
        )
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        loss = chunked_linear_cross_entropy(
            hidden, head_weight, labels,
            num_items_in_batch = num_items_in_batch, shift_labels = shift_labels, ignore_index = ignore_index,
            softcap = softcap, logit_scale = logit_scale, head_grad_link = link,
        )
        return CausalLMOutputWithPast(
            loss = loss, logits = None,
            **{output_cache_field: outputs.get("cache", outputs.get("past_key_values")) if hasattr(outputs, "get")
               else getattr(outputs, "past_key_values", None)},
            hidden_states = getattr(outputs, "hidden_states", None),
            attentions = getattr(outputs, "attentions", None),
        )

    model.forward = types.MethodType(forward, model)
    model._unsloth_cpu_chunked_ce = True
    return True
