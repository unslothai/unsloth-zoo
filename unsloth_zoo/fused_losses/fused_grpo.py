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

"""GRPO-family policy loss with its gradient taken in the forward (TRL GRPOTrainer._compute_loss math).

Every token-level GRPO loss is sum_i w_i * l(logp_i; side inputs) + sum_i e_i * H_i, where the weights w_i / e_i
depend only on the completion mask and constants (loss-type normaliser, gradient accumulation), and l is a closed
form of the token's own log-prob. So d loss / d logp_i and d loss / d H_i are known as soon as a chunk's statistics
are, while its logits are still in the buffer: dz is written in the same pass and the head GEMMs run there. The
backward only scales the stored gradients by the incoming scalar. Against the log-prob op this drops the logits
recompute: trainable head 4 -> 3 head GEMMs, frozen head with an entropy gradient 3 -> 2.

Not closed in one pass (the caller keeps the log-prob op + a torch loss): sequence-level importance sampling
(GSPO), VESPO's sequence gamma weights, the off-policy mask and the top-entropy-quantile mask (both read every
token's log-prob / entropy before any coefficient is known).
"""

import os
from typing import Optional

import torch

from . import fused_linear as fl

__all__ = ["fused_linear_grpo_loss", "grpo_token_weights", "GRPO_FWD_GRAD"]

GRPO_FWD_GRAD = os.environ.get("UNSLOTH_FUSED_LINEAR_GRPO_FWD_GRAD", "1") != "0"
_OP = "fused_linear_grpo_v1"
# 0: clipped surrogate (grpo, bnpo, dr_grpo, dapo, luspo), 1: cispo, 2: sapo.
_KIND = {"grpo": 0, "bnpo": 0, "dr_grpo": 0, "dapo": 0, "luspo": 0, "cispo": 1, "sapo": 2}


def _token_terms(lp, adv, old, ref, isr, kind, eps_low, eps_high, delta, beta, sapo_pos, sapo_neg, bias_kl):
    """Per-token loss l and dl/dlogp, as TRL builds them (gradients follow torch.clamp / torch.minimum backward)."""
    r = torch.exp(lp - old)
    if kind == 0:
        r1, d1 = r, r
        if delta > 0.0:
            r1, d1 = torch.clamp(r, max = delta), r * (r <= delta)
        lo, hi = 1.0 - eps_low, 1.0 + eps_high
        r2, d2 = torch.clamp(r, lo, hi), r * ((r >= lo) & (r <= hi))
        t1, t2 = r1 * adv, r2 * adv
        # torch.minimum: ties send half the gradient to each side.
        s1 = (t1 < t2).to(lp.dtype) + 0.5 * (t1 == t2).to(lp.dtype)
        loss = -torch.minimum(t1, t2)
        grad = -adv * (s1 * d1 + (1.0 - s1) * d2)
    elif kind == 1:
        c = torch.clamp(r, max = eps_high)
        loss = -c * adv * lp
        grad = -c * adv
    else:
        tau = torch.where(adv > 0, torch.full_like(adv, sapo_pos), torch.full_like(adv, sapo_neg))
        s = torch.sigmoid(tau * (r - 1.0))
        loss = -s * 4.0 / tau * adv
        grad = -4.0 * adv * s * (1.0 - s) * r
    if isr is not None:
        loss, grad = loss * isr, grad * isr
    if beta != 0.0:
        d = ref - lp
        e = torch.exp(d)
        kl, dkl = e - d - 1.0, 1.0 - e
        if bias_kl:
            kl, dkl = kl * r, dkl * r + kl * r
        loss, grad = loss + beta * kl, grad + beta * dkl
    return loss, grad


def grpo_token_weights(mask, loss_type, *, num_items_in_batch = None, max_completion_length = None,
                       gradient_accumulation_steps = 1, steps_per_generation = 1, num_processes = 1,
                       entropy_coef = 0.0, training = True):
    """(w, e): per-token weights of the policy loss and the entropy term for a [B, T] completion mask (TRL 1.15)."""
    m = mask.to(torch.float32)
    B = m.shape[0]
    ga = float(gradient_accumulation_steps) if training else 1.0
    if loss_type in ("grpo", "sapo"):
        w = m / m.sum(-1, keepdim = True).clamp(min = 1.0) / B / ga
    elif loss_type == "bnpo":
        w = m / m.sum().clamp(min = 1.0) / ga
    elif loss_type == "dr_grpo":
        w = m / (B * float(max_completion_length)) / ga
    elif loss_type in ("cispo", "dapo"):
        norm = torch.as_tensor(num_items_in_batch, dtype = torch.float32, device = m.device).clamp(min = 1.0)
        norm = norm / num_processes
        if training:
            norm = norm * float(gradient_accumulation_steps) / float(steps_per_generation)
        w = m / norm
    elif loss_type == "luspo":
        w = m / B / ga
    else:
        raise ValueError(f"Unsloth: GRPO loss type {loss_type} has no forward gradient.")
    e = -float(entropy_coef) * m / m.sum().clamp(min = 1.0) / ga if entropy_coef else torch.zeros_like(m)
    return w, e


def _grpo_impl(hidden, weight, bias, targets, mask, adv, old, ref, isr, w_tok, e_tok, scaling, kind, eps_low, eps_high,
               delta, beta, sapo_pos, sapo_neg, bias_kl, temperature, mul, div, cap, compute_dtype, need_h, need_w,
               need_b, rows):
    device = weight.device
    N, H = hidden.shape
    V = weight.shape[0]
    targets = targets.contiguous()
    idx, n_keep = fl._keep_index(targets, mask, V, None)
    f32 = dict(device = device, dtype = torch.float32)
    acc = torch.float64 if device.type in ("cuda", "cpu") else torch.float32
    logps = torch.zeros(N, **f32)
    ent = torch.zeros(N, **f32)
    grad_h = torch.zeros((N, H), device = device, dtype = hidden.dtype) if need_h else hidden.new_empty(0)
    acc_w = torch.zeros(weight.shape, device = device, dtype = weight.dtype) if need_w else None
    acc_b = torch.zeros(V, **f32) if need_b else None
    loss = torch.zeros((), device = device, dtype = acc)
    inv_t, scale, cap = fl._scalars(temperature, mul, div, cap)
    grad = need_h or need_w or need_b
    has_ref, has_isr = ref.numel() > 0, isr.numel() > 0
    if n_keep > 0:
        fp32_logits, exact_mode = fl._logits_mode(device, compute_dtype)
        if compute_dtype not in fl._HALF:
            fp32_logits, exact_mode = False, False
        R = min(fl._chunk_rows(rows, n_keep, H, V, fl._bytes_per_logit(fp32_logits, compute_dtype), device), n_keep)
        buf = fl._Buffers(R, V, device, fp32_logits, compute_dtype)
        w = weight.to(compute_dtype)
        s = scaling.to(**f32).reshape(())
        full = n_keep == N
        with torch.autocast(device.type, enabled = False):
            for a in range(0, n_keep, R):
                b = min(a + R, n_keep)
                n = b - a
                pick = (lambda x: x[a:b]) if full else (lambda x: x.index_select(0, idx[a:b]))
                h = fl._rows_of(hidden, idx, a, b, n_keep, N).to(compute_dtype)
                t = pick(targets)
                x = fl._project(h, w, buf, n, fp32_logits)
                exact = fl._exact_target(h, w, bias, t) if exact_mode else None
                st, _ = fl._row_forward(device, x, None, t, bias, exact, None, (inv_t, scale, cap, 1.0, 0.0),
                                        (True, False, False, False), False)
                lp, en = st["logp"], st["ent"]
                ell, dl = _token_terms(lp, pick(adv), pick(old), pick(ref) if has_ref else None,
                                       pick(isr) if has_isr else None, kind, eps_low, eps_high, delta, beta,
                                       sapo_pos, sapo_neg, bias_kl)
                wt, et = pick(w_tok), pick(e_tok)
                loss += (ell.to(acc) * wt.to(acc)).sum() + (en.to(acc) * et.to(acc)).sum()
                if full:
                    logps[a:b].copy_(lp)
                    ent[a:b].copy_(en)
                else:
                    logps.index_copy_(0, idx[a:b], lp)
                    ent.index_copy_(0, idx[a:b], en)
                if not grad:
                    continue
                # dz = g * d logp / dz + ge * d H / dz, written over the chunk's own logits (no recompute).
                dz = fl._row_backward(device, x, buf.grad_view(n, compute_dtype), t, bias, st["lse"], st["lpt"], en,
                                      (dl * wt * s).contiguous(), (et * s).contiguous(), (inv_t, scale, cap), 0, V)
                if need_h:
                    gh = fl._dz_w(dz, w)
                    if full:
                        grad_h[a:b].copy_(gh)
                    else:
                        grad_h.index_copy_(0, idx[a:b], gh.to(grad_h.dtype))
                if need_w:
                    fl._dz_t_h(acc_w, dz, h)
                if need_b:
                    acc_b += dz.sum(0, dtype = torch.float32)
    grad_w = acc_w if need_w else hidden.new_empty(0)
    grad_b = acc_b.to(bias.dtype) if need_b else hidden.new_empty(0)
    return loss.float(), logps, ent, grad_h, grad_w, grad_b


_OPS = fl._OPS
if "grpo_impl" not in _OPS:
    from torch.library import custom_op

    @custom_op(f"unsloth_zoo::{_OP}", mutates_args = ())
    def _grpo_op(
        hidden: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], targets: torch.Tensor,
        mask: Optional[torch.Tensor], adv: torch.Tensor, old: torch.Tensor, ref: torch.Tensor, isr: torch.Tensor,
        w_tok: torch.Tensor, e_tok: torch.Tensor, scaling: torch.Tensor, kind: int, eps_low: float, eps_high: float,
        delta: float, beta: float, sapo_pos: float, sapo_neg: float, bias_kl: bool, temperature: float, mul: float,
        div: float, cap: float, compute_dtype: torch.dtype, need_h: bool, need_w: bool, need_b: bool, rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fl.CALLS["grpo_fwd_grad"] = fl.CALLS.get("grpo_fwd_grad", 0) + 1
        return _OPS["grpo_impl"](hidden, weight, bias, targets, mask, adv, old, ref, isr, w_tok, e_tok, scaling, kind,
                                 eps_low, eps_high, delta, beta, sapo_pos, sapo_neg, bias_kl, temperature, mul, div,
                                 cap, compute_dtype, need_h, need_w, need_b, rows)

    @_grpo_op.register_fake
    def _grpo_fake(hidden, weight, bias, targets, mask, adv, old, ref, isr, w_tok, e_tok, scaling, kind, eps_low,
                   eps_high, delta, beta, sapo_pos, sapo_neg, bias_kl, temperature, mul, div, cap, compute_dtype,
                   need_h, need_w, need_b, rows):
        N = hidden.shape[0]
        f32 = dict(device = weight.device, dtype = torch.float32)
        return (
            torch.empty((), **f32), torch.empty(N, **f32), torch.empty(N, **f32),
            hidden.new_empty(hidden.shape) if need_h else hidden.new_empty(0),
            weight.new_empty(weight.shape) if need_w else hidden.new_empty(0),
            bias.new_empty(bias.shape) if need_b else hidden.new_empty(0),
        )

    def _grpo_setup(ctx, inputs, output):
        loss, logps, ent, grad_h, grad_w, grad_b = output
        ctx.mark_non_differentiable(logps, ent, grad_h, grad_w, grad_b)
        ctx.set_materialize_grads(False)
        ctx.need = inputs[25:28]
        ctx.save_for_backward(grad_h, grad_w, grad_b, inputs[11])

    def _grpo_backward(ctx, grad_loss, *_):
        if grad_loss is None:
            return (None,) * 29
        grad_h, grad_w, grad_b, scaling = ctx.saved_tensors
        acc = torch.float64 if grad_loss.device.type in ("cuda", "cpu") else torch.float32
        s = grad_loss.to(acc) / scaling.to(acc)
        need_h, need_w, need_b = ctx.need
        scale = fl._stored_grad_scaler(s, (grad_h, grad_w, grad_b))
        return (scale(grad_h) if need_h else None, scale(grad_w) if need_w else None,
                scale(grad_b) if need_b else None) + (None,) * 26

    _grpo_op.register_autograd(_grpo_backward, setup_context = _grpo_setup)
_OPS["grpo_impl"] = fl._guarded(_grpo_impl)


def fused_linear_grpo_loss(
    hidden, weight, targets, *,
    mask, advantages, old_logps,
    loss_type = "grpo",
    bias = None,
    ref_logps = None, beta = 0.0, use_bias_correction_kl = False,
    epsilon_low = 0.2, epsilon_high = 0.2, delta = None,
    sapo_temperature_pos = 1.0, sapo_temperature_neg = 1.05,
    importance_sampling_ratio = None,
    entropy_coef = 0.0,
    num_items_in_batch = None, max_completion_length = None,
    gradient_accumulation_steps = 1, steps_per_generation = 1, num_processes = 1, training = True,
    temperature = 1.0,
    logit_scale_multiply = 0.0, logit_scale_divide = 0.0, logit_softcapping = 0.0,
    scaling = None,
    rows_per_chunk = None,
):
    """TRL GRPOTrainer._compute_loss (token-level importance sampling) on hidden [B, T, H] -> (loss, logps, entropy).

    The returned log-probs / entropies are metrics (no gradient); the loss carries the whole gradient. With
    UNSLOTH_FUSED_LINEAR_GRPO_FWD_GRAD=0 the same loss is built from fused_linear_logprobs + torch."""
    if loss_type not in _KIND:
        raise ValueError(f"Unsloth: GRPO loss type {loss_type} has no forward gradient.")
    device = weight.device
    B, T = mask.shape
    adv = advantages.to(device = device, dtype = torch.float32)
    adv = (adv.reshape(B, 1) if adv.numel() == B else adv.reshape(B, T)).expand(B, T)
    w_tok, e_tok = grpo_token_weights(mask.to(device), loss_type, num_items_in_batch = num_items_in_batch,
                                      max_completion_length = max_completion_length,
                                      gradient_accumulation_steps = gradient_accumulation_steps,
                                      steps_per_generation = steps_per_generation, num_processes = num_processes,
                                      entropy_coef = entropy_coef, training = training)
    old = old_logps.to(device = device, dtype = torch.float32).reshape(B, T)
    ref = ref_logps.to(device = device, dtype = torch.float32).reshape(B, T) if beta != 0.0 else None
    isr = importance_sampling_ratio.to(device = device, dtype = torch.float32).expand(B, T) \
        if importance_sampling_ratio is not None else None
    kind = _KIND[loss_type]
    args = (float(epsilon_low), float(epsilon_high), float(delta or 0.0), float(beta), float(sapo_temperature_pos),
            float(sapo_temperature_neg), bool(use_bias_correction_kl))
    if not GRPO_FWD_GRAD:
        lp, en = fl.fused_linear_logprobs(hidden, weight, targets, bias = bias, temperature = temperature,
                                          logit_scale_multiply = logit_scale_multiply,
                                          logit_scale_divide = logit_scale_divide,
                                          logit_softcapping = logit_softcapping, return_entropy = True,
                                          entropy_requires_grad = bool(entropy_coef), rows_per_chunk = rows_per_chunk)
        lp, en = lp.reshape(B, T).float(), en.reshape(B, T).float()
        ell, _ = _token_terms(lp, adv, old, ref, isr, kind, *args)
        loss = (ell * w_tok).sum() + (en * e_tok).sum()
        return loss, lp.detach(), en.detach()
    f32 = dict(device = device, dtype = torch.float32)
    if scaling is None:
        scaling = torch.ones((), **f32)
    else:
        scaling = torch.as_tensor(scaling).to(**f32).reshape(())
        scaling = torch.where(scaling == 0, torch.ones_like(scaling), scaling)
    grad_on = torch.is_grad_enabled()
    need_h = grad_on and hidden.requires_grad
    need_w = grad_on and weight.requires_grad
    need_b = grad_on and bias is not None and bias.requires_grad
    bias2 = fl._head_bias(bias, weight)
    empty = torch.empty(0, **f32)
    flat = lambda x: x.reshape(-1).contiguous() if x is not None else empty  # noqa: E731
    loss, lp, en, _, _, _ = getattr(torch.ops.unsloth_zoo, _OP)(
        hidden.reshape(-1, hidden.shape[-1]).to(device), weight, bias2, targets.reshape(-1).to(device),
        mask.reshape(-1).to(device = device, dtype = torch.bool), flat(adv), flat(old), flat(ref), flat(isr),
        flat(w_tok), flat(e_tok), scaling, kind, *args, float(temperature), float(logit_scale_multiply or 0.0),
        float(logit_scale_divide or 0.0), float(logit_softcapping or 0.0), fl._compute_dtype(hidden, weight),
        need_h, need_w, need_b, int(rows_per_chunk or 0),
    )
    return loss, lp.reshape(B, T), en.reshape(B, T)
