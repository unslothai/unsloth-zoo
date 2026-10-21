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
"""Fused MoE router: logits -> (fp32 weights [T, k], int64 ids [T, k]) in one Triton launch,
plus a Triton backward matching eager autograd for the logits.

Modes:
  MODE_SOFTMAX_TOPK  softmax(fp32) -> top_k -> optional renorm (Qwen3/Qwen2/OLMoE/Mixtral MoE).
  MODE_TOPK_SOFTMAX  top_k on logits -> softmax over the selected (gpt-oss).
  MODE_SIGMOID       sigmoid + e_score_correction_bias -> group top_k -> UNBIASED scores ->
                     renorm (+1e-20) * routed_scaling_factor (DeepSeek-V3, GLM4-MoE). Masked
                     groups use -inf (transformers 5) or 0.0 (transformers 4).

Ties pick the lower index (same set as CUDA torch.topk; order inside exact ties may differ).
Logits a float32 ulp apart can rank differently from eager. NaN unsupported; bias gets no grad.
Opt-in: UNSLOTH_MOE_FUSED_ROUTER=1, read every eager call.
"""

from __future__ import annotations

__all__ = [
    "moe_router",
    "moe_router_reference",
    "fused_router_enabled",
    "fused_route_or_none",
    "ROUTER_CALLS",
    "patch_moe_fused_router",
    "enable_fused_router",
]

import contextlib
import inspect
import os
import re
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .common import TEMPORARY_PATCHES

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

MODE_SOFTMAX_TOPK = 0
MODE_TOPK_SOFTMAX = 1
MODE_SIGMOID = 2
_MODES = {"softmax_topk": MODE_SOFTMAX_TOPK, "topk_softmax": MODE_TOPK_SOFTMAX, "sigmoid": MODE_SIGMOID}

ROUTER_CALLS = {"fused": 0}
_SUPPORTED = {}


def fused_router_enabled():
    return os.environ.get("UNSLOTH_MOE_FUSED_ROUTER", "0") == "1"


if HAS_TRITON:
    @triton.jit
    def _router_fwd_kernel(
        Z, BIAS, W, IDS, T, E, GS, stride_z, scale, mask_value,
        K: tl.constexpr, BLOCK_K: tl.constexpr, MODE: tl.constexpr, RENORM: tl.constexpr,
        HAS_BIAS: tl.constexpr, N_GROUP: tl.constexpr, TOPK_GROUP: tl.constexpr, BLOCK_G: tl.constexpr,
        BLOCK_T: tl.constexpr, BLOCK_E: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        cols = tl.arange(0, BLOCK_E)
        row_ok = rows < T
        col_ok = cols < E
        ok = row_ok[:, None] & col_ok[None, :]
        z = tl.load(Z + rows[:, None].to(tl.int64) * stride_z + cols[None, :], mask = ok, other = 0.0).to(tl.float32)
        NEG = float("-inf")
        BIG = BLOCK_E + 1

        if MODE == 0:
            zm = tl.max(tl.where(col_ok[None, :], z, NEG), axis = 1)
            ez = tl.where(col_ok[None, :], tl.exp(z - zm[:, None]), 0.0)
            p = ez / tl.sum(ez, axis = 1)[:, None]
            choice = p
        elif MODE == 1:
            choice = z
        else:
            s = 1.0 / (1.0 + tl.exp(-z))
            choice = s
            if HAS_BIAS:
                b = tl.load(BIAS + cols, mask = col_ok, other = 0.0).to(tl.float32)
                choice = s + b[None, :]
            if N_GROUP > 1:
                gid = cols // GS
                gcols = tl.arange(0, BLOCK_G)
                gscore = tl.full((BLOCK_T, BLOCK_G), NEG, tl.float32)
                for g in tl.static_range(N_GROUP):
                    in_g = (gid == g) & col_ok
                    v = tl.where(in_g[None, :], choice, NEG)
                    m1 = tl.max(v, axis = 1)
                    i1 = tl.min(tl.where(in_g[None, :] & (v == m1[:, None]), cols[None, :], BIG), axis = 1)
                    m2 = tl.max(tl.where(cols[None, :] == i1[:, None], NEG, v), axis = 1)
                    gscore = tl.where(gcols[None, :] == g, (m1 + m2)[:, None], gscore)
                g_ok = gcols < N_GROUP
                g_sel = tl.zeros((BLOCK_T, BLOCK_G), tl.int32)
                for j in tl.static_range(TOPK_GROUP):
                    avail = g_ok[None, :] & (g_sel == 0)
                    gm = tl.max(tl.where(avail, gscore, NEG), axis = 1)
                    gi = tl.min(tl.where(avail & (gscore == gm[:, None]), gcols[None, :], BLOCK_G + 1), axis = 1)
                    g_sel = tl.where(gcols[None, :] == gi[:, None], 1, g_sel)
                keep = tl.zeros((BLOCK_T, BLOCK_E), tl.int32)
                for g in tl.static_range(N_GROUP):
                    sel_g = tl.max(tl.where(gcols[None, :] == g, g_sel, 0), axis = 1)
                    keep = tl.where((gid == g)[None, :] & (sel_g[:, None] > 0), 1, keep)
                choice = tl.where(keep > 0, choice, mask_value)

        kcols = tl.arange(0, BLOCK_K)
        taken = tl.zeros((BLOCK_T, BLOCK_E), tl.int32)
        vals = tl.zeros((BLOCK_T, BLOCK_K), tl.float32)
        ids = tl.zeros((BLOCK_T, BLOCK_K), tl.int32)
        for j in tl.static_range(K):
            avail = col_ok[None, :] & (taken == 0)
            m = tl.max(tl.where(avail, choice, NEG), axis = 1)
            idx = tl.min(tl.where(avail & (choice == m[:, None]), cols[None, :], BIG), axis = 1)
            idx = tl.minimum(idx, E - 1)
            hit = cols[None, :] == idx[:, None]
            taken = tl.where(hit, 1, taken)
            if MODE == 2:
                v = tl.sum(tl.where(hit, s, 0.0), axis = 1)
            else:
                v = m
            vals = tl.where(kcols[None, :] == j, v[:, None], vals)
            ids = tl.where(kcols[None, :] == j, idx[:, None], ids)

        k_ok = kcols < K
        if MODE == 1:
            vm = tl.max(tl.where(k_ok[None, :], vals, NEG), axis = 1)
            ev = tl.where(k_ok[None, :], tl.exp(vals - vm[:, None]), 0.0)
            w = ev / tl.sum(ev, axis = 1)[:, None]
        else:
            w = vals
            if RENORM:
                den = tl.sum(tl.where(k_ok[None, :], w, 0.0), axis = 1)
                if MODE == 2:
                    den = den + 1e-20
                w = w / den[:, None]
        w = w * scale
        out_ok = row_ok[:, None] & k_ok[None, :]
        offs = rows[:, None].to(tl.int64) * K + kcols[None, :]
        tl.store(W + offs, w, mask = out_ok)
        tl.store(IDS + offs, ids.to(tl.int64), mask = out_ok)

    @triton.jit
    def _router_bwd_kernel(
        Z, IDS, DW, DZ, T, E, stride_z, stride_dw, scale,
        K: tl.constexpr, MODE: tl.constexpr, RENORM: tl.constexpr,
        BLOCK_T: tl.constexpr, BLOCK_E: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        cols = tl.arange(0, BLOCK_E)
        row_ok = rows < T
        col_ok = cols < E
        ok = row_ok[:, None] & col_ok[None, :]
        r64 = rows.to(tl.int64)
        z = tl.load(Z + r64[:, None] * stride_z + cols[None, :], mask = ok, other = 0.0).to(tl.float32)
        NEG = float("-inf")
        g = tl.zeros((BLOCK_T, BLOCK_E), tl.float32)
        sel = tl.zeros((BLOCK_T, BLOCK_E), tl.int32)
        for j in tl.static_range(K):
            idx = tl.load(IDS + r64 * K + j, mask = row_ok, other = -1).to(tl.int32)
            dw = tl.load(DW + r64 * stride_dw + j, mask = row_ok, other = 0.0).to(tl.float32)
            hit = cols[None, :] == idx[:, None]
            g = tl.where(hit, dw[:, None] * scale, g)
            sel = tl.where(hit, 1, sel)
        is_sel = (sel > 0) & ok

        if MODE == 0:
            zm = tl.max(tl.where(col_ok[None, :], z, NEG), axis = 1)
            ez = tl.where(col_ok[None, :], tl.exp(z - zm[:, None]), 0.0)
            p = ez / tl.sum(ez, axis = 1)[:, None]
            if RENORM:
                den = tl.sum(tl.where(is_sel, p, 0.0), axis = 1)
                w = p / den[:, None]
                gw = tl.sum(tl.where(is_sel, g * w, 0.0), axis = 1)
                dp = tl.where(is_sel, (g - gw[:, None]) / den[:, None], 0.0)
            else:
                dp = tl.where(is_sel, g, 0.0)
            dz = p * (dp - tl.sum(dp * p, axis = 1)[:, None])
        elif MODE == 1:
            zm = tl.max(tl.where(is_sel, z, NEG), axis = 1)
            ez = tl.where(is_sel, tl.exp(z - zm[:, None]), 0.0)
            w = ez / tl.sum(ez, axis = 1)[:, None]
            gw = tl.sum(tl.where(is_sel, g * w, 0.0), axis = 1)
            dz = tl.where(is_sel, w * (g - gw[:, None]), 0.0)
        else:
            s = 1.0 / (1.0 + tl.exp(-z))
            if RENORM:
                den = tl.sum(tl.where(is_sel, s, 0.0), axis = 1) + 1e-20
                gs = tl.sum(tl.where(is_sel, g * s, 0.0), axis = 1)
                ds = tl.where(is_sel, g / den[:, None] - gs[:, None] / (den * den)[:, None], 0.0)
            else:
                ds = tl.where(is_sel, g, 0.0)
            dz = ds * s * (1.0 - s)
        tl.store(DZ + r64[:, None] * E + cols[None, :], dz.to(DZ.dtype.element_ty), mask = ok)


def _blocks(E):
    BLOCK_E = max(16, triton.next_power_of_2(E))
    BLOCK_T = max(1, min(64, 2048 // BLOCK_E))
    return BLOCK_T, BLOCK_E


def _device_guard(device):
    if device.index is None or device.index == torch.cuda.current_device():
        return contextlib.nullcontext()
    return torch.cuda.device(device)


def _router_fwd_impl(logits, bias, top_k, mode, renorm, n_group, topk_group, scale, mask_zero):
    ROUTER_CALLS["fused"] += 1   # counted at run time (opaque to Dynamo: no guard, no recompile)
    T, E = logits.shape
    w = torch.empty((T, top_k), device = logits.device, dtype = torch.float32)
    ids = torch.empty((T, top_k), device = logits.device, dtype = torch.int64)
    if T == 0:
        return w, ids
    z = logits if logits.stride(1) == 1 else logits.contiguous()
    BLOCK_T, BLOCK_E = _blocks(E)
    has_bias = bias is not None
    b = bias.contiguous() if has_bias else z
    with _device_guard(z.device):
        _router_fwd_kernel[(triton.cdiv(T, BLOCK_T),)](
            z, b, w, ids, T, E, max(1, E // max(n_group, 1)), z.stride(0), float(scale),
            0.0 if mask_zero else float("-inf"),
            K = top_k, BLOCK_K = max(2, triton.next_power_of_2(top_k)), MODE = mode, RENORM = bool(renorm),
            HAS_BIAS = has_bias, N_GROUP = max(n_group, 1), TOPK_GROUP = max(topk_group, 1),
            BLOCK_G = max(2, triton.next_power_of_2(max(n_group, 1))),
            BLOCK_T = BLOCK_T, BLOCK_E = BLOCK_E, num_warps = 4,
        )
    return w, ids


@torch.library.custom_op("unsloth_zoo::moe_router_fwd", mutates_args = ())
def _router_fwd_op(
    logits: torch.Tensor, bias: Optional[torch.Tensor], top_k: int, mode: int, renorm: bool,
    n_group: int, topk_group: int, scale: float, mask_zero: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _router_fwd_impl(logits, bias, top_k, mode, renorm, n_group, topk_group, scale, mask_zero)


@_router_fwd_op.register_fake
def _(logits, bias, top_k, mode, renorm, n_group, topk_group, scale, mask_zero):
    T = logits.shape[0]
    return (logits.new_empty((T, top_k), dtype = torch.float32),
            logits.new_empty((T, top_k), dtype = torch.int64))


def _router_bwd_impl(logits, ids, dw, mode, renorm, scale):
    T, E = logits.shape
    dz = torch.empty((T, E), device = logits.device, dtype = logits.dtype)
    if T == 0:
        return dz
    z = logits if logits.stride(1) == 1 else logits.contiguous()
    ids = ids.contiguous()
    dw = dw if dw.stride(-1) == 1 else dw.contiguous()
    BLOCK_T, BLOCK_E = _blocks(E)
    with _device_guard(z.device):
        _router_bwd_kernel[(triton.cdiv(T, BLOCK_T),)](
            z, ids, dw, dz, T, E, z.stride(0), dw.stride(0), float(scale),
            K = ids.shape[1], MODE = mode, RENORM = bool(renorm),
            BLOCK_T = BLOCK_T, BLOCK_E = BLOCK_E, num_warps = 4,
        )
    return dz


@torch.library.custom_op("unsloth_zoo::moe_router_bwd", mutates_args = ())
def _router_bwd_op(
    logits: torch.Tensor, ids: torch.Tensor, dw: torch.Tensor, mode: int, renorm: bool, scale: float,
) -> torch.Tensor:
    return _router_bwd_impl(logits, ids, dw, mode, renorm, scale)


@_router_bwd_op.register_fake
def _(logits, ids, dw, mode, renorm, scale):
    return torch.empty_like(logits, memory_format = torch.contiguous_format)


def _setup_context(ctx, inputs, output):
    logits, bias, top_k, mode, renorm, n_group, topk_group, scale, mask_zero = inputs
    ctx.mode, ctx.renorm, ctx.scale = mode, renorm, scale
    ctx.save_for_backward(logits, output[1])
    ctx.mark_non_differentiable(output[1])


def _backward(ctx, dw, dids):
    logits, ids = ctx.saved_tensors
    dz = None
    if ctx.needs_input_grad[0]:
        dz = _router_bwd_op(logits, ids, dw.float(), ctx.mode, ctx.renorm, ctx.scale)
    return dz, None, None, None, None, None, None, None, None


_router_fwd_op.register_autograd(_backward, setup_context = _setup_context)


class _RouterFunction(torch.autograd.Function):
    """Eager path: the same kernels without the custom-op dispatch (tens of us per call)."""
    @staticmethod
    def forward(ctx, logits, bias, top_k, mode, renorm, n_group, topk_group, scale, mask_zero):
        w, ids = _router_fwd_impl(logits, bias, top_k, mode, renorm, n_group, topk_group, scale, mask_zero)
        ctx.mode, ctx.renorm, ctx.scale = mode, renorm, scale
        ctx.save_for_backward(logits, ids)
        ctx.mark_non_differentiable(ids)
        return w, ids

    @staticmethod
    def backward(ctx, dw, dids):
        logits, ids = ctx.saved_tensors
        dz = None
        if ctx.needs_input_grad[0]:
            dz = _router_bwd_impl(logits, ids, dw.float(), ctx.mode, ctx.renorm, ctx.scale)
        return dz, None, None, None, None, None, None, None, None


def moe_router_reference(logits, top_k, mode = "softmax_topk", renorm = False, bias = None,
                         n_group = 1, topk_group = 1, scale = 1.0, mask_zero = False):
    """Eager routing with the same semantics (and the same lowest-index tie policy)."""
    mode = _MODES.get(mode, mode)
    if mode == MODE_SOFTMAX_TOPK:
        p = F.softmax(logits, dim = -1, dtype = torch.float32)
        w, ids = torch.topk(p, top_k, dim = -1)
        if renorm:
            w = w / w.sum(dim = -1, keepdim = True)
    elif mode == MODE_TOPK_SOFTMAX:
        v, ids = torch.topk(logits.float(), top_k, dim = -1)
        w = F.softmax(v, dim = -1)
    else:
        s = logits.float().sigmoid()
        choice = s if bias is None else s + bias.float()
        T, E = choice.shape
        if n_group > 1:
            gscore = choice.view(T, n_group, E // n_group).topk(2, dim = -1)[0].sum(dim = -1)
            gidx = torch.topk(gscore, topk_group, dim = -1)[1]
            gmask = torch.zeros_like(gscore).scatter_(1, gidx, 1)
            keep = gmask.unsqueeze(-1).expand(T, n_group, E // n_group).reshape(T, E).bool()
            choice = choice.masked_fill(~keep, 0.0 if mask_zero else float("-inf"))
        ids = torch.topk(choice, top_k, dim = -1)[1]
        w = s.gather(1, ids)
        if renorm:
            w = w / (w.sum(dim = -1, keepdim = True) + 1e-20)
    return w * scale, ids


def _supported(logits, top_k, n_group):
    if not HAS_TRITON or not isinstance(logits, torch.Tensor) or logits.dim() != 2 \
            or logits.device.type != "cuda" \
            or logits.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return False
    E = logits.shape[1]
    if not (1 <= top_k <= min(E, 64)) or E > 4096:
        return False
    if n_group > 1 and (E % n_group != 0 or E // n_group < 2 or n_group > 64):
        return False
    return True


def moe_router(logits, top_k, mode = "softmax_topk", renorm = False, bias = None,
               n_group = 1, topk_group = 1, scale = 1.0, mask_zero = False):
    """Fused router (fp32 weights [T, k], int64 ids [T, k]); eager reference off CUDA."""
    if not _supported(logits, top_k, n_group):
        return moe_router_reference(logits, top_k, mode, renorm, bias, n_group, topk_group, scale, mask_zero)
    args = (logits, bias, int(top_k), int(_MODES.get(mode, mode)), bool(renorm),
            int(n_group), int(topk_group), float(scale), bool(mask_zero))
    if torch.compiler.is_compiling():
        return _router_fwd_op(*args)
    return _RouterFunction.apply(*args)


def fused_route_or_none(logits, top_k, mode = "softmax_topk", renorm = False, bias = None,
                        n_group = 1, topk_group = 1, scale = 1.0, mask_zero = False):
    """moe_router when UNSLOTH_MOE_FUSED_ROUTER=1 and the input is supported, else None."""
    if not fused_router_enabled() or not _supported(logits, top_k, n_group):
        return None
    if n_group > 1 and not (1 <= topk_group <= n_group and top_k <= topk_group * (logits.shape[1] // n_group)):
        return None
    return moe_router(logits, top_k, mode, renorm, bias, n_group, topk_group, scale, mask_zero)


# transformers router classes. A class is patched only when its forward (and get_topk_indices)
# reads exactly as below, so a reworded upstream router keeps its own forward.
_SOFTMAX_HEAD = (
    "hidden_states = hidden_states.reshape(-1, self.hidden_dim)",
    "router_logits = F.linear(hidden_states, self.weight)",
)
_SOFTMAX_TAIL = ("router_scores = router_top_value", "return router_logits, router_scores, router_indices")
_TOPK = "router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)"
_DIV = "router_top_value /= router_top_value.sum(dim=-1, keepdim=True)"
_CAST = "router_top_value = router_top_value.to(router_logits.dtype)"
_SM = "router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)"
_ROUTER_FORMS = {
    # renorm: "attr" = self.norm_topk_prob, True = always; cast to the logits dtype or keep fp32.
    "softmax_attr_cast": (_SOFTMAX_HEAD + (_SM, _TOPK, "if self.norm_topk_prob:", _DIV, _CAST) + _SOFTMAX_TAIL, None),
    "softmax_true_cast": (_SOFTMAX_HEAD + (_SM, _TOPK, _DIV, _CAST) + _SOFTMAX_TAIL, None),
    "softmax_true_fp32": (_SOFTMAX_HEAD + (
        "router_probs = torch.nn.functional.softmax(router_logits.float(), dim=-1)", _TOPK, _DIV) + _SOFTMAX_TAIL, None),
    "gpt_oss": ((
        "router_logits = F.linear(hidden_states, self.weight, self.bias)",
        "router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)",
        "router_scores = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)",
        "return router_logits, router_scores, router_indices"), None),
}
_SIGMOID_GROUPS = (
    "group_scores = (",
    "scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group)",
    ".topk(2, dim=-1)[0]", ".sum(dim=-1)", ")",
    "group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]",
    "group_mask = torch.zeros_like(group_scores)",
    "group_mask.scatter_(1, group_idx, 1)",
    "score_mask = (", "group_mask.unsqueeze(-1)",
    ".expand(-1, self.num_group, self.num_experts // self.num_group)",
    ".reshape(-1, self.num_experts)", ")",
    'scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))',
    "topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]",
)
_SIGMOID_WEIGHTS = (
    "topk_weights = scores.gather(1, topk_indices)",
    "if self.norm_topk_prob:",
    "denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20",
    "topk_weights /= denominator",
    "topk_weights = topk_weights * self.routed_scaling_factor",
)
_ROUTER_FORMS["sigmoid_v5"] = ((
    "hidden_states = hidden_states.view(-1, self.hidden_dim)",
    "router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))",
    "scores = router_logits.sigmoid()",
    "scores_for_choice = scores + self.e_score_correction_bias",
) + _SIGMOID_GROUPS + _SIGMOID_WEIGHTS + ("return router_logits, topk_weights, topk_indices",), None)
# transformers 4: selection in a no_grad get_topk_indices, unselected groups filled with 0.0.
_ROUTER_FORMS["sigmoid_v4"] = ((
    "hidden_states = hidden_states.view(-1, self.config.hidden_size)",
    "router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))",
    "scores = router_logits.sigmoid()",
    "topk_indices = self.get_topk_indices(scores)",
) + _SIGMOID_WEIGHTS + ("return topk_indices, topk_weights",), (
    "scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)",
    "group_scores = (",
    "scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)",
    ".topk(2, dim=-1)[0]", ".sum(dim=-1)", ")",
    "group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]",
    "group_mask = torch.zeros_like(group_scores)",
    "group_mask.scatter_(1, group_idx, 1)",
    "score_mask = (", "group_mask.unsqueeze(-1)",
    ".expand(-1, self.n_group, self.n_routed_experts // self.n_group)",
    ".reshape(-1, self.n_routed_experts)", ")",
    "scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)",
    "topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]",
    "return topk_indices",
))


def _body_lines(fn):
    try:
        src = inspect.getsource(fn)
    except Exception:
        return None
    lines = [re.sub(r"\s+#.*$", "", x).strip() for x in src.splitlines()]
    while lines and not lines[0].startswith("def "):   # decorators
        lines.pop(0)
    return tuple(x for x in lines[1:] if x and not x.startswith("#"))


def _router_form(cls):
    fwd = cls.__dict__.get("forward")
    if fwd is None or getattr(fwd, "_unsloth_fused_router", False):
        return None
    body = _body_lines(fwd)
    # Unsloth's compiler rewrites forward as a call to a module-level `{Class}_forward` copy.
    if body == (f"return {cls.__name__}_forward(self, hidden_states=hidden_states)",):
        import sys
        fn = getattr(sys.modules.get(cls.__module__), f"{cls.__name__}_forward", None)
        body = _body_lines(fn) if fn is not None else None
    for name, (lines, topk_lines) in _ROUTER_FORMS.items():
        if body != lines:
            continue
        if topk_lines is not None:
            gti = cls.__dict__.get("get_topk_indices")
            if gti is None or _body_lines(gti) != topk_lines:
                return None
        return name
    return None


def _make_router_forward(form, original):
    if form.startswith("softmax"):
        def forward(self, hidden_states):
            if not fused_router_enabled():
                return original(self, hidden_states)
            hidden_states = hidden_states.reshape(-1, self.hidden_dim)
            router_logits = F.linear(hidden_states, self.weight)
            renorm = True if "_true_" in form else bool(self.norm_topk_prob)
            r = fused_route_or_none(router_logits, self.top_k, "softmax_topk", renorm)
            if r is None:
                return original(self, hidden_states)
            w, ids = r
            if form.endswith("_cast"):
                w = w.to(router_logits.dtype)
            return router_logits, w, ids
    elif form == "gpt_oss":
        def forward(self, hidden_states):
            if not fused_router_enabled():
                return original(self, hidden_states)
            router_logits = F.linear(hidden_states, self.weight, self.bias)
            r = fused_route_or_none(router_logits, self.top_k, "topk_softmax")
            if r is None:
                return original(self, hidden_states)
            return router_logits, r[0].to(router_logits.dtype), r[1]
    else:
        v5 = form == "sigmoid_v5"
        def forward(self, hidden_states):
            if not fused_router_enabled():
                return original(self, hidden_states)
            if v5:
                hidden_states = hidden_states.view(-1, self.hidden_dim)
                n_group = self.num_group
            else:
                hidden_states = hidden_states.view(-1, self.config.hidden_size)
                n_group = self.n_group
            router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
            r = fused_route_or_none(
                router_logits, self.top_k, "sigmoid", bool(self.norm_topk_prob), self.e_score_correction_bias,
                n_group, self.topk_group, float(self.routed_scaling_factor), mask_zero = not v5,
            )
            if r is None:
                return original(self, hidden_states)
            w, ids = r
            return (router_logits, w, ids) if v5 else (ids, w)
    forward._unsloth_fused_router = True
    forward.__wrapped__ = original
    return forward


_ROUTER_CLASSES = (
    ("qwen3_moe", "Qwen3MoeTopKRouter"), ("qwen2_moe", "Qwen2MoeTopKRouter"), ("olmoe", "OlmoeTopKRouter"),
    ("mixtral", "MixtralTopKRouter"), ("qwen3_vl_moe", "Qwen3VLMoeTextTopKRouter"),
    ("qwen3_5_moe", "Qwen3_5MoeTopKRouter"), ("gpt_oss", "GptOssTopKRouter"),
    ("deepseek_v3", "DeepseekV3TopkRouter"), ("glm4_moe", "Glm4MoeTopkRouter"),
    ("glm4_moe_lite", "Glm4MoeLiteTopkRouter"),
)


def patch_moe_fused_router():
    """Swap the eager routing in matching transformers router classes for moe_router.
    Installed only under UNSLOTH_MOE_FUSED_ROUTER=1; each call still re-reads the env."""
    if not fused_router_enabled() or not HAS_TRITON:
        return
    import importlib
    for module_name, cls_name in _ROUTER_CLASSES:
        try:
            module = importlib.import_module(f"transformers.models.{module_name}.modeling_{module_name}")
        except Exception:
            continue
        cls = getattr(module, cls_name, None)
        if cls is None:
            continue
        form = _router_form(cls)
        if form is None:
            continue
        cls.forward = _make_router_forward(form, cls.__dict__["forward"])
        _SUPPORTED[cls_name] = form


def enable_fused_router(model):
    """Loader hook: also patch router classes Unsloth's compiler re-created (compiled-cache
    modules), which the import-time class patch cannot reach. Returns the patched class names."""
    if not fused_router_enabled() or not HAS_TRITON or model is None or not hasattr(model, "modules"):
        return []
    names = {name for _, name in _ROUTER_CLASSES}
    done = []
    for module in model.modules():
        cls = type(module)
        if cls.__name__ not in names or getattr(cls.__dict__.get("forward"), "_unsloth_fused_router", False):
            continue
        form = _router_form(cls)
        if form is None:
            continue
        cls.forward = _make_router_forward(form, cls.__dict__["forward"])
        _SUPPORTED[f"{cls.__module__}.{cls.__name__}"] = form
        done.append(cls.__name__)
    return done


TEMPORARY_PATCHES.append(patch_moe_fused_router)
