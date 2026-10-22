# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import functools

import torch
import triton
import triton.language as tl

from fla.ops.common.backends import dispatch
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import (
    IS_NVIDIA,
    IS_NVIDIA_BLACKWELL,
    IS_NVIDIA_HOPPER,
    TRITON_ABOVE_3_4_0,
    TRITON_ABOVE_3_7_1,
    TRITON_ABOVE_3_8_0,
    autotune_cache_kwargs,
    check_shared_mem,
)

BKV_LIST = [64, 128] if check_shared_mem() else ([32, 64] if check_shared_mem('ada') else [32])
NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]
# Unsloth: backport of fla PR #1399. Triton<3.8 races at 8 warps on Blackwell: a
# tl.dot accumulator is read from TMEM by one warp while another warp zero-fills it
# for the next tl.dot (triton-lang/triton#10590, fla-org/flash-linear-attention#1327).
# IS_NVIDIA_BLACKWELL is frozen at import from a Blackwell device when any visible
# device is one (fla_vendor._blackwell_import_device), so mixed hosts drop 8 warps too.
BLACKWELL_8_WARP_RACE = IS_NVIDIA_BLACKWELL and not TRITON_ABOVE_3_8_0
CHUNK_BWD_DV_LOCAL_NUM_WARPS = [2, 4] if BLACKWELL_8_WARP_RACE else NUM_WARPS


# Unsloth: per-device Hopper probe for the fla #640 tile workaround below.
#
# ``IS_NVIDIA_HOPPER`` (fla/utils/_device.py) is a module constant frozen at import
# time from device 0 only:
#     IS_NVIDIA_HOPPER = (IS_NVIDIA and ('NVIDIA H' in torch.cuda.get_device_name(0)
#                                        or torch.cuda.get_device_capability()[0] == 9))
# while chunk_bwd_dqkwg picks CONST_TILING per tensor via ``k.device.index``. On a
# mixed host (cuda:0 Ada/Blackwell, cuda:1 H100) the global is False even though the
# tensor lives on Hopper, so the BK=64 step-down would not fire and the miscompiled
# tile would silently corrupt dk/dg. Probe the tensor's own device instead.
#
# Only ever ORed with the global, never used to clear it, so this can add a
# step-down but never remove one.
#
# Capability, not shared memory: ``check_shared_mem('hopper', idx)`` is a
# shared-memory *tier* test (>= 232448 bytes) that Blackwell B200 also passes, so it
# is not a Hopper detector. SM90 == Hopper. HIP is excluded because an AMD Instinct
# reports capability major 9 on a ROCm build without being Hopper, matching how
# fla_vendor._hopper_dqkwg_suspect gates the bare major==9 signal.
@functools.lru_cache(maxsize=None)
def _device_is_nvidia_hopper(index):
    if not IS_NVIDIA:
        return False
    try:
        if getattr(torch.version, 'hip', None) is not None:
            return False
        if index is None:
            index = torch.cuda.current_device()
        if torch.cuda.get_device_capability(index)[0] == 9:
            return True
        return 'NVIDIA H' in torch.cuda.get_device_name(index)
    except Exception:
        return None  # unknown, not "no": see _is_hopper_tensor


def _is_hopper_tensor(x):
    """True / False for a CUDA tensor whose device we could inspect, else None.

    The tri-state matters. ``None`` means "could not tell", and the caller then
    falls back to the import-time global, so an unexpected probe failure on a real
    Hopper still gets the step-down. Collapsing that into ``False`` would fail open
    into the miscompile, which is the one direction this must never take.
    """
    try:
        if x is None or not x.is_cuda:
            return None
        return _device_is_nvidia_hopper(x.device.index)
    except Exception:
        return None


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        # Unsloth: backported from fla PR #1229. The 8-warp config hits a Blackwell tl.dot race (wrong, nondeterministic o).
        # Kept off on every Blackwell Triton: fla PR #1399 re-enables it on Triton>=3.8, not yet stress-tested here.
        *([] if IS_NVIDIA_BLACKWELL else [triton.Config({'BK': 128, 'BV': 128}, num_warps=8, num_stages=3)]),
        triton.Config({'BK': 64, 'BV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BK': 32, 'BV': 32}, num_warps=2, num_stages=3),
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'STATE_V_FIRST'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    v += (bos * HV + i_h) * V
    o += (bos * HV + i_h) * V
    h += (i_tg * HV + i_h).to(tl.int64) * K*V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_v = i_v * BV + tl.arange(0, BV)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_q = q + o_t[:, None] * (H*K) + o_k[None, :]
        p_k = k + o_k[:, None] + o_t[None, :] * (H*K)
        if STATE_V_FIRST:
            p_h = h + o_v[:, None] * K + o_k[None, :]
            m_h = (o_v[:, None] < V) & m_k[None, :]
        else:
            p_h = h + o_k[:, None] * V + o_v[None, :]
            m_h = m_k[:, None] & (o_v[None, :] < V)
        # [BT, BK]
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k[None, :], other=0.0)
        # [BK, BT]
        b_k = tl.load(p_k, mask=m_k[:, None] & m_t[None, :], other=0.0)
        b_h = tl.load(p_h, mask=m_h, other=0.0)

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        if STATE_V_FIRST:
            b_o += tl.dot(b_q, tl.trans(b_h))
        else:
            b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * HV + i_h
        p_g = g + o_t * HV
        b_g = tl.load(p_g, mask=m_t, other=0.0)
        b_o = b_o * exp2(b_g)[:, None]
        b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_o = b_o * exp2(b_g)[:, None]
        b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
    p_o = o + o_t[:, None] * (HV*V) + o_v[None, :]

    b_v = tl.load(p_v, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0)
    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & (o_v < V)[None, :])


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'USE_DW': lambda args: args['dw'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G', 'USE_G_GAMMA', 'USE_DW', 'STATE_V_FIRST'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dqkwg(
    q,
    k,
    v,
    g,
    g_gamma,
    h,
    do,
    dh,
    dq,
    dk,
    dw,
    dv,
    dg,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_DW: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV

    all = B * T
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    v += (bos * HV + i_h) * V
    do += (bos * HV + i_h) * V
    h += (i_tg * HV + i_h).to(tl.int64) * K*V
    dh += (i_tg * HV + i_h).to(tl.int64) * K*V
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    dq += (bos * HV + i_h) * K
    dk += (bos * HV + i_h) * K

    # for delta rule only
    if USE_DW:
        dw += (bos * HV + i_h) * K
        dv += (bos * HV + i_h) * V

    if USE_G:
        dg += i_k * all * HV
        b_dg_last = tl.zeros([1], dtype=tl.float32) if USE_G else None
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_g_last = b_gamma * min(BT, T - i_t * BT)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32) if USE_DW else None

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    m_qk = m_t[:, None] & m_k[None, :]
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_hv = m_t[:, None] & m_v[None, :]
        p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
        p_do = do + o_t[:, None] * (HV*V) + o_v[None, :]
        if STATE_V_FIRST:
            p_h = h + o_v[:, None] * K + o_k[None, :]
            p_dh = dh + o_v[:, None] * K + o_k[None, :]
        else:
            p_h = h + o_v[:, None] + o_k[None, :] * V
            p_dh = dh + o_v[:, None] + o_k[None, :] * V
        m_h = m_v[:, None] & m_k[None, :]
        # [BT, BV]
        b_v = tl.load(p_v, mask=m_hv, other=0.0)
        b_do = tl.load(p_do, mask=m_hv, other=0.0)
        # [BV, BK]
        b_h = tl.load(p_h, mask=m_h, other=0.0)
        b_dh = tl.load(p_dh, mask=m_h, other=0.0)
        if USE_G:
            b_dg_last += (tl.sum(b_h * b_dh))
        # [BT, BV] @ [BV, BT] -> [BT, BT]
        b_ds += tl.dot(b_do, tl.trans(b_v))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))
        if USE_DW:
            p_dv = dv + o_t[:, None] * (HV*V) + o_v[None, :]
            b_dv = tl.load(p_dv, mask=m_hv, other=0.0)
            b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype))

    if USE_DW:
        p_dw = dw + o_t[:, None] * (HV*K) + o_k[None, :]
        tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), mask=m_qk)

    tl.debug_barrier()
    p_q = q + o_t[:, None] * (H*K) + o_k[None, :]
    p_k = k + o_t[:, None] * (H*K) + o_k[None, :]
    b_q = tl.load(p_q, mask=m_qk, other=0.0)
    b_k = tl.load(p_k, mask=m_qk, other=0.0)

    p_dq = dq + o_t[:, None] * (HV*K) + o_k[None, :]
    p_dk = dk + o_t[:, None] * (HV*K) + o_k[None, :]

    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    if USE_G:
        g += bos * HV + i_h
        dg += bos * HV + i_h
        p_g = g + o_t * HV
        b_g = tl.load(p_g, mask=m_t, other=0.0)
        b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * HV)
        b_dg_last *= exp2(b_g_last)
        b_dq = b_dq * exp2(b_g)[:, None] * scale
        b_dk = b_dk * tl.where(m_t, exp2(-b_g + b_g_last), 0)[:, None]
        b_dg_last += tl.sum(b_dk * b_k)

        b_ds = tl.where(m_A, b_ds * exp2(b_g[:, None] - b_g[None, :]), 0) * scale
        b_ds = b_ds.to(b_k.dtype)
        # [BT, BK]
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)

        b_dg = tl.sum(b_dq * b_q, axis=1) - tl.sum(b_dk * b_k, axis=1)

        p_dg = dg + o_t * HV
        # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
        # b_dg = tl.dot(tl.where(o_t[:, None] <= o_t[None, :], 1., 0.), b_dg, allow_tf32=False) + b_dg_last)
        b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_qk)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_qk)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_t)

    elif USE_G_GAMMA:
        b_dq = b_dq * exp2(b_g)[:, None] * scale
        b_dk = b_dk * tl.where(m_t, exp2(-b_g + b_g_last), 0)[:, None]
        b_ds = tl.where(m_A, b_ds * exp2(b_g[:, None] - b_g[None, :]), 0) * scale
        b_ds = b_ds.to(b_k.dtype)
        # [BT, BK]
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_qk)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_qk)

    else:
        b_ds = tl.where(m_A, b_ds, 0)
        b_ds = b_ds.to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q) * scale
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_qk)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_qk)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G', 'USE_G_GAMMA', 'STATE_V_FIRST'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dv(
    q,
    k,
    g,
    g_gamma,
    do,
    dv,
    dh,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    # offset calculation
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    do += (bos * HV + i_h) * V
    dv += (bos * HV + i_h) * V
    dh += (i_tg * HV + i_h).to(tl.int64) * K*V

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_v = i_v * BV + tl.arange(0, BV)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_k = k + o_t[:, None] * (H*K) + o_k[None, :]
        p_q = q + o_k[:, None] + o_t[None, :] * (H*K)
        b_q = tl.load(p_q, mask=m_k[:, None] & m_t[None, :], other=0.0)
        b_k = tl.load(p_k, mask=m_t[:, None] & m_k[None, :], other=0.0)
        b_A += tl.dot(b_k, b_q)
        if STATE_V_FIRST:
            p_dh = dh + o_v[:, None] * K + o_k[None, :]
            b_dh = tl.trans(tl.load(p_dh, mask=(o_v[:, None] < V) & m_k[None, :], other=0.0))
        else:
            p_dh = dh + o_k[:, None] * V + o_v[None, :]
            b_dh = tl.load(p_dh, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)
        b_dv += tl.dot(b_k, b_dh.to(b_k.dtype))

    if USE_G:
        g += bos * HV + i_h
        p_g = g + o_t * HV
        b_g = tl.load(p_g, mask=m_t, other=0.0)
        b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * HV)
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_g_last = b_gamma * min(BT, T - i_t * BT)

    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    if USE_G or USE_G_GAMMA:
        b_A = tl.where(m_A, b_A * exp2(b_g[None, :] - b_g[:, None]) * scale, 0).to(do.dtype.element_ty)
        b_dv *= tl.where(m_t, exp2(-b_g + b_g_last), 0)[:, None]
    else:
        b_A = tl.where(m_A, b_A * scale, 0).to(do.dtype.element_ty)
    p_do = do + o_t[:, None] * (HV*V) + o_v[None, :]
    p_dv = dv + o_t[:, None] * (HV*V) + o_v[None, :]
    b_do = tl.load(p_do, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0)
    b_dv += tl.dot(b_A.to(b_do.dtype), b_do)
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_t[:, None] & (o_v < V)[None, :])


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'USE_A': lambda args: args['A'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in CHUNK_BWD_DV_LOCAL_NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dv_local(
    q,
    k,
    g,
    g_gamma,
    A,
    do,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_A: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    do += (bos * HV + i_h) * V
    dv += (bos * HV + i_h) * V

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    if USE_A:
        p_A = A + (bos * HV + i_h) * BT + tl.arange(0, BT)[:, None] + o_t[None, :] * (HV*BT)
        b_A = tl.load(p_A, mask=m_t[None, :], other=0.0)
    else:
        if USE_G:
            g += bos * HV + i_h
            p_g = g + o_t * HV
            b_g = tl.load(p_g, mask=m_t, other=0.0)
        if USE_G_GAMMA:
            b_gamma = tl.load(g_gamma + i_h)
            b_g = b_gamma * (tl.arange(0, BT) + 1)

        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            p_k = k + o_t[:, None] * (H*K) + o_k[None, :]
            p_q = q + o_k[:, None] + o_t[None, :] * (H*K)

            b_k = tl.load(p_k, mask=m_t[:, None] & m_k[None, :], other=0.0)
            b_q = tl.load(p_q, mask=m_k[:, None] & m_t[None, :], other=0.0)
            b_A += tl.dot(b_k, b_q) * scale
        if USE_G or USE_G_GAMMA:
            b_A *= exp2(b_g[None, :] - b_g[:, None])
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        p_do = do + o_t[:, None] * (HV*V) + o_v[None, :]
        p_dv = dv + o_t[:, None] * (HV*V) + o_v[None, :]
        b_do = tl.load(p_do, mask=m_t[:, None] & m_v[None, :], other=0.0)
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V, HV = *q.shape, v.shape[-1], v.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * HV)
    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    return o


def chunk_bwd_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V, HV = *k.shape, do.shape[-1], do.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    # H100 can have larger block size
    if check_shared_mem('hopper', k.device.index):
        CONST_TILING = 128
    elif check_shared_mem('ada', k.device.index):
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NV = triton.cdiv(V, BV)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    dv = torch.empty_like(do)
    grid = (NV, NT, B * HV)
    chunk_bwd_kernel_dv[grid](
        q=q,
        k=k,
        g=g,
        g_gamma=g_gamma,
        do=do,
        dv=dv,
        dh=dh,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        STATE_V_FIRST=state_v_first,
    )
    return dv


def chunk_bwd_dv_local(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    A: torch.Tensor | None = None,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V, HV = *k.shape, do.shape[-1], do.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    # H100 can have larger block size
    if check_shared_mem('hopper', k.device.index):
        CONST_TILING = 128
    elif check_shared_mem('ada', k.device.index):
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dv = torch.empty_like(do)
    grid = (NT, B * HV)
    chunk_bwd_kernel_dv_local[grid](
        q=q,
        k=k,
        g=g,
        g_gamma=g_gamma,
        A=A,
        do=do,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dv


@dispatch('common')
def chunk_bwd_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if check_shared_mem('hopper', k.device.index):
        CONST_TILING = 128
    elif check_shared_mem('ada', k.device.index):
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    # Unsloth: extends the fla PR #983 (issue #640) backport. Upstream narrowed the
    # Hopper guard to Triton [3.4.0, 3.7.1) but still refused to run at all. The
    # root cause reported in #640 is narrower than that: a Triton codegen bug that
    # only corrupts dk (and, through it, dg and dbeta) at BK == 64 on Hopper. The
    # upstream CONST_TILING sweep measured BK 32 and BK 128 clean (< 0.02) and only
    # BK 64 broken (dk max error 14.65, dg 5.47), and explicitly ruled out the
    # autotune config space as the trigger. Hopper's CONST_TILING of 128 means the
    # bad tile is reachable only for head dims 33 <= K <= 64, so step those down to
    # the measured-clean 32 and keep the fast kernels instead of falling back to a
    # pure-torch layer. Scoped to the gated path (g is not None), matching the
    # upstream guard. Triton >= 3.7.1 fixes the miscompile, so nothing is stepped
    # down there.
    #
    # Whether *this* call runs on Hopper is a per-tensor question, so ask the
    # tensor's own device first. ``IS_NVIDIA_HOPPER`` is frozen at import from
    # device 0 and is wrong in both directions on a mixed host: it misses a Hopper
    # card at a nonzero index, and it also marks a call on an Ada / Ampere /
    # Blackwell card as affected when device 0 happens to be the Hopper one, which
    # would narrow BK for a miscompile that cannot occur there. The global is used
    # only as the fallback for when the probe cannot tell (``None``), so an
    # unexpected probe failure on a real Hopper still gets the step-down.
    _on_hopper = _is_hopper_tensor(k)
    if _on_hopper is None:
        _on_hopper = IS_NVIDIA_HOPPER
    HOPPER_DQKWG_BROKEN = (
        g is not None
        and _on_hopper
        and TRITON_ABOVE_3_4_0
        and not TRITON_ABOVE_3_7_1
    )
    if HOPPER_DQKWG_BROKEN and BK == 64:
        BK = 32

    # Unreachable fence: BK can no longer be 64 in the affected range. Kept so a
    # future edit that reintroduces the bad tile fails loudly instead of silently
    # training on corrupted gradients.
    if HOPPER_DQKWG_BROKEN and BK == 64:
        raise RuntimeError(
            "Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces incorrect results for "
            "gated chunk_bwd_dqkwg at BK=64 (see fla issue #640). Fix this with: "
            'pip install -U "triton>=3.7.1"\n'
            "Note: upstream also suggests `pip install tilelang`, which cannot help "
            "here because this vendored snapshot prunes the TileLang kernels. To take "
            "the slower pure-PyTorch path instead, set UNSLOTH_DISABLE_HOPPER_FLA_BWD=1."
        )

    NK = triton.cdiv(K, BK)
    dq = q.new_empty(B, T, HV, K)
    dk = k.new_empty(B, T, HV, K)
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device) if g is not None else None
    dw = torch.empty_like(w) if w is not None else None

    grid = (NK, NT, B * HV)
    chunk_bwd_kernel_dqkwg[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        h=h,
        do=do,
        dh=dh,
        dw=dw,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        B=B,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        STATE_V_FIRST=state_v_first,
    )

    if H != HV:
        dq = dq.view(B, T, H, HV // H, K).sum(3)
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    if dg is not None:
        dg = dg.sum(0)
    return dq, dk, dw, dg
